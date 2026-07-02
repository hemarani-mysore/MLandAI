"""
choose_chunk_size.py — Evaluate optimal RAG chunk size
=======================================================
Tests multiple chunk sizes on the Braintree corpus and reports:
  - Average response time
  - Faithfulness  (is the answer grounded in the retrieved context?)
  - Relevancy     (is the answer relevant to the question?)

Mirrors the approach from:
  github.com/NirDiamant/RAG_Techniques — choose_chunk_size.py
but uses LangChain + FAISS + OpenAI instead of llama_index.

Usage:
    python choose_chunk_size.py
    python choose_chunk_size.py --chunk-sizes 256 512 1000 2000
    python choose_chunk_size.py --num-questions 10 --sample-docs 40
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent))
from helper_functions import (
    create_question_answer_from_context_chain,
    retrieve_context_per_question,
)

RAW_DOCS_DIR = Path(__file__).parent / "corpus" / "data" / "raw_docs"

DEFAULT_CHUNK_SIZES = [256, 512, 1000, 2000]

# Braintree-specific eval questions covering the main topics in the corpus
BRAINTREE_EVAL_QUESTIONS = [
    "How do I create a transaction in Braintree?",
    "What are the steps to submit a transaction for settlement?",
    "How do I create a customer in Braintree?",
    "How do I store a payment method for future use?",
    "What is a payment method nonce and how is it used?",
    "How do I process a refund in Braintree?",
    "How do I void a transaction?",
    "What webhook notifications does Braintree support?",
    "How does Braintree handle 3D Secure authentication?",
    "What are the different transaction statuses in Braintree?",
    "How do I set up recurring billing or subscriptions?",
    "How can I accept PayPal payments using Braintree?",
    "What is Drop-in UI and how do I integrate it?",
    "How do I handle failed or declined transactions?",
    "What test card numbers can I use in the Braintree sandbox?",
    "What is a client token and how is it generated?",
    "How do I accept Apple Pay with Braintree?",
    "What is the difference between authorization and settlement?",
    "How do I search for transactions in Braintree?",
    "How do I handle disputes or chargebacks?",
]


# ─────────────────────────────────────────────────────────────────────
# Document loading
# ─────────────────────────────────────────────────────────────────────

def load_raw_docs(docs_dir: Path, sample_size: int | None = None) -> list[Document]:
    """Load .md files from raw_docs/ as LangChain Documents."""
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"{docs_dir} not found. Run 'python corpus/scraper.py' first."
        )

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")

    if sample_size and sample_size < len(md_files):
        md_files = random.sample(md_files, sample_size)

    docs = []
    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name},
            ))

    print(f"Loaded {len(docs)} documents from {docs_dir}")
    return docs


# ─────────────────────────────────────────────────────────────────────
# Index builder
# ─────────────────────────────────────────────────────────────────────

def build_index(docs: list[Document], chunk_size: int) -> FAISS:
    """Re-chunk docs at the given chunk_size and build an in-memory FAISS index."""
    overlap = max(20, chunk_size // 10)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"  chunk_size={chunk_size}: {len(chunks)} chunks — embedding...", end="", flush=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = FAISS.from_documents(chunks, embeddings)
    print(f" done ({index.index.ntotal} vectors)")
    return index


# ─────────────────────────────────────────────────────────────────────
# Evaluators
# ─────────────────────────────────────────────────────────────────────

def _ask_llm_yes_no(llm: ChatOpenAI, prompt: str) -> bool:
    """Send a YES/NO prompt to the LLM and return True for YES."""
    response = llm.invoke(prompt)
    return response.content.strip().upper().startswith("YES")


def evaluate_faithfulness(question: str, context: list[str], answer: str, eval_llm: ChatOpenAI) -> bool:
    """Return True if the answer is grounded in the retrieved context."""
    context_text = "\n\n".join(context)
    prompt = f"""You are an evaluator checking whether an answer is directly supported by the provided context.

Context:
{context_text}

Question: {question}
Answer: {answer}

Is the answer directly supported by the context above?
Respond with YES if the answer is grounded in the context, NO if it contains information not present in the context.
Answer with only YES or NO."""
    return _ask_llm_yes_no(eval_llm, prompt)


def evaluate_relevancy(question: str, answer: str, eval_llm: ChatOpenAI) -> bool:
    """Return True if the answer is relevant to the question."""
    prompt = f"""You are an evaluator checking whether an answer is relevant to a question.

Question: {question}
Answer: {answer}

Is this answer relevant to the question? A relevant answer directly addresses what was asked.
Respond with YES if the answer is relevant, NO if it is off-topic or does not address the question.
Answer with only YES or NO."""
    return _ask_llm_yes_no(eval_llm, prompt)


# ─────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────

def evaluate_chunk_size(
    chunk_size: int,
    docs: list[Document],
    questions: list[str],
    qa_llm: ChatOpenAI,
    eval_llm: ChatOpenAI,
    top_k: int = 3,
) -> dict:
    """
    Build an index at the given chunk_size, run all eval questions through the
    RAG pipeline, and return average response time, faithfulness, and relevancy.
    """
    index     = build_index(docs, chunk_size)
    retriever = index.as_retriever(search_kwargs={"k": top_k})
    qa_chain  = create_question_answer_from_context_chain(qa_llm)

    total_time        = 0.0
    total_faithfulness = 0
    total_relevancy    = 0

    for i, question in enumerate(questions, 1):
        print(f"    Q{i}/{len(questions)}: {question[:60]}...")
        start   = time.time()
        context = retrieve_context_per_question(question, retriever)
        context_text = "\n\n".join(context)

        result  = qa_chain.invoke({"question": question, "context": context_text})
        answer  = result.answer_based_on_content
        elapsed = time.time() - start

        faithful = evaluate_faithfulness(question, context, answer, eval_llm)
        relevant = evaluate_relevancy(question, answer, eval_llm)

        total_time         += elapsed
        total_faithfulness += int(faithful)
        total_relevancy    += int(relevant)

        print(f"      time={elapsed:.2f}s  faithful={'✓' if faithful else '✗'}  relevant={'✓' if relevant else '✗'}")

    n = len(questions)
    return {
        "chunk_size"  : chunk_size,
        "avg_time"    : total_time / n,
        "faithfulness": total_faithfulness / n,
        "relevancy"   : total_relevancy / n,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(chunk_sizes: list[int], num_questions: int, sample_docs: int | None):
    docs      = load_raw_docs(RAW_DOCS_DIR, sample_size=sample_docs)
    questions = BRAINTREE_EVAL_QUESTIONS[:num_questions]

    print(f"\nEval questions : {len(questions)}")
    print(f"Chunk sizes    : {chunk_sizes}")
    print(f"Documents used : {len(docs)}\n")

    qa_llm   = ChatOpenAI(model="gpt-3.5-turbo",  temperature=0)
    eval_llm = ChatOpenAI(model="gpt-4o-mini",     temperature=0)

    results = []
    for chunk_size in chunk_sizes:
        print(f"\n{'─'*60}")
        print(f"Testing chunk_size = {chunk_size}")
        print(f"{'─'*60}")
        result = evaluate_chunk_size(
            chunk_size, docs, questions, qa_llm, eval_llm
        )
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"{'RESULTS SUMMARY':^60}")
    print(f"{'═'*60}")
    print(f"{'Chunk Size':>12}  {'Avg Time (s)':>14}  {'Faithfulness':>14}  {'Relevancy':>10}")
    print(f"{'─'*60}")
    for r in results:
        print(
            f"{r['chunk_size']:>12}  "
            f"{r['avg_time']:>14.2f}  "
            f"{r['faithfulness']:>13.0%}  "
            f"{r['relevancy']:>9.0%}"
        )
    print(f"{'═'*60}")

    best = max(results, key=lambda r: (r["faithfulness"] + r["relevancy"]) / 2)
    print(f"\nBest chunk size by faithfulness + relevancy: {best['chunk_size']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare RAG quality across different chunk sizes on the Braintree corpus."
    )
    parser.add_argument(
        "--chunk-sizes", nargs="+", type=int, default=DEFAULT_CHUNK_SIZES,
        help=f"Chunk sizes to evaluate (default: {DEFAULT_CHUNK_SIZES})",
    )
    parser.add_argument(
        "--num-questions", type=int, default=5,
        help="Number of evaluation questions to use (default: 5, max: 20)",
    )
    parser.add_argument(
        "--sample-docs", type=int, default=50,
        help="Number of documents to sample from the corpus (default: 50, None = all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        chunk_sizes=args.chunk_sizes,
        num_questions=min(args.num_questions, len(BRAINTREE_EVAL_QUESTIONS)),
        sample_docs=args.sample_docs,
    )
