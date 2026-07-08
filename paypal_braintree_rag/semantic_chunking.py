"""
semantic_chunking.py — RAG with Semantic Chunking on the Braintree Corpus
==========================================================================
Instead of splitting text at fixed character counts (e.g. every 1000 chars),
SemanticChunker splits where the *meaning changes* — detected by measuring
the embedding distance between consecutive sentences.

Breakpoint types:
  percentile         — split when distance > Nth percentile of all distances (default: 90)
  standard_deviation — split when distance > mean + N * std_dev
  interquartile      — split when distance > Q3 + N * IQR
  gradient           — split at the steepest drops in distance

Mirrors the approach from:
  github.com/NirDiamant/RAG_Techniques — semantic_chunking.py
but loads scraped Braintree markdown docs instead of a PDF.

Usage:
    python semantic_chunking.py --query "How do I create a transaction?"
    python semantic_chunking.py --query "How do I set up webhooks?" --breakpoint-type gradient
    python semantic_chunking.py --query "What is a payment nonce?" --compare
    python semantic_chunking.py --query "How do refunds work?" --sample-docs 50
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helper_functions import (
    create_question_answer_from_context_chain,
    retrieve_context_per_question,
    show_context,
)
from corpus.semantic_embed import load_index as load_semantic_index

RAW_DOCS_DIR    = Path(__file__).parent / "corpus" / "data" / "raw_docs"
SEMANTIC_INDEX  = Path(__file__).parent / "corpus" / "data" / "faiss_index_semantic"


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
        raise FileNotFoundError(f"No .md files in {docs_dir}")
    if sample_size and sample_size < len(md_files):
        import random
        md_files = random.sample(md_files, sample_size)

    docs = []
    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    print(f"Loaded {len(docs)} documents")
    return docs


# ─────────────────────────────────────────────────────────────────────
# Semantic Chunking RAG
# ─────────────────────────────────────────────────────────────────────

class SemanticChunkingRAG:
    """
    RAG pipeline that splits documents at semantic boundaries rather than
    fixed character counts.

    On first run: builds in-memory index from docs (slow — re-chunks + re-embeds).
    After running corpus/semantic_parser.py + corpus/semantic_embed.py:
    loads the pre-built index from disk instantly.
    """

    def __init__(
        self,
        docs: list[Document] | None = None,
        n_retrieved: int = 3,
        breakpoint_type: BreakpointThresholdType = "percentile",
        breakpoint_amount: float = 90,
    ):
        print(f"\n--- Initializing Semantic Chunking RAG ---")

        self.llm      = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.qa_chain = create_question_answer_from_context_chain(self.llm)
        self.time_records: dict = {}

        # ── Load pre-built index if available ─────────────────────
        if (SEMANTIC_INDEX / "index.faiss").exists():
            print("Loading pre-built semantic FAISS index from disk...")
            start = time.time()
            self.vectorstore = load_semantic_index()
            self.time_records["Loading"] = time.time() - start
            print(f"Load time      : {self.time_records['Loading']:.2f}s")
            print(f"Vectors stored : {self.vectorstore.index.ntotal}")

        # ── Build in-memory from docs (fallback) ──────────────────
        else:
            if not docs:
                raise ValueError(
                    "No pre-built semantic index found. Either run:\n"
                    "  python corpus/semantic_parser.py\n"
                    "  python corpus/semantic_embed.py\n"
                    "or pass docs= to build an in-memory index."
                )
            print(f"No saved index found — building in-memory index from {len(docs)} docs.")
            print(f"Breakpoint type   : {breakpoint_type}  |  amount: {breakpoint_amount}")

            embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
            chunker      = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=breakpoint_type,
                breakpoint_threshold_amount=breakpoint_amount,
            )

            print("\nChunking semantically (embeds sentences to detect boundaries)...")
            start = time.time()
            all_chunks = []
            for doc in docs:
                chunks = chunker.create_documents([doc.page_content], metadatas=[doc.metadata])
                all_chunks.extend(chunks)
            self.time_records["Chunking"] = time.time() - start

            print(f"Chunking time  : {self.time_records['Chunking']:.2f}s")
            print(f"Total chunks   : {len(all_chunks)}")
            if all_chunks:
                sizes = [len(c.page_content) for c in all_chunks]
                print(f"Chunk sizes    : min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)//len(sizes)}")

            print("\nEmbedding into FAISS...")
            embed_start = time.time()
            self.vectorstore = FAISS.from_documents(all_chunks, embeddings)
            self.time_records["Embedding"] = time.time() - embed_start
            print(f"Embedding time : {self.time_records['Embedding']:.2f}s")
            print(f"Vectors stored : {self.vectorstore.index.ntotal}")

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": n_retrieved, "fetch_k": n_retrieved * 4},
        )

    def run(self, query: str, show_chunks: bool = False) -> dict:
        start = time.time()
        context = retrieve_context_per_question(query, self.retriever)
        self.time_records["Retrieval"] = time.time() - start
        print(f"\nRetrieval Time : {self.time_records['Retrieval']:.2f}s")

        if show_chunks:
            print("\n--- Retrieved Semantic Chunks ---")
            show_context(context)

        context_text = "\n\n".join(context)
        result = self.qa_chain.invoke({"question": query, "context": context_text})

        print(f"\n{'='*60}")
        print(f"Question : {query}")
        print(f"{'='*60}")
        print(f"Answer   : {result.answer_based_on_content}")
        print(f"{'='*60}")

        return self.time_records


# ─────────────────────────────────────────────────────────────────────
# Comparison: fixed-size vs semantic chunking
# ─────────────────────────────────────────────────────────────────────

def compare_chunking(docs: list[Document], query: str, chunk_size: int = 1000):
    """
    Run the same query through fixed-size chunking and semantic chunking,
    and print a side-by-side comparison of retrieved chunks.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\n" + "=" * 60)
    print("COMPARISON: Fixed-size vs Semantic Chunking")
    print("=" * 60)
    print(f"Query: {query}\n")

    # ── Fixed-size ─────────────────────────────────────────────────
    print("── Fixed-size chunks (1000 chars, overlap 100) ──")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    fixed_chunks = splitter.split_documents(docs)
    fixed_store  = FAISS.from_documents(fixed_chunks, embeddings)
    fixed_docs   = fixed_store.similarity_search(query, k=3)
    print(f"Total fixed chunks : {len(fixed_chunks)}")
    for i, doc in enumerate(fixed_docs, 1):
        print(f"\n  Chunk {i} ({len(doc.page_content)} chars):")
        print(f"  {doc.page_content[:250].strip()}...")

    # ── Semantic ───────────────────────────────────────────────────
    print("\n── Semantic chunks (percentile=90) ──")
    chunker       = SemanticChunker(embeddings, breakpoint_threshold_type="percentile",
                                    breakpoint_threshold_amount=90)
    sem_chunks    = []
    for doc in docs:
        sem_chunks.extend(chunker.create_documents([doc.page_content], metadatas=[doc.metadata]))
    sem_store     = FAISS.from_documents(sem_chunks, embeddings)
    sem_docs      = sem_store.similarity_search(query, k=3)
    print(f"Total semantic chunks : {len(sem_chunks)}")
    for i, doc in enumerate(sem_docs, 1):
        print(f"\n  Chunk {i} ({len(doc.page_content)} chars):")
        print(f"  {doc.page_content[:250].strip()}...")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG with semantic chunking on the Braintree corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python semantic_chunking.py --query "How do I create a transaction?"
  python semantic_chunking.py --query "How do refunds work?" --breakpoint-type gradient
  python semantic_chunking.py --query "What is a nonce?" --compare --sample-docs 20
        """,
    )
    parser.add_argument("--query", type=str, default="How do I create a transaction?",
                        help="Question to ask.")
    parser.add_argument("--breakpoint-type", type=str, default="percentile",
                        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
                        help="How to detect semantic boundaries (default: percentile).")
    parser.add_argument("--breakpoint-amount", type=float, default=90,
                        help="Threshold value for the breakpoint (default: 90).")
    parser.add_argument("--n-retrieved", type=int, default=3,
                        help="Number of chunks to retrieve (default: 3).")
    parser.add_argument("--sample-docs", type=int, default=30,
                        help="Number of docs to sample from corpus (default: 30). Use more for better coverage.")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Print the raw retrieved chunks alongside the answer.")
    parser.add_argument("--compare", action="store_true",
                        help="Show side-by-side comparison of fixed-size vs semantic chunking.")
    return parser.parse_args()


def main(args):
    if args.compare:
        # compare always needs to re-chunk, so load docs
        docs = load_raw_docs(RAW_DOCS_DIR, sample_size=args.sample_docs)
        compare_chunking(docs, args.query)
    elif (SEMANTIC_INDEX / "index.faiss").exists():
        # pre-built index available — no need to load docs
        rag = SemanticChunkingRAG(n_retrieved=args.n_retrieved)
        rag.run(args.query, show_chunks=args.show_chunks)
    else:
        # no saved index — build in-memory from sample docs
        print("No pre-built semantic index found.")
        print("Tip: run corpus/semantic_parser.py + corpus/semantic_embed.py to build it once.\n")
        docs = load_raw_docs(RAW_DOCS_DIR, sample_size=args.sample_docs)
        rag  = SemanticChunkingRAG(
            docs=docs,
            n_retrieved=args.n_retrieved,
            breakpoint_type=args.breakpoint_type,
            breakpoint_amount=args.breakpoint_amount,
        )
        rag.run(args.query, show_chunks=args.show_chunks)


if __name__ == "__main__":
    main(parse_args())
