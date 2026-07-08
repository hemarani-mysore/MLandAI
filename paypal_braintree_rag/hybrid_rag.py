"""
hybrid_rag.py — Hybrid RAG: Semantic Search + BM25 on Braintree Corpus
=======================================================================
Combines two retrieval strategies on the semantic chunks corpus:

  1. Semantic search (FAISS)  — finds chunks by meaning/context
  2. BM25 keyword search      — finds chunks by exact/partial keyword match

Results are merged using Reciprocal Rank Fusion (RRF), which rewards
chunks that rank highly in BOTH methods.

Why hybrid?
  - Semantic alone misses exact terms ("Transaction:Sale", "payment_method_nonce")
  - BM25 alone misses paraphrased queries ("how do I charge a card" ≠ "create transaction")
  - Together they cover both intent AND terminology

Pipeline:
    corpus/semantic_parser.py  →  chunks_semantic.jsonl   (run once)
    corpus/semantic_embed.py   →  faiss_index_semantic/   (run once)
    corpus/hybrid_embed.py     →  bm25_index.pkl          (run once)
    hybrid_rag.py              →  query + answer           (fast every time)

Usage:
    python hybrid_rag.py --query "How do I create a transaction?"
    python hybrid_rag.py --query "What is payment_method_nonce?" --show-context
    python hybrid_rag.py --query "How do I void a transaction?" --compare
"""

import argparse
import os
import sys
from pathlib import Path

# Fix macOS OpenMP conflict between conda FAISS and other libs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from helper_functions import create_question_answer_from_context_chain
from corpus.semantic_embed import load_index as load_faiss_index
from corpus.hybrid_embed import load_bm25


# ─────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    semantic_docs: list[Document],
    bm25_docs: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Merge two ranked lists using RRF.

    Score for each doc = sum of 1 / (rank + k) across both lists.
    A chunk appearing at rank 1 in both lists scores highest.

    k=60 is the standard constant that dampens the impact of very high ranks.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs):
        key = doc.page_content[:150]
        scores[key]  = scores.get(key, 0) + 1 / (rank + 1 + k)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:150]
        scores[key]  = scores.get(key, 0) + 1 / (rank + 1 + k)
        doc_map[key] = doc

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ranked_keys]


# ─────────────────────────────────────────────────────────────────────
# BM25 retrieval helper
# ─────────────────────────────────────────────────────────────────────

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 10,
) -> list[Document]:
    """Run BM25 keyword search and return top_k results as Documents."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        Document(
            page_content=chunks[i]["content"],
            metadata={
                "source_file" : chunks[i]["source_file"],
                "bm25_score"  : float(scores[i]),
            },
        )
        for i in top_indices
    ]


# ─────────────────────────────────────────────────────────────────────
# Hybrid RAG pipeline
# ─────────────────────────────────────────────────────────────────────

class HybridRAG:
    """
    Retrieves using both FAISS semantic search and BM25 keyword search,
    merges results with Reciprocal Rank Fusion, then generates an answer.
    """

    def __init__(self, n_retrieved: int = 3, candidates: int = 10):
        """
        Args:
            n_retrieved : Final number of chunks passed to the LLM.
            candidates  : How many candidates each retriever fetches before fusion.
        """
        print("\n--- Initializing Hybrid RAG (Semantic + BM25) ---")

        self.n_retrieved = n_retrieved
        self.candidates  = candidates

        # Load FAISS semantic index
        print("Loading semantic FAISS index...")
        self.vectorstore = load_faiss_index()
        self.faiss_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": candidates}
        )
        print(f"  FAISS vectors : {self.vectorstore.index.ntotal}")

        # Load BM25 index
        print("Loading BM25 index...")
        self.bm25, self.chunks = load_bm25()
        print(f"  BM25 docs     : {len(self.chunks)}")

        self.llm      = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.qa_chain = create_question_answer_from_context_chain(self.llm)
        print("Ready.\n")

    def retrieve(self, query: str) -> list[Document]:
        """Run hybrid retrieval and return top n_retrieved docs after RRF."""

        # Semantic retrieval
        semantic_docs = self.faiss_retriever.invoke(query)

        # BM25 retrieval
        bm25_docs = bm25_search(query, self.bm25, self.chunks, top_k=self.candidates)

        # Merge with RRF
        fused = reciprocal_rank_fusion(semantic_docs, bm25_docs)

        return fused[:self.n_retrieved]

    def run(self, query: str, show_context: bool = False) -> None:
        docs         = self.retrieve(query)
        context_text = "\n\n".join(doc.page_content for doc in docs)

        if show_context:
            print("\n--- Retrieved Chunks (after RRF) ---")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source_file", "unknown")
                print(f"\nChunk {i} [{src}]:")
                print(doc.page_content[:300].strip() + "...")

        result = self.qa_chain.invoke({"question": query, "context": context_text})

        print(f"\n{'='*60}")
        print(f"Question : {query}")
        print(f"{'='*60}")
        print(f"Answer   : {result.answer_based_on_content}")
        print(f"\nSources:")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source_file", "unknown")
            print(f"  [{i}] {src}")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────
# Comparison: semantic-only vs hybrid
# ─────────────────────────────────────────────────────────────────────

def compare(query: str, rag: HybridRAG) -> None:
    print("\n" + "=" * 60)
    print("COMPARISON: Semantic-only vs Hybrid (Semantic + BM25)")
    print("=" * 60)
    print(f"Query: {query}\n")

    print("── Semantic only (FAISS top 3) ──")
    semantic_docs = rag.faiss_retriever.invoke(query)[:3]
    for i, doc in enumerate(semantic_docs, 1):
        src = doc.metadata.get("source_file", "unknown")
        print(f"  [{i}] {src}: {doc.page_content[:150].strip()}...")

    print("\n── BM25 only (keyword top 3) ──")
    bm25_docs = bm25_search(query, rag.bm25, rag.chunks, top_k=3)
    for i, doc in enumerate(bm25_docs, 1):
        src = doc.metadata.get("source_file", "unknown")
        score = doc.metadata.get("bm25_score", 0)
        print(f"  [{i}] {src} (score={score:.2f}): {doc.page_content[:150].strip()}...")

    print("\n── Hybrid RRF (top 3) ──")
    fused = rag.retrieve(query)
    for i, doc in enumerate(fused, 1):
        src = doc.metadata.get("source_file", "unknown")
        print(f"  [{i}] {src}: {doc.page_content[:150].strip()}...")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG (semantic + BM25) on the Braintree corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_rag.py --query "How do I create a transaction?"
  python hybrid_rag.py --query "What is payment_method_nonce?" --show-context
  python hybrid_rag.py --query "How do I void a transaction?" --compare
        """,
    )
    parser.add_argument("--query", type=str, default="How do I create a transaction?",
                        help="Question to ask.")
    parser.add_argument("--n-retrieved", type=int, default=3,
                        help="Number of chunks to pass to the LLM (default: 3).")
    parser.add_argument("--candidates", type=int, default=10,
                        help="Candidates each retriever fetches before fusion (default: 10).")
    parser.add_argument("--show-context", action="store_true",
                        help="Print the retrieved chunks alongside the answer.")
    parser.add_argument("--compare", action="store_true",
                        help="Show semantic-only vs BM25-only vs hybrid results side by side.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rag  = HybridRAG(n_retrieved=args.n_retrieved, candidates=args.candidates)

    if args.compare:
        compare(args.query, rag)
    else:
        rag.run(args.query, show_context=args.show_context)
