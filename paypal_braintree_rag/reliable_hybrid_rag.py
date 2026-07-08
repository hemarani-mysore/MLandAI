"""
reliable_hybrid_rag.py — Hybrid + Reranking + Reliability + Source Attribution
================================================================================
The most complete RAG pipeline in this project. Stacks four techniques:

  Stage 1 — Hybrid retrieval (Semantic FAISS + BM25 + RRF)
            Casts a wide net: top-20 diverse candidates covering both
            meaning and exact keywords.

  Stage 2 — Cross-encoder reranking
            A sentence-transformer cross-encoder scores every (query, chunk)
            pair and keeps the top-5 highest-scoring chunks.

  Stage 3 — Reliability check
            Each of the top-5 chunks is validated: if its cross-encoder
            score (sigmoid-normalised) is below a threshold it is dropped.
            If fewer than 2 chunks survive, the pipeline refuses to answer
            rather than hallucinate.

  Stage 4 — Source-attributed answer generation
            The LLM is prompted to cite [1], [2], … inline, so every claim
            is traceable to a specific Braintree doc page.

Pipeline setup (one-time):
    python corpus/semantic_parser.py   →  chunks_semantic.jsonl
    python corpus/semantic_embed.py    →  faiss_index_semantic/
    python corpus/hybrid_embed.py      →  bm25_index.pkl

Usage:
    python reliable_hybrid_rag.py --query "How do I create a transaction?"
    python reliable_hybrid_rag.py --query "What is a payment nonce?" --show-stages
    python reliable_hybrid_rag.py --query "How do I void a transaction?" --threshold 0.3
"""

import argparse
import math
import os
import sys
from pathlib import Path

# Fix macOS OpenMP conflict between conda FAISS and other libs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

from corpus.semantic_embed import load_index as load_faiss_index
from corpus.hybrid_embed import load_bm25
from hybrid_rag import reciprocal_rank_fusion, bm25_search


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Normalise a cross-encoder logit to a 0-1 relevance probability."""
    return 1 / (1 + math.exp(-x))


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Cross-encoder reranking
# ─────────────────────────────────────────────────────────────────────

def rerank_with_cross_encoder(
    query: str,
    docs: list[Document],
    cross_encoder: CrossEncoder,
    top_n: int = 5,
) -> list[tuple[Document, float]]:
    """
    Score every (query, doc) pair with the cross-encoder.
    Returns list of (doc, normalised_score) sorted best-first.
    """
    pairs  = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored = sorted(
        zip(docs, [_sigmoid(float(s)) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )
    return scored[:top_n]


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Reliability check
# ─────────────────────────────────────────────────────────────────────

def reliability_check(
    scored_docs: list[tuple[Document, float]],
    threshold: float = 0.4,
    min_chunks: int = 2,
) -> tuple[list[tuple[Document, float]], bool]:
    """
    Filter out chunks whose cross-encoder score is below the threshold.
    Returns (passing_docs, is_reliable).
    is_reliable=False means the pipeline should refuse to answer.
    """
    passing = [(doc, score) for doc, score in scored_docs if score >= threshold]
    return passing, len(passing) >= min_chunks


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Source-attributed answer generation
# ─────────────────────────────────────────────────────────────────────

_ATTRIBUTION_PROMPT = PromptTemplate(
    input_variables=["sources", "question"],
    template="""You are a helpful Braintree payments expert.
Answer the question below using ONLY the numbered sources provided.
For every fact or step you mention, add an inline citation like [1] or [2].
If the sources do not contain enough information, say "I don't have enough information to answer this."

{sources}

Question: {question}

Answer (with inline citations):""",
)


def generate_with_attribution(
    query: str,
    scored_docs: list[tuple[Document, float]],
    llm: ChatOpenAI,
) -> tuple[str, list[dict]]:
    """
    Build a numbered source block and generate an answer with inline citations.
    Returns (answer_text, source_metadata_list).
    """
    source_block = ""
    source_meta  = []

    for i, (doc, score) in enumerate(scored_docs, 1):
        src  = doc.metadata.get("source_file", "unknown")
        source_block += f"[{i}] {src} (relevance: {score:.2f})\n{doc.page_content}\n\n"
        source_meta.append({
            "index"      : i,
            "source_file": src,
            "score"      : score,
            "preview"    : doc.page_content[:120].strip(),
        })

    chain  = _ATTRIBUTION_PROMPT | llm
    result = chain.invoke({"sources": source_block.strip(), "question": query})
    return result.content.strip(), source_meta


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

class ReliableHybridRAG:
    """
    Full pipeline: Hybrid retrieval → Cross-encoder rerank →
                   Reliability check → Source-attributed answer.
    """

    def __init__(
        self,
        hybrid_candidates: int = 20,
        rerank_top_n: int = 5,
        reliability_threshold: float = 0.4,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print("\n--- Initializing Reliable Hybrid RAG ---")

        self.hybrid_candidates      = hybrid_candidates
        self.rerank_top_n           = rerank_top_n
        self.reliability_threshold  = reliability_threshold

        # Stage 1 indexes
        print("Loading semantic FAISS index...")
        self.vectorstore = load_faiss_index()
        self.faiss_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": hybrid_candidates}
        )
        print(f"  FAISS vectors : {self.vectorstore.index.ntotal}")

        print("Loading BM25 index...")
        self.bm25, self.chunks = load_bm25()
        print(f"  BM25 docs     : {len(self.chunks)}")

        # Stage 2 cross-encoder
        print(f"Loading cross-encoder ({cross_encoder_model})...")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        # Stage 4 LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        print("Ready.\n")

    def run(self, query: str, show_stages: bool = False) -> None:
        # ── Stage 1: Hybrid retrieval ──────────────────────────────
        if show_stages:
            print(f"\n[Stage 1] Hybrid retrieval — fetching {self.hybrid_candidates} candidates")

        semantic_docs = self.faiss_retriever.invoke(query)
        bm25_docs     = bm25_search(query, self.bm25, self.chunks, top_k=self.hybrid_candidates)
        candidates    = reciprocal_rank_fusion(semantic_docs, bm25_docs)[:self.hybrid_candidates]

        if show_stages:
            print(f"  {len(candidates)} candidates after RRF fusion")

        # ── Stage 2: Cross-encoder reranking ──────────────────────
        if show_stages:
            print(f"\n[Stage 2] Cross-encoder reranking → keeping top {self.rerank_top_n}")

        scored_docs = rerank_with_cross_encoder(
            query, candidates, self.cross_encoder, top_n=self.rerank_top_n
        )

        if show_stages:
            for doc, score in scored_docs:
                src = doc.metadata.get("source_file", "unknown")
                print(f"  {score:.3f}  {src}")

        # ── Stage 3: Reliability check ─────────────────────────────
        if show_stages:
            print(f"\n[Stage 3] Reliability check (threshold={self.reliability_threshold})")

        reliable_docs, is_reliable = reliability_check(
            scored_docs, threshold=self.reliability_threshold
        )

        if show_stages:
            passing = len(reliable_docs)
            dropped = len(scored_docs) - passing
            print(f"  {passing} chunks passed  |  {dropped} dropped below threshold")

        if not is_reliable:
            print(f"\n{'='*60}")
            print(f"Question : {query}")
            print(f"{'='*60}")
            print("Answer   : I don't have enough reliable information in the "
                  "Braintree docs to answer this question confidently.")
            print(f"{'='*60}")
            return

        # ── Stage 4: Source-attributed generation ─────────────────
        if show_stages:
            print(f"\n[Stage 4] Generating answer with source attribution")

        answer, source_meta = generate_with_attribution(query, reliable_docs, self.llm)

        print(f"\n{'='*60}")
        print(f"Question : {query}")
        print(f"{'='*60}")
        print(f"Answer   :\n{answer}")
        print(f"\nSources used:")
        for s in source_meta:
            print(f"  [{s['index']}] {s['source_file']}  (relevance: {s['score']:.2f})")
            print(f"       {s['preview']}...")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reliable Hybrid RAG: Hybrid + Cross-encoder + Reliability + Citations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reliable_hybrid_rag.py --query "How do I create a transaction?"
  python reliable_hybrid_rag.py --query "What is a payment nonce?" --show-stages
  python reliable_hybrid_rag.py --query "How do I void a transaction?" --threshold 0.3
        """,
    )
    parser.add_argument("--query", type=str, default="How do I create a transaction?",
                        help="Question to ask.")
    parser.add_argument("--candidates", type=int, default=20,
                        help="Hybrid candidates to fetch before reranking (default: 20).")
    parser.add_argument("--rerank-top-n", type=int, default=5,
                        help="How many to keep after cross-encoder reranking (default: 5).")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Minimum cross-encoder score (0-1) to pass reliability check (default: 0.4).")
    parser.add_argument("--show-stages", action="store_true",
                        help="Print what happens at each stage of the pipeline.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rag  = ReliableHybridRAG(
        hybrid_candidates     = args.candidates,
        rerank_top_n          = args.rerank_top_n,
        reliability_threshold = args.threshold,
    )
    rag.run(args.query, show_stages=args.show_stages)
