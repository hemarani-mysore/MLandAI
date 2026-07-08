"""
reranking.py — RAG with Reranking on the Braintree Corpus
===========================================================
Implements two reranking strategies on top of the pre-built Braintree FAISS index:

  1. LLM Reranker   — GPT-4o-mini scores each retrieved doc 1-10 for relevance,
                      then returns the top-N highest-scoring docs.
  2. Cross-Encoder  — A lightweight cross-encoder model (ms-marco-MiniLM-L-6-v2)
                      scores (query, doc) pairs and reranks accordingly.

Both improve on basic similarity search by considering query intent, not just
keyword overlap.

Mirrors the approach from:
  github.com/NirDiamant/RAG_Techniques — reranking.py
but loads the pre-built Braintree FAISS index instead of a PDF.

Usage:
    python reranking.py --query "How do I create a transaction?"
    python reranking.py --query "What is a payment nonce?" --retriever-type cross_encoder
    python reranking.py --query "How do I set up webhooks?" --compare
"""

import argparse
import os
import sys
from typing import Any, List

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from helper_functions import create_question_answer_from_context_chain
from corpus.embed import load_index


# ─────────────────────────────────────────────────────────────────────
# LLM-based reranker
# ─────────────────────────────────────────────────────────────────────

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="Relevance score of the document to the query (1–10).")


def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    """
    Ask an LLM to score each document's relevance to the query (1–10),
    then return the top_n highest-scoring documents.
    """
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query.
Consider the specific context and intent of the query, not just keyword matches.

Query: {query}
Document: {doc}
Relevance Score:""",
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=50)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        try:
            score = llm_chain.invoke({"query": query, "doc": doc.page_content}).relevance_score
            score = float(score)
        except Exception:
            score = 0.0
        scored_docs.append((doc, score))
        print(f"    scored: {score:.1f} — {doc.page_content[:60].strip()}...")

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]


class LLMRerankerRetriever(BaseRetriever, BaseModel):
    """Retrieves 20 candidates via FAISS, reranks with GPT-4o-mini, returns top_n."""

    vectorstore: Any = Field(description="FAISS vector store for initial retrieval")
    initial_k: int = Field(default=20, description="Candidates to fetch before reranking")
    top_n: int = Field(default=3, description="Documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        print(f"\n[LLM Reranker] Fetching {self.initial_k} candidates...")
        candidates = self.vectorstore.similarity_search(query, k=self.initial_k)
        print(f"[LLM Reranker] Scoring candidates...")
        return rerank_documents(query, candidates, top_n=self.top_n)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async not implemented")


# ─────────────────────────────────────────────────────────────────────
# Cross-encoder reranker
# ─────────────────────────────────────────────────────────────────────

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    """Retrieves k candidates via FAISS, reranks with a cross-encoder, returns top rerank_top_k."""

    vectorstore: Any = Field(description="FAISS vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=20, description="Candidates to fetch before reranking")
    rerank_top_k: int = Field(default=3, description="Documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        print(f"\n[Cross-Encoder] Fetching {self.k} candidates...")
        candidates = self.vectorstore.similarity_search(query, k=self.k)
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        print(f"[Cross-Encoder] Top scores: {[round(float(s), 3) for _, s in scored[:5]]}")
        return [doc for doc, _ in scored[:self.rerank_top_k]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async not implemented")


# ─────────────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────────────

def compare_rag_techniques(query: str, vectorstore: Any) -> None:
    """
    Side-by-side comparison of baseline similarity search vs LLM reranker
    on the Braintree corpus.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs LLM Reranker")
    print("=" * 60)
    print(f"Query: {query}\n")

    print("── Baseline (similarity search, top 3) ──")
    baseline_docs = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(baseline_docs, 1):
        print(f"\nDoc {i}: {doc.page_content[:200].strip()}...")

    print("\n── LLM Reranker (fetch 20, score & pick top 3) ──")
    reranker = LLMRerankerRetriever(vectorstore=vectorstore, initial_k=20, top_n=3)
    reranked_docs = reranker._get_relevant_documents(query)
    for i, doc in enumerate(reranked_docs, 1):
        print(f"\nDoc {i}: {doc.page_content[:200].strip()}...")


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        print("\n--- Loading Braintree FAISS index ---")
        self.vectorstore = load_index()
        print(f"Vectors in index: {self.vectorstore.index.ntotal}")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    def run(self, query: str, retriever_type: str = "reranker", compare: bool = False):
        if retriever_type == "reranker":
            retriever = LLMRerankerRetriever(vectorstore=self.vectorstore, initial_k=20, top_n=3)
        elif retriever_type == "cross_encoder":
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            retriever = CrossEncoderRetriever(
                vectorstore=self.vectorstore,
                cross_encoder=cross_encoder,
                k=20,
                rerank_top_k=3,
            )
        else:
            raise ValueError(f"Unknown retriever type '{retriever_type}'. Use 'reranker' or 'cross_encoder'.")

        qa_chain = create_question_answer_from_context_chain(self.llm)

        print(f"\nRunning query with {retriever_type} retriever...")
        reranked_docs = retriever._get_relevant_documents(query)
        context_text  = "\n\n".join(doc.page_content for doc in reranked_docs)
        result        = qa_chain.invoke({"question": query, "context": context_text})

        print(f"\n{'='*60}")
        print(f"Question : {query}")
        print(f"{'='*60}")
        print(f"Answer   : {result.answer_based_on_content}")
        print(f"\nSource documents used:")
        for i, doc in enumerate(reranked_docs, 1):
            src = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            print(f"  [{i}] {src}: {doc.page_content[:120].strip()}...")
        print(f"{'='*60}")

        if compare:
            compare_rag_techniques(query, self.vectorstore)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG with reranking on the Braintree corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reranking.py --query "How do I create a transaction?"
  python reranking.py --query "How do I set up webhooks?" --retriever-type cross_encoder
  python reranking.py --query "What is a payment nonce?" --compare
        """,
    )
    parser.add_argument(
        "--query", type=str, default="How do I create a transaction in Braintree?",
        help="Question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--retriever-type", type=str, default="reranker",
        choices=["reranker", "cross_encoder"],
        help="'reranker' = LLM-based scoring; 'cross_encoder' = sentence-transformer model (default: reranker).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Show side-by-side comparison of baseline vs LLM reranker.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = RAGPipeline()
    pipeline.run(
        query=args.query,
        retriever_type=args.retriever_type,
        compare=args.compare,
    )
