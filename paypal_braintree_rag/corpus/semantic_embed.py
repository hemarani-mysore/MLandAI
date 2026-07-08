"""
corpus/semantic_embed.py — Embed Semantic Chunks into FAISS
============================================================
Loads corpus/data/chunks_semantic.jsonl (produced by semantic_parser.py),
embeds each chunk with OpenAI embeddings, and saves a FAISS index to
corpus/data/faiss_index_semantic/.

Usage:
    python corpus/semantic_embed.py
    python corpus/semantic_embed.py --batch-size 50

Run corpus/semantic_parser.py first to generate chunks_semantic.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

CHUNKS_FILE = Path(__file__).parent / "data" / "chunks_semantic.jsonl"
FAISS_DIR   = Path(__file__).parent / "data" / "faiss_index_semantic"
EMBED_DELAY = 0.05


def load_chunks() -> list[Document]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. Run corpus/semantic_parser.py first."
        )
    docs = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            docs.append(Document(
                page_content=chunk["content"],
                metadata={
                    "source_file"      : chunk["source_file"],
                    "chunk_index"      : chunk["chunk_index"],
                    "chunk_size"       : chunk["chunk_size"],
                    "breakpoint_type"  : chunk.get("breakpoint_type", "percentile"),
                    "breakpoint_amount": chunk.get("breakpoint_amount", 90),
                },
            ))
    print(f"Loaded {len(docs)} semantic chunks from {CHUNKS_FILE}")
    return docs


def build_faiss_index(docs: list[Document], batch_size: int) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    total_batches = -(-len(docs) // batch_size)
    print(f"\nEmbedding {len(docs)} chunks in {total_batches} batches of {batch_size}...")

    vectorstore = FAISS.from_documents(docs[:batch_size], embeddings)
    print(f"  Batch 1/{total_batches} done")

    for i in range(batch_size, len(docs), batch_size):
        batch     = docs[i : i + batch_size]
        batch_num = i // batch_size + 1
        try:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)
            print(f"  Batch {batch_num}/{total_batches} done ({i + len(batch)}/{len(docs)} chunks)")
        except Exception as e:
            print(f"  [error] Batch {batch_num}: {e} — retrying in 30s")
            time.sleep(30)
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)
        time.sleep(EMBED_DELAY)

    return vectorstore


def load_index() -> FAISS:
    """Load the saved semantic FAISS index from disk (used by semantic_chunking.py)."""
    if not (FAISS_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"No semantic FAISS index at {FAISS_DIR}. Run corpus/semantic_embed.py first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)


def main(batch_size: int):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    docs        = load_chunks()
    vectorstore = build_faiss_index(docs, batch_size)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"\n✅ Semantic FAISS index saved → {FAISS_DIR}")
    print(f"   Total vectors : {vectorstore.index.ntotal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed semantic chunks into a FAISS index.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Chunks to embed per API call (default: 100).")
    args = parser.parse_args()
    main(args.batch_size)
