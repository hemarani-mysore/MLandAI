"""
corpus/embed.py — Chunks → FAISS Index
========================================
Loads corpus/data/chunks.jsonl, embeds each chunk with OpenAI embeddings,
and saves a persistent FAISS index to corpus/data/faiss_index/.

Usage:
    python corpus/embed.py
    python corpus/embed.py --batch-size 100    # embed 100 chunks at a time
    python corpus/embed.py --doc-type guide    # embed only guide chunks

Run parser.py first to generate chunks.jsonl.
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

CHUNKS_FILE  = Path(__file__).parent / "data" / "chunks.jsonl"
FAISS_DIR    = Path(__file__).parent / "data" / "faiss_index"

EMBED_DELAY  = 0.05   # seconds between batches (avoids OpenAI rate limits)


# ─────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────

def load_chunks(doc_type_filter: str | None = None) -> list[Document]:
    """
    Load chunks.jsonl and return as LangChain Document objects.
    Metadata fields (source_file, section_title, doc_type) are preserved.
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. Run corpus/parser.py first."
        )

    docs = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if doc_type_filter and chunk["doc_type"] != doc_type_filter:
                continue
            docs.append(Document(
                page_content = chunk["content"],
                metadata     = {
                    "source_file"   : chunk["source_file"],
                    "section_title" : chunk["section_title"],
                    "doc_type"      : chunk["doc_type"],
                    "chunk_index"   : chunk["chunk_index"],
                },
            ))

    print(f"Loaded {len(docs)} chunks from {CHUNKS_FILE}")
    return docs


# ─────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────

def build_faiss_index(docs: list[Document], batch_size: int = 100) -> FAISS:
    """
    Embed docs in batches and build a FAISS vector store.
    Batching avoids hitting OpenAI's per-request token limit.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"\nEmbedding {len(docs)} chunks in batches of {batch_size}...")

    # Embed first batch to initialise the index
    first_batch = docs[:batch_size]
    vectorstore  = FAISS.from_documents(first_batch, embeddings)
    print(f"  Batch 1/{-(-len(docs) // batch_size)} done")

    # Embed remaining batches and merge
    for i in range(batch_size, len(docs), batch_size):
        batch      = docs[i : i + batch_size]
        batch_num  = i // batch_size + 1
        total_batches = -(-len(docs) // batch_size)

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


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main(batch_size: int, doc_type_filter: str | None):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    docs        = load_chunks(doc_type_filter)
    vectorstore = build_faiss_index(docs, batch_size=batch_size)

    vectorstore.save_local(str(FAISS_DIR))
    print(f"\n✅ FAISS index saved → {FAISS_DIR}")
    print(f"   Total vectors : {vectorstore.index.ntotal}")


def load_index() -> FAISS:
    """Load the saved FAISS index from disk (used by simple_rag.py)."""
    if not (FAISS_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index found at {FAISS_DIR}. Run corpus/embed.py first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks.jsonl into a FAISS index.")
    parser.add_argument("--batch-size", type=int,  default=100,
                        help="Number of chunks to embed per API call (default 100).")
    parser.add_argument("--doc-type",   type=str,  default=None,
                        help="Only embed chunks with this doc_type (e.g. 'guide').")
    args = parser.parse_args()

    main(args.batch_size, args.doc_type)
