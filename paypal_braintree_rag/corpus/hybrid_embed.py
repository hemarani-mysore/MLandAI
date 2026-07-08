"""
corpus/hybrid_embed.py — Build BM25 Index from Semantic Chunks
===============================================================
Loads corpus/data/chunks_semantic.jsonl (produced by semantic_parser.py)
and builds a BM25 keyword index saved to corpus/data/bm25_index.pkl.

The FAISS index is already built by corpus/semantic_embed.py.
This adds the keyword side of hybrid search.

Usage:
    python corpus/hybrid_embed.py

Run after:
    python corpus/semantic_parser.py
    python corpus/semantic_embed.py
"""

import json
import pickle
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path(__file__).parent / "data" / "chunks_semantic.jsonl"
BM25_FILE   = Path(__file__).parent / "data" / "bm25_index.pkl"


def build_bm25(chunks_file: Path) -> tuple[BM25Okapi, list[dict]]:
    """Load semantic chunks and build a BM25 index over their content."""
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"{chunks_file} not found. Run corpus/semantic_parser.py first."
        )

    chunks = []
    with chunks_file.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} semantic chunks from {chunks_file}")

    # Tokenise: lowercase + split on whitespace
    tokenised = [chunk["content"].lower().split() for chunk in chunks]
    bm25      = BM25Okapi(tokenised)

    print(f"BM25 index built over {len(tokenised)} documents")
    return bm25, chunks


def load_bm25() -> tuple[BM25Okapi, list[dict]]:
    """Load saved BM25 index and chunk list from disk."""
    if not BM25_FILE.exists():
        raise FileNotFoundError(
            f"{BM25_FILE} not found. Run corpus/hybrid_embed.py first."
        )
    with BM25_FILE.open("rb") as f:
        data = pickle.load(f)
    print(f"Loaded BM25 index ({data['num_docs']} docs) from {BM25_FILE}")
    return data["bm25"], data["chunks"]


def main():
    BM25_FILE.parent.mkdir(parents=True, exist_ok=True)

    bm25, chunks = build_bm25(CHUNKS_FILE)

    with BM25_FILE.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks, "num_docs": len(chunks)}, f)

    print(f"\n✅ BM25 index saved → {BM25_FILE}")
    print(f"   Documents indexed : {len(chunks)}")


if __name__ == "__main__":
    main()
