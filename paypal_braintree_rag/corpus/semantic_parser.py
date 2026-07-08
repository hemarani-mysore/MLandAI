"""
corpus/semantic_parser.py — Semantic Chunking of Braintree Docs
================================================================
Splits scraped .md files at semantic boundaries (where meaning shifts)
instead of fixed character counts.

Uses LangChain's SemanticChunker which:
  1. Splits text into sentences
  2. Embeds each sentence
  3. Measures cosine distance between adjacent sentences
  4. Splits where the distance exceeds a threshold

Saves output to corpus/data/chunks_semantic.jsonl.

Usage:
    python corpus/semantic_parser.py
    python corpus/semantic_parser.py --breakpoint-type gradient
    python corpus/semantic_parser.py --breakpoint-amount 85

Run corpus/scraper.py first to populate corpus/data/raw_docs/.
Then run corpus/semantic_embed.py to build the FAISS index.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_openai import OpenAIEmbeddings

RAW_DOCS_DIR    = Path(__file__).parent / "data" / "raw_docs"
CHUNKS_FILE     = Path(__file__).parent / "data" / "chunks_semantic.jsonl"


def parse_semantic(
    docs_dir: Path,
    breakpoint_type: BreakpointThresholdType,
    breakpoint_amount: float,
) -> list[dict]:
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}. Run corpus/scraper.py first.")

    print(f"Found {len(md_files)} markdown files")
    print(f"Breakpoint type   : {breakpoint_type}")
    print(f"Breakpoint amount : {breakpoint_amount}")
    print("\nInitialising SemanticChunker (uses OpenAI embeddings to detect boundaries)...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chunker    = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=breakpoint_amount,
    )

    all_chunks: list[dict] = []
    start = time.time()

    for i, md_path in enumerate(md_files, 1):
        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        try:
            chunks = chunker.create_documents([text])
        except Exception as e:
            print(f"  [skip] {md_path.name}: {e}")
            continue

        for idx, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if len(content) < 30:
                continue
            all_chunks.append({
                "content"       : content,
                "source_file"   : md_path.name,
                "chunk_index"   : idx,
                "chunk_size"    : len(content),
                "breakpoint_type"  : breakpoint_type,
                "breakpoint_amount": breakpoint_amount,
            })

        print(f"  [{i:>3}/{len(md_files)}] {md_path.name}: {len(chunks)} semantic chunks")

    elapsed = time.time() - start
    print(f"\nTotal time : {elapsed:.1f}s")
    return all_chunks


def main(breakpoint_type: str, breakpoint_amount: float):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    chunks = parse_semantic(RAW_DOCS_DIR, breakpoint_type, breakpoint_amount)

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    sizes = [c["chunk_size"] for c in chunks]
    print(f"\n✅ Saved {len(chunks)} semantic chunks → {CHUNKS_FILE}")
    print(f"   Avg chunk size : {sum(sizes) // len(sizes)} chars")
    print(f"   Min / Max      : {min(sizes)} / {max(sizes)} chars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic chunking of Braintree docs.")
    parser.add_argument(
        "--breakpoint-type", type=str, default="percentile",
        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
        help="Strategy to detect semantic boundaries (default: percentile).",
    )
    parser.add_argument(
        "--breakpoint-amount", type=float, default=90,
        help="Threshold value (default: 90 for percentile).",
    )
    args = parser.parse_args()
    main(args.breakpoint_type, args.breakpoint_amount)
