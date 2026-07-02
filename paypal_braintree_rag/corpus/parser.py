"""
corpus/parser.py — Markdown → Chunks
======================================
Walks all .md files in a directory, splits them into logical sections
(by heading), then further splits long sections with RecursiveCharacterTextSplitter.
Saves output to corpus/data/chunks.jsonl.

Usage:
    python corpus/parser.py --docs-dir /path/to/braintree_python
    python corpus/parser.py --docs-dir ./braintree_python --chunk-size 800
"""

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKS_FILE = Path(__file__).parent / "data" / "chunks.jsonl"


# ─────────────────────────────────────────────────────────
# Markdown section splitter
# ─────────────────────────────────────────────────────────

def split_by_headings(text: str) -> list[dict]:
    """
    Split markdown text into sections at every H1/H2/H3 heading.
    Each section includes its heading as the title.
    Returns list of {title, content} dicts.
    """
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [{"title": "", "content": text.strip()}]

    sections = []
    for i, match in enumerate(matches):
        title   = match.group(2).strip()
        start   = match.start()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append({"title": title, "content": content})

    # Capture any text before the first heading
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.insert(0, {"title": "Overview", "content": preamble})

    return sections


# ─────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────

def parse_md_files(
    docs_dir   : Path,
    chunk_size : int = 1000,
    overlap    : int = 200,
) -> list[dict]:
    """
    Walk all .md files under docs_dir, split into chunks, return list of chunk dicts.

    Each chunk dict contains:
        content       : str   — the text of the chunk
        source_file   : str   — relative path of the .md file
        section_title : str   — nearest heading above the chunk
        doc_type      : str   — inferred from filename/path
        chunk_index   : int   — position within this file's chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = overlap,
        length_function = len,
    )

    md_files = sorted(docs_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found under {docs_dir}")

    print(f"Found {len(md_files)} markdown files in {docs_dir}")

    all_chunks: list[dict] = []

    for md_path in md_files:
        rel_path = str(md_path.relative_to(docs_dir))
        doc_type = _infer_doc_type(rel_path)

        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  [skip] {rel_path}: {e}")
            continue

        sections = split_by_headings(text)

        chunk_index = 0
        for section in sections:
            # Further split long sections with the character splitter
            sub_chunks = splitter.split_text(section["content"])
            for sub in sub_chunks:
                sub = sub.strip()
                if len(sub) < 30:   # skip trivially short fragments
                    continue
                all_chunks.append({
                    "content"       : sub,
                    "source_file"   : rel_path,
                    "section_title" : section["title"],
                    "doc_type"      : doc_type,
                    "chunk_index"   : chunk_index,
                })
                chunk_index += 1

        print(f"  {rel_path}: {chunk_index} chunks")

    return all_chunks


def _infer_doc_type(rel_path: str) -> str:
    path_lower = rel_path.lower()
    if "readme" in path_lower:
        return "overview"
    if "changelog" in path_lower or "history" in path_lower:
        return "changelog"
    if "example" in path_lower or "sample" in path_lower:
        return "example"
    if "test" in path_lower or "spec" in path_lower:
        return "test"
    return "guide"


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main(docs_dir: Path, chunk_size: int, overlap: int):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    chunks = parse_md_files(docs_dir, chunk_size=chunk_size, overlap=overlap)

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\n✅ Saved {len(chunks)} chunks → {CHUNKS_FILE}")

    # Quick stats
    from collections import Counter
    types = Counter(c["doc_type"] for c in chunks)
    print("\nChunk breakdown by doc_type:")
    for dt, count in types.most_common():
        print(f"  {dt:<20} {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Braintree .md files into chunks.")
    parser.add_argument("--docs-dir",   type=Path, required=True,
                        help="Root directory of the braintree_python repo (or any dir with .md files).")
    parser.add_argument("--chunk-size", type=int,  default=1000,
                        help="Max characters per chunk (default 1000).")
    parser.add_argument("--overlap",    type=int,  default=200,
                        help="Overlap between chunks (default 200).")
    args = parser.parse_args()

    main(args.docs_dir, args.chunk_size, args.overlap)
