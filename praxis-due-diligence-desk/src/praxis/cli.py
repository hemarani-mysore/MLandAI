"""``praxis`` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from praxis.graph import run_dossier
from praxis.rag import get_corpus, ingest_source, search_corpus
from praxis.render import to_markdown
from praxis.schemas import DossierRequest


def _cmd_run(args: argparse.Namespace) -> int:
    for path in args.ingest or []:
        result = ingest_source(path=path)
        print(f"# ingested {result.title}: {result.chunks} chunks", flush=True)
    response = run_dossier(DossierRequest(subject=args.subject, depth=args.depth))
    print(response.model_dump_json(indent=2) if args.json else to_markdown(response))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    for path in args.paths:
        result = ingest_source(path=path)
        print(f"ingested {result.title}: {result.pages} pages, {result.chunks} chunks")
    stats = get_corpus().stats()
    print(f"corpus: {stats.documents} docs, {stats.chunks} chunks, v{stats.version}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    hits = search_corpus(args.query, k=args.k)
    if not hits:
        print("(no results — ingest documents first with `praxis ingest`)")
        return 0
    for i, hit in enumerate(hits, 1):
        rr = f" rerank={hit.rerank_score:.3f}" if hit.rerank_score is not None else ""
        print(f"{i}. {hit.chunk.title} {hit.chunk.locator}  (rrf={hit.score:.4f}{rr})")
        print(f"   {hit.chunk.raw_text[:200].strip()}...\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="praxis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Produce a due-diligence memo for a subject")
    run.add_argument("subject")
    run.add_argument("--depth", choices=("quick", "standard", "deep"), default="standard")
    run.add_argument("--json", action="store_true")
    run.add_argument(
        "--ingest",
        nargs="+",
        metavar="PATH",
        help="Ingest these docs into the corpus first (same process, so in-memory Qdrant works)",
    )
    run.set_defaults(func=_cmd_run)

    ingest = sub.add_parser("ingest", help="Add documents (.pdf/.md/.txt/.html) to the corpus")
    ingest.add_argument("paths", nargs="+")
    ingest.set_defaults(func=_cmd_ingest)

    search = sub.add_parser("search", help="Hybrid search the corpus")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=6)
    search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
