"""``praxis`` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from praxis.graph import run_dossier
from praxis.render import to_markdown
from praxis.schemas import DossierRequest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="praxis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Produce a due-diligence memo for a subject")
    run.add_argument("subject", help="Company / technology / vendor to research")
    run.add_argument("--depth", choices=("quick", "standard", "deep"), default="standard")
    run.add_argument("--json", action="store_true", help="Emit the raw DossierResponse JSON")

    args = parser.parse_args(argv)

    if args.command == "run":
        response = run_dossier(DossierRequest(subject=args.subject, depth=args.depth))
        if args.json:
            print(response.model_dump_json(indent=2))
        else:
            print(to_markdown(response))
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
