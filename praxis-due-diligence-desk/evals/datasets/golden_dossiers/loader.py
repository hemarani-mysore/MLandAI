"""Load `evals/datasets/golden_dossiers/index.yaml` into typed `GoldenItem`s.

Each item is a directory named after its `id`, holding `sources/*.md` (the
raw source material) and, for `dev`/`test` items, a frozen `memo.md` (the
memo text the judge grades). `online` items have no frozen memo — a fresh one
is generated from `sources/` at eval time instead (see `memo_eval.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

HERE = Path(__file__).parent
INDEX = HERE / "index.yaml"

Split = Literal["dev", "test", "online"]
Label = Literal["pass", "fail"]


@dataclass(frozen=True)
class GoldenItem:
    id: str
    split: Split
    label: Label
    critique: str
    sources_dir: Path
    memo_path: Path | None

    def read_sources(self) -> str:
        """Every source doc for this item, concatenated with a filename header
        — this is what the judge (and, for `online` items, `run_dossier`'s
        ingestion) actually sees."""
        paths = sorted(self.sources_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"{self.id}: no sources/*.md found in {self.sources_dir}")
        return "\n\n".join(f"# {p.name}\n\n{p.read_text()}" for p in paths)

    def read_memo(self) -> str:
        if self.memo_path is None:
            raise ValueError(f"{self.id}: no frozen memo.md (an 'online' item — generate one)")
        return self.memo_path.read_text()


def load_index(*, split: Split | None = None) -> list[GoldenItem]:
    """All golden items, optionally filtered to one split. Validates that
    every id is unique, every referenced path actually exists, and (unless
    filtering to a single split) every split has at least one item."""
    raw = yaml.safe_load(INDEX.read_text())
    items: list[GoldenItem] = []
    seen_ids: set[str] = set()

    for entry in raw["items"]:
        item_id = entry["id"]
        if item_id in seen_ids:
            raise ValueError(f"duplicate golden item id: {item_id}")
        seen_ids.add(item_id)

        item_dir = HERE / item_id
        sources_dir = item_dir / "sources"
        if not sources_dir.is_dir():
            raise FileNotFoundError(f"{item_id}: missing sources dir {sources_dir}")

        memo_path = item_dir / "memo.md"
        items.append(
            GoldenItem(
                id=item_id,
                split=entry["split"],
                label=entry["label"],
                critique=entry["critique"].strip(),
                sources_dir=sources_dir,
                memo_path=memo_path if memo_path.is_file() else None,
            )
        )

    covered_splits = {item.split for item in items}
    if split is None and covered_splits < {"dev", "test", "online"}:
        raise ValueError(f"every split needs >=1 item, got only: {sorted(covered_splits)}")

    return [item for item in items if split is None or item.split == split]
