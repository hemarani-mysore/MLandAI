"""`evals/datasets/golden_dossiers/loader.py` — index parses, every referenced
path resolves, and split coverage is what the eval harness expects."""

import importlib.util
import sys
from pathlib import Path

_LOADER = Path(__file__).parents[1] / "evals" / "datasets" / "golden_dossiers" / "loader.py"
_spec = importlib.util.spec_from_file_location("golden_dossiers_loader", _LOADER)
assert _spec and _spec.loader
loader = importlib.util.module_from_spec(_spec)
# Register before exec_module — loader.py's `@dataclass` needs `sys.modules
# [cls.__module__]` to resolve its (string, via `from __future__ import
# annotations`) type hints; skipping this crashes with an obscure
# `AttributeError: 'NoneType' object has no attribute '__dict__'`. The
# official importlib "import from a file path" recipe includes this line for
# exactly this reason (dataclasses/pickling need the module registered).
sys.modules[_spec.name] = loader
_spec.loader.exec_module(loader)


def test_loads_eight_items_with_unique_ids():
    items = loader.load_index()
    assert len(items) == 8
    assert len({item.id for item in items}) == 8


def test_every_split_has_at_least_one_item():
    items = loader.load_index()
    splits = {item.split for item in items}
    assert splits == {"dev", "test", "online"}


def test_dev_and_test_items_have_a_frozen_memo_online_does_not():
    for item in loader.load_index():
        if item.split == "online":
            assert item.memo_path is None
        else:
            assert item.memo_path is not None
            assert item.memo_path.is_file()


def test_sources_directory_has_at_least_one_markdown_file_per_item():
    for item in loader.load_index():
        assert item.sources_dir.is_dir()
        assert list(item.sources_dir.glob("*.md"))


def test_split_filter_returns_only_that_split():
    test_items = loader.load_index(split="test")
    assert test_items
    assert all(item.split == "test" for item in test_items)


def test_read_sources_concatenates_every_source_file():
    item = next(i for i in loader.load_index() if i.id == "driftwood-materials")
    text = item.read_sources()
    assert "company-overview.md" in text
    assert "customer-update.md" in text
    assert "Federal Trade Commission" in text


def test_read_memo_raises_for_an_online_item():
    item = next(i for i in loader.load_index() if i.split == "online")
    try:
        item.read_memo()
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError reading an online item's (nonexistent) memo")


def test_labels_are_pass_or_fail():
    for item in loader.load_index():
        assert item.label in ("pass", "fail")


def test_two_deliberate_fail_items_exist():
    fails = [item.id for item in loader.load_index() if item.label == "fail"]
    assert set(fails) == {"driftwood-materials", "ferrovia-rail-tech"}
