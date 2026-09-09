from praxis.rag.bm25 import BM25Index
from praxis.rag.fuse import rrf


def test_bm25_ranks_by_term_match():
    idx = BM25Index()
    idx.add("a", "Acme Robotics revenue grew to forty two million dollars")
    idx.add("b", "Globex raised a Series C led by Sequoia")
    idx.add("c", "The weather in Seattle is often rainy")

    hits = idx.search("what was acme revenue", k=3)
    assert hits[0][0] == "a"
    assert "c" not in [doc_id for doc_id, _ in hits]


def test_bm25_remove_and_readd():
    idx = BM25Index()
    idx.add("a", "patents and intellectual property")
    idx.add("a", "patents and intellectual property portfolio")  # replace
    assert len(idx) == 1
    idx.remove("a")
    assert len(idx) == 0
    assert idx.search("patents", k=5) == []


def test_bm25_empty_query_terms():
    idx = BM25Index()
    idx.add("a", "hello world")
    assert idx.search("zzz qqq", k=5) == []


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]
