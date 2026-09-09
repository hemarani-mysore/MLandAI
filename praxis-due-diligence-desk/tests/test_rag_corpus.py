from praxis.rag import ingest_source, search_corpus

DOCS = [
    (
        "Acme 10-K",
        "Acme Robotics 2025 revenue was 42 million dollars, gross margin 58 percent. "
        "Key risk: supplier concentration in Taiwan.",
    ),
    (
        "Acme Patents",
        "Acme Robotics holds 14 US patents covering actuator control and gripper design.",
    ),
    (
        "Globex Overview",
        "Globex Corp raised a 120 million dollar Series C led by Sequoia Capital in 2024. "
        "Globex builds autonomous warehouse drones.",
    ),
    ("Nimbus Team", "Nimbus Systems was founded by three ex-Boston-Dynamics engineers in 2021."),
]


def _load(corpus):
    for title, text in DOCS:
        ingest_source(text=text, title=title, corpus=corpus)


def test_hybrid_search_discriminates(corpus):
    _load(corpus)
    cases = {
        "what was acme robotics revenue": "Acme 10-K",
        "does acme hold patents": "Acme Patents",
        "who led the globex funding round": "Globex Overview",
        "who founded nimbus": "Nimbus Team",
    }
    for query, expected_title in cases.items():
        hits = search_corpus(query, k=2, corpus=corpus)
        assert hits, query
        assert hits[0].chunk.title == expected_title, (query, [h.chunk.title for h in hits])
        assert hits[0].rerank_score is not None  # lexical reranker ran
        assert hits[0].dense_rank is not None or hits[0].sparse_rank is not None


def test_retrieved_chunk_yields_a_citation(corpus):
    _load(corpus)
    hit = search_corpus("acme revenue", k=1, corpus=corpus)[0]
    cite = hit.citation
    assert cite.source_id == hit.chunk.source_id
    assert cite.quote


def test_stats_and_cache(corpus):
    _load(corpus)
    stats = corpus.stats()
    assert stats.documents == 4
    assert stats.chunks == stats.vectors == 4
    assert stats.version == 4

    first = search_corpus("acme patents", k=2, corpus=corpus)
    again = search_corpus("acme patents", k=2, corpus=corpus)
    assert first is again  # served from cache

    ingest_source(
        text="Acme also filed 3 European patents in 2025.", title="Acme EU", corpus=corpus
    )
    after = search_corpus("acme patents", k=2, corpus=corpus)
    assert after is not first  # version bump invalidated the cache


def test_empty_corpus_returns_nothing(corpus):
    assert search_corpus("anything", corpus=corpus) == []
