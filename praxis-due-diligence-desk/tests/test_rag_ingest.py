from praxis.rag import ingest_source
from praxis.rag.contextual import contextualize
from praxis.rag.models import Page, SourceDoc


def test_ingest_inline_text(corpus):
    result = ingest_source(
        text="Acme Robotics revenue was 42 million.", title="Acme", corpus=corpus
    )
    assert result.chunks >= 1
    assert result.corpus_version == 1
    assert result.source_id.startswith("acme-")


def test_ingest_file(tmp_path, corpus):
    p = tmp_path / "globex.md"
    p.write_text("Globex raised a Series C led by Sequoia.\n")
    result = ingest_source(path=str(p), corpus=corpus)
    assert result.chunks >= 1
    assert corpus.stats().documents == 1


def test_contextualize_prefixes_chunks():
    from praxis.rag.chunk import chunk_document

    doc = SourceDoc(
        source_id="a-1",
        title="Acme 10-K",
        uri="x",
        kind="text",
        pages=[Page(number=1, text="Revenue was 42 million dollars this year.")],
    )
    chunks = chunk_document(doc)
    out = contextualize(chunks, doc)
    assert out[0].raw_text == chunks[0].raw_text
    assert out[0].text != chunks[0].text
    assert out[0].raw_text in out[0].text
