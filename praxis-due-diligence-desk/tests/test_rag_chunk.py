from praxis.rag.chunk import chunk_document
from praxis.rag.models import Page, SourceDoc


def _doc(text: str, kind: str = "text") -> SourceDoc:
    return SourceDoc(
        source_id="acme-1",
        title="Acme",
        uri="inline:x",
        kind=kind,
        pages=[Page(number=1, text=text)],
    )


def test_short_text_is_one_chunk():
    chunks = chunk_document(_doc("Acme Robotics makes grippers."))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "acme-1::p1::0"
    assert chunks[0].locator == "chunk 1"
    assert chunks[0].raw_text == chunks[0].text


def test_long_text_splits_with_overlap():
    body = " ".join(f"sentence{i}." for i in range(400))
    chunks = chunk_document(_doc(body), size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c.raw_text) <= 340 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_pdf_locator_uses_page_numbers():
    doc = SourceDoc(
        source_id="r-1",
        title="Report",
        uri="x.pdf",
        kind="pdf",
        pages=[Page(number=1, text="alpha " * 50), Page(number=2, text="beta " * 50)],
    )
    chunks = chunk_document(doc, size=120, overlap=20)
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
    assert all(c.locator == f"p.{c.page}" for c in chunks)
