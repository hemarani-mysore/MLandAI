import pytest

from praxis.rag.parse import parse_source


def test_parse_inline_text():
    doc = parse_source(text="Acme Robotics makes grippers.", title="Note")
    assert doc.kind == "text"
    assert doc.source_id.startswith("note-")
    assert doc.pages[0].text == "Acme Robotics makes grippers."


def test_long_text_paginates():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(20))
    doc = parse_source(text=body, title="Big")
    assert len(doc.pages) > 1
    assert [p.number for p in doc.pages] == list(range(1, len(doc.pages) + 1))


def test_parse_markdown_file(tmp_path):
    p = tmp_path / "acme_notes.md"
    p.write_text("# Acme\n\nRevenue was $42M.\n")
    doc = parse_source(path=str(p))
    assert doc.kind == "markdown"
    assert doc.title == "Acme Notes"
    assert "Revenue was $42M." in doc.text


def test_parse_pdf_roundtrip(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "report.pdf"
    d = pymupdf.open()
    page = d.new_page()
    page.insert_text((72, 72), "Acme Robotics reported revenue of $42M in 2025.")
    d.save(str(pdf))
    d.close()

    doc = parse_source(path=str(pdf))
    assert doc.kind == "pdf"
    assert doc.pages[0].number == 1
    assert "42M" in doc.pages[0].text


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_source(path="/no/such/file.pdf")


def test_needs_input():
    with pytest.raises(ValueError):
        parse_source()
