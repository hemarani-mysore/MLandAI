"""Turn a file / URL / inline text into a ``SourceDoc`` with page-level text."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from praxis.rag.models import Page, SourceDoc
from praxis.rag.text import slugify

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WS.sub(" ", line).rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _source_id(title: str, uri: str) -> str:
    digest = hashlib.sha1(uri.encode()).hexdigest()[:6]
    return f"{slugify(title)}-{digest}"


def _pages_from_pdf(path: Path) -> list[Page]:
    import pymupdf  # PyMuPDF

    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append(Page(number=i, text=text))
    return pages or [Page(number=1, text="")]


def _paginate_text(text: str, *, chars: int = 3000) -> list[Page]:
    text = _clean(text)
    if len(text) <= chars:
        return [Page(number=1, text=text)]
    pages: list[Page] = []
    buf: list[str] = []
    num = 1
    for para in text.split("\n\n"):
        if buf and sum(len(p) for p in buf) + len(para) > chars:
            pages.append(Page(number=num, text="\n\n".join(buf)))
            buf, num = [], num + 1
        buf.append(para)
    if buf:
        pages.append(Page(number=num, text="\n\n".join(buf)))
    return pages


def parse_source(
    *,
    path: str | None = None,
    text: str | None = None,
    title: str | None = None,
    uri: str | None = None,
) -> SourceDoc:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        title = title or p.stem.replace("_", " ").replace("-", " ").title()
        uri = uri or str(p.resolve())
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            return SourceDoc(
                source_id=_source_id(title, uri),
                title=title,
                uri=uri,
                kind="pdf",
                pages=_pages_from_pdf(p),
            )
        body = p.read_text(encoding="utf-8", errors="replace")
        if suffix in {".html", ".htm"}:
            body = _HTML_TAG.sub(" ", body)
            kind = "html"
        elif suffix in {".md", ".markdown"}:
            kind = "markdown"
        else:
            kind = "text"
        return SourceDoc(
            source_id=_source_id(title, uri),
            title=title,
            uri=uri,
            kind=kind,
            pages=_paginate_text(body),
        )

    if text is not None:
        title = title or "Inline document"
        digest = hashlib.sha1(text.encode()).hexdigest()[:10]
        uri = uri or f"inline:{digest}"
        return SourceDoc(
            source_id=_source_id(title, uri),
            title=title,
            uri=uri,
            kind="text",
            pages=_paginate_text(text),
        )

    raise ValueError("parse_source needs either `path` or `text`")
