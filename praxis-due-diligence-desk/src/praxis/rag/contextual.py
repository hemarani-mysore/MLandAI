"""Optional contextual augmentation.

Prepend a one-sentence, LLM-written situating context to each chunk before it is
embedded (Anthropic's "contextual retrieval"). Off by default
(``PRAXIS_CONTEXTUAL_CHUNKS``) because it costs one LLM call per chunk; the
deployed config turns it on.
"""

from __future__ import annotations

from praxis.llm import StructuredLLM, get_llm
from praxis.rag.models import Chunk, SourceDoc

_SYSTEM = (
    "You situate a document excerpt for a retrieval index. Given the document "
    "title and one chunk, reply with a single short sentence naming what the "
    "chunk is about and where it sits in the document. No preamble."
)


def contextualize(
    chunks: list[Chunk], doc: SourceDoc, *, llm: StructuredLLM | None = None
) -> list[Chunk]:
    llm = llm or get_llm()
    out: list[Chunk] = []
    for ch in chunks:
        context = llm.complete(
            system=_SYSTEM,
            user=f"Document: {doc.title}\nChunk:\n{ch.raw_text}",
            role="contextualize",
        ).strip()
        out.append(ch.model_copy(update={"text": f"{context}\n\n{ch.raw_text}"}))
    return out
