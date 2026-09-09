import os
import uuid

os.environ.setdefault("PRAXIS_LLM_PROVIDER", "fake")
os.environ.setdefault("PRAXIS_EMBEDDING_PROVIDER", "fake")

import pytest  # noqa: E402

from praxis.rag import clear_cache  # noqa: E402
from praxis.rag.corpus import Corpus  # noqa: E402
from praxis.rag.embed import HashEmbedder  # noqa: E402
from praxis.rag.rerank import LexicalReranker  # noqa: E402
from praxis.rag.vector_store import QdrantVectorStore  # noqa: E402


def make_corpus(dim: int = 64) -> Corpus:
    embedder = HashEmbedder(dim)
    return Corpus(
        embedder=embedder,
        vector_store=QdrantVectorStore(collection=f"test_{uuid.uuid4().hex[:10]}", dim=dim),
        reranker=LexicalReranker(),
    )


@pytest.fixture
def corpus() -> Corpus:
    return make_corpus()


@pytest.fixture(autouse=True)
def _isolate_corpus_state():
    """Each test gets a fresh, empty process-wide corpus and retrieval cache."""
    from praxis.rag import corpus as _corpus_mod

    _corpus_mod.get_corpus.cache_clear()
    clear_cache()
    yield
    _corpus_mod.get_corpus.cache_clear()
    clear_cache()
