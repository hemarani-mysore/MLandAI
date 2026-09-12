import os
import uuid

os.environ.setdefault("PRAXIS_LLM_PROVIDER", "fake")
os.environ.setdefault("PRAXIS_EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("PRAXIS_CHECKPOINT_DB_PATH", ":memory:")  # never touch a real file in tests

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from praxis.api.deps import corpus_dep, db_session_dep  # noqa: E402
from praxis.api.main import app  # noqa: E402
from praxis.db import Base  # noqa: E402
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


@pytest_asyncio.fixture
async def db_session():
    """A fresh, isolated in-memory SQLite session per test. `StaticPool` shares
    one connection across the pool — without it, each pooled connection would
    see its own empty `:memory:` database."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def client(corpus, db_session):
    """A TestClient with the corpus and DB dependencies swapped for isolated,
    per-test fakes — shared by every API test module."""
    app.dependency_overrides[corpus_dep] = lambda: corpus

    async def _db_override():
        yield db_session

    app.dependency_overrides[db_session_dep] = _db_override
    yield TestClient(app)
    app.dependency_overrides.clear()
