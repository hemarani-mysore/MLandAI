"""`research_one`'s web-search fallback — when the corpus has nothing for a
question, an available external `web_search` MCP tool stands in for it;
without one configured, behaviour is unchanged (the `no-source` placeholder)."""

import json
import sys
from pathlib import Path

import pytest

from praxis.config import get_settings
from praxis.graph.nodes.researcher import ResearchTask, research_one
from praxis.llm import FakeStructuredLLM
from praxis.tools import load_external_tools

STUB_SERVER = Path(__file__).parent / "fixtures" / "stub_web_search_server.py"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _config(corpus, external_tools: dict) -> dict:
    return {
        "configurable": {
            "llm": FakeStructuredLLM(),
            "corpus": corpus,
            "external_tools": external_tools,
        }
    }


def _task() -> ResearchTask:
    return ResearchTask(
        subject="Acme Robotics", section="Overview", question="What is Acme's revenue?"
    )


async def test_falls_back_to_web_search_when_corpus_is_empty(corpus, monkeypatch):
    connections = {
        "stub": {"transport": "stdio", "command": sys.executable, "args": [str(STUB_SERVER)]}
    }
    monkeypatch.setenv("PRAXIS_MCP_SERVERS", json.dumps(connections))
    get_settings.cache_clear()
    tools = await load_external_tools()

    result = await research_one(_task(), _config(corpus, tools))

    evidence = result["evidence"][0]
    assert evidence.citation.source_id == "https://example.com/stub-result"
    assert evidence.citation.locator == "https://example.com/stub-result"
    assert evidence.confidence == 0.3  # unverified external source — deliberately capped
    assert evidence.question == "What is Acme's revenue?"
    assert evidence.section == "Overview"


async def test_unconfigured_stays_on_the_no_source_path(corpus):
    result = await research_one(_task(), _config(corpus, {}))

    evidence = result["evidence"][0]
    assert evidence.citation.source_id == "no-source"
    assert evidence.confidence == 0.0
