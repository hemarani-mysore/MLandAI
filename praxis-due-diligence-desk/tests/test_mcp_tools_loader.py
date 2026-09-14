"""`load_external_tools()` — empty when `PRAXIS_MCP_SERVERS` is unset (the
default, unchanged corpus-only behaviour); connects to configured servers
otherwise. Uses the stub `web_search` server (`tests/fixtures/`) — no real
external API key exists on this machine."""

import json
import sys
from pathlib import Path

import pytest

from praxis.config import get_settings
from praxis.tools import load_external_tools

STUB_SERVER = Path(__file__).parent / "fixtures" / "stub_web_search_server.py"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    # get_settings() is @lru_cache'd process-wide — clear around each test so
    # a monkeypatched PRAXIS_MCP_SERVERS here never leaks into another test.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_returns_empty_when_unset(monkeypatch):
    monkeypatch.delenv("PRAXIS_MCP_SERVERS", raising=False)
    get_settings.cache_clear()

    tools = await load_external_tools()

    assert tools == {}


async def test_connects_to_a_configured_stub_server_and_invokes_it(monkeypatch):
    connections = {
        "stub": {"transport": "stdio", "command": sys.executable, "args": [str(STUB_SERVER)]}
    }
    monkeypatch.setenv("PRAXIS_MCP_SERVERS", json.dumps(connections))
    get_settings.cache_clear()

    tools = await load_external_tools()

    assert "web_search" in tools
    raw = await tools["web_search"].ainvoke({"query": "Acme Robotics revenue"})
    # langchain-mcp-adapters returns a list of content blocks; the stub's
    # single-dict result comes back as one block whose `text` is its JSON dump.
    assert "https://example.com/stub-result" in json.dumps(raw)
