"""External MCP tools the researcher may fall back to when the local corpus
has nothing for a question — e.g. a real web-search server (Exa) in
production, or a stub in tests (no such key exists on this machine).

``PRAXIS_MCP_SERVERS`` (JSON, empty by default) configures zero or more
servers, keyed by name, values shaped like `langchain_mcp_adapters`'
``StdioConnection``/``SSEConnection``/etc. Empty/unset -> ``{}``, so
``research_one`` behaves exactly as it did before this existed.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from praxis.config import get_settings


async def load_external_tools() -> dict[str, BaseTool]:
    """Connect to every configured MCP server and return its tools by name."""
    raw = get_settings().mcp_servers
    if not raw.strip():
        return {}
    connections = json.loads(raw)
    client = MultiServerMCPClient(connections)
    tools = await client.get_tools()
    return {tool.name: tool for tool in tools}
