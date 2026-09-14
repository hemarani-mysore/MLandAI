"""A tiny stand-in for a real web-search MCP server (e.g. Exa), used only in
tests — no real external API key exists on this machine. Exposes one
`web_search` tool with a canned result, run over stdio like any real one."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stub-web-search")


@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Search the web (canned, for tests)."""
    return [
        {
            "title": f"Stub result for: {query}",
            "url": "https://example.com/stub-result",
            "snippet": f"A web search stand-in result about {query}, for offline tests.",
        }
    ]


if __name__ == "__main__":
    mcp.run("stdio")
