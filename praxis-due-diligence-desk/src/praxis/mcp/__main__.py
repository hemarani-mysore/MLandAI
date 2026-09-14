"""``python -m praxis.mcp`` — run the stdio MCP server. Also reachable via the
`praxis mcp` CLI subcommand (`praxis/cli.py`)."""

from praxis.mcp.server import main

if __name__ == "__main__":
    main()
