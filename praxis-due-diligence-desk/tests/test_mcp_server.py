"""`praxis-mcp` over its real stdio transport — a genuine subprocess, the same
way any MCP host would connect. Offline (fake provider), no Docker/Redis."""

from conftest import mcp_client
from praxis.schemas import RUBRIC_SECTIONS


async def test_lists_four_tools_three_resources_two_prompts():
    async with mcp_client() as session:
        tools = await session.list_tools()
        assert sorted(t.name for t in tools.tools) == [
            "create_dossier",
            "get_evidence",
            "ingest_document",
            "search_corpus",
        ]

        resources = await session.list_resources()
        templates = await session.list_resource_templates()
        assert [str(r.uri) for r in resources.resources] == ["corpus://stats"]
        assert sorted(t.uriTemplate for t in templates.resourceTemplates) == [
            "dossier://{run_id}",
            "dossier://{run_id}/citations",
        ]
        # "3 resources" per the acceptance criteria = 1 concrete + 2 templated
        assert len(resources.resources) + len(templates.resourceTemplates) == 3

        prompts = await session.list_prompts()
        assert sorted(p.name for p in prompts.prompts) == [
            "due-diligence-brief",
            "investment-memo",
        ]


async def test_ingest_then_search_then_evidence():
    async with mcp_client() as session:
        ingested = await session.call_tool(
            "ingest_document",
            {
                "text": "Acme Robotics 2025 revenue was $42M. Acme holds 14 US patents.",
                "title": "Acme brief",
            },
        )
        assert ingested.structuredContent["chunks"] >= 1

        hits = await session.call_tool("search_corpus", {"query": "acme revenue", "k": 3})
        assert hits.structuredContent["result"]

        evidence = await session.call_tool(
            "get_evidence", {"claim": "Acme's 2025 revenue was $42M"}
        )
        body = evidence.structuredContent
        assert body["claim"] == "Acme's 2025 revenue was $42M"
        assert body["supporting"]
        assert body["supporting"][0]["source_id"].startswith("acme-brief-")


async def test_get_evidence_on_an_empty_corpus_finds_nothing():
    async with mcp_client() as session:
        result = await session.call_tool("get_evidence", {"claim": "anything at all"})
        body = result.structuredContent
        assert body["supporting"] == []
        assert body["refuting"] == []


async def test_create_dossier_produces_all_six_sections():
    async with mcp_client() as session:
        result = await session.call_tool("create_dossier", {"subject": "Acme Robotics"})
        memo = result.structuredContent["memo"]
        assert [s["heading"] for s in memo["sections"]] == list(RUBRIC_SECTIONS)


async def test_corpus_stats_resource():
    async with mcp_client() as session:
        contents = await session.read_resource("corpus://stats")
        assert "documents" in contents.contents[0].text


async def test_unknown_dossier_resource_errors():
    from mcp.shared.exceptions import McpError

    async with mcp_client() as session:
        try:
            await session.read_resource("dossier://no-such-run")
        except McpError:
            pass
        else:
            raise AssertionError("expected an McpError for an unknown run id")


async def test_prompts_render():
    async with mcp_client() as session:
        brief = await session.get_prompt("due-diligence-brief", {"subject": "Acme Robotics"})
        assert "Acme Robotics" in brief.messages[0].content.text

        memo_prompt = await session.get_prompt("investment-memo", {"subject": "Acme Robotics"})
        assert "Acme Robotics" in memo_prompt.messages[0].content.text
