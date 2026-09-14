"""`create_dossier` over MCP produces the same result as calling `run_dossier`
directly — the MCP tool is a thin wrapper over library code, not a fork."""

from conftest import mcp_client
from praxis.graph import arun_dossier
from praxis.llm import FakeStructuredLLM
from praxis.schemas import DossierRequest


async def test_create_dossier_over_mcp_matches_run_dossier():
    # arun_dossier, not the sync run_dossier wrapper — this test is already
    # inside an event loop (pytest-asyncio), and run_dossier's asyncio.run()
    # cannot be called from within a running loop.
    direct = await arun_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    async with mcp_client() as session:
        over_mcp = await session.call_tool("create_dossier", {"subject": "Acme Robotics"})

    mcp_response = over_mcp.structuredContent
    direct_response = direct.model_dump(mode="json")

    assert mcp_response["memo"] == direct_response["memo"]
    assert mcp_response["verification"] == direct_response["verification"]
    assert mcp_response["plan"] == direct_response["plan"]
    assert mcp_response["evidence_count"] == direct_response["evidence_count"]
    assert mcp_response["sources"] == direct_response["sources"]
