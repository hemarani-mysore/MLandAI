"""`GET /dossiers/stream` — watch a dossier run node by node instead of waiting
for one blocking response.

Persists the same way `POST /dossiers` does (a `DossierRun` row, checkpointed
under its own id) plus a `RunEvent` per frame, so a stream that was watched
live is still inspectable afterwards via `GET /dossiers/{id}` and
`GET /dossiers/{id}/events`.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from praxis.config import get_settings
from praxis.db import DossierRun, RunEvent
from praxis.graph.build import NODE_NAMES, build_graph, to_response
from praxis.graph.checkpoint import open_checkpointer
from praxis.graph.state import initial_state
from praxis.llm import get_llm
from praxis.rag import Corpus
from praxis.schemas import DossierRequest


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def _serialize_partial(output: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize(v) for k, v in output.items()}


async def stream_dossier(
    request: DossierRequest, corpus: Corpus, session: AsyncSession
) -> AsyncIterator[str]:
    run = DossierRun(subject=request.subject, depth=request.depth, status="running")
    session.add(run)
    await session.commit()

    seq = 0

    async def persist(kind: str, node: str | None, payload: dict[str, Any] | None) -> None:
        nonlocal seq
        session.add(RunEvent(run_id=run.id, seq=seq, kind=kind, node=node, payload=payload))
        seq += 1
        await session.commit()

    settings = get_settings()
    config = {
        "configurable": {"llm": get_llm(), "corpus": corpus, "thread_id": run.id},
        "max_concurrency": settings.research_concurrency,
    }
    state = initial_state(request.subject, request.depth)

    try:
        async with open_checkpointer(settings.checkpoint_db_path) as checkpointer:
            graph = build_graph(checkpointer)
            async for event in graph.astream_events(state, config=config, version="v2"):
                name = event["name"]
                if name not in NODE_NAMES:
                    continue
                if event["event"] == "on_chain_start":
                    await persist("node_start", name, None)
                    yield _sse_frame("node_start", {"node": name})
                elif event["event"] == "on_chain_end":
                    partial = _serialize_partial(event["data"].get("output") or {})
                    await persist("node_end", name, partial)
                    yield _sse_frame("node_end", {"node": name, "partial": partial})

            snapshot = await graph.aget_state({"configurable": {"thread_id": run.id}})
            response = to_response(request, snapshot.values)
    except Exception as exc:  # noqa: BLE001 - the stream itself is the error channel now
        run.status = "failed"
        run.error = str(exc)
        run.finished_at = dt.datetime.now(dt.UTC)
        await session.commit()
        await persist("error", None, {"message": str(exc)})
        yield _sse_frame("error", {"message": str(exc)})
        return

    run.status = "succeeded"
    run.memo = response.memo.model_dump(mode="json")
    run.verification = response.verification.model_dump(mode="json")
    run.sources = response.sources
    run.finished_at = dt.datetime.now(dt.UTC)
    await session.commit()

    payload = response.model_dump(mode="json")
    await persist("complete", None, payload)
    yield _sse_frame("complete", payload)
