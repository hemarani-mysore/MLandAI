"""GET /dossiers/stream — node events over SSE, persisted the same as a run."""

import json


def _parse_sse(lines: list[str]) -> list[tuple[str, dict]]:
    frames = []
    event = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            frames.append((event, json.loads(line[len("data: ") :])))
    return frames


def test_stream_emits_node_events_then_completes(client):
    with client.stream("GET", "/dossiers/stream", params={"subject": "Acme Robotics"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in r.iter_lines() if line]

    frames = _parse_sse(lines)
    kinds = [kind for kind, _ in frames]

    assert kinds.count("node_start") + kinds.count("node_end") >= 7
    assert kinds[-1] == "complete"
    assert "error" not in kinds

    complete_payload = frames[-1][1]
    assert complete_payload["memo"]["subject"] == "Acme Robotics"
    assert complete_payload["verification"]["hallucinated_citation_rate"] == 0.0


def test_stream_persists_the_run_and_its_events(client):
    with client.stream("GET", "/dossiers/stream", params={"subject": "Acme Robotics"}) as r:
        list(r.iter_lines())  # drain the stream

    listed = client.get("/dossiers").json()
    assert len(listed) == 1
    assert listed[0]["status"] == "succeeded"
    assert listed[0]["memo"]["subject"] == "Acme Robotics"

    events = client.get(f"/dossiers/{listed[0]['id']}/events").json()
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[-1]["kind"] == "complete"
    assert sum(e["kind"] in ("node_start", "node_end") for e in events) >= 7


def test_stream_requires_a_subject(client):
    r = client.get("/dossiers/stream", params={"subject": "x"})
    assert r.status_code == 422
