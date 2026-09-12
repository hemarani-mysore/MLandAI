from fastapi.testclient import TestClient

from praxis.api.main import app


def test_health():
    assert TestClient(app).get("/health").json() == {"ok": True}


def test_ready_reports_corpus(client):
    body = client.get("/ready").json()
    assert body["ok"] is True
    assert body["corpus"]["chunks"] == 0


def test_create_dossier_returns_full_response(client):
    r = client.post("/dossiers", json={"subject": "Globex Corp", "depth": "quick"})
    assert r.status_code == 200
    body = r.json()
    assert body["memo"]["subject"] == "Globex Corp"
    assert body["plan"]["sub_questions"]
    assert body["evidence_count"] > 0
    assert body["sources"] == []  # empty corpus


def test_markdown_endpoint(client):
    r = client.post("/dossiers.md", json={"subject": "Globex Corp"})
    assert r.status_code == 200
    assert r.text.startswith("# Due-Diligence Memo — Globex Corp")


def test_bad_request_is_rejected(client):
    assert client.post("/dossiers", json={"subject": "G"}).status_code == 422


def test_corpus_ingest_search_and_dossier(client):
    ing = client.post(
        "/corpus/documents",
        json={
            "text": "Acme Robotics 2025 revenue was 42 million dollars. Acme holds 14 US patents.",
            "title": "Acme 10-K",
        },
    )
    assert ing.status_code == 200
    assert ing.json()["chunks"] >= 1

    stats = client.get("/corpus/stats").json()
    assert stats["documents"] == 1

    hits = client.post("/corpus/search", json={"query": "acme revenue", "k": 3}).json()
    assert hits and hits[0]["chunk"]["title"] == "Acme 10-K"

    dossier = client.post("/dossiers", json={"subject": "Acme Robotics"}).json()
    assert dossier["sources"]  # grounded in the ingested doc


def test_corpus_ingest_requires_input(client):
    assert client.post("/corpus/documents", json={}).status_code == 422


def test_corpus_state_is_isolated_per_test(client):
    # the previous test ingested a doc; this fresh client must not see it
    assert client.get("/corpus/stats").json()["chunks"] == 0


def test_ingest_missing_file_is_404(client):
    r = client.post("/corpus/documents", json={"path": "/no/such/file.pdf"})
    assert r.status_code == 404
