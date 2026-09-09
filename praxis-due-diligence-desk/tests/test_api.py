from fastapi.testclient import TestClient

from praxis.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"ok": True}


def test_create_dossier_returns_full_response():
    r = client.post("/dossiers", json={"subject": "Globex Corp", "depth": "quick"})
    assert r.status_code == 200
    body = r.json()
    assert body["memo"]["subject"] == "Globex Corp"
    assert body["plan"]["sub_questions"]
    assert body["evidence_count"] > 0
    assert "verification" in body


def test_markdown_endpoint():
    r = client.post("/dossiers.md", json={"subject": "Globex Corp"})
    assert r.status_code == 200
    assert r.text.startswith("# Due-Diligence Memo — Globex Corp")


def test_bad_request_is_rejected():
    assert client.post("/dossiers", json={"subject": "G"}).status_code == 422
