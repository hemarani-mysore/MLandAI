"""POST /dossiers persists a DossierRun; GET /dossiers[/{id}] read it back."""


def test_create_dossier_persists_a_run(client):
    posted = client.post("/dossiers", json={"subject": "Acme Robotics"}).json()

    # DossierResponse doesn't carry the run id, so fetch it via the list.
    listed = client.get("/dossiers").json()
    assert len(listed) == 1
    record = listed[0]

    assert record["status"] == "succeeded"
    assert record["subject"] == "Acme Robotics"
    assert record["memo"]["subject"] == "Acme Robotics"
    assert record["memo"] == posted["memo"]
    assert record["verification"] == posted["verification"]
    assert record["finished_at"] is not None

    detail = client.get(f"/dossiers/{record['id']}")
    assert detail.status_code == 200
    assert detail.json() == record


def test_list_dossiers_returns_newest_first(client):
    client.post("/dossiers", json={"subject": "Acme Robotics"})
    client.post("/dossiers", json={"subject": "Globex Corp"})

    listed = client.get("/dossiers").json()

    assert [r["subject"] for r in listed] == ["Globex Corp", "Acme Robotics"]


def test_list_dossiers_respects_limit(client):
    for subject in ("Acme Robotics", "Globex Corp", "Nimbus Systems"):
        client.post("/dossiers", json={"subject": subject})

    listed = client.get("/dossiers", params={"limit": 2}).json()

    assert len(listed) == 2


def test_get_unknown_run_is_404(client):
    assert client.get("/dossiers/no-such-run").status_code == 404


def test_markdown_endpoint_also_persists(client):
    r = client.post("/dossiers.md", json={"subject": "Acme Robotics"})
    assert r.status_code == 200

    listed = client.get("/dossiers").json()
    assert len(listed) == 1
    assert listed[0]["status"] == "succeeded"
