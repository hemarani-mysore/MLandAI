from praxis.cli import main


def test_cli_run_markdown(capsys):
    assert main(["run", "Initech", "--depth", "quick"]) == 0
    assert "Due-Diligence Memo — Initech" in capsys.readouterr().out


def test_cli_run_json(capsys):
    assert main(["run", "Initech", "--json"]) == 0
    assert '"recommendation"' in capsys.readouterr().out


def test_cli_ingest_then_search(tmp_path, capsys):
    doc = tmp_path / "acme.md"
    doc.write_text("Acme Robotics 2025 revenue was 42 million dollars. Acme holds 14 US patents.\n")

    assert main(["ingest", str(doc)]) == 0
    out = capsys.readouterr().out
    assert "1 chunks" in out or "chunks" in out

    assert main(["search", "acme revenue", "-k", "2"]) == 0
    assert "Acme" in capsys.readouterr().out


def test_cli_search_empty_corpus(capsys):
    assert main(["search", "anything"]) == 0
    assert "no results" in capsys.readouterr().out
