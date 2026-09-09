from praxis.cli import main


def test_cli_run_markdown(capsys):
    exit_code = main(["run", "Initech", "--depth", "quick"])
    assert exit_code == 0
    assert "Due-Diligence Memo — Initech" in capsys.readouterr().out


def test_cli_run_json(capsys):
    exit_code = main(["run", "Initech", "--json"])
    assert exit_code == 0
    assert '"recommendation"' in capsys.readouterr().out
