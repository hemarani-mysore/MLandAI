.PHONY: install fmt lint type test check run api demo eval-rag eval-rag-gate \
        eval-structure eval-structure-gate eval-dev eval-test eval-test-gate \
        eval-citation eval-citation-gate eval-redteam eval-redteam-gate clean

install:  ## Create .venv and install with dev extras
	uv sync --extra dev

fmt:  ## Auto-format
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint (no changes)
	uv run ruff check .
	uv run ruff format --check .

type:  ## Type-check
	uv run mypy src

test:  ## Unit tests
	uv run pytest

eval-rag:  ## Deterministic retrieval eval (report)
	uv run python evals/rag_eval.py

eval-rag-gate:  ## Retrieval eval as a pass/fail gate (used in CI)
	uv run python evals/rag_eval.py --gate

eval-structure:  ## Deterministic memo-structure eval (report)
	uv run python evals/memo_structure_eval.py

eval-structure-gate:  ## Memo-structure eval as a pass/fail gate (used in CI)
	uv run python evals/memo_structure_eval.py --gate

eval-dev:  ## LLM-as-judge vs expert labels, dev split (live model, calibration)
	uv run python evals/memo_eval.py --split dev

eval-test:  ## LLM-as-judge vs expert labels, test split (recorded, offline)
	uv run python evals/memo_eval.py --split test

eval-test-gate:  ## Judge-vs-expert F1 gate on the recorded test split (used in CI)
	uv run python evals/memo_eval.py --split test --gate

eval-citation:  ## Citation-integrity self-check (report)
	uv run python evals/citation_eval.py

eval-citation-gate:  ## Citation-integrity self-check as a gate (used in CI)
	uv run python evals/citation_eval.py --gate

eval-redteam:  ## Seeded-error detection, recorded fixtures (report)
	uv run python evals/redteam_eval.py

eval-redteam-gate:  ## Seeded-error detection rate as a gate (used in CI)
	uv run python evals/redteam_eval.py --gate

check: lint type test eval-rag-gate eval-structure-gate eval-test-gate \
       eval-citation-gate eval-redteam-gate  ## Everything CI runs

run: demo  ## alias

demo:  ## Ingest the fixture corpus, then run a grounded dossier (no keys needed)
	uv run praxis run "Acme Robotics" --ingest evals/datasets/corpus/*.md

api:  ## Serve the API on :8000
	uv run uvicorn praxis.api.main:app --reload --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
