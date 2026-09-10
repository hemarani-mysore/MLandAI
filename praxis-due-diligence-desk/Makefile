.PHONY: install fmt lint type test check run api demo eval-rag eval-rag-gate \
        eval-structure eval-structure-gate clean

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

check: lint type test eval-rag-gate eval-structure-gate  ## Everything CI runs

run: demo  ## alias

demo:  ## Ingest the fixture corpus, then run a grounded dossier (no keys needed)
	uv run praxis run "Acme Robotics" --ingest evals/datasets/corpus/*.md

api:  ## Serve the API on :8000
	uv run uvicorn praxis.api.main:app --reload --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
