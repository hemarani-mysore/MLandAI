.PHONY: install fmt lint type test check run api demo clean

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

check: lint type test  ## Everything CI runs

run: demo  ## alias

demo:  ## Run a dossier from the CLI (fake provider, no keys needed)
	uv run praxis run "Acme Robotics" --depth standard

api:  ## Serve the API on :8000
	uv run uvicorn praxis.api.main:app --reload --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
