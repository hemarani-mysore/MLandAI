# Praxis — FastAPI + LangGraph. Multi-stage, non-root, slim. One image, three
# roles (api/worker/mcp) selected by command override — see docker-compose.yml.
FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
# Not part of the installed package (hatchling only builds src/praxis) — a
# companion directory needed at runtime to run `alembic upgrade head` inside
# the container (see docker-compose.selfhost.yml's api entrypoint).
COPY alembic ./alembic
RUN pip install --no-cache-dir .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PRAXIS_LLM_PROVIDER=fake

RUN useradd --create-home --uid 10001 praxis
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app /app

WORKDIR /app
USER praxis
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "praxis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
