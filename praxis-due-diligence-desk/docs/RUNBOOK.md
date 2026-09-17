# Runbook — day-2 operations

Commands assume `praxis-due-diligence-desk/` as the working directory unless
noted. Kubernetes commands assume `kubectl` is pointed at the right cluster
(`kubectl config current-context`) — check before running anything here
against production.

## Redeploy

**Fly:**
```bash
fly deploy --config deploy/fly/api.toml --dockerfile Dockerfile
fly deploy --config deploy/fly/mcp.toml --dockerfile Dockerfile
```
Or push to `main` with `FLY_API_TOKEN` set as a repo secret — the CI `deploy`
job does this automatically (see `PRODUCTION.md`).

**Kubernetes:**
```bash
kubectl apply -k deploy/k8s/overlays/prod
kubectl -n praxis rollout status deployment/api
```

## Rollback

**Fly:**
```bash
fly releases -a praxis-api             # find the previous working image
fly deploy -a praxis-api --image <ref>
```

**Kubernetes:**
```bash
kubectl -n praxis rollout undo deployment/api
kubectl -n praxis rollout status deployment/api
```

## Reindex the corpus

The corpus is a process-wide singleton per running `api`/`worker`/`mcp`
instance — restarting any of them drops its in-memory corpus (SQLite-backed
runs keep `DossierRun` history, but the vector index itself is not persisted
unless `PRAXIS_QDRANT_URL` points at a real, durable Qdrant). To reindex:

```bash
# via the CLI, against whatever PRAXIS_QDRANT_URL is configured
uv run praxis ingest path/to/*.pdf path/to/*.md

# or via the API (each call runs synchronously; POST /corpus/jobs for many
# documents so ingestion runs on the worker instead of blocking each request)
curl -s -XPOST https://praxis-api.fly.dev/corpus/jobs \
  -H 'content-type: application/json' -d '{"text": "...", "title": "..."}'
```

## Scale

**Kubernetes** — the `api` HPA already targets 70% CPU, 2→10 replicas
automatically:
```bash
kubectl -n praxis get hpa api          # watch it react under load
kubectl -n praxis scale deployment/worker --replicas=2  # manual only — the
  # ingestion queue isn't sharded yet; more than one worker replica risks two
  # workers racing the same arq job, not documented as safe
```

**Fly:**
```bash
fly scale count 2 -a praxis-api
```

## Rotate keys

1. Get the new key from the provider (OpenAI/Anthropic/Nebius).
2. **Fly:** `fly secrets set -a praxis-api OPENAI_API_KEY=<new>` (triggers an
   automatic redeploy). Repeat for `praxis-mcp`.
3. **Kubernetes:** edit `deploy/k8s/base/secret.example.yaml`'s filled-in
   copy (`secret.yaml`, never committed) and re-apply:
   ```bash
   kubectl apply -f deploy/k8s/base/secret.yaml
   kubectl -n praxis rollout restart deployment/api deployment/worker deployment/mcp
   ```
4. Confirm: `curl .../ready` — the `llm.ok` field reflects whether the new
   key actually validates (client construction fails fast on a bad key,
   see `src/praxis/api/main.py`'s `/ready`).
5. Revoke the old key at the provider once the rollout is confirmed healthy.

## Diagnosing a red `/ready`

`/ready`'s response names exactly which dependency is unhealthy
(`qdrant`/`database`/`llm`, each with its own `ok` + `error`) — check that
field first rather than guessing:

```bash
curl -s https://praxis-api.fly.dev/ready | python3 -m json.tool
```

- `qdrant.ok: false` — `PRAXIS_QDRANT_URL` unreachable, wrong, or (Qdrant
  Cloud) missing an API key (`PRAXIS_QDRANT_API_KEY` isn't wired up yet —
  see `PRODUCTION.md`'s known-gap note).
- `database.ok: false` — `PRAXIS_DATABASE_URL` unreachable or migrations not
  applied (`alembic upgrade head`).
- `llm.ok: false` — the configured provider key is missing or invalid.
