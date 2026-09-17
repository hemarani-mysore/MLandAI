# deploy/ (Phase 5 — done)

```
k8s/
  base/            namespace, api/worker/mcp deployments (one shared image,
                   command override — no frontend exists in this project),
                   services, HPA (api, 70% CPU, 2→10), PDB, NetworkPolicy,
                   ingress, a configMapGenerator, secret.example.yaml
  overlays/
    dev/           NodePort, in-cluster Qdrant+Redis, SQLite, zero secrets
                   needed — what CI's kind-smoke job actually deploys
    prod/          real ingress hosts, managed Qdrant Cloud/Upstash
                   endpoints, self-hosted qdrant/redis deleted from base;
                   praxis-secrets applied manually (never generated here)
fly/
  api.toml  mcp.toml   (no frontend.toml — no frontend exists)
```

**Live URL:** deploy-ready, not deployed — no Fly/managed-service account
exists on the machine this was built on. `PRODUCTION.md` has the exact
signup + `fly deploy` steps; the CI `deploy` job is already gated on
`main` + a `FLY_API_TOKEN` repo secret and will start working the moment
both exist. `/health` (liveness) + `/ready` (a real deep check — Qdrant,
the database, and the LLM client, each reporting its own ok/error) already
work today; see `src/praxis/api/main.py`.

**k8s in CI:** `kustomize build overlays/{dev,prod} | kubeconform -strict`
(`k8s-validate`), then a *real* ephemeral `kind` cluster (`kind-smoke`) — the
dev overlay's full rollout, a `/health`/`/ready`/`POST /dossiers` smoke test
over a real port-forward, and a NetworkPolicy check that actually connects
(an unrelated pod is refused, an `api`-labelled one succeeds) — not just a
liveness ping, and not just "the YAML looks right." Both jobs pass in CI
today.

Security posture: non-root (uid 10001, matching the Dockerfile),
`readOnlyRootFilesystem` + dropped capabilities on every praxis-owned
container (api/worker/mcp — the third-party qdrant/redis images are left
un-hardened; forcing that blind, with no Docker on this machine to verify
against, risked a StatefulSet that never comes up), an `emptyDir` at `/data`
(the LangGraph checkpointer always needs a writable SQLite path regardless of
the main DB backend) and `/tmp` (readOnlyRootFilesystem leaves `/tmp`
read-only too otherwise), NetworkPolicy restricting Qdrant to
api/worker/mcp and Redis to api/worker.

**GitHub Actions workflows live one level above this directory**, at the
monorepo root (`../.github/workflows/`) — not inside
`praxis-due-diligence-desk/.github/`. GitHub Actions never discovers
workflows in a subdirectory; three phases of jobs were built and edited in
the wrong (dead) location before this was caught and fixed in Phase 5. See
`docs/IMPLEMENTATION_PLAN.md`'s Phase 5 section for the full story.
