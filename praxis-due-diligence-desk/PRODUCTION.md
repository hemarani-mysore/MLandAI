# Production runbook — Fly + managed services

Everything in `deploy/` is **deploy-ready, not deployed**: the Dockerfile,
`deploy/fly/*.toml`, and the CI `deploy` job all exist and are exercised by
CI (image build + push to GHCR, `kind-smoke`'s full Kubernetes rollout), but
no Fly account or managed-service account exists on the machine this was
built on, and creating one isn't something an agent can do on your behalf.
This is that missing manual step, in order.

## 1. Managed services (all have real free tiers)

| Service | Replaces | Sign up | Env var |
|---|---|---|---|
| [Neon](https://neon.tech) | Postgres | Create a project, copy the pooled connection string | `PRAXIS_DATABASE_URL=postgresql+asyncpg://...` |
| [Qdrant Cloud](https://cloud.qdrant.io) | self-hosted Qdrant | Create a free cluster, copy its URL + API key | `PRAXIS_QDRANT_URL=https://xxx.qdrant.io:6333` |
| [Upstash](https://upstash.com) | self-hosted Redis | Create a Redis database, copy the `rediss://` URL | `PRAXIS_REDIS_URL=rediss://...` |

Neon's connection string already includes `?sslmode=require` — keep it; swap
`postgresql://` for `postgresql+asyncpg://` (asyncpg's own scheme) if Neon's
copy-paste gives you the plain `postgresql://` form. Qdrant Cloud needs its
API key passed via `QDRANT_API_KEY` — `praxis`'s `QdrantVectorStore` doesn't
currently forward a separate API-key env var (it was built and tested only
against an unauthenticated local/in-cluster Qdrant); if Qdrant Cloud rejects
unauthenticated requests, `rag/vector_store.py`'s `QdrantClient(url=...)`
call needs an `api_key=` argument wired to a new `PRAXIS_QDRANT_API_KEY`
setting — a small, real gap worth closing before actually pointing at Qdrant
Cloud, not silently glossed over here.

## 2. Fly.io

```bash
# from praxis-due-diligence-desk/
brew install flyctl   # or: curl -L https://fly.io/install.sh | sh
fly auth signup        # or `fly auth login` if you already have an account

fly launch --config deploy/fly/api.toml --dockerfile Dockerfile --no-deploy
fly launch --config deploy/fly/mcp.toml --dockerfile Dockerfile --no-deploy

fly secrets set -a praxis-api \
  OPENAI_API_KEY=sk-... \
  PRAXIS_DATABASE_URL="postgresql+asyncpg://...neon.tech/praxis" \
  PRAXIS_QDRANT_URL="https://xxx.qdrant.io:6333" \
  PRAXIS_REDIS_URL="rediss://...upstash.io"

fly secrets set -a praxis-mcp \
  OPENAI_API_KEY=sk-... \
  PRAXIS_DATABASE_URL="postgresql+asyncpg://...neon.tech/praxis" \
  PRAXIS_QDRANT_URL="https://xxx.qdrant.io:6333"

# Run the migration once, against the real Postgres, before the first deploy
# — the k8s path does this via an initContainer; Fly has no equivalent, so
# it's a one-off manual step here.
fly ssh console -a praxis-api -C "alembic upgrade head" 2>/dev/null || \
  PRAXIS_DATABASE_URL="postgresql+asyncpg://...neon.tech/praxis" uv run alembic upgrade head

fly deploy --config deploy/fly/api.toml --dockerfile Dockerfile
fly deploy --config deploy/fly/mcp.toml --dockerfile Dockerfile

curl https://praxis-api.fly.dev/health
curl https://praxis-api.fly.dev/ready   # should report {"ok": true, "qdrant": ..., "database": ..., "llm": ...}
```

## 3. CI/CD

Once the two apps above exist, add `FLY_API_TOKEN` (`fly tokens create deploy
-a praxis-api`, works across both apps in the same org) as a repo secret
(`Settings -> Secrets and variables -> Actions`). The `deploy` job in the
real `.github/workflows/ci.yml` (repo root, **not**
`praxis-due-diligence-desk/.github/` — see
`docs/IMPLEMENTATION_PLAN.md`'s Phase 5 note on why that distinction
matters) is already gated on `github.ref == 'refs/heads/main'` *and* that
secret's presence — it will start deploying automatically on the next push
to `main` once both are true, no other change needed.

## 4. Rollback

```bash
fly releases -a praxis-api
fly deploy -a praxis-api --image <previous-image-ref-from-releases-list>
```

See `docs/RUNBOOK.md` for this and the other day-2 operations (reindexing the
corpus, scaling, rotating keys).
