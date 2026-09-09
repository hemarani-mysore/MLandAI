# deploy/ (Phase 5)

```
k8s/
  base/            namespace, deployments (api, mcp, worker, frontend), services,
                   hpa, pdb, networkpolicy, ingress, configmap, secrets.example.yaml
  overlays/
    dev/           kustomize overlay for a local `kind` cluster
    prod/          kustomize overlay for the managed target
fly/
  api.toml  mcp.toml  frontend.toml
```

**Live URL:** `api` + `mcp` + `frontend` on Fly.io, Qdrant Cloud + Neon Postgres
free tiers. `/health` (liveness) + `/ready` (checks Qdrant + Postgres + one LLM ping).

**k8s in CI:** the manifests are validated with `kubeconform` + `kustomize build`,
then applied to an ephemeral `kind` cluster in GitHub Actions with a rollout +
`/health` smoke test — so "runs on Kubernetes" is proven without paying for a cluster.

Security posture copied from `awesome-ai-apps/mcp_ai_agents/mcp_toolbox_security_agent`:
non-root, `readOnlyRootFilesystem`, dropped caps, NetworkPolicy so only `api`/`worker`
can reach the datastores.
