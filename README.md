# MLandAI

A collection of independent machine learning / AI projects. Each project lives on its **own branch**, not on `main` — check out the branch to see that project's code.

## Branches

| Branch | Project | What it does |
|---|---|---|
| [`IRISClassification`](https://github.com/hemarani-mysore/MLandAI/tree/IRISClassification) | Iris Flower Classification | Classic Iris dataset classification using an SVM model (scikit-learn), with the trained model and a walkthrough notebook. |
| [`titanic-ml-project`](https://github.com/hemarani-mysore/MLandAI/tree/titanic-ml-project) | Titanic Survival Prediction | End-to-end Kaggle ML pipeline — EDA → feature engineering → ensemble modeling (Logistic Regression, Random Forest, XGBoost, Voting Ensemble) → submission. Best model: **84.17%** 5-fold CV accuracy. |
| [`paypal-braintree-rag`](https://github.com/hemarani-mysore/MLandAI/tree/paypal-braintree-rag) | PayPal/Braintree RAG Pipeline | A RAG pipeline over scraped PayPal/Braintree developer docs, exploring multiple retrieval strategies: simple RAG, semantic chunking, hybrid (BM25 + embeddings) search, reranking, and reliability/evaluation techniques. |
| [`rag-test-generator`](https://github.com/hemarani-mysore/MLandAI/tree/rag-test-generator) | Agentic RAG Test Generator | Two agentic pipelines built with OpenAI + LangGraph: (1) ingests any public GitHub repo, uses RAG to find untested code paths, and auto-generates + PRs Playwright/pytest tests; (2) records a live browser scenario, generates an executable Playwright test for it, and self-heals it in a loop until it passes. |
| [`praxis-due-diligence-desk`](https://github.com/hemarani-mysore/MLandAI/tree/praxis-due-diligence-desk) | Praxis — Due-Diligence Research Desk | Production-grade multi-agent research agent (LangGraph + FastAPI) that turns a company name into a cited investment memo: parallel research sub-agents fan out over hybrid RAG (dense + BM25, reciprocal-rank fusion, reranking), and a **deterministic, LLM-free verifier grounds every citation** against the actual retrieved text before a bounded gap-fill loop re-queries and drafts. Ships with an LLM-as-judge eval suite (citation-injection + seeded-error red-team gates), OpenTelemetry cost/token tracing, and a real deploy path — Docker, Kubernetes (kustomize, validated on an ephemeral `kind` cluster in CI), Postgres/Alembic, Fly.io. |

`main` just holds this index — there's no project code here.

## Working on a project

```bash
git clone https://github.com/hemarani-mysore/MLandAI.git
cd MLandAI
git checkout <branch-name>   # e.g. git checkout rag-test-generator
```

Each project branch has its own setup instructions in that project's `README.md` (or `SETUP.md`).
