# PayPal / Braintree RAG — Setup & Run Guide

## Prerequisites

- Python 3.11+
- An OpenAI API key (for embeddings)

---

## 1. Install Dependencies

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters \
            faiss-cpu openai python-dotenv beautifulsoup4 markdownify requests \
            rank-bm25 pymupdf pydantic
```

---

## 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env   # if example exists, otherwise create manually
```

Edit `.env` and add your key:

```
OPENAI_API_KEY=sk-your-key-here
```

---

## 3. Scrape the Braintree Developer Docs

```bash
python corpus/scraper.py
```

**What it does:**
- BFS-crawls up to 300 pages under `developer.paypal.com/braintree/docs/`
- Extracts main article content from each page
- Saves each page as a `.md` file in `corpus/data/raw_docs/`

**Output:** `corpus/data/raw_docs/*.md` — one file per documentation page

Optional flags:
```bash
python corpus/scraper.py --max-pages 100   # crawl fewer pages
python corpus/scraper.py --delay 1.0       # slower/more polite (default 0.5s)
```

> **Note:** This replaces the previous step of cloning `braintree_python`. The braintree_python repo only contains changelogs; the actual how-to guides live on the developer docs site.

---

## 4. Parse Scraped Pages into Chunks

```bash
python corpus/parser.py --docs-dir corpus/data/raw_docs
```

**What it does:**
- Walks all `.md` files under `corpus/data/raw_docs/`
- Splits each file by headings (H1 / H2 / H3), then by character length
- Saves chunks to `corpus/data/chunks.jsonl`

**Output:** `corpus/data/chunks.jsonl` — one JSON line per chunk with fields:
`content`, `source_file`, `section_title`, `doc_type`, `chunk_index`

Optional flags:
```bash
python corpus/parser.py --docs-dir corpus/data/raw_docs --chunk-size 800 --overlap 150
```

---

## 5. Embed and Build FAISS Index

```bash
python corpus/embed.py
```

**What it does:**
- Loads `corpus/data/chunks.jsonl`
- Embeds each chunk using OpenAI `text-embedding-3-small`
- Saves the FAISS index to `corpus/data/faiss_index/`

> **Cost:** Very low — `text-embedding-3-small` is ~$0.02 per 1M tokens.
> 186 chunks ≈ fractions of a cent.

Optional flags:
```bash
python corpus/embed.py --batch-size 50     # smaller batches if hitting rate limits
python corpus/embed.py --doc-type guide    # embed only guide chunks
```

---

## 6. Find the Optimal Chunk Size (Optional)

Run `choose_chunk_size.py` to evaluate which chunk size gives the best faithfulness and relevancy for your queries:

```bash
python choose_chunk_size.py                                        # default: 256, 512, 1000, 2000
python choose_chunk_size.py --chunk-sizes 512 1000 2000           # custom sizes
python choose_chunk_size.py --num-questions 10 --sample-docs 40   # more thorough evaluation
```

This tests each chunk size by running eval questions through the RAG pipeline and scoring:
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Relevancy** — does the answer actually address the question?

Example output:
```
  Chunk Size    Avg Time (s)    Faithfulness   Relevancy
         256            1.91            70%        90%
         512            1.97            90%        70%
        1000            1.73           100%        90%
        2000            1.82            90%       100%

Best chunk size by faithfulness + relevancy: 1000
```

### Rebuilding the Index with a Different Chunk Size

> **Important:** Changing `chunk_size` inside `simple_rag.py` has **no effect** in corpus mode.
> `simple_rag.py` always loads the pre-built FAISS index from disk — it does not re-chunk.
> The `chunk_size` parameter in `simple_rag.py` only applies when using `--path` (PDF mode).

To actually use a different chunk size, rebuild the index:

```bash
# Re-parse with the new chunk size
python corpus/parser.py --docs-dir corpus/data/raw_docs --chunk-size 512

# Re-embed (overwrites the existing FAISS index)
python corpus/embed.py

# Now query — it will use the new 512-char chunks
python simple_rag.py --query "How do I create a transaction?"
```

---

## 7. Query the RAG

```bash
# Use the pre-built Braintree corpus (default) — returns a generated answer
python simple_rag.py --query "How do I create a transaction?"

# Also show the raw retrieved chunks alongside the answer
python simple_rag.py --query "How do I create a transaction?" --show-context

# Change number of retrieved chunks
python simple_rag.py --query "What payment methods does Braintree support?" --n_retrieved 5

# Use a PDF instead of the corpus
python simple_rag.py --path data/my_doc.pdf --query "What is Braintree?"

# Run with evaluation
python simple_rag.py --query "How do I void a transaction?" --evaluate
```

---

## Project Structure

```
paypal_braintree_rag/
├── .env                        # API keys (not committed)
├── simple_rag.py               # Step 7 — query entry point (retrieves + generates answer)
├── choose_chunk_size.py        # Step 6 — evaluate optimal chunk size
├── helper_functions.py         # Shared LangChain / FAISS utilities
├── SETUP.md                    # This file
│
├── corpus/
│   ├── scraper.py              # Step 3 — crawls Braintree docs → raw_docs/
│   ├── parser.py               # Step 4 — .md files → chunks.jsonl
│   ├── embed.py                # Step 5 — chunks.jsonl → FAISS index
│   └── data/
│       ├── raw_docs/           # Generated by scraper.py (250 .md files)
│       ├── chunks.jsonl        # Generated by parser.py
│       └── faiss_index/        # Generated by embed.py
│           ├── index.faiss
│           └── index.pkl
│
└── evaluation/
    └── evalute_rag.py          # RAG evaluation utilities
```

---

## Quick Recap

| Step | Command |
|------|---------|
| Scrape docs | `python corpus/scraper.py` |
| Parse | `python corpus/parser.py --docs-dir corpus/data/raw_docs` |
| Embed | `python corpus/embed.py` |
| Find best chunk size | `python choose_chunk_size.py` |
| Rebuild index (if chunk size changed) | `python corpus/parser.py --chunk-size <N>` + `python corpus/embed.py` |
| Query | `python simple_rag.py --query "How do I create a transaction?"` |
