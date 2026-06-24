"""
agents/embedding_agent.py — Code Chunking & Embedding Agent
============================================================
Responsible for:
  - Receiving a list of CodeFile objects from the ingestion agent
  - Chunking them into meaningful pieces (via code_chunker)
  - Embedding each chunk using Google's text-embedding model
  - Storing embeddings + metadata in ChromaDB
  - Providing a retrieval function: query → top-k relevant chunks

This is the SECOND agent in the pipeline.
Input  : List[CodeFile]  (from ingestion_agent)
Output : ChromaDB collection (persisted to disk) + retrieval function
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from google import genai
from google.genai import types, errors as genai_errors
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

from config import (
    GOOGLE_API_KEY,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    GEMINI_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)
from utils.file_parser import CodeFile
from utils.code_chunker import chunk_all_files, CodeChunk

console = Console()

_client: genai.Client | None = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


# ─────────────────────────────────────────────
# Embedding helper
# ─────────────────────────────────────────────

# Free tier: 100 requests/minute → pace at ~92/min to stay safely under the limit
_INTER_REQUEST_DELAY = 0.65   # seconds between each embed call
_MAX_RETRIES         = 6


def embed_text(text: str) -> list[float]:
    """
    Call Google's embedding API to convert text → vector, with retry-on-429.

    We use task_type="retrieval_document" when embedding chunks for storage,
    and "retrieval_query" when embedding a search query. This tells the model
    to optimize the vector for that use case.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            result = _get_client().models.embed_content(
                model   = GEMINI_EMBEDDING_MODEL,
                contents = text,
                config  = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return result.embeddings[0].values
        except genai_errors.ClientError as exc:
            if exc.code != 429 or attempt == _MAX_RETRIES - 1:
                raise
            m = re.search(r"retry in (\d+)", str(exc), re.IGNORECASE)
            wait = int(m.group(1)) + 5 if m else 60
            console.print(
                f"\n[yellow]⏳ Rate-limit hit (attempt {attempt + 1}/{_MAX_RETRIES}). "
                f"Waiting {wait}s...[/yellow]"
            )
            time.sleep(wait)
    return []  # unreachable, but satisfies type checker


def embed_query(query: str) -> list[float]:
    """Embed a search query (different task_type than document embedding)."""
    result = _get_client().models.embed_content(
        model    = GEMINI_EMBEDDING_MODEL,
        contents = query,
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


# ─────────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────────

def get_chroma_collection(reset: bool = False) -> chromadb.Collection:
    """
    Create or load the ChromaDB collection.

    ChromaDB is a local vector database — it stores vectors on disk in CHROMA_DIR.
    A "collection" is like a table — ours is named "codebase".

    Args:
        reset : If True, delete the existing collection and start fresh.
                Use this when re-ingesting a different repo.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            console.print(f"[yellow]🗑  Cleared existing ChromaDB collection[/yellow]")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name     = CHROMA_COLLECTION_NAME,
        # cosine distance is standard for semantic similarity
        metadata = {"hnsw:space": "cosine"},
    )
    return collection


def _existing_ids(collection: chromadb.Collection) -> set[str]:
    """Return the set of chunk IDs already stored in the collection."""
    result = collection.get(include=[])   # fetch only IDs, no vectors/docs
    return set(result["ids"])


def store_chunks(
    collection : chromadb.Collection,
    chunks     : list[CodeChunk],
) -> None:
    """
    Embed each chunk and store it in ChromaDB, skipping any already present.

    ChromaDB's add() takes parallel lists:
      - ids        : unique string ID per chunk
      - embeddings : the vector for each chunk
      - documents  : the raw text (stored for retrieval)
      - metadatas  : dict of metadata per chunk (file path, language, etc.)

    We batch in groups of 50 to avoid hitting API rate limits.
    """
    BATCH_SIZE = 50

    already_stored = _existing_ids(collection)
    pending = [c for c in chunks if c.chunk_id not in already_stored]

    if already_stored:
        console.print(
            f"[yellow]⏩ Resuming: {len(already_stored)} chunks already in ChromaDB, "
            f"{len(pending)} remaining.[/yellow]"
        )

    if not pending:
        console.print("[green]✅ All chunks already embedded — nothing to do.[/green]")
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(pending))

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]

            ids        = [c.chunk_id   for c in batch]
            documents  = [c.content    for c in batch]
            metadatas  = [
                {
                    "file_path"  : c.file_path,
                    "language"   : c.language,
                    "chunk_type" : c.chunk_type,
                    "start_line" : c.start_line,
                    "end_line"   : c.end_line,
                    "name"       : c.name,
                }
                for c in batch
            ]

            # Embed each text individually, pacing calls to stay under free-tier quota
            embeddings = []
            for doc in documents:
                embeddings.append(embed_text(doc))
                time.sleep(_INTER_REQUEST_DELAY)

            collection.add(
                ids        = ids,
                embeddings = embeddings,
                documents  = documents,
                metadatas  = metadatas,
            )

            progress.advance(task, len(batch))


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

def retrieve(
    collection : chromadb.Collection,
    query      : str,
    top_k      : int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Given a natural language query, find the most relevant code chunks.

    How it works:
      1. Embed the query into a vector
      2. ChromaDB computes cosine similarity between query vector and all stored vectors
      3. Returns the top_k closest matches

    Returns a list of dicts with keys: content, file_path, language, chunk_type, name, score
    """
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested response format
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "content"    : doc,
            "file_path"  : meta["file_path"],
            "language"   : meta["language"],
            "chunk_type" : meta["chunk_type"],
            "name"       : meta.get("name", ""),
            "score"      : round(1 - dist, 4),   # Convert distance → similarity (0-1)
        })

    return hits


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run(
    code_files  : list[CodeFile],
    reset       : bool = False,
) -> chromadb.Collection:
    """
    Main entry point for the embedding agent.

    Pipeline:
        1. Chunk all code files
        2. Get (or reset) ChromaDB collection
        3. Skip embedding if collection already has data (avoids re-embedding)
        4. Embed + store all chunks
        5. Return the collection (for the analysis agent to query)

    Args:
        code_files : Output from ingestion_agent.run()
        reset      : Wipe ChromaDB and re-embed everything from scratch

    Returns:
        ChromaDB collection ready for querying
    """
    console.print("\n" + "="*60)
    console.print("[bold cyan]EMBEDDING AGENT[/bold cyan]")
    console.print("="*60)

    # Step 1: Chunk
    console.print(f"\n[cyan]✂️  Chunking {len(code_files)} code files...[/cyan]")
    chunks = chunk_all_files(code_files, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    console.print(f"[green]✅ Created {len(chunks)} chunks[/green]")

    # Step 2: Connect to ChromaDB
    collection = get_chroma_collection(reset=reset)

    # Step 3: Skip only if the collection is fully populated (saves API quota).
    # A partial collection (e.g. after a mid-run crash) is handled inside
    # store_chunks, which resumes from where it left off.
    existing_count = collection.count()
    if existing_count >= len(chunks) and not reset:
        console.print(
            f"\n[yellow]⚡ ChromaDB already has {existing_count} vectors (fully embedded).[/yellow]"
            f"\n   Skipping embedding. Use reset=True to re-embed.\n"
        )
        return collection

    # Step 4: Embed and store
    console.print(f"\n[cyan]🔢 Embedding {len(chunks)} chunks → ChromaDB...[/cyan]")
    store_chunks(collection, chunks)

    console.print(f"\n[green]✅ Stored {collection.count()} vectors in ChromaDB[/green]")
    console.print(f"   Location: {CHROMA_DIR}")

    return collection


# ─────────────────────────────────────────────
# Quick test — run directly to verify
# python agents/embedding_agent.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from agents.ingestion_agent import run as ingest

    TEST_REPO = "https://github.com/fastapi/full-stack-fastapi-template"

    console.print("[bold]Testing Embedding Agent[/bold]")

    # Step 1: Ingest
    code_files = ingest(TEST_REPO)

    # Step 2: Embed
    collection = run(code_files)

    # Step 3: Test retrieval
    console.print("\n[bold]Testing retrieval...[/bold]")
    test_queries = [
        "authentication and login endpoints",
        "user registration and signup",
        "database models and schemas",
    ]

    for query in test_queries:
        console.print(f"\n[cyan]Query:[/cyan] {query}")
        results = retrieve(collection, query, top_k=3)
        for r in results:
            console.print(
                f"  [{r['score']:.3f}] {r['file_path']} "
                f"— {r['chunk_type']} {r['name']}"
            )
