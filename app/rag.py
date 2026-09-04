"""
Retrieval-Augmented Generation pipeline.

Responsibilities:
  1. Document ingestion + chunking
  2. Embedding generation (sentence-transformers, runs locally / free)
  3. Storage + similarity search via ChromaDB (persistent, local vector DB)
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings
from app.logging_config import get_logger
from app.schemas import SourceChunk

settings = get_settings()
logger = get_logger(__name__)

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.EMBEDDING_MODEL
)

# Ensure the persistence directory exists (and is a plain relative/local path
# by default) before ChromaDB tries to create its SQLite file inside it.
Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

_client = chromadb.PersistentClient(
    path=settings.VECTOR_DB_PATH,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name="assistant_docs",
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Simple, dependency-free sliding-window chunker operating on characters.
    Good enough for prose; swap for a token-aware splitter (e.g. tiktoken) if
    exact token budgets matter for your target model.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    text = " ".join(text.split())  # normalize whitespace

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_document(filename: str, raw_text: str) -> int:
    """Chunk a document, embed each chunk, and store it in the vector DB."""
    chunks = chunk_text(raw_text)
    if not chunks:
        logger.warning("No content extracted from %s", filename)
        return 0

    ids = [f"{filename}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    _collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info("Ingested %s (%d chunks)", filename, len(chunks))
    return len(chunks)


def ingest_file(path: str | Path) -> tuple[str, int]:
    """Read a .txt/.md file from disk and ingest it."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    n = ingest_document(path.name, text)
    return path.name, n


def retrieve(query: str, top_k: int = None) -> List[SourceChunk]:
    """Return the top-k most relevant chunks for a query."""
    top_k = top_k or settings.TOP_K
    if _collection.count() == 0:
        return []

    results = _collection.query(query_texts=[query], n_results=min(top_k, _collection.count()))

    sources: List[SourceChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, cid, dist in zip(docs, metas, ids, dists):
        sources.append(
            SourceChunk(
                document=meta.get("source", "unknown"),
                chunk_id=cid,
                text=doc,
                score=round(1 - dist, 4),  # cosine distance -> similarity
            )
        )
    return sources


def build_context_block(sources: List[SourceChunk]) -> str:
    """Format retrieved chunks into a context block for the system prompt."""
    if not sources:
        return ""
    parts = [f"[{s.document}#{s.chunk_id}] {s.text}" for s in sources]
    return "Relevant context retrieved from the knowledge base:\n" + "\n---\n".join(parts)


def collection_stats() -> dict:
    return {"total_chunks": _collection.count()}
