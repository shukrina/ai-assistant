"""
Bulk-ingest all .txt / .md files from a directory into the vector store.

Usage:
    python scripts/ingest_documents.py --source ./knowledge_base
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag import ingest_file  # noqa: E402
from app.logging_config import configure_logging, get_logger  # noqa: E402

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk document ingestion for the RAG knowledge base.")
    parser.add_argument("--source", type=str, required=True, help="Directory containing .txt/.md files")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    files = list(source_dir.glob("*.txt")) + list(source_dir.glob("*.md"))
    if not files:
        logger.warning("No .txt or .md files found in %s", source_dir)
        return

    total_chunks = 0
    for f in files:
        filename, n_chunks = ingest_file(f)
        total_chunks += n_chunks
        logger.info("  -> %s: %d chunks", filename, n_chunks)

    logger.info("Done. Ingested %d files, %d total chunks.", len(files), total_chunks)


if __name__ == "__main__":
    main()
