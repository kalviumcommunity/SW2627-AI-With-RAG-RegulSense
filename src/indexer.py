"""Indexer Entrypoint for RegulSense Banking Compliance Assistant.

Provides convenient alias and command-line execution for the corpus indexing pipeline.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_indexer import (
    CorpusIndexer,
    IndexingRunSummary,
    SpotCheckResult,
    run_corpus_indexing,
)

__all__ = [
    "CorpusIndexer",
    "IndexingRunSummary",
    "SpotCheckResult",
    "run_corpus_indexing",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RegulSense Corpus Indexer Entrypoint")
    parser.add_argument("--persist-dir", type=str, default=None, help="ChromaDB persistence directory")
    parser.add_argument("--collection-name", type=str, default=None, help="Target collection name")
    parser.add_argument("--in-memory", action="store_true", help="Run with ephemeral in-memory storage")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection before indexing")
    parser.add_argument("--spot-check-all", action="store_true", help="Audit all chunks in spot-check")
    args = parser.parse_args()

    p_dir = Path(args.persist_dir) if args.persist_dir else None
    run_corpus_indexing(
        persist_dir=p_dir,
        collection_name=args.collection_name,
        in_memory=args.in_memory,
        recreate=args.recreate,
        spot_check_all=args.spot_check_all,
    )
