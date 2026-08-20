"""
Full ingestion pipeline: download → flatten → chunk → index.

This is the single entry point to go from zero to a queryable index.

Usage:
    python -m ingest.run_pipeline --split hi
    python -m ingest.run_pipeline --split hi --raw-dir ./data/raw --index-dir ./data/index
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", "./data/raw"))
DEFAULT_INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "./data/index"))


def run_full_pipeline(
    split: str = "hi",
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> None:
    """Run the complete ingestion pipeline."""
    # Step 1: Download
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading MSMARCO-XI (%s) …", split)
    from voicerag.ingest.download import ingest_split
    path, rows = ingest_split(split, raw_dir)
    logger.info("✓ Downloaded %d rows → %s", rows, path)

    # Step 2: Flatten
    logger.info("=" * 60)
    logger.info("STEP 2: Flattening corpus …")
    from voicerag.ingest.flatten import flatten_corpus
    p_path, q_path = flatten_corpus(raw_dir, split, raw_dir)
    logger.info("✓ Passages: %s  |  Eval QA: %s", p_path, q_path)

    # Step 3: Chunk
    logger.info("=" * 60)
    logger.info("STEP 3: Running chunking strategies …")
    from voicerag.ingest.chunking import run_all_strategies
    chunks_df = run_all_strategies(raw_dir, index_dir)
    logger.info("✓ %d chunks created", len(chunks_df))

    # Step 4: Build indices
    logger.info("=" * 60)
    logger.info("STEP 4: Building FAISS + BM25 + metadata indices …")
    from voicerag.ingest.build_index import build_all
    build_all(raw_dir, index_dir)
    logger.info("✓ All indices built")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — ready for queries")
    logger.info("  Index dir: %s", index_dir)
    logger.info("  Raw data:  %s", raw_dir)
    logger.info("  Test with: python -m harness.retriever --query 'your test query'")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Full ingestion pipeline: download → flatten → chunk → index")
    parser.add_argument("--split", default=os.environ.get("HF_DATASET_SPLIT", "hi"),
                        help="Language split to download")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    run_full_pipeline(args.split, args.raw_dir, args.index_dir)


if __name__ == "__main__":
    main()
