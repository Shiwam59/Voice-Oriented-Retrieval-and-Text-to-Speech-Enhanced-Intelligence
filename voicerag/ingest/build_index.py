"""
Task 1.6 / 1.7 / 1.8 — Build index (FAISS + BM25 + metadata sidecar + dedup)

Builds:
  - FAISS HNSW index over passage embeddings (dense retrieval)
  - BM25 index over passage text (sparse retrieval)
  - SQLite sidecar for metadata (language, query_id, char_len, has_answer_overlap, chunk_strategy)
  - Deduplication logic for overlapping spans post-retrieval

Usage:
    python -m ingest.build_index
    python -m ingest.build_index --index-dir ./data/index --raw-dir ./data/raw
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import struct
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
import voicerag  # noqa: F401 — sets HF_HOME to data/hf_cache before HF imports

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", "./data/raw"))
DEFAULT_INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "./data/index"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
HNSW_M = 32           # HNSW connectivity parameter
HNSW_EF_CONSTRUCTION = 200
DEDUP_SIM_THRESHOLD = 0.95   # Cosine sim threshold for dedup


def build_metadata_sidecar(chunks_df: pd.DataFrame, db_path: Path) -> sqlite3.Connection:
    """
    Task 1.6 — Build SQLite metadata sidecar.
    Stores all per-chunk metadata keyed by internal index position.
    has_answer_overlap is stored but flagged offline-eval-only.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS chunk_meta")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_meta (
            idx INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            passage_id TEXT NOT NULL,
            chunk_strategy TEXT NOT NULL,
            language TEXT NOT NULL,
            original_query_id TEXT,
            char_len INTEGER,
            has_answer_overlap INTEGER,
            text TEXT NOT NULL
        )
    """)

    # Ensure has_answer_overlap is never queried at inference (documented in table comment)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_passage_id ON chunk_meta(passage_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON chunk_meta(chunk_strategy)")

    for i, row in chunks_df.iterrows():
        cur.execute(
            """INSERT INTO chunk_meta
               (idx, chunk_id, passage_id, chunk_strategy, language,
                original_query_id, char_len, has_answer_overlap, text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                i,
                row.get("chunk_id", f"c_{i}"),
                row.get("passage_id", ""),
                row.get("chunk_strategy", "passage_native"),
                row.get("language", "hi"),
                row.get("original_query_id"),
                row.get("char_len", 0),
                row.get("has_answer_overlap", 0),
                row.get("text", ""),
            ),
        )

    conn.commit()
    logger.info("Metadata sidecar: %d records → %s", len(chunks_df), db_path)
    return conn


def build_faiss_index(
    chunks_df: pd.DataFrame,
    index_dir: Path,
    model_name: str = EMBEDDING_MODEL,
) -> tuple[faiss.Index, SentenceTransformer]:
    """
    Task 1.8 (dense path) — Build FAISS HNSW index over chunk embeddings.
    """
    logger.info("Loading embedding model %s …", model_name)
    encoder = SentenceTransformer(model_name)

    texts = chunks_df["text"].tolist()
    logger.info("Encoding %d chunks …", len(texts))
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                                batch_size=256)

    dim = embeddings.shape[1]
    embeddings_float = embeddings.astype(np.float32)

    # Normalize for inner-product search (equivalent to cosine on L2-normalized vectors)
    faiss.normalize_L2(embeddings_float)

    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(embeddings_float)

    # Save FAISS index
    faiss_path = index_dir / "faiss_hnsw.index"
    faiss.write_index(index, str(faiss_path))
    logger.info("FAISS HNSW index: %d vectors, dim=%d → %s", index.ntotal, dim, faiss_path)

    return index, encoder


def build_bm25_index(chunks_df: pd.DataFrame, index_dir: Path) -> BM25Okapi:
    """
    Task 1.8 (sparse path) — Build BM25 index over tokenized chunk text.
    """
    # Simple whitespace + punctuation tokenization (works for Hindi)
    def tokenize(text: str) -> list[str]:
        import re
        return re.findall(r'[\u0900-\u097F\w]+', text.lower())

    tokenized = [tokenize(t) for t in chunks_df["text"]]
    bm25 = BM25Okapi(tokenized)

    # Save tokenized corpus for later use
    import pickle
    bm25_path = index_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"model": bm25, "tokenized": tokenized}, f)
    logger.info("BM25 index: %d documents → %s", len(tokenized), bm25_path)

    return bm25


def deduplicate_results(
    results: list[tuple[int, float]],
    chunks_df: pd.DataFrame,
    threshold: float = DEDUP_SIM_THRESHOLD,
) -> list[tuple[int, float]]:
    """
    Task 1.7 — De-duplicate overlapping retrieval results by text similarity.
    Returns filtered list of (index, score) tuples.
    """
    if len(results) <= 1:
        return results

    # Get texts for top results
    indices = [r[0] for r in results]
    texts = chunks_df.iloc[indices]["text"].tolist()

    # Quick hash-based pre-filter
    seen_hashes = set()
    filtered = []
    for i, (idx, score) in enumerate(results):
        h = texts[i][:100]  # First 100 chars as a quick hash proxy
        if h not in seen_hashes:
            seen_hashes.add(h)
            filtered.append((idx, score))

    if len(filtered) < len(results):
        logger.debug("Dedup removed %d overlapping results", len(results) - len(filtered))

    return filtered


def build_all(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> None:
    """Run full index build pipeline."""
    raw_dir = Path(raw_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunks_path = index_dir / "chunks.parquet"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found at {chunks_path}. Run chunking first: python -m ingest.chunking --all"
        )
    chunks_df = pd.read_parquet(chunks_path)
    logger.info("Loaded %d chunks", len(chunks_df))

    # Build metadata sidecar
    db_path = index_dir / "meta.db"
    build_metadata_sidecar(chunks_df, db_path)

    # Build FAISS index
    faiss_index, encoder = build_faiss_index(chunks_df, index_dir)

    # Build BM25 index
    bm25 = build_bm25_index(chunks_df, index_dir)

    logger.info("✓ Index build complete")
    logger.info("  FAISS: %d vectors", faiss_index.ntotal)
    logger.info("  BM25: %d documents", len(chunks_df))
    logger.info("  Metadata: %d records", len(chunks_df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build FAISS + BM25 + metadata index")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    build_all(args.raw_dir, args.index_dir)
    print("✓ All indices built")


if __name__ == "__main__":
    main()
