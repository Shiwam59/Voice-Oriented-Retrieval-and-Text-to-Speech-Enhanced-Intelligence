"""
Build index from passage_native + India Knowledge Base.
Keeps it simple and accurate — India KB passages are not drowned out.

Usage:
    python data/build_india_index.py
"""

import sys
import os
from pathlib import Path

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "voicerag" / "data" / "hf_cache"))

# Set offline mode if model is cached
_model_cache = Path(os.environ["HF_HOME"]) / "hub" / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
if _model_cache.exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import faiss
import pickle
import sqlite3

from sentence_transformers import SentenceTransformer

from india_knowledge_base import INDIA_KB

# ── Config ──────────────────────────────────────────────────────
INDEX_DIR = Path(__file__).resolve().parent / "index"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200


def build():
    print("=" * 60)
    print("Building Index (passage_native + India KB)")
    print("=" * 60)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load passage_native chunks ───────────────────────
    native_path = INDEX_DIR / "chunks_passage_native.parquet"
    native_df = pd.read_parquet(native_path)
    print(f"Loaded {len(native_df)} passage_native chunks")

    all_chunks = []
    for _, row in native_df.iterrows():
        all_chunks.append({
            "chunk_id": row.get("chunk_id", ""),
            "passage_id": row.get("passage_id", ""),
            "text": row.get("text", ""),
            "chunk_strategy": "passage_native",
            "language": row.get("language", "hi"),
            "original_query_id": row.get("original_query_id"),
            "char_len": row.get("char_len", 0),
            "has_answer_overlap": 0,
        })

    # ── 2. Add India Knowledge Base ──────────────────────────
    for p in INDIA_KB:
        all_chunks.append({
            "chunk_id": f"{p['passage_id']}_kb",
            "passage_id": p["passage_id"],
            "text": p["text"],
            "chunk_strategy": "passage_native",
            "language": "hi",
            "original_query_id": None,
            "char_len": len(p["text"]),
            "has_answer_overlap": 0,
        })
    print(f"Added {len(INDIA_KB)} India Knowledge Base passages")

    # ── 3. Combine ───────────────────────────────────────────
    chunks_df = pd.DataFrame(all_chunks)
    print(f"\nTotal chunks: {len(chunks_df)}")
    chunks_df.to_parquet(INDEX_DIR / "chunks.parquet", index=False)

    # ── 4. Build FAISS index ─────────────────────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    texts = chunks_df["text"].tolist()
    print(f"Encoding {len(texts)} chunks...")
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    dim = embeddings.shape[1]
    embeddings_float = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings_float)

    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(embeddings_float)

    faiss_path = INDEX_DIR / "faiss_hnsw.index"
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    # ── 5. Build BM25 index ──────────────────────────────────
    import re
    def tokenize(text):
        return re.findall(r'[\u0900-\u097F\w]+', text.lower())

    tokenized = [tokenize(t) for t in texts]
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized)

    bm25_path = INDEX_DIR / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"model": bm25, "tokenized": tokenized}, f)
    print(f"BM25 index: {len(tokenized)} documents")

    # ── 6. Build metadata sidecar ────────────────────────────
    db_path = INDEX_DIR / "meta.db"
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_passage_id ON chunk_meta(passage_id)")

    for i, row in chunks_df.iterrows():
        cur.execute(
            "INSERT INTO chunk_meta VALUES (?,?,?,?,?,?,?,?,?)",
            (i, row.get("chunk_id", ""), row.get("passage_id", ""),
             row.get("chunk_strategy", "passage_native"), row.get("language", "hi"),
             row.get("original_query_id"), row.get("char_len", 0),
             row.get("has_answer_overlap", 0), row.get("text", "")),
        )
    conn.commit()
    conn.close()
    print(f"Metadata sidecar: {len(chunks_df)} records")

    print("\n" + "=" * 60)
    print(f"BUILD COMPLETE — {len(chunks_df)} total passages indexed")
    print("=" * 60)


if __name__ == "__main__":
    build()
