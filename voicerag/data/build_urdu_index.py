"""
Build FAISS index for Urdu MSMARCO-XI passages.

Usage:
    python data/build_urdu_index.py
"""

import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "voicerag" / "data" / "hf_cache"))

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

INDEX_DIR = Path(__file__).resolve().parent / "index_urdu"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200


def build():
    print("=" * 60)
    print("Building Urdu Index")
    print("=" * 60)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load Urdu passages
    passages_path = Path(__file__).resolve().parent / "raw" / "passages_urdu.parquet"
    if not passages_path.exists():
        print(f"ERROR: {passages_path} not found. Run download_urdu.py first.")
        return

    passages_df = pd.read_parquet(passages_path)
    print(f"Loaded {len(passages_df)} Urdu passages")

    # Build chunks
    chunks = []
    for _, row in passages_df.iterrows():
        chunks.append({
            "chunk_id": row["passage_id"],
            "passage_id": row["passage_id"],
            "text": row["text"],
            "chunk_strategy": "passage_native",
            "language": "ur",
            "original_query_id": row.get("query_id"),
            "char_len": len(row["text"]),
            "has_answer_overlap": 0,
        })

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_parquet(INDEX_DIR / "chunks.parquet", index=False)
    print(f"Chunks: {len(chunks_df)}")

    # Build FAISS index
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

    # Build BM25 index
    import re
    def tokenize(text):
        return re.findall(r'[\u0600-\u06FF\u0900-\u097F\w]+', text.lower())

    tokenized = [tokenize(t) for t in texts]
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized)

    bm25_path = INDEX_DIR / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"model": bm25, "tokenized": tokenized}, f)
    print(f"BM25 index: {len(tokenized)} documents")

    # Build metadata sidecar
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
            (i, row["chunk_id"], row["passage_id"], "passage_native", "ur",
             row.get("original_query_id"), row["char_len"], 0, row["text"]),
        )
    conn.commit()
    conn.close()
    print(f"Metadata sidecar: {len(chunks_df)} records")

    print("\n" + "=" * 60)
    print(f"BUILD COMPLETE — {len(chunks_df)} Urdu passages indexed")
    print("=" * 60)


if __name__ == "__main__":
    build()
