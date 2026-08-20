"""
Latency Benchmark — P50 / P70 / P100 measurement.

Measures the full pipeline latency across multiple queries:
  1. Embedding the query (SentenceTransformer encode)
  2. FAISS dense search
  3. BM25 sparse search
  4. RRF fusion
  5. Extractive answer (no LLM — fast path)

Reports P50, P70, P90, P100 latencies in milliseconds.

Usage:
    python -m voicerag.benchmark_latency
    python benchmark_latency.py
"""

import os
import sys
import time
import statistics
from pathlib import Path

# Setup
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.environ["HF_HOME"] = str(_ROOT / "voicerag" / "data" / "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import faiss
import pickle
import sqlite3
from sentence_transformers import SentenceTransformer


# ── Test Queries ─────────────────────────────────────────────────
TEST_QUERIES = [
    # Hindi queries
    "गोवा कहाँ है?",
    "गंगा नदी कहाँ से बहती है?",
    "दिल्ली की राजधानी क्या है?",
    "ताजमहल कहाँ है?",
    "महात्मा गांधी कौन थे?",
    "होली कब मनाई जाती है?",
    "क्रिकेट विश्व कप",
    "चंद्रयान-3 क्या है?",
    "ISRO क्या है?",
    "उत्तर प्रदेश की राजधानी क्या है?",
    "भारत का राष्ट्रीय पशु क्या है?",
    "बनारस कहाँ है?",
    "केरल किसके लिए प्रसिद्ध है?",
    "वाराणसी में क्या प्रसिद्ध है?",
    "राजस्थान का सबसे बड़ा शहर कौन सा है?",
    # English queries (will be translated)
    "Where is Goa?",
    "What is the capital of India?",
    "Who is Mahatma Gandhi?",
    "What is ISRO?",
    "Tell me about the Ganga river",
    "What is Diwali?",
    "Where is the Taj Mahal?",
    "What is Chandrayaan-3?",
]


class FastRetriever:
    """Optimized retriever with in-memory indices."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self._load_indices()

    def _load_indices(self):
        """Load all indices into memory once."""
        # FAISS
        faiss_path = self.index_dir / "faiss_hnsw.index"
        self.faiss_index = faiss.read_index(str(faiss_path))

        # Embedding model
        self.encoder = SentenceTransformer(
            os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        )

        # BM25
        bm25_path = self.index_dir / "bm25_index.pkl"
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["model"]

        # Metadata
        db_path = self.index_dir / "meta.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Preload all metadata into memory for O(1) lookup
        self._meta_cache = {}
        cur = self.conn.execute(
            "SELECT idx, chunk_id, passage_id, text FROM chunk_meta"
        )
        for row in cur:
            self._meta_cache[row["idx"]] = {
                "chunk_id": row["chunk_id"],
                "passage_id": row["passage_id"],
                "text": row["text"],
            }

    def retrieve(self, query: str, top_k: int = 5):
        """Fast retrieval with pre-loaded indices."""
        import re

        # Dense search
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        q_emb = np.ascontiguousarray(q_emb.astype(np.float32))
        faiss.normalize_L2(q_emb)

        n_retrieve = min(top_k * 3, self.faiss_index.ntotal)
        dense_scores, dense_indices = self.faiss_index.search(q_emb, n_retrieve)

        # BM25 search
        tokens = re.findall(r'[\u0900-\u097F\w]+', query.lower())
        bm25_scores = self.bm25.get_scores(tokens)
        sparse_top = np.argsort(bm25_scores)[::-1][:n_retrieve]

        # RRF fusion with BM25 boost
        rrf = {}
        dense_map = {}
        sparse_map = {}

        all_sparse = [float(bm25_scores[i]) for i in sparse_top if bm25_scores[i] > 0]
        max_sparse = max(all_sparse) if all_sparse else 1.0
        max_sparse = max(max_sparse, 1e-6)

        K = 60
        for rank, idx in enumerate(dense_indices[0]):
            if idx < 0:
                continue
            idx = int(idx)
            raw = float(dense_scores[0][rank])
            rrf_term = 1.0 / (K + rank + 1) * (1.0 + raw)
            rrf[idx] = rrf.get(idx, 0) + rrf_term
            dense_map[idx] = raw

        for rank, idx in enumerate(sparse_top):
            if idx < 0:
                continue
            idx = int(idx)
            raw = float(bm25_scores[idx])
            norm = raw / max_sparse
            bm25_boost = 1.0 + norm * 5.0
            rrf_term = 1.0 / (K + rank + 1) * bm25_boost
            rrf[idx] = rrf.get(idx, 0) + rrf_term
            sparse_map[idx] = raw

        # Sort and return top-k
        sorted_idx = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)

        results = []
        for idx in sorted_idx[:top_k]:
            meta = self._meta_cache.get(idx, {})
            # Find the dense score for this idx
            d_score = dense_map.get(idx, 0.0)
            s_score = sparse_map.get(idx, 0.0)
            results.append({
                "passage_id": meta.get("passage_id", f"unknown_{idx}"),
                "text": meta.get("text", ""),
                "dense_score": d_score,
                "sparse_score": s_score,
                "fused_score": rrf[idx],
            })

        return results

    def close(self):
        self.conn.close()


def run_benchmark():
    """Run latency benchmark and report P50/P70/P90/P100."""
    print("=" * 70)
    print("VoiceRAG Latency Benchmark")
    print("=" * 70)

    index_dir = str(_ROOT / "voicerag" / "data" / "index")
    print(f"Index: {index_dir}")

    # Load retriever
    print("\nLoading indices into memory...")
    t0 = time.perf_counter()
    retriever = FastRetriever(index_dir)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Index load time: {load_time:.1f}ms")
    print(f"FAISS vectors: {retriever.faiss_index.ntotal}")
    print(f"BM25 documents: {len(retriever.bm25.doc_len)}")
    print(f"Metadata entries: {len(retriever._meta_cache)}")

    # Warmup
    print("\nWarmup (3 queries)...")
    for q in TEST_QUERIES[:3]:
        retriever.retrieve(q, top_k=5)

    # Benchmark
    print(f"\nBenchmarking {len(TEST_QUERIES)} queries (10 runs each)...")
    latencies = []
    query_latencies = []

    for q in TEST_QUERIES:
        query_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            results = retriever.retrieve(q, top_k=5)
            elapsed = (time.perf_counter() - t0) * 1000
            query_times.append(elapsed)

        avg = statistics.mean(query_times)
        latencies.extend(query_times)
        query_latencies.append((q, avg, min(query_times), max(query_times)))

    # Report per-query stats
    print("\n" + "-" * 70)
    print(f"{'Query':<40} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for q, avg, mn, mx in query_latencies:
        q_short = q[:38] + ".." if len(q) > 40 else q
        print(f"{q_short:<40} {avg:>7.1f}ms {mn:>7.1f}ms {mx:>7.1f}ms")

    # Overall percentiles
    latencies.sort()
    n = len(latencies)

    p50 = latencies[int(n * 0.50)]
    p70 = latencies[int(n * 0.70)]
    p90 = latencies[int(n * 0.90)]
    p95 = latencies[int(n * 0.95)]
    p100 = latencies[-1]
    avg = statistics.mean(latencies)

    print("\n" + "=" * 70)
    print("LATENCY RESULTS (retrieval only, no STT/LLM)")
    print("=" * 70)
    print(f"  Total measurements: {n}")
    print(f"  Mean:    {avg:.1f}ms")
    print(f"  P50:     {p50:.1f}ms")
    print(f"  P70:     {p70:.1f}ms")
    print(f"  P90:     {p90:.1f}ms")
    print(f"  P95:     {p95:.1f}ms")
    print(f"  P100:    {p100:.1f}ms")
    print(f"  Target:  <200ms")
    print(f"  {'PASS' if p90 < 200 else 'FAIL'}: P90 {'<' if p90 < 200 else '>'} 200ms")
    print("=" * 70)

    retriever.close()
    return {"p50": p50, "p70": p70, "p90": p90, "p95": p95, "p100": p100, "avg": avg}


if __name__ == "__main__":
    run_benchmark()
