"""
Task 1.8 (retrieval) — Hybrid retrieval: dense FAISS + sparse BM25, fused.

Provides the core retrieval function that runs dense and sparse search in
parallel (conceptually; BM25 is in-process and fast), fuses results via
reciprocal rank fusion, and returns top-k passages with scores.

Usage:
    python -m harness.retriever --query "test query" --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Optional

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

from voicerag.schemas.contracts import ChunkStrategy, Passage, RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "./data/index"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Reciprocal Rank Fusion constant
RRF_K = 60
RETRIEVAL_CANDIDATES = int(os.environ.get("RETRIEVAL_CANDIDATES", "20"))
QUERY_EMBED_CACHE_SIZE = int(os.environ.get("QUERY_EMBED_CACHE_SIZE", "256"))


class HybridRetriever:
    """Hybrid retriever combining FAISS dense search and BM25 sparse search."""

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._load_indices()

    def _load_indices(self) -> None:
        """Load FAISS index, BM25 index, metadata sidecar, and embedding model."""
        # FAISS
        faiss_path = self.index_dir / "faiss_hnsw.index"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}. Run build_index first.")
        self.faiss_index = faiss.read_index(str(faiss_path))
        logger.info("Loaded FAISS index: %d vectors, dim=%d",
                     self.faiss_index.ntotal, self.faiss_index.d)

        # Embedding model (same one used at index time)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        # BM25
        bm25_path = self.index_dir / "bm25_index.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}. Run build_index first.")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["model"]

        # Metadata sidecar
        db_path = self.index_dir / "meta.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Metadata DB not found: {db_path}. Run build_index first.")
        # check_same_thread=False: FastAPI runs sync pipeline code in a
        # thread pool, so the SQLite connection (created in the main thread)
        # must be usable from other threads.  The retriever only issues
        # read-only SELECTs, so this is safe without external locking.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    @staticmethod
    def _row_to_passage(row) -> Passage:
        return Passage(
            passage_id=row["passage_id"],
            text=row["text"],
            query_id=row["original_query_id"],
            source_lang=row["language"],
            target_lang=row["language"],
            language=row["language"],
            char_len=row["char_len"],
            chunk_strategy=ChunkStrategy(row["chunk_strategy"]) if row["chunk_strategy"] else None,
        )

    def _idx_to_passage(self, idx: int) -> Passage:
        """Look up metadata for a given index position."""
        cur = self.conn.execute(
            "SELECT chunk_id, passage_id, chunk_strategy, language, original_query_id, char_len, text "
            "FROM chunk_meta WHERE idx = ?",
            (idx,),
        )
        row = cur.fetchone()
        return self._row_to_passage(row) if row is not None else Passage(passage_id=f"unknown_{idx}", text="[missing]")

    def _idx_to_passages(self, indices: list[int]) -> dict[int, Passage]:
        """Fetch all selected metadata rows in one SQLite query."""
        if not indices:
            return {}
        placeholders = ",".join("?" for _ in indices)
        rows = self.conn.execute(
            f"SELECT idx, passage_id, chunk_strategy, language, original_query_id, char_len, text "
            f"FROM chunk_meta WHERE idx IN ({placeholders})",
            tuple(indices),
        ).fetchall()
        result = {int(row["idx"]): self._row_to_passage(row) for row in rows}
        return {idx: result.get(idx, Passage(passage_id=f"unknown_{idx}", text="[missing]")) for idx in indices}

    def retrieve(self, query: str, top_k: int = 10, strategy: Optional[ChunkStrategy] = None) -> RetrievalResult:
        """
        Hybrid retrieval: dense FAISS + sparse BM25, fused via RRF.

        Args:
            query: User query text
            top_k: Number of results to return
            strategy: If specified, filter to a specific chunk strategy

        Returns:
            RetrievalResult with fused passages and per-path scores
        """
        # ── Dense search (FAISS) ──────────────────────────────────
        cache_key = query.strip().lower()
        q_emb = self._embedding_cache.get(cache_key)
        if q_emb is None:
            q_emb = self.encoder.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )
            q_emb = np.ascontiguousarray(q_emb.astype(np.float32))
            faiss.normalize_L2(q_emb)
            self._embedding_cache[cache_key] = q_emb
            self._embedding_cache.move_to_end(cache_key)
            if len(self._embedding_cache) > QUERY_EMBED_CACHE_SIZE:
                self._embedding_cache.popitem(last=False)
        else:
            self._embedding_cache.move_to_end(cache_key)

        # Keep the candidate set bounded; 20 is enough for fusion on this
        # corpus and avoids sorting/scoring unnecessary BM25 candidates.
        n_retrieve = min(max(top_k * 4, RETRIEVAL_CANDIDATES), self.faiss_index.ntotal)
        dense_scores, dense_indices = self.faiss_index.search(q_emb, n_retrieve)

        # ── Sparse search (BM25) ─────────────────────────────────
        import re
        tokens = re.findall(r'[\u0900-\u097F\w]+', query.lower())
        bm25_scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        if n_retrieve < len(bm25_scores):
            candidate_indices = np.argpartition(bm25_scores, -n_retrieve)[-n_retrieve:]
            sparse_top_indices = candidate_indices[np.argsort(bm25_scores[candidate_indices])[::-1]]
        else:
            sparse_top_indices = np.argsort(bm25_scores)[::-1][:n_retrieve]

        # ── Score-Weighted Reciprocal Rank Fusion ────────────────
        # Pure rank-based RRF treats rank-1 and rank-30 nearly the same,
        # letting noisy dense matches outrank strong BM25 keyword hits.
        # Fix: when BM25 finds an exact keyword match (high score), multiply
        # the RRF term by a large bonus so that passage jumps to the top.
        rrf_scores: dict[int, float] = {}
        dense_score_map: dict[int, float] = {}
        sparse_score_map: dict[int, float] = {}

        # Collect all raw BM25 scores for normalization
        all_sparse_raw = [float(bm25_scores[i]) for i in sparse_top_indices if bm25_scores[i] > 0]
        max_sparse_raw = max(all_sparse_raw) if all_sparse_raw else 1.0
        max_sparse_raw = max(max_sparse_raw, 1e-6)

        # Phase 1: collect RRF scores from both sources
        dense_rrf: dict[int, float] = {}
        sparse_rrf: dict[int, float] = {}

        for rank, idx in enumerate(dense_indices[0]):
            if idx < 0:
                continue
            idx = int(idx)
            raw = float(dense_scores[0][rank])
            dense_score_map[idx] = raw
            dense_rrf[idx] = 1.0 / (RRF_K + rank + 1)

        for rank, idx in enumerate(sparse_top_indices):
            if idx < 0:
                continue
            idx = int(idx)
            raw = float(bm25_scores[idx])
            sparse_score_map[idx] = raw
            sparse_rrf[idx] = 1.0 / (RRF_K + rank + 1)

        # Phase 2: hybrid fusion with BM25 keyword boost
        all_indices = set(dense_rrf.keys()) | set(sparse_rrf.keys())
        for idx in all_indices:
            d = dense_rrf.get(idx, 0.0)
            s = sparse_rrf.get(idx, 0.0)
            if d > 0 and s > 0:
                # Both sources agree — strongest signal
                rrf_scores[idx] = (d + s) * 1.2
            elif d > 0:
                # Dense only — semantic match
                rrf_scores[idx] = d * 0.8
            else:
                # Sparse only — keyword match
                # Strong BM25 (score > 8) is a reliable exact match
                raw = sparse_score_map.get(idx, 0.0)
                if raw > 8.0:
                    rrf_scores[idx] = s * 1.5  # Strong keyword match
                elif raw > 5.0:
                    rrf_scores[idx] = s * 1.0  # Medium match
                else:
                    rrf_scores[idx] = s * 0.3  # Weak match

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # Filter by strategy if requested
        if strategy:
            filtered = []
            for idx in sorted_indices:
                passage = self._idx_to_passage(idx)
                if passage.chunk_strategy == strategy:
                    filtered.append(idx)
                if len(filtered) >= top_k:
                    break
            sorted_indices = filtered

        # Build result with one metadata query instead of one SQLite round-trip
        selected_indices = sorted_indices[:top_k]
        metadata = self._idx_to_passages(selected_indices)
        passages = []
        fused_score_list = []
        for idx in selected_indices:
            passage = metadata[idx]
            passage.score = rrf_scores[idx]
            passages.append(passage)
            fused_score_list.append(rrf_scores[idx])

        dense_only = [dense_score_map.get(idx, 0.0) for idx in sorted_indices[:top_k]]
        sparse_only = [sparse_score_map.get(idx, 0.0) for idx in sorted_indices[:top_k]]

        return RetrievalResult(
            passages=passages,
            dense_scores=dense_only,
            sparse_scores=sparse_only,
            fused_scores=fused_score_list,
        )

    def close(self) -> None:
        self.conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Hybrid retrieval smoke test")
    parser.add_argument("--query", required=True, help="Query to test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    retriever = HybridRetriever(args.index_dir)
    result = retriever.retrieve(args.query, args.top_k)

    print(f"\nTop-{args.top_k} results for: '{args.query}'\n")
    for i, p in enumerate(result.passages):
        print(f"  {i+1}. [{p.passage_id}] (score={p.score:.4f}, strategy={p.chunk_strategy})")
        print(f"     {p.text[:120]}…\n")

    print(f"Dense scores:  {[f'{s:.4f}' for s in result.dense_scores]}")
    print(f"Sparse scores: {[f'{s:.4f}' for s in result.sparse_scores]}")
    print(f"Fused scores:  {[f'{s:.4f}' for s in result.fused_scores]}")

    retriever.close()


# ── Multi-Index Retriever ──────────────────────────────────────────

class MultiIndexRetriever:
    """Routes queries to the correct language index.

    Supports:
    - Hindi: ./data/index (default)
    - Urdu: ./data/index_urdu (separate index)
    """

    def __init__(self, base_dir: str | Path = None):
        if base_dir is None:
            base_dir = Path(os.environ.get("VECTOR_DB_PATH", "./data/index")).parent
        self.base_dir = Path(base_dir)
        self._retrievers: dict[str, HybridRetriever] = {}
        self._load_retrievers()

    def _load_retrievers(self):
        """Load all available language indices."""
        # Hindi index (default)
        hindi_dir = self.base_dir / "index"
        if hindi_dir.exists() and (hindi_dir / "faiss_hnsw.index").exists():
            self._retrievers["hi"] = HybridRetriever(hindi_dir)
            logger.info("Loaded Hindi index: %d vectors", self._retrievers["hi"].faiss_index.ntotal)

        # Urdu index
        urdu_dir = self.base_dir / "index_urdu"
        if urdu_dir.exists() and (urdu_dir / "faiss_hnsw.index").exists():
            self._retrievers["ur"] = HybridRetriever(urdu_dir)
            logger.info("Loaded Urdu index: %d vectors", self._retrievers["ur"].faiss_index.ntotal)

    def retrieve(self, query: str, top_k: int = 5, language: str = "hi") -> RetrievalResult:
        """Retrieve from the appropriate language index."""
        # Map language to retriever
        lang_key = language if language in self._retrievers else "hi"
        if lang_key not in self._retrievers:
            logger.warning("No index for language '%s', falling back to Hindi", language)
            lang_key = "hi"

        retriever = self._retrievers[lang_key]
        return retriever.retrieve(query, top_k)

    def close(self):
        for r in self._retrievers.values():
            r.close()


if __name__ == "__main__":
    main()
