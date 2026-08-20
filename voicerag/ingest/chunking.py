"""
Task 1.3 / 1.4 / 1.5 — Chunking strategies

Implements three chunking strategies over the flattened passage store:
  - Passage-native (1.3): each passage = one chunk
  - Sentence-window (1.4): sliding window over longer passages
  - Semantic (1.5): merge adjacent sentences by cosine similarity threshold

All strategies produce records tagged with chunk_strategy, passage_id,
language, original_query_id, char_len, and has_answer_overlap (offline-only).

Usage:
    python -m ingest.chunking --all
    python -m ingest.chunking --strategy sentence_window
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
import voicerag  # noqa: F401 — sets HF_HOME to data/hf_cache before HF imports

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", "./data/raw"))
DEFAULT_INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "./data/index"))

# Decision: passages above 150 chars trigger sentence-window chunking
SENTENCE_WINDOW_THRESHOLD = 150
SENTENCE_WINDOW_SIZE = 3      # 2-3 sentences per chunk
SENTENCE_WINDOW_OVERLAP = 1    # 1-sentence overlap

# Decision: semantic chunking cosine-sim threshold
SEMANTIC_SIM_THRESHOLD = 0.7

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for Hindi/Indic text. Splits on । and ."""
    import re
    # Split on Hindi danda (।), period, question mark, exclamation
    parts = re.split(r'[।.!?]+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _answer_map(eval_df: Optional[pd.DataFrame]) -> dict[str, str]:
    """qid -> gold answer dict for O(1) has_answer_overlap lookups."""
    if eval_df is None or eval_df.empty:
        return {}
    return dict(zip(eval_df["query_id"].astype(str), eval_df["answer"].astype(str)))


def _has_overlap(answer_map: dict[str, str], query_id, text: str) -> int:
    answer = answer_map.get(str(query_id), "")
    return 1 if answer and answer in text else 0


def passage_native_chunking(
    passages_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Task 1.3 — Each passage = one chunk."""
    logger.info("Passage-native chunking: %d passages", len(passages_df))
    answer_map = _answer_map(eval_df)

    chunks = []
    for _, row in passages_df.iterrows():
        chunks.append({
            "chunk_id": f"{row['passage_id']}_native",
            "passage_id": row["passage_id"],
            "text": row["text"],
            "chunk_strategy": "passage_native",
            "language": row.get("target_lang", "hi"),
            "original_query_id": row.get("query_id"),
            "char_len": row.get("char_len", len(row["text"])),
            "has_answer_overlap": _has_overlap(answer_map, row.get("query_id"), row["text"]),
        })

    result = pd.DataFrame(chunks)
    logger.info("Produced %d passage-native chunks", len(result))
    return result


def sentence_window_chunking(
    passages_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    threshold: int = SENTENCE_WINDOW_THRESHOLD,
    window_size: int = SENTENCE_WINDOW_SIZE,
    overlap: int = SENTENCE_WINDOW_OVERLAP,
) -> pd.DataFrame:
    """Task 1.4 — Sentence-window chunking for longer passages."""
    long_passages = passages_df[passages_df["char_len"] > threshold]
    logger.info(
        "Sentence-window chunking: %d passages above %d-char threshold (of %d total)",
        len(long_passages), threshold, len(passages_df),
    )
    answer_map = _answer_map(eval_df)

    chunks = []
    for _, row in long_passages.iterrows():
        sentences = _split_sentences(row["text"])
        if len(sentences) <= 1:
            continue  # Skip single-sentence passages (already covered by native)

        overlap_flag = _has_overlap(answer_map, row.get("query_id"), row.get("text", ""))

        step = max(1, window_size - overlap)
        for start in range(0, len(sentences), step):
            end = min(start + window_size, len(sentences))
            window_text = " ".join(sentences[start:end])
            chunk_id = f"{row['passage_id']}_sw_{start}_{end}"
            chunks.append({
                "chunk_id": chunk_id,
                "passage_id": row["passage_id"],
                "text": window_text,
                "chunk_strategy": "sentence_window",
                "language": row.get("target_lang", "hi"),
                "original_query_id": row.get("query_id"),
                "char_len": len(window_text),
                "has_answer_overlap": overlap_flag,
            })

    result = pd.DataFrame(chunks) if chunks else pd.DataFrame(
        columns=["chunk_id", "passage_id", "text", "chunk_strategy",
                 "language", "original_query_id", "char_len", "has_answer_overlap"]
    )
    logger.info("Produced %d sentence-window chunks", len(result))
    return result


def semantic_chunking(
    passages_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    sim_threshold: float = SEMANTIC_SIM_THRESHOLD,
    model_name: str = EMBEDDING_MODEL,
) -> pd.DataFrame:
    """Task 1.5 — Semantic chunking: merge adjacent sentences while cosine sim > threshold."""
    long_passages = passages_df[passages_df["char_len"] > SENTENCE_WINDOW_THRESHOLD]
    if len(long_passages) == 0:
        logger.warning("No passages long enough for semantic chunking")
        return pd.DataFrame(
            columns=["chunk_id", "passage_id", "text", "chunk_strategy",
                     "language", "original_query_id", "char_len", "has_answer_overlap"]
        )

    logger.info("Loading embedding model %s for semantic chunking …", model_name)
    encoder = SentenceTransformer(model_name)
    answer_map = _answer_map(eval_df)

    chunks = []
    passage_texts = long_passages["text"].tolist()
    passage_ids = long_passages["passage_id"].tolist()
    query_ids = long_passages.get("query_id", pd.Series(dtype=str)).tolist()
    target_langs = long_passages.get("target_lang", pd.Series(dtype=str)).tolist()

    # Encode all sentences from all long passages
    all_sentences = []
    sentence_map = []  # (passage_idx, sentence_start, sentence_end)
    for p_idx, text in enumerate(passage_texts):
        sents = _split_sentences(text)
        if len(sents) <= 1:
            continue
        start = len(all_sentences)
        all_sentences.extend(sents)
        sentence_map.append((p_idx, start, start + len(sents)))

    if not all_sentences:
        logger.warning("No sentences found for semantic chunking")
        return pd.DataFrame(
            columns=["chunk_id", "passage_id", "text", "chunk_strategy",
                     "language", "original_query_id", "char_len", "has_answer_overlap"]
        )

    logger.info("Encoding %d sentences …", len(all_sentences))
    embeddings = encoder.encode(all_sentences, normalize_embeddings=True, show_progress_bar=False)

    # Merge sentences within each passage by cosine similarity
    overlap_flags = {}
    for p_idx, s_start, s_end in sentence_map:
        overlap_flags[p_idx] = _has_overlap(
            answer_map, query_ids[p_idx] if p_idx < len(query_ids) else None, passage_texts[p_idx]
        )

    for p_idx, s_start, s_end in sentence_map:
        p_embeddings = embeddings[s_start:s_end]
        p_sents = all_sentences[s_start:s_end]

        current_chunk_sents = [p_sents[0]]
        chunk_start = 0

        for i in range(1, len(p_sents)):
            prev_emb = p_embeddings[i - 1]
            curr_emb = p_embeddings[i]
            sim = float(np.dot(prev_emb, curr_emb))

            if sim >= sim_threshold:
                current_chunk_sents.append(p_sents[i])
            else:
                # Emit current chunk
                chunk_text = " ".join(current_chunk_sents)
                chunks.append({
                    "chunk_id": f"{passage_ids[p_idx]}_sem_{chunk_start}_{i}",
                    "passage_id": passage_ids[p_idx],
                    "text": chunk_text,
                    "chunk_strategy": "semantic",
                    "language": target_langs[p_idx] if p_idx < len(target_langs) else "hi",
                    "original_query_id": query_ids[p_idx] if p_idx < len(query_ids) else None,
                    "char_len": len(chunk_text),
                    "has_answer_overlap": overlap_flags.get(p_idx, 0),
                })
                current_chunk_sents = [p_sents[i]]
                chunk_start = i

        # Emit last chunk
        if current_chunk_sents:
            chunk_text = " ".join(current_chunk_sents)
            chunks.append({
                "chunk_id": f"{passage_ids[p_idx]}_sem_{chunk_start}_{s_end}",
                "passage_id": passage_ids[p_idx],
                "text": chunk_text,
                "chunk_strategy": "semantic",
                "language": target_langs[p_idx] if p_idx < len(target_langs) else "hi",
                "original_query_id": query_ids[p_idx] if p_idx < len(query_ids) else None,
                "char_len": len(chunk_text),
                "has_answer_overlap": overlap_flags.get(p_idx, 0),
            })

    result = pd.DataFrame(chunks) if chunks else pd.DataFrame(
        columns=["chunk_id", "passage_id", "text", "chunk_strategy",
                 "language", "original_query_id", "char_len", "has_answer_overlap"]
    )

    logger.info("Produced %d semantic chunks (threshold=%.2f)", len(result), sim_threshold)
    return result


def fixed_size_chunking(
    passages_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    chunk_size: int = 200,
    overlap: int = 50,
) -> pd.DataFrame:
    """Fixed-size character chunking with overlap.
    Splits passages into overlapping chunks of ~chunk_size characters."""
    answer_map = _answer_map(eval_df)
    chunks = []

    for _, row in passages_df.iterrows():
        text = row["text"]
        if len(text) <= chunk_size:
            # Short passage — keep as single chunk
            chunks.append({
                "chunk_id": f"{row['passage_id']}_fs_0",
                "passage_id": row["passage_id"],
                "text": text,
                "chunk_strategy": "fixed_size",
                "language": row.get("target_lang", "hi"),
                "original_query_id": row.get("query_id"),
                "char_len": len(text),
                "has_answer_overlap": _has_overlap(answer_map, row.get("query_id"), text),
            })
        else:
            # Split into overlapping chunks
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append({
                    "chunk_id": f"{row['passage_id']}_fs_{chunk_idx}",
                    "passage_id": row["passage_id"],
                    "text": chunk_text,
                    "chunk_strategy": "fixed_size",
                    "language": row.get("target_lang", "hi"),
                    "original_query_id": row.get("query_id"),
                    "char_len": len(chunk_text),
                    "has_answer_overlap": _has_overlap(answer_map, row.get("query_id"), chunk_text),
                })
                start += chunk_size - overlap
                chunk_idx += 1

    result = pd.DataFrame(chunks)
    logger.info("Fixed-size chunking: %d chunks (size=%d, overlap=%d)", len(result), chunk_size, overlap)
    return result


def run_all_strategies(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> pd.DataFrame:
    """Run all four chunking strategies and merge into a single chunk store.

    Each strategy's output is saved to its own parquet as it completes, so a
    killed build resumes without redoing finished strategies (skips any whose
    file already exists unless force=True).
    """
    raw_dir = Path(raw_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    passages_path = raw_dir / "passages.parquet"
    eval_path = raw_dir / "eval_qa.parquet"

    passages_df = pd.read_parquet(passages_path)
    eval_df = pd.read_parquet(eval_path) if eval_path.exists() else None

    logger.info("Running all chunking strategies over %d passages …", len(passages_df))

    strategy_files = {
        "passage_native": index_dir / "chunks_passage_native.parquet",
        "sentence_window": index_dir / "chunks_sentence_window.parquet",
        "semantic": index_dir / "chunks_semantic.parquet",
        "fixed_size": index_dir / "chunks_fixed_size.parquet",
    }

    if not strategy_files["passage_native"].exists():
        native = passage_native_chunking(passages_df, eval_df)
        native.to_parquet(strategy_files["passage_native"], index=False)
        logger.info("Saved %d passage-native chunks → %s", len(native), strategy_files["passage_native"])
    else:
        logger.info("passage_native chunks already exist — skipping (%s)", strategy_files["passage_native"])

    if not strategy_files["sentence_window"].exists():
        windowed = sentence_window_chunking(passages_df, eval_df)
        windowed.to_parquet(strategy_files["sentence_window"], index=False)
        logger.info("Saved %d sentence-window chunks → %s", len(windowed), strategy_files["sentence_window"])
    else:
        logger.info("sentence_window chunks already exist — skipping")

    if not strategy_files["semantic"].exists():
        semantic = semantic_chunking(passages_df, eval_df)
        semantic.to_parquet(strategy_files["semantic"], index=False)
        logger.info("Saved %d semantic chunks → %s", len(semantic), strategy_files["semantic"])
    else:
        logger.info("semantic chunks already exist — skipping")

    if not strategy_files["fixed_size"].exists():
        fixed = fixed_size_chunking(passages_df, eval_df)
        fixed.to_parquet(strategy_files["fixed_size"], index=False)
        logger.info("Saved %d fixed-size chunks → %s", len(fixed), strategy_files["fixed_size"])
    else:
        logger.info("fixed_size chunks already exist — skipping")

    # Merge all strategies
    all_chunks = pd.concat(
        [pd.read_parquet(f) for f in strategy_files.values()], ignore_index=True
    )

    chunks_path = index_dir / "chunks.parquet"
    all_chunks.to_parquet(chunks_path, index=False)
    logger.info("Saved %d total chunks → %s", len(all_chunks), chunks_path)
    for name, f in strategy_files.items():
        logger.info("  %s: %d", name, len(pd.read_parquet(f)))

    return all_chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run chunking strategies")
    parser.add_argument("--strategy", choices=["passage_native", "sentence_window", "semantic", "all"],
                        default="all")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    if args.strategy == "all":
        run_all_strategies(args.raw_dir, args.index_dir)
        print("✓ All chunking strategies complete")
    else:
        passages_df = pd.read_parquet(Path(args.raw_dir) / "passages.parquet")
        eval_path = Path(args.raw_dir) / "eval_qa.parquet"
        eval_df = pd.read_parquet(eval_path) if eval_path.exists() else None

        if args.strategy == "passage_native":
            df = passage_native_chunking(passages_df, eval_df)
        elif args.strategy == "sentence_window":
            df = sentence_window_chunking(passages_df, eval_df)
        else:
            df = semantic_chunking(passages_df, eval_df)

        out = Path(args.index_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"chunks_{args.strategy}.parquet"
        df.to_parquet(path, index=False)
        print(f"✓ {args.strategy}: {len(df)} chunks → {path}")


if __name__ == "__main__":
    main()
