"""
Task 1.2 — Flatten corpus

Reads the raw MSMARCO-XI parquet, extracts every passage from the nested
`passages` dict (Translated_passages + English_passages) into a flat
deduplicated passage store, and holds out query/answer pairs separately
for evaluation.

Dataset schema (actual):
  passages = {
      'English_passages': np.array of strings,
      'Translated_passages': np.array of strings,
      'is_selected': np.array of 0/1,
  }
  Answer = str (Hindi translated answer)
  query = str (Hindi translated query)
  query_id = int

Outputs:
    data/raw/passages.parquet      — flat, deduplicated passage store
    data/raw/eval_qa.parquet       — held-out query/answer pairs (for eval §10)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", "./data/raw"))


def _text_hash(text: str) -> str:
    """Deterministic hash for dedup."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def flatten_corpus(
    raw_path: str | Path = None,
    split: str = "hi",
    output_dir: str | Path = None,
) -> tuple[Path, Path]:
    """
    Flatten nested passages into a deduplicated store and extract eval QA pairs.
    Returns (passages_path, eval_qa_path).
    """
    raw_dir = Path(raw_path) if raw_path else DEFAULT_RAW_DIR
    out_dir = Path(output_dir) if output_dir else DEFAULT_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = raw_dir / f"msmarco_xi_{split}.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Raw parquet not found: {parquet_file}. Run download first.")

    df = pd.read_parquet(parquet_file)
    logger.info("Read %d rows from %s", len(df), parquet_file)

    # ── Extract eval QA pairs ─────────────────────────────────────
    eval_rows = []
    for idx, row in df.iterrows():
        eval_rows.append({
            "query_id": str(row.get("query_id", idx)),
            "query": row.get("query", ""),
            "answer": str(row.get("Answer", "") or ""),
            "eng_query": row.get("Eng_Query", ""),
            "eng_answer": row.get("Eng_Answer", ""),
            "source_lang": row.get("source_lang", ""),
            "target_lang": row.get("target_lang", ""),
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_path = out_dir / "eval_qa.parquet"
    eval_df.to_parquet(eval_path, index=False)
    logger.info("Saved %d eval QA pairs → %s", len(eval_df), eval_path)

    # ── Flatten passages ─────────────────────────────────────────
    # download.py already exploded the passages struct into list columns
    passage_records = []
    seen_hashes = set()

    for idx, row in df.iterrows():
        query_id = str(row.get("query_id", idx))

        def _as_list(val):
            """Parquet round-trips list columns as numpy arrays or None."""
            if val is None:
                return []
            if hasattr(val, "tolist"):
                return val.tolist()
            return list(val)

        translated = _as_list(row.get("Translated_passages"))
        english = _as_list(row.get("English_passages"))
        is_selected = _as_list(row.get("is_selected"))
        source_lang = row.get("source_lang", "")
        target_lang = row.get("target_lang", "hi")

        for i, trans_text in enumerate(translated):
            trans_text = str(trans_text).strip()
            if not trans_text:
                continue

            h = _text_hash(trans_text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            passage_records.append({
                "query_id": query_id,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": trans_text,
                "eng_text": str(english[i]) if i < len(english) else "",
                "is_selected": int(is_selected[i]) if i < len(is_selected) else 0,
            })

    passages_df = pd.DataFrame(passage_records)
    passages_df["passage_id"] = [f"p_{i:06d}" for i in range(len(passages_df))]
    passages_df["char_len"] = passages_df["text"].str.len()

    passages_path = out_dir / "passages.parquet"
    passages_df.to_parquet(passages_path, index=False)
    logger.info(
        "Flattened %d unique passages from %d rows → %s",
        len(passages_df), len(df), passages_path,
    )
    logger.info("  Selected (is_selected=1): %d", int(passages_df["is_selected"].sum()))
    logger.info("  Avg char_len: %.0f", passages_df["char_len"].mean())

    return passages_path, eval_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Flatten MSMARCO-XI passages")
    parser.add_argument("--raw-path", default=None, help="Path to raw parquet directory")
    parser.add_argument("--split", default=os.environ.get("HF_DATASET_SPLIT", "hi"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    p_path, q_path = flatten_corpus(args.raw_path, args.split, args.output_dir)
    print(f"✓ Passages: {p_path}  |  Eval QA: {q_path}")


if __name__ == "__main__":
    main()
