"""
Task 1.1 — Ingest MSMARCO-XI (hi)

Downloads the Hindi split of ai4bharat/MSMARCO-XI from HuggingFace,
caches the raw parquet locally, and logs the row count + schema.

The dataset stores per-language parquet files directly in the repo:
  train/hintrain.parquet (3.5GB), validation/hinval.parquet (441MB)

The `passages` column is a nested struct (dict of arrays) that pandas
cannot convert directly ("Nested data conversions not implemented"), so
we read via pyarrow row-groups and explode the struct into plain list
columns.

Ingestion is capped at INGEST_LIMIT_ROWS rows (default 6000) — embedding
the full 4.7M-passage corpus on CPU is infeasible for the hackathon;
see decisions.md.

Usage:
    python -m ingest.download --split hi
    python -m ingest.download --split hi --limit 6000
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
import voicerag  # noqa: F401 — sets HF_HOME to data/hf_cache before HF imports

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(os.environ.get("RAW_DATA_DIR", "./data/raw"))

# PRD §4 schema check (dataset uses 'Answer'; PRD's 'answers' maps to it)
EXPECTED_FIELDS = {"source_lang", "target_lang", "meta", "query", "Answer", "passages"}

# Mapping of language code -> (train file, validation file) in the HF repo
LANG_FILE_MAP = {
    "hi": ("train/hintrain.parquet", "validation/hinval.parquet"),
    "te": ("train/teltrain.parquet", "validation/telval.parquet"),
    "ta": ("train/tamtrain.parquet", "validation/tamval.parquet"),
    "bn": ("train/bentrain.parquet", "validation/benval.parquet"),
    "mr": ("train/martrain.parquet", "validation/marval.parquet"),
}

DATASET_REPO = os.environ.get("HF_DATASET", "ai4bharat/MSMARCO-XI")


def _read_parquet_rows(file_path: str, limit: int) -> pd.DataFrame:
    """
    Read a nested MSMARCO-XI parquet via pyarrow row-groups, exploding the
    `passages` struct into plain list columns pandas can handle.
    Stops after `limit` rows.
    """
    pf = pq.ParquetFile(file_path)
    frames = []
    taken = 0

    for rg_idx in range(pf.num_row_groups):
        if taken >= limit:
            break
        table = pf.read_row_group(rg_idx)

        # Schema validation on the first row group
        if rg_idx == 0:
            cols = set(table.column_names)
            missing = EXPECTED_FIELDS - cols
            if missing:
                raise ValueError(f"Schema mismatch — missing fields: {missing}")
            logger.info("Schema validated (row group 0): %s", table.column_names)

        # Flat columns convert cleanly to pandas
        flat = table.select(["query", "Answer", "query_id", "source_lang", "target_lang"]).to_pandas()

        # Nested passages struct -> plain python lists
        passages = table.column("passages").combine_chunks()
        try:
            flat["Translated_passages"] = passages.field("Translated_passages").to_pylist()
            flat["English_passages"] = passages.field("English_passages").to_pylist()
            flat["is_selected"] = passages.field("is_selected").to_pylist()
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Unexpected passages struct layout: {e}")

        remaining = limit - taken
        if len(flat) > remaining:
            flat = flat.iloc[:remaining]
        frames.append(flat)
        taken += len(flat)
        logger.info("  row group %d: %d rows (total %d/%d)", rg_idx, len(flat), taken, limit)

    return pd.concat(frames, ignore_index=True)


def ingest_split(
    split: str = "hi",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
) -> tuple[Path, int]:
    """
    Download the given language split from HuggingFace and save as parquet
    with the passages struct exploded to list columns.

    Validation rows are ingested first (they're the eval split per §10),
    then train rows up to `limit`.

    Returns (parquet_path, row_count).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"msmarco_xi_{split}.parquet"

    if limit is None:
        limit = int(os.environ.get("INGEST_LIMIT_ROWS", 6000))

    if split not in LANG_FILE_MAP:
        raise ValueError(f"Unknown language '{split}'. Supported: {list(LANG_FILE_MAP.keys())}")

    from huggingface_hub import hf_hub_download

    # Validation first (eval queries per PRD §10), then train
    files = [LANG_FILE_MAP[split][1], LANG_FILE_MAP[split][0]]

    frames = []
    taken = 0
    for file_path in files:
        if taken >= limit:
            break
        logger.info("Fetching %s/%s …", DATASET_REPO, file_path)
        cached = hf_hub_download(repo_id=DATASET_REPO, filename=file_path, repo_type="dataset")
        rows_left = limit - taken
        df = _read_parquet_rows(cached, rows_left)
        frames.append(df)
        taken += len(df)
        logger.info("  %s: took %d rows", file_path, len(df))

    df = pd.concat(frames, ignore_index=True)
    n_passages = sum(len(p) for p in df["Translated_passages"])
    logger.info("Ingested %d rows (~%d passages) from %s", len(df), n_passages, split)

    df.to_parquet(out_path, index=False)
    logger.info("Cached to %s", out_path)

    return out_path, len(df)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI from HuggingFace")
    parser.add_argument("--split", default=os.environ.get("HF_DATASET_SPLIT", "hi"),
                        help="Language split to download")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to ingest (default: INGEST_LIMIT_ROWS env or 6000)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    path, rows = ingest_split(args.split, args.output_dir, args.limit)
    print(f"OK: ingested {rows} rows -> {path}")


if __name__ == "__main__":
    main()
