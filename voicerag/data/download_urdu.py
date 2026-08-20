"""
Download Urdu MSMARCO-XI dataset from HuggingFace.

Usage:
    python data/download_urdu.py
"""

from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Urdu MSMARCO-XI dataset...")

# Download train data
print("  Downloading train split...")
path_train = hf_hub_download(
    "ai4bharat/MSMARCO-XI",
    "train/urdtrain.parquet",
    repo_type="dataset"
)
df_train = pd.read_parquet(path_train)
print(f"  Train: {len(df_train)} examples")

# Extract passages
passages = []
for _, row in df_train.iterrows():
    if "passages" in row and isinstance(row["passages"], dict):
        tp = row["passages"].get("Translated_passages", [])
        ep = row["passages"].get("English_passages", [])
        is_sel = row["passages"].get("is_selected", [])
        for i, text in enumerate(tp):
            passages.append({
                "query_id": row.get("query_id", ""),
                "source_lang": "eng_Latn",
                "target_lang": "urd_Arab",
                "text": text,
                "eng_text": ep[i] if i < len(ep) else "",
                "is_selected": is_sel[i] if i < len(is_sel) else 0,
                "passage_id": f"ur_{row.get('query_id', '')}_{i}",
                "char_len": len(text),
            })

passages_df = pd.DataFrame(passages)
passages_df.to_parquet(RAW_DIR / "passages_urdu.parquet", index=False)
print(f"  Passages: {len(passages_df)} extracted")

# Download validation data
print("  Downloading validation split...")
path_val = hf_hub_download(
    "ai4bharat/MSMARCO-XI",
    "validation/urdval.parquet",
    repo_type="dataset"
)
df_val = pd.read_parquet(path_val)
print(f"  Validation: {len(df_val)} examples")

# Extract eval QA pairs
eval_rows = []
for _, row in df_val.iterrows():
    eval_rows.append({
        "query_id": str(row.get("query_id", "")),
        "query": row.get("query", ""),
        "answer": row.get("Answer", ""),
        "eng_query": row.get("Eng_Query", ""),
        "eng_answer": row.get("Eng_Answer", ""),
        "source_lang": "eng_Latn",
        "target_lang": "urd_Arab",
    })
eval_df = pd.DataFrame(eval_rows)
eval_df.to_parquet(RAW_DIR / "eval_qa_urdu.parquet", index=False)
print(f"  Eval QA: {len(eval_df)} pairs")

print("\nDone! Urdu dataset saved to data/raw/")
print("Next step: Run build_urdu_index.py to build the index")
