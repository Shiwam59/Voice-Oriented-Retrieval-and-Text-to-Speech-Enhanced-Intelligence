"""VoiceRAG — low-latency voice-enabled RAG over MSMARCO-XI (Hindi)."""

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent

# All heavy artifacts (raw parquet, HF caches, FAISS/BM25 index, SQLite
# sidecar) live under <pkg>/data so the whole folder is movable as one unit.
# HF_HOME must be set before huggingface_hub / sentence_transformers import.
os.environ.setdefault("HF_HOME", str(_PKG_ROOT / "data" / "hf_cache"))

# Auto-detect offline mode: if the embedding model is already cached
# locally, skip network lookups to avoid hanging on slow/absent internet.
_hf_home = Path(os.environ["HF_HOME"])
_model_cache = _hf_home / "hub" / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
if _model_cache.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Set correct index path BEFORE .env loads (so .env can't override it)
os.environ.setdefault("VECTOR_DB_PATH", str(_PKG_ROOT / "data" / "index"))

# Load .env from the package directory so SARVAM_API_KEY / LLM_API_KEY /
# VECTOR_DB_PATH are available without manual sourcing.
try:
    from dotenv import load_dotenv
    load_dotenv(_PKG_ROOT / ".env")
except ImportError:
    pass
