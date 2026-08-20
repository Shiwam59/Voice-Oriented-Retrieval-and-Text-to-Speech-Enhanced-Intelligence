#!/usr/bin/env python3
"""Run the VoiceRAG API server.

Usage:
    python run.py              # default port 8000
    python run.py --port 8080  # custom port
"""

import os
import sys
from pathlib import Path

# Ensure the parent of the voicerag package is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Set HF_HOME and auto-detect offline mode BEFORE any imports.
# uvicorn with reload=True spawns child processes that inherit os.environ,
# so setting env vars here guarantees they're available everywhere.
_VOICERAG_DIR = _PROJECT_ROOT / "voicerag"
os.environ.setdefault("HF_HOME", str(_VOICERAG_DIR / "data" / "hf_cache"))
os.environ["VECTOR_DB_PATH"] = str(_VOICERAG_DIR / "data" / "index")

_model_cache = (
    Path(os.environ["HF_HOME"])
    / "hub"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
)
if _model_cache.exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = 8000
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    print(f"Starting VoiceRAG API on http://localhost:{port}")
    print(f"Open http://localhost:{port} in your browser")
    uvicorn.run(
        "voicerag.api.app:app",
        host="0.0.0.0",
        port=port,
        # Reload can create duplicate listener processes on Windows and keep
        # an old .env value alive. Restart manually after changing .env.
        reload=False,
    )
