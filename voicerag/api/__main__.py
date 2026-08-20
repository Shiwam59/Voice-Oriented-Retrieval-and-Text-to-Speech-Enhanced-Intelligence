"""API entry point.

Usage:
    python -m voicerag.api          # runs on port 8000
    python -m voicerag.api 8080     # custom port
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import uvicorn

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

if __name__ == "__main__":
    print(f"Starting VoiceRAG API on http://localhost:{port}")
    uvicorn.run("voicerag.api.app:app", host="0.0.0.0", port=port, reload=True)
