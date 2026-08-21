# VoiceRAG

Voice-enabled Retrieval-Augmented Generation system for Hindi/English/Urdu.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup (downloads data & builds index)
bash setup.sh

# Run server
python run.py
```

Then open http://localhost:8000

## Features

- Hindi/English/Urdu bilingual support
- Hybrid retrieval (FAISS + BM25)
- Extractive fast path (~21ms)
- Ollama local LLM (GPU-accelerated)
- TTS speaker button
- Citations & latency breakdown
