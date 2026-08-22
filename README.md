<div align="center">

# 🎙️ VoiceRAG

### Voice-Enabled Retrieval-Augmented Generation

**Hindi | English | Urdu** — Ask anything about India, get accurate answers with citations

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-purple)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🗣️ **Bilingual RAG** | Ask in Hindi, English, or Urdu — get answers in the same language |
| ⚡ **Hybrid Retrieval** | FAISS + BM25 for accurate document search |
| 🎯 **Extractive Fast Path** | Simple answers in ~21ms (no LLM needed) |
| 🤖 **Local LLM** | Ollama + qwen2.5:1.5b — no cloud dependency |
| 🔊 **Text-to-Speech** | Auto-speak answers in Hindi/English |
| 🎤 **Voice Input** | Record questions via microphone |
| 📊 **Citations** | Every answer linked to source documents |
| 📈 **Latency Analytics** | P50/P70/P100 latency breakdown |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Voice/Input │────▶│  STT (Sarvam) │────▶│  Normalize   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                   ┌──────────────┐     ┌───────▼──────┐
                   │  LLM (Ollama)│◀────│  Hybrid RAG   │
                   └──────┬───────┘     │ FAISS + BM25  │
                          │             └──────────────┘
                   ┌──────▼───────┐
                   │  TTS (Speak) │
                   └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama (for local LLM)

### 1. Clone & Setup
```bash
git clone https://github.com/Shiwam59/Voice-Oriented-Retrieval-and-Text-to-Speech-Enhanced-Intelligence.git
cd Voice-Oriented-Retrieval-and-Text-to-Speech-Enhanced-Intelligence
bash setup.sh
```

### 2. Configure API Keys
Create `voicerag/.env`:
```env
SARVAM_API_KEY=your_sarvam_api_key_here
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### 3. Run
```bash
python run.py
```

Open **http://localhost:8000** in your browser! 🎉

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **RAG Retrieval** | ~20ms ⚡ |
| **Extractive Path** | ~21ms ⚡ |
| **LLM Generation** | ~950ms (GPU) |
| **Total (Extractive)** | ~21ms ✅ |
| **Total (LLM)** | ~970ms |
| **Index Size** | 1,617 vectors |
| **Languages** | Hindi, English, Urdu |

### Latency Breakdown
| Stage | Time | Description |
|-------|------|-------------|
| Normalize | ~0.2ms | Query cleaning |
| Retrieve | ~20ms | FAISS + BM25 hybrid search |
| Rerank | ~0ms | (skipped for speed) |
| Guardrail | ~0ms | Content filtering |
| Generate | ~0.7ms | Extractive answer |
| **Total** | **~21ms** | **No LLM needed** |

---

## 🗂️ Dataset

Based on [AI4Bharat MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) multilingual dataset:

- **1,617 passages** indexed (Hindi)
- **Topics**: Geography, History, Culture, Science, Technology
- **Languages**: Hindi, English, Urdu (via translation)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Vector DB** | FAISS (HNSW index) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Embeddings** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| **LLM** | Ollama + qwen2.5:1.5b (local, GPU-accelerated) |
| **STT** | Sarvam AI (saaras:v3) |
| **Frontend** | HTML/CSS/JS (dark theme) |

---

## 📁 Project Structure

```
VoiceRAG/
├── run.py                  # Server entry point
├── setup.sh               # Setup script
├── requirements.txt       # Python dependencies
├── voicerag/
│   ├── api/               # FastAPI server
│   │   └── app.py         # REST API endpoints
│   ├── harness/           # RAG pipeline
│   │   ├── orchestrator.py # Pipeline orchestration
│   │   ├── retriever.py   # Hybrid retrieval
│   │   └── stages.py      # STT, normalize, generate
│   ├── ingest/            # Index builder
│   │   └── build_index.py # FAISS + BM25 builder
│   ├── schemas/           # Data models
│   ├── tests/             # Unit tests
│   └── ui/                # Frontend
│       └── index.html     # Dark theme UI
└── data/
    ├── raw/               # Downloaded datasets
    └── index/             # Built search index
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Text query → answer |
| `POST` | `/query/audio` | Audio query → transcript → answer |
| `GET` | `/stats` | Latency statistics |

### Example Request
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "गोवा कहाँ है?"}'
```

### Example Response
```json
{
  "answer": "गोवा भारत का एक छोटा राज्य है...",
  "citations": ["p_goa_001", "p_goa_002"],
  "confidence": 0.58,
  "transcript": "गोवा कहाँ है?",
  "latency_breakdown": {
    "normalize_ms": 0.2,
    "retrieve_ms": 20.0,
    "generate_ms": 0.7
  }
}
```

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing`
5. Open Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for Indian languages**

⭐ Star this repo if you found it useful!

</div>
