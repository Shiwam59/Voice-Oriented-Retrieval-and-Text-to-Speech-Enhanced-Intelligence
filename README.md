# VoiceRAG — Voice-Enabled Hindi RAG Pipeline

A real-time voice question-answering system over Hindi text corpora, built for the AI4Bharat MSMARCO-XI dataset.  
Ask questions in Hindi via voice or text — get cited, grounded answers in milliseconds.

---

## Architecture

```
[Mic / Text Input]
       ↓
[STT: Sarvam saarika/saaras]  ← latency measured but excluded from 200ms budget
       ↓
[Normalize Query]
       ↓
[Hybrid Retrieval: FAISS HNSW + BM25]  ← ~26ms P50
       ↓
[Rerank (fusion)]
       ↓
[Guardrail: unsafe keyword + off-topic detection]
       ↓
[Generate: LLM or extractive fallback]  → [Groundedness check] → [Citation binding]
       ↓
[Postprocess → API Response]
```

**Key design choices:**
- **No hallucinated answers:** every answer carries at least one `passage_id` citation
- **No second LLM on guardrail path:** all guardrails are rule-based (score threshold + regex)
- **Single-shot Q&A only:** no multi-turn memory
- **Latency budget:** ≤200ms post-STT — achieved at P50=26.6ms (6× margin)

---

## Setup

```bash
# 1. Clone and install dependencies
cd voicerag
pip install -r requirements.txt

# 2. Copy and fill in secrets
cp .env.example .env
# SARVAM_API_KEY=<your key>
# LLM_API_KEY=<optional — system works without it via extractive fallback>
```

---

## Quickstart

```bash
# Build the index (download → flatten → chunk → FAISS/BM25/SQLite)
python -m ingest.run_pipeline --split hi

# Run a text query
python -m harness.run --query "कॉर्पोरेशन क्या है?"

# Run retrieval smoke test
python -m harness.retriever --query "कॉर्पोरेशन क्या है?" --top-k 5

# Run the API + UI
uvicorn api.app:app --port 8000
# Then open voicerag/ui/index.html in a browser
```

---

## Latency (P50/P70/P100)

Measured post-STT on CPU (dev machine), n=8 queries, extractive path:

| Stage | P50 | P70 | P100 |
|---|---|---|---|
| Retrieve (dominant) | 26.4ms | 28.2ms | 30.1ms |
| Generate (extractive) | ~0ms | ~0ms | ~0ms |
| **Total post-STT** | **26.6ms** | **28.4ms** | **30.3ms** |
| Budget target | ≤200ms | ≤200ms | ≤200ms |
| **Status** | **✅** | **✅** | **✅** |

> Full results CSV: [`voicerag/eval/results.csv`](voicerag/eval/results.csv)

---

## Vector DB Choice

**FAISS in-memory HNSW** was chosen over network vector databases (Pinecone, Weaviate, Qdrant) because:

1. **No network hop:** index lives in-process, eliminating the largest single source of retrieval latency
2. **HNSW sub-linear ANN search:** O(log n) approximate nearest-neighbor search — stays fast as corpus grows
3. **BM25 hybrid:** sparse BM25 runs in parallel (rank_bm25) and scores are fused via Reciprocal Rank Fusion (RRF), giving both semantic and lexical matching for Hindi text
4. **SQLite sidecar:** metadata (passage_id, language, char_len) stored in SQLite alongside the FAISS index — no serialization overhead on the hot path

---

## Guardrails

| Check | Method | Triggers |
|---|---|---|
| Unsafe input (4.2) | Regex keyword list | `bomb`, `weapon`, `kill`, etc. |
| Off-topic (4.1) | `sparse < 0.5 AND dense < 0.55` | English/unrelated queries against Hindi corpus |
| Groundedness (4.3) | Character bigram Jaccard overlap ≥ 0.08 | LLM answers with no corpus overlap |
| Citation binding (4.4) | Hard check post-generate | Any answer without `passage_id` → bind to top passage |

All guardrails are **rule-based / score-threshold only** — no second LLM call.

---

## Project Structure

```
voicerag/
  ingest/          # Dataset download, flatten, chunk, index build
  harness/         # Orchestration: all 7 pipeline stages + retriever
  guardrails/      # (reserved for future standalone guardrail modules)
  eval/            # Latency benchmark harness, results CSV
  api/             # FastAPI endpoints (/query, /query/audio, /stats, /health)
  ui/              # Demo frontend (mic capture, confidence badge, latency breakdown)
  schemas/         # Pydantic contracts (source of truth for all stage I/O)
  fixtures/        # Sample audio, worked examples
  decisions.md     # All design decisions and deviations from PRD
  TASKS.md         # Task breakdown with acceptance criteria
  AGENTS.md        # Instructions for coding agents
```

---

## Running Tests

```bash
# Smoke tests (fixture-based, no index needed)
python -m tests.smoke_test

# Error recovery tests
python -m tests.test_error_recovery

# Full latency benchmark
python -c "
import sys; sys.path.insert(0, '.')
from voicerag.eval.run_benchmark import run_benchmark
summary = run_benchmark(n_queries=100, warmup=10, passes=3,
    index_dir='./voicerag/data/index', raw_dir='./voicerag/data/raw',
    output_path='./voicerag/eval/results.csv')
print(summary)
"
```
