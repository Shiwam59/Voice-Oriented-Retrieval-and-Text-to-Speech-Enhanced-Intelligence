# decisions.md — VoiceRAG

Running log of decisions, assumptions, and deviations made during build.
See `AGENTS.md` §1 for the locked-decisions table.

---

## Locked decisions (from AGENTS.md §1)

| Decision | Locked choice | Status |
|---|---|---|
| STT provider | Sarvam (`saarika`/`saaras`) | ✅ Implemented |
| Vector DB | FAISS + in-memory HNSW + SQLite/JSON sidecar | ✅ Implemented |
| Generation model | Extractive fallback as default (no LLM key required); LLM path available via `LLM_API_KEY` | ✅ Documented below |

---

## Decision log

### 2026-08-18 — Ingest limit capped at 150 rows

**Reason:** Full MSMARCO-XI Hindi split is ~10k rows. Building FAISS on CPU with all rows
takes >5min per ingest run, killing dev iteration speed.

**Decision:** `INGEST_LIMIT_ROWS=150` during development. Remove cap for production.

**Impact:** Retrieval results will be limited to 150 passages; recall will be lower than production.

---

### 2026-08-18 — Chunking strategy thresholds

**Sentence-window chunking:**
- Applied to passages with character length > 300 chars
- Window: 2-3 sentences, 1-sentence overlap

**Semantic chunking:**
- Cosine similarity threshold: 0.75 (sentences merged while similarity stays above this)
- Uses `paraphrase-multilingual-MiniLM-L12-v2` embedding model (same as dense retrieval)

Both thresholds recorded here per Task 1.4/1.5 acceptance criteria.

---

### 2026-08-18 — Off-topic detection signal (Task 4.1)

**Problem:** RRF fusion normalizes all fused scores to a similar range (~0.016–0.030),
making a simple fused-score threshold unreliable for separating English off-corpus queries
from Hindi on-corpus queries.

**Calibration data:**
- Off-topic query ("who won IPL 2026?"): dense=0.48, sparse=0.0
- On-topic query ("कॉर्पोरेशन क्या है?"): dense=0.62, sparse=8.8

**Decision:** Use combined signal:
- `max(sparse_scores) < 0.5 AND max(dense_scores) < 0.55` → off-topic refusal
- Rationale: BM25=0 means no Hindi vocabulary overlap at all (corpus is Hindi).
  Dense < 0.55 catches low-quality semantic matches.

---

### 2026-08-18 — Groundedness check threshold (Task 4.3)

**Method:** Character bigram (n=2) Jaccard overlap between generated answer and top-3
retrieved passages.

**Calibration:**
- Clearly unrelated English text against Hindi corpus: overlap = 0.0 → correctly flagged
- Self-overlap (passage against itself): overlap = 0.67 → correctly passes
- Hindi text from same domain (e.g., extractive answer): overlap > 0.08 → correctly passes

**Decision:** threshold = 0.08

**Scope:** Only applied to LLM-generated answers (`extractive=False`). Extractive answers
bypass the check since they are definitionally grounded.

---

### 2026-08-18 — Generation model (Task 5.1)

**Status:** No LLM API key available in dev environment (`LLM_API_KEY` not set).

**Default path:** Extractive fallback — top passage text (truncated to 300 chars) is returned
directly, always with a `passage_id` citation. This satisfies the "fastest model" requirement
from the locked-decisions table and the ≤200ms latency budget.

**LLM path:** When `LLM_API_KEY` is set, `harness/stages.py::generate()` calls the
configured LLM endpoint. Citation binding is enforced: if LLM output contains no `[pN]`
reference, it is retried once then falls back to extractive.

**Benchmark numbers (n=8, 1 pass, extractive path, CPU, dev machine):**
| Metric | Value |
|---|---|
| Post-STT P50 | 26.6ms |
| Post-STT P70 | 28.4ms |
| Post-STT P100 | 30.3ms |
| Retrieve P50 | 26.4ms |
| Generate P50 | ~0ms (extractive) |
| Budget target | ≤200ms |
| **Status** | **✅ Met (by ~6×)** |

---

### 2026-08-18 — Extractive fallback (Task 5.2)

Extractive fallback is triggered by:
1. `LLM_API_KEY` not set (dev/test environment)
2. LLM generation stage timeout or retry exhaustion
3. Groundedness check failure (post-generate)
4. Any stage where `top_passage` is available

All extractive answers carry the `passage_id` of the source passage — satisfying the
no-uncited-answer constraint from `AGENTS.md §4`.

### 2026-08-20 — Retrieval hot-path optimization

**Change:** Reduced hybrid retrieval over-fetch from `top_k * 10` to a bounded candidate set of 20 (configurable with `RETRIEVAL_CANDIDATES`), replaced full BM25 sorting with `argpartition` plus candidate-only sorting, added a 256-entry LRU cache for normalized query embeddings, and changed selected metadata loading to one SQLite `IN` query.

**Reason:** The post-STT budget is ≤200ms. Dense embedding inference and unnecessary BM25/SQLite work were avoidable hot-path costs; repeated UI/demo queries also re-encoded the same text.

**Tradeoff:** A smaller candidate set can reduce recall on larger corpora. The default of 20 is appropriate for the current 150-row development index and must be rebenchmarked after removing `INGEST_LIMIT_ROWS` for production.
