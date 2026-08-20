# AGENTS.md — VoiceRAG

Instructions for any AI coding agent (Claude Code, etc.) working on this repo.
Read this before touching code. If something here conflicts with the PRD
(`Voice-RAG-PRD.md`), **this file wins for implementation details; the PRD
wins for product intent.**

---

## 1. Locked decisions (assumed — flag if you disagree)

The PRD left three items open (§12). To unblock building, these are assumed
locked. If you're a human reviewing this, confirm or override in
`decisions.md` before the agent starts — an agent should not re-litigate
these mid-build.

| Decision | Locked choice | Why |
|---|---|---|
| STT provider | **Sarvam** (`saarika`/`saaras`) | Matches the Hindi corpus; PRD's stated lean |
| Vector DB | **FAISS + in-memory HNSW** (metadata in a local SQLite/JSON sidecar) | In-process, no network hop, best shot at the 200ms budget; PRD's stated lean |
| Generation model | **Smallest/fastest available fast-tier model**, short `max_tokens`, no runtime prompt construction | PRD requires a benchmarking spike — do this in Task 5.1 before finalizing, but default to the fastest model available so early integration isn't blocked |

**If you change any of these, update this table and `decisions.md` in the
same commit — don't let the two drift.**

---

## 2. Repo layout (target)

```
/voicerag
  /ingest          # dataset download, corpus flattening, chunking, indexing
  /harness          # orchestration: transcribe -> normalize -> retrieve -> rerank -> guardrail -> generate -> postprocess
  /guardrails       # off-topic / unsafe / groundedness checks
  /eval             # latency benchmark harness, test set, results CSV
  /api              # HTTP layer exposing the pipeline (typed request/response)
  /ui               # demo frontend (mic capture, answer + citations + latency display)
  /schemas          # shared typed request/response contracts (source of truth)
  /fixtures         # sample records, sample audio, one worked example end-to-end
  decisions.md      # running log of decisions/assumptions made during build
  TASKS.md          # task breakdown with acceptance criteria
  .env.example
  README.md
```

Do not invent an alternate top-level structure without updating this file.

---

## 3. Environment & secrets

Copy `.env.example` to `.env`. Required keys:

```
SARVAM_API_KEY=
LLM_API_KEY=
VECTOR_DB_PATH=./data/index        # local FAISS index path, not a network DB
HF_DATASET=ai4bharat/MSMARCO-XI
HF_DATASET_SPLIT=hi
```

If a key is missing, the relevant module should fail fast with a clear
error — never silently mock a real API call without an explicit
`--mock` / `MOCK_MODE=true` flag.

---

## 4. Hard constraints (non-negotiable)

- **Latency budget:** chunking + retrieval + generation must be measured
  and reported at P50/P70; target is ≤200ms combined, **post-STT only**.
  Any change that risks this budget must be flagged in `decisions.md`,
  not silently merged.
- **No hallucinated answers:** every non-refused answer must carry at
  least one `passage_id` citation (§9.4 of PRD). If the harness can't
  bind a citation, it must retry once, then fall back to extractive
  answer or refusal — never return an uncited generative answer.
- **No second LLM call on the guardrail hot path** — guardrail gating is
  rule-based / score-threshold only (§6 of PRD).
- **Single-shot Q&A only** — do not build multi-turn memory; it's an
  explicit non-goal.
- **One STT provider only** — do not add a fallback/second provider
  mid-build without updating the locked-decisions table above.

---

## 5. Definition of done (per task)

A task in `TASKS.md` is done only when:
1. Code runs locally per the "how to run" section below.
2. It matches the typed contract in `/schemas` (no ad-hoc shapes).
3. It has at least a smoke test (real call or fixture-based).
4. Any deviation from the PRD or this file is recorded in `decisions.md`.
5. Latency-sensitive stages log a timestamp at entry/exit for the
   telemetry hooks required in §8 of the PRD.

---

## 6. How to run / test

All commands run from `/voicerag` (the Python package root).

```bash
pip install -r requirements.txt

# ingest + build index (download → flatten → chunk → FAISS/BM25/metadata)
python -m ingest.download --split hi          # cached raw parquet
python -m ingest.flatten --split hi           # dedup passage store + eval QA
python -m ingest.chunking --all               # 3 chunking strategies
python -m ingest.build_index                  # FAISS HNSW + BM25 + SQLite sidecar
# or all four at once:
python -m ingest.run_pipeline --split hi

# run the harness end-to-end on a sample query (text path)
python -m harness.run --query "कॉर्पोरेशन क्या है?"
python -m harness.run --query "..." --audio fixtures/sample1.wav

# retrieval-only smoke test
python -m harness.retriever --query "कॉर्पोरेशन क्या है?" --top-k 5

# smoke + error-recovery tests (fixtures only, no index needed)
python -m tests.smoke_test
python -m tests.test_error_recovery

# run the latency benchmark
python -m eval.run_benchmark --n 100

# serve API
uvicorn api.app:app --port 8000    # UI at ui/index.html
```

(Agent: update these commands as soon as real entry points exist — this
section must never drift from what actually works.)

---

## 7. Coding conventions

- Prefer explicit typed schemas (Pydantic/TypedDict/TS types) over dicts
  for anything crossing a stage boundary in the harness.
- Each harness stage (`transcribe`, `normalize_query`, `retrieve`,
  `rerank`, `guardrail_check`, `generate`, `postprocess`) is its own
  function/module, independently retryable and independently testable —
  do not collapse stages into one function even if it's shorter.
- No prompt strings built at runtime via string concatenation in the
  hot path — use pre-built templates (per §6 mitigation in the PRD).
- Every stage emits a telemetry event (start, end, success/failure) —
  this feeds §10's latency analytics, so it's not optional instrumentation.

---

## 8. When you're unsure

If a requirement is ambiguous and picking wrong would be expensive to
undo (e.g. index format, schema shape), stop and add an entry to
`decisions.md` describing the fork and your chosen default, rather than
silently picking one. If it's cheap to change later (e.g. a variable
name), just proceed.
