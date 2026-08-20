# TASKS.md — VoiceRAG

Each task is meant to be small enough for a single agent session. Check
off only when the acceptance criteria are met. Record any deviation in
`decisions.md`.

---

## Milestone 1 — Data & Index

- [x] **1.1 Ingest MSMARCO-XI (hi)**
  Download `hi` split (train + validation) from `ai4bharat/MSMARCO-XI`.
  *Done when:* raw parquet cached locally; row count logged; schema
  (`source_lang`, `target_lang`, `meta`, `query`, `answers`, `passages`)
  matches PRD §4.

- [x] **1.2 Flatten corpus**
  Flatten all `passages` across the split into one deduplicated passage
  store, holding `query`/`answers` out separately.
  *Done when:* passage store has unique `passage_id`s; held-out
  query/answer set saved separately for eval (§10).

- [x] **1.3 Passage-native chunking**
  Each passage = one chunk, tagged with `passage_id`, `query_id`,
  `source_lang`, `target_lang`.
  *Done when:* index-ready records produced for 100% of passages.

- [x] **1.4 Sentence-window chunking**
  2–3 sentence sliding window, 1-sentence overlap, for longer passages.
  *Done when:* applied to all passages above a defined length threshold
  (define threshold in `decisions.md`).

- [x] **1.5 Semantic chunking**
  Merge adjacent sentences while cosine similarity stays above a tuned
  threshold.
  *Done when:* alternate index populated; threshold value recorded in
  `decisions.md`.

- [x] **1.6 Metadata-aware indexing**
  Every chunk carries `language`, `original_query_id`, `char_len`,
  `has_answer_overlap` (offline-eval only, never surfaced at inference).
  *Done when:* spot-check confirms `has_answer_overlap` is excluded from
  any inference-time payload.

- [x] **1.7 Overlap de-duplication**
  Post-retrieval de-dup of overlapping spans by passage similarity.
  *Done when:* a query with known-duplicate spans returns de-duped
  results in a smoke test.

- [x] **1.8 Stand up vector DB + hybrid retrieval smoke test**
  FAISS/in-memory HNSW (see locked decision) + BM25 in parallel, merged.
  *Done when:* a hand-picked query returns plausible top-k passages from
  both dense and sparse paths.

---

## Milestone 2 — Harness

- [x] **2.1 Define typed schemas**
  `/schemas`: input `{audio | transcript, language, session_id}`,
  output `{answer, citations[], confidence, refused, refusal_reason?, latency_breakdown}`.
  *Done when:* schemas importable by both harness and API layer; no
  duplicate ad-hoc shapes elsewhere in the codebase.

- [x] **2.2 Stage functions**
  Implement `transcribe()`, `normalize_query()`, `retrieve(top_k, strategy)`,
  `rerank()`, `guardrail_check()`, `generate()`, `postprocess()` as
  independently callable, independently retryable units.
  *Done when:* each has a standalone smoke test with a fixture input.

- [x] **2.3 Retries with backoff**
  Capped retry count per stage on transient failure (STT timeout, DB
  blip, LLM rate limit).
  *Done when:* a simulated transient failure recovers within the cap;
  exceeding the cap degrades gracefully (§2.5 below), never hangs.

- [x] **2.4 Per-stage timeouts**
  Hard deadline per stage; on exceed, short-circuit to extractive
  fallback (PRD §6).
  *Done when:* an artificially slowed stage triggers the fallback path
  in a test, not a budget blowout.

- [x] **2.5 Structured error recovery**
  STT failure → re-record prompt; empty retrieval → refusal path (not
  hallucination); malformed LLM output → schema validation + one repair
  retry → extractive fallback.
  *Done when:* each of the three failure modes has a test proving the
  correct fallback path fires.

- [x] **2.6 Telemetry hooks**
  Every stage boundary emits start/end/success/failure events.
  *Done when:* a single pipeline run produces a complete latency
  breakdown consumable by Milestone 5's eval harness.

---

## Milestone 3 — STT Integration

- [x] **3.1 Wire Sarvam end-to-end**
  Real API call from `transcribe()`, config via `.env`.
  *Done when:* a real audio fixture produces a real transcript.

- [x] **3.2 Streaming/near-real-time capture**
  Browser mic capture (WebRTC/MediaRecorder) or chunked upload feeding
  `transcribe()`.
  *Done when:* a live mic recording reaches the harness without a
  full-file round trip blocking the UI.

- [x] **3.3 STT latency measured separately**
  *Done when:* STT timing is logged but excluded from the 200ms budget
  calculation, per PRD §6.

---

## Milestone 4 — Guardrails

- [x] **4.1 Off-topic detection**
  Threshold on dense+sparse retrieval scores; below threshold → refusal.
  *Done when:* a known off-corpus query returns the refusal message, not
  a forced answer.

- [x] **4.2 Unsafe/inappropriate input handling**
  Keyword + classifier pass on transcribed query; blocks generation for
  unsafe categories; logged separately from normal refusals.
  *Done when:* a test unsafe query is blocked and logged distinctly.

- [x] **4.3 Groundedness / hallucination check**
  N-gram/entailment overlap check between generated answer and retrieved
  passages; ungrounded output → regenerate once (forced extractive) →
  else refusal + closest-passages fallback.
  *Done when:* an intentionally ungrounded generation is caught and
  either corrected or refused.

- [x] **4.4 Citation binding**
  Reject and retry any generation missing a `passage_id` citation.
  *Done when:* a citation-less generation output triggers a retry, and
  a persistent failure falls back per §2.5.

- [x] **4.5 Confidence indicator in UI**
  Surface retrieval score + groundedness result as a visible confidence
  signal, distinguishing refusals/low-confidence from confident answers.
  *Done when:* three demo queries (confident, low-confidence, refused)
  are visibly distinct in the UI.

---

## Milestone 5 — Latency Pass

- [x] **5.1 Generation model benchmarking spike**
  Compare candidate small/fast models against the remaining post-retrieval
  budget.
  *Done when:* a chosen model is recorded in `decisions.md` with the
  benchmark numbers that justified it (updates the locked-decision table
  in `AGENTS.md` if it changes from the default).

- [x] **5.2 Extractive fallback**
  When top passage's answer field closely matches the query, return the
  extractive span directly instead of calling generation.
  *Done when:* a high-confidence-extractive query bypasses the LLM call
  in a test and still returns a valid, cited answer.

- [x] **5.3 Budget tuning**
  Tune per-stage timings against the ≤200ms P50/P70 target from §6's
  table.
  *Done when:* a full pipeline run on the eval set hits the target, or
  the shortfall is documented with a plan.

- [x] **5.4 P50/P70/P100 measured and documented**
  *Done when:* results committed as CSV + summary table in the README,
  per §10.

---

## Milestone 6 — Demo & Deploy

- [ ] **6.1 Live hosted link**
  *Done when:* a public URL serves the working demo end-to-end.

- [x] **6.2 UI polish**
  Answer, citations, confidence indicator, latency breakdown all visible.
  *Done when:* a first-time user can complete a query without
  explanation.

- [x] **6.3 Telemetry dashboard**
  *Done when:* stage-level latency and refusal rates are viewable
  somewhere (even a simple table/plot page).

---

## Milestone 7 — Submission Assets

- [x] **7.1 GitHub repo cleanup + README**
  Include architecture summary, setup instructions, latency table,
  vector DB choice justification (per PRD §7.7).

- [ ] **7.2 Team/process video** (90s, process not product)

- [ ] **7.3 Demo video** (end-to-end, includes guardrail examples:
  off-topic, unsafe, ungrounded — per PRD §13)

- [ ] **7.4 Submission checklist**
  Cross-check against PRD §13 in full before final submission (no
  resubmissions allowed).
