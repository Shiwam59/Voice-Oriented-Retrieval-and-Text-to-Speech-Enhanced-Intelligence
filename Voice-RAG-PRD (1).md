# PRD: Voice-Enabled RAG System (VoiceRAG)

**Status:** Draft v1.0
**Owner:** [Team Name]
**Last updated:** Aug 13, 2026
**Target:** Hackathon submission (single, final submission — no resubmissions)

---

## 1. Summary

VoiceRAG is a low-latency, voice-in/voice-answer-out Retrieval-Augmented Generation system. A user speaks a question, the system transcribes it, retrieves grounded context from an indexed corpus built on `ai4bharat/MSMARCO-XI`, and generates a guarded, citation-backed answer — with the full chunking → retrieval → generation path (post-transcription) completing in under 200ms at P50/P70.

**Pipeline:** Voice input → Speech-to-Text (Sarvam or ElevenLabs) → Retrieval (multi-strategy chunked vector DB) → Guarded answer generation → Voice/text output.

---

## 2. Goals & Non-Goals

### Goals
- End-to-end voice question → grounded answer, working live, with a public demo link.
- Sub-200ms latency for the chunking+retrieval+generation segment of the pipeline (STT is measured separately, see §6).
- Multiple, justified chunking strategies — not a single fixed-size splitter.
- A real orchestration harness (not a single prompt call) with retries, structured I/O, and error recovery.
- Guardrails that make the system *refuse or hedge* on off-topic, unsafe, or ungrounded queries.
- Reproducible latency analytics (P50/P70/P100) over a real query set.

### Non-Goals
- Multi-turn conversational memory (single-shot Q&A is sufficient for v1).
- Full multilingual UI — we scope to one primary language pair for the demo (see §4) but design the schema to extend to all MSMARCO-XI languages.
- Production-grade auth, billing, or multi-tenant infra.

---

## 3. Users & Use Case

**Primary user:** Someone who asks a spoken question in an Indic language (or English) and wants a fast, grounded, spoken/text answer sourced from the MS MARCO passage corpus — a stand-in for a voice search / voice assistant / customer-support-over-voice scenario.

**Core user story:** "As a user, I speak a question. Within ~1–2 seconds I hear/read an answer that is actually backed by retrieved passages, and if my question isn't answerable from the corpus (or is unsafe/off-topic), the system tells me that instead of guessing."

---

## 4. Dataset

**Source:** [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (part of AI4Bharat's IndicRAGSuite).

- Machine-translated, multi-stage-validated variant of MS MARCO for Indian languages, with `train`/`validation` splits stored as per-language-pair parquet files (e.g. `telval.parquet` for Telugu validation).
- Each record includes: `source_lang`, `target_lang`, translation `meta` (model name, temperature, max_tokens, top_p, penalties), `query`, `answers`/`Answer`, and `passages` (the retrieval corpus per query).
- **Language scope decision:** we will build the index over the **Hindi** subset (`hi`) as the primary demo language, since Sarvam STT is strong on Hindi/Indic languages; schema is language-agnostic so other subsets (te, ta, bn, mr, ...) can be added by re-running the ingestion job.
- **Corpus construction:** we flatten all `passages` fields across the chosen split into a single deduplicated passage store — this is the retrieval corpus. `query`/`answers` pairs are held out separately and used both as (a) seed content for grounding checks and (b) our latency/eval test set (see §9).

---

## 5. System Architecture

```
┌──────────────┐   ┌───────────────┐   ┌─────────────────────────────┐   ┌───────────────────┐
│  Voice Input  │→ │  STT (Sarvam / │→ │   Retrieval Harness           │→ │  Answer Generation  │
│  (mic/upload) │   │  ElevenLabs)  │   │  - query rewrite/normalize   │   │  - grounded LLM     │
└──────────────┘   └───────────────┘   │  - multi-strategy chunk index │   │  - guardrail checks │
                                        │  - hybrid vector+keyword      │   │  - citation binding │
                                        │  - re-rank + top-k select     │   └───────────────────┘
                                        └─────────────────────────────┘             │
                                                                                     ▼
                                                                          ┌───────────────────┐
                                                                          │ Output: text + TTS │
                                                                          │ + latency/telemetry │
                                                                          └───────────────────┘
```

### Components
1. **Voice capture** — browser mic (WebRTC/MediaRecorder) or file upload, streamed/chunked audio.
2. **STT** — one provider only, chosen up front: **Sarvam** (`saarika` / `saaras` ASR API) for strong Hindi/Indic coverage, or **ElevenLabs** Scribe if we prioritize English + broad accent robustness. *Decision: Sarvam*, to match the Hindi corpus and keep the language story coherent end-to-end (see §12 open decision if this changes).
3. **Retrieval harness** — the orchestration layer described in §8; owns chunking, indexing, retrieval, re-ranking, and guardrail gating.
4. **Answer generation** — an LLM call constrained to retrieved context, with a strict "answer only from context" system prompt, citation tagging back to passage IDs, and a refusal path.
5. **Output** — text answer (always) + optional TTS playback; structured JSON also returned for the demo UI (answer, citations, confidence, latency breakdown).

---

## 6. Latency Budget

Target: **chunking + vector DB retrieval + generation ≤ 200ms** (P50/P70). STT is inherently streaming/variable (network + provider-side) and is tracked as a **separate, reported metric**, not counted inside the 200ms budget, per the requirement wording ("full process — chunking + vector DB retrieval + everything through to final output").

| Stage | Budget (target) | Notes |
|---|---|---|
| Query normalization + embedding | ≤ 25ms | Cached small embedding model, batched warm client |
| Vector search (ANN) | ≤ 40ms | HNSW/IVF index, pre-warmed in-memory |
| Keyword/BM25 pass (hybrid) | ≤ 20ms | Runs in parallel with vector search |
| Re-rank top-k (cross-encoder or lightweight scorer) | ≤ 50ms | Small model or heuristic re-rank to stay in budget |
| Guardrail gate (relevance/grounding check) | ≤ 15ms | Rule-based + score-threshold, not a second LLM call on the hot path |
| Answer generation (LLM call) | ≤ 50ms–remaining budget | Small/fast model, streamed; this is the largest variable cost — see mitigation below |
| **Total (post-STT)** | **≤ 200ms** | |

**Mitigation for the generation step** (the riskiest part of the budget): use a small, fast model with short max_tokens, pre-built prompt templates (no runtime prompt construction overhead), and where the top retrieved passage's answer field already closely matches the query (high-confidence extractive case), short-circuit to an **extractive answer** (return the passage/answer span directly) instead of a full generative call — falling back to generation only when extraction confidence is low.

---

## 7. Chunking & Indexing Strategy

We explicitly avoid a single fixed-size chunker. The MSMARCO-XI corpus is already passage-sized (short, self-contained), so "chunking" here means building **multiple complementary representations** of the same corpus and letting retrieval pick the right one per query:

1. **Passage-native chunking (baseline):** each MS MARCO passage kept as its own chunk (already near-optimal size for this corpus; passages are short/atomic). Metadata: `passage_id`, `query_id` (source query it was collected for), `source_lang`, `target_lang`.
2. **Sentence-window chunking:** for any longer/concatenated passages, split by sentence boundaries with a sliding window (e.g. 2–3 sentences per chunk, 1-sentence overlap) to preserve local context without diluting embeddings.
3. **Semantic chunking:** embed sentences and merge adjacent sentences into a chunk while cosine similarity stays above a threshold (topic-coherent chunks rather than fixed token counts) — used as an alternate index for queries where passage-native retrieval scores are weak.
4. **Metadata-aware indexing:** every chunk is indexed with `language`, `original_query_id`, `char_len`, and a `has_answer_overlap` flag (does this passage's text overlap the known gold answer for its source query — used only for offline eval, never leaked at inference time).
5. **Overlap handling:** sentence-window and semantic chunks use overlapping spans (10–20%) so answer-bearing sentences aren't cut across a chunk boundary; overlapping duplicates are de-duplicated post-retrieval by passage similarity before re-ranking.
6. **Hybrid retrieval:** dense vector search (embedding similarity) + sparse BM25/keyword search run in parallel; results are merged and passed through a lightweight re-ranker (cross-encoder or reciprocal rank fusion) to pick the final top-k.
7. **Index storage:** a vector DB (e.g. Qdrant/Weaviate/FAISS+metadata store — final choice logged in repo README) holding all chunk variants with a `chunk_strategy` field, so retrieval can query one strategy, compare across strategies, or fuse results.

**Why this satisfies "vast":** the system indexes the same corpus three distinct ways (native, sentence-window, semantic) with overlap and metadata, and retrieval is hybrid (dense+sparse+re-rank) rather than a single embedding lookup — a deliberate strategy choice per query characteristics, not one-size-fits-all.

---

## 8. Harness Design (Orchestration)

The model is never called as a single raw prompt-in/text-out step. The harness owns:

- **Structured request/response schema** (typed input: `{audio | transcript, language, session_id}`; typed output: `{answer, citations[], confidence, refused: bool, refusal_reason?, latency_breakdown}`).
- **Tool-call style steps**, each independently retryable: `transcribe()` → `normalize_query()` → `retrieve(top_k, strategy)` → `rerank()` → `guardrail_check()` → `generate()` → `postprocess()`.
- **Retries with backoff** on transient failures (STT timeout, vector DB connection blip, LLM rate limit) — capped retry count so we never blow the latency budget silently; on retry exhaustion, degrade gracefully (return best-effort retrieved passages with a "generation unavailable" flag rather than hanging).
- **Timeouts per stage**, enforced so one slow stage can't eat the whole budget — each stage has a hard deadline; if exceeded, the harness short-circuits to the extractive fallback (§6).
- **Structured error recovery:** STT failure → prompt re-record; empty retrieval → guardrail refusal path (§9), not a hallucinated answer; malformed LLM output → schema validation + one repair retry before falling back to extractive answer.
- **Telemetry hooks** at every stage boundary feeding the latency analytics in §10.

---

## 9. Guardrails

The system must demonstrably know when *not* to answer:

1. **Off-topic detection:** if top retrieval scores (both dense and sparse) fall below a tuned threshold, the query is classified off-corpus and the system responds with a refusal ("I don't have information about that in this dataset") instead of forcing an answer.
2. **Unsafe/inappropriate input handling:** a lightweight moderation pass on the transcribed query (keyword + classifier) blocks generation and returns a safe refusal for unsafe categories, logged separately from normal refusals.
3. **Groundedness / hallucination check:** generated answers are checked against retrieved passage text (e.g. n-gram/entailment overlap check) before being returned; answers that can't be traced to a retrieved chunk are either regenerated once (forcing extractive quoting) or replaced with a refusal + "closest passages" fallback.
4. **Citation binding:** every non-refused answer must carry at least one `passage_id` citation; the harness rejects (and retries) any generation output missing citations.
5. **Confidence scoring surfaced to the user:** the UI shows a confidence indicator (from retrieval score + groundedness check) so refusals and low-confidence answers are visibly distinguished from confident, grounded ones.

---

## 10. Latency Analytics Plan

- **Test set:** sample of held-out `query` entries from MSMARCO-XI (target: ≥100 queries, spanning short/long/ambiguous phrasing) run through the retrieval+generation path (STT stage measured separately using pre-recorded audio prompts of the same queries).
- **Metrics reported:** P50, P70, P100 (max) latency for:
  - retrieval-only (chunking/index lookup + rerank)
  - generation-only
  - full post-STT pipeline
  - STT-only (reported separately, not counted toward the 200ms target)
- **Method:** warm-cache runs only after a fixed warm-up set (documented), repeated ≥3 passes to avoid single-best-case cherry-picking, results committed to the repo as a CSV + summary table/plot in the README.
- **Failure/refusal cases included:** refusal-path latency is measured too (guardrail short-circuit should be fast, not slower than a normal answer).

---

## 11. Milestones

| Phase | Deliverable |
|---|---|
| 1. Data & Index | Ingest MSMARCO-XI (hi), build 3 chunking strategies, stand up vector DB, hybrid retrieval smoke-tested |
| 2. Harness | Orchestration layer with typed I/O, retries, timeouts, telemetry |
| 3. STT integration | Sarvam wired end-to-end, streaming or near-real-time capture |
| 4. Guardrails | Off-topic/unsafe/groundedness checks + refusal UX |
| 5. Latency pass | Extractive fallback, budget tuning, P50/P70/P100 measured and documented |
| 6. Demo & deploy | Live link (hosted), UI polish, telemetry dashboard |
| 7. Submission assets | GitHub repo cleanup + README, team-process video, demo video, promo posts |

---

## 12. Open Decisions

- **STT provider final pick:** Sarvam (leaning, for Hindi/Indic alignment with corpus) vs ElevenLabs (leaning, if we prioritize polish/latency on English demo). Must be locked before build starts — no switching mid-build per "pick one."
- **Vector DB choice:** FAISS (simplest, in-process, best shot at the 200ms budget) vs Qdrant/Weaviate (nicer ops, slight network hop risk). Leaning FAISS or an in-memory HNSW index for latency headroom, given the hard <200ms requirement.
- **Generation model:** must be small/fast enough to fit the remaining latency budget after retrieval — needs a benchmarking spike before final selection.

---

## 13. Submission Checklist (from brief)

- [ ] Submission form filled: https://forms.gle/MNvCjcv23Hn2Eeu58
- [ ] GitHub repo link (public, final state — no resubmissions allowed)
- [ ] Live working link
- [ ] Video 1 — Team/process video, 90s, shows process not product
- [ ] Video 2 — Demo video, end-to-end working demo
- [ ] Both videos posted to Instagram, X, and LinkedIn by **every** team member individually
- [ ] At least 1 public Instagram account among the team
- [ ] Confirm exact required hashtags/mentions/caption text for every post (brief was cut off — verify before posting)
- [ ] P50/P70/P100 latency table included in repo README
- [ ] Guardrail examples (off-topic, unsafe, ungrounded) demonstrated in demo video
