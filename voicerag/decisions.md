# decisions.md — VoiceRAG

Running log of decisions and assumptions made during the build.

---

## Locked decisions (from AGENTS.md)

| Decision | Choice | Rationale |
|---|---|---|
| STT provider | Sarvam (saarika/saaras) | Matches Hindi corpus; PRD's stated lean |
| Vector DB | FAISS + in-memory HNSW, metadata in local SQLite sidecar | In-process, no network hop, best shot at 200ms budget |
| Generation model | Smallest/fastest fast-tier model (TBD in Task 5.1) | Default to fastest; benchmark spike before finalizing |

---

## Build decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-18 | Sentence-window threshold: passages with >150 chars trigger sentence-window chunking | Short MSMARCO passages are already near-optimal; only longer ones benefit from sliding window |
| 2026-08-18 | Semantic chunking cosine-sim threshold: 0.7 | Standard default for topical coherence; tunable after eval |
| 2026-08-18 | BM25 via rank_bm25 (pure Python, no Java dep) | Hackathon-friendly; no Elasticsearch/Jina dependency |
| 2026-08-18 | Embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Multilingual support (incl. Hindi), 384-dim, fast inference |
| 2026-08-18 | Metadata sidecar: SQLite via stdlib sqlite3 | Zero dependency; schema versioned for future migration |
| 2026-08-18 | Ingestion cap: 6000 rows (~60k passages), validation split first, via `INGEST_LIMIT_ROWS` | Full corpus is ~4.7M passages — CPU embedding would take hours; hackathon needs a buildable index. Validation rows first because they double as the §10 eval set. Raising the env var re-runs the full pipeline. |
| 2026-08-18 | pandas cannot read the nested `passages` struct ("Nested data conversions not implemented") — download.py reads via pyarrow row-groups and explodes the struct to list columns | pyarrow-native read path avoids the pandas nested-type limitation entirely |
