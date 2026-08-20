"""
Harness stage functions — Task 2.2

Each function is independently callable, independently retryable, and
emits telemetry events (Task 2.6). No stage calls another directly;
orchestration is handled in orchestrator.py.

Stages:
    transcribe()      — audio → text (Sarvam STT)
    normalize_query() — clean/normalize the transcribed query
    retrieve()        — hybrid dense+sparse retrieval → top-k passages
    rerank()          — re-rank top-k passages
    guardrail_check() — off-topic / unsafe / groundedness gates
    generate()        — LLM answer from context with citations
    postprocess()     — format final output
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import httpx
import numpy as np

from voicerag.schemas.contracts import (
    GenerationResult,
    GuardrailResult,
    PipelineRequest,
    PipelineResponse,
    RefusalReason,
    RetrievalResult,
    StageTiming,
)

logger = logging.getLogger(__name__)

# ── Telemetry helper ────────────────────────────────────────────────

_telemetry_store: list[StageTiming] = []


def get_telemetry() -> list[StageTiming]:
    """Return all recorded stage timings for the current pipeline run."""
    return list(_telemetry_store)


def clear_telemetry() -> None:
    """Clear telemetry for a new pipeline run."""
    _telemetry_store.clear()


def _emit_timing(stage_name: str, start_ms: float, status: str = "success") -> StageTiming:
    """Record and return a stage timing event."""
    end_ms = time.perf_counter() * 1000
    timing = StageTiming(
        stage_name=stage_name,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
        status=status,
    )
    _telemetry_store.append(timing)
    logger.info("[telemetry] %s: %.2f ms (%s)", stage_name, timing.duration_ms, status)
    return timing


# ── Helpers ────────────────────────────────────────────────────────

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"


def _get_sarvam_key():
    """Read SARVAM_API_KEY lazily so .env is loaded first."""
    return os.environ.get("SARVAM_API_KEY", "")


# ── Language detection ──────────────────────────────────────────────
_DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')
_ARABIC_RANGE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')


def detect_language(text: str) -> str:
    """
    Detect whether the query is Hindi, Urdu, or English.
    - Devanagari >30% → Hindi
    - Arabic >30% → Urdu
    - Otherwise → English
    """
    if not text.strip():
        return "hi"
    devanagari_chars = len(_DEVANAGARI_RANGE.findall(text))
    arabic_chars = len(_ARABIC_RANGE.findall(text))
    total_alpha = len(re.findall(r'\w', text)) + devanagari_chars + arabic_chars
    if total_alpha == 0:
        return "hi"
    dev_ratio = devanagari_chars / total_alpha
    ar_ratio = arabic_chars / total_alpha
    if dev_ratio > 0.3:
        lang = "hi"
    elif ar_ratio > 0.3:
        lang = "ur"
    else:
        lang = "en"
    logger.info("[lang] detected '%s' (devanagari=%.1f%%, arabic=%.1f%%)", lang, dev_ratio * 100, ar_ratio * 100)
    return lang


def translate(
    text: str,
    source_lang: str,
    target_lang: str,
    timeout_ms: float = 5000,
) -> tuple[str, StageTiming]:
    """
    Translate text using Sarvam API.
    source_lang/target_lang: 'hi' or 'en' (converted to BCP-47 internally).
    Returns (translated_text, timing).
    """
    start_ms = time.perf_counter() * 1000

    sarvam_key = _get_sarvam_key()
    if not sarvam_key:
        logger.warning("SARVAM_API_KEY not set — skipping translation")
        _emit_timing("translate", start_ms, "mock")
        return text, _emit_timing("translate", start_ms, "mock")

    # Convert short codes to BCP-47
    bcp_map = {"hi": "hi-IN", "en": "en-IN"}
    src = bcp_map.get(source_lang, source_lang)
    tgt = bcp_map.get(target_lang, target_lang)

    try:
        with httpx.Client(timeout=timeout_ms / 1000) as client:
            resp = client.post(
                SARVAM_TRANSLATE_URL,
                headers={
                    "api-subscription-key": sarvam_key,
                    "Content-Type": "application/json",
                },
                json={
                    "input": text,
                    "source_language_code": src,
                    "target_language_code": tgt,
                    "model": "mayura:v1",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            translated = data.get("translated_text", text)
            logger.info("[translate] '%s' -> '%s'", text[:50], translated[:50])
    except Exception as e:
        logger.warning("Translation failed: %s — using original text", e)
        _emit_timing("translate", start_ms, "error")
        return text, _emit_timing("translate", start_ms, "error")

    _emit_timing("translate", start_ms)
    return translated, _telemetry_store[-1]


def transcribe(
    audio: bytes,
    language: str = "hi",
    timeout_ms: float = 10000,
    content_type: str = "audio/wav",
) -> tuple[str, StageTiming]:
    """
    Transcribe audio using Sarvam STT (saarika model).
    Returns (transcript_text, timing).
    """
    start_ms = time.perf_counter() * 1000

    sarvam_key = _get_sarvam_key()
    if not sarvam_key:
        logger.warning("SARVAM_API_KEY not set — returning mock transcript")
        _emit_timing("transcribe", start_ms, "mock")
        return "[mock transcript: SARVAM_API_KEY not set]", _emit_timing("transcribe", start_ms, "mock")

    # Map browser MIME types to file extensions for Sarvam
    ext_map = {
        "audio/webm": "audio.webm",
        "audio/webm;codecs=opus": "audio.webm",
        "audio/wav": "audio.wav",
        "audio/wave": "audio.wav",
        "audio/x-wav": "audio.wav",
        "audio/ogg": "audio.ogg",
        "audio/ogg;codecs=opus": "audio.ogg",
        "audio/opus": "audio.opus",
        "audio/mp3": "audio.mp3",
        "audio/mpeg": "audio.mp3",
        "audio/aac": "audio.aac",
        "audio/mp4": "audio.m4a",
        "audio/x-m4a": "audio.m4a",
    }
    # Browsers commonly report values such as
    # `audio/webm;codecs=opus`. Keep the codec information for choosing the
    # filename, but send Sarvam the base media type in the multipart part.
    # Some API gateways reject MIME parameters in the file Content-Type.
    normalized_content_type = (content_type or "audio/webm").split(";", 1)[0].strip().lower()
    filename = ext_map.get(content_type, ext_map.get(normalized_content_type, "audio.webm"))
    logger.info("[transcribe] sending %s as %s (%d bytes) to Sarvam", content_type, normalized_content_type, len(audio))

    # Use 'unknown' for auto-detection when language is 'auto'
    if language == "auto":
        lang_code = "unknown"
    elif len(language) == 2:
        lang_code = f"{language}-IN"
    else:
        lang_code = language

    try:
        with httpx.Client(timeout=timeout_ms / 1000) as client:
            resp = client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": sarvam_key},
                files={"file": (filename, audio, normalized_content_type)},
                data={
                    "model": "saaras:v3",
                    "mode": "transcribe",
                    "language_code": lang_code,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            transcript = data.get("transcript", "")
    except httpx.TimeoutException:
        _emit_timing("transcribe", start_ms, "timeout")
        raise
    except httpx.HTTPStatusError as e:
        # Preserve Sarvam's response body in logs so invalid audio, key,
        # quota, and request-shape failures are diagnosable without exposing
        # the API key to the client.
        detail = e.response.text[:500] if e.response is not None else str(e)
        logger.error("Sarvam STT request failed (%s): %s", e.response.status_code if e.response else "unknown", detail)
        _emit_timing("transcribe", start_ms, "error")
        raise RuntimeError(f"Sarvam STT request failed: {detail}") from e
    except Exception:
        _emit_timing("transcribe", start_ms, "error")
        raise

    _emit_timing("transcribe", start_ms)
    return transcript, _telemetry_store[-1]


# ── Normalize query ────────────────────────────────────────────────

# Common Hindi text normalization patterns
_NORMALIZE_PATTERNS = [
    (r'\s+', ' '),           # Collapse whitespace
    (r'[।]+', '।'),          # Collapse multiple dandas
    (r'[\u200c\u200d]', ''), # Remove zero-width joiners
]


# ── Query expansion (C) ──────────────────────────────────────────
# Maps common Hindi query terms to related terms for better recall.
_HINDI_SYNONYMS = {
    "कॉर्पोरेशन": ["निगम", "कंपनी"],
    "निगम": ["कॉर्पोरेशन", "कंपनी"],
    "पानी": ["जल", "द्रव्य"],
    "जल": ["पानी", "द्रव्य"],
    "सूत्र": ["फॉर्मूला", "समीकरण"],
    "राजधानी": ["शहर", "नगर"],
    "शहर": ["नगर", "राजधानी"],
    "देश": ["राष्ट्र"],
    "राष्ट्रपति": ["प्रेसिडेंट"],
    "प्रधानमंत्री": ["पीएम"],
    "शिक्षा": ["पढ़ाई", "विद्या"],
    "स्वास्थ्य": ["चिकित्सा"],
    "अर्थव्यवस्था": ["आर्थिक"],
    "कानून": ["कानूनी"],
    "इतिहास": ["ऐतिहासिक"],
    "भूगोल": ["भौगोलिक"],
    "विज्ञान": ["वैज्ञानिक"],
    "प्रौद्योगिकी": ["तकनीक"],
    "खेल": ["क्रिकेट"],
    "संविधान": ["कानून"],
    "मौसम": ["जलवायु"],
    "त्योहार": ["उत्सव"],
    "भाषा": ["बोली"],
    "संस्कृति": ["सभ्यता"],
}


def _expand_query(query: str) -> str:
    """
    Expand a Hindi query with related terms for better recall.
    Appends synonyms to the original query so both BM25 and dense
    retrieval can find more relevant passages.
    """
    words = query.split()
    expansions = []
    for word in words:
        clean = word.strip("।?!,.")
        if clean in _HINDI_SYNONYMS:
            expansions.extend(_HINDI_SYNONYMS[clean])
    if expansions:
        # Append expansions separated by spaces — BM25 benefits from
        # extra keywords, and the embedding model handles multi-word
        # queries well.
        expanded = query + " " + " ".join(expansions)
        logger.info("[expand] '%s' -> '%s'", query, expanded)
        return expanded
    return query


def normalize_query(query: str) -> tuple[str, StageTiming]:
    """
    Clean, normalize, and expand the transcribed query.
    Returns (expanded_query, timing).
    """
    start_ms = time.perf_counter() * 1000

    normalized = query.strip()
    for pattern, repl in _NORMALIZE_PATTERNS:
        normalized = re.sub(pattern, repl, normalized)
    normalized = normalized.strip()

    # Expand with synonyms for better recall
    expanded = _expand_query(normalized)

    _emit_timing("normalize_query", start_ms)
    return expanded, _telemetry_store[-1]


# ── Retrieve ─────────────────────────────────────────────────────────

def retrieve(
    query: str,
    retriever,
    top_k: int = 10,
    strategy: Optional[str] = None,
    timeout_ms: float = 100,
) -> tuple[RetrievalResult, StageTiming]:
    """
    Hybrid retrieval via the retriever (FAISS + BM25).
    Returns (result, timing).
    """
    start_ms = time.perf_counter() * 1000
    result = retriever.retrieve(query, top_k=top_k, strategy=strategy)
    _emit_timing("retrieve", start_ms)
    return result, _telemetry_store[-1]


# ── Rerank ──────────────────────────────────────────────────────────

def rerank(
    retrieval_result: RetrievalResult,
    query: str,
    top_k: int = 5,
) -> tuple[RetrievalResult, StageTiming]:
    """
    Re-rank passages using combined scores with a small position boost
    for is_selected passages and fused-score weighting.

    This is a lightweight heuristic reranker (no cross-encoder) to stay
    within the 50ms budget per PRD §6.
    """
    start_ms = time.perf_counter() * 1000

    passages = retrieval_result.passages
    if len(passages) <= top_k:
        _emit_timing("rerank", start_ms)
        return retrieval_result, _telemetry_store[-1]

    # Combine dense + sparse + fused scores for reranking
    scored = []
    for i, p in enumerate(passages):
        dense = retrieval_result.dense_scores[i] if i < len(retrieval_result.dense_scores) else 0
        sparse = retrieval_result.sparse_scores[i] if i < len(retrieval_result.sparse_scores) else 0
        fused = retrieval_result.fused_scores[i] if i < len(retrieval_result.fused_scores) else 0

        # Weighted combination: fused is primary, with dense/sparse as tiebreakers
        final_score = fused * 0.6 + (dense / max(dense, 0.001)) * 0.2 + (sparse / max(sparse, 0.001)) * 0.2
        scored.append((final_score, i, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    reranked_passages = [s[2] for s in scored[:top_k]]
    reranked_dense = [retrieval_result.dense_scores[s[1]] for s in scored[:top_k]]
    reranked_sparse = [retrieval_result.sparse_scores[s[1]] for s in scored[:top_k]]
    reranked_fused = [retrieval_result.fused_scores[s[1]] for s in scored[:top_k]]

    for p, score in zip(reranked_passages, [s[0] for s in scored[:top_k]]):
        p.score = score

    result = RetrievalResult(
        passages=reranked_passages,
        dense_scores=reranked_dense,
        sparse_scores=reranked_sparse,
        fused_scores=reranked_fused,
    )

    _emit_timing("rerank", start_ms)
    return result, _telemetry_store[-1]


# ── Guardrail check ──────────────────────────────────────────────────

# Unsafe content patterns (basic keyword list)
_UNSAFE_PATTERNS = re.compile(
    r'(?i)(bomb|kill|murder|terrorist|hack.*account|steal.*password|'
    r'self.?harm|suicide|weapon|drug.*manufacture)',
)

# Off-topic detection: require BOTH sparse AND dense to be very low.
# Only refuse truly unrecognised queries (e.g. pure English against Hindi corpus).
_OFF_TOPIC_DENSE_MIN = 0.15   # much lower — only refuse if dense match is very weak
_OFF_TOPIC_SPARSE_MIN = 0.0   # disabled — BM25 sparse scores are unnormalized and unreliable


def guardrail_check(
    query: str,
    retrieval_result: RetrievalResult,
) -> tuple[GuardrailResult, StageTiming]:
    """
    Rule-based guardrail checks (no LLM call):
    1. Unsafe input detection (keyword match)
    2. Off-topic detection (retrieval score threshold)

    Returns (guardrail_result, timing).
    """
    start_ms = time.perf_counter() * 1000

    # ── Unsafe check ───────────────────────────────────────────
    if _UNSAFE_PATTERNS.search(query):
        _emit_timing("guardrail_check", start_ms, "refused_unsafe")
        return (
            GuardrailResult(passed=False, refused=True, refusal_reason=RefusalReason.UNSAFE, confidence=0.0),
            _telemetry_store[-1],
        )

    # ── Off-topic check ────────────────────────────────────────
    # Only refuse if dense similarity is very low (truly irrelevant query).
    # Refuse only if BOTH dense AND BM25 are weak.
    # BM25 > 5.0 means a meaningful keyword match exists.
    top_dense = max(retrieval_result.dense_scores) if retrieval_result.dense_scores else 0.0
    top_sparse = max(retrieval_result.sparse_scores) if retrieval_result.sparse_scores else 0.0
    top_fused = retrieval_result.fused_scores[0] if retrieval_result.fused_scores else 0.0
    sparse_is_strong = top_sparse > 5.0
    dense_is_weak = top_dense < _OFF_TOPIC_DENSE_MIN
    is_off_topic = dense_is_weak and not sparse_is_strong

    logger.info("[guardrail] scores: dense=%.3f sparse=%.3f fused=%.3f off_topic=%s",
                top_dense, top_sparse, top_fused, is_off_topic)

    if is_off_topic:
        _emit_timing("guardrail_check", start_ms, "refused_off_topic")
        return (
            GuardrailResult(
                passed=False,
                refused=True,
                refusal_reason=RefusalReason.OFF_TOPIC,
                top_score=top_fused,
                confidence=0.0,
            ),
            _telemetry_store[-1],
        )

    # ── Confidence from retrieval score ──────────────────────
    confidence = min(1.0, max(0.1, top_dense))

    _emit_timing("guardrail_check", start_ms)
    return (
        GuardrailResult(passed=True, refused=False, top_score=top_fused, confidence=confidence),
        _telemetry_store[-1],
    )


# ── Groundedness check ──────────────────────────────────────────────

_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
    "है", "हैं", "था", "थे", "की", "के", "का", "को", "में", "से", "और", "यह",
    "वह", "क्या", "कहाँ", "किस", "कौन", "कब", "क्यों", "कैसे", "एक", "पर",
    "स्थित", "स्थान", "जगह", "रहता", "रही", "रहे", "होना",
}


def _content_terms(text: str) -> set[str]:
    """Return normalized non-stopword terms for lightweight relevance checks."""
    tokens = re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", text.lower())
    return {token for token in tokens if token not in _CONTENT_STOPWORDS and len(token) > 1}


def query_answer_relevance(
    query: str,
    answer: str,
    min_shared_terms: int = 2,
    alternate_queries: Optional[list[str]] = None,
) -> bool:
    """Require the answer to address meaningful terms in either query form."""
    answer_terms = _content_terms(answer)
    if not answer_terms:
        return False
    query_variants = [query, _expand_query(query)] + list(alternate_queries or [])
    for query_variant in query_variants:
        query_terms = _content_terms(query_variant)
        if not query_terms:
            continue
        shared = query_terms & answer_terms
        required = 1 if len(query_terms) == 1 else min_shared_terms
        if len(shared) >= required:
            return True
    return False


def answer_is_complete(answer: str) -> bool:
    """Reject obvious question echoes and truncated English generations."""
    if not answer or len(_content_terms(answer)) < 2:
        return False
    stripped = answer.strip()
    if "?" in stripped:
        return False
    return not bool(re.search(
        r"\b(?:is|are|was|were|the|a|an|its|it's|located|where|what|who|when|why|how)\s*[.!…]*$",
        stripped.lower(),
    ))


def _ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """Jaccard overlap of character n-grams between two strings."""
    def ngrams(s: str) -> set:
        s = re.sub(r'\s+', ' ', s.strip().lower())
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()
    a, b = ngrams(text_a), ngrams(text_b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def groundedness_check(
    answer: str,
    retrieval_result: RetrievalResult,
    threshold: float = 0.08,
) -> bool:
    """Check both textual overlap and meaningful evidence-term overlap."""
    if not answer or not retrieval_result.passages or not answer_is_complete(answer):
        return False
    answer_terms = _content_terms(answer)
    for passage in retrieval_result.passages[:3]:
        ngram_score = _ngram_overlap(answer, passage.text)
        shared_terms = answer_terms & _content_terms(passage.text)
        if ngram_score >= threshold and len(shared_terms) >= 2:
            logger.debug(
                "[groundedness] ngram=%.3f shared_terms=%d", ngram_score, len(shared_terms)
            )
            return True
    return False


# ── Generate ───────────────────────────────────────────────────────

LLM_BASE_URL_DEFAULT = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL_DEFAULT = "allam-2-7b"  # Fastest Groq model (~400ms)
FAST_EXTRACTIVE_CONFIDENCE = float(os.environ.get("FAST_EXTRACTIVE_CONFIDENCE", "0.45"))


def _get_llm_config():
    """Read LLM config lazily so .env is loaded first."""
    return (
        os.environ.get("LLM_API_KEY", ""),
        os.environ.get("LLM_BASE_URL", LLM_BASE_URL_DEFAULT),
        os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT),
    )


def _extract_best_sentences(query: str, text: str, max_sentences: int = 2) -> str:
    """
    Extract the most relevant sentences from a passage for a given query.
    Scores each sentence by keyword overlap + position (earlier = better).
    Returns the top sentences joined together.
    """
    # Split into sentences (Hindi: ।  or  ./!/?  as delimiters)
    sentences = re.split(r'[।!?\.]+\s*', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return text[:300]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Score each sentence
    query_words = set(re.findall(r'[\u0900-\u097F\w]+', query.lower()))
    scored = []
    for i, sent in enumerate(sentences):
        sent_words = set(re.findall(r'[\u0900-\u097F\w]+', sent.lower()))
        # Keyword overlap score
        overlap = len(query_words & sent_words) / max(len(query_words), 1)
        # Position bonus (earlier sentences are usually more important)
        position_bonus = 1.0 / (1 + i * 0.3)
        # Length penalty (very short sentences are less informative)
        length_bonus = min(1.0, len(sent) / 50)
        score = overlap * 2.0 + position_bonus + length_bonus
        scored.append((score, i, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])  # Re-sort by position
    return " ".join(s[2] for s in top)


def _extractive_result(
    query: str,
    passages: list,
    confidence: float,
) -> GenerationResult:
    """Build an extractive answer only when the evidence matches the query."""
    all_texts = " ".join(p.text for p in passages[:3])
    answer = _extract_best_sentences(query, all_texts, max_sentences=2)
    if not query_answer_relevance(query, answer):
        logger.warning("[groundedness] retrieved passages do not answer the query")
        return GenerationResult(answer="", citations=[], confidence=0.0, extractive=True)
    citations = list(dict.fromkeys(p.passage_id for p in passages[:3]))
    return GenerationResult(
        answer=answer,
        citations=citations,
        confidence=confidence,
        extractive=True,
    )


def generate(
    query: str,
    retrieval_result: RetrievalResult,
    guardrail_result: GuardrailResult,
    timeout_ms: float = 100,
) -> tuple[GenerationResult, StageTiming]:
    """
    Generate an answer from retrieved context.

    Strategy:
      1. If LLM API key is available → use LLM for best answers
      2. Otherwise → extract the most relevant sentences (A improvement)

    Returns (generation_result, timing).
    """
    start_ms = time.perf_counter() * 1000

    passages = retrieval_result.passages
    if not passages:
        _emit_timing("generate", start_ms, "error")
        return (
            GenerationResult(answer="", citations=[], confidence=0.0),
            _telemetry_store[-1],
        )

    top_passage = passages[0]

    # Fast path: when retrieval is already strong and the passage contains a
    # relevant extractive answer, avoid a network LLM round-trip entirely.
    # The LLM remains available for lower-confidence or synthesis queries.
    fast_result = _extractive_result(
        query,
        passages,
        guardrail_result.confidence * 0.7,
    )
    if not fast_result.answer:
        # No relevant evidence means an LLM call cannot produce a grounded
        # answer. Refuse immediately instead of paying network latency.
        _emit_timing("generate", start_ms, "no_evidence")
        return fast_result, _telemetry_store[-1]

    force_llm = os.environ.get("FORCE_LLM_GENERATION", "").lower() in {"1", "true", "yes"}
    if not force_llm and guardrail_result.confidence >= FAST_EXTRACTIVE_CONFIDENCE:
        _emit_timing("generate", start_ms, "extractive_fast_path")
        return fast_result, _telemetry_store[-1]

    # ── LLM generation ──────────────────────────────────
    llm_api_key, llm_base_url, llm_model = _get_llm_config()
    if llm_api_key:
        try:
            # Keep the request small enough for the 200ms budget.
            context = "\n\n".join(
                f"[Passage {p.passage_id}]: {p.text[:900]}" for p in passages[:2]
            )
            logger.info("[generate] context passages: %s", [p.passage_id for p in passages[:3]])
            logger.info("[generate] context preview: %s", context[:200])

            prompt = (
                "Answer using ONLY the provided passages. "
                "Rules:\n"
                "1. Use the SAME language as the question\n"
                "2. Return exactly one concise sentence\n"
                "3. Do NOT repeat the question or invent information\n"
                "4. Reference a passage ID like [p_XXXXXX]\n\n"
                f"Passages:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer (one sentence):"
            )

            with httpx.Client(timeout=max(timeout_ms / 1000, 0.05)) as client:
                resp = client.post(
                    llm_base_url,
                    headers={"Authorization": f"Bearer {llm_api_key}"},
                    json={
                        "model": llm_model,
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 48,
                        "temperature": 0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                if not answer_is_complete(answer):
                    raise ValueError("LLM returned a question echo or incomplete answer")

            # Extract citations from answer
            citations = re.findall(r'\[?(p_\d+)\]?', answer)
            citations = list(set(citations)) or [top_passage.passage_id]

            _emit_timing("generate", start_ms)
            return (
                GenerationResult(
                    answer=answer,
                    citations=citations,
                    confidence=guardrail_result.confidence * 0.9,
                    extractive=False,
                ),
                _telemetry_store[-1],
            )
        except Exception as e:
            logger.warning("LLM generation failed: %s — falling back to extractive", e)
            _emit_timing("generate", start_ms, "error_fallback")

    # ── Smart extractive answer (no LLM) ──────────────────
    result = _extractive_result(
        query,
        passages,
        guardrail_result.confidence * 0.7,
    )

    _emit_timing("generate", start_ms)
    return result, _telemetry_store[-1]


# ── Postprocess ─────────────────────────────────────────────────────

def postprocess(
    generation_result: GenerationResult,
    guardrail_result: GuardrailResult,
    session_id: Optional[str] = None,
    stt_timing: Optional[float] = None,
) -> tuple[PipelineResponse, StageTiming]:
    """
    Assemble the final PipelineResponse with latency breakdown.

    Returns (pipeline_response, timing).
    """
    start_ms = time.perf_counter() * 1000

    # Compute latency breakdown from telemetry
    from voicerag.schemas.contracts import LatencyBreakdown

    breakdown_dict = {}
    post_stt_start = None
    for t in _telemetry_store:
        if t.stage_name == "transcribe":
            breakdown_dict["stt_ms"] = t.duration_ms
        else:
            if post_stt_start is None:
                post_stt_start = t.start_ms
            breakdown_dict[f"{t.stage_name}_ms"] = t.duration_ms

    if post_stt_start:
        breakdown_dict["total_post_stt_ms"] = time.perf_counter() * 1000 - post_stt_start

    breakdown = LatencyBreakdown(
        stt_ms=breakdown_dict.get("stt_ms"),
        normalize_ms=breakdown_dict.get("normalize_query_ms"),
        retrieve_ms=breakdown_dict.get("retrieve_ms"),
        rerank_ms=breakdown_dict.get("rerank_ms"),
        guardrail_ms=breakdown_dict.get("guardrail_check_ms"),
        generate_ms=breakdown_dict.get("generate_ms"),
        postprocess_ms=None,  # Filled after this completes
        total_post_stt_ms=breakdown_dict.get("total_post_stt_ms"),
        timings=list(_telemetry_store),
    )

    response = PipelineResponse(
        answer=generation_result.answer,
        citations=generation_result.citations,
        confidence=generation_result.confidence,
        refused=guardrail_result.refused,
        refusal_reason=guardrail_result.refusal_reason,
        latency_breakdown=breakdown,
        session_id=session_id,
    )

    _emit_timing("postprocess", start_ms)
    response.latency_breakdown.postprocess_ms = _telemetry_store[-1].duration_ms

    return response, _telemetry_store[-1]
