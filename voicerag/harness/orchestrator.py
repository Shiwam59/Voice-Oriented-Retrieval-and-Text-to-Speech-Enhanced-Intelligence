"""
Harness orchestrator — Tasks 2.3 / 2.4 / 2.5

Orchestrates the full pipeline:
    transcribe → normalize → retrieve → rerank → guardrail → generate → postprocess

With:
    - Capped retries with backoff on transient failures (2.3)
    - Per-stage timeouts — short-circuit to extractive on exceed (2.4)
    - Structured error recovery (2.5):
        STT failure → error with re-record prompt
        Empty retrieval → refusal path
        Malformed LLM output → schema validation + repair → extractive fallback
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Optional

from voicerag.schemas.contracts import (
    GenerationResult,
    GuardrailResult,
    PipelineRequest,
    PipelineResponse,
    RefusalReason,
    RetrievalResult,
)
from voicerag.harness.stages import (
    clear_telemetry,
    detect_language,
    generate,
    get_telemetry,
    groundedness_check,
    guardrail_check,
    normalize_query,
    postprocess,
    query_answer_relevance,
    rerank,
    retrieve,
    transcribe,
    translate,
    _extractive_result,
)

logger = logging.getLogger(__name__)

# ── Query cache for repeated questions ────────────────────────────
import hashlib
from collections import OrderedDict

_query_cache: OrderedDict[str, PipelineResponse] = OrderedDict()
_CACHE_MAX_SIZE = 100


def _cache_key(transcript: str, language: str) -> str:
    """Generate a cache key from transcript + language."""
    return hashlib.md5(f"{transcript}:{language}".encode()).hexdigest()


def _get_cached(transcript: str, language: str) -> Optional[PipelineResponse]:
    """Look up cached response."""
    key = _cache_key(transcript, language)
    return _query_cache.get(key)


def _set_cached(transcript: str, language: str, response: PipelineResponse) -> None:
    """Cache a response (LRU eviction)."""
    key = _cache_key(transcript, language)
    _query_cache[key] = response
    _query_cache.move_to_end(key)
    while len(_query_cache) > _CACHE_MAX_SIZE:
        _query_cache.popitem(last=False)


# ── Retry / timeout config ──────────────────────────────────────────

MAX_RETRIES = 2
BACKOFF_BASE = 0.05  # 50ms base backoff

STAGE_TIMEOUTS_MS = {
    "transcribe": 10000,
    "normalize_query": 25,
    "retrieve": 60,
    "rerank": 50,
    "guardrail_check": 15,
    # Keep the LLM on the 200ms post-STT budget. On timeout, use the
    # grounded extractive fallback instead of retrying on the hot path.
    "generate": 120,
    "postprocess": 10,
}


class StageTimeoutError(Exception):
    """A stage exceeded its hard deadline (2x slack for cold-start jitter)."""


def _retry_with_backoff(fn, *args, max_retries: int = MAX_RETRIES, timeout_ms: float = None, **kwargs):
    """
    Execute a stage function with capped retries and deadline enforcement.

    - Only passes timeout_ms to functions that accept it (httpx-backed stages).
    - After the call, if elapsed time blew the deadline (>2x slack), raises
      StageTimeoutError so the orchestrator's fallback paths fire (Task 2.4).
    - On retry exhaustion, raises the last error.
    """
    accepts_timeout = "timeout_ms" in inspect.signature(fn).parameters
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            call_kwargs = dict(kwargs)
            if timeout_ms is not None and accepts_timeout:
                call_kwargs["timeout_ms"] = timeout_ms
            start = time.perf_counter()
            result = fn(*args, **call_kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if timeout_ms is not None and elapsed_ms > timeout_ms * 2:
                raise StageTimeoutError(
                    f"{fn.__name__} blew deadline: {elapsed_ms:.0f}ms > {timeout_ms:.0f}ms budget"
                )
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                backoff = BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "[retry] %s attempt %d/%d failed: %s — retrying in %.0f ms",
                    fn.__name__, attempt + 1, max_retries + 1, e, backoff * 1000,
                )
                time.sleep(backoff)
            else:
                logger.error("[retry] %s exhausted retries — last error: %s", fn.__name__, e)
    raise last_error


def run_pipeline(
    request: PipelineRequest,
    retriever,
    top_k: int = 5,
    generate_timeout_ms: Optional[float] = None,
) -> PipelineResponse:
    """
    Execute the full VoiceRAG pipeline.

    Args:
        request: Typed pipeline input (audio or transcript)
        retriever: HybridRetriever instance
        top_k: Number of passages to retrieve

    Returns:
        PipelineResponse with answer, citations, confidence, latency breakdown
    """
    clear_telemetry()

    # ── Step 0: Detect language ────────────────────────────────
    transcript = request.transcript or ""
    query_lang = detect_language(transcript) if transcript else "hi"

    # ── Cache check (skip full pipeline for repeated queries) ──
    if transcript and not request.audio:
        cached = _get_cached(transcript, query_lang)
        if cached:
            logger.info("[cache] HIT for '%s' (lang=%s)", transcript[:50], query_lang)
            return cached
        logger.info("[cache] MISS for '%s' (lang=%s)", transcript[:50], query_lang)

    # ── Step 1: Transcribe (if audio provided) ──────────────────
    if request.audio and not transcript:
        try:
            transcript, _ = _retry_with_backoff(
                transcribe, request.audio, request.language.value,
                timeout_ms=STAGE_TIMEOUTS_MS["transcribe"],
                content_type=request.audio_content_type or "audio/wav",
            )
            # Re-detect language from the transcript
            query_lang = detect_language(transcript)
        except Exception as e:
            logger.error("STT failed after retries: %s", e)
            return PipelineResponse(
                error=f"Speech-to-text failed: {e}. Please try re-recording.",
                refused=False,
                session_id=request.session_id,
            )

    if not transcript:
        return PipelineResponse(
            error="No audio or transcript provided.",
            refused=False,
            session_id=request.session_id,
        )

    # ── Step 1b: English → Hindi translation for search ──────────
    # Route to correct index based on language:
    # - Hindi queries → search Hindi index directly
    # - English queries → translate to Hindi → search Hindi index
    # - Urdu queries → search Urdu index directly (if available)
    #                 or translate to Hindi → search Hindi index
    search_query = transcript
    search_lang = query_lang

    if query_lang == "ur":
        # Urdu → translate to Hindi, search Hindi index
        try:
            search_query, _ = _retry_with_backoff(
                translate, transcript, "ur", "hi",
                timeout_ms=5000,
            )
            logger.info("[bilingual] UR->HI: '%s' -> '%s'", transcript[:50], search_query[:50])
        except Exception as e:
            logger.warning("Translation UR->HI failed: %s — searching with Urdu", e)
            search_query = transcript
    elif query_lang == "en":
        try:
            search_query, _ = _retry_with_backoff(
                translate, transcript, "en", "hi",
                timeout_ms=5000,
            )
            logger.info("[bilingual] EN->HI: '%s' -> '%s'", transcript[:50], search_query[:50])
        except Exception as e:
            logger.warning("Translation EN->HI failed: %s — searching with English", e)
            search_query = transcript

    # ── Step 2: Normalize query ────────────────────────────────
    try:
        query, _ = _retry_with_backoff(
            normalize_query, search_query,
            timeout_ms=STAGE_TIMEOUTS_MS["normalize_query"],
        )
    except Exception:
        query = search_query.strip()  # Best-effort fallback

    # ── Step 3: Retrieve ────────────────────────────────────────
    try:
        retrieval_result, _ = _retry_with_backoff(
            retrieve, query, retriever, top_k=top_k,
            timeout_ms=STAGE_TIMEOUTS_MS["retrieve"],
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return PipelineResponse(
            answer="",
            confidence=0.0,
            refused=True,
            refusal_reason=RefusalReason.OFF_TOPIC,
            error=f"Retrieval error: {e}",
            session_id=request.session_id,
        )

    # Empty retrieval → refusal
    if not retrieval_result.passages:
        return PipelineResponse(
            answer="",
            confidence=0.0,
            refused=True,
            refusal_reason=RefusalReason.OFF_TOPIC,
            session_id=request.session_id,
        )

    # ── Step 4: Rerank ──────────────────────────────────────────
    try:
        retrieval_result, _ = _retry_with_backoff(
            rerank, retrieval_result, query, top_k=top_k,
            timeout_ms=STAGE_TIMEOUTS_MS["rerank"],
        )
    except Exception:
        pass  # Use pre-rerank results

    # ── Step 5: Guardrail check ─────────────────────────────────
    # Run unsafe check on ORIGINAL transcript (not translated query)
    # so English keywords like "bomb" are still caught.
    try:
        guardrail_result, _ = _retry_with_backoff(
            guardrail_check, transcript, retrieval_result,
            timeout_ms=STAGE_TIMEOUTS_MS["guardrail_check"],
        )
    except Exception as e:
        logger.warning("Guardrail check failed: %s — passing through", e)
        guardrail_result = GuardrailResult(passed=True, refused=False, confidence=0.5)

    if guardrail_result.refused:
        return PipelineResponse(
            answer=_refusal_message(guardrail_result.refusal_reason, query_lang),
            confidence=0.0,
            refused=True,
            refusal_reason=guardrail_result.refusal_reason,
            session_id=request.session_id,
        )

    # ── Step 6: Generate ─────────────────────────────────────
    try:
        generation_result, _ = _retry_with_backoff(
            generate, query, retrieval_result, guardrail_result,
            max_retries=0,
            timeout_ms=(
                generate_timeout_ms
                if generate_timeout_ms is not None
                else STAGE_TIMEOUTS_MS["generate"]
            ),
        )
    except Exception:
        # Extractive fallback only when the retrieved evidence matches the query.
        generation_result = _extractive_result(
            query,
            retrieval_result.passages,
            guardrail_result.confidence * 0.3,
        )

    # ── Step 6b: Groundedness and query-relevance gates (4.3) ──
    # A fluent answer about the wrong subject must not pass merely because
    # it overlaps with an unrelated retrieved passage.
    if generation_result.answer and not query_answer_relevance(
        query,
        generation_result.answer,
        alternate_queries=[transcript] if query_lang == "en" else None,
    ):
        logger.warning("[relevance] answer does not address the query — falling back to extractive")
        generation_result = _extractive_result(
            query,
            retrieval_result.passages,
            guardrail_result.confidence * 0.5,
        )
    elif generation_result.answer and not generation_result.extractive:
        if not groundedness_check(generation_result.answer, retrieval_result):
            logger.warning("[groundedness] answer ungrounded — falling back to extractive")
            generation_result = _extractive_result(
                query,
                retrieval_result.passages,
                guardrail_result.confidence * 0.5,
            )

    # If neither generation nor extraction can bind a relevant answer,
    # refuse instead of returning an incomplete or unrelated passage.
    if not generation_result.answer:
        return PipelineResponse(
            answer=_refusal_message(RefusalReason.OFF_TOPIC, query_lang),
            confidence=0.0,
            refused=True,
            refusal_reason=RefusalReason.OFF_TOPIC,
            session_id=request.session_id,
        )

    # ── Step 6c: Citation binding (4.4) ───────────────────────
    # Every non-refused answer MUST carry at least one passage_id citation.
    if not generation_result.citations:
        logger.warning("[citation] no citations — binding to top passage")
        top_p = retrieval_result.passages[0]
        generation_result = GenerationResult(
            answer=generation_result.answer,
            citations=[top_p.passage_id],
            confidence=generation_result.confidence,
            extractive=generation_result.extractive,
        )

    # ── Step 7: Postprocess ────────────────────────────────────
    response, _ = postprocess(generation_result, guardrail_result, session_id=request.session_id)

    # Attach the transcript so the UI can display what was heard
    response.transcript = transcript

    # ── Step 8: Translate answer back to user's language ─────────
    if response.answer and not response.refused:
        if query_lang == "en":
            # English query → translate Hindi answer back to English
            pass  # handled below
        elif query_lang == "ur" and search_lang != "ur":
            # Urdu query → translate Hindi answer back to Urdu
            pass  # handled below
        else:
            # Hindi query or Urdu index used directly — no translation needed
            pass

    if query_lang == "en" and response.answer and not response.refused:
        try:
            translated_answer, _ = _retry_with_backoff(
                translate, response.answer, "hi", "en",
                timeout_ms=5000,
            )
            response.answer = translated_answer
            logger.info("[bilingual] HI->EN answer: '%s'", translated_answer[:80])
        except Exception as e:
            logger.warning("Translation HI->EN failed: %s — returning Hindi answer", e)
    elif query_lang == "ur" and search_lang != "ur" and response.answer and not response.refused:
        try:
            translated_answer, _ = _retry_with_backoff(
                translate, response.answer, "hi", "ur",
                timeout_ms=5000,
            )
            response.answer = translated_answer
            logger.info("[bilingual] HI->UR answer: '%s'", translated_answer[:80])
        except Exception as e:
            logger.warning("Translation HI->UR failed: %s — returning Hindi answer", e)

    # ── Cache the response for repeated queries ─────────────
    if transcript and not request.audio and not response.error:
        _set_cached(transcript, query_lang, response)
        logger.info("[cache] STORED '%s' (cache_size=%d)", transcript[:50], len(_query_cache))

    return response


def _refusal_message(reason: Optional[RefusalReason], language: str = "hi") -> str:
    """Return a user-friendly refusal message in the query language."""
    if language == "en":
        messages = {
            RefusalReason.OFF_TOPIC: "I could not find information about this topic in the dataset. Please ask a different question.",
            RefusalReason.UNSAFE: "This query is not safe to answer.",
            RefusalReason.UNGROUNDED: "I could not find enough information in the dataset to answer this question.",
            RefusalReason.GENERATION_UNAVAILABLE: "I could not generate an answer. Please try again.",
        }
        return messages.get(reason, "The request could not be completed. Please try again.")

    messages = {
        RefusalReason.OFF_TOPIC: "मुझे इस डेटासेट में इस विषय के बारे में जानकारी नहीं है। कृपया अलग सवाल पूछें।",
        RefusalReason.UNSAFE: "यह क्वेरी सुरक्षित नहीं है और मैं इसका उत्तर नहीं दे सकता।",
        RefusalReason.UNGROUNDED: "मुझे इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।",
        RefusalReason.GENERATION_UNAVAILABLE: "उत्तर उत्पन्न करने में समस्या हुई। कृपया पुनः प्रयास करें।",
    }
    return messages.get(reason, "अनुरोध पूरा नहीं हो सका। कृपया पुनः प्रयास करें।")
