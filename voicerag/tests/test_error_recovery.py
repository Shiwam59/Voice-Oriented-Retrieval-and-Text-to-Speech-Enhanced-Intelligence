"""
Error-recovery / retry / timeout tests — Tasks 2.3, 2.4, 2.5 acceptance.

Runs entirely on fakes — no FAISS index or network required.

Usage:
    python -m tests.test_error_recovery
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from voicerag.schemas.contracts import (
    ChunkStrategy,
    Language,
    Passage,
    PipelineRequest,
    RefusalReason,
    RetrievalResult,
)
from voicerag.harness import orchestrator
from voicerag.harness.orchestrator import StageTimeoutError, _retry_with_backoff, run_pipeline


# ── Fakes ───────────────────────────────────────────────────────────

class FakeRetriever:
    """Returns a canned RetrievalResult without touching FAISS."""

    def __init__(self, empty: bool = False, delay_s: float = 0.0):
        self.empty = empty
        self.delay_s = delay_s

    def retrieve(self, query, top_k=10, strategy=None):
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.empty:
            return RetrievalResult(passages=[], dense_scores=[], sparse_scores=[], fused_scores=[])
        p = Passage(
            passage_id="p_000001",
            text="निगम एक कंपनी या लोगों का समूह होता है।",
            chunk_strategy=ChunkStrategy.PASSAGE_NATIVE,
        )
        # rank-1 in both dense and sparse ⇒ fused RRF ≈ 2/61 ≈ 0.0328
        return RetrievalResult(passages=[p], dense_scores=[0.9], sparse_scores=[8.0], fused_scores=[0.0328])

    def close(self):
        pass


# ── Task 2.3: retries with backoff ─────────────────────────────────

def test_transient_failure_recovers():
    """A stage failing once then succeeding must recover within the cap."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient blip")
        return "ok"

    result = _retry_with_backoff(flaky, max_retries=2)
    assert result == "ok"
    assert calls["n"] == 2
    print("  OK: transient failure recovered on retry 2 (cap respected)")


def test_retry_exhaustion_never_hangs():
    """A permanently failing stage must raise after the cap, not hang."""
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise ConnectionError("down")

    try:
        _retry_with_backoff(always_fails, max_retries=2)
        raise AssertionError("should have raised")
    except ConnectionError:
        pass
    assert calls["n"] == 3  # 1 initial + 2 retries
    print("  OK: exhaustion raised after 3 attempts (never hangs)")


# ── Task 2.4: per-stage timeouts ────────────────────────────────────

def test_stage_timeout_raises():
    """A stage blowing its deadline raises StageTimeoutError."""
    def slow():
        time.sleep(0.3)
        return "late"

    try:
        _retry_with_backoff(slow, timeout_ms=50, max_retries=0)
        raise AssertionError("should have raised StageTimeoutError")
    except StageTimeoutError:
        pass
    print("  OK: slow stage (300ms vs 50ms budget) raised StageTimeoutError")


def test_generate_timeout_falls_back_extractive():
    """On generate failure/timeout the pipeline returns a cited extractive answer."""
    original = orchestrator.generate

    def broken_generate(*a, **kw):
        raise RuntimeError("LLM down")

    orchestrator.generate = broken_generate
    try:
        resp = run_pipeline(
            PipelineRequest(transcript="कॉर्पोरेशन क्या है?", language=Language.HI),
            FakeRetriever(),
        )
    finally:
        orchestrator.generate = original

    assert not resp.refused
    assert resp.answer, "no extractive fallback answer"
    assert resp.citations, "fallback answer missing citation"
    print(f"  OK: generate failure -> extractive fallback (citations={resp.citations})")


# ── Task 2.5: structured error recovery ─────────────────────────────

def test_stt_failure_returns_rerecord_error():
    """STT failure must surface a re-record prompt, never a hang/crash."""
    original = orchestrator.transcribe

    def broken_stt(*a, **kw):
        raise TimeoutError("STT timeout")

    orchestrator.transcribe = broken_stt
    try:
        resp = run_pipeline(
            PipelineRequest(audio=b"fake-bytes", language=Language.HI),
            FakeRetriever(),
        )
    finally:
        orchestrator.transcribe = original

    assert resp.error and "re-record" in resp.error.lower(), f"unexpected error: {resp.error}"
    assert not resp.answer, "STT failure must not produce an answer"
    print("  OK: STT failure -> re-record prompt (no crash, no hallucinated answer)")


def test_empty_retrieval_refuses():
    """Empty retrieval must take the refusal path, not hallucinate."""
    resp = run_pipeline(
        PipelineRequest(transcript="कुछ भी", language=Language.HI),
        FakeRetriever(empty=True),
    )
    assert resp.refused, "empty retrieval was not refused"
    assert resp.refusal_reason == RefusalReason.OFF_TOPIC
    assert not resp.citations
    print("  OK: empty retrieval -> refusal (off_topic), no citations")


def test_malformed_llm_output_falls_back():
    """Malformed generation output must fall back to extractive, cited answer."""
    original = orchestrator.generate

    def malformed_generate(*a, **kw):
        from voicerag.harness import stages
        # Simulate malformed output followed by persistent failure:
        raise ValueError("malformed JSON from LLM")

    orchestrator.generate = malformed_generate
    try:
        resp = run_pipeline(
            PipelineRequest(transcript="कॉर्पोरेशन क्या है?", language=Language.HI),
            FakeRetriever(),
        )
    finally:
        orchestrator.generate = original

    assert resp.answer and resp.citations, "malformed LLM output lost the fallback answer"
    print("  OK: malformed LLM output -> schema-safe extractive fallback with citation")


# ── Runner ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Error recovery / retry / timeout tests (Tasks 2.3-2.5)")
    print("=" * 60)
    tests = [
        test_transient_failure_recovers,
        test_retry_exhaustion_never_hangs,
        test_stage_timeout_raises,
        test_generate_timeout_falls_back_extractive,
        test_stt_failure_returns_rerecord_error,
        test_empty_retrieval_refuses,
        test_malformed_llm_output_falls_back,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failures += 1
    print("=" * 60)
    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
