"""
Shared typed request/response contracts (source of truth).
All stage boundaries in the harness must use these schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────

class Language(str, Enum):
    HI = "hi"
    EN = "en"
    UR = "ur"
    AUTO = "auto"


class ChunkStrategy(str, Enum):
    PASSAGE_NATIVE = "passage_native"
    SENTENCE_WINDOW = "sentence_window"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"


class RefusalReason(str, Enum):
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"
    UNGROUNDED = "ungrounded"
    GENERATION_UNAVAILABLE = "generation_unavailable"


# ── Pipeline input ─────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    """Input to the harness — either audio bytes or a pre-transcribed query."""
    audio: Optional[bytes] = Field(None, description="Raw audio bytes")
    audio_content_type: Optional[str] = Field(None, description="MIME type of audio (e.g. audio/wav, audio/webm)")
    transcript: Optional[str] = Field(None, description="Pre-transcribed query text")
    language: Language = Field(Language.HI, description="Query language")
    session_id: Optional[str] = Field(None, description="Correlation / session ID")

    model_config = {"frozen": True}


# ── Internal stage I/O ─────────────────────────────────────────────

class Passage(BaseModel):
    """A single retrieved passage / chunk."""
    passage_id: str
    text: str
    query_id: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    language: Optional[str] = None
    char_len: Optional[int] = None
    chunk_strategy: Optional[ChunkStrategy] = None
    # has_answer_overlap is offline-eval only — never included at inference
    score: Optional[float] = Field(None, description="Retrieval score")


class RetrievalResult(BaseModel):
    passages: list[Passage]
    dense_scores: list[float] = Field(default_factory=list)
    sparse_scores: list[float] = Field(default_factory=list)
    fused_scores: list[float] = Field(default_factory=list)


class GuardrailResult(BaseModel):
    passed: bool
    refused: bool = False
    refusal_reason: Optional[RefusalReason] = None
    top_score: Optional[float] = None
    confidence: float = 0.0


class GenerationResult(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)  # passage_ids
    confidence: float = 0.0
    extractive: bool = False


class StageTiming(BaseModel):
    """Latency for a single stage (milliseconds)."""
    stage_name: str
    start_ms: float
    end_ms: float
    duration_ms: float
    status: str = "success"  # success | timeout | error


# ── Pipeline output ────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    stt_ms: Optional[float] = None
    normalize_ms: Optional[float] = None
    retrieve_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    guardrail_ms: Optional[float] = None
    generate_ms: Optional[float] = None
    postprocess_ms: Optional[float] = None
    total_post_stt_ms: Optional[float] = None
    timings: list[StageTiming] = Field(default_factory=list)


class PipelineResponse(BaseModel):
    """Output from the harness."""
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    refusal_reason: Optional[RefusalReason] = None
    transcript: Optional[str] = Field(None, description="STT transcript (set when audio input was transcribed)")
    latency_breakdown: Optional[LatencyBreakdown] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
