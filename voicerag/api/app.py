"""
API layer — FastHTTP API exposing the VoiceRAG pipeline.

Endpoints:
    POST /query           — text query → answer with citations
    POST /query/audio     — audio query → answer with citations
    GET  /health          — health check
    GET  /stats           — latency stats from recent runs

Usage:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from voicerag.schemas.contracts import (
    Language,
    LatencyBreakdown,
    PipelineResponse,
)

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────

app = FastAPI(
    title="VoiceRAG API",
    description="Voice-enabled RAG system with grounded, cited answers",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (UI)
ui_path = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_path):
    app.mount("/static", StaticFiles(directory=ui_path), name="static")

# Global retriever instance (lazy-loaded)
_retriever = None
_pipeline_runs: list[dict] = []  # In-memory stats store


def _get_retriever():
    global _retriever
    if _retriever is None:
        import logging
        index_dir = os.environ.get("VECTOR_DB_PATH", "./data/index")
        abs_index = os.path.abspath(index_dir)
        logging.warning("[retriever] Loading index from: %s (abs=%s)", index_dir, abs_index)
        try:
            from voicerag.harness.retriever import HybridRetriever
            _retriever = HybridRetriever(index_dir)
            logging.warning("[retriever] Loaded %d vectors", _retriever.faiss_index.ntotal)
        except Exception as e:
            logging.error("[retriever] Failed to load index: %s", e)
            return None
    return _retriever


# ── Request / Response models ───────────────────────────────────────

class TextQueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    language: Language = Field(Language.AUTO)
    top_k: int = Field(5, ge=1, le=20)
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    refused: bool
    error: Optional[str] = None
    refusal_reason: Optional[str] = None
    extractive: bool = False
    transcript: Optional[str] = Field(None, description="STT transcript from audio input")
    latency_breakdown: Optional[LatencyBreakdown] = None
    session_id: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the UI at the root path."""
    ui_file = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    if os.path.exists(ui_file):
        return FileResponse(ui_file)
    return {"message": "VoiceRAG API running"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "voicerag"}


@app.post("/query", response_model=QueryResponse)
async def query_text(req: TextQueryRequest):
    """Text-in → answer-out endpoint."""
    from voicerag.schemas.contracts import PipelineRequest
    from voicerag.harness.orchestrator import run_pipeline

    request = PipelineRequest(
        transcript=req.query,
        language=req.language,
        session_id=req.session_id,
    )

    retriever = _get_retriever()
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index not ready yet. Wait 1-2 minutes for first-time setup.",
        )
    try:
        response = run_pipeline(request, retriever, top_k=req.top_k)
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))

    # Record stats
    _record_run(response)

    return QueryResponse(
        answer=response.answer,
        citations=response.citations,
        confidence=response.confidence,
        refused=response.refused,
        refusal_reason=response.refusal_reason.value if response.refusal_reason else None,
        transcript=response.transcript,
        latency_breakdown=response.latency_breakdown,
        session_id=response.session_id,
        error=response.error,
    )


@app.post("/query/audio", response_model=QueryResponse)
async def query_audio(
    audio: UploadFile = File(...),
    language: str = "auto",
    top_k: int = 5,
    session_id: Optional[str] = None,
):
    """Audio-in → answer-out endpoint."""
    from voicerag.schemas.contracts import PipelineRequest
    from voicerag.harness.orchestrator import run_pipeline

    audio_bytes = await audio.read()
    audio_content_type = audio.content_type or "audio/webm"

    request = PipelineRequest(
        audio=audio_bytes,
        audio_content_type=audio_content_type,
        language=Language(language),
        session_id=session_id,
    )

    retriever = _get_retriever()
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index not ready yet. Wait 1-2 minutes for first-time setup.",
        )
    try:
        response = run_pipeline(request, retriever, top_k=top_k)
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))

    _record_run(response)

    return QueryResponse(
        answer=response.answer,
        citations=response.citations,
        confidence=response.confidence,
        refused=response.refused,
        refusal_reason=response.refusal_reason.value if response.refusal_reason else None,
        transcript=response.transcript,
        latency_breakdown=response.latency_breakdown,
        session_id=response.session_id,
        error=response.error,
    )


@app.get("/stats")
async def stats():
    """Return latency stats from recent pipeline runs."""
    if not _pipeline_runs:
        return {"runs": 0, "message": "No pipeline runs recorded yet"}

    post_stt_times = [r["total_post_stt_ms"] for r in _pipeline_runs if r.get("total_post_stt_ms")]

    import numpy as np
    if post_stt_times:
        sorted_times = sorted(post_stt_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p70 = sorted_times[int(len(sorted_times) * 0.7)]
        p100 = sorted_times[-1]
        stats = {
            "total_runs": len(_pipeline_runs),
            "post_stt": {
                "p50_ms": round(p50, 2),
                "p70_ms": round(p70, 2),
                "p100_ms": round(p100, 2),
                "mean_ms": round(float(np.mean(post_stt_times)), 2),
                "n": len(post_stt_times),
            },
            "recent_runs": _pipeline_runs[-10:],
        }
    else:
        stats = {"total_runs": len(_pipeline_runs), "recent_runs": _pipeline_runs[-10:]}

    return stats


def _record_run(response: PipelineResponse) -> None:
    """Record a pipeline run for stats."""
    run = {
        "session_id": response.session_id,
        "refused": response.refused,
        "confidence": response.confidence,
        "answer_len": len(response.answer),
    }
    if response.latency_breakdown:
        lb = response.latency_breakdown
        run["total_post_stt_ms"] = lb.total_post_stt_ms
        run["retrieve_ms"] = lb.retrieve_ms
        run["generate_ms"] = lb.generate_ms
        for t in lb.timings:
            run[f"{t.stage_name}_ms"] = t.duration_ms
    _pipeline_runs.append(run)
    # Keep last 1000 runs
    if len(_pipeline_runs) > 1000:
        _pipeline_runs[:] = _pipeline_runs[-1000:]
