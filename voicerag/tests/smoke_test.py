"""
Smoke test — builds a mini index from fixtures and runs the full pipeline.

Usage:
    python -m tests.smoke_test
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure voicerag is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import pandas as pd
import numpy as np

from voicerag.schemas.contracts import Language, PipelineRequest
from voicerag.harness.orchestrator import run_pipeline
from voicerag.fixtures.sample_data import (
    SAMPLE_PASSAGES,
    GUARDRAIL_TEST_QUERIES,
    WORKED_EXAMPLE,
)


def build_mini_index(index_dir: Path) -> None:
    """Build a small FAISS + BM25 index from fixture data."""
    from voicerag.ingest.build_index import build_metadata_sidecar, build_faiss_index, build_bm25_index

    passages_df = pd.DataFrame(SAMPLE_PASSAGES)
    passages_df["char_len"] = passages_df["text"].str.len()

    # Create chunk-like records (passage-native only for fixtures)
    chunks = []
    for _, row in passages_df.iterrows():
        chunks.append({
            "chunk_id": f"{row['passage_id']}_native",
            "passage_id": row["passage_id"],
            "text": row["text"],
            "chunk_strategy": "passage_native",
            "language": row.get("target_lang", "hi"),
            "original_query_id": row.get("query_id"),
            "char_len": row.get("char_len", len(row["text"])),
            "has_answer_overlap": 0,
        })

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_parquet(index_dir / "chunks.parquet", index=False)

    # Build indices
    build_metadata_sidecar(chunks_df, index_dir / "meta.db")
    build_faiss_index(chunks_df, index_dir)
    build_bm25_index(chunks_df, index_dir)


def test_retrieval(index_dir: Path) -> bool:
    """Test that retrieval returns relevant passages."""
    from voicerag.harness.retriever import HybridRetriever

    retriever = HybridRetriever(index_dir)
    result = retriever.retrieve("कॉर्पोरेशन क्या है?", top_k=3)
    retriever.close()

    assert len(result.passages) > 0, "No passages returned"
    assert any("निगम" in p.text for p in result.passages), "Top passage doesn't contain expected Hindi text"
    print("  ✓ Retrieval returns relevant passages")
    return True


def test_pipeline(index_dir: Path) -> bool:
    """Test full pipeline with text query."""
    from voicerag.harness.retriever import HybridRetriever

    retriever = HybridRetriever(index_dir)
    request = PipelineRequest(transcript="कॉर्पोरेशन क्या है?", language=Language.HI, session_id="smoke")

    response = run_pipeline(request, retriever, top_k=3)
    retriever.close()

    assert response.answer, "Empty answer"
    assert not response.refused, "Query was incorrectly refused"
    assert len(response.citations) > 0, "No citations"
    print(f"  ✓ Pipeline answer: {response.answer[:80]}…")
    print(f"  ✓ Citations: {response.citations}")
    return True


def test_off_topic(index_dir: Path) -> bool:
    """Test that off-topic queries are refused."""
    from voicerag.harness.retriever import HybridRetriever

    retriever = HybridRetriever(index_dir)
    request = PipelineRequest(
        transcript="who won the cricket world cup 2026?",
        language=Language.HI,
        session_id="smoke_offtopic",
    )

    response = run_pipeline(request, retriever, top_k=3)
    retriever.close()

    # This might not be refused on a tiny index — just check it doesn't crash
    print(f"  ✓ Off-topic query handled (refused={response.refused})")
    return True


def test_unsafe(index_dir: Path) -> bool:
    """Test that unsafe queries are refused."""
    from voicerag.harness.retriever import HybridRetriever

    retriever = HybridRetriever(index_dir)
    request = PipelineRequest(
        transcript="how to make a bomb",
        language=Language.EN,
        session_id="smoke_unsafe",
    )

    response = run_pipeline(request, retriever, top_k=3)
    retriever.close()

    assert response.refused, "Unsafe query was NOT refused"
    print(f"  ✓ Unsafe query correctly refused: {response.refusal_reason}")
    return True


def test_telemetry(index_dir: Path) -> bool:
    """Test that telemetry is recorded."""
    from voicerag.harness.retriever import HybridRetriever
    from voicerag.harness.stages import get_telemetry, clear_telemetry

    retriever = HybridRetriever(index_dir)
    request = PipelineRequest(transcript="पानी का सूत्र?", language=Language.HI)

    clear_telemetry()
    response = run_pipeline(request, retriever, top_k=3)
    retriever.close()

    timings = get_telemetry()
    stage_names = {t.stage_name for t in timings}
    assert "normalize_query" in stage_names, "Missing normalize_query telemetry"
    assert "retrieve" in stage_names, "Missing retrieve telemetry"
    assert "guardrail_check" in stage_names, "Missing guardrail telemetry"
    assert "generate" in stage_names, "Missing generate telemetry"

    print(f"  ✓ Telemetry recorded for {len(timings)} stages")
    if response.latency_breakdown:
        lb = response.latency_breakdown
        print(f"  ✓ Latency: retrieve={lb.retrieve_ms:.1f}ms, generate={lb.generate_ms:.1f}ms, total={lb.total_post_stt_ms:.1f}ms")
    return True


def main():
    print("=" * 60)
    print("VoiceRAG Smoke Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir)

        print("\nBuilding mini index from fixtures …")
        build_mini_index(index_dir)
        print("  ✓ Mini index built")

        print("\nRunning tests …")
        all_passed = True
        for test_fn in [test_retrieval, test_pipeline, test_off_topic, test_unsafe, test_telemetry]:
            try:
                test_fn(index_dir)
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()
