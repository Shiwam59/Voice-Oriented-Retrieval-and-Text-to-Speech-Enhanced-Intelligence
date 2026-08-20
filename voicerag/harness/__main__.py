"""
Harness entry point — run the pipeline end-to-end on a query.

Usage:
    python -m harness.run --query "..."
    python -m harness.run --query "..." --audio fixtures/sample1.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import sys

# Ensure voicerag is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicerag.schemas.contracts import Language, PipelineRequest
from voicerag.harness.orchestrator import run_pipeline
from voicerag.harness.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run VoiceRAG pipeline")
    parser.add_argument("--query", help="Text query (skip STT)")
    parser.add_argument("--audio", help="Audio file path (run STT first)")
    parser.add_argument("--language", default="hi", choices=["hi", "en"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default=os.environ.get("VECTOR_DB_PATH", "./data/index"))
    args = parser.parse_args()

    # Build request
    audio_bytes = None
    audio_content_type = None
    if args.audio:
        with open(args.audio, "rb") as f:
            audio_bytes = f.read()
        audio_content_type = mimetypes.guess_type(args.audio)[0] or "application/octet-stream"

    request = PipelineRequest(
        transcript=args.query,
        audio=audio_bytes,
        audio_content_type=audio_content_type,
        language=Language(args.language),
        session_id="cli-test",
    )

    # Load retriever
    retriever = HybridRetriever(args.index_dir)

    # Run pipeline
    response = run_pipeline(request, retriever, top_k=args.top_k)

    # Print results
    print("\n" + "=" * 60)
    if response.error:
        print(f"ERROR: {response.error}")
    elif response.refused:
        print(f"REFUSED ({response.refusal_reason}): {response.answer}")
    else:
        print(f"ANSWER: {response.answer}")
        print(f"CITATIONS: {response.citations}")
        print(f"CONFIDENCE: {response.confidence:.2f}")

    if response.latency_breakdown:
        lb = response.latency_breakdown
        print(f"\n--- Latency ---")
        if lb.stt_ms:
            print(f"  STT: {lb.stt_ms:.2f} ms")
        if lb.normalize_ms:
            print(f"  Normalize: {lb.normalize_ms:.2f} ms")
        if lb.retrieve_ms:
            print(f"  Retrieve: {lb.retrieve_ms:.2f} ms")
        if lb.rerank_ms:
            print(f"  Rerank: {lb.rerank_ms:.2f} ms")
        if lb.guardrail_ms:
            print(f"  Guardrail: {lb.guardrail_ms:.2f} ms")
        if lb.generate_ms:
            print(f"  Generate: {lb.generate_ms:.2f} ms")
        if lb.total_post_stt_ms:
            print(f"  TOTAL (post-STT): {lb.total_post_stt_ms:.2f} ms")

    print("=" * 60)

    retriever.close()


if __name__ == "__main__":
    main()
