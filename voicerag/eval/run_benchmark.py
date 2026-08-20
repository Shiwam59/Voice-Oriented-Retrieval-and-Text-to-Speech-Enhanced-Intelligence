"""
Eval benchmark harness — Task 5.4

Runs the retrieval+generation path over a sample of held-out queries
and measures P50/P70/P100 latency. Refusal-path latency is also measured.

Usage:
    python -m eval.run_benchmark --n 100
    python -m eval.run_benchmark --n 100 --warmup 10 --passes 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "./data/index"))
DEFAULT_RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", "./data/raw"))
DEFAULT_OUTPUT = Path("./eval/results.csv")


def run_benchmark(
    n_queries: int = 100,
    warmup: int = 10,
    passes: int = 3,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict:
    """
    Run latency benchmark over held-out queries.
    Returns summary statistics.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from voicerag.schemas.contracts import Language, PipelineRequest
    from voicerag.harness.orchestrator import run_pipeline
    from voicerag.harness.retriever import HybridRetriever

    # Load retriever
    retriever = HybridRetriever(index_dir)

    # Load eval queries
    eval_path = Path(raw_dir) / "eval_qa.parquet"
    if not eval_path.exists():
        logger.warning("No eval_qa.parquet found — using built-in test queries")
        test_queries = [
            "कॉर्पोरेशन क्या है?",
            "भारत की राजधानी कहाँ है?",
            "पानी का रासायनिक सूत्र क्या है?",
        ]
    else:
        eval_df = pd.read_parquet(eval_path)
        test_queries = eval_df["query"].tolist()[:n_queries + warmup]

    # Warmup
    logger.info("Warmup: running %d queries …", warmup)
    for q in test_queries[:warmup]:
        req = PipelineRequest(transcript=q, language=Language.HI, session_id="warmup")
        run_pipeline(req, retriever, top_k=5)

    # Benchmark passes
    results = []
    total = min(n_queries, len(test_queries) - warmup)
    benchmark_queries = test_queries[warmup:warmup + total]

    logger.info("Benchmark: %d queries × %d passes", total, passes)

    for pass_num in range(passes):
        for i, query in enumerate(benchmark_queries):
            req = PipelineRequest(
                transcript=query,
                language=Language.HI,
                session_id=f"bench_p{pass_num}_q{i}",
            )

            start = time.perf_counter()
            response = run_pipeline(req, retriever, top_k=5)
            elapsed = (time.perf_counter() - start) * 1000

            results.append({
                "pass": pass_num,
                "query_idx": i,
                "query": query[:100],
                "refused": response.refused,
                "refusal_reason": str(response.refusal_reason) if response.refusal_reason else "",
                "confidence": response.confidence,
                "total_ms": elapsed,
                "post_stt_ms": response.latency_breakdown.total_post_stt_ms if response.latency_breakdown else None,
                "retrieve_ms": response.latency_breakdown.retrieve_ms if response.latency_breakdown else None,
                "generate_ms": response.latency_breakdown.generate_ms if response.latency_breakdown else None,
            })

    retriever.close()

    # ── Compute stats ──────────────────────────────────────────
    df = pd.DataFrame(results)
    post_stt = df["post_stt_ms"].dropna().values
    retrieve = df["retrieve_ms"].dropna().values
    generate = df["generate_ms"].dropna().values

    summary = {}
    for label, arr in [("post_stt", post_stt), ("retrieve", retrieve), ("generate", generate)]:
        sorted_arr = np.sort(arr)
        summary[label] = {
            "n": len(arr),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p70_ms": round(float(np.percentile(arr, 70)), 2),
            "p100_ms": round(float(np.percentile(arr, 100)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
        }

    # Refusal path timing
    refusal_df = df[df["refused"] == True]
    summary["refusal_rate"] = round(len(refusal_df) / len(df) * 100, 2) if len(df) > 0 else 0
    if len(refusal_df) > 0:
        refusal_times = refusal_df["post_stt_ms"].dropna().values
        summary["refusal_latency"] = round(float(np.mean(refusal_times)), 2)

    # Save CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Results saved → %s", output_path)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run latency benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of queries to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup queries")
    parser.add_argument("--passes", type=int, default=3, help="Number of passes")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    summary = run_benchmark(args.n, args.warmup, args.passes, args.index_dir, args.raw_dir, args.output)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for label, stats in summary.items():
        if isinstance(stats, dict):
            print(f"\n  {label.upper()}:")
            for k, v in stats.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {label}: {stats}")
    print("=" * 60)


if __name__ == "__main__":
    main()
