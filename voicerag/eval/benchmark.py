"""Full VoiceRAG harness benchmark with local Ollama generation.

This benchmark measures the real pipeline, not a raw model call:

    PipelineRequest -> normalize -> retrieve -> rerank -> guardrail
    -> generate -> citation/groundedness handling -> postprocess

It uses verbatim Hindi questions from ``data/raw/eval_qa.parquet`` by default,
so the benchmark matches the Hindi MSMARCO-XI evaluation corpus.

Run from the repository root::

    python -m voicerag.eval.benchmark --n 20 --warmup 2

Or from the ``voicerag`` package root::

    python -m eval.benchmark --n 20 --warmup 2

The first warm-up requests are excluded from summary statistics. By default,
benchmark requests bypass the orchestrator response cache so every measured
row exercises the full harness.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", str(_REPO_ROOT / "voicerag" / "data" / "index")))
DEFAULT_RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", str(_REPO_ROOT / "voicerag" / "data" / "raw")))
DEFAULT_OUTPUT = _REPO_ROOT / "voicerag" / "eval" / "harness_results.csv"
DEFAULT_SUMMARY = _REPO_ROOT / "voicerag" / "eval" / "harness_summary.json"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_LLM_MODEL = "qwen2.5:1.5b"


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Return a linear-interpolated percentile, or None for no observations."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return round(ordered[lower], 2)
    value = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return round(value, 2)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    """Return consistent latency statistics for one stage."""
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 2) if values else None,
        "p50_ms": percentile(values, 50),
        "p70_ms": percentile(values, 70),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": round(max(values), 2) if values else None,
    }


def _load_hindi_queries(raw_dir: Path, n: int, warmup: int) -> list[str]:
    """Load exact Hindi query strings from the local evaluation parquet."""
    eval_path = raw_dir / "eval_qa.parquet"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Hindi evaluation file not found: {eval_path}. "
            "Run the ingest pipeline first or pass --raw-dir."
        )

    import pandas as pd

    frame = pd.read_parquet(eval_path)
    if "query" not in frame.columns:
        raise ValueError(f"Expected a 'query' column in {eval_path}; found {list(frame.columns)}")

    queries = [str(value) for value in frame["query"].dropna().tolist() if str(value).strip()]
    required = n + warmup
    if len(queries) < required:
        raise ValueError(f"Requested {required} queries, but only {len(queries)} are available")
    return queries[:required]


def _clear_harness_cache(orchestrator_module: Any) -> None:
    """Disable response-cache hits so measured rows execute the full harness."""
    cache = getattr(orchestrator_module, "_query_cache", None)
    if cache is not None:
        cache.clear()


def _timing_map(telemetry: list[Any]) -> dict[str, float]:
    """Aggregate the latest timing event for each stage in one pipeline run."""
    result: dict[str, float] = {}
    for event in telemetry:
        stage_name = getattr(event, "stage_name", "")
        duration = getattr(event, "duration_ms", None)
        if stage_name and duration is not None:
            result[stage_name] = round(float(duration), 4)
    return result


def _stage_statuses(telemetry: list[Any], stage_name: str) -> list[str]:
    """Return all statuses emitted by one stage, preserving event order."""
    return [
        str(getattr(event, "status", ""))
        for event in telemetry
        if getattr(event, "stage_name", "") == stage_name
    ]


def _post_stt_from_timings(timings: dict[str, float]) -> float | None:
    """Calculate post-STT stage sum when the response ended before postprocess."""
    excluded = {"transcribe"}
    values = [duration for stage, duration in timings.items() if stage not in excluded]
    return round(sum(values), 4) if values else None


def _configure_local_llm(model: str, base_url: str) -> None:
    """Configure the harness process to call the local Ollama OpenAI API."""
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["LLM_MODEL"] = model
    # Ollama ignores this value, but the harness uses a non-empty key as the
    # switch that enables its LLM path.
    os.environ.setdefault("LLM_API_KEY", "ollama")


def run_benchmark(
    n_queries: int = 20,
    warmup: int = 2,
    passes: int = 1,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
    summary_path: str | Path = DEFAULT_SUMMARY,
    top_k: int = 5,
    budget_ms: float = 200.0,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    use_cache: bool = False,
    force_llm: bool = True,
    generation_timeout_ms: float = 1500.0,
) -> dict[str, Any]:
    """Run the complete VoiceRAG harness and aggregate stage telemetry."""
    _configure_local_llm(llm_model, llm_base_url)
    if force_llm:
        os.environ["FORCE_LLM_GENERATION"] = "true"
    else:
        os.environ.pop("FORCE_LLM_GENERATION", None)

    from voicerag.harness import orchestrator as orchestrator_module
    from voicerag.harness.orchestrator import run_pipeline
    from voicerag.harness.retriever import HybridRetriever
    from voicerag.harness.stages import get_telemetry
    from voicerag.schemas.contracts import Language, PipelineRequest

    queries = _load_hindi_queries(Path(raw_dir), n_queries, warmup)
    retriever = HybridRetriever(index_dir)
    rows: list[dict[str, Any]] = []

    try:
        logger.info("Warming up %d full-harness requests", warmup)
        for index, query in enumerate(queries[:warmup]):
            if not use_cache:
                _clear_harness_cache(orchestrator_module)
            request = PipelineRequest(
                transcript=query,
                language=Language.HI,
                session_id=f"benchmark-warmup-{index}",
            )
            run_pipeline(
                request,
                retriever,
                top_k=top_k,
                generate_timeout_ms=generation_timeout_ms,
            )

        benchmark_queries = queries[warmup:]
        logger.info(
            "Benchmarking %d Hindi queries × %d pass(es) with model %s",
            len(benchmark_queries),
            passes,
            llm_model,
        )

        for pass_index in range(passes):
            for query_index, query in enumerate(benchmark_queries):
                if not use_cache:
                    _clear_harness_cache(orchestrator_module)

                request = PipelineRequest(
                    transcript=query,
                    language=Language.HI,
                    session_id=f"benchmark-p{pass_index}-q{query_index}",
                )
                started = time.perf_counter()
                response = run_pipeline(
                    request,
                    retriever,
                    top_k=top_k,
                    generate_timeout_ms=generation_timeout_ms,
                )
                wall_ms = (time.perf_counter() - started) * 1000.0
                telemetry = get_telemetry()
                timings = _timing_map(telemetry)
                generate_statuses = _stage_statuses(telemetry, "generate")
                llm_succeeded = bool(generate_statuses == ["success"] and force_llm)
                llm_attempted = force_llm and bool(generate_statuses)
                response_breakdown = response.latency_breakdown
                post_stt_ms = (
                    response_breakdown.total_post_stt_ms
                    if response_breakdown and response_breakdown.total_post_stt_ms is not None
                    else _post_stt_from_timings(timings)
                )

                row: dict[str, Any] = {
                    "pass": pass_index,
                    "query_idx": query_index,
                    "query": query,
                    "refused": response.refused,
                    "refusal_reason": response.refusal_reason.value if response.refusal_reason else "",
                    "error": response.error or "",
                    "confidence": response.confidence,
                    "extractive": not llm_succeeded,
                    "llm_attempted": llm_attempted,
                    "llm_succeeded": llm_succeeded,
                    "generate_statuses": ";".join(generate_statuses),
                    "citations": ";".join(response.citations),
                    "wall_ms": round(wall_ms, 4),
                    "post_stt_ms": round(float(post_stt_ms), 4) if post_stt_ms is not None else None,
                    "answer_chars": len(response.answer or ""),
                }
                for stage_name, duration in timings.items():
                    row[f"{stage_name}_ms"] = duration
                rows.append(row)
    finally:
        retriever.close()

    frame_columns = sorted({key for row in rows for key in row})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    stage_names = sorted(
        {
            key.removesuffix("_ms")
            for row in rows
            for key in row
            if key.endswith("_ms") and key not in {"wall_ms", "post_stt_ms"}
        }
    )
    summary: dict[str, Any] = {
        "config": {
            "n_queries": n_queries,
            "warmup": warmup,
            "passes": passes,
            "top_k": top_k,
            "budget_ms": budget_ms,
            "llm_model": llm_model,
            "llm_base_url": llm_base_url,
            "cache_enabled": use_cache,
            "force_llm": force_llm,
            "generation_timeout_ms": generation_timeout_ms,
            "input": "verbatim Hindi queries from eval_qa.parquet",
        },
        "counts": {
            "measured_rows": len(rows),
            "refused_rows": sum(1 for row in rows if row["refused"]),
            "error_rows": sum(1 for row in rows if row["error"]),
            "cited_rows": sum(1 for row in rows if row["citations"]),
            "llm_attempted_rows": sum(1 for row in rows if row["llm_attempted"]),
            "llm_succeeded_rows": sum(1 for row in rows if row["llm_succeeded"]),
        },
        "latency": {
            "wall": _stats([float(row["wall_ms"]) for row in rows]),
            "post_stt": _stats([float(row["post_stt_ms"]) for row in rows if row["post_stt_ms"] is not None]),
            **{
                stage: _stats(
                    [float(row[f"{stage}_ms"]) for row in rows if row.get(f"{stage}_ms") is not None]
                )
                for stage in stage_names
            },
        },
    }
    post_stt_p70 = summary["latency"]["post_stt"]["p70_ms"]
    summary["pass"] = bool(
        post_stt_p70 is not None
        and post_stt_p70 <= budget_ms
        and summary["counts"]["llm_succeeded_rows"] > 0
    )

    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved benchmark rows to %s", output)
    logger.info("Saved benchmark summary to %s", summary_file)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark the full VoiceRAG harness with local Ollama Qwen")
    parser.add_argument("--n", type=int, default=20, help="Number of unique Hindi queries to measure")
    parser.add_argument("--warmup", type=int, default=2, help="Full-harness warmup requests excluded from statistics")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes over the measured queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument(
        "--generation-timeout-ms",
        type=float,
        default=1500.0,
        help="Benchmark-only LLM timeout; production remains bounded by the harness default",
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Include orchestrator response-cache hits; disabled by default for full-pipeline measurement",
    )
    parser.add_argument(
        "--no-force-llm",
        action="store_false",
        dest="force_llm",
        default=True,
        help="Allow the grounded extractive fast path; by default the benchmark forces local LLM generation",
    )
    args = parser.parse_args()

    summary = run_benchmark(
        n_queries=args.n,
        warmup=args.warmup,
        passes=args.passes,
        index_dir=args.index_dir,
        raw_dir=args.raw_dir,
        output_path=args.output,
        summary_path=args.summary,
        top_k=args.top_k,
        budget_ms=args.budget_ms,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        use_cache=args.use_cache,
        force_llm=args.force_llm,
        generation_timeout_ms=args.generation_timeout_ms,
    )

    print("\n" + "=" * 72)
    print("FULL VOICERAG HARNESS BENCHMARK")
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 72)
    print("PASS" if summary["pass"] else "FAIL")
    if not summary["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
