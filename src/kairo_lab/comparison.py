"""Correctness-aware comparison of two reproducible benchmark logs."""

from __future__ import annotations

import statistics
from pathlib import Path

from .stability import summarize


def _usable(summary: dict[str, object]) -> bool:
    correctness = summary.get("correctness") or {}
    return bool(
        summary.get("bench_repeats", 0)
        and summary.get("all_requests_successful")
        and correctness.get("passed") == correctness.get("total")
    )


def compare_logs(
    candidate_path: Path,
    baseline_path: Path,
    *,
    minimum_ratio: float = 1.20,
    drop_first: bool = False,
) -> dict[str, object]:
    """Compare candidate throughput against a same-workload baseline.

    The promotion gate is false unless both logs pass correctness/request
    gates, expose identical workload configs, and clear ``minimum_ratio``.
    ``drop_first`` is explicit for Graph captures whose first repeat includes
    one-time compilation; the returned report retains both raw summaries.
    """

    if minimum_ratio <= 0:
        raise ValueError("minimum_ratio must be positive")
    candidate = summarize(candidate_path)
    baseline = summarize(baseline_path)
    candidate_workload = candidate.get("workload")
    baseline_workload = baseline.get("workload")
    workload_match = (
        isinstance(candidate_workload, dict)
        and isinstance(baseline_workload, dict)
        and candidate_workload == baseline_workload
    )
    candidate_values = list(candidate.get("throughput_tok_s") or [])
    baseline_values = list(baseline.get("throughput_tok_s") or [])
    if drop_first:
        if len(candidate_values) > 1:
            candidate_values = candidate_values[1:]
        if len(baseline_values) > 1:
            baseline_values = baseline_values[1:]
    candidate_median = statistics.median(candidate_values) if candidate_values else 0.0
    baseline_median = statistics.median(baseline_values) if baseline_values else 0.0
    ratio = candidate_median / baseline_median if baseline_median else 0.0
    correctness_ok = _usable(candidate) and _usable(baseline)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "workload_match": workload_match,
        "drop_first": drop_first,
        "candidate_median_tok_s": round(candidate_median, 2),
        "baseline_median_tok_s": round(baseline_median, 2),
        "throughput_ratio": round(ratio, 4),
        "throughput_delta_percent": round((ratio - 1.0) * 100, 2) if ratio else 0.0,
        "minimum_ratio": minimum_ratio,
        "correctness_and_success_ok": correctness_ok,
        "promotion_gate": bool(
            correctness_ok and workload_match and ratio >= minimum_ratio
        ),
    }
