"""Correctness-aware comparison of two reproducible benchmark logs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .stability import summarize


def render_markdown(result: dict[str, object]) -> str:
    """Render a comparison result as a compact, publication-ready table."""

    candidate = result.get("candidate") or {}
    baseline = result.get("baseline") or {}
    workload = candidate.get("workload") or baseline.get("workload") or {}

    def value(summary: dict[str, object], key: str, default: str = "n/a") -> object:
        item = summary.get(key, default)
        if key == "source" and isinstance(item, str):
            return item.replace("\\", "/")
        if key == "correctness" and isinstance(item, dict):
            return f"{item.get('passed', 'n/a')}/{item.get('total', 'n/a')}"
        return item

    gate = "PASS" if result.get("promotion_gate") else "FAIL"
    lines = [
        "## Kairo benchmark comparison",
        "",
        f"**Promotion gate: {gate}** (minimum ratio "
        f"{result.get('minimum_ratio', 'n/a')}, effective repeats "
        f"{result.get('effective_minimum_repeats', 'n/a')})",
        "",
        "| Metric | Candidate | Baseline |",
        "|---|---:|---:|",
        f"| Source | `{value(candidate, 'source')}` | `{value(baseline, 'source')}` |",
        f"| Repeats | {value(candidate, 'bench_repeats')} | {value(baseline, 'bench_repeats')} |",
        f"| Correctness | {value(candidate, 'correctness')} | {value(baseline, 'correctness')} |",
        f"| Workload consistent | {value(candidate, 'workload_consistent')} | {value(baseline, 'workload_consistent')} |",
        f"| Throughput median (tok/s) | {value(candidate, 'throughput_median_tok_s')} | {value(baseline, 'throughput_median_tok_s')} |",
        f"| TTFT P50 median (ms) | {value(candidate, 'ttft_p50_ms_median')} | {value(baseline, 'ttft_p50_ms_median')} |",
        f"| Total P99 median (ms) | {value(candidate, 'total_p99_ms_median')} | {value(baseline, 'total_p99_ms_median')} |",
        "",
        f"**Throughput ratio:** {result.get('throughput_ratio', 'n/a')}x "
        f"({result.get('throughput_delta_percent', 'n/a')}%)",
        f"**Workload match:** `{result.get('workload_match', False)}`; "
        f"**Correctness/request gate:** `{result.get('correctness_and_success_ok', False)}`",
        "",
        "### Workload identity",
        "",
        "```json",
        json.dumps(workload, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _usable(summary: dict[str, object], minimum_repeats: int) -> bool:
    correctness = summary.get("correctness") or {}
    return bool(
        int(summary.get("bench_repeats", 0)) >= minimum_repeats
        and summary.get("all_requests_successful")
        and correctness.get("passed") == correctness.get("total")
    )


def compare_logs(
    candidate_path: Path,
    baseline_path: Path,
    *,
    minimum_ratio: float = 1.20,
    drop_first: bool = False,
    minimum_repeats: int = 2,
) -> dict[str, object]:
    """Compare candidate throughput against a same-workload baseline.

    The promotion gate is false unless both logs pass correctness/request
    gates, expose identical workload configs, and clear ``minimum_ratio``.
    ``drop_first`` is explicit for Graph captures whose first repeat includes
    one-time compilation; the returned report retains both raw summaries.
    """

    if minimum_ratio <= 0:
        raise ValueError("minimum_ratio must be positive")
    if minimum_repeats < 1:
        raise ValueError("minimum_repeats must be positive")
    candidate = summarize(candidate_path)
    baseline = summarize(baseline_path)
    candidate_workload = candidate.get("workload")
    baseline_workload = baseline.get("workload")
    workload_match = (
        isinstance(candidate_workload, dict)
        and isinstance(baseline_workload, dict)
        and candidate.get("workload_consistent", False)
        and baseline.get("workload_consistent", False)
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
    effective_minimum_repeats = minimum_repeats + (1 if drop_first else 0)
    correctness_ok = (
        _usable(candidate, effective_minimum_repeats)
        and _usable(baseline, effective_minimum_repeats)
    )
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
        "minimum_repeats": minimum_repeats,
        "effective_minimum_repeats": effective_minimum_repeats,
        "correctness_and_success_ok": correctness_ok,
        "promotion_gate": bool(
            correctness_ok and workload_match and ratio >= minimum_ratio
        ),
    }
