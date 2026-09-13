#!/usr/bin/env python3
"""Validate a pinned serving protocol against its raw benchmark logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit("PyYAML is required to validate a serving protocol") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kairo_lab.stability import summarize


def _close(actual: float, expected: float, tolerance: float = 0.02) -> bool:
    return abs(actual - expected) <= tolerance


def _validate_lane(
    repo_root: Path,
    lane_name: str,
    lane: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    raw_path = repo_root / str(lane["raw_output"])
    summary = summarize(raw_path)
    expected_workload = {
        "concurrency": workload["concurrency"],
        "requests": workload["requests"],
        "prompt_tokens_requested": workload["prompt_tokens_requested"],
        "generation_tokens": workload["generation_tokens"],
        "warmup": workload["warmup"],
        "disable_thinking": True,
        "ignore_eos": True,
        "context_tokens": workload["context_tokens"],
        "prompt_tokens_actual": workload["prompt_tokens_actual_mean"],
        "generation_tokens_actual": workload["generation_tokens_actual"],
    }
    actual_workload = summary.get("workload") or {}
    workload_ok = all(
        key in actual_workload
        and (
            _close(float(actual_workload[key]), float(expected))
            if isinstance(expected, (int, float))
            else actual_workload[key] == expected
        )
        for key, expected in expected_workload.items()
    )
    expected_correctness = str(lane["correctness"]).split("/", 1)
    correctness = summary.get("correctness") or {}
    correctness_ok = (
        len(expected_correctness) == 2
        and correctness.get("passed") == int(expected_correctness[0])
        and correctness.get("total") == int(expected_correctness[1])
    )
    expected_throughput = [float(value) for value in lane["throughput_tok_s"]]
    actual_throughput = [float(value) for value in summary["throughput_tok_s"]]
    throughput_ok = len(actual_throughput) == len(expected_throughput) and all(
        _close(actual, expected)
        for actual, expected in zip(actual_throughput, expected_throughput)
    )
    expected_success = [int(value) for value in lane["successful_requests_per_repeat"]]
    success_ok = summary["successful_requests_per_repeat"] == expected_success
    median_ok = _close(
        float(summary["throughput_median_tok_s"]), float(lane["median_tok_s"])
    )
    checks = {
        "raw_exists": raw_path.exists(),
        "workload": workload_ok,
        "correctness": correctness_ok,
        "requests_successful": success_ok and bool(summary["all_requests_successful"]),
        "throughput_repeats": throughput_ok,
        "throughput_median": median_ok,
    }
    return {
        "lane": lane_name,
        "raw_output": str(raw_path),
        "checks": checks,
        "valid": all(checks.values()),
        "summary": summary,
    }


def validate(protocol_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("pinned_current_validation"), dict
    ):
        raise ValueError("protocol has no pinned_current_validation section")
    section = document["pinned_current_validation"]
    workload = section.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("pinned_current_validation has no workload")
    repo_root = protocol_path.resolve().parents[2]
    lanes = [
        _validate_lane(repo_root, "graph", section["graph"], workload),
        _validate_lane(repo_root, "eager_control", section["eager_control"], workload),
    ]
    comparison = section.get("comparison") or {}
    actual_ratio = lanes[0]["summary"]["throughput_median_tok_s"] / lanes[1]["summary"]["throughput_median_tok_s"]
    comparison_checks = {
        "ratio": _close(actual_ratio, float(comparison["median_throughput_speedup"])),
        "protocol_gate": comparison.get("promotion_gate") is True,
    }
    return {
        "protocol": str(protocol_path),
        "lanes": lanes,
        "comparison": {
            "checks": comparison_checks,
            "actual_median_throughput_speedup": round(actual_ratio, 4),
        },
        "valid": all(lane["valid"] for lane in lanes) and all(comparison_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    result = validate(args.protocol)
    print(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
