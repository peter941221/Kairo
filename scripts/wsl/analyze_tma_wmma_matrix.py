#!/usr/bin/env python3
"""Summarize TMA-WMMA JSONL runs without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[tuple[int, ...], str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        shape = tuple(int(value) for value in record["shape"])
        variant = str(record["variant"])
        grouped[(shape, variant)].append(record)

    variants: list[dict[str, object]] = []
    by_shape: dict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    for (shape, variant), runs in sorted(grouped.items()):
        values = [float(run["tma_gflops"]) for run in runs]
        all_ok = all(
            bool(run.get("tma_ok"))
            and float(run.get("max_abs_error_vs_cublas", 1.0)) < 0.02
            for run in runs
        )
        summary = {
            "shape": list(shape),
            "variant": variant,
            "samples": len(values),
            "median_gflops": statistics.median(values),
            "min_gflops": min(values),
            "max_gflops": max(values),
            "all_correct": all_ok,
        }
        variants.append(summary)
        by_shape[shape].append(summary)

    winners = []
    for shape, candidates in sorted(by_shape.items()):
        valid = [candidate for candidate in candidates if candidate["all_correct"]]
        winner = max(valid, key=lambda candidate: float(candidate["median_gflops"])) if valid else None
        winners.append(
            {
                "shape": list(shape),
                "winner": winner["variant"] if winner else None,
                "winner_median_gflops": winner["median_gflops"] if winner else None,
            }
        )
    return {"variants": variants, "winners": winners}


def load_records(path: str | None) -> list[dict[str, object]]:
    handle = sys.stdin if path in {None, "-"} else Path(path).open(encoding="utf-8")
    try:
        return [json.loads(line) for line in handle if line.strip()]
    finally:
        if handle is not sys.stdin:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="JSONL path, or stdin when omitted")
    args = parser.parse_args()
    print(json.dumps(summarize(load_records(args.input)), indent=2))


if __name__ == "__main__":
    main()
