#!/usr/bin/env python3
"""Aggregate NVFP4 regular-vs-CUDA-Graph pipeline JSONL records."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(records: list[dict[str, object]], tolerance: float = 1.0e-3) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        shape = tuple(int(value) for value in record["shape"])
        grouped[shape].append(record)
    summaries: list[dict[str, object]] = []
    for shape, runs in sorted(grouped.items()):
        regular = [float(r["quantized_pipeline_ms"]) for r in runs]
        graph = [float(r["cuda_graph_pipeline_ms"]) for r in runs if r.get("cuda_graph_pipeline_ms")]
        errors = [
            float(r["cuda_graph_max_abs_error_vs_pipeline"])
            for r in runs
            if isinstance(r.get("cuda_graph_max_abs_error_vs_pipeline"), (int, float))
        ]
        dynamic_errors = [
            float(r["cuda_graph_dynamic_max_abs_error_vs_pipeline"])
            for r in runs
            if isinstance(r.get("cuda_graph_dynamic_max_abs_error_vs_pipeline"), (int, float))
        ]
        graph_correct = (
            bool(graph)
            and len(errors) == len(graph)
            and max(errors) <= tolerance
            and (not dynamic_errors or len(dynamic_errors) == len(graph))
            and (not dynamic_errors or max(dynamic_errors) <= tolerance)
        )
        regular_median = statistics.median(regular)
        graph_median = statistics.median(graph) if graph else None
        reduction = (1.0 - graph_median / regular_median) * 100.0 if graph_median else None
        summaries.append(
            {
                "shape": list(shape),
                "runs": len(runs),
                "regular_pipeline_ms_median": regular_median,
                "cuda_graph_pipeline_ms_median": graph_median,
                "pipeline_reduction_percent": reduction,
                "graph_correct": graph_correct,
                "dynamic_correctness_checked": bool(dynamic_errors),
                "recommendation": "cuda_graph" if graph_correct and reduction and reduction > 0 else "regular_pipeline",
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(summarize(records, args.tolerance), indent=2))


if __name__ == "__main__":
    main()
