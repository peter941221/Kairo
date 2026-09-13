#!/usr/bin/env python3
"""Summarize request-arrival waves from a bench_openai JSON result.

TTFT is a client-visible proxy for admission/scheduling waves.  The analyzer
does not infer internal scheduler state; it reports clusters using an explicit
gap threshold so the heuristic remains reviewable and reproducible.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path


def analyze(result: dict, gap_ms: float = 2000.0) -> dict:
    samples = [
        sample
        for sample in result.get("samples", [])
        if sample.get("ok") and sample.get("ttft_ms") is not None
    ]
    ordered = sorted(samples, key=lambda sample: sample["ttft_ms"])
    waves: list[list[dict]] = []
    for sample in ordered:
        if not waves or sample["ttft_ms"] - waves[-1][-1]["ttft_ms"] > gap_ms:
            waves.append([])
        waves[-1].append(sample)

    wave_rows = []
    for index, wave in enumerate(waves, start=1):
        ttfts = [sample["ttft_ms"] for sample in wave]
        wave_rows.append(
            {
                "wave": index,
                "requests": len(wave),
                "first_ttft_ms": min(ttfts),
                "last_ttft_ms": max(ttfts),
                "span_ms": max(ttfts) - min(ttfts),
                "request_ids": [sample["request_id"] for sample in wave],
            }
        )
    gaps = [
        wave_rows[index]["first_ttft_ms"] - wave_rows[index - 1]["last_ttft_ms"]
        for index in range(1, len(wave_rows))
    ]
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_config": result.get("config", {}),
        "gap_threshold_ms": gap_ms,
        "summary": {
            "successful_samples": len(ordered),
            "wave_count": len(wave_rows),
            "wave_sizes": [row["requests"] for row in wave_rows],
            "inter_wave_gaps_ms": gaps,
            "median_ttft_ms": statistics.median(
                [sample["ttft_ms"] for sample in ordered]
            )
            if ordered
            else None,
        },
        "waves": wave_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--gap-ms", type=float, default=2000.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")), args.gap_ms)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
