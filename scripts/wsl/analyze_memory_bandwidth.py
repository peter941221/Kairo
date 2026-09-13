#!/usr/bin/env python3
"""Summarize repeated RTX memory-bandwidth probe records."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(records: list[dict[str, object]], minimum_repeats: int = 2) -> dict[str, object]:
    if minimum_repeats < 1:
        raise ValueError("minimum_repeats must be positive")
    values: list[float] = []
    for record in records:
        value = record.get("read_write_gbps")
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("each record must contain a positive read_write_gbps")
        values.append(float(value))
    if len(values) < minimum_repeats:
        raise ValueError(f"need at least {minimum_repeats} valid repeats, got {len(values)}")
    median = statistics.median(values)
    return {
        "repeats": len(values),
        "read_write_gbps_per_repeat": [round(value, 3) for value in values],
        "read_write_gbps_median": round(median, 3),
        "relative_range_percent_of_median": round(
            (max(values) - min(values)) / median * 100.0, 2
        ),
        "gate": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--minimum-repeats", type=int, default=2)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(json.dumps(summarize(records, args.minimum_repeats), indent=2) + "\n")


if __name__ == "__main__":
    main()
