#!/usr/bin/env python3
"""Summarize repeated bench_openai JSON objects from a raw service log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kairo_lab.stability import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.log)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    correctness = result.get("correctness") or {}
    healthy = (
        result["bench_repeats"] > 0
        and result["all_requests_successful"]
        and correctness.get("passed") == correctness.get("total")
    )
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
