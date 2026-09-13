#!/usr/bin/env python3
"""Compare a candidate benchmark log against a correctness-aware baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kairo_lab.comparison import compare_logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--minimum-ratio", type=float, default=1.20)
    parser.add_argument(
        "--drop-first",
        action="store_true",
        help="exclude the first repeat from both medians (explicit capture-cost analysis)",
    )
    args = parser.parse_args()
    result = compare_logs(
        args.candidate,
        args.baseline,
        minimum_ratio=args.minimum_ratio,
        drop_first=args.drop_first,
    )
    print(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["promotion_gate"] else 1)


if __name__ == "__main__":
    main()
