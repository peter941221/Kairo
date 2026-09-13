#!/usr/bin/env python3
"""Verify a local Hugging Face snapshot revision before serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def discover_revisions(model_path: Path) -> set[str]:
    metadata_dir = model_path / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    for metadata in sorted(metadata_dir.glob("*.metadata")):
        try:
            first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if first_line:
            revisions.add(first_line)
    return revisions


def verify(model_path: Path, expected_revision: str) -> dict[str, object]:
    revisions = discover_revisions(model_path)
    if not revisions:
        raise ValueError(f"no Hugging Face metadata revisions found under {model_path}")
    if revisions != {expected_revision}:
        raise ValueError(
            f"snapshot revision mismatch: expected {expected_revision}, found {sorted(revisions)}"
        )
    return {
        "model_path": str(model_path),
        "expected_revision": expected_revision,
        "found_revisions": sorted(revisions),
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.model_path, args.expected_revision), indent=2) + "\n")


if __name__ == "__main__":
    main()
