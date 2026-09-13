#!/usr/bin/env python3
"""Dependency-free deterministic probes for an OpenAI-compatible model server.

This is deliberately a small semantic gate, not a benchmark: it checks that
the serving lane can answer exact-control prompts without protocol errors and
that the returned text matches the expected answer after whitespace trimming.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


CASES = (
    ("sentinel", "Reply with exactly: KAIRO_OK", "KAIRO_OK"),
    ("arithmetic", "What is 2 + 2? Reply with only the integer.", "4"),
    ("token", "Return the exact token KAIRO_FP4 and nothing else.", "KAIRO_FP4"),
    ("chinese", "只回答：正确", "正确"),
)


def probe(base_url: str, model: str, name: str, prompt: str, expected: str, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        choices = payload.get("choices") or []
        content = (((choices[0].get("message") or {}).get("content")) if choices else "") or ""
        actual = content.strip()
        return {
            "name": name,
            "ok": actual == expected,
            "expected": expected,
            "actual": actual,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": None,
        }
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "expected": expected,
            "actual": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="smoke")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [probe(args.base_url, args.model, *case, args.timeout) for case in CASES]
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "protocol": "OpenAI chat completions, temperature=0, thinking disabled",
        "summary": {"passed": sum(case["ok"] for case in cases), "total": len(cases)},
        "cases": cases,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["summary"]["passed"] == result["summary"]["total"] else 1)


if __name__ == "__main__":
    main()
