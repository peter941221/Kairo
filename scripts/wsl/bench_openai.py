#!/usr/bin/env python3
"""Small, dependency-free OpenAI-compatible benchmark runner for Kairo."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Sample:
    request_id: int
    ok: bool
    ttft_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * percentile / 100
    low, high = int(rank), min(int(rank) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (rank - low)


def _prompt(token_target: int, request_id: int) -> str:
    # The server's usage field is authoritative; this deterministic text gives
    # the runner a stable target without requiring a tokenizer dependency.
    prefix = f"Kairo request {request_id}. "
    return (prefix + ("Kairo benchmark token. " * max(1, token_target // 4 + 1)))[: token_target * 5]


def _one(args: argparse.Namespace, request_id: int) -> Sample:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": _prompt(args.prompt_tokens, request_id)}],
        "max_tokens": args.generation_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.disable_thinking:
        # SGLang accepts this as a top-level OpenAI-compatible extension;
        # runtimes that ignore the key still receive the same prompt/body.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.ignore_eos:
        # Keep decode runs at an exact length so throughput comparisons are
        # not skewed by requests that happen to emit EOS early.
        body["ignore_eos"] = True
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    usage = {}
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                payload = line.split(b":", 1)[1].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices and first is None and (
                    (choices[0].get("delta") or {}).get("content")
                    or choices[0].get("finish_reason")
                ):
                    first = time.perf_counter()
        ended = time.perf_counter()
        if first is None:
            first = ended
        return Sample(
            request_id=request_id,
            ok=True,
            ttft_ms=(first - started) * 1000,
            total_ms=(ended - started) * 1000,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
    except Exception as exc:  # benchmark output should preserve failures
        return Sample(request_id=request_id, ok=False, error=f"{type(exc).__name__}: {exc}")


def run(args: argparse.Namespace) -> dict:
    if args.warmup:
        for index in range(args.warmup):
            sample = _one(args, -(index + 1))
            if not sample.ok:
                raise RuntimeError(f"warmup failed: {sample.error}")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        samples = list(pool.map(lambda i: _one(args, i), range(args.requests)))
    elapsed = time.perf_counter() - started
    good = [sample for sample in samples if sample.ok]
    ttft = [sample.ttft_ms for sample in good if sample.ttft_ms is not None]
    total = [sample.total_ms for sample in good if sample.total_ms is not None]
    completion = [sample.completion_tokens for sample in good if sample.completion_tokens]
    prompt = [sample.prompt_tokens for sample in good if sample.prompt_tokens]
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "lane": args.lane,
        "config": {
            "concurrency": args.concurrency,
            "prompt_tokens_requested": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "warmup": args.warmup,
            "requests": args.requests,
            "disable_thinking": args.disable_thinking,
            "ignore_eos": args.ignore_eos,
            "prompt_tokens_actual": statistics.mean(prompt) if prompt else None,
            "generation_tokens_actual": statistics.mean(completion) if completion else None,
            **({"model_revision": args.model_revision} if args.model_revision else {}),
            **({"source_revision": args.source_revision} if args.source_revision else {}),
            **({"context_tokens": args.context_tokens} if args.context_tokens else {}),
        },
        "summary": {
            "ok": len(good),
            "failed": len(samples) - len(good),
            "wall_s": elapsed,
            "ttft_p50_ms": _percentile(ttft, 50),
            "ttft_p99_ms": _percentile(ttft, 99),
            "total_p50_ms": _percentile(total, 50),
            "total_p99_ms": _percentile(total, 99),
            "output_tokens_per_s": sum(completion) / elapsed if completion else None,
            "input_tokens_per_s": sum(prompt) / elapsed if prompt else None,
            "prompt_tokens_actual": statistics.mean(
                prompt
            )
            if prompt
            else None,
        },
        "samples": [asdict(sample) for sample in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="smoke")
    parser.add_argument("--lane", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument(
        "--context-tokens",
        type=int,
        help="declared service context limit to include in the audit record",
    )
    parser.add_argument(
        "--model-revision",
        help="pinned model snapshot revision to include in the audit record",
    )
    parser.add_argument(
        "--source-revision",
        help="Kairo source revision to include in the audit record",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
