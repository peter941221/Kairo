"""Parse and summarize repeated benchmark JSON objects from raw logs."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _json_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    clean = ANSI.sub("", text)
    objects: list[dict] = []
    cursor = 0
    while True:
        start = clean.find("{", cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(clean, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        cursor = end
    return objects


def summarize(path: Path) -> dict[str, object]:
    objects = _json_objects(path.read_text(encoding="utf-8", errors="replace"))
    benches = [
        value for value in objects
        if isinstance(value.get("summary"), dict)
        and "output_tokens_per_s" in value["summary"]
    ]
    correctness = [
        value for value in objects
        if isinstance(value.get("summary"), dict)
        and "passed" in value["summary"]
        and "total" in value["summary"]
    ]
    throughputs = [float(value["summary"]["output_tokens_per_s"]) for value in benches]
    successes = [int(value["summary"].get("ok", 0)) for value in benches]
    failures = [int(value["summary"].get("failed", 0)) for value in benches]
    result: dict[str, object] = {
        "source": str(path),
        "bench_repeats": len(benches),
        "throughput_tok_s": [round(value, 2) for value in throughputs],
        "successful_requests_per_repeat": successes,
        "failed_requests_per_repeat": failures,
        "all_requests_successful": bool(benches) and all(value == 0 for value in failures),
        "correctness": (
            {"passed": correctness[-1]["summary"]["passed"], "total": correctness[-1]["summary"]["total"]}
            if correctness else None
        ),
    }
    for metric in ("ttft_p50_ms", "ttft_p99_ms", "total_p50_ms", "total_p99_ms"):
        values = [
            float(value["summary"][metric])
            for value in benches
            if isinstance(value["summary"].get(metric), (int, float))
        ]
        if values:
            result[f"{metric}_per_repeat"] = [round(item, 2) for item in values]
            result[f"{metric}_median"] = round(statistics.median(values), 2)
    workloads = []
    for bench in benches:
        config = bench.get("config")
        if isinstance(config, dict):
            workloads.append(
                {
                    key: config[key]
                    for key in (
                        "concurrency",
                        "requests",
                        "prompt_tokens_requested",
                        "prompt_tokens_actual",
                        "generation_tokens",
                        "generation_tokens_actual",
                        "warmup",
                        "disable_thinking",
                        "ignore_eos",
                        "context_tokens",
                    )
                    if key in config
                }
            )
    if workloads:
        result["workload"] = workloads[0]
        result["workload_consistent"] = (
            len(workloads) == len(benches)
            and all(item == workloads[0] for item in workloads)
        )
    if throughputs:
        median = statistics.median(throughputs)
        result.update({
            "throughput_median_tok_s": round(median, 2),
            "throughput_mean_tok_s": round(statistics.mean(throughputs), 2),
            "throughput_range_tok_s": [round(min(throughputs), 2), round(max(throughputs), 2)],
            "relative_range_percent_of_median": round((max(throughputs) - min(throughputs)) / median * 100, 2),
        })
    return result
