"""Dependency-light commands for recording reproducible Kairo experiments."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def recommend_profile(model: str, concurrency: int, prompt_tokens: int) -> dict[str, object]:
    """Return the most conservative measured Qwen3.8 serving profile.

    The ratio-8 choice is intentionally limited to the shape where it was
    measured to win (short prompt, c16). Unknown shapes stay on the baseline
    ratio instead of extrapolating a benchmark result.
    """
    if model != "qwen38":
        raise ValueError(f"Unsupported profile model: {model}")
    if concurrency < 1 or prompt_tokens < 1:
        raise ValueError("concurrency and prompt_tokens must be positive")
    short_high_concurrency = concurrency >= 16 and prompt_tokens <= 1024
    if short_high_concurrency:
        return {
            "model": model,
            "profile": "qwen38-sglang-ratio8-short-c16",
            "mamba_full_memory_ratio": 8.0,
            "confidence": "measured_c16",
            "reason": (
                "ratio=8 increased the measured short-prompt c16 throughput; "
                "revalidate for new shapes"
            ),
        }
    return {
        "model": model,
        "profile": "qwen38-sglang-ratio459-baseline",
        "mamba_full_memory_ratio": 4.59,
        "confidence": "baseline_or_unvalidated",
        "reason": (
            "ratio=8 is not yet a measured win for this concurrency/prompt shape"
        ),
    }


def recommend_runtime(model: str, concurrency: int, prompt_tokens: int) -> dict[str, object]:
    """Select only runtime cells that have an observed cross-runtime result.

    Unknown cells intentionally return ``manual`` instead of extrapolating a
    benchmark point. This keeps the policy useful for an experiment router
    without turning a two-point matrix into an unbounded performance claim.
    """
    if model != "qwen38":
        raise ValueError(f"Unsupported runtime model: {model}")
    if concurrency < 1 or prompt_tokens < 1:
        raise ValueError("concurrency and prompt_tokens must be positive")
    if concurrency == 16 and prompt_tokens in {512, 2048}:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-b12x-c16",
            "confidence": "measured_cross_runtime",
            "reason": "vLLM/B12X wins both measured c16 prompt cells",
        }
    if concurrency in {1, 4} and prompt_tokens == 512:
        return {
            "model": model,
            "backend": "sglang",
            "profile": "qwen38-sglang-ratio8-short",
            "confidence": "measured_cross_runtime",
            "reason": "SGLang is slightly faster at the measured c1/c4 short cells",
        }
    return {
        "model": model,
        "backend": "manual",
        "profile": None,
        "confidence": "unvalidated",
        "reason": "No cross-runtime measurement covers this shape yet",
    }


def _nvcc_command() -> str | None:
    for candidate in ("/usr/local/cuda-13.0/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc"):
        if Path(candidate).exists():
            return candidate
    return shutil.which("nvcc")


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, str | None]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_revision": _command_output(["git", "rev-parse", "HEAD"]),
        "gpu": _command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "cuda": (
            _command_output([nvcc, "--version"])
            if (nvcc := _nvcc_command())
            else None
        ),
    }


def init_run(lane: str) -> Path:
    if lane not in {"decode", "prefill", "discovery"}:
        raise ValueError(f"Unsupported lane: {lane}")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{lane}"
    output = ROOT / "runs" / f"{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "lane": lane,
        "status": "initialized",
        "environment": environment(),
        "protocol": "experiments/protocols/v0.yaml",
        "workloads": "experiments/workloads/candidates.yaml",
        "result_contract": {
            "raw_measurements": "required",
            "correctness_status": "required_before_performance_claim",
            "baseline_revision": "required",
        },
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(prog="kairo-lab")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("env", help="print the current reproducibility environment")
    run_parser = subcommands.add_parser("init-run", help="create an ignored run record")
    run_parser.add_argument("--lane", default="decode", choices=["decode", "prefill", "discovery"])
    profile_parser = subcommands.add_parser(
        "recommend-profile", help="recommend a measured serving profile"
    )
    profile_parser.add_argument("--model", default="qwen38", choices=["qwen38"])
    profile_parser.add_argument("--concurrency", type=int, required=True)
    profile_parser.add_argument("--prompt-tokens", type=int, required=True)
    profile_parser.add_argument("--format", choices=["json", "shell"], default="json")
    runtime_parser = subcommands.add_parser(
        "recommend-runtime", help="select a measured cross-runtime serving cell"
    )
    runtime_parser.add_argument("--model", default="qwen38", choices=["qwen38"])
    runtime_parser.add_argument("--concurrency", type=int, required=True)
    runtime_parser.add_argument("--prompt-tokens", type=int, required=True)
    args = parser.parse_args()

    if args.command == "env":
        print(json.dumps(environment(), indent=2))
    elif args.command == "init-run":
        print(init_run(args.lane))
    elif args.command == "recommend-profile":
        result = recommend_profile(args.model, args.concurrency, args.prompt_tokens)
        if args.format == "shell":
            print(f"export KAIRO_MAMBA_FULL_MEMORY_RATIO={result['mamba_full_memory_ratio']}")
            print(f"export KAIRO_PROFILE_ID={result['profile']}")
        else:
            print(json.dumps(result, indent=2))
    else:
        print(
            json.dumps(
                recommend_runtime(args.model, args.concurrency, args.prompt_tokens),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
