"""Dependency-light commands for recording reproducible Kairo experiments."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .blueprint import make_compile_plan, validate_blueprint
from .cache import RuntimeKernelCache


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


def recommend_runtime(
    model: str,
    concurrency: int,
    prompt_tokens: int,
    *,
    context_tokens: int | None = None,
    generation_tokens: int | None = None,
) -> dict[str, object]:
    """Select only runtime cells that have an observed cross-runtime result.

    Unknown cells intentionally return ``manual`` instead of extrapolating a
    benchmark point. When context and generation lengths are supplied, Graph
    routes are additionally gated by their measured 1K/128-token envelope.
    Omitting them preserves the original shape-only API for existing callers.
    """
    if model not in {"qwen38", "qwen3_8b"}:
        raise ValueError(f"Unsupported runtime model: {model}")
    if concurrency < 1 or prompt_tokens < 1:
        raise ValueError("concurrency and prompt_tokens must be positive")
    if context_tokens is not None and context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    if generation_tokens is not None and generation_tokens < 1:
        raise ValueError("generation_tokens must be positive")
    if (context_tokens is None) != (generation_tokens is None):
        raise ValueError("context_tokens and generation_tokens must be provided together")
    graph_envelope = (
        context_tokens is None
        or generation_tokens is None
        or (context_tokens <= 1024 and generation_tokens <= 128)
    )
    if model == "qwen3_8b":
        if (
            concurrency == 16
            and prompt_tokens == 2048
            and context_tokens == 4096
            and generation_tokens == 128
        ):
            return {
                "model": model,
                "backend": "vllm-nightly",
                "profile": "qwen3-8b-vllm-nightly-cutlass-full-decode-graph-c16-p2048-4k",
                "linear_backend": "cutlass",
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "max_num_seqs": 16,
                "max_model_len": 4096,
                "confidence": "measured_repeated",
                "reason": "Qwen3-8B NVFP4 Graph retained a measured 1.69x steady-state lead at c16/prompt2048 with 4K context",
            }
        if concurrency in {16, 32} and prompt_tokens == 512 and graph_envelope:
            return {
                "model": model,
                "backend": "vllm-nightly",
                "profile": (
                    "qwen3-8b-vllm-nightly-cutlass-full-decode-graph-"
                    f"c{concurrency}-p512"
                ),
                "linear_backend": "cutlass",
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "max_num_seqs": 32,
                "confidence": "measured_repeated",
                "reason": (
                    "Qwen3-8B NVFP4 Graph reached a measured >2x eager throughput "
                    f"lead at c{concurrency}/prompt512 in the 1K/128 envelope"
                ),
            }
        if concurrency in {16, 32} and prompt_tokens == 512:
            return {
                "model": model,
                "backend": "vllm-nightly",
                "profile": f"qwen3-8b-vllm-nightly-cutlass-c{concurrency}",
                "linear_backend": "cutlass",
                "confidence": "measured_repeated",
                "reason": "Qwen3-8B NVFP4 eager control is covered by repeated c16/prompt512 runs",
            }
        return {
            "model": model,
            "backend": "manual",
            "profile": None,
            "confidence": "unvalidated",
            "reason": "No Qwen3-8B measurement covers this shape yet",
        }
    if concurrency == 8 and prompt_tokens == 256 and graph_envelope:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-cutlass-full-decode-graph-c8",
            "linear_backend": "cutlass",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_num_seqs": 32,
            "confidence": "measured_pilot",
            "reason": "FULL_DECODE_ONLY Graph reached 2.57x eager throughput in paired repeats",
        }
    if concurrency == 32 and prompt_tokens == 256 and graph_envelope:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-cutlass-full-decode-graph-c32",
            "linear_backend": "cutlass",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_num_seqs": 32,
            "confidence": "measured_repeated",
            "reason": "FULL_DECODE_ONLY Graph reached 2.52x median eager throughput across c32 repeats",
        }
    if concurrency == 32 and prompt_tokens == 512 and graph_envelope:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-cutlass-full-decode-graph-c32-p512",
            "linear_backend": "cutlass",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_num_seqs": 32,
            "confidence": "measured_repeated",
            "reason": "FULL_DECODE_ONLY Graph reached 2.45x median eager throughput at c32/prompt512 in the 1K/128 envelope",
        }
    if concurrency == 16 and prompt_tokens == 256 and graph_envelope:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-cutlass-full-decode-graph-c16",
            "linear_backend": "cutlass",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_num_seqs": 32,
            "confidence": "measured_repeated",
            "reason": "FULL_DECODE_ONLY Graph reached 2.45x median eager throughput across c16 repeats",
        }
    if concurrency == 16 and prompt_tokens == 512 and graph_envelope:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-cutlass-full-decode-graph-c16-p512",
            "linear_backend": "cutlass",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_num_seqs": 32,
            "confidence": "measured_repeated",
            "reason": "FULL_DECODE_ONLY Graph reached 2.59x median eager throughput at c16/prompt512 in the 1K/128 envelope",
        }
    if concurrency == 16 and prompt_tokens == 512:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-cutlass-c16",
            "linear_backend": "cutlass",
            "confidence": "measured_repeated",
            "reason": "CUTLASS won two warm-cache c16 repeats at the short prompt",
        }
    if concurrency == 16 and prompt_tokens == 2048:
        return {
            "model": model,
            "backend": "vllm-nightly",
            "profile": "qwen38-vllm-nightly-cutlass-c16",
            "linear_backend": "cutlass",
            "confidence": "measured_repeated",
            "reason": "CUTLASS won two warm-cache c16 repeats at the long prompt",
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


def recommend_gemm(m: int, n: int, k: int) -> dict[str, object]:
    """Choose only a TMA-WMMA variant covered by the measured matrix.

    The m128 candidate is deliberately promoted only for exact measured
    cells. Other aligned shapes use the single-buffer baseline as a safe
    control, while unaligned shapes stay on the caller's fallback path.
    """
    if min(m, n, k) < 1:
        raise ValueError("m, n, and k must be positive")
    measured_winners = {
        (1024, 1024, 1024): "m128",
        (2048, 1024, 4096): "m128",
        (4096, 4096, 4096): "m256",
        (512, 1024, 1024): "m128",
        (8192, 1024, 1024): "m256",
        (2048, 2048, 2048): "m256",
    }
    if (winner := measured_winners.get((m, n, k))) is not None:
        return {
            "shape": [m, n, k],
            "variant": winner,
            "confidence": "measured_shape",
            "reason": f"{winner} is the fastest measured TMA-WMMA candidate for this cell",
        }
    if m >= 64 and m % 16 == 0 and n % 16 == 0 and k % 16 == 0:
        return {
            "shape": [m, n, k],
            "variant": "single",
            "confidence": "aligned_control_unvalidated_shape",
            "reason": "use the single-buffer control until this shape is measured",
        }
    return {
        "shape": [m, n, k],
        "variant": "fallback",
        "confidence": "unaligned_unvalidated",
        "reason": "TMA-WMMA probes require M>=64 and dimensions divisible by 16",
    }


def recommend_nvfp4_pipeline(m: int, n: int, k: int) -> dict[str, object]:
    """Choose the measured NVFP4 pipeline for an exact shape bucket."""

    if min(m, n, k) < 1:
        raise ValueError("m, n, and k must be positive")
    measured_graph_shapes = {
        (1, 4096, 4096): {"reduction_percent": 27.5, "replays_to_amortize": 3800},
        (32, 4096, 4096): {"reduction_percent": 27.5, "replays_to_amortize": 3769},
        (128, 4096, 4096): {"reduction_percent": 36.0, "replays_to_amortize": 3367},
    }
    shape = (m, n, k)
    if (evidence := measured_graph_shapes.get(shape)) is not None:
        return {
            "shape": [m, n, k],
            "strategy": "cuda_graph_shape_bucket",
            "confidence": "measured_repeated",
            **evidence,
            "reason": "exact shape has two-process correctness-aware Graph measurements",
        }
    if n % 32 == 0 and k % 32 == 0:
        return {
            "shape": [m, n, k],
            "strategy": "regular_nvfp4_pipeline",
            "confidence": "aligned_control_unvalidated_shape",
            "reason": "shape is NVFP4-aligned but Graph benefit is not measured",
        }
    return {
        "shape": [m, n, k],
        "strategy": "fallback",
        "confidence": "unaligned_unvalidated",
        "reason": "NVFP4 probe requires N and K divisible by 32",
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


def _load_blueprint(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError("blueprint is not JSON and PyYAML is unavailable") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("blueprint root must be an object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(prog="kairo-lab")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("env", help="print the current reproducibility environment")
    cache_parser = subcommands.add_parser(
        "cache-inspect", help="audit persistent runtime cache entries"
    )
    cache_parser.add_argument("--root", type=Path, required=True)
    blueprint_parser = subcommands.add_parser(
        "validate-blueprint", help="validate a declarative kernel blueprint"
    )
    blueprint_parser.add_argument("--file", type=Path, required=True)
    plan_parser = subcommands.add_parser(
        "plan-blueprint", help="create a cache-aware compile/benchmark plan"
    )
    plan_parser.add_argument("--file", type=Path, required=True)
    plan_parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"), required=True)
    plan_parser.add_argument("--driver-version", required=True)
    plan_parser.add_argument("--gpu-capability", required=True)
    plan_parser.add_argument("--template-version", default="v1")
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
    runtime_parser.add_argument(
        "--model", default="qwen38", choices=["qwen38", "qwen3_8b"]
    )
    runtime_parser.add_argument("--concurrency", type=int, required=True)
    runtime_parser.add_argument("--prompt-tokens", type=int, required=True)
    runtime_parser.add_argument("--context-tokens", type=int)
    runtime_parser.add_argument("--generation-tokens", type=int)
    gemm_parser = subcommands.add_parser(
        "recommend-gemm", help="select a measured TMA-WMMA experiment variant"
    )
    gemm_parser.add_argument("--m", type=int, required=True)
    gemm_parser.add_argument("--n", type=int, required=True)
    gemm_parser.add_argument("--k", type=int, required=True)
    nvfp4_parser = subcommands.add_parser(
        "recommend-nvfp4", help="select a measured NVFP4 pipeline strategy"
    )
    nvfp4_parser.add_argument("--m", type=int, required=True)
    nvfp4_parser.add_argument("--n", type=int, required=True)
    nvfp4_parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    if args.command == "env":
        print(json.dumps(environment(), indent=2))
    elif args.command == "cache-inspect":
        print(json.dumps(RuntimeKernelCache(args.root).inspect(), indent=2))
    elif args.command == "validate-blueprint":
        print(json.dumps(validate_blueprint(_load_blueprint(args.file)), indent=2))
    elif args.command == "plan-blueprint":
        report = validate_blueprint(_load_blueprint(args.file))
        print(
            json.dumps(
                make_compile_plan(
                    report,
                    tuple(args.shape),
                    driver_version=args.driver_version,
                    gpu_capability=args.gpu_capability,
                    template_version=args.template_version,
                ),
                indent=2,
            )
        )
    elif args.command == "init-run":
        print(init_run(args.lane))
    elif args.command == "recommend-profile":
        result = recommend_profile(args.model, args.concurrency, args.prompt_tokens)
        if args.format == "shell":
            print(f"export KAIRO_MAMBA_FULL_MEMORY_RATIO={result['mamba_full_memory_ratio']}")
            print(f"export KAIRO_PROFILE_ID={result['profile']}")
        else:
            print(json.dumps(result, indent=2))
    elif args.command == "recommend-runtime":
        print(
            json.dumps(
                recommend_runtime(
                    args.model,
                    args.concurrency,
                    args.prompt_tokens,
                    context_tokens=args.context_tokens,
                    generation_tokens=args.generation_tokens,
                ),
                indent=2,
            )
        )
    elif args.command == "recommend-nvfp4":
        print(json.dumps(recommend_nvfp4_pipeline(args.m, args.n, args.k), indent=2))
    else:
        print(json.dumps(recommend_gemm(args.m, args.n, args.k), indent=2))


if __name__ == "__main__":
    main()
