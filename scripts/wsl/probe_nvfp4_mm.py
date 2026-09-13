#!/usr/bin/env python3
"""Benchmark the installed vLLM CUTLASS NVFP4 path on the local GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time


def _prepare_import_path() -> None:
    # The isolated nightly environment intentionally omits a few pure-Python
    # packages that are present in the shared environment. Keep nightly torch
    # first so its CUDA extension ABI remains authoritative.
    sys.path.insert(0, "/home/peter/venv-vllm-nightly/lib/python3.12/site-packages")
    sys.path.append("/home/peter/venv-gpu/lib/python3.12/site-packages")


def run(m: int, n: int, k: int, iterations: int, warmups: int) -> dict[str, object]:
    _prepare_import_path()
    import torch
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        cutlass_fp4_supported,
        swizzle_blockscale,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if n % 32 or k % 32:
        raise ValueError("NVFP4 CUTLASS probe requires N and K divisible by 32")
    capability = torch.cuda.get_device_capability()
    if not cutlass_fp4_supported():
        raise RuntimeError(f"CUTLASS FP4 is not supported on capability {capability}")
    device = torch.device("cuda")
    torch.manual_seed(5090)
    x = torch.randn((m, k), device=device, dtype=torch.float16)
    weight = torch.randn((n, k), device=device, dtype=torch.float16)
    global_scale = torch.ones((), device=device, dtype=torch.float32)
    x_fp4, x_scale = scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=True, backend="cutlass"
    )
    weight_fp4, weight_scale_raw = scaled_fp4_quant(
        weight, global_scale, is_sf_swizzled_layout=False, backend="cutlass"
    )
    weight_scale = swizzle_blockscale(weight_scale_raw)
    alpha = torch.ones((), device=device, dtype=torch.float32)
    reference = x.float() @ weight.float().T

    def invoke() -> torch.Tensor:
        return cutlass_scaled_fp4_mm(
            x_fp4,
            weight_fp4,
            x_scale,
            weight_scale,
            alpha,
            torch.float16,
        )

    for _ in range(warmups):
        invoke()
    torch.cuda.synchronize()
    output = invoke().float()
    torch.cuda.synchronize()
    error = (output - reference).abs()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        invoke()
    stop.record()
    stop.synchronize()
    elapsed_ms = start.elapsed_time(stop) / iterations
    operations = 2.0 * m * n * k
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(capability),
        "shape": [m, n, k],
        "iterations": iterations,
        "variant": "vllm_cutlass_nvfp4",
        "fp4_supported": True,
        "input_packed_shape": list(x_fp4.shape),
        "weight_packed_shape": list(weight_fp4.shape),
        "output_dtype": "float16",
        "mm_ms": elapsed_ms,
        "gflops": operations / (elapsed_ms * 1.0e6),
        "max_abs_error_vs_fp16": float(error.max().item()),
        "mean_abs_error_vs_fp16": float(error.mean().item()),
        "relative_mean_error_vs_fp16": float((error.mean() / reference.abs().mean()).item()),
        "finite": bool(torch.isfinite(output).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.m, args.n, args.k, args.iterations, args.warmups)))


if __name__ == "__main__":
    main()
