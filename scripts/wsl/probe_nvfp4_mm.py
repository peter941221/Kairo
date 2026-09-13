#!/usr/bin/env python3
"""Benchmark the installed vLLM CUTLASS NVFP4 path on the local GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time


def _prepare_import_path(backend: str) -> None:
    # The isolated nightly environment intentionally omits a few pure-Python
    # packages that are present in the shared environment. Keep nightly torch
    # first so its CUDA extension ABI remains authoritative.
    sys.path.insert(0, "/home/peter/venv-vllm-nightly/lib/python3.12/site-packages")
    if backend == "b12x":
        sys.path.insert(
            1,
            "/home/peter/venv-vllm-nightly/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages",
        )
    sys.path.append("/home/peter/venv-gpu/lib/python3.12/site-packages")


def run(
    m: int,
    n: int,
    k: int,
    iterations: int,
    warmups: int,
    backend: str,
    control_first: bool,
) -> dict[str, object]:
    _prepare_import_path(backend)
    import torch
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import cutlass_fp4_supported
    if backend == "b12x":
        from vllm.utils.b12x import get_b12x_blockscaled, get_b12x_intrinsics
        blockscaled = get_b12x_blockscaled()
        intrinsics = get_b12x_intrinsics()
        if blockscaled is None or intrinsics is None or not blockscaled.is_supported():
            raise RuntimeError("B12X NVFP4 backend is unavailable")
    else:
        from vllm._custom_ops import cutlass_scaled_fp4_mm
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

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
        x,
        global_scale,
        is_sf_swizzled_layout=True,
        backend="cutlass" if backend == "cutlass" else "none",
    )
    weight_fp4, weight_scale_raw = scaled_fp4_quant(
        weight, global_scale, is_sf_swizzled_layout=False, backend="cutlass"
    )
    weight_scale = (
        intrinsics.swizzle_block_scale(weight_scale_raw)
        if backend == "b12x"
        else swizzle_blockscale(weight_scale_raw)
    )
    alpha = torch.ones((), device=device, dtype=torch.float32)
    reference = x.float() @ weight.float().T
    torch.cuda.synchronize()

    def invoke() -> torch.Tensor:
        if backend == "b12x":
            return blockscaled.mm_nvfp4(
                x_fp4, x_scale, weight_fp4, weight_scale, alpha, out_dtype=torch.float16
            )
        return cutlass_scaled_fp4_mm(x_fp4, weight_fp4, x_scale, weight_scale, alpha, torch.float16)

    def invoke_fp16() -> torch.Tensor:
        return torch.mm(x, weight.T)

    def invoke_pipeline() -> torch.Tensor:
        dynamic_fp4, dynamic_scale = scaled_fp4_quant(
            x,
            global_scale,
            is_sf_swizzled_layout=True,
            backend="cutlass" if backend == "cutlass" else "none",
        )
        if backend == "b12x":
            return blockscaled.mm_nvfp4(
                dynamic_fp4,
                dynamic_scale,
                weight_fp4,
                weight_scale,
                alpha,
                out_dtype=torch.float16,
            )
        return cutlass_scaled_fp4_mm(
            dynamic_fp4, weight_fp4, dynamic_scale, weight_scale, alpha, torch.float16
        )

    def measure(function) -> float:
        for _ in range(warmups):
            function()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) / iterations

    if control_first:
        fp16_elapsed_ms = measure(invoke_fp16)
        output = invoke().float()
        torch.cuda.synchronize()
        error = (output - reference).abs()
        elapsed_ms = measure(invoke)
    else:
        output = invoke().float()
        torch.cuda.synchronize()
        error = (output - reference).abs()
        elapsed_ms = measure(invoke)
        fp16_elapsed_ms = measure(invoke_fp16)
    pipeline_elapsed_ms = measure(invoke_pipeline)
    operations = 2.0 * m * n * k
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(capability),
        "shape": [m, n, k],
        "iterations": iterations,
        "variant": f"vllm_{backend}_nvfp4",
        "measurement_order": "fp16_then_nvfp4" if control_first else "nvfp4_then_fp16",
        "fp4_supported": True,
        "input_packed_shape": list(x_fp4.shape),
        "weight_packed_shape": list(weight_fp4.shape),
        "output_dtype": "float16",
        "mm_ms": elapsed_ms,
        "gflops": operations / (elapsed_ms * 1.0e6),
        "fp16_mm_ms": fp16_elapsed_ms,
        "fp16_gflops": operations / (fp16_elapsed_ms * 1.0e6),
        "speedup_vs_fp16": fp16_elapsed_ms / elapsed_ms,
        "quantized_pipeline_ms": pipeline_elapsed_ms,
        "quantized_pipeline_gflops": operations / (pipeline_elapsed_ms * 1.0e6),
        "quantized_pipeline_speedup_vs_fp16": fp16_elapsed_ms / pipeline_elapsed_ms,
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
    parser.add_argument("--backend", choices=["cutlass", "b12x"], default="cutlass")
    parser.add_argument("--control-first", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.m,
                args.n,
                args.k,
                args.iterations,
                args.warmups,
                args.backend,
                args.control_first,
            )
        )
    )


if __name__ == "__main__":
    main()
