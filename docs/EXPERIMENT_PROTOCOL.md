# Kairo experiment protocol v0

## Purpose

Every result must answer: *under which model, quantization, workload, hardware,
software revision, and correctness tolerance did this approach win?*

## Required controls

1. Pin the model revision, tokenizer, quantization recipe, framework/container,
   NVIDIA driver, CUDA version, and Kairo Git revision.
2. Compare the same model weights, prompt set, output length, concurrency,
   precision, warm-up policy, and CUDA Graph policy.
3. Report compilation/build time separately from steady-state latency.
4. Run warm-up before measurement; report at least P50 and P99, not only a mean.
5. Run numerical checks against the declared reference before performance claims.
6. Retain raw result JSON locally and commit the aggregation tables and command
   lines used for any public claim.
7. For throughput comparisons, request a fixed generation length with the
   runtime's `ignore_eos` extension (or an equivalent control) and record the
   actual usage token counts.

## Initial matrix

| Lane | Concurrency | Prompt tokens | Generated tokens | Primary measures |
|---|---:|---:|---:|---|
| Decode | 1, 4, 16 | 512 | 256 | tokens/s, P50/P99 ITL, memory |
| Prefill | 1, 4 | 4K, 16K | 1 | TTFT, tokens/s, memory |

The matrix is intentionally small. Expand it only after a baseline result shows
an actionable hotspot.

## Baselines

The first report should pin and benchmark whichever of TensorRT-LLM, SGLang, and
vLLM can execute the exact workload. For kernel-level work, also record the
cuBLASLt/CUTLASS implementation and settings when accessible.

## Promotion gate

A candidate enters kernel development only when all are true:

- it is at least 15% faster in an isolated operator or subgraph experiment;
- its parent hotspot has enough runtime share for a 20% end-to-end result;
- its result survives correctness, non-aligned-shape, and repeated-run checks;
- the mechanism is explainable (fusion, layout, scale handling, scheduling, or
  data movement), rather than an unfair protocol difference.
