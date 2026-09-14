---
layout: default
title: "Building Kairo: From Measured Routes to Native Blackwell Components"
description: "Measured CUDA Graph routing and native Blackwell baselines on an RTX 5090."
---

# Building Kairo: From Measured Routes to Native Blackwell Components

Kairo is an RTX 5090 inference workbench. Its current output is a strict
runtime policy: enable a serving profile only for an exact workload that has
passed a paired benchmark and correctness gate. Unknown workloads return
`manual`.

This article records the Phase 1 evidence and the boundary for Phase 2 native
work.

## 1. Serving result

Paired runs used the same pinned model revision, vLLM nightly, CUTLASS linear
backend, FP8 KV cache, prompt workload, actual token count, concurrency,
request count, warm-up, and fixed 128-token generation. The only intended
variable was `FULL_DECODE_ONLY` CUDA Graph replay versus eager execution.
Every listed route passed 4/4 deterministic semantic checks.

```text
+----------------------+-------+--------+--------+--------+--------+-------+
| model                | conc. | prompt | context| graph  | eager  | ratio |
+----------------------+-------+--------+--------+--------+--------+-------+
| Qwen3.8-27B-NVFP4   | 32    | 512    | 1K     | 645.08 | 308.51 | 2.09x |
| Qwen3.8-27B-NVFP4   |  8    | 2048   | 4K     | 322.78 | 123.22 | 2.62x |
| Qwen3.8-27B-NVFP4   | 16    | 2048   | 4K     | 260.07 | 199.52 | 1.30x |
| Qwen3-8B-NVFP4      | 16    | 512    | 1K     |2177.49 |1073.63 | 2.03x |
+----------------------+-------+--------+--------+--------+--------+-------+
                         throughput: output tokens / second
```

The 1.30x cell prevents a global “enable Graphs” rule. The policy stores exact
model, prompt, context, generation, concurrency, and sequence-limit bounds.

```text
request
  -> exact measured route: launch recorded Graph/eager profile
  -> no exact route:       return manual; do not extrapolate
```

The routed launcher was exercised on the 27B c32 / prompt-512 / 1K bucket. It
selected Graph, passed 4/4 checks, completed 32/32 requests, and measured
665.76 tok/s. A 4K request selected its separate eager profile.

Per-repeat public evidence, derived table, SVG, source protocols, and runners
are in the repository:

```text
evidence/phase1/serving-benchmark-repeats.jsonl
docs/generated/phase1-serving-results.{md,svg}
experiments/protocols/qwen38-vllm-cudagraph-serving.yaml
scripts/wsl/run_routed_bench.sh
```

## 2. What the benchmark enforces

```text
+-----------------------+------------------------------------------------------+
| control               | requirement                                          |
+-----------------------+------------------------------------------------------+
| model identity        | pinned checkpoint revision                           |
| workload              | requested and actual input/output token counts       |
| comparison            | same runtime, cache format, warm-up, and request mix |
| correctness           | deterministic semantic gate before speed claim       |
| publication           | protocol, per-repeat evidence, and render script     |
+-----------------------+------------------------------------------------------+
```

This is not a model-quality evaluation or a universal RTX 5090 result. It is a
reproducible result for the recorded hardware and software configuration.

## 3. Native baseline

The same machine was used to establish an SM120 CUDA baseline before claiming a
custom execution path.

```text
+----------------------------+-------------------------------+----------------------+
| experiment                 | result                        | decision             |
+----------------------------+-------------------------------+----------------------+
| 2D TMA copy                | max error 0                   | supported            |
| tcgen05 probe              | unavailable in current stack  | do not depend on it  |
| TMA + WMMA, 1K square      | 35.2 TFLOP/s, correct         | baseline only        |
| TMA + WMMA, 4K square      | 41.6 TFLOP/s, correct         | baseline only        |
| m256 swizzle, 1K square    | max error 18.52               | rejected             |
+----------------------------+-------------------------------+----------------------+
```

The fastest measured block candidate reached 49.9 TFLOP/s at a selected 4K
shape. cuBLAS remained substantially faster. The swizzle failure identified a
specific requirement for Phase 2: TMA's shared-memory layout must be paired
with a compatible WMMA consumer layout.

## 4. Phase 2

Phase 2 targets narrow native components selected by measured bottlenecks:
data movement, shared-memory layout, subgraphs, or execution paths. A component
is not promoted without numerical correctness, boundary coverage, repeated
measurements, and a fair end-to-end baseline.

## Reproduce

```text
PYTHONPATH=src python -m unittest discover -s tests -q
python scripts/render_phase1_results.py
bash scripts/wsl/run_lab.sh recommend-runtime \
  --model qwen38 --concurrency 32 --prompt-tokens 512 \
  --context-tokens 1024 --generation-tokens 128
```

Local model and environment paths are configured through `KAIRO_*` variables.
The repository does not contain model weights, local logs, credentials, or
private machine notes.
