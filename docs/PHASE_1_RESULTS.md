# Kairo Phase 1 results

Kairo Phase 1 establishes a reproducible measurement and decision layer for
consumer-Blackwell LLM inference. The numbers below are local measurements on
one RTX 5090 32GB under the pinned software and model revisions in the linked
protocols. They are not universal hardware, framework, quality, or production
cost claims.

## What Phase 1 delivered

| Capability | Evidence | Current conclusion |
|---|---|---|
| Exact-workload runtime routing | [`qwen38-vllm-cudagraph-serving.yaml`](../experiments/protocols/qwen38-vllm-cudagraph-serving.yaml) | Measured routes are executable; uncovered routes fail closed to `manual`. |
| Cross-model Graph control | [`qwen3-8b-vllm-cudagraph-serving.yaml`](../experiments/protocols/qwen3-8b-vllm-cudagraph-serving.yaml) | The Graph result transfers to a smaller NVFP4 control model under measured buckets. |
| Native SM120 movement path | [`tma-copy-phase0.yaml`](../experiments/protocols/tma-copy-phase0.yaml) | A real 2D TMA copy is correct on the local RTX 5090. |
| Native matrix-compute baseline | [`tma-wmma-gemm-phase1.yaml`](../experiments/protocols/tma-wmma-gemm-phase1.yaml) | TMA+WMMA GEMM is correct, but not competitive with cuBLAS. |
| Rejected layout experiment | [`tma-wmma-swizzle-probe.yaml`](../experiments/protocols/tma-wmma-swizzle-probe.yaml) | TMA swizzle requires a consumer layout that matches the producer. |

## Headline serving results

All Graph/eager pairs below use fixed-generation workloads and pass the stated
deterministic correctness gate. Consult the protocol before comparing values:
context, concurrency, request count, source revision, and repeat policy are
part of each result.

| Model and measured workload | Graph | Eager | Result |
|---|---:|---:|---:|
| Qwen3.8-27B-NVFP4, c32, prompt 512, context 1K, generation 128 | 645.08 tok/s | 308.51 tok/s | 2.09x |
| Qwen3.8-27B-NVFP4, c8, prompt 2048, context 4K, generation 128 | 322.78 tok/s | 123.22 tok/s | 2.62x |
| Qwen3.8-27B-NVFP4, c16, prompt 2048, context 4K, generation 128 | 260.07 tok/s | 199.52 tok/s | 1.30x |
| Qwen3-8B-NVFP4, c16, prompt 512, context 1K, generation 128 | 2177.49 tok/s | 1073.63 tok/s | 2.03x |

The 1.30x Qwen3.8 result is intentional context: Graph is not promoted as a
global default. Kairo promotes only exact measured workload buckets.

## What Phase 1 does not claim

- CUDA Graphs are not a Kairo invention.
- These results do not establish a universal “enable Graphs” rule.
- The current TMA+WMMA experiment does not beat cuBLAS.
- Four deterministic checks are a serving-semantic gate, not a model-quality
  evaluation or full token-level equivalence proof.

## Phase 2 direction

Phase 2 will use Phase 1's measurement and routing layer to select narrowly
scoped native components—data movement, layouts, subgraphs, or execution
paths—and will promote them only after correctness, boundary, repeatability,
and fair end-to-end comparison gates pass.
