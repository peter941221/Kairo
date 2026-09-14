STATUS: DRAFT. Phase-one launch article.
This replaces the earlier CUDA-Graph-only outline. Graph routing is phase-one
evidence, not Kairo's long-term identity. Do not present these local results as
universal claims.

# Building Kairo: From Reproducible Measurements to Native Blackwell Inference

One CUDA Graph setting more than doubled serving throughput on a single RTX
5090. That was not the interesting result.

The same setting helped much less under another valid workload. We used that
discrepancy to build Kairo: a workbench that records why an inference choice
won, applies it only where it was measured, and leaves unmeasured requests
alone.

Each result enabled the next: controlled benchmark -> Graph boundary ->
executable routing policy -> native-kernel exploration.

## 1. A fast benchmark was not yet a result we could trust

LLM-serving comparisons are easy to accidentally make unfair. A benchmark can
change model revisions, actual token counts, warm-up policy, concurrency,
context window, Graph mode, or whether the server answers correctly. A large
throughput number alone cannot say which variable caused it.

Kairo began with a narrower rule: each result states the model and weight
revision, GPU, driver, CUDA and runtime revisions, workload, correctness gate,
repetitions, command, and raw-artifact identity. Both sides request a fixed
output length so early EOS cannot manufacture a win.

This is deliberately unglamorous. It is also what allows a later decision to
be reviewed instead of remembered as a configuration anecdote.

First conclusion: a benchmark is an input to an optimization decision only
after its controls are explicit.

## 2. A controlled comparison exposed the CUDA Graph boundary

We served the pinned `nvidia/Qwen3.8-27B-NVFP4` checkpoint on one 32 GB RTX
5090 through the same vLLM-nightly/CUTLASS configuration. The paired runs used
the same model revision, prompt workload, actual token count, concurrency,
request count, warm-up policy, FP8 KV cache, and fixed 128-token generation.
The intended variable was `FULL_DECODE_ONLY` CUDA Graph replay versus eager
execution. Both paths passed four deterministic semantic checks.

The short-context result was strong:

```text
27B NVFP4, c32, prompt 512, context 1K, generation 128
CUDA Graph:  645.08 tok/s median
Eager:       308.51 tok/s median
Result:      2.09x
```

The long-context result was also strong:

```text
27B NVFP4, c8, prompt 2048, context 4K, generation 128
CUDA Graph:  322.78 tok/s median
Eager:       123.22 tok/s median
Result:      2.62x
```

But a third point changed the interpretation:

```text
27B NVFP4, c16, prompt 2048, context 4K, generation 128
CUDA Graph:  260.07 tok/s median
Eager:       199.52 tok/s median
Result:      1.30x
```

CUDA Graphs are not a newly discovered technique. The useful result is that
their value on this consumer-Blackwell serving path depends on the workload
bucket.

Second conclusion: a setting can be clearly beneficial without being a safe
global default.

<!-- RELEASE TODO: Replace blocks with a generated table containing every
sample, P50/P99, stability summary, and links to public JSONL. -->

## 3. The benchmark boundary became a routing policy

Most benchmark reports end with “enable Graphs.” That discards the conditions
under which the claim was true.

Kairo stores measured winners as exact routes. A request is matched on model,
workload, context envelope, sequence limits, and runtime profile. A known
matching route enables its measured Graph or eager profile. An unknown route
returns `manual`; it does not silently inherit the nearest-looking win.

The routing launcher was tested end to end. On the measured Qwen3.8 c32 /
prompt-512 bucket, it selected Graph automatically, passed 4/4 correctness,
completed all 32 requests, and recorded 665.76 tok/s. A separate 4K-context
request selected its eager route and also completed with correctness enabled.
The policy is an executable constraint, not a table in a report.

We repeated the core observation on `nvidia/Qwen3-8B-NVFP4`. At c16 / prompt
512, Graph reached 2177.49 tok/s median versus 1073.63 eager, a 2.03x result.
At c32 it reached 3946.07 versus 1835.39 tok/s, or 2.15x. This does not prove
every NVFP4 model behaves this way. It shows the result was not one checkpoint
and one accidental run.

Third conclusion: the deliverable is not a Graph flag. It is a policy that can
explain, reproduce, and decline an inference decision.

## 4. The policy made clear where configuration tuning ends

Route selection cannot repair every bottleneck. To see what a native component
would require, we began with small, independently checkable CUDA experiments
on the same RTX 5090.

The local CUDA 13.0 toolchain can execute an asynchronous copy on SM120. A real
two-dimensional TMA transfer using a host-created Tensor Map and shared-memory
barrier returned a tile with zero maximum absolute error. The same environment
did not expose a usable `tcgen05` path, so Kairo does not assume every
Blackwell instruction is available merely because the GPU name is Blackwell.

We then moved from a scalar tiled FP16 GEMM, through WMMA, to TMA+WMMA. The
integrated path was correct against cuBLAS and reached about 35.2 TFLOP/s at a
1K square shape and 41.6 TFLOP/s at 4K. Larger block candidates reached roughly
49.9 TFLOP/s at selected 4K shapes.

These are capability results, not a replacement claim: cuBLAS remains far
faster. They establish the data-movement and matrix-compute path on which a
future Kairo component can be built.

Fourth conclusion: native work starts with a verified boundary, not with a
claim to have beaten a mature library.

## 5. One failed swizzle identified the real constraint

The most informative kernel result was a rejection. An experimental
32-byte-swizzle TMA variant compiled and ran, but its 1K output differed from
cuBLAS by a maximum absolute error of 18.52.

The failure did not show that TMA swizzle is unavailable. It exposed a layout
contract: the TMA producer wrote swizzled shared memory while the existing WMMA
consumer assumed row-major data. Asynchronous transfer, shared-memory layout,
and tensor-core loads have to be designed together.

Kairo retains that failed result in its protocol. It prevents the same invalid
optimization from being rediscovered and gives the next implementation a clear
requirement: provide a matching swizzled consumer mapping.

Fifth conclusion: an optimization that passes compilation but fails its layout
contract is not an unfinished speedup; it is a rejected design.

## 6. What Kairo builds next

Phase one establishes a measurement and decision layer for consumer Blackwell
inference: controlled protocols, correctness-aware serving comparisons,
fail-closed routing, native capability probes, and explicit negative results.

Phase two begins where that layer identifies persistent gaps. The goal is not a
generic compiler announced in advance. It is to build and validate narrowly
scoped native components—data movement, layouts, subgraphs, or execution paths
—when evidence shows they can improve end-to-end inference under the same
fairness rules.

That makes the roadmap falsifiable. If a native candidate does not pass
correctness, boundary coverage, repeatability, and a fair baseline comparison,
it remains an experiment rather than becoming Kairo's default.

Final conclusion: Kairo is moving from reproducible measurements toward native
Blackwell inference, one verified boundary at a time.

## Reproduce and inspect

Kairo's repository contains experiment protocols, portable contract tests,
serving runners, route selection logic, and CUDA probes.

<!-- RELEASE BLOCKER: Before public publication, add a public results bundle:
raw/normalized JSONL, environment locks, chart-generation script, exact source
commit, and artifact hashes. Link every numerical claim above to it. -->

## Sources and scope

This article reports local measurements on one RTX 5090 / WSL2 system under
recorded software revisions. It is not a claim about all RTX 5090 machines,
all vLLM versions, all NVFP4 checkpoints, model quality, or production cost.

<!-- RELEASE TODO: List Kairo protocol files, public data artifacts, vLLM and
CUDA Graph documentation, plus CUDA/TMA references. -->
