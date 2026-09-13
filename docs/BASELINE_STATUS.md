# Baseline runtime status

Last verified in WSL Ubuntu 24.04 on the local RTX 5090:

| Component | State | Version / detail |
|---|---|---|
| GPU | usable | NVIDIA GeForce RTX 5090, 32,607 MiB |
| Driver | usable | 596.36 |
| CUDA compiler | usable for native probe | CUDA 13.0 (`/usr/local/cuda-13.0/bin/nvcc`) |
| PyTorch | GPU visible | 2.12.0+cu130; `torch.cuda.is_available()=True` |
| TensorRT-LLM | importable | 1.3.0rc25 |
| vLLM | CLI importable | 0.29.0 |
| SGLang | service smoke passed in isolated env | 0.5.19 |

## Fresh-model gate (2026-09-13)

The workbench's primary candidate is now `nvidia/Qwen3.8-27B-NVFP4`, with
`nvidia/Qwen3-8B-NVFP4` as the fast control. The Qwen3.8 checkpoint is fully
downloaded at `/home/peter/kairo-models/Qwen3.8-27B-NVFP4` (about 21 GiB).

- vLLM 0.29.0 resolved the `Qwen3_5ForConditionalGeneration` architecture,
  selected the GDN decode path and FlashInfer NVFP4 GEMM, and loaded all three
  safetensors shards in about 42 seconds using roughly 19 GiB of GPU memory.
- The service did **not** pass `/health`: initialization remained in the
  multimodal/FP8 autotune stage and the WSL GPU service became unresponsive.
  A second attempt with `--skip-mm-profiling --language-model-only`, 4K context,
  and 0.65 GPU utilization reproduced the stall. This is a runtime/initial-
  bring-up blocker, not evidence that the weights are invalid.
- The 8B NVFP4 control is also downloaded at
  `/home/peter/kairo-models/Qwen3-8B-NVFP4`; its first vLLM gate hit the same
  WSL failure before `/health`, so the next run must use an isolated/newer
  serving environment rather than mutate the working TensorRT stack.
- The checkpoint README itself recommends `vllm-openai:nightly` or the SGLang
  `dev` image (and shows a four-GPU GB300 reference deployment). Our pinned
  wheels are therefore a compatibility probe, not the vendor-supported path;
  the next bring-up should use a disposable nightly/dev container or source
  build, then return to the pinned wheel only for fair comparisons.
- A source checkout of SGLang main (`14b647c`) was tested in the isolated
  SGLang environment. With `--mamba-ssm-dtype bfloat16` and 0.80 static-memory
  fraction it loaded the weights, allocated 4.22 GiB of Mamba state plus a
  33,685-token FP8 KV pool, and reached "server fired up". The detokenizer then
  stopped heartbeating while compiling a first-use FlashInfer FP4 extension, so
  `/health` stayed 503 until the harness timeout. This narrows the next task to
  first-use kernel compilation/heartbeat handling rather than model loading or
  raw memory capacity.
- Building the missing `fp4_gemm_cutlass_sm120` target directly with Ninja
  reproduced a WSL service failure during the 18-kernel CUDA build. An
  experimental Marlin backend override did not clear the health gate, so this
  remains an explicit pending backend/compile experiment rather than a claimed
  workaround.

## Qwen3.8 first serving baseline

After selecting `flashinfer_cudnn` for FP4 GEMM, the source SGLang lane passed
the full smoke contract on the RTX 5090 (`/health` plus exact `KAIRO_OK`). The
following warm-cache decode observations used 256 requested prompt tokens,
64 generated tokens, FP8 KV cache, 4K context, Mamba bfloat16 state, and
`chat_template_kwargs.enable_thinking=false`:

| Concurrency | Requests | TTFT P50 | Aggregate output |
|---:|---:|---:|---:|
| 1 | 4 | 187 ms | 13.7 tok/s |
| 4 | 8 | 277 ms | 44.0 tok/s |

All requests succeeded. These are the first Qwen3.8 serving baselines, not a
Kairo win claim; the Mamba bfloat16 setting and SGLang-main build are recorded
as part of the configuration, and a pinned-runtime comparison is still needed.

A fixed-length decode repeat used the same runtime but added the top-level
`ignore_eos=true` request extension, so every successful request generated
exactly 256 tokens. The prompt target was 512 tokens (572 actual), with two
warmups and thinking disabled:

| Concurrency | Requests | TTFT P50 / P99 | Total P50 | Aggregate output |
|---:|---:|---:|---:|---:|
| 1 | 4 | 183 / 185 ms | 17.80 s | 14.30 tok/s |
| 4 | 8 | 367 / 885 ms | 18.75 s | 54.59 tok/s |
| 16 | 16 | 10,432 / 39,140 ms | 30.78 s | 68.37 tok/s |

The fixed-length pass confirms the earlier scaling direction while removing
early-EOS noise: four-way batching delivered 3.82x and c16 delivered 4.78x the
single-request output rate. At c16, however, TTFT P99 reached 39.1 s and the
samples arrived in visibly separated waves, making scheduler capacity/queueing
the first end-to-end optimization hypothesis. These are still baseline
observations; the next comparison should repeat the matrix across backends and
include longer prompt/prefill points.

As a first scheduler experiment, the same c16 run was repeated with
`KAIRO_MAX_RUNNING_REQUESTS=16` (the default was left untouched in the baseline
run). It produced 16/16 successes, 64.95 tok/s, TTFT P50/P99 of 22.36/45.02 s,
and a 63.06 s wall time. The result is slightly slower than the default
configuration (68.37 tok/s) and has worse queueing, so simply raising the
running-request cap is rejected as an optimization. The wave pattern points to
batch formation or decode scheduling as the next controlled variable.

A matching prefill probe (2,048 requested prompt tokens, 2,241 actual tokens,
one generated token, concurrency 1, four requests) produced TTFT P50 238 ms and
input throughput 9,080 tok/s. It is a warm-cache observation and should be
repeated before using it to select a prefill kernel target.

The fixed-length repeat (two warmups, thinking disabled, `ignore_eos=true`)
returned 4/4 successful requests with the same 2,241 actual input tokens,
TTFT P50/P99 of 227/259 ms, and 9,460 input tok/s. This is close enough to the
earlier reading to treat the prefill point as stable for the current runtime,
while still requiring cross-backend repetition before optimization claims.

For the vLLM comparison, forcing its `flashinfer_cudnn` linear backend selected
`FlashInferCudnnNvFp4LinearKernel` and reduced weight-load time to roughly 11 s,
but the Qwen3.8 engine still failed to reach `/health` during GDN/attention
initialization. The serving baseline above therefore uses SGLang main; vLLM is
kept as a pending runtime comparison rather than treated as a failed model.
- Until a newer pinned vLLM/SGLang environment clears this gate, the 0.5B model
  remains the CI canary and the 8B NVFP4 model is the reproducible performance
  control. No hero-model performance claim is made yet.

## Service smoke gate (Qwen2.5-0.5B-Instruct)

- **vLLM 0.29.0: passed** on the RTX 5090. With CUDA 13.0 selected and
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, the server reached `/health` and returned
  the exact deterministic response `KAIRO_OK` from `/v1/chat/completions`.
- **SGLang 0.5.19: passed in an isolated environment.** The server reached
  `/health` and returned `KAIRO_OK` from `/v1/chat/completions` using
  `/home/peter/venv-sglang` (PyTorch 2.13.0+cu130 and
  `sglang-kernel==0.4.6.post1`).
- The original shared environment remains unsuitable for SGLang: its
  `sgl_kernel` SM120 object was built against a different PyTorch C++ ABI
  (`undefined symbol: c10::ValueError...`). Keep it reserved for TensorRT-LLM
  and the already-validated vLLM lane.

## Important qualification

vLLM and SGLang were installed into the existing `/home/peter/venv-gpu`
environment without dependency resolution so that the TensorRT CUDA 13 stack
would not be replaced. Their package metadata requests a newer/different set of
versions (notably torch 2.13.0 and framework-specific CUDA wheels). Therefore:

- the service smoke result is valid only for the explicitly recorded backend
  environment and launch flags;
- no optimization or performance-win claim is valid yet; measurements below are
  baseline observations only;
- the next gate is a pinned decode/prefill baseline with repeated samples;
- if either backend fails, create a dedicated pinned environment instead of
  mutating `venv-gpu` further.

The smoke harness accepts `KAIRO_VLLM_BIN` and `KAIRO_SGLANG_PYTHON`, so an
isolated backend environment can be tested without changing the TensorRT
environment.

The old Moonmath records remain useful historical evidence: vLLM 0.26.0 and
SGLang 0.5.17 were previously run on this same 5090, with WSL-specific pin-memory
and CUDA Graph settings.

## First decode measurement (smoke workload)

The dependency-free `scripts/wsl/bench_openai.py` was run with 512 requested
prompt tokens, 64 generated tokens, one warmup, four requests, and concurrency
1. The server usage reported 587 input tokens for each request:

| Backend | TTFT P50 / P99 (ms) | Total P50 (ms) | Output tok/s |
|---|---:|---:|---:|
| vLLM 0.29.0 | 19.5 / 21.8 | 491.6 | 129.0 |
| SGLang 0.5.19 | 22.6 / 181.3 | 469.0 | 125.6 |

These are wiring and reproducibility checks on a 0.5B model, not the Kairo
result. The next run expands concurrency and prompt lengths under the v0
protocol before selecting a kernel hotspot.

An additional concurrency-4 pass (same prompt/generation sizes, eight requests)
already exposes a useful direction signal:

| Backend | TTFT P50 / P99 (ms) | Total P50 (ms) | Aggregate output tok/s |
|---|---:|---:|---:|
| vLLM 0.29.0 | 25.9 / 27.7 | 497.2 | 512.4 |
| SGLang 0.5.19 | 301.0 / 576.7 | 879.4 | 290.4 |

This is still a tiny smoke workload, but the 4-way queueing gap is large enough
to justify profiling scheduler/batching and decode-kernel behavior next. It is
not yet a claim of a Kairo win: both backends need the same longer matrix,
steady-state sampling, and correctness checks.

The protocol's concurrency-16 point (16 requests) strengthens that signal:

| Backend | TTFT P50 / P99 (ms) | Total P50 (ms) | Aggregate output tok/s |
|---|---:|---:|---:|
| vLLM 0.29.0 | 40.8 / 45.3 | 595.1 | 1,691.1 |
| SGLang 0.5.19 | 147.9 / 149.8 | 784.5 | 1,296.9 |

At this workload vLLM is about 30% ahead in aggregate decode throughput and has
roughly 3.6x lower TTFT P50. Treat this as a hypothesis-generating baseline;
the next experiment is to reproduce it with longer generations and profiler
traces before touching kernels.

## First prefill measurement (4K class prompt)

With the per-request nonce enabled to avoid prefix-cache reuse, the runner used
4,484 actual input tokens, one generated token, four requests, and concurrency
1. Services were configured for a 16K context window:

| Backend | TTFT P50 / P99 (ms) | Input tok/s |
|---|---:|---:|
| vLLM 0.29.0 | 38.6 / 40.3 | 112,448 |
| SGLang 0.5.19 | 61.6 / 119.9 | 58,170 |

The roughly 1.9x prefill gap makes prefill scheduling/attention a second strong
profiling candidate. These numbers are still smoke-scale and require repeated
trials plus profiler traces before a target is selected.

Note: the earlier decode tables were collected before the per-request nonce was
added. They remain useful for queueing direction, but formal decode comparisons
must be rerun with the nonce-enabled runner and repeated trials.

## First operator profiles

Nsight Systems is installed, but its service trace did not contain child-worker
CUDA kernels. Nsight Compute is blocked by the host policy
`ERR_NVGPUCTRPERM` (GPU performance counters are not exposed to this WSL user).
The in-process fallback `scripts/wsl/profile_transformers.py` does capture CUDA
operators without those counters:

- 4K-class prefill, two forwards: `aten::mm`/CUTLASS accounted for about 63%
  of self CUDA time; FlashAttention accounted for about 18%.
- 512-token KV-cache decode, 16 steps: GEMV/GEMM kernels accounted for about
  74% combined; FlashAttention was about 8.5%.

These are direct Transformers profiles rather than vLLM/SGLang worker traces,
so they define kernel hypotheses—not proof of the serving-runtime bottleneck.
The next experiment should use the same shapes in an in-process backend probe,
or enable GPU performance counters, before implementing a custom kernel.
