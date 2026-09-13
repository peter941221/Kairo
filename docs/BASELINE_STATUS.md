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
