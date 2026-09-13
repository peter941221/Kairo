# Baseline runtime status

Last verified in WSL Ubuntu 24.04 on the local RTX 5090:

| Component | State | Version / detail |
|---|---|---|
| GPU | usable | NVIDIA GeForce RTX 5090, 32,607 MiB |
| Driver | usable | 596.36 |
| CUDA compiler | usable for native probe | CUDA 13.0 (`/usr/local/cuda-13.0/bin/nvcc`) |
| PyTorch | GPU visible | 2.12.0+cu130; `torch.cuda.is_available()=True` |
| TensorRT-LLM | importable | 1.3.0rc25 |
| vLLM | CLI importable | 0.29.0 (pinned) |
| vLLM nightly + B12X | service smoke passed in isolated env | 0.29.1rc1.dev18 + Torch 2.15 nightly |
| SGLang | service smoke passed in isolated env | 0.5.19 |

## Phase 0 hardware capability probe

`scripts/wsl/probe_capabilities.sh` now compiles and runs a minimal CUDA
data-movement closure with the CUDA 13.0 toolkit and `-arch=sm_120`. The probe
reported:

```json
{"gpu":"NVIDIA GeForce RTX 5090","compute_capability":"12.0","runtime_version":13000,"compiled_arch":"sm_120","async_copy_checksum":32896,"expected_checksum":32896,"async_copy_ok":true}
```

This proves the local toolchain can target the physical GPU and that an
asynchronous shared-memory copy executes correctly on it. It is intentionally
not a claim that every TMA/WGMMA/FP4 instruction is available; those remain
separate probes. The raw report is kept in the ignored
`.kairo-local/capability-probe.json` file so hardware facts do not get mixed
with source-controlled result tables.

The companion `scripts/wsl/probe_tcgen05.sh` tested the CUDA 13.0 CCCL
`tcgen05.fence` entry point with `-arch=sm_120`. It returns
`supported:false`: `ptxas` resolves the generated
`not_supported_before_SM_100a_SM_101a` stub and fails the link. This is an
actionable SM120 boundary, not a missing-header problem. Kairo therefore keeps
WMMA/cp.async as the RTX 5090 fallback and does not claim a `tcgen05` kernel
until a toolchain and hardware path explicitly support SM120.

In contrast, `scripts/wsl/run_tma_copy_probe.sh` successfully executed a real
2D `cp.async.bulk.tensor` transfer on the same SM120 GPU. It uses a host-created
`CUtensorMap`, a shared `mbarrier`, and a 16x16 FP32 tile; the returned tile had
maximum absolute error `0.0`. This separates the capability decision cleanly:
`tcgen05` is unavailable through the current SM120 path, while TMA is available
and ready to be integrated with the WMMA GEMM. The reproducible probe is
captured in `experiments/protocols/tma-copy-phase0.yaml` and the raw result in
`.kairo-local/tma-copy-probe.json`.

## Phase 1 FP16 GEMM closure

The first Kairo-owned tiled GEMM is implemented in
`scripts/wsl/fp16_gemm_probe.cu` and built for `sm_120` by
`scripts/wsl/run_fp16_gemm_probe.sh`. It handles non-multiple dimensions and
accumulates in FP32. On `[128,131,113]` it matched cuBLAS within `1.43e-6`
maximum absolute error. On `[1024,1024,1024]`, the initial 16x16/1x1 tile
reached 9,121 GFLOP/s; a 32x32 tile with 2x2 output reuse reached 22,515
GFLOP/s, a 2.47x improvement. cuBLAS measured 117,397 GFLOP/s on the same
run, so the optimized reference is still 5.2x behind and remains a target for
asynchronous staging and tensor-core instructions. Reproducible parameters are
recorded in `experiments/protocols/fp16-gemm-phase1.yaml`.

The `tile32x32_output2x2_async` variant uses 4-byte
`__pipeline_memcpy_async` copies and also passes the odd-boundary case. Its
1K result was 22,705 GFLOP/s versus 22,515 GFLOP/s for the synchronous variant,
within run-to-run noise. We therefore reject async copy alone as the current
performance lever and will reserve the shared-memory pipeline for a
tensor-core MMA implementation.

The controlled `wmma_fp16` variant now uses one warp per 16x16 output tile and
passes the aligned 1K shape with a maximum absolute error of `3.43e-5` versus
cuBLAS. Three 50-iteration runs measured 33,253--33,330 GFLOP/s, roughly 3.4x
the tiled scalar reference. It still trails cuBLAS, but this is the first
Kairo-owned tensor-core result and provides a credible starting point for
Blackwell-specific WGMMA/TMA work. Non-16-divisible shapes are explicitly
rejected by this variant and must use the fallback path.

At the larger `[4096,4096,4096]` shape, ten iterations measured 27,376
GFLOP/s versus 209,516 GFLOP/s for cuBLAS. The probe skips the impractical
O(MNK) CPU reference at this size and checks the full output against cuBLAS
instead (maximum absolute error `0.0` in this run). The scale-up result keeps
the optimization target honest: a tensor-core API alone is not enough; tile
occupancy, shared-memory movement, and Blackwell-native instruction selection
must be addressed.

The follow-up `wmma_fp16_shared` variant stages one 64x16 A tile and one 16x16
B tile per block so four warps reuse B. It reached 34,307--34,475 GFLOP/s on
three 4K runs (about +25% over direct WMMA), while measuring 31,632 GFLOP/s at
1K (about 5% slower). Kairo therefore needs a shape-aware tile/staging policy;
one kernel configuration is not optimal across the matrix.

The first integrated TMA+WMMA GEMM is now in
`scripts/wsl/tma_wmma_gemm_probe.cu`. It uses two host-created Tensor Maps,
one transaction-counted mbarrier, and a 64x16 block tile. The 1K result is
35,234 GFLOP/s and the 4K result is 41,612 GFLOP/s, both with zero maximum
absolute error against cuBLAS on the checked shapes. Relative to direct WMMA,
TMA improves the 4K point by roughly 1.5x; cuBLAS remains faster, so the next
experiments are double-buffered TMA and layout/swizzle tuning. Parameters are
recorded in `experiments/protocols/tma-wmma-gemm-phase1.yaml`.

Run it directly on the WSL 5090:

```bash
bash scripts/wsl/run_fp16_gemm_probe.sh 1024 1024 1024 50
```

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

### Isolated vLLM nightly probe

The official CUDA 13.0 nightly index exposed
`vllm-0.29.1rc1.dev18+gfa1b3b192`, which was installed into the isolated
`/home/peter/venv-vllm-nightly` without changing the shared environments. The
wheel imports only with a newer Torch ABI: using shared Torch 2.12.0 or
isolated SGLang Torch 2.13.0 fails inside `torch._inductor` (a `CSE` generic
signature mismatch). A matching CUDA 13.0 Torch 2.15 nightly wheel was
installed in the same isolated environment, together with the compatible
Transformers/FastAPI stack. The nightly engine resolved Qwen3.8, loaded all
three shards (19.08 GiB GPU memory), and selected the expected NVFP4/GDN
paths.

The first serving attempt then exposed a concrete SM120 dependency boundary:
the default FP8 scaled-MM path asks shared FlashInfer to JIT-build a
`sm120` extension during KV-cache profiling. That path is slow and can stall
the WSL service. The isolated environment now includes the optional B12X
backend and its CUDA 13 CUTLASS DSL stack; with
`--linear-backend b12x`, the nightly engine reaches `/health` and returns the
exact `KAIRO_OK` smoke response.

The isolated launcher is kept in `scripts/wsl/vllm_nightly.py`; it makes the
nightly vLLM/Torch site win while reusing shared CUDA Python dependencies, so
future backend experiments do not mutate the pinned serving environment.

### vLLM nightly B12X serving comparison

The first fixed-length cross-runtime point used the same Qwen3.8-NVFP4
checkpoint, 572 actual prompt tokens, 256 generated tokens, two warmups,
thinking disabled, `ignore_eos=true`, and c16 requests. Both services ran on
one RTX 5090 with FP8 KV cache and `max-model-len=1024`; the 1024 limit is
intentional for this bounded scheduling probe and is not the 4K protocol.

| Runtime | Backend | Requests | Success | TTFT P50 / P99 | Aggregate output |
|---|---|---:|---:|---:|---:|
| vLLM nightly | B12X | 16 | 16/16 | 912 / 1,254 ms | **186.06 tok/s** |
| SGLang main | FP4 FlashInfer cuDNN, Mamba ratio 8 | 16 | 16/16 | 10,288 / 19,590 ms | 108.49 tok/s |

The vLLM point is 71.5% faster on this bounded 1K/c16 shape, while its
per-request completion time is about 22 s and the server admits more requests
in one wave. This is a runtime scheduling result, not yet a general Kairo win;
the 4K-configured matrix below is the relevant comparison.

The same matrix was then repeated with `max-model-len=4096` and
`max-num-seqs=16` for vLLM (SGLang used its 4K context profile). The prompt and
generation lengths remained fixed, so startup/context capacity was the only
changed serving envelope:

| Runtime | c1 output | c4 output | c16 output | c16 TTFT P50 / P99 |
|---|---:|---:|---:|---:|
| vLLM nightly + B12X | 13.20 tok/s | 49.13 tok/s | **191.93 tok/s** | 967 / 1,316 ms |
| SGLang main, ratio 8 | 14.42 tok/s | 53.04 tok/s | 107.37 tok/s | 10,080 / 19,670 ms |

The vLLM c16 result was repeated three times (191.93, 193.95, and 194.11
tok/s; all 48 requests succeeded), giving a conservative +78.7% throughput
delta over the paired SGLang point. At c1/c4, SGLang remains slightly faster.
The shape-specific reversal is the strongest current lead: vLLM/B12X keeps a
much larger effective decode batch, while SGLang's Mamba cache forms multiple
waves. It is a measurable runtime/scheduling wedge, not a kernel-publication
claim; the long-prompt and deterministic correctness repeats below qualify it
for continued investigation.

The long-prompt repeat used 2,048 requested prompt tokens (2,241--2,243
actual), 128 generated tokens, c16, two warmups, and the same 4K serving
envelope. vLLM/B12X remained ahead, but the gap narrowed as prefill work grew:

| Runtime | Output tok/s | TTFT P50 / P99 | Success |
|---|---:|---:|---:|
| vLLM nightly + B12X | **133.38** | 3,167 / 5,628 ms | 16/16 |
| SGLang main, ratio 8 | 56.59 (first) | — | 16/16 |
| SGLang main, ratio 4.59 | 57.65 | — | 16/16 |

The first SGLang long-prompt run recorded output throughput but not a comparable
TTFT series. Against that first point, the vLLM long-prompt result was about
2.31x faster, so the c16 lead survived beyond the short-prompt wedge while the
absolute advantage was smaller than at 512 tokens.

A second vLLM run under the same protocol reached 133.45 tok/s (versus
133.38 tok/s initially), with 32/32 requests successful across the two runs.
This narrow spread makes the long-prompt vLLM point repeatable enough for the
next profiling stage; the repeated SGLang measurement below adds identical TTFT
capture for the cross-runtime comparison.

That SGLang repeat is now complete: ratio 8 reached 61.90 tok/s with TTFT P50/P99
of 11,995/24,228 ms and 16/16 success. Across the two SGLang runs the output
range is 56.59--61.90 tok/s; using the latest pair, vLLM is still about 2.15x
faster at c16. The large, visible three-wave queue remains the optimization
target rather than a model-quality difference.

The CLI now exposes this evidence as a bounded runtime policy via
`recommend-runtime`: measured c16 cells select vLLM nightly+B12X, while the
measured c1/c4 short cells select SGLang. Unmeasured shapes return `manual`
instead of silently extrapolating. This is the first executable form of the
workload-aware routing hypothesis.

The new `scripts/wsl/analyze_waves.py` turns that observation into a reproducible
metric using a documented 2,000 ms TTFT-gap threshold. On the latest long-prompt
JSON, vLLM forms one 16-request wave; SGLang forms waves of 6 + 6 + 4, with
10.55 s and 10.80 s inter-wave gaps. This is strong evidence for a scheduling or
state-cache admission bottleneck and gives Kairo a concrete target to attack.
For any saved benchmark result:

```bash
python3 scripts/wsl/analyze_waves.py result.json --gap-ms 2000
```

### Deterministic correctness gate

`scripts/wsl/check_correctness.py` sends four temperature-0 OpenAI-compatible
chat requests (exact sentinel, arithmetic, low-bit token, and Chinese text)
with thinking disabled. Both serving lanes passed all four cases on the same
Qwen3.8 checkpoint:

| Runtime | Passed |
|---|---:|
| vLLM nightly + B12X | **4/4** |
| SGLang main, ratio 8 | **4/4** |

This is a semantic/protocol smoke gate, not a proof of token-level equivalence
for arbitrary prompts. It is sufficient to prevent a throughput result from
being reported when the lane cannot answer deterministic control tasks; future
kernel work should add task-suite and regression prompts.

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

### Mamba budget experiment

The SGLang source log identified the concrete capacity limit: with the default
`mamba_full_memory_ratio=4.59`, the Mamba state cache capped the server at 8
running requests. Raising only this ratio to 8.0 increased the automatic cap to
9 (all other model, backend, context, KV, and precision settings stayed fixed):

| Mamba ratio | Auto cap | c1 output | c4 output | c16 output | c16 TTFT P50 / P99 |
|---:|---:|---:|---:|---:|---:|
| 4.59 | 8 | 14.30 tok/s | 54.59 tok/s | 62.79 tok/s | 23.12 / 45.75 s |
| 8.0 | 9 | 14.42 tok/s | 53.04 tok/s | **107.37 tok/s** | 10.08 / 19.67 s |

The c16 pair used fresh services, two warmups, 16 requests, 512 requested
prompt tokens (572 actual), exactly 256 generated tokens, and
`ignore_eos=true`. The ratio-8 point therefore delivers a measured +71% c16
throughput over the paired ratio-4.59 run, while c1/c4 remain effectively flat.
This is the first credible end-to-end performance wedge, but it is
high-concurrency-specific and must survive repeated runs, longer prompts, and a
correctness matrix before being promoted as the default configuration.

The required longer-context check qualifies that result. With 2,048 requested
prompt tokens (2,241--2,243 actual), 128 generated tokens, c16, and the same
warmup/fixed-length protocol, ratio 8.0 reached 56.59 tok/s while ratio 4.59
reached 57.65 tok/s. The ratio-8 allocation is therefore not a global win: its
extra Mamba slot helps short-prompt c16 batching, but the smaller KV pool erases
the benefit at this longer prompt shape. A production profile must choose the
ratio from prompt/concurrency forecasts rather than hard-code 8.0.

An additional ratio-12 probe makes the trade-off sharper. At short prompt/c16
it reached 111.25 tok/s (versus 107.37 at ratio 8) and admitted 8+8 requests
in two waves. At the 2K prompt it fell to 53.54 tok/s and formed four waves of
4, with roughly 9.1--9.2 s between waves. The ratio-12 configuration is recorded
as an experimental profile in
`experiments/protocols/qwen38-sglang-ratio12-probe.yaml`; it is not promoted to
the recommender until repeated trials and a correctness pass are complete.

A ratio-16 short-prompt probe reached 111.30 tok/s with the same 8+8 wave
pattern as ratio 12, so increasing the Mamba budget beyond 12 provides no
measured gain on this shape. The complete short-prompt sweep is captured in
`experiments/protocols/qwen38-sglang-budget-sweep.yaml`; ratio 8 remains the
documented default and ratios 12/16 are explicit experiment overrides.

One additional scheduler probe set `KAIRO_NUM_CONTINUOUS_DECODE_STEPS=4` on the
ratio-8 service. The identical c16 workload reached 103.08 tok/s with TTFT
P50/P99 of 10.65/20.53 s, about 4% below the ratio-8 default of 107.37 tok/s.
The extra continuous steps are therefore rejected for this workload; the
current candidate remains ratio-only.

A matching prefill probe (2,048 requested prompt tokens, 2,241 actual tokens,
one generated token, concurrency 1, four requests) produced TTFT P50 238 ms and
input throughput 9,080 tok/s. It is a warm-cache observation and should be
repeated before using it to select a prefill kernel target.

The fixed-length repeat (two warmups, thinking disabled, `ignore_eos=true`)
returned 4/4 successful requests with the same 2,241 actual input tokens,
TTFT P50/P99 of 227/259 ms, and 9,460 input tok/s. This is close enough to the
earlier reading to treat the prefill point as stable for the current runtime,
while still requiring cross-backend repetition before optimization claims.

The pinned vLLM 0.29.0 lane remains a compatibility probe and still stalls on
Qwen3.8 initialization. The isolated nightly+B12X lane is the reproducible
vLLM comparison path. The 0.5B model remains the CI canary; Qwen3.8 is now the
hero performance lane, but any public win claim must use the 4K matrix and a
correctness repeat.

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
