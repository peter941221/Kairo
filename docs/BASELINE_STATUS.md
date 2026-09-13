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

The Phase 0 roofline input is now measured by
`scripts/wsl/run_memory_bandwidth_probe.sh`: a contiguous 512 MiB `uint4`
device read/write kernel reached 1417.461, 1528.578, and 1531.124 GB/s across
three independent processes (median 1528.578 GB/s, 7.44% range). The result is
an achieved workload bandwidth, not a theoretical GDDR7 peak; raw JSONL is
kept in `.kairo-local/memory-bandwidth-probe-repeats.jsonl` and the protocol is
`experiments/protocols/memory-bandwidth-phase0.yaml`.
The dependency-free analyzer `scripts/wsl/analyze_memory_bandwidth.py` enforces
the repeated-measurement gate when this baseline is regenerated.

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
TMA improves the 4K point by roughly 1.5x; cuBLAS remains faster.

After recovering WSL from the later SGLang autotune failure, a fresh process
reran the 1K TMA+WMMA cell at 35,465 GFLOP/s with zero error versus cuBLAS.
This post-restart validation is recorded in the protocol's
`post_wsl_restart_validation` block and confirms the SM120 TMA path itself was
not damaged by the runtime incident.

The follow-up double-buffered control is also correct (zero error), but did
not improve these shapes: 32,108 versus 34,926 GFLOP/s at 1K and 40,801 versus
41,917 GFLOP/s at 4K. The extra barrier/synchronization overhead currently
outweighs overlap, so the single-buffer kernel remains the default. The double
variant is retained as a regression fixture for future swizzle, larger-K, and
low-bit experiments. Parameters are recorded in
`experiments/protocols/tma-wmma-gemm-phase1.yaml`.

The next block-shape experiment uses a 128x16 A tile and eight WMMA warps per
block (`m128` variant). It stays correct on square 1K/4K and a rectangular
`[2048,1024,4096]` cell. It reaches 46,491--47,091 GFLOP/s at 4K (about
11--13% above the 64-row baseline) and 42,332 GFLOP/s on the rectangular cell
(about 8% above its control). This is the leading aligned-shape candidate,
but routing remains conservative until boundary and occupancy coverage grows.

The shape-expansion matrix confirms why that conservatism matters: at the
small `[256,1024,1024]` smoke cell both larger blocks lose to the 64-row
control. In the stable rerun, m128 wins `[512,1024,1024]` (32,316 vs 30,663
GFLOP/s), while m256 narrowly wins 2K square (42,977 vs 42,717 vs 40,378
for m128/single), 4K square (49,850 vs 47,316 vs 42,854), and 8K×1K×1K
(45,821 vs 44,887 vs 41,322). These figures use 50 iterations (20 for the
4K cell), not the noisier 5--10 iteration smoke points. The policy therefore uses an explicit measured-winner
allowlist, recorded in `experiments/protocols/tma-wmma-shape-expansion.yaml`,
rather than extrapolating from divisibility alone.

The first shared-memory swizzle probe (`m256_s32`) is rejected at the
correctness gate: its 1K run produced maximum absolute error 18.52 against
cuBLAS. This is a layout-contract failure, not a slow implementation—TMA's
swizzled destination cannot be consumed by the existing row-major WMMA loads.
The result is recorded in `experiments/protocols/tma-wmma-swizzle-probe.yaml`;
future swizzle work must provide a matching shared-memory access mapping.

### NVFP4 vendor baseline

The isolated vLLM nightly environment exposes a working CUTLASS NVFP4 path on
SM120 (`cutlass_fp4_supported=true`). In same-process comparisons with the
same tensors and warmups, CUTLASS NVFP4 reaches 1.63, 42.20, and 203.79
TFLOP/s at M=1, 32, and 128, versus FP16 `torch.mm` at 1.48, 49.63, and
147.89 TFLOP/s. That is 1.105×, 0.850×, and 1.378× respectively. The native
B12X path reaches only 0.355, 11.60, and 37.46 TFLOP/s (0.240×, 0.211×,
0.243×), so it is not a low-batch default despite being available. The NVFP4
output is finite; quantization error against the original FP16 operands is
13--14% relative mean and is not a substitute for model-level accuracy. The full command and raw records are captured in
`experiments/protocols/nvfp4-cutlass-phase0.yaml` and
`.kairo-local/nvfp4-cutlass-4096.jsonl`. This is now the highest-value path
for a Kairo-owned optimization, but no custom-kernel win is claimed yet.

A 1K-iteration threshold sweep (same-process FP16 control) shows the largest
stable opportunity at M=128 (about 1.4--1.75× across ordered repeats). M=1 is
at most marginal, while M=8--64 can move with process ordering and GPU clocks;
earlier runs even crossed 1.0× at M=1/8/16. Therefore there is no universal
hard-coded crossover yet: routing must use repeated per-shape measurements and
include activation-quantization overhead. The measurements are recorded in
`experiments/protocols/nvfp4-cutlass-threshold.yaml`.

The pipeline probe then included per-call activation quantization. At
N=K=4096, pre-quantized CUTLASS NVFP4 was 1.02×/1.53×/1.49× FP16 at
M=1/32/128, but quantization-plus-GEMM fell to 0.71×/1.19×/0.95×. This makes
the actionable Kairo target explicit: fuse activation quantization with the
NVFP4 GEMM (or amortize it across decode steps), rather than merely calling
the vendor GEMM. The pipeline data is recorded in
`experiments/protocols/nvfp4-pipeline-overhead.yaml`.

The first Kairo-owned pipeline optimization is now measured: capturing
activation quantization plus CUTLASS GEMM in `torch.cuda.CUDAGraph` reduces
pipeline time by 27.5--36.0% across two independent process repeats at
M=1/32/128 (N=K=4096), with graph output
matching the regular pipeline at max absolute error 0.0. Across the repeated
matrix, the graph path reaches median 1.08x/1.34x/1.54x FP16 controls at
M=1/32/128, respectively. This requires static shape buckets and
recapture when dimensions change; it is an integration candidate, not yet a
drop-in vLLM backend. Reproduce with `--cuda-graph`; the full matrix is in
`experiments/protocols/nvfp4-cuda-graph.yaml`.

The reusable shape-bucket helper now reports capture cost and lookup reuse. On
the unified cache probe, capture took 15.7--23.0 ms and the measured replay
savings amortized after roughly 2.2k--3.4k calls; each bucket showed one
capture and one subsequent cache hit. Buckets are now isolated by operation
namespace as well as shape, so changing backend/model cannot reuse an
incompatible graph. Replacing activation values in-place
after capture still matched the regular pipeline at max error 0.0. This gives
the decode scheduler a concrete retention threshold instead of assuming that
graph capture is free.

Attempting to transfer this directly to the Qwen3.8 service by removing
`--enforce-eager` did not pass the 300-second health gate in the nightly
environment; the process remained in initialization and produced no valid
throughput sample. A follow-up with `cudagraph_mode=PIECEWISE` also failed its
210-second health gate. `smoke_serve.sh` now exposes `KAIRO_ENFORCE_EAGER` and
compilation controls, but keeps the safe eager default. The microbenchmark
result therefore stands as a shape-static building block, not a claim that
vLLM service graphs already work.

The workbench now has a content-addressed runtime cache primitive in
`src/kairo_lab/cache.py`. It keys opaque PTX/Cubin/graph artifacts by blueprint
hash, full shape, driver version, GPU capability, and template version, writes
payload and metadata atomically, verifies SHA-256 integrity, and reports hit,
miss, and explainable fingerprint-invalidation counters. This closes the PRD
cache contract at the library layer; wiring it into a compiled-kernel loader
remains the next integration step. Concurrent first misses now use a per-key
single-flight lock, so only one caller builds an artifact while waiters reuse
the published result. Runtime dispatch now records artifact lookup/build time
and correctness-gate time separately from launch time, preserving the PRD's
startup-versus-steady-state boundary and refusing to launch an unvalidated
artifact. The `cache-inspect` CLI audits on-disk metadata, payload
hashes, missing artifacts, and orphan files without changing runtime counters.

`src/kairo_lab/comparison.py` and `scripts/wsl/compare_benchmarks.py` now turn
the north-star claim into a machine gate. They require matching benchmark
configs, complete correctness and request-success gates on both logs, then
compare medians against a configurable minimum ratio. Every repeat within each
log must also keep the same workload identity, so parameter drift cannot
manufacture a speedup; promotion additionally requires two effective repeats
per log by default (three raw repeats when dropping the first). On the Qwen3-8B Graph /
eager logs this gate reports 2.0282x (+102.82%) and `promotion_gate=true`; an
explicit `--drop-first` analysis still reports 2.0063x.
The benchmark runner now records an optional declared `context_tokens` field;
routed and SGLang entry points populate it automatically, preventing a 1K/4K
context mix-up from being treated as a fair A/B.
Stability summaries also retain TTFT and total-latency P50/P99 per repeat and
their medians, so a throughput win can be checked against interactive latency.

`recommend-nvfp4` now exposes the measured Graph allowlist: exact 4K
N/K shapes at M=1/32/128 select `cuda_graph_shape_bucket` with replay
amortization estimates; aligned but unmeasured shapes select the regular NVFP4
pipeline, and unaligned shapes select fallback. This prevents the new
optimization from silently extrapolating to unsupported dimensions.

At a larger, layer-like N=K=8192 shape, the quantized pipeline remains fast:
2.82, 105.48, and 471.08 TFLOP/s at M=1/32/128, versus same-process FP16
controls of 1.61, 48.48, and 177.72 TFLOP/s (1.75×/2.18×/2.65×). An
independent C++ cuBLAS control at M=128 measured 172.83 TFLOP/s, consistent
with the torch control. This is the strongest current low-bit opportunity; the
next milestone is model-level accuracy and serving integration, not another
isolated FP4 wrapper. Details and raw JSONL are in
`experiments/protocols/nvfp4-large-shape.yaml` and
`.kairo-local/nvfp4-pipeline-8192.jsonl`.

Run it directly on the WSL 5090:

```bash
bash scripts/wsl/run_fp16_gemm_probe.sh 1024 1024 1024 50
```

## Fresh-model gate (2026-09-13)

The workbench's primary candidate is now `nvidia/Qwen3.8-27B-NVFP4`, with
`nvidia/Qwen3-8B-NVFP4` as the fast control. The Qwen3.8 checkpoint is fully
downloaded at `/home/peter/kairo-models/Qwen3.8-27B-NVFP4` (about 21 GiB).

### Declarative blueprint gate

`src/kairo_lab/blueprint.py` now provides the first fail-closed configuration
constraint layer. `validate-blueprint` accepts only registered template /
precision topologies, checks tile alignment, TMA+mbarrier requirements,
shared-memory budgets, and explicit dynamic dimensions, then emits a stable
blueprint hash and cache-key plan. The checked-in
`experiments/blueprints/fp16-tma-wmma.yaml` is a valid SM120 example; invalid
combinations stop before compilation rather than being silently guessed.
`plan-blueprint` extends the report into an executable shape-specific plan,
constructing the same content-addressed `RuntimeKernelCache` key (including
driver, GPU capability, template version, and shape) and making the correctness
gate precede benchmarking.
`RuntimeDispatcher` now wires that plan to `RuntimeKernelCache` with injected
compiler and launcher callbacks. It preserves single-flight builds and returns
explicit `kairo` versus `fallback` results, including the failure stage and
reason, so a future CUDA extension can be integrated without changing the
experiment contract.

The repository now has a portable GitHub Actions contract gate at
`.github/workflows/ci.yml`: it runs the full Python suite, parses every tracked
experiment manifest, and checks all WSL shell entry points without requiring a
GPU. Hardware probes and serving measurements remain separate local gates.

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

### vLLM nightly CUTLASS serving pilot

The same Qwen3.8 checkpoint was then launched through the isolated nightly
vLLM environment with `--linear-backend cutlass`. The fixed-length envelope
matches the B12X point above (4K max length, FP8 KV cache, 572-token prompt,
256 generated tokens, two warmups, c16, 16 requests, thinking disabled and
`ignore_eos=true`). Across two warm-cache repeats, all 32 requests succeeded;
CUTLASS reached **288.03 and 295.76 tok/s** (mean 291.90) with TTFT P50 of
1,127 and 1,094 ms, versus 191.93 tok/s for the paired B12X run (mean delta
approximately +52.1%). This is the first model-level
signal that the CUTLASS path can turn the isolated NVFP4 kernel advantage into
a serving advantage on the 5090.

The same 2K-prompt c16 workload also reached 154.60 and 152.33 tok/s across two
repeats (about +15.0% over the paired B12X 133.38--133.45 tok/s). This extends
the CUTLASS lead beyond short prompts, although the absolute gain is smaller.

This is a pilot, not yet a blanket production default: the comparisons use one
fresh service per backend and need broader concurrency/shape coverage. The exact
command and raw output are captured in
`experiments/protocols/qwen38-vllm-cutlass-serving.yaml` and
`.kairo-local/qwen-cutlass-c16-fair.out`. The runner supports
`KAIRO_BENCH_REPEATS=N` to repeat warm-cache measurements without reloading
the model.
The same CUTLASS service also passed the deterministic 4/4 semantic gate
(`KAIRO_RUN_CORRECTNESS=1`) before a short c16 throughput run; its raw record is
`.kairo-local/qwen-cutlass-correctness.out`.

### vLLM FULL_DECODE_ONLY Graph breakthrough

The nightly runtime can use CUDA Graphs for decode-only execution when its
Mamba cache capacity is respected. With `cudagraph_mode=FULL_DECODE_ONLY`,
`max_num_seqs=32`, and a 1K context envelope, Qwen3.8 CUTLASS serving reached
**397.56 and 392.82 output tok/s** on two fresh services (mean 395.19), while
the identical eager CUTLASS control reached 154.62 and 152.94 tok/s (mean
153.78). That is a **2.57x / +157.0%** throughput result, with median TTFT
falling from 390--444 ms to about 230 ms. The Graph service also passed the
4/4 deterministic correctness gate and all 16 throughput requests succeeded.

This is the first system-level Kairo lead combining a current model, NVFP4
CUTLASS, and runtime scheduling. It is deliberately bounded: 8-way
concurrency, 292 actual prompt tokens, 128 generated tokens, 1K context, and
`max_num_seqs=32`. Higher concurrency/context and fresh-service repeats remain
required before making it a blanket default. The full protocol is in
`experiments/protocols/qwen38-vllm-cudagraph-serving.yaml`.

The same Graph route also held at c16 under the identical 1K-context envelope:
four fresh services reached 652.09, 750.95, 710.00, and 727.22 output tok/s
(median **718.61**), while the eager control reached 296.10 and 290.10
(median **293.10**). This is a **2.45x / +145.2%** median throughput lead;
the latest Graph run passed the 4/4 correctness gate and all 16 requests
succeeded. This c16 result established the stable Graph route; c32 is now the
strongest measured serving point, while
remaining bounded to the exact prompt/context/sequence-cap envelope.

To test whether the lead was only a very short-prompt effect, the same c16
Graph/eager pair was repeated at 512 requested prompt tokens (572 actual),
still with 1K context and 128 generated tokens. Graph reached 676.95 and
667.05 tok/s (median **672.00**) versus eager 262.21 and 257.51 (median
**259.86**): **2.59x / +158.6%**, with 4/4 correctness and 16/16 success.
This is strong shape evidence, but the current router deliberately does not
map it onto the existing 4K-context/256-output c16 profile; context and output
length must become explicit routing dimensions before that promotion.

The 4K-context boundary was also tested directly at 2,048 requested prompt
tokens (2,241.75 actual), 128 generated tokens, and c16 with
`max_num_seqs=16`. Graph reached 254.28 and 265.86 tok/s (median **260.07**)
versus a same-cap eager control at 174.71 and 224.34 (median **199.52**), a
more modest but real **1.30x / +30.3%** lead. Correctness was 4/4 and all
requests succeeded. The result supports Graph as a long-context optimization,
but its smaller margin and sequence-cap sensitivity argue for keeping this
cell experimental until more fresh-service repeats are collected.

The 1K-context short-prompt cell was then pushed to c32. Graph delivered
715.84 and 726.37 tok/s (median **721.10**) versus an identical eager c32
control at 286.99 and 286.27 (median **286.63**): **2.52x / +151.7%**.
All 32 requests succeeded and the 4/4 correctness gate passed. This is the
current headline throughput point, still bounded by the 1K context and
`max_num_seqs=32` memory envelope.

At c32/prompt512 in the same 1K/128 envelope, Graph reached 675.77 and
658.25 tok/s (median **667.01**) versus eager 266.68 and 277.32 (median
**272.00**): **2.45x / +145.2%**, with 4/4 correctness and 32/32 success.
The c32 lead therefore survives both tested prompt buckets.

The routed c32/prompt512 Graph was then held in one service for five
consecutive batches. Every batch completed 32/32 requests with zero failures;
throughput was 665.29, 638.58, 639.73, 618.27, and 657.79 tok/s (median
**639.73**, range **7.35%** of median), after a 4/4 correctness pass. This is
a same-service stability gate, not a substitute for fresh-service variance.
The raw log can be checked mechanically with
`python3 scripts/wsl/analyze_stability.py`; it extracts all benchmark JSON
objects, strips PTY control codes, and exits non-zero on any failed request or
incomplete correctness gate.

The CLI now exposes this evidence as a bounded runtime policy via
`recommend-runtime`: the measured c8/prompt256 cell selects vLLM nightly
CUTLASS `FULL_DECODE_ONLY` Graph with `max_num_seqs=32`; both repeated c16
prompt cells select vLLM nightly CUTLASS (the exact c16/prompt256 cell uses
the Graph profile), while measured c1/c4 short cells select SGLang. Unmeasured
shapes return `manual` instead of silently extrapolating.
This is the first executable form of the workload-aware routing hypothesis.
Supplying `--context-tokens` and `--generation-tokens` activates the envelope
gate: Graph is eligible only for the measured 1K/128 route, while a 4K/256
request stays on its separately measured eager profile.
The same policy is executable through `scripts/wsl/run_routed_bench.sh`, which
sets the vLLM nightly Graph/eager flags and invokes the common correctness plus
throughput runner; uncovered routes fail closed. Its first real WSL/5090
integration run auto-selected c32/prompt512 Graph, passed 4/4 correctness,
completed 32/32 requests, and measured 665.76 tok/s. The runner now captures
each raw invocation automatically for post-hoc stability analysis and stores
the vLLM service log beside it as `*.server.log`. A second
integration run with c16/prompt512, 4K context, and 256 output tokens selected
the eager route, passed 4/4 correctness, completed 16/16 requests, and measured
292.53 tok/s; both automatic branches are therefore executable.
The routed launcher enforces the measured context and sequence limits by
default; broader settings require the explicit `KAIRO_ROUTED_ALLOW_OVERRIDES=1`
escape hatch and are not covered by the published route.
The routed runner now enables correctness by default and runs the machine
stability audit after capture; skipping it requires the explicit
`KAIRO_ROUTED_SKIP_GATE=1` override.
An invocation with no explicit correctness flag was also verified: the default
gate ran 4/4, completed 32/32 requests, measured 630.56 tok/s, and passed the
post-run stability audit.

### New-model transfer check: Qwen3-8B NVFP4

The local `nvidia/Qwen3-8B-NVFP4` checkpoint is now a measured control rather
than a discovery-only candidate. Under vLLM nightly + CUTLASS, c16/prompt512,
1K context, 128 generated tokens, and 16 requests, the Graph lane measured
1942.58, 2216.16, and 2177.49 tok/s (median **2177.49**). The identical eager
control measured 1043.74, 1073.63, and 1116.30 tok/s (median **1073.63**), a
**2.028x / +102.8%** median lead. Both lanes passed 4/4 deterministic
correctness and 16/16 request-success gates. The first Graph sample includes
capture overhead; its two steady-state samples vary by 1.77%. Full settings and
raw logs are recorded in
`experiments/protocols/qwen3-8b-vllm-cudagraph-serving.yaml`.
The runtime recommender accepts `--model qwen3_8b` for this exact measured
cell; all other Qwen3-8B shapes remain `manual` until measured. The model-aware
routed launcher was also run end to end: it selected Graph automatically,
passed 4/4 correctness and 16/16 request success, measured 1847.96 tok/s, and
passed the post-run stability audit.
The same A/B was extended to c32/prompt512: Graph measured 3393.21, 3958.74,
and 3946.07 tok/s (steady-state median **3946.07**) versus eager 1836.39 and
1834.39 (median **1835.39**), a **2.15x / +115.0%** lead. Every repeat passed
4/4 correctness and 32/32 request success; the two steady-state Graph repeats
varied by only 0.32%. The recommender now covers both c16 and c32 for this
exact Qwen3-8B workload bucket. At the 4K boundary (c16/prompt2048,
generation128), Graph's two steady-state repeats averaged 1869.92 tok/s versus
the eager post-warmup point of 1102.76 tok/s, a provisional **1.69x / +69.5%**
lead. The exact long-context route is exposed for experiments, but remains
experimental until the eager control receives another fresh-service repeat.
An isolated SGLang 0.5.19 control was also brought up for the short cell with
FP4 `flashinfer_cudnn`, FlashInfer autotune disabled, and CUDA Graphs disabled.
It passed 4/4 correctness and 16/16 requests at **353.35 tok/s** (TTFT P50
260.27 ms), versus 1073.63 tok/s for vLLM eager and 2177.49 tok/s for vLLM
Graph on the paired c16 workload. Thus vLLM eager is 3.038x faster and Graph
is 6.162x faster than this SGLang control. The reusable
`scripts/wsl/run_sglang_bench.sh` entry point captures both server and benchmark
logs; the no-autotune variant is the reproducible SGLang result. The default
SGLang autotune variant destabilized WSL before health, so it remains a failed
compatibility experiment rather than a performance claim.

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
The latest reproducible run is recorded in
`experiments/protocols/transformers-profiler.yaml`: at 512 tokens, four
prefill steps consumed 17.597 ms self-CUDA time with `aten::mm` at 57.07%; 16
KV-cache decode steps consumed 37.841 ms with `aten::mm` at 41.60%, two GEMV
kernel buckets at 32.04% + 12.00%, and FlashAttention at 8.50%. The traces are
kept locally under `.kairo-local` because they are large. These figures remain
operator hypotheses for the 0.5B canary, not attribution of the Qwen3.8 Graph
serving gain.
The next experiment should use the same shapes in an in-process backend probe,
or enable GPU performance counters, before implementing a custom kernel.
