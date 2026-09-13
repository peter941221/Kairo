# Kairo

Kairo is a Blackwell inference-kernel experiment workbench. Its first purpose is
to discover a **reproducible, end-to-end performance wedge** on a single RTX
5090—not to prematurely become a general compiler.

## North star

On one RTX 5090, improve a public model under a public workload by at least 20%
over a pinned, fair baseline, without changing model semantics or hiding startup
costs. The first target is selected from measured evidence, not intuition.

## Initial experiment lanes

- **Decode:** tokens/s and P50/P99 inter-token latency under interactive load.
- **Prefill:** time-to-first-token and throughput at 4K/16K context lengths.
- **Discovery:** a dense low-bit model is the controlled primary lane; MoE is a
  high-upside discovery lane once an executable baseline is available.

The current hero candidate is `nvidia/Qwen3.8-27B-NVFP4`; the fast control is
`nvidia/Qwen3-8B-NVFP4`. `Qwen2.5-0.5B-Instruct` remains only the CI/service
canary. Candidate metadata lives in
[`experiments/workloads/candidates.yaml`](experiments/workloads/candidates.yaml).

Validate a declarative kernel blueprint before compiling or running it:

```bash
bash scripts/wsl/run_lab.sh validate-blueprint \
  --file experiments/blueprints/fp16-tma-wmma.yaml
```

The validator fail-closes unknown template/precision combinations, enforces
tile alignment, asynchronous TMA/mbarrier requirements, shared-memory budgets,
and explicit dynamic dimensions. It returns a blueprint hash and the cache-key
dimensions used by the runtime.

Turn a validated blueprint into an executable cache-aware plan for one shape:

```bash
bash scripts/wsl/run_lab.sh plan-blueprint \
  --file experiments/blueprints/fp16-tma-wmma.yaml \
  --shape 1024 1024 1024 --driver-version 596.36 --gpu-capability sm120
```

The plan makes the order explicit: validate, lookup/build, load, correctness
gate, then benchmark. Its digest is directly compatible with
`RuntimeKernelCache`.

`src/kairo_lab/dispatch.py` provides the same contract in-process for a real
compiler/launcher or a test double: cache misses build once, an injected
correctness callback must pass before launch, launches are reported as
`kairo`, and failures use an explicit fallback with a machine-readable reason.
Each result also records artifact, correctness, and launch timings, so
compilation/caching cost cannot be confused with steady-state execution.

## Repository layout

```text
experiments/         versioned workload and protocol manifests
src/kairo_lab/       small, dependency-light lab CLI and result schema
scripts/wsl/         commands that run inside the Ubuntu WSL environment
tests/               protocol/schema tests
docs/                product and experiment documentation
.kairo-local/        ignored local rules, notes, credentials, and machine facts
```

GitHub CI runs the portable contract checks on every push and pull request:
unit tests, all YAML manifests, and every WSL shell entry point. GPU probes and
serving benchmarks remain explicitly local because they require the RTX 5090.

## Quick start (from PowerShell)

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && ./scripts/wsl/doctor.sh'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && ./scripts/wsl/run_lab.sh env'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && ./scripts/wsl/run_lab.sh init-run --lane decode'
```

Run the Phase 0 RTX 5090 compiler and asynchronous-copy gate:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/probe_capabilities.sh'
```

The measured memory roofline can be rechecked from its raw repeats:

```bash
python3 scripts/wsl/analyze_memory_bandwidth.py \
  .kairo-local/memory-bandwidth-probe-repeats.jsonl
```

Before serving a pinned local checkpoint, verify its Hugging Face snapshot
without re-hashing multi-gigabyte weight shards:

```bash
python3 scripts/wsl/verify_model_snapshot.py \
  --model-path /home/peter/kairo-models/Qwen3.8-27B-NVFP4 \
  --expected-revision dbb8f445b3145f8a4c18ddc769f032d57d32867c
```

The Qwen serving runners perform this check automatically for the known pinned
model directories (or when `KAIRO_MODEL_REVISION` is supplied). Set
`KAIRO_VERIFY_MODEL_REVISION=0` only for an explicitly unversioned exploratory
model.

Check whether the current toolkit exposes Blackwell `tcgen05` on SM120 (the
probe reports a structured unsupported result rather than failing the lab):

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/probe_tcgen05.sh'
```

Run the verified 2D TMA + mbarrier copy closure:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/run_tma_copy_probe.sh'
```

Run the first Kairo-owned FP16 tiled GEMM (correctness plus cuBLAS timing):

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/run_fp16_gemm_probe.sh 1024 1024 1024 50'
```

The probe defaults to the `tile32x32_output2x2` variant; pass
`tile16x16_output1x1`, `tile32x32_output2x2_async`, or `wmma_fp16` as a fifth
argument to reproduce the comparison variants. `wmma_fp16` and
`wmma_fp16_shared` require dimensions
divisible by 16; for very large shapes the probe uses the independent cuBLAS
output as its correctness oracle instead of an O(MNK) CPU reference.

Run the integrated TMA+WMMA GEMM path (aligned dimensions, cuBLAS comparison):

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/run_tma_wmma_gemm_probe.sh 1024 1024 1024 50'
```

Pass `double` as a fifth argument to run the experimental double-buffered
control, or `m128` to use the measured 128-row block candidate:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/run_tma_wmma_gemm_probe.sh 4096 4096 4096 10 double'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/run_tma_wmma_gemm_probe.sh 4096 4096 4096 10 m128'
```

`m128` requires M divisible by 128 and `m256` requires M divisible by 256
(all dimensions must be divisible by 16). They are measured candidates rather
than global defaults; use the protocol record to compare them against the
single-buffer control.

`m256_s32` is an experimental 32-byte TMA-swizzle probe; it is disabled from
the default matrix until its shared-memory layout passes correctness.

Run a reproducible shape × variant matrix (single, m128, m256, and double by
default). Records are printed as JSONL; pass an output path to save them:

```bash
bash scripts/wsl/run_tma_wmma_matrix.sh .kairo-local/tma-wmma-matrix.jsonl
```

Use `KAIRO_TMA_SHAPES=M,N,K,iters;...` or `KAIRO_TMA_VARIANTS=single,m128,m256`
to narrow a sweep. Existing output files are protected unless
`KAIRO_ALLOW_OVERWRITE=1` is set.

Summarize a saved matrix by median throughput and correctness-aware winner:

```bash
python scripts/wsl/analyze_tma_wmma_matrix.py .kairo-local/tma-wmma-shape-stable.jsonl
```

Ask the lab policy which CUDA variant is justified for a shape:

```bash
./scripts/wsl/run_lab.sh recommend-gemm --m 4096 --n 4096 --k 4096
```

The policy promotes the measured winner (`m128` or `m256`) only for exact
measured cells; unknown aligned shapes remain on the single-buffer control
until the matrix covers them.

Probe the installed vLLM CUTLASS NVFP4 path (isolated nightly environment):

```bash
bash scripts/wsl/run_nvfp4_probe.sh --m 32 --n 4096 --k 4096 --iterations 50
```

Select the native B12X SM120 implementation for a side-by-side low-batch
measurement:

```bash
bash scripts/wsl/run_nvfp4_probe.sh --backend b12x --m 1 --n 4096 --k 4096 --iterations 50
```

For a larger layer-like cell, use N=K=8192 and include pipeline timing:

```bash
bash scripts/wsl/run_nvfp4_probe.sh --m 128 --n 8192 --k 8192 --iterations 200
```

This is a vendor-kernel baseline for the NVFP4 phase, not yet a Kairo kernel
claim; it reports quantization error against the FP16 reference and includes
the activation-quantization pipeline cost.

For the experimental Kairo pipeline path, add `--cuda-graph` to capture the
quantization and CUTLASS GEMM as one static-shape replay. Current 4K matrix
two-repeat results show roughly 27--36% lower pipeline time with bit-exact agreement to
the regular pipeline; see
[`nvfp4-cuda-graph.yaml`](experiments/protocols/nvfp4-cuda-graph.yaml).
The vLLM service probe with `KAIRO_ENFORCE_EAGER=0` did not reach health within
five minutes, so service-level CUDA Graph mode is not enabled by default.

Run the shape-bucket matrix and keep machine-readable records:

```bash
bash scripts/wsl/run_nvfp4_graph_matrix.sh .kairo-local/nvfp4-graph.jsonl
```

Override cells with `KAIRO_NVFP4_GRAPH_SHAPES=M,N,K,iterations,warmups;...`.
Set `KAIRO_NVFP4_GRAPH_REPEATS=3` for independent process repeats; the analyzer
reports per-shape medians.
Summarize the JSONL with the correctness-aware selector:

```bash
python scripts/wsl/analyze_nvfp4_graph.py .kairo-local/nvfp4-graph.jsonl
```

The capture/replay implementation is in
[`cuda_graph_bucket.py`](scripts/wsl/cuda_graph_bucket.py); it
keeps one graph per operation namespace and exact shape, preventing a model or
backend switch from reusing an incompatible capture. It exposes capture cost,
replay break-even, and cache-hit statistics.

The model-level CUTLASS serving gate is reproducible with:

```bash
bash scripts/wsl/run_qwen_cutlass_bench.sh 18088
```

Add `KAIRO_RUN_CORRECTNESS=1` to run the four-case deterministic semantic gate
before throughput measurement.

Add `--control-first` to reverse the FP16/NVFP4 timing order when checking
clock or launch-order sensitivity.

`init-run` writes a machine-readable run record under `runs/`, which is ignored
by Git. Keep the command, input manifests, source revision, and published result
table together when reporting an experiment.

To run the backend service gate, use the shared GPU environment for vLLM and the
isolated PyTorch 2.13 environment for SGLang:

```bash
./scripts/wsl/smoke_serve.sh vllm
KAIRO_SGLANG_PYTHON=/home/peter/venv-sglang/bin/python \
  ./scripts/wsl/smoke_serve.sh sglang
```

For the Qwen3.8 text-only bring-up gate, use conservative single-5090 settings:

```bash
KAIRO_MAX_MODEL_LEN=4096 KAIRO_GPU_MEMORY_UTILIZATION=0.65 \
KAIRO_KV_CACHE_DTYPE=fp8_e4m3 KAIRO_TRUST_REMOTE_CODE=1 \
KAIRO_SKIP_MM_PROFILING=1 KAIRO_LANGUAGE_MODEL_ONLY=1 \
./scripts/wsl/smoke_serve.sh vllm /home/peter/kairo-models/Qwen3.8-27B-NVFP4 18085
```

To try the current SGLang main checkout without changing the pinned wheels:

```bash
KAIRO_SGLANG_PYTHON=/home/peter/venv-sglang/bin/python \
KAIRO_SGLANG_PYTHONPATH=/home/peter/src/sglang/python \
KAIRO_SGLANG_QWEN38_FLAGS=1 KAIRO_DISABLE_FLASHINFER_AUTOTUNE=1 \
KAIRO_MAMBA_SSM_DTYPE=bfloat16 KAIRO_SKIP_SERVER_WARMUP=1 \
KAIRO_HEALTH_TIMEOUT=240 KAIRO_GPU_MEMORY_UTILIZATION=0.80 \
KAIRO_CONTEXT_LENGTH=4096 KAIRO_KV_CACHE_DTYPE=fp8_e4m3 \
KAIRO_TRUST_REMOTE_CODE=1 \
./scripts/wsl/smoke_serve.sh sglang /home/peter/kairo-models/Qwen3.8-27B-NVFP4 18086
```

The vendor-recommended vLLM nightly path is isolated in
`/home/peter/venv-vllm-nightly` and launched through
`scripts/wsl/vllm_nightly.py`. It is a bring-up lane only until the SM120
CUTLASS/B12X dependencies are complete; the current isolated environment has
those dependencies and passes the Qwen3.8 smoke gate. Do not mix it into the
pinned `venv-gpu` baseline. The essential 5090 backend override is:

```bash
KAIRO_MAX_MODEL_LEN=4096 KAIRO_GPU_MEMORY_UTILIZATION=0.80 \
KAIRO_KV_CACHE_DTYPE=fp8_e4m3 KAIRO_TRUST_REMOTE_CODE=1 \
KAIRO_SKIP_MM_PROFILING=1 KAIRO_LANGUAGE_MODEL_ONLY=1 \
KAIRO_LINEAR_BACKEND=b12x KAIRO_MAX_RUNNING_REQUESTS=16 \
KAIRO_DISABLE_THINKING=1 KAIRO_KEEP_ALIVE=1 \
./scripts/wsl/smoke_serve.sh vllm-nightly \
  /home/peter/kairo-models/Qwen3.8-27B-NVFP4 18087
```

The underlying launcher can also be invoked directly when capturing custom
vLLM arguments:

```bash
export LD_LIBRARY_PATH=/home/peter/venv-vllm-nightly/lib/python3.12/site-packages/nvidia/nvshmem/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cublas/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cusparselt/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/nccl/lib
/home/peter/venv-vllm-nightly/bin/python scripts/wsl/vllm_nightly.py serve \
  /home/peter/kairo-models/Qwen3.8-27B-NVFP4 --port 18087 \
  --served-model-name smoke --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 --max-model-len 1024 \
  --enforce-eager --generation-config vllm --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --skip-mm-profiling --language-model-only --max-num-seqs 16 \
  --linear-backend b12x
```

For the experimentally validated high-concurrency variant, add
`KAIRO_MAMBA_FULL_MEMORY_RATIO=8`. It currently improves the fixed c16 decode
point substantially, while c1/c4 are neutral to slightly slower; see
[`docs/BASELINE_STATUS.md`](docs/BASELINE_STATUS.md) before using it as a
general default.

## Decision gate

After baseline profiling, promote a hotspot only if it has a credible route to a
20% end-to-end win. By Amdahl's law, a hotspot must consume at least one third of
runtime even if Kairo can make it twice as fast.

See [the v0 experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for the exact
fairness rules and selection gate.

The validated high-concurrency Qwen3.8 candidate is captured as a runnable
profile in
[`experiments/protocols/qwen38-sglang-high-concurrency.yaml`](experiments/protocols/qwen38-sglang-high-concurrency.yaml).

Use the conservative workload-aware recommender before launching an experiment:

```bash
./scripts/wsl/run_lab.sh recommend-profile \
  --model qwen38 --concurrency 16 --prompt-tokens 512
```

It selects ratio 8 only for the measured short-prompt c16 shape; all other
shapes remain on the ratio-4.59 baseline until measured.

Ratio 12 is now captured as a separate experimental probe: it improves the
512-token c16 point slightly but loses at 2K prompts, reinforcing the need for
workload-aware selection rather than a single global setting.
Ratio 16 was also tested and plateaued; see the
[`qwen38-sglang-budget-sweep.yaml`](experiments/protocols/qwen38-sglang-budget-sweep.yaml)
manifest for the measured wave sizes.

The first fair cross-runtime matrix is now recorded in
[`docs/BASELINE_STATUS.md`](docs/BASELINE_STATUS.md): vLLM nightly+B12X reaches
191.93–194.11 tok/s at the 4K-configured c16 point versus 107.37 tok/s for
SGLang ratio 8, while SGLang remains slightly ahead at c1/c4. The lead survives
the 2K-prompt repeats at 133.38–133.45 versus 56.59–61.90 tok/s, and both lanes pass the
deterministic 4/4 correctness gate. Treat the result as a shape-specific
scheduling lead until a broader task suite and repeated trials are complete.

The newer CUTLASS path has now passed two same-service repeats on the short
prompt c16 cell at 288.03 and 295.76 tok/s (32/32 successful), about 52% above
the paired B12X point. The recommender therefore selects CUTLASS for both
measured c16 prompt cells; other shapes stay manual until measured.

The strongest current end-to-end result is vLLM nightly + CUTLASS with
`FULL_DECODE_ONLY` CUDA Graphs: at c16, four fresh runs reached a median
718.61 tok/s versus 293.10 tok/s eager (2.45x, +145.2%). The c8 pilot reached
392.82--397.56 tok/s versus 152.94--154.62 tok/s eager (2.57x mean, 4/4
correctness). Both are bounded 1K-context results with `max_num_seqs=32`; see
[`qwen38-vllm-cudagraph-serving.yaml`](experiments/protocols/qwen38-vllm-cudagraph-serving.yaml)
before extrapolating it to other workloads.

At c16 on the same envelope, four fresh Graph runs reached a median 718.61
tok/s versus 293.10 tok/s eager (2.45x, +145.2%); the latest Graph run passed
4/4 correctness. The measured runtime policy routes the exact c16/prompt256
cell to this Graph profile and keeps other shapes explicit.

At c16/prompt512 with the same 1K context and 128-token output, Graph reached
672.00 tok/s median versus 259.86 eager (2.59x, +158.6%; 4/4 correctness).
That evidence is recorded separately because the current router still models
the existing prompt512 cell as a 4K-context/256-output workload.

At 4K context and c16/prompt2048, Graph still led the same-cap eager control:
260.07 versus 199.52 tok/s median (1.30x, +30.3%; 4/4 correctness). The
smaller margin is why this long-context cell remains experimental rather than
becoming an automatic default.

At c32/prompt256 in the 1K envelope, Graph reached 721.10 tok/s median versus
286.63 eager (2.52x, +151.7%; 4/4 correctness, 32/32 success), the current
headline point for the workbench.

The c32/prompt512 point also held: Graph median **667.01** versus eager
**272.00 tok/s** (2.45x, +145.2%; 4/4 correctness, 32/32 success).
The routed entry point then sustained five consecutive c32 batches with 0
failures each and 639.73 tok/s median (618.27–665.29 range).

The latest pinned Qwen3.8-27B validation repeats that exact c32/prompt512
workload in fresh services: Graph **645.08 tok/s** versus eager **308.51 tok/s**,
or **2.0909x / +109.09%**. Both sides generated exactly 128 tokens, passed 4/4
correctness, and completed 32/32 requests; Graph also improved TTFT P50
(899.54ms vs 1040.60ms) and Total P99 (6325.84ms vs 13257.14ms). Raw logs,
commands, and pinned model revision are recorded in the protocol's
`pinned_current_validation` block.

Validate that protocol block against the ignored raw logs at any time:

```bash
python3 scripts/wsl/validate_serving_protocol.py \
  experiments/protocols/qwen38-vllm-cudagraph-serving.yaml
```

The newer `Qwen3-8B-NVFP4` control is now covered as a separate measured
runtime profile. At c16/prompt512 with 1K context and 128 generated tokens,
three Graph repeats measured 1942.58, 2216.16, and 2177.49 tok/s (median
**2177.49**), while the eager control measured 1043.74, 1073.63, and 1116.30
(median **1073.63**): **2.03x / +102.8%**. Both lanes passed 4/4 correctness and
16/16 requests. The first Graph repeat includes capture overhead; the two
steady-state repeats varied by only 1.77%. See
[`qwen3-8b-vllm-cudagraph-serving.yaml`](experiments/protocols/qwen3-8b-vllm-cudagraph-serving.yaml).
The model-aware routed entry point was then exercised on the same cell: it
selected Graph automatically, passed the default 4/4 correctness and 16/16
success gates, measured 1847.96 tok/s, and passed the post-run stability audit.
At c32/prompt512, the same new-model Graph lane measured 3393.21, 3958.74,
and 3946.07 tok/s (steady-state median **3946.07**) versus eager 1836.39 and
1834.39 (median **1835.39**): **2.15x / +115.0%**, with 4/4 correctness and
32/32 success on every repeat. The c32 steady-state range was 0.32%.
At the 4K boundary (c16/prompt2048/context4096/generation128), a fresh
source-pinned pair measured Graph **1473.70 tok/s** versus eager **966.22
tok/s**, a **1.5252x / +52.52%** lead. Both lanes passed 4/4 correctness and
16/16 requests across two repeats; the exact raw logs, model revision, and
source revision are recorded in the protocol's `pinned_long_context_validation`
section. Validate that section with:

```bash
python3 scripts/wsl/validate_serving_protocol.py \
  experiments/protocols/qwen3-8b-vllm-cudagraph-serving.yaml \
  --section pinned_long_context_validation
```

The same Qwen3-8B c16/p512/1K/128 workload also has an isolated SGLang control:
with FP4 `flashinfer_cudnn`, FlashInfer autotune disabled, and CUDA Graphs
disabled, it passed 4/4 correctness and 16/16 requests at **353.35 tok/s**.
The vLLM eager control is 3.04x faster and the vLLM Graph route is 6.16x
faster on this exact cell. Reproduce it with
`bash scripts/wsl/run_sglang_bench.sh 18144` after setting
`KAIRO_SGLANG_MODEL` and `KAIRO_SGLANG_BENCH_PYTHON`; the full record is in
the Qwen3-8B protocol.

Summarize any repeated raw log and fail if correctness or request success is
incomplete:

```bash
python3 scripts/wsl/analyze_stability.py \
  .kairo-local/routed-c32-p512-stability5.out
```

The summary retains TTFT and total-latency P50/P99 for every repeat and reports
their cross-repeat medians alongside throughput.

Compare a candidate against a same-workload baseline and enforce the 20%
north-star gate:

```bash
python3 scripts/wsl/compare_benchmarks.py \
  .kairo-local/qwen3-8b-c16-p512-graph-repeats3.out \
  .kairo-local/qwen3-8b-c16-p512-eager-repeats3.out
```

Add `--format markdown` to emit a compact table suitable for a GitHub issue or
experiment report; JSON remains the default machine-readable output.

The command refuses promotion on workload mismatch, inconsistent parameters
across repeats, failed correctness, failed requests, or insufficient speedup.
It also requires at least two effective measured repeats per log by default;
`--drop-first` therefore requires three raw repeats. Use
`--minimum-repeats 1` only for explicitly exploratory output. Use `--drop-first`
only as an explicit steady-state capture-cost analysis.
Routed and SGLang runners pass their declared context limit, model revision, and
Kairo source revision into each raw benchmark record, so context, weight, or
source mismatches are visible to the gate.

Run the lightweight correctness gate against any OpenAI-compatible service:

```bash
python3 scripts/wsl/check_correctness.py \
  --base-url http://127.0.0.1:18087 --model smoke
```

Quantify scheduler waves from a saved benchmark result:

```bash
python3 scripts/wsl/analyze_waves.py result.json --gap-ms 2000
```

Ask the measured cross-runtime policy which lane covers a shape:

```bash
python -m kairo_lab.cli recommend-runtime \
  --model qwen38 --concurrency 16 --prompt-tokens 512 \
  --context-tokens 1024 --generation-tokens 128
```

The policy selects the measured c8/prompt256 cell as vLLM/CUTLASS
`FULL_DECODE_ONLY` Graph (`max_num_seqs=32`), vLLM/CUTLASS for both measured
c16 prompt cells, and SGLang for measured c1/c4 short cells; every other shape
returns `manual` until measured.
When context and generation lengths are supplied, Graph is selected only inside
the measured 1K-context/128-output envelope; the same prompt at a 4K/256
workload remains on its separately measured eager profile.

The same policy can target the newer control model for its measured cell:

```bash
python -m kairo_lab.cli recommend-runtime \
  --model qwen3_8b --concurrency 16 --prompt-tokens 512 \
  --context-tokens 1024 --generation-tokens 128
```

For the routed launcher, set `KAIRO_PROFILE_MODEL=qwen3_8b` together with
`KAIRO_QWEN_MODEL=/home/peter/kairo-models/Qwen3-8B-NVFP4`.

Run the routed benchmark entry point (WSL/5090):

```bash
KAIRO_WORKLOAD_CONCURRENCY=32 \
KAIRO_WORKLOAD_PROMPT_TOKENS=512 \
KAIRO_WORKLOAD_CONTEXT_TOKENS=1024 \
KAIRO_WORKLOAD_GENERATION_TOKENS=128 \
  bash scripts/wsl/run_routed_bench.sh 18120
```

It applies the measured vLLM/CUTLASS Graph or eager settings automatically,
and refuses an uncovered route. Set `KAIRO_ROUTED_DRY_RUN=1` to inspect the
selected profile without starting the model. The first real WSL/5090 run
selected c32/prompt512 Graph automatically and passed 4/4 correctness with
32/32 successful requests. By default the entry point also captures the raw
log under `.kairo-local`; set `KAIRO_ROUTED_OUTPUT` to choose another path or
`KAIRO_ROUTED_NO_CAPTURE=1` to disable capture.
The launcher also writes the vLLM service log beside the raw output as
`*.server.log`, preserving health-check and kernel-compilation failures.
The measured memory/context limits are enforced by default; set
`KAIRO_ROUTED_ALLOW_OVERRIDES=1` only for an explicitly experimental run.
The complementary c16/prompt512 4K/256 workload automatically selects eager
and has also passed its 4/4 correctness and 16/16 success gate. The
correctness gate and post-run stability audit are enabled by default;
`KAIRO_ROUTED_SKIP_GATE=1` is an explicit experimental escape hatch.

The profiler fallback records CUDA operator attribution when Nsight counters are
not exposed:

```bash
bash -lc 'cd /mnt/c/Projects/Kairo && \
  /home/peter/venv-gpu/bin/python scripts/wsl/profile_transformers.py \
  --model /home/peter/kairo-models/Qwen2.5-0.5B-Instruct \
  --mode decode --prompt-tokens 512 --steps 16'
```

The measured operator breakdown is captured in
[`transformers-profiler.yaml`](experiments/protocols/transformers-profiler.yaml).

The runtime cache primitive in `src/kairo_lab/cache.py` provides content-
addressed AOT/JIT artifact storage. Its key includes blueprint hash, full
shape, driver version, GPU capability, and template version; metadata records
integrity, hit/miss counts, and explainable fingerprint invalidation. It is
ready to back PTX/Cubin or serialized graph packages without silently reusing
artifacts across hardware changes. Concurrent first misses are single-flight,
so a burst of requests cannot trigger duplicate builds for one key.

Inspect a persistent cache without altering runtime hit/miss counters:

```bash
python -m kairo_lab.cli cache-inspect --root .kairo-local/kernel-cache
```

Ask the executable policy whether an NVFP4 shape is eligible for Graph replay:

```bash
python -m kairo_lab.cli recommend-nvfp4 --m 128 --n 4096 --k 4096
```

To apply that decision automatically when launching SGLang:

```bash
KAIRO_WORKLOAD_CONCURRENCY=16 KAIRO_WORKLOAD_PROMPT_TOKENS=512 \
  ./scripts/wsl/serve_profile.sh sglang \
  /home/peter/kairo-models/Qwen3.8-27B-NVFP4 18086
```

For a deliberately labeled budget experiment, set
`KAIRO_PROFILE_RATIO_OVERRIDE=12`; the launcher keeps the recommended profile
metadata and appends an override marker to the run identity.
