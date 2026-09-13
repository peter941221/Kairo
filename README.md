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

## Repository layout

```text
experiments/         versioned workload and protocol manifests
src/kairo_lab/       small, dependency-light lab CLI and result schema
scripts/wsl/         commands that run inside the Ubuntu WSL environment
tests/               protocol/schema tests
docs/                product and experiment documentation
.kairo-local/        ignored local rules, notes, credentials, and machine facts
```

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

`m128` requires M divisible by 128 (and all dimensions divisible by 16). It
is currently a measured candidate rather than the global default; use the
protocol record to compare it against the single-buffer control.

Run a reproducible shape × variant matrix (single, m128, m256, and double by
default). Records are printed as JSONL; pass an output path to save them:

```bash
bash scripts/wsl/run_tma_wmma_matrix.sh .kairo-local/tma-wmma-matrix.jsonl
```

Use `KAIRO_TMA_SHAPES=M,N,K,iters;...` or `KAIRO_TMA_VARIANTS=single,m128,m256`
to narrow a sweep. Existing output files are protected unless
`KAIRO_ALLOW_OVERWRITE=1` is set.

Ask the lab policy which CUDA variant is justified for a shape:

```bash
./scripts/wsl/run_lab.sh recommend-gemm --m 4096 --n 4096 --k 4096
```

The policy promotes the measured winner (`m128` or `m256`) only for exact
measured cells; unknown aligned shapes remain on the single-buffer control
until the matrix covers them.

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
  --model qwen38 --concurrency 16 --prompt-tokens 2048
```

The policy selects vLLM/B12X for the measured c16 cells and SGLang for the
measured c1/c4 short cells; every other shape returns `manual` until measured.

To apply that decision automatically when launching SGLang:

```bash
KAIRO_WORKLOAD_CONCURRENCY=16 KAIRO_WORKLOAD_PROMPT_TOKENS=512 \
  ./scripts/wsl/serve_profile.sh sglang \
  /home/peter/kairo-models/Qwen3.8-27B-NVFP4 18086
```

For a deliberately labeled budget experiment, set
`KAIRO_PROFILE_RATIO_OVERRIDE=12`; the launcher keeps the recommended profile
metadata and appends an override marker to the run identity.
