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
