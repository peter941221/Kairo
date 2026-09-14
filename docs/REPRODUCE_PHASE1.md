# Reproduce Phase 1

This guide has three levels. Levels 1 and 2 run without an RTX 5090. Level 3
requires an RTX 5090, WSL2 Ubuntu 24.04, CUDA 13-compatible software, local
NVFP4 model weights, and the vLLM nightly environment used by the runner.

## 1. Inspect the published evidence

```bash
git clone https://github.com/peter941221/Kairo.git
cd Kairo
python scripts/render_phase1_results.py
```

This regenerates `docs/generated/phase1-serving-results.md` and `.svg` from
the reviewed per-repeat JSONL. It should report 2.09x for the Qwen3.8 1K c32
cell and 2.62x for the Qwen3.8 4K c8 cell.

## 2. Run portable checks

```bash
python -m pip install pyyaml
PYTHONPATH=src python -m unittest discover -s tests -q
bash -n scripts/wsl/*.sh
bash scripts/wsl/run_lab.sh validate-blueprint \
  --file experiments/blueprints/fp16-tma-wmma.yaml
```

The unit tests, protocol parsing, and blueprint validator do not require a GPU.

## 3. Re-run a measured serving route

Set only local paths through environment variables. The values below are
examples; do not commit them.

```bash
export KAIRO_MODEL_DIR=/path/to/kairo-models
export KAIRO_VLLM_NIGHTLY_VENV=/path/to/venv-vllm-nightly
export KAIRO_GPU_VENV=/path/to/venv-gpu
export KAIRO_QWEN_MODEL="$KAIRO_MODEL_DIR/Qwen3.8-27B-NVFP4"
```

Verify that the exact c32 / prompt-512 / context-1K request has a measured
route before starting a server:

```bash
KAIRO_ROUTED_DRY_RUN=1 \
KAIRO_PROFILE_MODEL=qwen38 \
KAIRO_WORKLOAD_CONCURRENCY=32 \
KAIRO_WORKLOAD_PROMPT_TOKENS=512 \
KAIRO_WORKLOAD_CONTEXT_TOKENS=1024 \
KAIRO_WORKLOAD_GENERATION_TOKENS=128 \
bash scripts/wsl/run_routed_bench.sh 18150
```

The output must identify the measured `FULL_DECODE_ONLY` route. To execute it
with correctness enabled and two independent repeats:

```bash
KAIRO_PROFILE_MODEL=qwen38 \
KAIRO_WORKLOAD_CONCURRENCY=32 \
KAIRO_WORKLOAD_PROMPT_TOKENS=512 \
KAIRO_WORKLOAD_CONTEXT_TOKENS=1024 \
KAIRO_WORKLOAD_GENERATION_TOKENS=128 \
KAIRO_BENCH_REPEATS=2 \
bash scripts/wsl/run_routed_bench.sh 18150
```

The published comparison uses a separately launched eager control with the
same workload. The complete commands, model revision, runtime revision, raw
artifact hashes, and expected per-repeat throughput are pinned in
[`qwen38-vllm-cudagraph-serving.yaml`](../experiments/protocols/qwen38-vllm-cudagraph-serving.yaml).

Do not compare an unpinned model revision, a different actual token count, or
a run with a different context/sequence limit to the Phase 1 headline.
