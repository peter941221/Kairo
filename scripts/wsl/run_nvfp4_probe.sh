#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/home/peter/venv-vllm-nightly/lib/python3.12/site-packages/nvidia_cutlass_dsl/cu13/lib:/home/peter/venv-vllm-nightly/lib/python3.12/site-packages/nvidia/nvshmem/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cublas/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cusparselt/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
exec /home/peter/venv-vllm-nightly/bin/python \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/wsl/probe_nvfp4_mm.py" "$@"
