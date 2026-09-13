#!/usr/bin/env bash
set -euo pipefail

# Runtime wheels carry their own CUDA user-space libraries. This enables baseline
# profiling; native custom kernels still require CUDA Toolkit 12.8+ in the WSL OS.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.14.0"
.venv/bin/python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'
