#!/usr/bin/env bash
set -euo pipefail

echo "Kairo WSL environment"
echo "PWD: $(pwd)"
python3 --version
git --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
  cuda_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
  if [[ "${cuda_release%%.*}" -lt 13 && "$cuda_release" != 12.8* && "$cuda_release" != 12.9* ]]; then
    echo "WARNING: CUDA Toolkit $cuda_release is not accepted for native RTX 5090 kernel builds. Install CUDA 12.8+ before custom CUDA work."
  fi
else
  echo "CUDA compiler: not found (fine for baseline profiling; required for custom CUDA kernels)"
fi

if [[ -x .venv/bin/python ]]; then
  echo "Kairo venv: .venv/bin/python"
fi

if python3 -c 'import torch' 2>/dev/null; then
  python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch_gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY
else
  echo "PyTorch: not installed"
fi
