#!/usr/bin/env bash
set -euo pipefail

echo "Kairo WSL environment"
echo "PWD: $(pwd)"
python_bin="${KAIRO_PYTHON:-python3}"
if [[ "$python_bin" == "python3" && -x /home/peter/venv-gpu/bin/python ]]; then
  python_bin="/home/peter/venv-gpu/bin/python"
fi
"$python_bin" --version
git --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

nvcc_bin="$(command -v nvcc || true)"
if [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  nvcc_bin="/usr/local/cuda-13.0/bin/nvcc"
elif [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  nvcc_bin="/usr/local/cuda-12.8/bin/nvcc"
fi
if [[ -n "$nvcc_bin" ]]; then
  "$nvcc_bin" --version | tail -n 1
  cuda_release="$($nvcc_bin --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
  if [[ "${cuda_release%%.*}" -lt 13 && "$cuda_release" != 12.8* && "$cuda_release" != 12.9* ]]; then
    echo "WARNING: CUDA Toolkit $cuda_release is not accepted for native RTX 5090 kernel builds. Install CUDA 12.8+ before custom CUDA work."
  fi
else
  echo "CUDA compiler: not found (fine for baseline profiling; required for custom CUDA kernels)"
fi

if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  python_bin=".venv/bin/python"
  echo "Kairo venv: .venv/bin/python"
elif [[ -x /home/peter/venv-gpu/bin/python ]]; then
  echo "Existing GPU venv: /home/peter/venv-gpu/bin/python"
fi

if "$python_bin" -c 'import torch' 2>/dev/null; then
  "$python_bin" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch_gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY
else
  echo "PyTorch: not installed"
fi

"$python_bin" - <<'PY'
import importlib.metadata
for package in ("vllm", "sglang", "tensorrt-llm"):
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}=not-installed")
PY
