#!/usr/bin/env bash
set -euo pipefail

# Build an isolated SGLang lane without replacing the TensorRT/vLLM runtime.
# Large wheels are intentionally local/ignored; download them through the host
# network when WSL's configured mirror cannot transfer CUDA artifacts.
env_dir="${KAIRO_SGLANG_ENV:-/home/peter/venv-sglang}"
torch_wheel="${KAIRO_TORCH_WHEEL:-/mnt/c/Projects/Kairo/.kairo-local/torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl}"
kernel_wheel="${KAIRO_SGLANG_KERNEL_WHEEL:-/mnt/c/Projects/Kairo/.kairo-local/sglang_kernel-0.4.6.post1-cp310-abi3-manylinux2014_x86_64.whl}"
sglang_wheel="${KAIRO_SGLANG_WHEEL:-/home/peter/kairo-packages/sglang-0.5.19-cp312-cp312-manylinux_2_34_x86_64.whl}"

for wheel in "$torch_wheel" "$kernel_wheel" "$sglang_wheel"; do
  if [[ ! -f "$wheel" ]]; then
    echo "missing wheel: $wheel" >&2
    exit 1
  fi
done

python3 -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --no-deps --progress-bar off \
  "$torch_wheel" "$sglang_wheel" "$kernel_wheel"

# Reuse the already-installed pure-Python/CUDA support packages while keeping
# this environment's torch first on sys.path.
printf '%s\n' /home/peter/venv-gpu/lib/python3.12/site-packages \
  > "$env_dir/lib/python3.12/site-packages/kairo_shared.pth"
"$env_dir/bin/python" -m pip install --no-deps --progress-bar off gguf
"$env_dir/bin/python" - <<'PY'
import torch
import sglang
print(f"sglang_env torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.is_available()}")
print(f"sglang={sglang.__version__}")
PY
