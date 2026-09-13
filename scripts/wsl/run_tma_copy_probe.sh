#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda_root="${KAIRO_CUDA_ROOT:-/usr/local/cuda-13.0}"
nvcc="$cuda_root/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
  echo "CUDA 13 nvcc not found at $nvcc; set KAIRO_CUDA_ROOT to a CUDA 13+ toolkit" >&2
  exit 2
fi
build_dir="${KAIRO_PROBE_BUILD_DIR:-/tmp/kairo-tma-copy-probe}"
mkdir -p "$build_dir"
binary="$build_dir/tma_copy_probe"
"$nvcc" -std=c++17 -O3 -arch=sm_120 \
  "$root/scripts/wsl/tma_copy_probe.cu" -lcuda -o "$binary"
"$binary"
