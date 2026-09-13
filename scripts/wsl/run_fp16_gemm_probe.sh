#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda_root="${KAIRO_CUDA_ROOT:-/usr/local/cuda-13.0}"
nvcc="$cuda_root/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
  echo "CUDA 13 nvcc not found at $nvcc; set KAIRO_CUDA_ROOT to a CUDA 13+ toolkit" >&2
  exit 2
fi

build_dir="${KAIRO_PROBE_BUILD_DIR:-/tmp/kairo-fp16-gemm-probe}"
mkdir -p "$build_dir"
binary="$build_dir/fp16_gemm_probe"
"$nvcc" -std=c++17 -O3 -arch=sm_120 \
  "$root/scripts/wsl/fp16_gemm_probe.cu" \
  -I"$cuda_root/include" -L"$cuda_root/lib64" -lcublas -o "$binary"
"$binary" "${1:-1024}" "${2:-1024}" "${3:-1024}" "${4:-50}"
