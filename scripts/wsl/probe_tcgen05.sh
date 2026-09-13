#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda_root="${KAIRO_CUDA_ROOT:-/usr/local/cuda-13.0}"
nvcc="$cuda_root/bin/nvcc"
build_dir="${KAIRO_PROBE_BUILD_DIR:-/tmp/kairo-tcgen05-probe}"
mkdir -p "$build_dir"
binary="$build_dir/tcgen05_compile_probe"
error_file="$build_dir/compile.err"

if [[ ! -x "$nvcc" ]]; then
  printf '{"supported":false,"reason":"nvcc_not_found","nvcc":"%s"}\n' "$nvcc"
  exit 0
fi
set +e
"$nvcc" -std=c++17 -arch=sm_120 \
  "$root/scripts/wsl/tcgen05_compile_probe.cu" -o "$binary" 2>"$error_file"
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
  reason="$(tr '\n' ' ' < "$error_file" | sed 's/[[:space:]]\+/ /g; s/[[:space:]]*$//')"
  printf '{"supported":false,"target":"sm_120","compile_exit":%d,"reason":"%s"}\n' "$status" "$reason"
  exit 0
fi
set +e
"$binary" >/dev/null 2>&1
run_status=$?
set -e
printf '{"supported":true,"target":"sm_120","compile_exit":0,"run_exit":%d}\n' "$run_status"
