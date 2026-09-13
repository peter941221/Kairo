#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda_root="${KAIRO_CUDA_ROOT:-/usr/local/cuda-13.0}"
nvcc="$cuda_root/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
  echo "CUDA 13 nvcc not found at $nvcc; set KAIRO_CUDA_ROOT to a CUDA 13+ toolkit" >&2
  exit 2
fi

build_dir="${KAIRO_PROBE_BUILD_DIR:-/tmp/kairo-tma-wmma-gemm-probe}"
mkdir -p "$build_dir"
binary="$build_dir/tma_wmma_gemm_probe"
"$nvcc" -std=c++17 -O3 -arch=sm_120 \
  "$root/scripts/wsl/tma_wmma_gemm_probe.cu" -lcuda -lcublas -o "$binary"

output="${1:-}"
if [[ -n "$output" ]]; then
  if [[ -e "$output" && "${KAIRO_ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "refusing to overwrite existing output: $output (set KAIRO_ALLOW_OVERWRITE=1)" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$output")"
  : > "$output"
fi

emit() {
  local record="$1"
  printf '%s\n' "$record"
  if [[ -n "$output" ]]; then
    printf '%s\n' "$record" >> "$output"
  fi
}

# Each cell is M,N,K,iterations. Override with a semicolon-separated list.
shapes="${KAIRO_TMA_SHAPES:-1024,1024,1024,20;2048,1024,4096,10;4096,4096,4096,10}"
variants="${KAIRO_TMA_VARIANTS:-single,m128,m256,double}"
IFS=';' read -r -a shape_cells <<< "$shapes"
IFS=',' read -r -a variant_cells <<< "$variants"
for shape in "${shape_cells[@]}"; do
  IFS=',' read -r m n k iterations <<< "$shape"
  if [[ -z "${m:-}" || -z "${n:-}" || -z "${k:-}" || -z "${iterations:-}" ]]; then
    echo "invalid shape cell (expected M,N,K,iterations): $shape" >&2
    exit 2
  fi
  for variant in "${variant_cells[@]}"; do
    if [[ "$variant" == "m128" && $((m % 128)) -ne 0 ]]; then
      echo "skip variant=m128 shape=[$m,$n,$k] (M is not divisible by 128)" >&2
      continue
    fi
    if [[ "$variant" == "m256" && $((m % 256)) -ne 0 ]]; then
      echo "skip variant=m256 shape=[$m,$n,$k] (M is not divisible by 256)" >&2
      continue
    fi
    if [[ "$variant" == "m256_s32" && $((m % 256)) -ne 0 ]]; then
      echo "skip variant=m256_s32 shape=[$m,$n,$k] (M is not divisible by 256)" >&2
      continue
    fi
    emit "$("$binary" "$m" "$n" "$k" "$iterations" "$variant")"
  done
done
