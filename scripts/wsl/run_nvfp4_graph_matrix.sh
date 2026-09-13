#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
  [[ -n "$output" ]] && printf '%s\n' "$record" >> "$output"
}

# Each cell is M,N,K,iterations,warmups. Static M buckets mirror decode-like
# row counts; override this list for a model-specific experiment.
shapes="${KAIRO_NVFP4_GRAPH_SHAPES:-1,4096,4096,500,30;32,4096,4096,300,30;128,4096,4096,200,20}"
backend="${KAIRO_NVFP4_BACKEND:-cutlass}"
IFS=';' read -r -a shape_cells <<< "$shapes"
for shape in "${shape_cells[@]}"; do
  IFS=',' read -r m n k iterations warmups <<< "$shape"
  if [[ -z "${m:-}" || -z "${n:-}" || -z "${k:-}" || -z "${iterations:-}" || -z "${warmups:-}" ]]; then
    echo "invalid shape cell (expected M,N,K,iterations,warmups): $shape" >&2
    exit 2
  fi
  if (( n % 32 != 0 || k % 32 != 0 )); then
    echo "invalid NVFP4 shape (N and K must be divisible by 32): $shape" >&2
    exit 2
  fi
  emit "$(bash "$root/scripts/wsl/run_nvfp4_probe.sh" \
    --backend "$backend" --m "$m" --n "$n" --k "$k" \
    --iterations "$iterations" --warmups "$warmups" --cuda-graph)"
done
