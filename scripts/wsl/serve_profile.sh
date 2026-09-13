#!/usr/bin/env bash
set -euo pipefail

backend="${1:-sglang}"
model="${2:-/home/peter/kairo-models/Qwen3.8-27B-NVFP4}"
port="${3:-18086}"
concurrency="${4:-${KAIRO_WORKLOAD_CONCURRENCY:-16}}"
prompt_tokens="${5:-${KAIRO_WORKLOAD_PROMPT_TOKENS:-512}}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$backend" != "sglang" ]]; then
  echo "profile selection currently supports sglang only" >&2
  exit 2
fi
if [[ "$model" != *Qwen3.8-27B-NVFP4* ]]; then
  echo "profile selection currently supports the Qwen3.8-27B-NVFP4 lane only" >&2
  exit 2
fi

python_bin="${KAIRO_PROFILE_PYTHON:-python3}"
profile_shell="$(PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m kairo_lab.cli \
  recommend-profile --model qwen38 --concurrency "$concurrency" \
  --prompt-tokens "$prompt_tokens" --format shell)"
eval "$profile_shell"

echo "profile=$KAIRO_PROFILE_ID concurrency=$concurrency prompt_tokens=$prompt_tokens ratio=$KAIRO_MAMBA_FULL_MEMORY_RATIO" >&2
if [[ "${KAIRO_PROFILE_DRY_RUN:-0}" == "1" ]]; then
  echo "would_exec=$root/scripts/wsl/smoke_serve.sh $backend $model $port"
  exit 0
fi
exec "$root/scripts/wsl/smoke_serve.sh" "$backend" "$model" "$port"
