#!/usr/bin/env bash
set -euo pipefail

# Run a measured Qwen benchmark using only a measured runtime route.  The workload
# dimensions are required so a short-prompt Graph result cannot silently be
# reused for a long-context request.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
port="${1:-18120}"
model="${KAIRO_QWEN_MODEL:-/home/peter/kairo-models/Qwen3.8-27B-NVFP4}"
profile_model="${KAIRO_PROFILE_MODEL:-qwen38}"
concurrency="${KAIRO_WORKLOAD_CONCURRENCY:-16}"
prompt_tokens="${KAIRO_WORKLOAD_PROMPT_TOKENS:-512}"
context_tokens="${KAIRO_WORKLOAD_CONTEXT_TOKENS:-}"
generation_tokens="${KAIRO_WORKLOAD_GENERATION_TOKENS:-}"

if [[ -z "$context_tokens" || -z "$generation_tokens" ]]; then
  echo "set KAIRO_WORKLOAD_CONTEXT_TOKENS and KAIRO_WORKLOAD_GENERATION_TOKENS" >&2
  exit 2
fi

python_bin="${KAIRO_PROFILE_PYTHON:-python3}"
route_json="$(PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m kairo_lab.cli \
  recommend-runtime --model "$profile_model" --concurrency "$concurrency" \
  --prompt-tokens "$prompt_tokens" --context-tokens "$context_tokens" \
  --generation-tokens "$generation_tokens")"

readarray -t route_values < <("$python_bin" -c 'import json, sys; r=json.load(sys.stdin); print(r["backend"]); print(r.get("profile") or ""); print(r.get("cudagraph_mode") or ""); print(r.get("linear_backend") or ""); print(r.get("max_num_seqs") or ""); print(r.get("max_model_len") or "")' <<<"$route_json")
backend="${route_values[0]}"
profile="${route_values[1]}"
cudagraph_mode="${route_values[2]}"
linear_backend="${route_values[3]}"
max_num_seqs="${route_values[4]}"
route_max_model_len="${route_values[5]}"

echo "route=$route_json" >&2
if [[ "$backend" != "vllm-nightly" ]]; then
  echo "route is not executable by this Qwen CUTLASS runner: backend=$backend profile=$profile" >&2
  exit 3
fi

export KAIRO_QWEN_MODEL="$model"
export KAIRO_BENCH_CONCURRENCY="$concurrency"
export KAIRO_BENCH_PROMPT_TOKENS="$prompt_tokens"
export KAIRO_BENCH_GENERATION_TOKENS="$generation_tokens"
export KAIRO_BENCH_CONTEXT_TOKENS="$context_tokens"
export KAIRO_BENCH_REQUESTS="${KAIRO_BENCH_REQUESTS:-$concurrency}"
export KAIRO_RUN_CORRECTNESS="${KAIRO_RUN_CORRECTNESS:-1}"
export KAIRO_LINEAR_BACKEND="${linear_backend:-cutlass}"
if [[ "$cudagraph_mode" == "FULL_DECODE_ONLY" ]]; then
  export KAIRO_ENFORCE_EAGER=0
  export KAIRO_CUDAGRAPH_MODE="$cudagraph_mode"
  # Use the measured per-route context/sequence limits. Do not inherit a
  # broader caller setting by accident.
  if [[ "${KAIRO_ROUTED_ALLOW_OVERRIDES:-0}" == "1" ]]; then
    export KAIRO_MAX_MODEL_LEN="${KAIRO_MAX_MODEL_LEN:-1024}"
    export KAIRO_MAX_RUNNING_REQUESTS="${KAIRO_MAX_RUNNING_REQUESTS:-${max_num_seqs:-$concurrency}}"
  else
    export KAIRO_MAX_MODEL_LEN="${route_max_model_len:-1024}"
    export KAIRO_MAX_RUNNING_REQUESTS="${max_num_seqs:-32}"
  fi
else
  export KAIRO_ENFORCE_EAGER=1
  unset KAIRO_CUDAGRAPH_MODE
  if [[ "${KAIRO_ROUTED_ALLOW_OVERRIDES:-0}" == "1" ]]; then
    export KAIRO_MAX_MODEL_LEN="${KAIRO_MAX_MODEL_LEN:-4096}"
  else
    export KAIRO_MAX_MODEL_LEN=4096
  fi
fi

if [[ "${KAIRO_ROUTED_DRY_RUN:-0}" == "1" ]]; then
  printf 'would_exec=%s\n' "$root/scripts/wsl/run_qwen_cutlass_bench.sh $port"
  exit 0
fi
if [[ "${KAIRO_ROUTED_NO_CAPTURE:-0}" == "1" ]]; then
  exec "$root/scripts/wsl/run_qwen_cutlass_bench.sh" "$port"
fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_file="${KAIRO_ROUTED_OUTPUT:-$root/.kairo-local/routed-${stamp}-${port}.out}"
mkdir -p "$(dirname "$output_file")"
export KAIRO_QWEN_CUTLASS_LOG="${KAIRO_QWEN_CUTLASS_LOG:-${output_file%.out}.server.log}"
echo "server_log=$KAIRO_QWEN_CUTLASS_LOG" >&2
echo "raw_output=$output_file" >&2
set +e
"$root/scripts/wsl/run_qwen_cutlass_bench.sh" "$port" 2>&1 | tee "$output_file"
status="${PIPESTATUS[0]}"
set -e
if [[ "$status" -eq 0 && "${KAIRO_ROUTED_SKIP_GATE:-0}" != "1" ]]; then
  "$python_bin" "$root/scripts/wsl/analyze_stability.py" "$output_file"
fi
exit "$status"
