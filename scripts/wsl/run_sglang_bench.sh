#!/usr/bin/env bash
set -euo pipefail

# Run a reproducible SGLang OpenAI-compatible decode benchmark. This lane is
# deliberately eager (smoke_serve disables CUDA Graphs) so it is a fair runtime
# control for the vLLM Graph experiments.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
port="${1:-18140}"
model="${KAIRO_SGLANG_MODEL:-/home/peter/kairo-models/Qwen3-8B-NVFP4}"
python_bin="${KAIRO_SGLANG_BENCH_PYTHON:-/home/peter/venv-sglang/bin/python}"
server_log="${KAIRO_SGLANG_SERVER_LOG:-/tmp/kairo-sglang-${port}.log}"
repeats="${KAIRO_BENCH_REPEATS:-1}"
concurrency="${KAIRO_BENCH_CONCURRENCY:-16}"
prompt_tokens="${KAIRO_BENCH_PROMPT_TOKENS:-512}"
generation_tokens="${KAIRO_BENCH_GENERATION_TOKENS:-128}"
requests="${KAIRO_BENCH_REQUESTS:-$concurrency}"
warmup="${KAIRO_BENCH_WARMUP:-1}"
health_timeout="${KAIRO_HEALTH_TIMEOUT:-300}"
service_pid=""

cleanup() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  echo "server_log=$server_log" >&2
}
trap cleanup EXIT

export KAIRO_SGLANG_PYTHON="${KAIRO_SGLANG_PYTHON:-$python_bin}"
export KAIRO_SGLANG_QWEN38_FLAGS="${KAIRO_SGLANG_QWEN38_FLAGS:-0}"
export KAIRO_SERVER_LOG="$server_log"
export KAIRO_CONTEXT_LENGTH="${KAIRO_CONTEXT_LENGTH:-1024}"
export KAIRO_GPU_MEMORY_UTILIZATION="${KAIRO_GPU_MEMORY_UTILIZATION:-0.80}"
export KAIRO_TRUST_REMOTE_CODE="${KAIRO_TRUST_REMOTE_CODE:-1}"
export KAIRO_SKIP_SERVER_WARMUP="${KAIRO_SKIP_SERVER_WARMUP:-1}"
export KAIRO_DISABLE_THINKING="${KAIRO_DISABLE_THINKING:-1}"
export KAIRO_KEEP_ALIVE=1
export KAIRO_HEALTH_TIMEOUT="$health_timeout"

"$root/scripts/wsl/smoke_serve.sh" sglang "$model" "$port" >"$server_log" 2>&1 &
service_pid=$!
for _ in $(seq 1 "$health_timeout"); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    tail -120 "$server_log" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${port}/health" >/dev/null

if [[ "${KAIRO_RUN_CORRECTNESS:-1}" == "1" ]]; then
  "$python_bin" "$root/scripts/wsl/check_correctness.py" \
    --base-url "http://127.0.0.1:${port}" --model smoke
fi

for repeat in $(seq 1 "$repeats"); do
  echo "benchmark_repeat=$repeat"
  bench_args=(
    --base-url "http://127.0.0.1:${port}" --model smoke --lane decode
    --concurrency "$concurrency" --prompt-tokens "$prompt_tokens"
    --generation-tokens "$generation_tokens" --warmup "$warmup"
    --requests "$requests" --disable-thinking --ignore-eos
  )
  [[ -n "${KAIRO_BENCH_CONTEXT_TOKENS:-}" ]] && bench_args+=(
    --context-tokens "$KAIRO_BENCH_CONTEXT_TOKENS"
  )
  "$python_bin" "$root/scripts/wsl/bench_openai.py" "${bench_args[@]}"
done
