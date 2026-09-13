#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
port="${1:-18088}"
model="${KAIRO_QWEN_MODEL:-/home/peter/kairo-models/Qwen3.8-27B-NVFP4}"
log_file="${KAIRO_QWEN_CUTLASS_LOG:-/tmp/kairo-qwen-vllm-cutlass-${port}.log}"
service_pid=""

cleanup() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  echo "server_log=$log_file" >&2
}
trap cleanup EXIT

export KAIRO_MAX_MODEL_LEN="${KAIRO_MAX_MODEL_LEN:-4096}"
export KAIRO_GPU_MEMORY_UTILIZATION="${KAIRO_GPU_MEMORY_UTILIZATION:-0.80}"
export KAIRO_KV_CACHE_DTYPE="${KAIRO_KV_CACHE_DTYPE:-fp8_e4m3}"
export KAIRO_TRUST_REMOTE_CODE="${KAIRO_TRUST_REMOTE_CODE:-1}"
export KAIRO_SKIP_MM_PROFILING="${KAIRO_SKIP_MM_PROFILING:-1}"
export KAIRO_LANGUAGE_MODEL_ONLY="${KAIRO_LANGUAGE_MODEL_ONLY:-1}"
export KAIRO_LINEAR_BACKEND="${KAIRO_LINEAR_BACKEND:-cutlass}"
export KAIRO_DISABLE_THINKING="${KAIRO_DISABLE_THINKING:-1}"
export KAIRO_KEEP_ALIVE=1
export KAIRO_HEALTH_TIMEOUT="${KAIRO_HEALTH_TIMEOUT:-300}"
repeats="${KAIRO_BENCH_REPEATS:-1}"
run_correctness="${KAIRO_RUN_CORRECTNESS:-0}"

if [[ "${KAIRO_VERIFY_MODEL_REVISION:-1}" == "1" && -n "${KAIRO_MODEL_REVISION:-}" ]]; then
  "${KAIRO_VERIFY_PYTHON:-python3}" "$root/scripts/wsl/verify_model_snapshot.py" \
    --model-path "$model" --expected-revision "$KAIRO_MODEL_REVISION" >&2
fi

"$root/scripts/wsl/smoke_serve.sh" vllm-nightly "$model" "$port" >"$log_file" 2>&1 &
service_pid=$!
for _ in $(seq 1 "$KAIRO_HEALTH_TIMEOUT"); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    tail -120 "$log_file" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${port}/health" >/dev/null

if [[ "$run_correctness" == "1" ]]; then
  /home/peter/venv-vllm-nightly/bin/python "$root/scripts/wsl/check_correctness.py" \
    --base-url "http://127.0.0.1:${port}" --model smoke
fi

for repeat in $(seq 1 "$repeats"); do
  echo "benchmark_repeat=$repeat"
  bench_args=(
    --base-url "http://127.0.0.1:${port}" --model smoke --lane decode
    --concurrency "${KAIRO_BENCH_CONCURRENCY:-16}"
    --prompt-tokens "${KAIRO_BENCH_PROMPT_TOKENS:-512}"
    --generation-tokens "${KAIRO_BENCH_GENERATION_TOKENS:-256}"
    --warmup "${KAIRO_BENCH_WARMUP:-1}"
    --requests "${KAIRO_BENCH_REQUESTS:-32}" --disable-thinking --ignore-eos
  )
  [[ -n "${KAIRO_MODEL_REVISION:-}" ]] && bench_args+=(
    --model-revision "$KAIRO_MODEL_REVISION"
  )
  [[ -n "${KAIRO_BENCH_CONTEXT_TOKENS:-}" ]] && bench_args+=(
    --context-tokens "$KAIRO_BENCH_CONTEXT_TOKENS"
  )
  /home/peter/venv-vllm-nightly/bin/python "$root/scripts/wsl/bench_openai.py" \
    "${bench_args[@]}"
done
