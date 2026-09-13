#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
model="${2:-/home/peter/kairo-models/Qwen2.5-0.5B-Instruct}"
port="${3:-18080}"
vllm_bin="${KAIRO_VLLM_BIN:-/home/peter/venv-gpu/bin/vllm}"
sglang_python="${KAIRO_SGLANG_PYTHON:-/home/peter/venv-gpu/bin/python}"
log_file="$(mktemp /tmp/kairo-${backend:-unknown}.XXXXXX.log)"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  echo "server_log=$log_file"
}
trap cleanup EXIT

if [[ "$backend" == "vllm" ]]; then
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
  "$vllm_bin" serve "$model" \
    --host 127.0.0.1 --port "$port" \
    --served-model-name smoke \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 2048 \
    --enforce-eager >"$log_file" 2>&1 &
elif [[ "$backend" == "sglang" ]]; then
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
  "$sglang_python" -m sglang.launch_server \
    --model-path "$model" \
    --host 127.0.0.1 --port "$port" \
    --served-model-name smoke \
    --tp-size 1 \
    --mem-fraction-static 0.45 \
    --context-length 2048 \
    --disable-cuda-graph >"$log_file" 2>&1 &
else
  echo "usage: $0 {vllm|sglang} [model_path] [port]" >&2
  exit 2
fi
server_pid=$!

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited before health check" >&2
    tail -80 "$log_file" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  echo "health check timed out" >&2
  tail -80 "$log_file" >&2
  exit 1
fi

echo "backend=$backend health=ok"
curl -fsS "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke","messages":[{"role":"user","content":"Reply with exactly: KAIRO_OK"}],"max_tokens":8,"temperature":0}'
echo
if [[ "${KAIRO_KEEP_ALIVE:-0}" == "1" ]]; then
  echo "keep_alive=1; press Ctrl-C to stop"
  while kill -0 "$server_pid" 2>/dev/null; do
    sleep 5
  done
fi
