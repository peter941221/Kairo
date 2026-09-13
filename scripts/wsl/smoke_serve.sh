#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
model="${2:-/home/peter/kairo-models/Qwen2.5-0.5B-Instruct}"
port="${3:-18080}"
vllm_bin="${KAIRO_VLLM_BIN:-/home/peter/venv-gpu/bin/vllm}"
sglang_python="${KAIRO_SGLANG_PYTHON:-/home/peter/venv-gpu/bin/python}"
sglang_pythonpath="${KAIRO_SGLANG_PYTHONPATH:-}"
gpu_memory_utilization="${KAIRO_GPU_MEMORY_UTILIZATION:-0.45}"
max_model_len="${KAIRO_MAX_MODEL_LEN:-2048}"
context_length="${KAIRO_CONTEXT_LENGTH:-2048}"
kv_cache_dtype="${KAIRO_KV_CACHE_DTYPE:-}"
trust_remote_code="${KAIRO_TRUST_REMOTE_CODE:-0}"
skip_mm_profiling="${KAIRO_SKIP_MM_PROFILING:-0}"
language_model_only="${KAIRO_LANGUAGE_MODEL_ONLY:-0}"
sglang_qwen38="${KAIRO_SGLANG_QWEN38_FLAGS:-0}"
disable_flashinfer_autotune="${KAIRO_DISABLE_FLASHINFER_AUTOTUNE:-0}"
health_timeout="${KAIRO_HEALTH_TIMEOUT:-120}"
skip_server_warmup="${KAIRO_SKIP_SERVER_WARMUP:-0}"
mamba_ssm_dtype="${KAIRO_MAMBA_SSM_DTYPE:-float32}"
mamba_full_memory_ratio="${KAIRO_MAMBA_FULL_MEMORY_RATIO:-4.59}"
fp4_gemm_backend="${KAIRO_FP4_GEMM_BACKEND:-}"
fp8_gemm_backend="${KAIRO_FP8_GEMM_BACKEND:-}"
disable_thinking="${KAIRO_DISABLE_THINKING:-0}"
linear_backend="${KAIRO_LINEAR_BACKEND:-}"
moe_backend="${KAIRO_MOE_BACKEND:-}"
max_running_requests="${KAIRO_MAX_RUNNING_REQUESTS:-}"
num_continuous_decode_steps="${KAIRO_NUM_CONTINUOUS_DECODE_STEPS:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
  vllm_args=(serve "$model" --host 127.0.0.1 --port "$port"
    --served-model-name smoke --tensor-parallel-size 1
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len" --enforce-eager)
  [[ "$trust_remote_code" == "1" ]] && vllm_args+=(--trust-remote-code)
  [[ -n "$kv_cache_dtype" ]] && vllm_args+=(--kv-cache-dtype "$kv_cache_dtype")
  [[ "$skip_mm_profiling" == "1" ]] && vllm_args+=(--skip-mm-profiling)
  [[ "$language_model_only" == "1" ]] && vllm_args+=(--language-model-only)
  [[ -n "$linear_backend" ]] && vllm_args+=(--linear-backend "$linear_backend")
  [[ -n "$moe_backend" ]] && vllm_args+=(--moe-backend "$moe_backend")
  [[ -n "$max_running_requests" ]] && vllm_args+=(--max-num-seqs "$max_running_requests")
  [[ "$disable_flashinfer_autotune" == "1" ]] && vllm_args+=(--no-enable-flashinfer-autotune)
  "$vllm_bin" "${vllm_args[@]}" >"$log_file" 2>&1 &
elif [[ "$backend" == "vllm-nightly" ]]; then
  nightly_python="${KAIRO_VLLM_NIGHTLY_PYTHON:-/home/peter/venv-vllm-nightly/bin/python}"
  nightly_launcher="${KAIRO_VLLM_NIGHTLY_LAUNCHER:-$root/scripts/wsl/vllm_nightly.py}"
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH="$CUDA_HOME/bin:$PATH"
  # The nightly wheel/Torch and B12X use their own CUDA libraries; the
  # remaining Python CUDA support is reused from the pinned GPU environment.
  export LD_LIBRARY_PATH="/home/peter/venv-vllm-nightly/lib/python3.12/site-packages/nvidia/nvshmem/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cublas/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/cusparselt/lib:/home/peter/venv-gpu/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
  nightly_args=(serve "$model" --host 127.0.0.1 --port "$port"
    --served-model-name smoke --tensor-parallel-size 1
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len" --enforce-eager
    --generation-config vllm)
  [[ "$trust_remote_code" == "1" ]] && nightly_args+=(--trust-remote-code)
  [[ -n "$kv_cache_dtype" ]] && nightly_args+=(--kv-cache-dtype "$kv_cache_dtype")
  [[ "$skip_mm_profiling" == "1" ]] && nightly_args+=(--skip-mm-profiling)
  [[ "$language_model_only" == "1" ]] && nightly_args+=(--language-model-only)
  [[ -n "$linear_backend" ]] && nightly_args+=(--linear-backend "$linear_backend")
  [[ -n "$moe_backend" ]] && nightly_args+=(--moe-backend "$moe_backend")
  [[ -n "$max_running_requests" ]] && nightly_args+=(--max-num-seqs "$max_running_requests")
  "$nightly_python" "$nightly_launcher" "${nightly_args[@]}" >"$log_file" 2>&1 &
elif [[ "$backend" == "sglang" ]]; then
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
  if [[ -n "$sglang_pythonpath" ]]; then
    export PYTHONPATH="$sglang_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
  fi
  sglang_args=( -m sglang.launch_server --model-path "$model"
    --host 127.0.0.1 --port "$port" --served-model-name smoke --tp-size 1
    --mem-fraction-static "$gpu_memory_utilization" --context-length "$context_length"
    --disable-cuda-graph)
  if [[ "$sglang_qwen38" == "1" ]]; then
    sglang_args+=(--chunked-prefill-size 2048 --mamba-full-memory-ratio "$mamba_full_memory_ratio"
      --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype "$mamba_ssm_dtype")
  fi
  [[ "$disable_flashinfer_autotune" == "1" ]] && sglang_args+=(--disable-flashinfer-autotune)
  [[ -n "$fp4_gemm_backend" ]] && sglang_args+=(--fp4-gemm-backend "$fp4_gemm_backend")
  [[ -n "$fp8_gemm_backend" ]] && sglang_args+=(--fp8-gemm-backend "$fp8_gemm_backend")
  [[ "$skip_server_warmup" == "1" ]] && sglang_args+=(--skip-server-warmup)
  [[ "$trust_remote_code" == "1" ]] && sglang_args+=(--trust-remote-code)
  [[ -n "$kv_cache_dtype" ]] && sglang_args+=(--kv-cache-dtype "$kv_cache_dtype")
  [[ -n "$max_running_requests" ]] && sglang_args+=(--max-running-requests "$max_running_requests")
  [[ -n "$num_continuous_decode_steps" ]] && sglang_args+=(--num-continuous-decode-steps "$num_continuous_decode_steps")
  "$sglang_python" "${sglang_args[@]}" >"$log_file" 2>&1 &
else
  echo "usage: $0 {vllm|vllm-nightly|sglang} [model_path] [port]" >&2
  exit 2
fi
server_pid=$!

for _ in $(seq 1 "$health_timeout"); do
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
request_body='{"model":"smoke","messages":[{"role":"user","content":"Reply with exactly: KAIRO_OK"}],"max_tokens":16,"temperature":0}'
if [[ "$disable_thinking" == "1" ]]; then
  request_body='{"model":"smoke","messages":[{"role":"user","content":"Reply with exactly: KAIRO_OK"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
fi
curl -fsS "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$request_body"
echo
if [[ "${KAIRO_KEEP_ALIVE:-0}" == "1" ]]; then
  echo "keep_alive=1; press Ctrl-C to stop"
  while kill -0 "$server_pid" 2>/dev/null; do
    sleep 5
  done
fi
