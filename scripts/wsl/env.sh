#!/usr/bin/env bash
# User-overridable local paths. Keep machine-specific values out of Git.
: "${KAIRO_MODEL_DIR:=${HOME}/kairo-models}"
: "${KAIRO_GPU_VENV:=${HOME}/venv-gpu}"
: "${KAIRO_VLLM_NIGHTLY_VENV:=${HOME}/venv-vllm-nightly}"
: "${KAIRO_SGLANG_ENV:=${HOME}/venv-sglang}"
export KAIRO_MODEL_DIR KAIRO_GPU_VENV KAIRO_VLLM_NIGHTLY_VENV KAIRO_SGLANG_ENV
