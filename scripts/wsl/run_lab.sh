#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
python_bin="python3"
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  python_bin=".venv/bin/python"
elif [[ -x /home/peter/venv-gpu/bin/python ]]; then
  python_bin="/home/peter/venv-gpu/bin/python"
fi
PYTHONPATH=src "$python_bin" -m kairo_lab.cli "$@"
