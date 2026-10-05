#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-loopformer}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for arg in "$@"; do
  if [[ "$arg" == --dry-run || "$arg" == --help ]]; then
    exec python -u -m scripts.eval.controller_training_audit "$@"
  fi
done
mkdir -p eval/pointer_diagnostics
log="eval/pointer_diagnostics/controller-training-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
python -u -m scripts.eval.controller_training_audit "$@" 2>&1 | tee "$log"
