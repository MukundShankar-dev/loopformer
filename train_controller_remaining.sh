#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run && "$1" != --smoke-test ) ]]; then
  echo 'Usage: bash train_controller_remaining.sh [--dry-run|--smoke-test]' >&2; exit 2
fi
source .venv/bin/activate
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-loopformer}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ${1:-} == --dry-run ]]; then
  exec python -u -m scripts.training.compare_controllers --dry-run
fi
mkdir -p models/stage1_pointer
log="models/stage1_pointer/controller-remaining-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
python -u -m scripts.training.compare_controllers "$@" 2>&1 | tee "$log"
