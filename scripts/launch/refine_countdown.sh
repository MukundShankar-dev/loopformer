#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
preview=()
if [[ ${1:-} == --dry-run ]]; then
  preview=(--dry-run)
elif [[ $# -gt 0 ]]; then
  echo 'Usage: bash scripts/launch/refine_countdown.sh [--dry-run]' >&2
  exit 2
fi
python -m scripts.training.refine_countdown ${preview[@]+"${preview[@]}"}
python -m scripts.eval.countdown_precision_test ${preview[@]+"${preview[@]}"}
