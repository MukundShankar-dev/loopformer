#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -gt 1 || ($# -eq 1 && $1 != --dry-run) ]]; then
  echo 'Usage: bash repair_controller_prefix.sh [--dry-run]' >&2; exit 2
fi
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-loopformer}"
if [[ ${1:-} == --dry-run ]]; then
  python -u -m scripts.training.prepare_controller_prefix --dry-run
  exec python -u -m scripts.training.train_controller --config configs/controller_prefix.json --dry-run
fi
mkdir -p models/stage1_pointer
run=models/stage1_pointer/controller-prefix-seed83
cache=models/stage1_pointer/controller-prefix-features.pt
log="models/stage1_pointer/controller-prefix-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
{
  python -u -m scripts.training.prepare_controller_prefix --output "$cache"
  python -u -m scripts.training.train_controller --config configs/controller_prefix.json \
    --features-cache "$cache" --output "$run"
  python -u -m scripts.eval.controller_candidate --run "$run" \
    --deep-features-cache eval/pointer_diagnostics/controller-remaining-seed61/deep_features.pt
} 2>&1 | tee "$log"
