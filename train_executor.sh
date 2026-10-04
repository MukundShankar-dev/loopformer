#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run && "$1" != --smoke-test ) ]]; then
  echo 'Usage: bash train_executor.sh [--dry-run|--smoke-test]' >&2; exit 2
fi
source .venv/bin/activate
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-loopformer}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
config="${EXECUTOR_CONFIG:-configs/executor_r.json}"
run="${EXECUTOR_RUN:-models/stage1_pointer/$(basename -- "$config" .json)-seed61}"
if [[ ${1:-} == --dry-run ]]; then
  exec python -u -m scripts.training.train_pointer --config "$config" --output "$run" --dry-run
fi
extra=()
if [[ ${1:-} == --smoke-test ]]; then
  run="${run}-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  extra=(--smoke-test)
fi
if [[ -e "$run" ]]; then
  echo "Output exists: $run. Choose a fresh EXECUTOR_RUN; no automatic resume." >&2; exit 1
fi
if [[ $(uname -s) != Linux ]] || ! command -v script >/dev/null; then
  echo 'This CUDA launcher requires Linux/WSL with util-linux script.' >&2; exit 1
fi
mkdir -p models/stage1_pointer
log="${run}-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
# Bash %q prevents config/output paths from becoming shell code in script -c.
printf -v command '%q ' python -u -m scripts.training.train_pointer --config "$config" --output "$run" "${extra[@]}"
echo "Training: $run"
echo "Terminal log: $log"
exec script --quiet --return --flush --command "$command" "$log"
