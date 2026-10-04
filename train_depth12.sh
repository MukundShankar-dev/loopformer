#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run && "$1" != --smoke-test ) ]]; then
  echo 'Usage: bash train_depth12.sh [--dry-run|--smoke-test]' >&2
  exit 2
fi
if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing virtual environment: $PWD/.venv" >&2
  exit 1
fi
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
config=configs/stage1_pointer_depth12_gaps.json
run_dir=models/stage1_pointer/depth12-fixed-prompt-seed47-gaps

if [[ ${1:-} == --dry-run ]]; then
  exec python -u -m scripts.training.train_pointer --config "$config" \
    --device cuda --output "$run_dir" --dry-run
fi
extra=""
if [[ ${1:-} == --smoke-test ]]; then
  run_dir="${run_dir}-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  extra="--smoke-test"
fi
if [[ -e "$run_dir" ]]; then
  echo "Output already exists: $run_dir. Refusing to overwrite or resume it." >&2
  exit 1
fi
if [[ $(uname -s) != Linux ]] || ! command -v script >/dev/null; then
  echo 'This CUDA launcher requires Linux/WSL and the util-linux script command.' >&2
  exit 1
fi
mkdir -p models/stage1_pointer
log="${run_dir}-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Training: $run_dir"
echo "Terminal log: $log"
# A PTY preserves the live Rich dashboard while recording stdout and stderr.
# --return propagates training failure; paths below are fixed repo-relative literals.
exec script --quiet --return --flush --command \
  "python -u -m scripts.training.train_pointer --config $config --device cuda --output $run_dir $extra" "$log"
