#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Use `bash probe_controls.sh full` only after reviewing the small run.
mode="${1:-small}"
case "$mode" in
  small) limit=32 ;;
  full) limit=1000 ;;
  *) echo 'Usage: bash probe_controls.sh [small|full]' >&2; exit 2 ;;
esac
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="eval/pointer_probes/controls-$mode-$run_id"
mkdir -p "$output"
run_controls() {
  source .venv/bin/activate
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  local checkpoint=models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500
  for file in recurrent_config.json adapter_model.pt tokenizer.json; do
    if [[ ! -f "$checkpoint/$file" ]]; then
      echo "Missing checkpoint file: $PWD/$checkpoint/$file" >&2
      return 1
    fi
  done
  echo "Checkpoint: $checkpoint; mode: $mode; outputs/log: $output"
  for split in validation depth_test; do
    python -u -m scripts.eval.probe_pointer \
      --model "$checkpoint" --data "data/pointer/seed-17/$split.jsonl" \
      --device cuda --controls --restart-after 6 --limit "$limit" \
      --output "$output/$split" || return $?
  done
}
run_controls 2>&1 | tee "$output/run.log"
