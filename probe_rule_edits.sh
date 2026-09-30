#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="eval/pointer_rule_edits/baseline2500-$run_id"
mkdir -p "$output"
run_probe() {
  source .venv/bin/activate
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  local checkpoint=models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500
  for file in recurrent_config.json adapter_model.pt tokenizer.json; do
    if [[ ! -f "$checkpoint/$file" ]]; then
      echo "Missing checkpoint file: $PWD/$checkpoint/$file" >&2
      return 1
    fi
  done
  echo "Checkpoint: $checkpoint; dataset: seed-17 depth_test; outputs/log: $output"
  python -u -m scripts.eval.rule_edit_probe \
    --model "$checkpoint" --data data/pointer/seed-17/depth_test.jsonl \
    --device cuda --limit 32 --early-loop 6 --output "$output/results"
}
run_probe 2>&1 | tee "$output/run.log"
