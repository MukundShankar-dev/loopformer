#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="eval/pointer_mechanism/matched-2500-3250-3750-$run_id"
mkdir -p "$output"
run_diagnostic() {
  source .venv/bin/activate
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  local root=models/stage1_pointer/depth6-fresh30k-seed37-batch4
  local checkpoint=$root/step-002500
  local comparison_one=$root/step-003250
  local comparison_two=$root/step-003750
  for directory in "$checkpoint" "$comparison_one" "$comparison_two"; do
    for file in recurrent_config.json adapter_model.pt tokenizer.json; do
      if [[ ! -f "$directory/$file" ]]; then
        echo "Missing checkpoint file: $PWD/$directory/$file" >&2
        return 1
      fi
    done
  done
  echo "Checkpoints: $checkpoint $comparison_one $comparison_two; cohorts: seed-17 validation/depth_test; outputs/log: $output"
  python -u -m scripts.eval.mechanism_diagnostic \
    --model "$checkpoint" --device cuda \
    --comparison-model "$comparison_one" --comparison-model "$comparison_two" \
    --validation data/pointer/seed-17/validation.jsonl \
    --deep data/pointer/seed-17/depth_test.jsonl \
    --fit-per-depth 24 --eval-per-depth 8 --deep-per-depth 8 \
    --output "$output/results"
}
run_diagnostic 2>&1 | tee "$output/run.log"
