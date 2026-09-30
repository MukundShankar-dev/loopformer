#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="eval/pointer_mechanism/baseline2500-$run_id"
mkdir -p "$output"
run_diagnostic() {
  source .venv/bin/activate
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  local checkpoint=models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500
  for file in recurrent_config.json adapter_model.pt tokenizer.json; do
    if [[ ! -f "$checkpoint/$file" ]]; then
      echo "Missing checkpoint file: $PWD/$checkpoint/$file" >&2
      return 1
    fi
  done
  echo "Checkpoint: $checkpoint; cohorts: seed-17 validation/depth_test; outputs/log: $output"
  python -u -m scripts.eval.mechanism_diagnostic \
    --model "$checkpoint" --device cuda \
    --validation data/pointer/seed-17/validation.jsonl \
    --deep data/pointer/seed-17/depth_test.jsonl \
    --fit-per-depth 24 --eval-per-depth 8 --deep-per-depth 8 \
    --output "$output/results"
}
run_diagnostic 2>&1 | tee "$output/run.log"
