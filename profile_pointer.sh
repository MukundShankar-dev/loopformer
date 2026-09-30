#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_profiles() {
  source .venv/bin/activate
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  POINTER_CHECKPOINT=models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

  for file in recurrent_config.json adapter_model.pt; do
    if [[ ! -f "$POINTER_CHECKPOINT/$file" ]]; then
      echo "Missing checkpoint file: $PWD/$POINTER_CHECKPOINT/$file" >&2
      exit 1
    fi
  done

  echo "Checkpoint: $POINTER_CHECKPOINT"
  echo "Run ID: $RUN_ID"
  echo "Log: $PWD/profile-pointer-debug.txt"

  for batch in 4 8; do
    python -m scripts.eval.profile_pointer \
      --model "$POINTER_CHECKPOINT" \
      --data data/pointer/seed-37-depth6-30k/train.jsonl \
      --device cuda --mode train \
      --batch-size "$batch" --effective-batch 8 --train-max-depth 6 \
      --output "eval/pointer_profiles/baseline2500-$RUN_ID/train-b$batch"
  done

  python -m scripts.eval.probe_pointer \
    --model "$POINTER_CHECKPOINT" \
    --data data/pointer/seed-17/depth_test.jsonl \
    --device cuda --limit 32 --restart-after 6 \
    --output "eval/pointer_probes/baseline2500-$RUN_ID"
}

run_profiles 2>&1 | tee profile-pointer-debug.txt
