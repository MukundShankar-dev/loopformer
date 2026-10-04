#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-loopformer}"
if [[ $# -gt 1 || ( $# == 1 && $1 != --dry-run ) ]]; then
  echo 'Usage: bash probe_depth12.sh [--dry-run]' >&2
  exit 2
fi
run=models/stage1_pointer/depth12-fixed-prompt-seed47-gaps
data=data/pointer/seed-47-depth12-gaps/depth_test.jsonl
if [[ ${1:-} == --dry-run ]]; then
  exec python -m scripts.eval.paired_steps --model "$run/step-005000" --data "$data" \
    --depths 12 14 16 18 20 --loops 24 --count 32 --dry-run
fi
for step in 005000 007500; do
  for file in recurrent_config.json adapter_model.pt tokenizer.json; do
    [[ -f "$run/step-$step/$file" ]] || { echo "Missing $run/step-$step/$file" >&2; exit 1; }
  done
done
output="eval/pointer_probes/depth12-paired-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$output"
probe() {
  for step in 005000 007500; do
    python -u -m scripts.eval.paired_steps --model "$run/step-$step" --data "$data" \
      --depths 12 14 16 18 20 --loops 24 --count 32 --batch-size 8 \
      --output "$output/step-$step" || return $?
  done
}
probe 2>&1 | tee "$output/run.log"
