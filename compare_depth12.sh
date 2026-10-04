#!/usr/bin/env bash
# Matched, forced-loop comparison; no training or learned stopping.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash compare_depth12.sh' >&2
  exit 2
fi
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run=models/stage1_pointer/depth12-fixed-prompt-seed47-gaps
data=data/pointer/seed-47-depth12-gaps/depth_test.jsonl
for step in 004500 005000 005500 007500; do
  for file in recurrent_config.json adapter_model.pt tokenizer.json; do
    [[ -f "$run/step-$step/$file" ]] || { echo "Missing $run/step-$step/$file" >&2; exit 1; }
  done
done
[[ -f "$data" ]] || { echo "Missing $data" >&2; exit 1; }
output="eval/pointer_loops/depth12-checkpoint-comparison-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$output"
compare() {
  echo "Results: $output"
  for step in 004500 005000 005500 007500; do
    python -u -m scripts.eval.loop_test \
      --model "$run/step-$step" --data "$data" \
      --device cuda --batch-size 16 --loops 24 \
      --output "$output/step-$step" || return $?
  done
  python -u -m scripts.eval.compare_checkpoints --results "$output" || return $?
}
compare 2>&1 | tee "$output/run.log"
