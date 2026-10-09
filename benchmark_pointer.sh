#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
full=false
preview=false
for option in "$@"; do
  case "$option" in
    --full) full=true ;;
    --dry-run) preview=true ;;
    *) echo 'Usage: bash benchmark_pointer.sh [--full] [--dry-run]' >&2; exit 2 ;;
  esac
done
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
config=configs/pointer_benchmark.json
root=eval/pointer_benchmark/frozen-20261008
dataset=data/pointer/benchmark-seeds211-223-227
model=models/stage1_pointer/controller-affine-seed83/best
if $full; then
  config=configs/pointer_full_benchmark.json
  root=eval/pointer_benchmark/final-full-20261008
  dataset=data/pointer/benchmark-seeds307-311-313
  model=models/stage1_pointer/controller-shared-number-precision-seed83/best
fi
if $preview; then
  exec python -m scripts.eval.frozen_pointer_benchmark --config "$config" --dry-run
fi
mkdir -p "$root"
{
  python -m scripts.eval.freeze_pointer_benchmark --config "$config" --output "$root/freeze.json"
  if [[ ! -f "$root/existing_test/summary.json" ]]; then
    python -u -m scripts.eval.loop_test \
      --model "$model" \
      --data data/pointer/seed-61-independent/test.jsonl \
      --device cuda --attention sdpa --loops 272 --stop-policy completion --stop-threshold .5 \
      --output "$root/existing_test" --wandb-mode disabled
  fi
  python -m scripts.eval.freeze_pointer_benchmark --config "$config" --output "$root/freeze.json" \
    --verify-result "$root/existing_test/summary.json"
  python -m scripts.eval.audit_pointer_benchmark --native-results "$root/existing_test" \
    --native-data data/pointer/seed-61-independent/test.jsonl
  python -u -m scripts.eval.frozen_pointer_benchmark --config "$config" \
    --freeze "$root/freeze.json" --output "$root/independent" --dataset-output "$dataset" --wandb-mode disabled
  python -m scripts.eval.audit_pointer_benchmark --results "$root/independent" --graphs "$dataset/graphs.jsonl"
  python -m scripts.eval.audit_pointer_benchmark --native-results "$root/independent/native" \
    --native-data "$root/independent/native_tasks.jsonl"
} 2>&1 | tee "$root/benchmark-launch.log"
