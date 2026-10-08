#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -gt 1 || ($# -eq 1 && $1 != --dry-run) ]]; then
  echo 'Usage: bash benchmark_pointer.sh [--dry-run]' >&2; exit 2
fi
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ${1:-} == --dry-run ]]; then
  exec python -m scripts.eval.frozen_pointer_benchmark --dry-run
fi
root=eval/pointer_benchmark/frozen-20261008
mkdir -p "$root"
{
  python -m scripts.eval.freeze_pointer_benchmark
  if [[ ! -f "$root/existing_test/summary.json" ]]; then
    python -u -m scripts.eval.loop_test \
      --model models/stage1_pointer/controller-affine-seed83/best \
      --data data/pointer/seed-61-independent/test.jsonl \
      --device cuda --attention sdpa --loops 272 --stop-policy completion --stop-threshold .5 \
      --output "$root/existing_test" --wandb-mode disabled
  fi
  python -m scripts.eval.freeze_pointer_benchmark --verify-result "$root/existing_test/summary.json"
  python -u -m scripts.eval.frozen_pointer_benchmark --wandb-mode disabled
} 2>&1 | tee "$root/benchmark-launch.log"
