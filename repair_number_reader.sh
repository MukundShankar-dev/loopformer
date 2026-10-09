#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
preview=()
if [[ ${1:-} == --dry-run ]]; then
  preview=(--dry-run)
elif [[ $# -gt 0 ]]; then
  echo 'Usage: bash repair_number_reader.sh [--dry-run]' >&2
  exit 2
fi
python -m scripts.training.repair_number_reader "${preview[@]}"
if [[ ${#preview[@]} -gt 0 ]]; then
  python -m scripts.eval.frozen_pointer_benchmark \
    --config configs/pointer_number_reader_benchmark.json --dry-run
  exit 0
fi
root=eval/pointer_benchmark/shared-number-20261008
python -m scripts.eval.freeze_pointer_benchmark \
  --config configs/pointer_number_reader_benchmark.json --output "$root/freeze.json"
python -m scripts.eval.frozen_pointer_benchmark \
  --config configs/pointer_number_reader_benchmark.json --freeze "$root/freeze.json" \
  --output "$root/independent" --dataset-output data/pointer/benchmark-seeds281-283-293 \
  --wandb-mode disabled
python -m scripts.eval.number_reader_diagnostic --output "$root/numeric"
python -m scripts.eval.audit_pointer_benchmark --results "$root/independent" \
  --graphs data/pointer/benchmark-seeds281-283-293/graphs.jsonl
