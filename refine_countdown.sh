#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
preview=()
if [[ ${1:-} == --dry-run ]]; then
  preview=(--dry-run)
elif [[ $# -gt 0 ]]; then
  echo 'Usage: bash refine_countdown.sh [--dry-run]' >&2
  exit 2
fi
python -m scripts.training.refine_countdown "${preview[@]}"
python -m scripts.eval.countdown_precision_test "${preview[@]}"
