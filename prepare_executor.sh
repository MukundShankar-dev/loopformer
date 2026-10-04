#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo 'Usage: bash prepare_executor.sh [--dry-run]' >&2; exit 2
fi
source .venv/bin/activate
output=data/pointer/seed-61-independent
if [[ $# == 0 && -e "$output" ]]; then
  exec python -m scripts.dataset --verify "$output"
fi
exec python -m scripts.dataset --seed 61 --graph-mode mixture --paired-horizons \
  --train-count 36000 --validation-count 1536 --test-count 1536 --depth-test-count 1664 \
  --max-train-depth 12 --train-depths 1 2 3 4 5 6 8 10 12 --max-eval-depth 64 \
  --output "$output" "$@"
