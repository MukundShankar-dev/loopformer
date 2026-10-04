#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export POINTER_DATA_ROOT=data/pointer/seed-47-depth12-gaps
export POINTER_LOOPS=24
export POINTER_COHORTS=depth12
if [[ $# == 0 ]]; then
  set -- models/stage1_pointer/depth12-fixed-prompt-seed47-gaps
fi
exec bash eval_ckpts.sh "$@"
