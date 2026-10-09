#!/usr/bin/env bash
# One finite desktop-run follow-up: copy completed evidence and retire old renders.
set -euo pipefail
cd "$(dirname "$0")/../.."
output="eval/pointer_analysis/paper-20261009"
remote_root="/home/mukund/loopformer/$output"
while true; do
  if ssh -o BatchMode=yes -o ConnectTimeout=10 desktop "test -f '$remote_root/COMPLETE'" 2>/dev/null; then
    state=complete
    break
  fi
  if ssh -o BatchMode=yes -o ConnectTimeout=10 desktop "test -f '$remote_root/FAILED'" 2>/dev/null; then
    state=failed
    break
  fi
  sleep 60
done
mkdir -p "$output"
rsync -a --partial -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' "desktop:$remote_root/" "$output/"
if [[ "$state" == failed ]]; then
  .venv/bin/python - <<'PY'
from pathlib import Path
from scripts.eval.paper_report import replace_status
message='The desktop suite **failed**. Read `eval/pointer_analysis/paper-20261009/run.log` and `FAILED`. Recoverable chunks are retained; no complete-population claim or clean-export retirement is made.'
for path in (Path('docs/analysis.md'), Path('docs/status.md')):
    replace_status(path,message)
PY
  printf 'Suite failed; partial evidence copied. Existing render exports retained.\n'
  exit 1
fi
.venv/bin/python -m scripts.eval.retire_pointer_plots --input "$output"
.venv/bin/python -m scripts.eval.paper_report --input "$output"
printf 'Audited results and clean figures are now on the Mac: %s/plots/index.md\n' "$output"
