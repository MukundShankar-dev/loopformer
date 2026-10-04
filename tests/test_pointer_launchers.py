"""Desktop launchers preserve logs, select the intended artifact and fail fast."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def launcher_repo(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ('train_fixed_prompt.sh', 'eval_ckpts.sh'):
        shutil.copy(root / name, tmp_path / name)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    activation = tmp_path / '.venv/bin'
    activation.mkdir(parents=True)
    (activation / 'activate').write_text(f'export PATH={shlex.quote(str(bin_dir))}:"$PATH"\n')
    python = bin_dir / 'python'
    python.write_text(f'#!{sys.executable}\n' + '''import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-':
    os.execv(sys.executable, [sys.executable, *args])
with Path('calls.jsonl').open('a') as f:
    f.write(json.dumps(args)+'\\n')
if 'scripts.training.train_pointer' in args:
    if '--dry-run' not in args:
        Path(args[args.index('--output')+1]).mkdir(parents=True)
    print('training output', flush=True)
    print('training stderr', file=sys.stderr, flush=True)
    sys.exit(int(os.environ.get('FAIL_TRAIN', '0')))
count = len(Path('calls.jsonl').read_text().splitlines())
if count == int(os.environ.get('FAIL_CALL', '-1')):
    print('eval failed', file=sys.stderr)
    sys.exit(7)
out = Path(args[args.index('--output')+1]); out.mkdir(parents=True)
if '--stop-policy' in args:
    summary = dict(accuracy=.5, mean_loops=3, completion=dict(exact_stop_rate=.5, cap_fallback_rate=.5))
else:
    summary = dict(trajectory_accuracy=.5, final_accuracy=.5, completion=dict(first_stop_exact_rate=.5,
        first_stop_early_rate=.25, first_stop_late_or_missing_rate=.25))
(out/'summary.json').write_text(json.dumps(summary))
''')
    python.chmod(0o755)
    # Emulate util-linux script on the Mac test host; the real command supplies a PTY.
    script = bin_dir / 'script'
    script.write_text(f'#!{sys.executable}\n' + '''import subprocess, sys
args=sys.argv[1:]
result=subprocess.run(args[args.index('--command')+1], shell=True, capture_output=True, text=True)
text=result.stdout+result.stderr
open(args[-1], 'w').write(text)
print(text, end='')
sys.exit(result.returncode)
''')
    script.chmod(0o755)
    (bin_dir / 'uname').write_text('#!/bin/sh\nprintf "Linux\\n"\n')
    (bin_dir / 'uname').chmod(0o755)
    # The metadata snippet is real Python; model loading is covered by CLI tests.
    package = tmp_path / 'scripts/recurrent_qwen'
    package.mkdir(parents=True)
    (package / 'checkpoint.py').write_text('def checkpoint_mode(spec):\n    return spec.get("recurrence_mode", "full_sequence")\n')
    run = tmp_path / 'models/stage1_pointer/depth6-fixed-prompt-seed37'
    checkpoint = run / 'step-003250'
    checkpoint.mkdir(parents=True)
    (run / 'best_checkpoint.json').write_text(json.dumps({'path': 'step-003250'}))
    (checkpoint / 'recurrent_config.json').write_text(json.dumps({'format': 'loopformer-stage1-fixed-prompt-v1',
        'recurrence_mode': 'fixed_prompt', 'completion_head': {'intermediate': 128}}))
    for name in ('adapter_model.pt', 'tokenizer.json'):
        (checkpoint / name).write_text('stub')
    data = tmp_path / 'data/pointer/seed-17'
    data.mkdir(parents=True)
    for split in ('validation', 'depth_test'):
        (data / f'{split}.jsonl').write_text('stub')
    return tmp_path, run


@pytest.mark.parametrize('fail_call', [None, 1, 2])
def test_eval_launcher_uses_selected_checkpoint_and_stops_on_failure(launcher_repo, fail_call):
    root, run = launcher_repo
    env = {**os.environ, **({'FAIL_CALL': str(fail_call)} if fail_call else {})}
    result = subprocess.run(['bash', 'eval_ckpts.sh'], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == (7 if fail_call else 0), result.stderr
    calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == (fail_call or 4)
    for i, args in enumerate(calls):
        assert args[args.index('--model') + 1] == str(run / 'step-003250')
        assert args[args.index('--loops') + 1] == '20'
        assert ('--stop-policy' in args) == (i % 2 == 1)
    logs = list((root / 'eval/pointer_loops').glob('*/run.log'))
    assert len(logs) == 1
    if fail_call:
        assert 'eval failed' in logs[0].read_text()


def test_training_launcher_preview_logs_and_no_overwrite(launcher_repo):
    root, run = launcher_repo
    shutil.rmtree(run)
    preview = subprocess.run(['bash', 'train_fixed_prompt.sh', '--dry-run'], cwd=root, capture_output=True, text=True)
    assert preview.returncode == 0 and not run.exists()
    assert not list(run.parent.glob('*launch*'))
    result = subprocess.run(['bash', 'train_fixed_prompt.sh'], cwd=root,
                            env={**os.environ, 'FAIL_TRAIN': '9'}, capture_output=True, text=True)
    assert result.returncode == 9
    logs = list(run.parent.glob('*launch*.log'))
    assert len(logs) == 1 and 'training stderr' in logs[0].read_text()
    before = (root / 'calls.jsonl').read_text()
    repeated = subprocess.run(['bash', 'train_fixed_prompt.sh'], cwd=root, capture_output=True, text=True)
    assert repeated.returncode != 0 and 'already exists' in repeated.stderr
    assert (root / 'calls.jsonl').read_text() == before
