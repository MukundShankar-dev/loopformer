import json
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from scripts.eval import paper_status
from scripts.eval.paper_watch import dashboard, fetch_status


def panel(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_status, '_active_process', lambda root: {})
    config = dict(graph_count=4, models=[dict(name='ce_full'), dict(name='joint_full')],
                  final_executor_history=dict(retained_weight_steps=[0, 250]))
    (tmp_path / 'protocol.json').write_text(json.dumps(config))
    return tmp_path


def chunk(root, arm, name, ids, seconds=None):
    directory = root / arm
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(graph_indices=np.array(ids, dtype=np.int64))
    if seconds is not None:
        arrays['seconds'] = np.array(seconds)
    np.savez(directory / name, **arrays)
    return directory


def test_partial_chunk_is_excluded_and_pause_is_not_failure(tmp_path, monkeypatch):
    root = panel(tmp_path, monkeypatch)
    directory = chunk(root, 'ce_full', 'graphs-0000-0002.npz', [0, 1], 20)
    chunk(root, 'ce_full', 'graphs-0002-0004.tmp.npz', [2, 3], 20)
    (directory / 'native_fidelity.json').write_text('{"passed": false}')
    (root / 'PAUSED').touch()
    result = paper_status.suite_status(root)
    assert result['status'] == 'paused'
    assert result['models'][0]['graphs'] == 2
    assert result['models'][0]['remaining_inference_seconds'] == 20
    assert not result['models'][0]['native_checked']
    assert result['models'][0]['state'] == 'partial'


def test_snapshot_coverage_and_loading_stage_are_not_reported_complete(tmp_path, monkeypatch):
    root = panel(tmp_path, monkeypatch)
    directory = chunk(root, 'executor_snapshots/step-000000', 'graphs-0000-0002.npz', [0, 1])
    (directory / 'summary.json').write_text('{"status": "partial"}')
    (root / 'RUNNING').touch()
    joint = root / 'joint_full'
    joint.mkdir()
    (joint / 'freeze.json').write_text('{}')
    monkeypatch.setattr(paper_status, '_active_process', lambda root: dict(module='paper_suite', label='Model extraction', pid=123))
    result = paper_status.suite_status(root)
    assert result['snapshots_complete'] == 0
    assert result['snapshots'][0]['graphs'] == 2
    assert result['models'][1]['state'] == 'loading / native checks'
    assert 'joint_full' in result['activity']
    assert result['stages'][1]['state'] == 'running'
    (root / 'independent_audit.json').write_text('{"passed": false}')
    assert not paper_status.suite_status(root)['independently_audited']


def test_overlapping_chunks_cannot_inflate_progress(tmp_path, monkeypatch):
    root = panel(tmp_path, monkeypatch)
    chunk(root, 'ce_full', 'graphs-0000-0002.npz', [0, 1])
    chunk(root, 'ce_full', 'graphs-0001-0003.npz', [1, 2])
    with pytest.raises(ValueError, match='Duplicate'):
        paper_status.suite_status(root)


def test_remote_status_is_read_only_and_shell_quotes_paths(monkeypatch):
    import subprocess
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"status":"paused","models":[]}', '')
    monkeypatch.setattr(subprocess, 'run', run)
    assert fetch_status(Path('eval/a b'), 'desktop', '/home/a b')['status'] == 'paused'
    assert "cd '/home/a b'" in calls[0][-1]
    assert "--input 'eval/a b' --json" in calls[0][-1]
    assert 'paper_status' in calls[0][-1] and 'analyze_pointer.sh' not in calls[0][-1]
    with pytest.raises(ValueError):
        fetch_status(Path('eval'), '-oProxyCommand=bad', '/home')


def test_dashboard_labels_partial_coverage_and_eta_scope(tmp_path, monkeypatch):
    root = panel(tmp_path, monkeypatch)
    chunk(root, 'ce_full', 'graphs-0000-0002.npz', [0, 1], 20)
    result = paper_status.suite_status(root)
    console = Console(width=120, record=True)
    console.print(dashboard(result, 'desktop', '12:34:56', 'SSH unavailable'))
    output = console.export_text()
    assert '50.0%' in output and '2/4' in output
    assert 'last successful update' in output
    assert 'subsequent models and reporting are excluded' in output
    assert 'Ctrl+C exits this monitor only' in output


def test_ordinary_baseline_uses_query_progress_and_requires_its_own_audit(tmp_path, monkeypatch):
    root=panel(tmp_path,monkeypatch);(root/'RUNNING').touch()
    directory=root/'ordinary_qwen';directory.mkdir()
    (directory/'progress.json').write_text(json.dumps(dict(status='extracting',queries=256,evaluation_seconds=10)))
    monkeypatch.setattr(paper_status,'_active_process',lambda root:dict(module='paper_baseline',label='Ordinary Qwen baseline',pid=321))
    result=paper_status.suite_status(root)
    assert result['ordinary_baseline']['queries']==256 and result['ordinary_baseline']['total']==1024
    assert result['ordinary_baseline']['remaining_inference_seconds']==30
    assert any(s['module']=='paper_baseline' and s['state']=='running' for s in result['stages'])
    (root/'independent_audit.json').write_text('{"passed": true}')
    assert not paper_status.suite_status(root)['independently_audited']
    (directory/'independent_audit.json').write_text('{"passed": true}')
    assert paper_status.suite_status(root)['independently_audited']
