from pathlib import Path

import pytest

from scripts.training.cleanup_pointer_runs import BASELINE, KEEP_STEPS, cleanup_candidates


def test_cleanup_preserves_baseline_evidence_and_unknown_runs(tmp_path):
    baseline = tmp_path / 'models/stage1_pointer' / BASELINE
    for name in [*KEEP_STEPS, 'step-000250']:
        (baseline / name).mkdir(parents=True)
    (baseline / 'metrics.jsonl').write_text('evidence')
    old = tmp_path / 'models/stage1_pointer/depth6-seed17'
    old.mkdir()
    new = tmp_path / 'models/stage1_pointer/loopbalanced'
    new.mkdir()
    assert set(cleanup_candidates(tmp_path)) == {old, baseline / 'step-000250'}
    assert (baseline / 'metrics.jsonl').read_text() == 'evidence'


def test_cleanup_rejects_symlinked_run(tmp_path):
    outside = tmp_path / 'preserve'
    outside.mkdir()
    parent = tmp_path / 'models/stage1_pointer'
    parent.mkdir(parents=True)
    (parent / 'depth6-seed17').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinked'):
        cleanup_candidates(tmp_path)
