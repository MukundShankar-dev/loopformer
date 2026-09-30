from pathlib import Path

import pytest

from scripts.training.cleanup_pointer_runs import BASELINE, KEEP_STEPS, cleanup_candidates
from scripts.training.cleanup_pointer_runs import LOOPBALANCED, LOOPBALANCED_EVALS


def test_loopbalanced_cleanup_is_exact_and_includes_checkpoint_payloads(tmp_path):
    run = tmp_path / 'models/stage1_pointer' / LOOPBALANCED
    (run / 'step-003250').mkdir(parents=True)
    (run / 'step-003250/adapter_model.pt').write_bytes(b'weights')
    evaluation = tmp_path / 'eval/pointer_loops' / LOOPBALANCED_EVALS[0]
    evaluation.mkdir(parents=True)
    unknown = evaluation.parent / ('future-' + LOOPBALANCED + '-step-003250')
    unknown.mkdir()
    baseline = run.parent / BASELINE
    baseline.mkdir()
    assert set(cleanup_candidates(tmp_path, loopbalanced_only=True)) == {run, evaluation}
    assert (run / 'step-003250/adapter_model.pt').exists()  # Preview never deletes.
    run.rename(run.with_name('preserve'))
    run.symlink_to(run.with_name('preserve'), target_is_directory=True)
    with pytest.raises(ValueError, match='symlinked'):
        cleanup_candidates(tmp_path, loopbalanced_only=True)


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
