import json
import csv

import numpy as np
import pytest

from scripts.eval.paper_writeup import baseline_counts, binned_training, weighted_depth_metric


def test_requested_depth_cohorts_use_question_denominators():
    validation = {'by_depth': {'1': {'examples': 10, 'trajectory_accuracy': 1.},
                               '2': {'examples': 30, 'trajectory_accuracy': .5},
                               '3': {'examples': 80, 'trajectory_accuracy': 0.}}}
    assert weighted_depth_metric(validation, [1, 2], 'trajectory_accuracy') == .625


def test_training_windows_weight_examples_and_preserve_update_axis():
    events = [{'event': 'train', 'step': 1, 'train': {'loss': 4., 'examples': 2}},
              {'event': 'validation', 'step': 1},
              {'event': 'train', 'step': 50, 'train': {'loss': 1., 'examples': 6}},
              {'event': 'train', 'step': 51, 'train': {'loss': 3., 'examples': 2}}]
    np.testing.assert_equal(binned_training(events), [[50, 1.75], [51, 3.]])


def test_baseline_recounts_predictions_and_rejects_wrong_summary(tmp_path):
    fields = ['example_id', 'task_depth', 'valid_answer', 'prediction', 'target', 'correct']
    with (tmp_path/'predictions.csv').open('w') as handle:
        writer = csv.writer(handle); writer.writerow(fields)
        writer.writerow(['a', 1, 'True', 'A', 'A', 'True'])
        writer.writerow(['b', 1, 'True', 'B', 'A', 'False'])
    summary = {'total': 2, 'correct': 1, 'by_depth': {'1': {'total': 2, 'correct': 1}}}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    depths, counts, _ = baseline_counts(tmp_path)
    np.testing.assert_equal(depths, [1]); np.testing.assert_equal(counts, [[1], [2]])
    summary['by_depth']['1']['correct'] = 2
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='depth summary'):
        baseline_counts(tmp_path)


def test_writeup_evaluation_draws_identical_metrics_once(monkeypatch, tmp_path):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.eval.paper_writeup import population_panel
    arrays = {'final__nominal_final_correct': np.ones((2, 256)),
              'final__joint_success': np.ones((2, 256)),
              'final__complete_trajectory': np.zeros((2, 256))}
    fig, ax = plt.subplots(); population_panel(ax, arrays, ['final'])
    assert len(ax.lines) == 2
    assert 'identical' in ax.lines[0].get_label()
    assert 'exact first stop' in ' '.join(ax.lines[0].get_label().split())
    np.testing.assert_equal(ax.lines[0].get_xdata(), np.arange(1, 257))
    plt.close(fig)
