import numpy as np
import pytest

from scripts.eval.paper_pareto import pareto_mask, tradeoff_points


def test_pareto_dominance_is_strict_and_retains_exact_ties():
    mask = pareto_mask(np.array([1, 2, 2, 3, 1]), np.array([0, .5, .5, .4, -.1]))
    assert mask.tolist() == [True, True, True, False, False]


def test_missing_stop_costs_cap_and_headless_stopping_is_excluded():
    counts = np.array([1, 2, 3])
    arrays = {
        'ce_full__stops': np.full((2, 3), -2),
        'final__stops': np.array([[1, -1, 3], [1, 2, 3]]),
        'final__joint_success': np.array([[True, False, True], [True, True, True]]),
        'final__strict_success': np.array([[True, False, False], [True, True, True]]),
    }
    rows = tradeoff_points(arrays, ['ce_full', 'final'], counts, 4, ranges=((1, 3),))
    assert len(rows) == 2 and all(r['model'] == 'final' for r in rows)
    assert rows[0]['mean_loops'] == pytest.approx(14/6)
    assert rows[0]['success_rate'] == pytest.approx(5/6)
    assert rows[1]['success_rate'] == pytest.approx(4/6)
    assert rows[0]['missing_stop_rate'] == pytest.approx(1/6)
    assert all(r['pareto'] for r in rows)
    # Matching a cyclic final letter at loop 3 for requested N=1 is not success.
    arrays['final__stops'][0, 0] = 3
    with pytest.raises(ValueError, match='Wrong-loop'):
        tradeoff_points(arrays, ['final'], counts, 4, ranges=((1, 3),))


def test_undefined_and_observed_stops_cannot_mix():
    with pytest.raises(ValueError, match='mixed undefined'):
        tradeoff_points({'final__stops': np.array([[-2, 2]])}, ['final'], np.array([1, 2]), 4, ranges=((1, 2),))
