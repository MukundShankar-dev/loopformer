"""Dataset loop means and gradients must be independent of microbatch partition."""

from dataclasses import replace

import pytest
import torch

from scripts.training.config import TrainingConfig
from scripts.training.objective import loop_loss_weights, step_loss, training_selection_loss
from scripts.training.runner import resume_identity, validate_resume_identity


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5])
def test_loop_means_and_gradients_across_unequal_microbatches(batch_size):
    depths = [1, 1, 2, 3, 3]
    torch.manual_seed(8)
    scores = torch.randn(5, 3, 4, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(4, (5, 3))
    mask = torch.arange(3)[None, :] < torch.tensor(depths)[:, None]
    ce = torch.nn.functional.cross_entropy(scores.flatten(0, 1), targets.flatten(), reduction='none').reshape(5, 3)
    expected = torch.stack([ce[mask[:, t], t].mean() for t in range(3)]).mean()
    expected.backward()
    gradient = scores.grad.clone()
    scores.grad = None
    weights = torch.tensor(loop_loss_weights(depths), dtype=scores.dtype)
    total = 0.0
    for start in range(0, 5, batch_size):
        end = min(start + batch_size, 5)
        # Actual unrolls may be shorter than the full training loop range.
        loops = max(depths[start:end])
        loss, _ = step_loss(scores[start:end, :loops], targets[start:end, :loops], mask[start:end, :loops], loop_weights=weights)
        scaled = loss * (end - start) / 5
        scaled.backward()
        total += scaled.item()
    assert total == pytest.approx(expected.item())
    torch.testing.assert_close(scores.grad, gradient)
    assert torch.count_nonzero(scores.grad[~mask]) == 0


def test_loop_weight_coefficients_are_equal_over_the_dataset():
    depths = list(range(1, 7)) * 5
    weights = loop_loss_weights(depths)
    for t, weight in enumerate(weights, 1):
        assert weight * sum(d >= t for d in depths) / len(depths) == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        loop_loss_weights([])


def test_selection_excludes_ood_examples_even_at_early_loops():
    metrics = {'by_depth': {
        '1': {'examples': 2, 'loss': 2., 'per_loop': {'1': {'count': 2, 'loss': 2.}}},
        '2': {'examples': 1, 'loss': 5., 'per_loop': {'1': {'count': 1, 'loss': 4.}, '2': {'count': 1, 'loss': 6.}}},
        '3': {'examples': 999, 'loss': 1000., 'per_loop': {'1': {'count': 999, 'loss': 1000.}}},
    }}
    assert training_selection_loss(metrics, 2, 'loop_mean') == pytest.approx(((2 * 2 + 4) / 3 + 6) / 2)
    assert training_selection_loss(metrics, 2, 'example_mean') == pytest.approx(3.)


def test_loss_change_cannot_be_disguised_as_batch_change_on_resume():
    baseline = TrainingConfig()
    balanced = replace(baseline, loss_reduction='loop_mean')
    assert 'loss_reduction' not in resume_identity(baseline, {})['config']
    with pytest.raises(ValueError, match='Resume identity differs'):
        validate_resume_identity(resume_identity(baseline, {}), resume_identity(balanced, {}), allow_batch_change=True)
    with pytest.raises(ValueError, match='loss_reduction'):
        replace(baseline, loss_reduction='unknown').validate()
