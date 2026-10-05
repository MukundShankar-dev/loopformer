"""Training-only numerical readout; neither its targets nor predictions enter recurrence."""
import math

import torch
from torch import Tensor, nn


def remaining_predictions(readout: nn.Linear, initial: Tensor, memories: Tensor,
                          scale: float, detach: bool = False) -> Tensor:
    """Read [B,T+1] remaining work in steps from initial and recurrent [B,T,M] memory.

    A fixed global scale conditions regression; it never depends on an example's
    requested count. The stop-only control detaches memory before this readout.
    """
    states = torch.cat((initial[:, None], memories), dim=1)
    return readout(states.detach() if detach else states).squeeze(-1) * scale


def remaining_loss(predicted: Tensor, depths: Tensor, scale: float) -> Tensor:
    """Mean per-example squared error on (N-t)/scale for t=0..N, inclusive.

    Later unsupervised passes have zero gradient. Targets are built here only;
    no gold value is fed back, and no decrement is implemented in the forward.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Remaining-work scale must be finite and positive')
    if (predicted.ndim != 2 or depths.shape != predicted.shape[:1] or
            depths.dtype not in (torch.int32, torch.int64) or
            (depths < 1).any() or (depths >= predicted.shape[1]).any()):
        raise ValueError('Need [B,T+1] predictions and integer depths in 1..T')
    if not torch.isfinite(predicted).all():
        raise FloatingPointError('Nonfinite remaining-work predictions')
    t = torch.arange(predicted.shape[1], device=predicted.device)[None, :]
    mask = t <= depths[:, None]
    target = depths[:, None] - t
    error = ((predicted - target) / scale).square() * mask
    return (error.sum(-1) / (depths + 1)).mean()


def remaining_metrics(predicted: Tensor, logits: Tensor, depths: Tensor) -> dict[str, float | int | None]:
    """Metrics in actual steps, per-example averaged; only t<=N has gold targets.

    Readout zero means predicted <=0.5 (nearest integer zero or lower), evaluated
    from loop one. This is diagnostic only, never the deployed stop policy.
    """
    t = torch.arange(predicted.shape[1], device=predicted.device)[None, :]
    mask = t <= depths[:, None]
    error = (predicted - (depths[:, None] - t)).abs()
    terminal = predicted.gather(1, depths[:, None]).squeeze(1)
    delta_mask = t[:, 1:] <= depths[:, None]
    delta_error = (predicted[:, 1:] - predicted[:, :-1] + 1).abs()
    zero = predicted[:, 1:] <= .5
    zero_exists = zero.any(-1)
    first_zero = zero.int().argmax(-1) + 1
    stop = logits >= 0
    stop_exists = stop.any(-1)
    first_stop = stop.int().argmax(-1) + 1
    near_zero = predicted[:, 1:].abs() <= .5
    near_zero &= delta_mask
    numerator = (stop & near_zero).sum().item()
    denominator = near_zero.sum().item()
    return {
        'remaining_initial_mae': error[:, 0].mean().item(),
        'remaining_initial_within_half': (error[:, 0] <= .5).float().mean().item(),
        'remaining_mae': ((error * mask).sum(-1) / (depths + 1)).mean().item(),
        'remaining_terminal_mae': terminal.abs().mean().item(),
        'remaining_decrement_mae': ((delta_error * delta_mask).sum(-1) / depths).mean().item(),
        'remaining_zero_exact': (zero_exists & (first_zero == depths)).float().mean().item(),
        'remaining_zero_missing': (~zero_exists).float().mean().item(),
        'remaining_stop_zero_agreement': (zero_exists & stop_exists & (first_zero == first_stop)).float().mean().item(),
        'remaining_near_zero_steps': denominator,
        'remaining_stop_near_zero_rate': numerator / denominator if denominator else None,
    }


def remaining_trace_rows(predicted: Tensor, logits: Tensor, tasks: list) -> list[dict]:
    """Observational forced replay, including explicitly unlabeled post-request loops."""
    values, probs = predicted.tolist(), logits.sigmoid().tolist()
    return [{'example_id': task.example_id, 'depth': task.task_depth, 'loop': t,
             'predicted_remaining': value,
             'target_remaining': task.task_depth - t if t <= task.task_depth else None,
             'nominal': t <= task.task_depth,
             'stop_probability': probs[i][t - 1] if t else None}
            for i, task in enumerate(tasks) for t, value in enumerate(values[i])]
