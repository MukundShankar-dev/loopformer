"""Fit shared token composition from existing count labels, never new depths."""
import torch
from torch import Tensor

from scripts.recurrent_qwen.number_reader import SharedNumberReader


def fit_number_reader(reader: SharedNumberReader, context: Tensor, counts: list[int]) -> dict:
    """Training-only two-stage least squares; no prescribed radix or digit values.

    Single-token examples give scalar labels for matching token embeddings.
    Two-token examples containing two such tokens identify the shared gain.
    Then all training count labels fit a *single* embedding projection, using
    that learned gain at every position. Held-out inputs/labels are not accepted.
    """
    if len(context) != len(counts) or len(set(counts)) != len(counts) or min(counts) < 1:
        raise ValueError('Need aligned unique positive training counts')
    embeddings, masks = reader.unpack(context.detach().cpu().double())
    valid = masks.squeeze(-1)
    if not ((valid == 0) | (valid == 1)).all() or (valid[:, 1:] > valid[:, :-1]).any():
        raise ValueError('Reader tokens must be left aligned with binary validity masks')
    lengths = valid.sum(1).long()
    labels = torch.tensor(counts, dtype=torch.float64)
    singles = {tuple(embeddings[i, 0].tolist()): labels[i] for i in range(len(counts)) if lengths[i] == 1}
    first_values, remainders, gain_counts = [], [], []
    for i in range(len(counts)):
        if lengths[i] != 2:
            continue
        first, second = (singles.get(tuple(v.tolist())) for v in embeddings[i, :2])
        if first is not None and second is not None:
            first_values.append(first); remainders.append(labels[i] - second); gain_counts.append(counts[i])
    if not first_values:
        raise ValueError('Training strings do not identify a shared gain from known single-token values')
    a, b = torch.stack(first_values), torch.stack(remainders)
    gain = a.dot(b) / a.dot(a)
    if not torch.isfinite(gain) or gain <= 0:
        raise ValueError('Training labels did not identify a positive finite reader gain')
    # Accumulate token features, including a per-token bias, exactly as inference.
    features = torch.cat((embeddings, torch.ones_like(masks)), -1)
    composed = features.new_zeros(len(context), reader.width + 1)
    for feature, mask in zip(features.unbind(1), valid.unbind(1), strict=True):
        composed = torch.where(mask[:, None].bool(), gain * composed + feature, composed)
    fit = torch.linalg.lstsq(composed, labels, rcond=1e-8, driver='gelsd')
    with torch.no_grad():
        reader.gain.copy_(gain.to(reader.gain))
        reader.token_value.weight.copy_(fit.solution[:-1][None].to(reader.token_value.weight))
        reader.token_value.bias.copy_(fit.solution[-1:].to(reader.token_value.bias))
    predictions = reader(context.to(reader.token_value.weight)).detach().cpu().flatten()
    error = (predictions.double() - labels).abs().max().item()
    if not torch.isfinite(predictions).all() or error > 1e-3:
        raise ValueError(f'Shared reader training fit failed: maximum error {error:g}')
    return {'method': 'training-label shared gain and projection least squares, rcond=1e-8',
            'training_counts': counts, 'gain_fit_counts': gain_counts, 'learned_gain': float(gain),
            'feature_rank': int(fit.rank), 'training_max_absolute_error': error,
            'digit_value_labels': False, 'reader_frozen_after_fit': True}


def refine_countdown(head, context: Tensor, counts: list[int], loops: int) -> dict:
    """Improve only two cell parameters using the original free-running labels.

    Initial memory comes from the frozen reader; N-t constructs loss targets
    only. Double-precision L-BFGS reduces optimization residuals which otherwise
    accumulate quadratically far beyond the supervised prefix. No decrement or
    identity gain is assigned. Deployment still uses the fitted float32 cell.
    """
    if head.kind != 'shared_number' or len(context)!=len(counts) or loops<1:
        raise ValueError('Need shared-number head and original aligned training prefix')
    head.requires_grad_(False)
    with torch.no_grad():
        initial = head.initialize(context).double()
    head.cell.double().requires_grad_(True)
    labels = torch.tensor(counts,dtype=torch.float64)
    times = torch.arange(1,loops+1,dtype=torch.float64)
    mask = times[None]<=labels[:,None]
    targets = labels[:,None]-times[None]  # Supervision, never a recurrence input.
    def objective() -> Tensor:
        memory, predicted = initial, []
        for _ in range(loops):
            memory = head.cell(memory); predicted.append(memory.squeeze(-1))
        error = (torch.stack(predicted,1)-targets).square()*mask
        # Same per-example weighting as prefix_losses, omitting constant loop-zero error.
        return (error.sum(1)/(mask.sum(1)+1)).mean()
    before = float(objective().detach())
    optimizer = torch.optim.LBFGS(head.cell.parameters(),lr=1,max_iter=100,
        tolerance_grad=1e-12,tolerance_change=1e-25,line_search_fn='strong_wolfe')
    evaluations = 0
    def closure() -> Tensor:
        nonlocal evaluations
        optimizer.zero_grad(); loss=objective(); loss.backward(); evaluations+=1
        return loss
    optimizer.step(closure)
    fitted = float(objective().detach())
    head.cell.float().requires_grad_(False)
    with torch.no_grad():
        memory, predicted = head.initialize(context), []
        for _ in range(loops):
            memory=head.cell(memory);predicted.append(memory.squeeze(-1))
        deployed=(torch.stack(predicted,1).double()-targets).square()*mask
        after=float((deployed.sum(1)/(mask.sum(1)+1)).mean())
    if not torch.isfinite(head.cell.weight).all() or not torch.isfinite(head.cell.bias).all() or after>1e-9:
        raise ValueError('Countdown training-prefix fit failed after float32 export')
    return {'method':'float64 L-BFGS on original free-running numerical prefix labels; float32 deployment',
        'training_counts':counts,'training_loops':loops,'objective_evaluations':evaluations,
        'before_mse':before,'fitted_mse':fitted,'deployed_mse':after,
        'learned_gain':float(head.cell.weight.item()),'learned_offset':float(head.cell.bias.item()),
        'numeric_targets_are_inputs':False}
