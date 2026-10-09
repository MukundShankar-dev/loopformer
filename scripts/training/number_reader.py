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
