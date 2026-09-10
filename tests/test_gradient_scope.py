import torch
from torch.nn import functional as F

from scripts.recurrent_qwen import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora


def test_only_recurrent_lora_receives_gradients_through_coda_and_unroll(base, inputs):
    originals = tuple(base.parameters())
    snapshots = [p.detach().clone() for p in originals]
    model = RecurrentQwen(base, 1, 3)
    attach_recurrent_lora(model, rank=2, alpha=4)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 8  # Two layers, q/v, A/B.
    assert all(name.startswith("recurrent.") and "lora_" in name for name, _ in trainable)
    optimizer = torch.optim.SGD([p for _, p in trainable], lr=0.1)
    # First backward: zero-initialized B can make A's gradient zero.
    # After one diagnostic update, both factors must receive nonzero gradients.
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(inputs, num_loops=3, return_hidden_states=True)
        for hidden in out.hidden_states:
            hidden.retain_grad()
        F.cross_entropy(out.logits, torch.tensor([2, 3])).backward()
        assert all(not p.requires_grad and p.grad is None for p in originals)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
        assert all(h.grad is not None and h.grad.abs().sum() > 0 for h in out.hidden_states)
        for name, parameter in trainable:
            if step == 1 or "lora_B" in name:
                assert parameter.grad.abs().sum() > 0, name
        if step == 0:
            optimizer.step()
    for original, snapshot in zip(originals, snapshots):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)
