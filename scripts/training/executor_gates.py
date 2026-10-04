"""Startup contracts for isolated control and richer executor supervision."""
import torch
from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.recurrent_qwen.interfaces import RecurrentController
from .objective import completion_loss


def validate_executor(model: RecurrentQwen, inputs: dict, token_ids: list[int]) -> dict:
    model.eval()
    model.zero_grad(set_to_none=True)
    prefix_error = None
    if model.prefix_reuse:
        with torch.no_grad():
            optimized = model(inputs['input_ids'][:1], inputs['attention_mask'][:1],
                              num_loops=2, readout_token_ids=token_ids)
            for block in (model.prelude, model.recurrent, model.coda):
                block.prefix_reuse = False
            try:
                dense = model(inputs['input_ids'][:1], inputs['attention_mask'][:1],
                              num_loops=2, readout_token_ids=token_ids)
            finally:
                for block in (model.prelude, model.recurrent, model.coda):
                    block.prefix_reuse = True
            for a, b in zip(optimized.loop_logits, dense.loop_logits):
                torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
            prefix_error = max((a - b).abs().max().item() for a, b in zip(optimized.loop_logits, dense.loop_logits))
    result = model(inputs['input_ids'][:1], inputs['attention_mask'][:1], num_loops=2,
                   readout_token_ids=token_ids, return_hidden_states=True)
    for hidden in result.hidden_states:
        hidden.retain_grad()
    torch.nn.functional.cross_entropy(result.loop_logits[-1].float(), inputs['targets'][:1, 0]).backward(retain_graph=True)
    position = int(inputs['attention_mask'][0].nonzero()[-1])
    norms = [h.grad[0, position].float().norm().item() for h in result.hidden_states]
    if not all(n > 0 for n in norms):
        raise RuntimeError('Executor loss must reach all recurrent working states')
    if model.completion_head is not None and any(p.grad is not None for p in model.completion_head.parameters()):
        raise RuntimeError('Pointer loss reached the controller')
    if model.bridge is not None and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.bridge.parameters()):
        raise RuntimeError('Re-entry bridge receives no executor gradient')
    model.zero_grad(set_to_none=True)
    if isinstance(model.completion_head, RecurrentController):
        loss, _ = completion_loss(result.stop_logits.float(), torch.ones_like(result.stop_logits, dtype=torch.bool))
        loss.backward()
        for name, p in model.named_parameters():
            if not name.startswith('completion_head.') and p.grad is not None:
                raise RuntimeError(f'Controller gradients escaped isolation: {name}')
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.completion_head.parameters()):
            raise RuntimeError('Controller receives no gradient')
    model.zero_grad(set_to_none=True)
    return {'passed': True, 'prefix_dense_max_logit_error': prefix_error, 'pointer_to_controller_gradient': False,
            'controller_to_executor_gradient': False, 'working_state_gradient_norms': norms,
            'input_routing': 'steps_removed' if model.router else 'legacy',
            'train_scope': model.train_scope}
