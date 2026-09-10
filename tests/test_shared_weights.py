import torch

from scripts.recurrent_qwen import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora


def test_shared_objects_and_recurrent_state_flow(base, inputs):
    original_layers = tuple(base.model.layers)
    model = RecurrentQwen(base, 1, 3)
    assert tuple(model.prelude.layers) == original_layers[:1]
    assert tuple(model.recurrent.layers) == original_layers[1:3]
    assert tuple(model.coda.layers) == original_layers[3:]
    attach_recurrent_lora(model, rank=2, alpha=4)
    parameter_ids = tuple(id(p) for p in model.parameters())
    recurrent_calls, prelude_calls, layer_calls = [], [], []
    handles = [
        model.recurrent.register_forward_hook(
            lambda module, args, output: recurrent_calls.append((id(module), args[0], output))
        ),
        model.prelude.register_forward_hook(lambda *args: prelude_calls.append(1)),
    ]
    for layer in model.recurrent.layers:
        handles.append(layer.register_forward_hook(lambda module, args, output: layer_calls.append(id(module))))
    try:
        with torch.no_grad():
            for depth in (1, 2, 5):
                recurrent_calls.clear()
                prelude_calls.clear()
                layer_calls.clear()
                result = model(inputs, num_loops=depth, return_hidden_states=True)
                assert len(prelude_calls) == 1
                assert len(recurrent_calls) == depth
                assert layer_calls == [id(layer) for layer in model.recurrent.layers] * depth
                assert all(call[0] == id(model.recurrent) for call in recurrent_calls)
                assert recurrent_calls[0][1] is result.initial_hidden_state
                for index in range(1, depth):
                    assert recurrent_calls[index][1] is recurrent_calls[index - 1][2]
                assert tuple(id(p) for p in model.parameters()) == parameter_ids
    finally:
        for handle in handles:
            handle.remove()
