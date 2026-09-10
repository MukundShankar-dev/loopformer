# Stage 0: recurrent architecture validation

Status: planned. This is the first implementation milestone. Source: project plan sections 2–4 and 32.

## Purpose and scope

Prove that the model surgery preserves one-pass behavior and that repeated passes use the intended trainable parameters. This stage contains no pointer training or asymmetric objectives.

## Execution

1. Inspect repository state and local Python, PyTorch, Transformers, PEFT, device support, and available model files. Record actual versions and constraints.
2. Inspect the model configuration, decoder classes, and exact forward arguments. Identify attention/position handling and output conventions.
3. Update [architecture](../architecture.md) with the concrete wrapper design and any necessary bridge decision.
4. Implement the configurable prelude/recurrent/coda split with shared middle weights and caching disabled.
5. Add recurrent LoRA placement and freezing. Preserve initial one-pass behavior after adapter attachment and any bridge initialization.
6. Expose per-loop hidden states, logits, and label-dependent answer margins through explicit output structures.
7. Implement and run the validation below. Record exact commands, model revision, device, dtype, tolerances, and outcomes.
8. Stop and report the Stage 0 result before starting the data milestone.

## Validation and gate

| Check | Required evidence |
| --- | --- |
| One-loop equivalence | Base and recurrent logits agree on identical inputs in evaluation mode, with stated tolerances and observed error |
| Shared weights | Multiple iterations invoke the same module objects and parameter set; parameter count does not grow with loop count |
| Gradient scope | Backpropagation reaches intended adapter/bridge paths while all original parameters remain frozen and gradient-free |
| Observability | Requested hidden states, logits, and margins align with documented loop indices and shapes |
| Configurable depth | More than one requested loop count works without rebuilding or duplicating recurrent weights |

Account for adapter initialization when interpreting gradients: an individual factor can legitimately have zero gradient initially. Verify the intended trainable paths rather than requiring every trainable element to be nonzero on one backward pass.

Small local model configurations can support focused tests, but distinguish them from integration validation on the specified pretrained Qwen checkpoint. Do not report pretrained equivalence from a substitute model alone.

The hard gate is passing one-loop equivalence, shared weights, and gradient scope. An unavailable dependency or checkpoint is an unverified gate, not a pass.

## Implementation record

A preliminary [inference smoke test](../../scripts/smoke_test_qwen.py) now loads `Qwen/Qwen2.5-0.5B-Instruct` and generates a short chat response. See [setup](../setup.md) for commands and validation limits. Syntax and CLI checks do not establish pretrained inference or pass any Stage 0 gate.

No recurrent wrapper or architecture tests exist yet. Populate this section with actual interfaces, commands, results, and limitations during implementation; update [status](../status.md) with the gate evidence.
