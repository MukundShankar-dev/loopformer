# Architecture and implementation boundaries

Status: planned; model interfaces and environment have not been inspected. The research specification is [the project plan](project_plan.md).

The implemented [inference smoke test](../scripts/smoke_test_qwen.py) composes Transformers loading, chat templating, and generation directly, with explicit CPU/MPS selection, float32 weights, disabled gradients, and no KV cache. It exercises ordinary Qwen only; the recurrent modules below remain planned. Usage and validation limits are in [setup](setup.md).

## Recurrent computation

Start with `Qwen/Qwen2.5-0.5B-Instruct`, PyTorch, Hugging Face Transformers, and PEFT/LoRA. Confirm the downloaded model configuration and installed decoder implementation before writing the wrapper.

The initial configurable split is layers 0–5 for the prelude, 6–17 for recurrence, and 18–23 for the coda:

```text
tokens -> embeddings -> prelude -> h_0
h_0 -> shared R -> h_1 -> shared R -> h_2 -> ... -> h_T
                       each h_t -> coda -> final norm -> LM head -> logits_t
```

The next recurrent iteration consumes the recurrent state, not the coda output or a decoded token. The prompt is processed once by the prelude. There is one recurrent parameter set, reused for every iteration. Intermediate coda reads expose predictions without changing the recurrent trajectory.

Preserve the base model's causal masks, positions, rotary embeddings, layer ordering, final normalization, and LM head behavior. Inspect the actual decoder forward arguments rather than assuming compatibility across Transformers versions. Initially set `use_cache=False`.

Before training, `T=1` must reproduce the ordinary Qwen forward pass within documented numerical tolerance. Start without a bridge where possible. If a bridge is necessary, document its placement, initialization, and effect on equivalence and the shared-transition interpretation. Do not assume `h_0` already decodes to the start symbol.

## Trainability and autograd

Freeze token embeddings, prelude, original middle weights, coda, final norm, and LM head. Train only LoRA adapters within the recurrent block and an optional bridge. Record the adapter targets and trainable parameter names/counts when implemented.

Frozen parameters still participate in differentiable operations. During training, gradients must pass through the frozen coda to the recurrent adapters and through frozen operations inside the middle block. Do not disable autograd around these paths merely because their base parameters are frozen.

Stage 1 uses differentiable recurrent unrolls. Stage 3 first generates a detached state without gradients, then differentiates one additional recurrent transition and its coda readout. No graph should connect that update to the detached prefix.

## Planned module ownership

Create modules as needed by the current stage; this table is not a request to scaffold later-stage code.

| Location | Responsibility |
| --- | --- |
| `recurrent_qwen/model.py` | Model partition, shared recurrence, and per-loop readout |
| `recurrent_qwen/bridge.py` | Optional re-entry transformation, only if justified |
| `recurrent_qwen/lora_utils.py` | Adapter placement, freezing, and trainability inspection |
| `recurrent_qwen/outputs.py` | Explicit output structures and tensor/index conventions |
| `tasks/` | Symbol validation, task records, prompt rendering, exact state trajectories, and generation |
| `training/` | Stage-specific training, detached rollout, and objective computation |
| `eval/` | Transition classification, aggregate metrics, depth sweeps, survival, transfer, and plots |
| `configs/` | Reproducible stage configurations |
| `scripts/` | Thin command-line entry points that compose library code |
| `tests/` | Focused contract tests and necessary model integration checks |

Task generators should not depend on training loops. Loss functions should not own model loading or artifact writing. Evaluation should operate on explicit predictions/targets or trajectory records wherever practical.

## Interface details to document during implementation

- Input tensor shapes, attention masks, position handling, padding, and answer readout position.
- Loop indexing (`h_0` follows the prelude; `h_t` follows `t` middle passes).
- Which outputs are optional, their shapes, and whether they retain gradients.
- Allowed answer token IDs and the distinction between intermediate and final targets.
- Layer split, LoRA settings, optional bridge, device, dtype, and model revision.
- Checkpoint contents and how they reconstruct the same frozen base plus adapters.

Expose intermediate observability without requiring every evaluation run to retain all full-sequence hidden states. Start with short prompts and tiny batches on the target 32 GB Apple Silicon machine. Benchmark memory and runtime before extending training unrolls; use long trajectories mostly for evaluation without gradients.

See [Stage 0](phases/stage0_architecture.md) for validation and [decisions](decisions.md) for unresolved semantics.
