# Architecture and implementation boundaries

Status: Stage 0 and Stage 1 data implemented; training remains planned. See the [validation report](experiments/stage0_validation.md) for device-specific evidence and limits, and [data report](experiments/stage1_data_validation.md) for pointer validation. The research specification is [the project plan](project_plan.md).

The [inference smoke test](../scripts/smoke_test_qwen.py) exercises ordinary Qwen. The [Stage 0 entry point](../scripts/validate_stage0.py) loads the checkpoint, constructs the recurrent model, attaches LoRA, and validates the architecture. Stage 0 defaults to cached files at an immutable revision. Usage is in [setup](setup.md).

## Recurrent computation

Inspection of Transformers 5.17.0 confirmed `Qwen2ForCausalLM`, `Qwen2Model`, and tensor-returning `Qwen2DecoderLayer.forward`. The Qwen2.5-0.5B-Instruct checkpoint has 24 layers, hidden size 896, 14 attention heads, two KV heads, full attention, default RoPE, and tied embedding/head weights.

The configurable default split is layers 0–5 for P, 6–17 for shared R, and 18–23 for C:

```text
tokens -> embeddings -> P -> h_0
h_0 -> R -> h_1 -> R -> h_2 -> ... -> h_T
             each h_t -> C -> final norm -> LM head -> logits_t
```

`RecurrentQwen` groups the original decoder objects into `prelude`, `recurrent`, and `coda` (`DecoderBlock` modules), without copying weights or retaining duplicate registered paths. Embeddings, RoPE, final RMSNorm, and the LM head are reused; tied weights remain tied. Construction freezes and clears gradients on the supplied base **in place**. Adapter injection subsequently mutates its shared middle layers, so compute unadapted reference logits first or load a separate reference.

The next iteration consumes the recurrent state, never the coda output or a decoded token. P runs once; intermediate C reads expose predictions without modifying the recurrent trajectory. There is no bridge: direct middle-to-middle recurrence passes Stage 0. This does not establish useful recurrent task execution or imply that `h_0` decodes to a start symbol.

Mask construction uses the installed `create_causal_mask`. Every decoder receives `attention_mask`, `position_ids`, `position_embeddings`, `past_key_values=None`, and `use_cache=False`. The mask and RoPE are prepared once and reused across loops. Position IDs default to `arange(sequence_length)` as in ordinary Qwen forward, including padding; callers can supply explicit positions. Only full attention and default RoPE are accepted in Stage 0. The validation CLI selects eager attention and float32 explicitly; tiny tests also cover SDPA. Compatibility with other Transformers versions is not promised.

## Trainability and autograd

All 494,032,768 original parameters are frozen: embeddings, P, original R weights, C, final norm, and LM head. `attach_recurrent_lora(model)` uses PEFT in-place injection on `model.recurrent` only. Defaults are q/v projections, rank 8, alpha 16, dropout 0, bias `none`, and standard random-A/zero-B initialization. This adds 270,336 trainable parameters in 48 tensors. Rank, alpha, and projection targets are configurable in the library; the CLI fixes q/v targets.

Trainable names have the form `recurrent.layers.<relative-index>.self_attn.<projection>.lora_[A|B].default.weight`. Add `recurrent_start` to that relative layer index to recover the original Qwen layer number.

Frozen operations remain differentiable. Gradients pass through C to the adapters and through earlier R iterations. There is no internal `no_grad` around those paths. Zero A gradients on the first backward are expected when B is zero; focused tests verify both factors after a diagnostic tiny-model update. No optimizer step is taken on the pretrained model during validation.

Future Stage 1 uses differentiable unrolls; future Stage 3 needs explicitly detached rollouts. Neither training objective is implemented here.

## Implemented forward interface

`model(input_ids, attention_mask=None, *, num_loops=1, position_ids=None, answer_positions=None, labels=None, allowed_token_ids=None, return_hidden_states=False, logits_mode="answer")` returns `RecurrentOutput`.

- Inputs are long `[B,S]` token IDs and a binary `[B,S]` mask. Each row needs an unmasked token. Positions are long `[1,S]` or `[B,S]`; loops do not append token positions.
- Answer positions are long `[B]`, defaulting to the last unmasked token for either padding side. They select a next-token prediction location, not a token to insert into the prompt. Padding positions are rejected. Integer slices implement this readout because advanced indexing and gather backward fail under strict MPS determinism; indices synchronize to a Python list once per forward. This is appropriate for the intended tiny batches.
- `loop_logits` is a tuple of length T, with tensors `[B,V]` by default or `[B,S,V]` in `"all"` mode. `.logits` returns the final item. Every loop reads C; no final-only supervision is imposed.
- With hidden capture enabled, `initial_hidden_state` is `h_0`, and `hidden_states[t-1]` is `h_t`, each `[B,S,H]`. Otherwise both fields are `None`; autograd can still retain the activations required for backward.
- Optional labels are long `[B,T]` **token IDs**, with an explicit unique allowed set `[A]` of at least two tokens. `margins[B,T]` uses the target raw logit minus the largest other allowed logit. Labels can differ by loop. Repeating a final label is an explicit caller choice. This interface neither computes CE nor shifts labels.
- Outputs retain gradients when autograd is enabled. Use caller-side `torch.no_grad()` for evaluation. No bridge, KV cache, generation API, detached-rollout training, or checkpoint save/load interface is implemented.

The wrapper inherits model device/dtype without conversion or downloads. Stage 0 measures short inputs and small loop counts; its resource results do not establish a training budget for longer sequences or unrolls.

## Module ownership

Only current-stage modules exist; future locations below are not scaffolded requirements.

| Location | Responsibility |
| --- | --- |
| `scripts/recurrent_qwen/model.py` | Model partition, shared recurrence, per-loop readout |
| `scripts/recurrent_qwen/lora_utils.py` | Adapter placement and freezing configuration |
| `scripts/recurrent_qwen/outputs.py` | Output shapes, loop conventions, allowed-answer margins |
| `scripts/recurrent_qwen/validation.py` | Stage 0 architecture checks and measurements |
| `scripts/validate_stage0.py` | Loading, CLI configuration, Rich presentation, JSON evidence |
| `tests/` | Focused scientific and implementation contracts |
| `scripts/dataset/pointer.py` | Pointer records, random tables, exact reference execution, decoded-symbol checking |
| `scripts/dataset/symbols.py` | Pinned tokenizer identity and context-validated single-token vocabulary |
| `scripts/dataset/dataset.py` | Seeded splits, JSONL/manifest persistence, independent verification |
| `scripts/dataset/cli.py` | Cached tokenizer loading, dataset CLI, Rich dry run and summaries |
| Future `training/` | Stage-specific objectives and training |
| Future `eval/` | Metrics, sweeps, survival, transfer, plots |

Scripts compose library functions. Task generation must remain separate from training, and loss functions must not own loading or artifact writing. See [Stage 0](phases/stage0_architecture.md) for commands and [decisions](decisions.md) for remaining research semantics.

## Stage 1 data boundary

Pointer generation has no dependency on the recurrent wrapper and loads no model. Logical state targets and actual answer-token IDs are saved together. A future trainer can construct labels `[B,T]` from `intermediate_token_ids` and the allowed answer set from the manifest; depth masking/reduction is still a training-stage decision. Saved answer positions apply to unpadded raw prompts; batching must account for padding. Only nominal steps 1..d are labeled, with no h_0 or post-completion supervision. The [Stage 1 guide](phases/stage1_pointer.md) owns the data format and reproduction commands.
