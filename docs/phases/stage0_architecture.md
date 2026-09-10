# Stage 0: recurrent architecture validation

Status: implemented. The gate outcome and device-specific evidence are in the [validation report](../experiments/stage0_validation.md). Source: project plan sections 2–4 and 32.

## Purpose and scope

Prove that model surgery preserves one-pass behavior and repeated passes use the intended trainable parameters. No pointer data, research training, bridge, or asymmetric objectives are included.

## Implementation

- [Model mechanics](../../scripts/recurrent_qwen/model.py): original Qwen decoder objects grouped into P (0–5), shared R (6–17), and C (18–23), with configurable half-open split boundaries. Preserve embeddings, masks, positions, RoPE, final norm, and tied LM head; freeze original weights and disable caches.
- [LoRA](../../scripts/recurrent_qwen/lora_utils.py): PEFT injection into R only, q/v projections, rank 8, alpha 16, no dropout or bias training. A single shared set adds 270,336 trainable parameters to the frozen 494,032,768-parameter model.
- [Outputs](../../scripts/recurrent_qwen/outputs.py): per-loop logits, optional `h_0` and `h_1...h_T`, and allowed-token margins with explicit per-loop labels. [Architecture](../architecture.md#implemented-forward-interface) defines shapes and indexing.
- [Validation](../../scripts/recurrent_qwen/validation.py): ordinary references captured before mutation, equivalence before/after adapter attachment, module/parameter identity hooks, recurrent input identity, observability, and a two-loop backward without an optimizer step.
- [CLI](../../scripts/validate_stage0.py): explicit checkpoint loading, Rich results, and JSON evidence. Imports never load models. Default is local-only, fixed-revision CPU float32 with eager attention and strict deterministic algorithms.

The wrapper shares and freezes the supplied base in place. Compute reference outputs before LoRA attachment; a base sharing adapted layers is no longer an independent unadapted reference.

## Run validation

From the repository root:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
python -m scripts.validate_stage0 --device cpu --output artifacts/stage0/cpu.json
```

If the pinned checkpoint is not cached, explicitly add `--download`. `--revision` accepts another commit; the report records the resolved revision. The default is `7ae557604adf67be50417f59c2c2f167def9a775`. Run without Python's `-O` flag because gate checks use assertions.

Use `--recurrent-start`, `--recurrent-end`, `--rank`, `--alpha`, `--loops`, `--seed`, and `--threads` for explicit variations. Default loop counts are 1, 2, and 4. Positive loop counts have no architectural upper bound, but resource limits still apply. `--device mps` requests Apple Silicon execution and fails if unavailable, without falling back to CPU. Omitting `--output` creates timestamped JSON under ignored `artifacts/stage0/`.

## Validation and gate

| Check | Evidence |
| --- | --- |
| T=1 equivalence | Every token's full-vocabulary logits match ordinary Qwen with `atol=rtol=1e-5`, before and after initial LoRA; pretrained cases include no padding and both padding sides |
| Shared weights | Original layer identities retained; every loop invokes the same R object, decoder objects, and parameter IDs; P runs once and C once per loop |
| Recurrent flow | Next R input is the previous R output object; tiny-model tests also check causality |
| Gradient scope | Every original parameter stays frozen and gradient-free; only R's A/B tensors are trainable, all gradients are finite, and all initial B gradients are nonzero |
| Gradient connectivity | Final-loop CE reaches both recurrent hidden states through frozen C and earlier recurrence; a tiny-model diagnostic update additionally verifies nonzero A gradients |
| Observability/depth | Correct logits, hidden-state, and margin shapes at multiple depths; independent margin arithmetic and readout-selection tests |

Focused tests use tiny random Qwen models for split extremes, explicit/default positions, padding, eager/SDPA equivalence, causality, gradients, ties, and invalid boundaries. They are separate from validation of the actual pretrained checkpoint. Exact commands, measurements, and limitations are in the [report](../experiments/stage0_validation.md).

The hard gate is passing equivalence, sharing, and gradient scope. An unavailable or failed device is not a pass on that device. Stage 0 establishes implementation correctness on tested inputs, not useful recurrence, pointer execution, repair, or overscaling damage.

## Stop boundary

Stage 0 ends here. Next is the [symbol vocabulary and pointer-data generator](stage1_pointer.md), including tests, followed by another stop before training. Completion semantics, cycles, and task-specific readout remain decisions for that milestone.
