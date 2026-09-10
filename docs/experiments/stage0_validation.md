# Stage 0 validation — 2026-09-10

## Outcome and scope

**Gate 0 passes on the specified pretrained Qwen checkpoint on CPU and Apple Silicon MPS.** Full logits match exactly at T=1 before and after LoRA attachment. Shared recurrence, adapter-only gradient scope, per-loop observability, and configurable depth pass. The focused suite has **36 passing tests**.

This establishes architecture correctness on the tested inputs. No pointer data, research training, checkpoint update, repair/damage experiment, or transfer evaluation was performed. The only optimizer step was a diagnostic update of a tiny random model in a unit test.

## Configuration and provenance

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, cached revision `7ae557604adf67be50417f59c2c2f167def9a775`. No model download was needed.
- Python 3.11.8; torch 2.14.0; transformers 5.17.0; huggingface-hub 1.31.0; peft 0.20.0; rich 15.0.0; pytest 9.1.1. All transitive versions are in the JSON artifacts; direct pins are not a complete environment lock.
- Float32, eager attention, no KV cache, evaluation mode for equivalence, strict deterministic algorithms, seed 17, four CPU threads. The backward check switches to training mode with zero attention/adapter dropout.
- P/R/C layer ranges: 0–5 / 6–17 / 18–23. No bridge. Recurrent q/v LoRA: rank 8, alpha 16, dropout 0, no trainable bias, zero-initialized B.
- 494,032,768 frozen original parameters; 270,336 trainable adapter parameters in 48 tensors (about 0.055% of the original count).
- Two plain-text probes: `The sky is blue.` and `What is two plus two? Answer briefly.` They are architecture fixtures, not task benchmarks or chat-generation tests.
- Cases: left-padded `[2,9]`, right-padded `[2,9]`, and unpadded `[1,5]` input IDs. Exact IDs and masks are saved. Default Qwen forward positions are used; tiny tests additionally cover explicit positions.
- Code: worktree based on `8af78e3bf8c24cdbf95e5fbb67c50d29667bf347`, with the implementation uncommitted at validation. JSON includes Git status and SHA-256 for source/test/requirements files, including the final `scripts/recurrent_qwen/` layout. Documentation edits made after the runs do not change those source hashes.

## Exact commands

From `/Users/mukunds/Desktop/loopformer`:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.validate_stage0 --device cpu --output artifacts/stage0/cpu.json
.venv/bin/python -m scripts.validate_stage0 --device mps --output artifacts/stage0/mps.json
```

Requirements installation completed and `pip check` returned `No broken requirements found.` The final focused suite reported `36 passed in 0.58s`. The MPS validation ran outside the agent sandbox, which otherwise reported MPS unavailable. CLI help and invalid-argument handling were also checked. All 90 local Markdown file links resolve, and both JSON artifacts' saved source hashes match the final code and requirements.

## Pretrained results

| Measurement | CPU | MPS |
| --- | --- | --- |
| Final run start (UTC) | 16:14:49 | 16:15:23 |
| Gate outcome | PASS | PASS |
| T=1 maximum / mean absolute error | 0 / 0 | 0 / 0 |
| Tolerance, absolute / relative | 1e-5 / 1e-5 | 1e-5 / 1e-5 |
| Loop counts verified | 1, 2, 4 | 1, 2, 4 |
| Original parameters with gradients | 0 | 0 |
| Trainable tensors with finite gradients | 48 / 48 | 48 / 48 |
| Initially nonzero B gradient tensors | 24 / 24 | 24 / 24 |
| Gradient L2 at h_1 / h_2 from final-loop loss | 1.27722 / 1.52975 | 1.27714 / 1.52973 |
| T=1 / T=2 / T=4 forward time, seconds | 0.1285 / 0.2393 / 0.4385 | 0.0345 / 0.0583 / 0.1104 |
| Two-loop forward + backward, seconds | 0.3794 | 0.1479 |
| Total run, seconds | 3.1230 | 2.1798 |
| Peak process RSS, GiB | 3.1900 | 3.1892 |

Each device's zero-error result compares its own ordinary forward to its own recurrent forward, across all three input cases both before and after adapter attachment. It is not a claim of bitwise CPU/MPS agreement. All token positions, including padding positions, and all 151,936 vocabulary logits were compared.

Hook checks observe one P call, T calls to the same R object, `12*T` recurrent decoder calls in order, and T C calls. Parameter IDs remain unchanged as T varies. Each next R input is the previous R output object. Captured hidden states, logits, and margins have the expected shapes and finite values.

A final-loop CE backward through two loops produces gradients in both recurrent states and all adapter tensors, while leaving all original parameters frozen and gradient-free. Initial A gradients are zero as expected from B=0. The tiny-model test performs one diagnostic update and verifies both A and B subsequently receive nonzero gradients while original weights remain exactly unchanged.

Margins use arbitrary diagnostic token labels drawn from the input IDs, including potentially special tokens. They check observability only; they do not constitute a validated symbolic answer vocabulary or a task-success result.

## MPS failure and fix

The first MPS attempt passed forward checks but failed backward because advanced-index answer selection invokes `index_put_with_accumulate_mps`, which lacks a strict deterministic implementation. An equivalent gather readout also failed, at `scatter_reduce_mps`.

The final readout uses per-example integer slices and stacking. It preserves values, shapes, gradient paths, and the selected answer positions while avoiding those scatter operations. The full MPS gate then passed with strict deterministic algorithms still enabled and unchanged tolerances. The tradeoff is one answer-index synchronization per forward plus a tiny batch-sized Python loop. No bridge, dtype conversion, nondeterministic fallback, or research-design change was introduced.

## Artifacts and limitations

- [CPU JSON](../../artifacts/stage0/cpu.json)
- [MPS JSON](../../artifacts/stage0/mps.json)

Both files live under Git-ignored `artifacts/stage0/`. They contain the exact configuration, inputs, resolved revision, packages, source hashes, checks, per-loop predictions/margins, trainable names and gradient norms, timings, and process RSS. They must be copied separately when sharing a checkout, or regenerated with the commands above. No trained checkpoint was saved.

Timing figures are individual short validation runs, not throughput benchmarks or repeated estimates. MPS intervals synchronize before timing ends. Peak process RSS is not a complete accounting of Metal/driver allocations or total unified-memory use. These figures do not establish a safe resource budget for Stage 1 training, longer prompts, mixed precision, or deeper differentiable unrolls.

Tiny tests cover SDPA and configurable split extremes; pretrained validation here covers eager float32 at the default split only. Pretrained numerical equivalence uses three small cases, not an exhaustive language evaluation. MPS CE backward is verified; MPS backward through margin-based objectives is not. No checkpoint serialization/resumption interface is implemented. Other Transformers versions, sliding attention, non-default RoPE, generation/KV caching, and large-depth performance are outside the verified scope.

The next milestone is [pointer symbols and data validation](../phases/stage1_pointer.md), with a stop before training. See [status](../status.md) and [decisions](../decisions.md) for remaining task semantics.
