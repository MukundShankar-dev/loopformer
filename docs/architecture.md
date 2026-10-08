# Architecture and implementation boundaries

Status: Stage 0, pointer data, Stage 1 training, and saved-checkpoint full-loop evaluation implemented. Pretrained CUDA training and final-answer/full-loop runs are audited in [status](status.md). Absorbing-terminal overscaling evaluation is implemented but pretrained execution is deferred. See the [validation report](experiments/stage0_validation.md) for architecture evidence and limits, and [data report](experiments/stage1_data_validation.md) for pointer validation. The research specification is [the project plan](project_plan.md).

The [inference smoke test](../scripts/smoke_test_qwen.py) exercises ordinary Qwen. The [Stage 0 entry point](../scripts/validate_stage0.py) loads the checkpoint, constructs the recurrent model, attaches LoRA, and validates the architecture. Stage 0 defaults to cached files at an immutable revision. Usage is in [setup](setup.md).

## Recurrent computation

The following full-sequence recurrence remains the historical default. The new `fixed_prompt` mode is an explicit checkpoint/configuration choice, described below; loading an old checkpoint never changes its recurrence.

Inspection of Transformers 5.17.0 confirmed `Qwen2ForCausalLM`, `Qwen2Model`, and tensor-returning `Qwen2DecoderLayer.forward`. The Qwen2.5-0.5B-Instruct checkpoint has 24 layers, hidden size 896, 14 attention heads, two KV heads, full attention, default RoPE, and tied embedding/head weights.

The configurable default split is layers 0–5 for P, 6–17 for shared R, and 18–23 for C:

```text
tokens -> embeddings -> P -> h_0
h_0 -> R -> h_1 -> R -> h_2 -> ... -> h_T
             each h_t -> C -> final norm -> LM head -> logits_t
```

`RecurrentQwen` groups the original decoder objects into `prelude`, `recurrent`, and `coda` (`DecoderBlock` modules), without copying weights or retaining duplicate registered paths. Embeddings, RoPE, final RMSNorm, and the LM head are reused; tied weights remain tied. Construction freezes and clears gradients on the supplied base **in place**. Adapter injection subsequently mutates its shared middle layers, so compute unadapted reference logits first or load a separate reference.

The next iteration consumes the recurrent state, never the coda output or a decoded token. P runs once; intermediate C reads expose predictions without modifying the recurrent trajectory. There is no bridge: direct middle-to-middle recurrence passes Stage 0. This does not establish useful recurrent task execution or imply that `h_0` decodes to a start symbol.

### Fixed prompt memory (2026-10-04)

`RecurrentQwen(..., recurrence_mode="fixed_prompt")` keeps the final unmasked prompt position writable and holds all other positions to their layer-specific first-pass context. [PromptMemory](../scripts/recurrent_qwen/memory.py) stores differentiable inputs and outputs separately for R and C within one forward call. Every R loop and C readout reuses those values; only the answer-position R output feeds the next R pass. No loop/depth scalar, generated token, reference symbol, new prompt or extra positional coordinate enters that update. The final unmasked answer position is enforced at the API boundary.

Memory tensors remain in the autograd graph, so later losses reach adapters that formed the context as well as earlier working states. They are discarded after the forward and recomputed with current parameters for the next batch. This reference implementation still executes dense sequence layers and restores fixed context at each layer; it does not implement a KV cache or claim an inference/training speedup. Returned `[B,S,H]` states contain a fixed first-pass prefix and the evolving working position. Full-state norm summaries therefore have different semantics from the old full-sequence recurrence.

One-pass equivalence, prefix invariance, parameter sharing, and the loss gradient path pass tiny-model tests. An independent recomputation reference matches both logits and adapter gradients. The pretrained startup gate is still required. New checkpoints use `loopformer-stage1-fixed-prompt-v1` with explicit `recurrence_mode: fixed_prompt`; the loader and adapter restore reject conflicting modes. Historical formats retain their old behavior. See [design, supervision and commands](learned_loop_completion.md#fixed-prompt-memory-and-recurrent-working-state).

`completion_threshold` activates the checkpoint's hidden-state head during eval/no-grad, batch 1. It stops on the first threshold crossing under `num_loops` as a safety cap, without receiving task depth. It cannot be combined with the older numeric-feature `stop_policy`. Intermediate C readouts are retained on this correctness path; timing includes their overhead. The CLI uses `--stop-policy completion` to select this distinct path.

### Literature cross-check (2026-09-30)

The computation is a valid **depth-recurrent, weight-tied middle-block transformer**, not an exact reproduction of every published looped transformer. The [Universal Transformer](https://arxiv.org/html/1807.03819) repeatedly updates the entire sequence representation with shared self-attention; our `h_t = R(h_{t-1})` has that basic depth-recurrence property. Its paper also adds an explicit recurrent-time coordinate and optionally adaptive per-position halting. We currently have neither. Reusing Qwen's same RoPE positions each pass preserves each token's sequence position but does **not** give R an explicit loop index.

The [ICLR 2024 looped-transformer study](https://proceedings.iclr.cc/paper_files/paper/2024/file/b8402301e7f06bdc97a31bfaa653dc32-Paper-Conference.pdf) compares the plain weight-tied update `Y_(t+1)=M(Y_t)` with **input injection** `Y_(t+1)=M(Y_t+P)` and reports deterioration beyond trained iterations without injection on its in-context regression setting. Our P output initializes recurrence once; it is not explicitly re-injected every loop. This is a plausible mechanism to investigate, not proof that missing injection causes the pointer failures. [Kapl et al. (2026)](https://proceedings.mlr.press/v306/kapl26a.html) explicitly study unique encoding/decoding blocks around a repeated middle block, which is closer to our P/R/C split; their positive results likewise do not validate our chosen 6/12/6 partition or imply one pointer transition per loop.

The code audit confirms the intended data flow: token embeddings pass through six frozen P layers once; a single shared 12-layer R object with q/v LoRA updates the full `[batch, sequence, hidden]` state at each loop; six frozen C layers, final norm, and the LM head read the answer position at every loop. C's output is discarded for recurrence. The same causal mask and position embeddings are reused, with no KV cache, new text tokens, external decoded state, learned time embedding, or injected P state. T=1 equivalence, shared-object identity, gradient-scope, and earlier-loop gradient tests establish implementation contracts, but do not establish that the trained hidden state encodes the current pointer correctly at every loop.

In evaluation, `h_0` and `h_t` are optional tensors exposed by the wrapper. The small probe saves **decoded A–Z readouts after C**, their target ranks/margins/entropy, and scalar RMS/update/cosine summaries of R's answer-position and whole-sequence states. It does not decode every token position as a symbolic pointer, save complete hidden vectors, or inspect layer-level attention. A correct readout after C is evidence of the observable output at that depth, not a direct measurement of the internal pointer representation or a proof that R alone performed exactly one transition. The [diagnostic guide](diagnostics_and_performance.md#frozen-rule-edit-probe) defines the next targeted measurement.

Mask construction uses the installed `create_causal_mask`. Every decoder receives `attention_mask`, `position_ids`, `position_embeddings`, `past_key_values=None`, and `use_cache=False`. The mask and RoPE are prepared once and reused across loops. Position IDs default to `arange(sequence_length)` as in ordinary Qwen forward, including padding; callers can supply explicit positions. Only full attention and default RoPE are accepted in Stage 0. The validation CLI selects eager attention and float32 explicitly; tiny tests also cover SDPA. Compatibility with other Transformers versions is not promised.

## Trainability and autograd

All 494,032,768 original parameters are frozen: embeddings, P, original R weights, C, final norm, and LM head. `attach_recurrent_lora(model)` uses PEFT in-place injection on `model.recurrent` only. Defaults are q/v projections, rank 8, alpha 16, dropout 0, bias `none`, and standard random-A/zero-B initialization. This adds 270,336 trainable parameters in 48 tensors. Rank, alpha, and projection targets are configurable in the library; the CLI fixes q/v targets.

Trainable names have the form `recurrent.layers.<relative-index>.self_attn.<projection>.lora_[A|B].default.weight`. Add `recurrent_start` to that relative layer index to recover the original Qwen layer number.

Frozen operations remain differentiable. Gradients pass through C to the adapters and through earlier R iterations. There is no internal `no_grad` around those paths. Zero A gradients on the first backward are expected when B is zero; focused tests verify both factors after a diagnostic tiny-model update. No optimizer step is taken on the pretrained model during validation.

Stage 1 uses differentiable unrolls with masked intermediate 26-symbol CE in `scripts/training/objective.py`. Later losses retain gradients through earlier recurrent states. Future Stage 3 detached rollouts and asymmetric objectives remain unimplemented.

The optional `loss_reduction=loop_mean` weights each nominal CE by N/(D*N_t), based on the entire selected training dataset. Per-example weighted sums and existing effective-batch accumulation estimate the dataset's equal-loop mean loss without microbatch-dependent denominators. Default `example_mean` is unchanged. Logs distinguish optimized loss from legacy example-mean CE; loop-balanced checkpoint selection averages loop means on trained-depth validation tasks only. The recurrent forward pass and evaluator's historical loss semantics are unchanged. See [objective and commands](training_pointer.md#loop-balanced-loss-experiment).

Batching changes only execution grouping: an optimizer update uses `batch_size * gradient_accumulation` examples, each with equal total nominal-loss weight. Mixed-depth microbatches unroll to their deepest task and mask later labels for shorter tasks. Explicit `--resume --allow-batch-change` permits repartitioning the same update group while retaining optimizer/RNG/cursor state; all other resume identities remain strict. Floating-point results need not be identical. Run metadata records the change; see [batching and resume](training_pointer.md#resume-with-larger-microbatches).

The training startup gate compares ordinary-Qwen and fresh-adapter T=1 logits over the full vocabulary at the answer position. Its reference uses `logits_to_keep` to project only that position, matching the wrapper's compact LM-head readout instead of projecting the whole prompt and slicing afterward. This removes a matrix-shape difference from the numerical comparison; `atol=rtol=1e-5` remains unchanged. Full-sequence equivalence remains covered separately by Stage 0. The change does not alter recurrent forward behavior, the training objective, or saved-checkpoint evaluation.

## Implemented forward interface

`model(input_ids, attention_mask=None, *, num_loops=1, position_ids=None, answer_positions=None, labels=None, allowed_token_ids=None, return_hidden_states=False, logits_mode="answer", stop_policy=None)` returns `RecurrentOutput`.

- Inputs are long `[B,S]` token IDs and a binary `[B,S]` mask. Each row needs an unmasked token. Positions are long `[1,S]` or `[B,S]`; loops do not append token positions.
- Answer positions are long `[B]`, defaulting to the last unmasked token for either padding side. They select a next-token prediction location, not a token to insert into the prompt. Padding positions are rejected. Integer slices implement this readout because advanced indexing and gather backward fail under strict MPS determinism; indices synchronize to a Python list once per forward. This is appropriate for the intended tiny batches.
- `loop_logits` is a tuple of length T for fixed-depth calls (or the executed prefix when stopping), with tensors `[B,V]` by default or `[B,S,V]` in `"all"` mode. `.logits` returns the final item. Every loop reads C; no final-only supervision is imposed.
- With hidden capture enabled, `initial_hidden_state` is `h_0`, and `hidden_states[t-1]` is `h_t`, each `[B,S,H]`. Otherwise both fields are `None`; autograd can still retain the activations required for backward.
- Optional labels are long `[B,T]` **token IDs**, with an explicit unique allowed set `[A]` of at least two tokens. `margins[B,T]` uses the target raw logit minus the largest other allowed logit. Labels can differ by loop. Repeating a final label is an explicit caller choice. This interface neither computes CE nor shifts labels.
- Outputs retain gradients when autograd is enabled. Use caller-side `torch.no_grad()` for evaluation. No bridge, KV cache, generation API, or detached-rollout training is implemented. Adapter save/load is provided separately by `scripts/recurrent_qwen/checkpoint.py`.

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
| `scripts/eval/naive_test.py` | Ordinary-model final-answer evaluation CLI, progress, CSV/JSON artifacts |
| `scripts/eval/pointer_task.py` and `loading.py` | Exact final-answer scoring, generation batches, and ordinary-model loading |
| `scripts/training/config.py`, `data.py`, `objective.py` | Explicit configuration, validated subsets/batches, masked per-loop loss and metrics |
| `scripts/training/gates.py`, `runner.py`, `evaluation.py` | Startup checks, optimizer/accumulation/resume, fixed per-loop validation |
| `scripts/training/train_pointer.py` | Composing training CLI and compact Rich dashboard |
| `scripts/recurrent_qwen/checkpoint.py` | Adapter/tokenizer/optimizer persistence and recurrent model reconstruction |
| `scripts/eval/recurrent_pointer.py` | Saved recurrent final-answer readout through the naive-test CLI |
| `scripts/eval/loop_test.py`, `loop_metrics.py` | Fixed-depth checkpoint CLI, per-example first-error diagnostics, and depth-by-loop CSV export; optional adaptive routing |
| `scripts/eval/adaptive_*.py` | Offline oracle/heuristic analysis, causal stopping, latency summaries, and gated head fitting; see [one adaptive guide](adaptive_compute.md) |
| `scripts/dataset/terminal.py` | Deterministic, evaluation-only terminal-rule transform with unchanged nominal targets |
| `scripts/eval/overscaling_test.py`, `overscaling_metrics.py` | Terminal sweep through the shared loop CLI, conditional dynamics and censored survival |
| Future evaluation extensions | Knowledge retention, transfer, hidden-state diagnostics, and plots |

Scripts compose library functions. Task generation must remain separate from training, and loss functions must not own loading or artifact writing. See [Stage 0](architecture.md) for commands and [decisions](decisions.md) for remaining research semantics.

## Stage 1 data boundary

Pointer generation has no dependency on the recurrent wrapper and loads no model. Logical state targets and actual answer-token IDs are saved together. The trainer revalidates raw prompts with its tokenizer, derives class indices `[B,T]` from exact logical intermediate states, and selects the validated symbol-token logits. Right-padded batches mask nominal labels at `t > d`; CE is averaged within each example and then across examples. Saved answer positions apply to unpadded raw prompts; batching must account for padding. Only nominal steps 1..d are labeled, with no h_0 or post-completion supervision. The [Stage 1 guide](training_pointer.md) owns the data format and reproduction commands.

The [ordinary-model evaluator](evaluation.md) loads `AutoModelForCausalLM` directly and does not use `RecurrentQwen`. It renders the shared `prompts/pointer_task.txt` instructions, tokenizes with the evaluated model's tokenizer, and scores generated final answers. Its chat-wrapped autoregressive output is distinct from frozen-coda readouts after recurrent loops; the CLI separately recognizes recurrent checkpoint directories and routes them to allowed-symbol readout at the requested depth. Training validation also records per-loop trajectories. See [training usage](training_pointer.md) for the checkpoint format and runtime boundaries.

The [full-loop CLI](evaluation.md) restores the same saved model/tokenizer and uses training's `encode_tasks` and `evaluate` under `no_grad`. Each example runs the full requested sweep; nominal masking and loss reduction are shared with training. CSV rows additionally expose intermediate CE and intermediate margins, without changing training objectives or forward behavior. Diagnostic aggregation reads those rows to locate first errors and decoded repeats; it does not introduce hidden-state probes or post-completion targets.


The deferred terminal experiment composes that same inference path via `overscaling_test.py`. Before encoding, `terminal.py` returns new task records with only the final outgoing edge changed to a self-loop. `tasks.jsonl` and its hash preserve the exact changed inputs. Nominal CE remains masked after d; post-nominal fixed-final scoring lives exclusively in `overscaling_metrics.py` and never enters the optimizer. No hidden states, model internals, or gradient behavior are changed. Conditional counts and survival are validated independently of model inference; see [usage and semantics](evaluation.md).


Depth-stage initialization is separate from resume: `scripts/training/initialization.py` validates the saved adapter interpretation and training-exposure metadata, while the training CLI restores only adapter tensors before creating a new optimizer. Checkpoint lineage survives subsequent resume. `scripts/eval/depth_generalization.py` composes matched `loop_test` sweeps; `depth_comparison.py` checks pairing/provenance and reports absolute and relative depth. No recurrent forward or gradient semantics change. See [depth-6 setup](training_pointer.md).

## Adaptive-compute boundary

[Adaptive inference compute](adaptive_compute.md) owns the implemented opt-in stopping interface and its remaining research gates. `RecurrentQwen.forward(stop_policy=...)` accepts a causal callback in eval/no-grad batch-1 mode, reads the shared recurrent state after each loop, and returns the executed prefix. The default fixed-depth forward and training gradients remain unchanged. A separate lightweight head may consume target-free confidence/history features; no pretrained head or adaptive benefit has been established.

## Explicit-step learned completion ablation

The separate [completion training path](learned_loop_completion.md) adds an optional LayerNorm → Linear → GELU → Linear head to `RecurrentQwen`. At each loop it reads only `h_t` at the answer position and returns `stop_logits[B,T]` alongside the unchanged coda logits. It does not receive the loop index or parsed requested depth. The trainer jointly optimizes the existing intermediate 26-symbol CE and a weighted binary continue/stop objective; the latter is supervised only through each example's nominal depth and backpropagates through R. No predicted or reference symbol is inserted into recurrence. Fixed-depth outputs of a head-equipped checkpoint still use the same R/C readout; the self-stopped inference path is not yet implemented. The head and adapters use a versioned checkpoint, while historical adapter-only checkpoints remain loadable. Tiny-model checks passed; pretrained training has not run.

---

## Stage 0 validation protocol (completed)

This section retains the implementation contract and validation commands. Current evidence is in [status](status.md) and the [Stage 0 report](experiments/stage0_validation.md).

Status: implemented. The gate outcome and device-specific evidence are in the [validation report](experiments/stage0_validation.md). Source: project plan sections 2–4 and 32.

### Purpose and scope

Prove that model surgery preserves one-pass behavior and repeated passes use the intended trainable parameters. No pointer data, research training, bridge, or asymmetric objectives are included.

### Implementation

- [Model mechanics](../scripts/recurrent_qwen/model.py): original Qwen decoder objects grouped into P (0–5), shared R (6–17), and C (18–23), with configurable half-open split boundaries. Preserve embeddings, masks, positions, RoPE, final norm, and tied LM head; freeze original weights and disable caches.
- [LoRA](../scripts/recurrent_qwen/lora_utils.py): PEFT injection into R only, q/v projections, rank 8, alpha 16, no dropout or bias training. A single shared set adds 270,336 trainable parameters to the frozen 494,032,768-parameter model.
- [Outputs](../scripts/recurrent_qwen/outputs.py): per-loop logits, optional `h_0` and `h_1...h_T`, and allowed-token margins with explicit per-loop labels. [Architecture](architecture.md#implemented-forward-interface) defines shapes and indexing.
- [Validation](../scripts/recurrent_qwen/validation.py): ordinary references captured before mutation, equivalence before/after adapter attachment, module/parameter identity hooks, recurrent input identity, observability, and a two-loop backward without an optimizer step.
- [CLI](../scripts/validate_stage0.py): explicit checkpoint loading, Rich results, and JSON evidence. Imports never load models. Default is local-only, fixed-revision CPU float32 with eager attention and strict deterministic algorithms.

The wrapper shares and freezes the supplied base in place. Compute reference outputs before LoRA attachment; a base sharing adapted layers is no longer an independent unadapted reference.

The [integrated mechanism diagnostic](diagnostics_and_performance.md#integrated-frozen-mechanism-diagnostic) uses scoped eval-only hooks to record `h_t` at Answer after R, normalized Answer vectors after frozen C, and selected eager-attention weights from Answer to the 26 rule source/destination tokens. It also reads frozen C(`h_0`) once before recurrence, with the same mask and RoPE, to audit what P+C can already decode; that extra readout never feeds the model. Adapter ablation temporarily disables recurrent LoRA in memory and restores it even on errors. These probes do not alter `RecurrentQwen.forward`, training gradients, checkpoint files, or what is fed into the next loop. Rule tokens precede Start/Answer under the causal mask, so their representations cannot attend backward to the answer-position state; the answer position can attend to the rule table. Saved attention ranks and linear decodability are observations, not causal proof of which computation R performs.

### Run validation

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

### Validation and gate

| Check | Evidence |
| --- | --- |
| T=1 equivalence | Every token's full-vocabulary logits match ordinary Qwen with `atol=rtol=1e-5`, before and after initial LoRA; pretrained cases include no padding and both padding sides |
| Shared weights | Original layer identities retained; every loop invokes the same R object, decoder objects, and parameter IDs; P runs once and C once per loop |
| Recurrent flow | Next R input is the previous R output object; tiny-model tests also check causality |
| Gradient scope | Every original parameter stays frozen and gradient-free; only R's A/B tensors are trainable, all gradients are finite, and all initial B gradients are nonzero |
| Gradient connectivity | Final-loop CE reaches both recurrent hidden states through frozen C and earlier recurrence; a tiny-model diagnostic update additionally verifies nonzero A gradients |
| Observability/depth | Correct logits, hidden-state, and margin shapes at multiple depths; independent margin arithmetic and readout-selection tests |

Focused tests use tiny random Qwen models for split extremes, explicit/default positions, padding, eager/SDPA equivalence, causality, gradients, ties, and invalid boundaries. They are separate from validation of the actual pretrained checkpoint. Exact commands, measurements, and limitations are in the [report](experiments/stage0_validation.md).

The hard gate is passing equivalence, sharing, and gradient scope. An unavailable or failed device is not a pass on that device. Stage 0 establishes implementation correctness on tested inputs, not useful recurrence, pointer execution, repair, or overscaling damage.

### Stop boundary

Stage 0 ends here. Next is the [symbol vocabulary and pointer-data generator](training_pointer.md), including tests, followed by another stop before training. Completion semantics, cycles, and task-specific readout remain decisions for that milestone.

## Performance investigation boundary

The [diagnostics and performance proposal](diagnostics_and_performance.md) records inspected overhead in symbolic projection, metric synchronization, batching and execution settings. No fast-path model change has been implemented. Preserve full-vocabulary one-loop equivalence, intermediate readouts and gradients, frozen base weights, and shared recurrence while measuring candidates.

The initial profiler uses temporary module hooks and opt-in evaluation ranges to attribute existing computation, without changing recurrent tensors or gradient paths. The separate small probe uses the existing hidden-state capture interface one example at a time. Disposable profile updates use the production objective/backward with fresh optimizer state and restore the source adapter values in memory; source checkpoint files are never rewritten. See [implemented diagnostics](diagnostics_and_performance.md#implemented-diagnostic-commands).

The opt-in [controlled probes](diagnostics_and_performance.md#controlled-restart-and-rule-context-experiments) use scoped evaluation hooks in `pointer_probes.py`: capture P's rule-prefix output, then substitute that prefix at R input only after the configured boundary. Remaining sequence positions stay recurrent; C/readout and weights remain unchanged. An identical-copy control checks output equality. This is an inference intervention, not a trained architecture or an extension to the production forward API. The hooks require eval/no-grad and are removed after each forward, including exceptions.

## Depth-12 execution settings

The broader-depth recipe retains the fixed-prompt model, shared LoRA, completion head and exact per-loop labels. Optional `attention: sdpa` selects PyTorch SDPA through Qwen's existing attention interface. Optional `precision: bf16` uses CUDA autocast only during forward computation; original/trainable parameter storage, AdamW state, pointer CE and completion BCE stay float32. No tensors are detached from the training recurrence. The startup one-pass equivalence gate remains float32, and BF16 launches additionally check a full deepest-depth backward pass for finite gradients and frozen base parameters. CUDA support is required explicitly. Defaults remain float32/eager/deterministic for old configs. SDPA output/adapter-gradient agreement is tested locally on a tiny CPU model; CUDA BF16 validation is deferred to the desktop probe/smoke run. Fixed prompt memory still uses dense layers; no projected-memory cache was added.

## Isolated executor interfaces — 2026-10-04

The user-authorized [pipeline upgrade](pipeline_upgrade.md) adds opt-in interfaces
while preserving the legacy full-sequence and fixed-prompt models.

```text
raw Rules/Start/Steps prompt
  ├─ remove Steps line → P → R → [bridge → same R] … → C → symbolic logits
  └─ full prompt → P (no gradient) → controller initial memory
                                  ↑ detached R working state after each loop
                                  GRU → stop logit
```

Routing is deterministic token/text plumbing, not a learned lookup or a numeric
counter. It requires the raw dataset format. The executor's count line is removed;
masked gaps preserve the public answer index and compact RoPE indices remove
count-token-length leakage. The controller reads the full prompt representation,
has its own GRU memory, and never receives t, d, remaining depth or reference
states as numerical features. Its output cannot enter R. BCE gradients and
clipping cannot rescale executor updates. Full-model training may change the
shared P weights through pointer loss, so controller prompt features can still
change between optimizer updates.

The bridge acts only before loops 2..T. It normalizes the returned working vector
by RMS, restores the initial executor-state RMS scale, and applies a learned gain
and identity-initialized linear projection. It does not freeze states or enforce
contraction. A direct learned linear A–Z head on R adds optional local CE;
C's symbolic output remains the primary answer path. Neither decoded output is
fed back. T=1 equivalence is with ordinary Qwen on the **executor view**, not with
ordinary Qwen on the unmodified full prompt.

`train_scope` supports LoRA, all recurrent-block weights, or all model weights.
Full R is the default new experiment; full model is a separate config. The new
`loopformer-executor-v2` checkpoint stores all trainable tensors in the historical
`adapter_model.pt` filename and records the architecture explicitly. That filename
no longer implies LoRA-only contents. The shared loader reconstructs the correct
trainability and interfaces; naive/full-loop/learned-stop evaluation reuse it.
Legacy checkpoint interpretation is unchanged.

Differentiable prefix reuse computes full R/C layers on pass one, retaining
prefix K/V and outputs within that forward. Later loops compute only the writable
query and replace its K/V; prefix tensors are **not detached**. This is separate
from Hugging Face's generation cache, which remains disabled. Non-reentrant
checkpointing recomputes layer activations during backward. Reuse requires zero
attention dropout. Selected-row LM projection computes the same 26 answer logits
without materializing the whole vocabulary; full-vocabulary output remains
available for legacy callers and equivalence tests.

Tiny-model tests compare dense/reused outputs and all parameter gradients for
LoRA/full R/full model under eager and SDPA, including checkpointing and different
prompt lengths. Other tests cover count invariance, separate gradients, bridge
T=1 behavior, save/reload and complete training/evaluation wiring. These prove
implementation contracts to numerical tolerance, not pretrained learnability or
unbounded algorithmic correctness. Pretrained CUDA startup checks remain required.


## Frozen controller diagnosis

The [controller diagnostic](diagnostics_and_performance.md#controller-diagnostic--current-desktop-command)
uses scoped read-only hooks on the existing `RecurrentController` modules. It
captures the input/output of `context`, the input of `observation`, and the GRU
output. The normal model forward and recurrent executor are unchanged. Cached
observations have shape `[questions, loops, hidden_width]`; controller memories
are `[questions, loops, controller_width]`. Prompt context has no loop axis.

Replay initializes the controller once and advances its own memory through every
cached R state. The existing stop loss constructs labels from requested depth;
neither these labels nor probe predictions enter the controller's forward. The
source model and original controller stay frozen; tiny-set fits update copies.
Diagnostic classifiers are separate measurement tools and are never installed in
the inference model. The only new binaries are ignored caches and controller
copies in the diagnostic output directory; no checkpoint format changes.


## Controller-only training and portable export

The [separate controller run](training_pointer.md#separate-controller-training--current-desktop-run)
freezes the selected executor and caches normal FP32 forward observations. Only
a copy of its `RecurrentController` enters the optimizer. The controller consumes
cached full-prompt context and R vectors, carries its own differentiable memory,
and receives no numeric time/depth feature. Per-loop continue/stop labels remain
supervision only. Cache replay is checked against the original live controller.

The portable `best/` checkpoint uses the existing `loopformer-executor-v2` format.
Export copies the original saved tensor dictionary and replaces exactly the
`completion_head.*` keys after name, shape and finiteness validation. It does not
use `requires_grad` to infer the export payload, which would omit the frozen
trained R. Source tensors/tokenizer/config remain untouched; metadata records
controller-only optimization and the source hash. The joint trainer must not use
this export as an optimizer-resume checkpoint.


## Controller remaining-work supervision — 2026-10-05

The [matched comparison](training_pointer.md#remaining-work-comparison--current-desktop-run)
adds a training-only `Linear(controller_width, 1)` readout without changing
`RecurrentController.initialize` or `advance`. `replay` returns initial memory
`[B,M]` and subsequent memory `[B,T,M]`; concatenating them gives `[B,T+1,M]`.
The shared linear readout, multiplied by fixed scale 12, predicts `[B,T+1]`
remaining work. Only the objective constructs targets N−t for t=0..N. There is
no teacher forcing, numerical feedback, clock input, hard-coded state update or
readout-dependent stopping. Full controller BPTT remains intact; input features
are detached at the existing prompt/observation interfaces.

The stop-only arm detaches memory before numerical regression. Its readout has a
separate optimizer and gradient clipping, so measurement does not alter controller
updates. The auxiliary arm lets regression gradients train memory. Export still
replaces exactly `completion_head.*`; the numerical readout is saved separately
with the matching selection step/scale and discarded for ordinary inference.
A scalar regression loss encourages an interpretable numerical representation;
it does not prove a counting algorithm or remove the bounded GRU's possible
extrapolation limitations. Count interpolation and longer-depth results remain
separate empirical questions.


## Read-only controller training audit

The [training audit](diagnostics_and_performance.md#controller-training-audit--current-desktop-command)
reuses cached `[B,H]` context and `[B,T,H]` executor observations and the existing
`replay`/`evaluate_controller` functions. It loads only saved controller and
numerical-readout weights. Forward inputs, initialization and recurrent memory
updates are unchanged; no target, decoded number or external counter enters them.

`controller_audit_metrics.loss_components` decomposes the existing scalar objective
into stop, initial (t=0), interior (0<t<N) and terminal (t=N) terms, retaining
original normalization. `torch.autograd.grad` observes each component's parameter
gradients without populating `.grad`, clipping tensors or creating an optimizer.
Controller states stay differentiable through the full rollout; frozen feature
inputs retain their existing detach boundaries. Weight-zero controls additionally
measure explicitly counterfactual auxiliary gradients, which are excluded from
the recorded actual controller objective. No numerical readout replaces stopping.

Best/final selection is an inspection label, not checkpoint promotion. Full-panel
fit and gradient diagnostics remain development evidence. Parameter tensors and
input hashes are checked unchanged after the audit, and no new weights are saved.

## Independent initialization supervision — 2026-10-08

The [repair intervention](controller_repair.md) adds an independently weighted
loop-zero regression objective to the same controller and numerical readout.
Its direct gradient reaches context initialization and the training-only readout;
it does not directly supervise recurrent updates or stop logits. Existing
trajectory and stop losses still train those paths through full BPTT. No forward
input, recurrence, export tensor name or inference policy changes. A zero default
weight preserves historical objectives. Generalization remains unverified.
