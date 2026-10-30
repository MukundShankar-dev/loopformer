# Architecture

The code is the source of truth. This page describes what `scripts/recurrent_qwen/` and `scripts/training/` implement. Commands are in [usage](usage.md); history and results are in [history](history.md); research design is in [plan](plan.md).

## Base model and task

- Base: `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775` ([symbols.py](../scripts/dataset/symbols.py)). There are 24 full-attention decoder layers, hidden size H=896, 14 query heads and 2 KV heads (head dim 64), vocabulary 151,936 and tied embeddings. Only full attention and default RoPE are accepted (`RecurrentQwen.__init__`).
- Prompt ([pointer.py](../scripts/dataset/pointer.py) `render_prompt`): raw text, no chat template, no special tokens:
  `Rules: ( A, C) ( B, D) ...\nStart: X\nSteps: N\nAnswer:`
- Each `' A'`..`' Z'` is one unique token in rule, start and answer contexts (`validate_symbols`). The readout position is the final `:` token.
- Targets: `intermediate_states` = the states after transitions 1..N. Loop t is supervised with the state after t transitions. `EncodedExample.targets` holds class indices into `SYMBOLS` (0..25), not tokenizer IDs ([data.py](../scripts/training/data.py)).

## P / R / C split

`RecurrentQwen(base, recurrent_start=6, recurrent_end=18, recurrence_mode=...)` ([model.py](../scripts/recurrent_qwen/model.py)):

| Block | Layers | Calls per forward |
| --- | --- | --- |
| P `prelude` | 0–5 | 1 |
| R `recurrent` | 6–17 (12 layers) | T (`num_loops`) |
| C `coda` | 18–23, then final `norm` and `lm_head` | T (once per loop; the output is never fed back) |

`DecoderBlock` wraps the base model's own layer objects, so there are no copies, and R uses one parameter set for every loop. Construction freezes the base in place (`requires_grad_(False)`) and sets `use_cache=False`. Every decoder layer call passes `past_key_values=None, use_cache=False`. There is no KV cache across forwards or generated tokens.

### Forward pass (`RecurrentQwen.forward`)

Inputs: `input_ids`/`attention_mask` [B,S]; `position_ids` [1,S] or [B,S] (default `arange(S)`); `answer_positions` [B] (default: last unmasked token). Per call:

1. If a `PromptRouter` is attached (isolated executor), first compute `controller_prompt_state` from the **raw** prompt. Then route the inputs (see below).
2. `hidden = P(embed(ids))`. Keep the answer-position vector `bridge_initial` [B,H].
3. For `fixed_prompt`, create `PromptMemory` objects for R and C, with `write_mask` [B,S,1] true only at the answer position.
4. For each loop t = 1..T:
   - If t>1 and a bridge exists, replace the working position with `bridge(h_working, bridge_initial)`.
   - `hidden = R(hidden)`; `working = hidden[b, answer_pos]` [B,H].
   - Controller/completion head → stop logit [B]; `state_head(working)` → [B,26].
   - `scores = lm_head(norm(C(hidden)))` at the answer position: [B,V], or [B,26] when `readout_token_ids` is given (a row-sliced `lm_head`). `logits_mode="all"` gives [B,S,V].
   - Optional `loop_callback`, margins and stopping (below).
5. Return `RecurrentOutput` ([outputs.py](../scripts/recurrent_qwen/outputs.py)): `loop_logits` (tuple of T; index t-1 is loop t), `hidden_states` (T × [B,S,H], if requested), `initial_hidden_state` (P output), `margins` [B,T], `state_logits` [B,T,26], `stop_logits` [B,T].

C stays in the autograd graph even when frozen, so readout losses reach R.

### Recurrence modes

- `full_sequence` (legacy): every position is rewritten by R on every loop. h_t = R(h_{t-1}) over the whole sequence.
- `fixed_prompt` (all executor checkpoints): only the final unmasked position is writable. `PromptMemory` ([memory.py](../scripts/recurrent_qwen/memory.py)) stores every layer input and the block output on pass 1. On later passes, `torch.where(write_mask, new, stored)` restores all non-working positions. The prompt therefore keeps its first-pass representation, and only the working state evolves. The stored tensors are not detached, so later losses reach parameters through the first-pass prompt. The memory lives for one forward only. `answer_positions` must equal the last unmasked token.
- `prefix_reuse` (executor option; [prefix.py](../scripts/recurrent_qwen/prefix.py) `fixed_prefix_layer`): this is the compute-saving form of `fixed_prompt`. Pass 1 runs densely and records each layer's rotary K/V. Later passes compute only the working query [B,1,H] and splice its K/V into the recorded prefix K/V. It needs zero attention dropout. `validate_executor` checks it against the dense path at 1e-4. This is reuse within a forward, not a KV cache.
- `gradient_checkpointing`: non-reentrant `torch.utils.checkpoint` per layer, only when grad is enabled.

### Executor interfaces (`configure_executor`)

`configure_executor(tokenizer, train_scope, bridge, direct_readout, controller_size, isolated, prefix_reuse, gradient_checkpointing, controller_kind)` requires `fixed_prompt`. Legacy checkpoints never call it.

- `PromptRouter` ([interfaces.py](../scripts/recurrent_qwen/interfaces.py)), attached when `isolated=True`. It decodes the valid tokens, requires the exact regex `Rules: …\nStart: [A-Z]\nSteps: [1-9][0-9]*\nAnswer:`, deletes the `\nSteps: N` line (the value is never parsed) and re-encodes. The routed tokens are left-aligned. The final token stays at the original answer index, the gap is masked, and `position_ids = cumsum(mask)-1`, so RoPE positions are contiguous. R and C therefore never see the count digits or their token length. Tokenization is cached by `lru_cache(4096)`.
- `ReentryBridge(H)`: `projection(gain * state/rms(state) * rms(initial))`, with the projection initialized to identity. It applies only for t>1, so T=1 is unchanged.
- `state_head = Linear(H, 26)` when `direct_readout=True`: a direct readout of R's working state. It is never fed back and does not feed C.
- Controller: `controller_head(H, controller_size, controller_kind)` is stored as `model.completion_head`. Its inputs are always detached, so no stop gradient reaches the executor.

### Controllers

| Class | `kind` | Initial memory | Per-loop update | Reads executor state? |
| --- | --- | --- | --- | --- |
| `CompletionHead` ([completion.py](../scripts/recurrent_qwen/completion.py)) | legacy, non-isolated | none | `MLP(LN(working))` → stop logit | yes, with gradient into R |
| `RecurrentController` | `gru` | `tanh(Linear(LN(P(raw prompt)[answer])))` (128-d) | `GRUCell(tanh(Linear(LN(working.detach()))), m)` → `Linear` | yes (detached) |
| `AffineSuffixController` | `affine_suffix` | `Linear` over the last 8 raw-prompt token embeddings flattened (8·H) → scalar | `m ← a·m + b` (initialized a=1, b=0); stop = `readout(m.detach())` | **no** |
| `SharedNumberController` | `shared_number` | `SharedNumberReader`: frozen embeddings of the Steps digit tokens (≤8, left-aligned, plus a validity channel), composed as `s ← gain·s + w·e_i + c` in float64 | same affine cell | **no** |

The final checkpoint (`controller-shared-number-precision-seed83`) uses `SharedNumberController`. It reads only the count tokens' frozen embeddings and runs a learned scalar countdown. It ignores the executor state (`advance` calls `observation` only to keep hooks working). Stopping does not depend on the pointer's identity or correctness. `number_token_ids` ([number_reader.py](../scripts/recurrent_qwen/number_reader.py)) extracts the Steps token span by regex. It checks one token per decimal character, but it never converts the text to an integer. Stopping uses the first loop where `sigmoid(stop) >= threshold`. The benchmarks and the demo use a threshold of 0.5.

### Stopping and observation (inference only)

All three of these require eval mode, `no_grad`, batch 1, no labels and `logits_mode="answer"`:

- `completion_threshold`: break after the first loop with stop probability ≥ the threshold. `num_loops` is the safety cap (272 in the benchmarks and the demo).
- `stop_policy(t, hidden, scores) -> bool`: heuristic policies from [scripts/eval/adaptive_policy.py](../scripts/eval/adaptive_policy.py).
- `loop_callback(t, scores[1,A], stop[1]|None, state[1,26]|None)`: receives detached clones after each readout, including the stopping loop. Its return value is ignored. The demo uses it to stream output ([scripts/demo/inference.py](../scripts/demo/inference.py)).

`answer_margin` gives the correct raw logit minus the best other allowed logit. Labels are token IDs [B,T].

### Historical partitioned path

[partitioned.py](../scripts/recurrent_qwen/partitioned.py) `partitioned_forward` is an inference-only factorization for historical count-dependent models. It computes a shared prompt prefix once and runs count-specific suffixes separately. It rejects models with a router, bridge or state head.

## Gradient scope (`train_scope`)

| Scope | Trainable | Frozen |
| --- | --- | --- |
| `lora` | PEFT LoRA on R's `q_proj`/`v_proj` (rank 8, α 16, no dropout, zero-init B; [lora_utils.py](../scripts/recurrent_qwen/lora_utils.py)): 12×2×2 = 48 tensors | all base weights |
| `recurrent` (`executor_r.json`) | every weight of layers 6–17 | P, C, embeddings/`lm_head`, final norm |
| `full` | all base parameters | — |

Bridge, `state_head` and controller parameters are trainable in every scope. Isolation guarantees, checked by `validate_executor`:

- Pointer and state losses do not reach the controller.
- The stop loss does not reach any non-controller parameter: `initialize` and `advance` detach their inputs, and `controller_prompt_state` runs under `no_grad`.
- `clip_training_gradients` ([runner.py](../scripts/training/runner.py)) clips the executor group and the controller group separately to `max_grad_norm`.

## Training objective ([objective.py](../scripts/training/objective.py), [runner.py](../scripts/training/runner.py) `train`)

The forward pass runs `num_loops = max target depth in the microbatch` through `forward_symbols` (26-row readout, scores [B,T,26]). With `target_mask` [B,T], loops past an example's depth contribute zero loss.

- Pointer: `step_loss`, cross-entropy over the 26 symbols at every loop t ≤ depth (per-loop intermediate supervision).
  - `example_mean` (default): mean over the example's valid loops, then mean over examples.
  - `loop_mean`: weighted sum with `loop_loss_weights`, w_t = N/(D·N_t), computed from the whole training set's depth counts.
- Direct state: `+ state_loss_weight × step_loss(state_logits)` with the same targets. The weight is 0.25 in `executor_r.json`.
- Stop: `+ completion_loss_weight × completion_loss(stop_logits, mask)`. This is BCE with continue labels at 1..d-1 and a stop label at d. The two classes are averaged per example (only the stop class when d=1). The weight is 0.1 in `executor_r.json`.
- Microbatch losses are scaled by `len(items)/len(group)` for accumulation.
- AdamW (`foreach=False`), linear warmup, optional cosine schedule to `min_lr_ratio` over `schedule_steps`.
- BF16 autocast is CUDA-only. Parameters and optimizer state stay FP32 ([precision.py](../scripts/training/precision.py)).
- `bucket_by_depth` shuffles depth-homogeneous update groups.
- Checkpoint selection (`training_selection_loss`) uses validation pointer CE on trained depths only. Stop and state losses are logged but are not used for selection.

Controller-only training ([train_controller.py](../scripts/training/train_controller.py), [controller.py](../scripts/training/controller.py), [controller_remaining.py](../scripts/training/controller_remaining.py)) runs on a cached, frozen executor:

- **Cache.** `extract_features` stores `context` [N,C], `working` [N,T,H] and predictions. `replay` re-runs the head over the cache and must match the live logits.
- **Losses.** `completion_loss`, or `prefix_losses` (stop BCE plus countdown MSE through min(N, `training_loops`)). Optionally added: `remaining_loss` (MSE on (N−t)/scale for t=0..N) and `initialization_loss` (loop-0 MSE).
- **Affine heads.** The initializer is fitted in closed form by `fit_suffix_initializer` ([controller_cache.py](../scripts/training/controller_cache.py)) and then frozen. The cell gain uses `affine_gain_lr_ratio`. The remaining-work readout is fixed at weight 1/scale, so the scalar memory itself is the predicted remaining count.
- **Selection.** Trained-count development exact-stop rate, with ties broken by BCE.
- **Final controller.** Two steps follow this training:
  1. `fit_number_reader` (least squares on training count labels only; [training/number_reader.py](../scripts/training/number_reader.py)) replaces the affine initializer with the shared reader.
  2. `refine_countdown` (float64 L-BFGS on the same labels, float32 deployment) refits only the two cell parameters.

  Each step exports through `export_checkpoint`. Executor tensors stay bitwise identical, and the script asserts this.

## Startup gates

[train_pointer.py](../scripts/training/train_pointer.py) runs these checks in order before any update:

1. `initialize` ([gates.py](../scripts/training/gates.py)) on the **bare** `RecurrentQwen` after the scope is set and before `configure_executor`:
   - T=1 logits match ordinary Qwen at the answer position (atol/rtol 1e-5).
   - A forward hook shows two R calls with identical parameter IDs.
   - In `fixed_prompt`, prefix positions are bitwise equal across loops.
   - A last-loop CE reaches both loop states.
   - No frozen parameter gets a gradient, and LoRA scope trains only `recurrent.*lora_*`.
2. `executor_view_equivalence` ([executor_gates.py](../scripts/training/executor_gates.py)) after `configure_executor`: the T=1 26-symbol logits match ordinary Qwen run on the **routed**, contiguous token sequence (1e-5). Routing changes R's input, so step 1 alone is not enough.
3. `validate_executor`:
   - Prefix-reuse and dense two-loop logits agree within 1e-4.
   - Loop-2 CE reaches every working state.
   - No pointer-loss gradient reaches the controller, and the bridge does receive gradient.
   - Controller BCE stays inside `completion_head.*`.
4. `validate_completion_gradients` (non-isolated completion head only) and `validate_compute_batch` (BF16, at the deepest training microbatch).

`python -m scripts.validate_stage0` ([validation.py](../scripts/recurrent_qwen/validation.py)) is the standalone pretrained gate. It checks all-position T=1 logits with and without fresh LoRA, the call counts of P, R and C, depth configurability, and a two-loop backward pass.

## Checkpoints ([checkpoint.py](../scripts/recurrent_qwen/checkpoint.py))

A checkpoint directory (`step-XXXXXX/` or an exported `best/`) contains:

- `adapter_model.pt`: every `requires_grad` tensor by module name. For the `recurrent` scope this means the full R layer weights plus bridge, `state_head` and `completion_head.*`, not only adapters.
- `recurrent_config.json`: the spec, including `format`, `base_model`, `revision`, the R range, LoRA settings, `symbols`/`token_ids`, `prompt_format: dataset_raw`, `loss_vocabulary: symbols`, `train_depths`, the `executor` options, `completion_head`, and `controller_training` provenance on exports.
- `training_state.pt`: optimizer, RNG, data cursor and resume identity. On controller exports it is provenance only and cannot be resumed.
- Tokenizer files.

| `format` | Mode |
| --- | --- |
| `loopformer-stage1-v1` | full_sequence LoRA (the only format `--init-from` accepts) |
| `loopformer-stage1-completion-v1` | full_sequence plus a `CompletionHead` |
| `loopformer-stage1-fixed-prompt-v1` | fixed_prompt LoRA |
| `loopformer-executor-v2` | fixed_prompt plus the `executor` block (required) |

`load_recurrent_checkpoint(path, device, dtype, download=False)`:

1. Validates the format against the mode and the symbol IDs against the saved tokenizer.
2. Loads the base with `local_files_only=not download` at the spec revision, using `attn_implementation="eager"`. Evaluation and demo callers then set `model.config._attn_implementation = "sdpa"`.
3. Rebuilds the model: LoRA if the scope is `lora`, then `configure_executor(**spec["executor"])`, then the completion head.
4. `restore_adapters` requires an exact name-set match, equal shapes and finite values.

Restored tensors are bitwise exact. CPU-loaded base weights may be memory-unaligned, which changes CPU matmul rounding by about 1 ulp; `tests/test_executor_upgrade.py` uses 1e-6 for this reason. GPU loading is not affected.

## Sandbox ([scripts/sandbox/](../scripts/sandbox/))

`run(files, entry, stdin, limits, mounts)` runs `python3 -I /code/<entry>` and returns exit code (128 + N for signal N), timeout and OOM flags, wall and CPU time, and truncated stdout (head) and stderr (tail). CPU time counts only processes that were reaped, so it undercounts after a timeout. The layers and what the program can see are in the [runner.py](../scripts/sandbox/runner.py) docstring. Runs share nothing: each gets fresh `/work` and `/tmp`, and tools pass the full program every call. Python's `/dev/shm` is bwrap's private one, bounded by the memory limit.

`grade(solution, tests, limits, workers)` in [grader.py](../scripts/sandbox/grader.py) first compiles the solution inside the sandbox, then runs each test case as its own sandbox run given only that case's input. `stdio` cases feed stdin and compare stdout by tokens, with numbers compared as decimals. `call` cases send JSON arguments to [call_harness.py](../scripts/sandbox/call_harness.py), which returns the solution's JSON value; it is compared structurally, with bools distinct from ints and floats within `atol`. The host makes the comparison; expected values never enter the sandbox. Each case allows output up to twice the expected size. The grade is `pass` or the first failing case's class, plus the pass fraction. Failure classes are parsed from program-controlled stderr, so they are diagnostics only.

## Known limitations visible in code

- Isolated models require the exact raw prompt regex and support only `logits_mode="answer"`. MMLU evaluation therefore sets `model.router = None` and `model.completion_head = None` and runs T=1 ([scripts/eval/mmlu.py](../scripts/eval/mmlu.py)).
- Stopping, `stop_policy` and `loop_callback` run at batch size 1 only. `loop_test` additionally requires FP32 for stopped runs.
- Readout indexing (`_answer_readout`) uses Python integer slices, which sync once per forward. This avoids MPS's nondeterministic scatter backward.
- Count digits are limited to 8 characters for `shared_number`. The affine countdown is an unbounded scalar, so precision beyond the trained range depends on the fitted gain and offset.
- `controller_kind` in `ControllerConfig` accepts only `gru` and `affine_suffix`. `shared_number` is produced only by `repair_number_reader`.
