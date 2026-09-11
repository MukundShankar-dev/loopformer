# Absorbing-terminal overscaling evaluation

Status: implemented as a deferred experiment; no pretrained terminal sweep has been run. The user's current priority is further Stage 1 training and depth generalization. Preparing these scripts does not establish Gate 1 or authorize launching Stage 2 now.

## Task and interpretation

`scripts/dataset/terminal.py` deterministically transforms each validated source task by replacing the final state's outgoing edge with a self-loop. Rule order, initial state, requested depth, seed, source ID, and nominal targets are retained. The prompt and mapping hash are recomputed and validated. Source files and checkpoints are never changed.

For `A -> C -> E`, the terminal version contains `E -> E`. Reference execution is `C, E, E, E, ...`. Because the nominal path has no repeated states, replacing that outgoing edge cannot alter any target through depth d. The source dataset's seeds and file hash plus `absorbing-terminal-v1` fully determine the transformed tasks; there is no additional sampling RNG.

This is a **different input distribution** from training. The terminal self-loop can also provide an answer shortcut: early final matches must be inspected alongside complete nominal trajectories. A nominal-performance drop must be diagnosed before drawing stability conclusions. High terminal-task accuracy alone does not demonstrate stepwise execution or transfer. Cyclic tasks in general remain outside this experiment.

For t >= d, the fixed final target is also the valid continuing reference state. Wrong→right is post-nominal recovery; right→wrong is damage. For t < d, movement toward the final answer is ordinary execution, so those transition counts are explicitly observational. No retention training, detached rollouts, anchor loss, hidden-state probes, or model architecture changes are introduced.

## Preview now; inference later

From the repository root with the environment activated:

```bash
python -m scripts.eval.overscaling_test --dry-run
python -m scripts.eval.overscaling_test --dry-run 10 --loops 16
```

These show the first five or ten transformed tasks, changed terminal rules, and exact reference targets. No checkpoint is required, no model weights are loaded, and nothing is written. Unlike the main dataset generator's depth-spanning preview, this preview selects the input file's prefix and is not a distribution estimate.

When Stage 1 is ready for review, use a complete saved checkpoint directory on the desktop. The following are **unrun future commands**, not the current training instructions:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m scripts.eval.overscaling_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 \
  --device cuda --loops 16 --test

python -m scripts.eval.overscaling_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 \
  --device cuda --loops 32
```

Update 500 is the historical validation-selected checkpoint, not necessarily the checkpoint for the future experiment. Select that future checkpoint using validation before test sweeps. Use 625 only as the already identified secondary historical comparison. Do not choose the best overscaling result after looking at test scores.

The default is all 1,000 source test examples and 32 loops. `--loops` must exceed every selected task depth. `--test [2|3]` performs real inference and prints per-loop predictions and correct targets, including terminal targets after d. `--limit N` runs a prefix; these subset flags and `--dry-run` are mutually exclusive. `--data` changes the source file. `--output` must be a new directory. CPU/float32, batch 1, seed 17, and strict determinism match `loop_test`; explicitly select CUDA/MPS. Downloads are opt-in with `--download`.

Each sweep observes every loop up to its budget; a 32-loop run supplies the 1/2/3/4/6/8/16/32 prefixes. A 64-loop run is supported but memory and timing at these budgets are unmeasured on pretrained checkpoints. Start small when these experiments are eventually launched.

## Artifacts

Outputs go under `eval/pointer_overscaling/<timestamp>-<run>-<step>/`:

| File | Purpose |
| --- | --- |
| `tasks.jsonl` | Exact transformed prompts, mappings, source IDs/seeds, and nominal targets |
| `trajectories.csv` | Every prediction and final margin; nominal CE/intermediate targets only through d |
| `examples.csv`, `depth_by_loop.csv` | Shared full-loop execution diagnostics and final-accuracy matrix |
| `transitions.csv` | Every adjacent prediction pair, target, margins, correctness, transition type, and post-nominal flag |
| `transition_rates.csv` | Per-depth and aggregate counts, conditional denominators, repair, damage, net gain, and adjacent accuracies |
| `solutions.csv` | First correct loop at/after d, early-final-match flag, nominal and last correctness |
| `survival.csv` | Continuous solution survival with observed, censored, and never-solved counts |
| `summary.json` | Run completion, settings, source/checkpoint/data hashes, transform version, transformed-task hash, timing, nominal metrics, and dynamics summaries |

Only `status: complete` runs have finished all exports. Run time covers inference and trajectory export; derived dynamics/export time is excluded from the inference throughput, as are loading and encoding. No hidden states or full logits are saved. CSV exports support downstream figures; plotting is not implemented in this change.

## Metric conventions

`transition_rates.csv` separates `all_loops_observational` from `post_nominal`. For the latter, include an example in transition t→t+1 only when t >= its requested depth. This aggregate cohort can grow with t. Both adjacent accuracies on any row use that row's same cohort, so `net_gain = accuracy_at_next - accuracy_at_t`. Compare depth-specific rows to avoid mixing depths. Empty conditional denominators are blank in CSV, never reported as zero measured repair/damage.

Survival starts at the first final-correct readout at or after d. An early final match does not start survival. At offset k, include only examples observed through first-correct+k. Success requires all readouts in that interval to remain correct; later recovery does not restore continuous survival. Never-solved examples and censored tails are explicit. This is an observed-cohort proportion, not a censoring-adjusted population estimate.

The summary includes a **hold-the-nominal-prediction baseline**: stop changing the decoded answer at d. Its later accuracy equals nominal final accuracy and it cannot repair wrong answers. This is a readout-halting comparison, not evidence from an implemented hidden-state no-op ablation. Confidence/stability halting and hidden-state diagnostics remain future work.

Nominal CE is unchanged and excludes t>d. Final-target margins remain available after completion. A positive raw-logit margin is stricter than the A–Z argmax convention in a tie. These metrics do not automatically declare a research gate passed.

## Validation

Focused tests cover terminal execution through 64 steps, original-record preservation, deterministic transformation, conditional counts, the net-gain identity, censoring, recovery after damage, malformed trajectories, a no-write preview, and offline tiny-Qwen checkpoint inference. The integration case checks nominal agreement with `naive_test` on the transformed tasks and inference without optimizer state. The Mac full suite passed 92 tests in 54.83 seconds; the five focused cases passed in 19.82 seconds. The continuation data/tokenizer preview and CLI help checks also passed. See [current status](status.md). No pretrained inference, training, or performance claims are made by these tests.
