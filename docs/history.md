# LoopFormer v1 history

v1 asked whether a looped Qwen2.5-0.5B-Instruct, with its middle layers reused
as one shared recurrent block R and one loop meaning one transition, could learn
a reusable in-context state-transition rule on pointer chasing over 26-letter
graphs. The plan then asked whether extra loops repair wrong states without
damaging correct ones (Gate 2 onward), and whether the rule transfers across
algorithm families. v1 ended with a count-free executor that runs 1,350 fresh
graphs through 256 transitions with 96.89% of trajectories wholly correct, and a
separately trained numeric timer that stops at exactly the requested loop for
every count 1–256. The repair/damage and transfer questions were never reached.
This file is the dated log of how it got there, including every negative result.
The current design is in [architecture](architecture.md), commands are in
[usage](usage.md), and the next plan is in [plan](plan.md).

## v1 in one page

### The final system, as built

- **Backbone.** Pinned `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`. The 24 layers are split
  P (layers 0–5, run once), R (layers 6–17, reused every loop) and C (layers 18–23,
  final norm and tied head, read after every loop). The readout is restricted
  to the 26 space-prefixed letter tokens. The next loop gets R's continuous
  state, never C's output or a decoded letter. `use_cache=False`.
- **Executor** (`models/stage1_pointer/executor_r-seed61/step-002250`). The prompt
  is routed so the executor sees Rules/Start/Answer with the Steps line removed,
  with compacted positions so digit count cannot leak in. Prompt positions are fixed
  layer-specific memory; only the answer position (the working vector) recurs.
  Before loops 2 onward, a learned re-entry bridge RMS-normalizes the working
  vector, rescales it, and applies a per-channel gain and a 896×896 projection.
  All R weights are trainable; P, C, embeddings and head are frozen. A direct
  896→26 R readout is supervised alongside C (coefficient 0.25). Training data:
  36,000 independent seed-61 graphs (generator v2, depth-independent, cycles
  allowed) at requested counts 1–6/8/10/12. Recipe: one epoch, 2,250 updates,
  batch 4 × accumulation 4, BF16/SDPA, LR 2e-5 with cosine decay and weight decay
  0.01. Per-loop CE at every loop through N, with full BPTT. 180,108,059
  trainable parameters.
- **Controller** (final checkpoint
  `models/stage1_pointer/controller-shared-number-precision-seed83/best`,
  adapter SHA-256 `0f8a3f2572ec2caf4fafbcdd074b5283bd8c610602502763b3c04487ce1f123d`).
  A shared linear projection of the frozen embeddings of the Steps tokens,
  accumulated left to right with a learned gain (it came out at 10), gives the
  initial memory m0. Each executor loop applies one scalar affine update
  `m = a·m + b` (fitted a = 1.0, b = −0.9999999403953552), and
  `sigmoid(w·m + c) ≥ 0.5` stops the run. Safety cap 272. The reader was fitted
  on 58 count labels (1–63 excluding 9/17/29/41/53) with at most 12 supervised
  loops. The cell was refitted with float64 L-BFGS.

### Headline results

| Result | Number | Artifact |
| --- | --- | --- |
| Existing seed-61 test, native stopping (128 graphs × counts 1–12) | 100%, 1,536/1,536 joint | `eval/pointer_benchmark/final-full-20261008/existing_test/` |
| Fresh panel (seeds 307/311/313), answer plus exact stop, mean over counts 1–256 | 97.81%, 338,026/345,600 (graph-cluster 95% CI 97.04–98.53%) | `eval/pointer_benchmark/final-full-20261008/independent/` |
| Same panel, strict trajectory through 256 | 96.89%, 1,308/1,350 graphs | same |
| Same panel, exact stopping | 100% at every count 1–256; zero early/late/missing | same |
| Controller alone, numeric requests (no R executed) | 8,192/8,192 exact; reader error ≤0.001953125 through 10,000 | `eval/pointer_benchmark/shared-number-precision-20261008/` |
| Same executor with GRU controller (population) | 4.57% mean exact stop | `eval/pointer_analysis/paper-20261009/` |
| Ordinary Qwen, 3-shot final letter, same 345,600 questions | 6.25% (21,614/345,600) | `eval/pointer_analysis/paper-20261009/ordinary_qwen/` |
| Original Qwen, 3-shot trace prompt, strict | 0.013% | `eval/pointer_traces/seed61-20261010/original/` |
| Full-model trace SFT, strict (trained / held-out 7,9,11 / 13–256) | 3.52% overall (100% / 0.15% / 0%) | `eval/pointer_traces/seed61-20261010/sft/` |
| MMLU 5-shot, original vs final checkpoint forced to one loop | 46.92% vs 24.90% (−22.02 pp, CI −23.07 to −20.96) | `eval/general/mmlu-retention-20261012/` |

### Limitations (stated once here)

1. **The stopping controller is a counter, not adaptive compute.** It reads
   the Steps count from the prompt with a fitted linear number reader and counts
   down with `m = a·m + b`. It never looks at R's state, so it cannot detect an
   execution error or spend more loops on a hard instance.
2. **One executor seed.** Only one successful executor training seed exists (61).
   Controller "seeds" 83/89/97 share data and initializer and differ only in
   minibatch order, so they test optimization robustness, not independent replication.
3. **26-state graphs cycle.** Every run longer than 26 steps revisits states.
   All 42 first errors on the final panel occur at loops 2–30, so "depth 256"
   is mostly evidence of stability on a cycle, not 256 distinct transitions.
   Long non-repeating chains were never tested.
4. **Stopping confirmation is not independent.** Each controller fix (affine
   cell, shared reader, precision refit) was designed after seeing a failing
   benchmark panel, and the count integers 1–256 were reused across those panels.
   Held-out counts 9/17/29/41/53/64 appeared in acceptance panels. Fresh graphs
   confirm the executor, not the timer on unseen integers.
5. **Seed 29 is stale.** The reserved seed-29 split was never generated or
   opened, but it is defined by the older v1 generator (non-repeating paths,
   depths ≤16), not the cyclic generator used for the final model.
6. **MMLU drop is confounded.** At one loop MMLU fell from 46.92% to 24.90%.
   But the pointer readout was trained on letter tokens that include exactly
   " A"–" D" (ids 362/425/356/422), and both prompt formats end in `Answer:`.
   The recurrent model picks A 6,400 times out of 14,042. So the drop cannot be
   separated from the pointer readout taking over the answer slot.
7. **The trace-SFT baseline fails only on stopping, and it is not exposure-matched.**
   Full SFT scores 100% on trained counts, 0.15% on held-out 7/9/11 and 0% above 12.
   Every failure is a stop at the wrong length (it snaps to a trained length or
   keys on individual digits). The controller saw counts 1–63; SFT saw nine
   counts. See former report `ordinary_trace_baselines.md`, section "Completed
   benchmark results" (git 8d4cc19).
8. **The core question was never run.** Gate 2 (overscaling repair and damage),
   the plan's central question, was never run on the pretrained model.
   The absorbing-terminal tools exist but were never executed. The prepared one-field
   ablations ([no bridge](../configs/executor_no_bridge.json),
   [no direct loss](../configs/executor_no_direct.json),
   [LoRA-only](../configs/executor_lora.json),
   [full model](../configs/executor_full.json)) were never run either, so no single
   executor change can be given credit for the improvement.

## Chronological log

Commit hashes are the ones that introduced the work or recorded its results.
"Former report X" means a file deleted from `docs/experiments/` in the
2026-10-29 restructuring. It can be read at git 8d4cc19.

**2026-09-09 – init** (`015ddef`).

**2026-09-10 – Stage 0: recurrent wrapper and Gate 0** (`8af78e3`, `a6de686`).
P/R/C split 0–5/6–17/18–23, no bridge, rank-8 q/v LoRA on R (alpha 16, 270,336
trainable parameters, B zero-initialized). Gate 0 passes on CPU and MPS: T=1 logits
match exactly (max error 0) before and after LoRA; shared weights, adapter-only
gradients and loop counts 1/2/4 also verified; 36 tests pass. The MPS backward
pass failed on deterministic scatter ops, so the answer readout became integer
slicing plus stacking.
Decisions D01/D02: Transformers 5.17.0 semantics, FP32/eager, tolerance 1e-5, no bridge.

**2026-09-10 – Pointer data** (`c3cb249`). Seed 17: 10,000 train, plus 1,000 each
for validation, test and depth_test. Depths 1–8, with depth_test at 9–16. Prompt
format `( A, C)`, 26 rules, raw text, readout at the final `Answer:` colon.
Decisions D03–D05: targets exactly at steps 1..d; nominal paths of d+1 distinct
states (so depth ≤25); no post-completion labels. 59 tests pass.
Later moved to `scripts/dataset/`, with byte-identical regeneration.

**2026-09-10 – Ordinary-Qwen baseline.** Three-shot chat prompt (revised after
the user saw three failures), greedy decoding, 8 tokens: **60/1,000 (6.00%)** on
the seed-17 test, 22/125 at depth 1 and 4.34% at depths 2–8. Z and T make up 41.3%
of responses. Reproduced byte-identically on the new WSL2/RTX 5070 Ti desktop
(`8ff477a`). There, one exact-resume test differs by 1.49e-8. This is a known
CUDA resume limitation.
Artifacts: `eval/pointer_task/20260910T172432.074426Z-Qwen-Qwen2.5-0.5B-Instruct/`,
`eval/pointer_task/20260910-wsl-cuda-instruct-retry/`.

**2026-09-10 – First training** (`8e80f76`, D06). Loss is per-loop 26-symbol CE,
averaged over loops within an example and then over examples, with BPTT. The
overfit run (32 examples, 160 updates) fits the training probes (84.375% trajectories)
but validation is weak (25%) (`94d1887`). The 5k depth-1–4 run (625 updates) gets
complete trajectories at trained depths of 86.2% (update 500) and 92.4% (update 625).
Depth 5 is 11 and 50 out of 125. Depths 7–8 are 0. Most first failures repeat the
previous symbol. Full-loop evaluator added (`7521554`). Continuation to three epochs,
update 1875 (`04179d3`, `5a3b954`): 99.4% at depths 1–4, 82.4% at depth 5, 20.8%
at depth 6, 1.6% at 7, 0% at 8. Seed 29 reserved for confirmation. The
absorbing-terminal overscaling diagnostic was prepared but deferred by the user,
who prioritized depth generalization first.

**2026-09-10/11 – Depth-6 curriculum** (`cc83e66`, `c4c3d21`, `cfd6b29`). Initialized
from update 1875 with `--init-from`: 7,500 depth-1–6 mappings, 938 updates. Step
500 scores 99.1% at depths 1–6, 79.2% at 7–8, 33.6% at 9, 5.6% at 10 and 0% at 11–16.
These artifacts were pruned 2026-09-17 and are recoverable at `1edacda`.

**2026-09-11 – Fresh 30k depth-6 baseline** (`715d0eb`, `af8420b`, `b4416cd`, `1edacda`).
The T=1 startup gate failed on one logit (about 1.3e-5). The fix projects the
reference LM head only at the answer position, with tolerance unchanged.
Batch changes on resume were allowed only when the effective batch is equal.
Run: seed-37, 30,000 mappings, depths 1–6, 3,750 updates, batch 4 × accumulation 2,
FP32. Took 60m54s and 7.01 GiB. Complete trajectories, steps 2500/3250/3750:
depths 1–6 98.0/99.2/99.33%, depth 9 73.6/56.0/60.8%, depth 10 41.6/15.2/27.2%,
depth 12 1.6/0/0%, depths 13–16 0%. Step 2500 became the working extrapolation
checkpoint; 3250 is the loss-selected one. More training improved trained depths
but hurt extension. `models/stage1_pointer/depth6-fresh30k-seed37-batch4/`,
`eval/pointer_loops/20260911T2*`.

**2026-09-17 – Loop-balanced CE; pruning** (`422e5a0`). Approved: equal total
weight per loop (N/(D·N_t)) in place of equal weight per example, everything else
fixed. Superseded pre-30k artifacts were pruned (about 29.6 MiB). Old tracked
evidence is at `1edacda`.

**2026-09-29 – Loop-balanced result: negative** (`7c53915`, `3040342`). No
checkpoint beat baseline step 2500 at far depths. Equal-loop 2500 drops to 66.4%
at depth 8 and 32.0% at depth 9 (baseline 94.4/73.6%). Artifacts deleted at the
user's request (24.08 MiB). The same day: reframing around adaptive compute and
docs consolidation (`3d4ab9d`, `9294fde`, `2ea98c6`). Oracle/heuristic stopping
infrastructure was built but gated. The offline check on step-2500 traces is
observational only (`eval/pointer_adaptive_analysis/`). Diagnostics and profiling
added (`8408e17`, `c33e299`): eval at batch 16 takes 9.083 s for 128×16 loops, and
batch 8 gave no training speedup over batch 4. Paired probe on 32 deep examples:
6/32 complete; 30/32 when restarted from the reference state after loop 6; 9/32
with depth minus one. Late failures show the target margin falling from a median
10.31 to −2.51 at first error (`eval/pointer_diagnostics/offline-review/`).
Decision: diagnose before another training run.

**2026-09-30 – Controlled restarts and rule edits** (`9e63553`, `3b8d2ba`, `04d6665`,
`75ec1af`). Showing the original Steps in the suffix restart changes nothing
(30/32), so the restart benefit is not the displayed count. Re-encoding only the
Rules prefix after loop 6 is harmful: deep complete 6/32→0/32, and loop 7 drops
from 28/32 to 5/32 (`eval/pointer_probes/controls-small-20260930T112605Z-473/`).
Decision (`7fafb78`): externally decoding the state and re-prompting is rejected,
because the controller would do the state transfer. Single-edge rule edits:
early loop 6 follows 21/21; at late first errors only 6/19 actually change to the
new target (`eval/pointer_rule_edits/`).

**2026-10-03 – Integrated frozen-mechanism suite** (`9ec77bb`, `9d22c86`, `10a70a6`).
Steps 2500/3250/3750 on matched cohorts. Disabling LoRA collapses deep readouts
from 529/800 to 21/800. Early relevant-edge edits are followed 106/106, and all 53
two-target cases follow both. Late first errors follow 17/70, while irrelevant edits
change 19/70. Steps 17 versus native barely moves late accuracy. Linear probes
find the correct target in R at 1 of 38 late errors. Conclusion: local transitions
are reliable but degrade as recurrent history grows; the locus is unresolved
(`eval/pointer_mechanism/matched-2500-3250-3750-20261004T001803Z-552/`).

**2026-10-03/04 – Joint learned completion: negative** (`6112b70`, `e76dea7`, `03d4995`).
Adds a 128-wide MLP stop head on R's answer state, balanced BCE weight 0.1,
coupled gradients. At step 3250, forced trajectories match CE-only at depths 1–6
(98.93% vs 99.2%) but collapse beyond: depth 8 27.2% vs 86.4%, depth 9 2.4% vs 56.0%,
depths 9–16 0.3% vs 9.2%. The head stops early on all 1,000 depth-9–16 examples
(at loop 3/4/5). So stopping early is not the whole failure.
`models/stage1_pointer/depth6-completion-seed37/`, `eval/pointer_loops/completion-best-20261004T040328Z-1574/`.

**2026-10-04 – Fixed prompt memory** (`e58c3fb`, `da2f707`). Only the answer position
recurs; prompt positions are fixed per-layer memory. Same data/recipe and
joint stop head (387,073 trainable). Step 3250 complete trajectories: depth 8
96.0%, depth 9 72.8%, depth 10 3.2%, depths 9–16 pooled 9.5%. Better than joint
completion, below CE-only 2500 (16.4%). Stopping is exact on all 750 trained-depth
examples, but every depth-9–16 example stops at loop 5 or 6, even though 985/1,000
are still correct there. No single threshold fixes it (max 106/1,000 deep). Steps
10 tokenizes as two digits, so it is out of distribution
(`eval/pointer_diagnostics/fixed-prompt-review/`). Paired-Steps probe: same maps
with Steps 9→16 lose 10 correct loop-9 predictions and gain none, so the requested
count causally affects execution (`eval/pointer_probes/paired-steps-20261004T150542Z-407/`).

**2026-10-04 – Depth-12 sparse counts** (`9b3154d`, `e505903`, `2a9296c`, `8602241`).
Approved after the fixed-prompt review. Seed-47 data, counts 1–6/8/10/12, with 7/9/11
held out. BF16/SDPA, batch 4 (smoke peak 16.42 GiB), 7,500 updates. W&B logging
added (`fca1dfa`). Selected step 5,000, cap 24 (`eval/pointer_loops/depth12-fixed-prompt-seed47-gaps-best-20261004T184317Z-1034/cohorts.json`):
joint success 96.98% on trained counts and 91.73% on held-out 7/9/11, but 0% at
13–20 (all stop early). Progression (`2b6247f`): complete deep trajectories 238/1,000
at step 5,000 versus 63 at 7,500. Late training regresses at a constant LR.
Paired counts (`572c71c`, `3799729`): identical maps give 30/32 correct loop-14
readouts with Steps 18 but 8/32 with Steps 20
(`eval/pointer_probes/depth12-paired-20261004T194923Z-5303/`). This motivated
removing the count from the executor's view.

**2026-10-04 – Pipeline reassessment and executor upgrade** (`d2eae3a`). Decision:
a stateless head did not isolate control, because its features and BCE were
coupled to R. Authorized changes: route the count only to an isolated recurrent
controller (GRU) with no gradient path into R; add the normalized re-entry bridge
and direct R supervision; train all of R (full-model SFT as a separate control);
switch to a versioned depth-independent cyclic dataset. Considered and rejected:
a LoopCD-style target-subtraction loss, numeric progress features, and oracle
re-entry.

**2026-10-04 – Executor seed 61** (`86619b0`). `configs/executor_r.json`, 127.9 min,
4.43 GiB. Validation (128 graphs × counts 1–12) has 99.09% complete trajectories.
Deep (32 graphs × counts 13–64) has 1,664/1,664; 125/128 validation graphs are
correct across all 64 loops. All 64,064 nominal deep predictions match an
independent reference execution. The jointly trained GRU stops correctly
on only 10.74% of validation and 0% of deep queries (stops at loop 3 or 4)
(`eval/pointer_diagnostics/executor_r-seed61-20261004T231253Z-799/`).

**2026-10-04 – Controller diagnostic** (`9282d01`, `bdd4c7c`). Count is linearly
decodable from the prompt feature (99.48%) and from initial memory (95.57%), but
falls to chance (6.25%) by loop 12. Tiny controller-only fits reach 72/72 but
generalize weakly. Decision: train the controller alone, with R frozen.

**2026-10-04/05 – Controller-only GRU training** (`4877638`, `5b1cad2`). 2,048 graphs,
counts 1–6/8/10/12, 3,000 updates; selected step 2,700. Exact stops: 95.31% on
trained counts, 45.83% on held-out 7/9/11 (count 9: 0/128), 1.68% on deep. Only
count 13 succeeds; 14–64 all stop at loops 10–15
(`eval/pointer_diagnostics/controller-eval-20261005T024736Z-791/`).

**2026-10-05 – Remaining-work supervision: negative** (`44c6417`, `400b1a3`). Adds an
N−t regression on controller memory, 3 paired seeds 83/89/97. Exact stopping is
unchanged: trained 95.54→95.31%, held-out 46.61→46.70%, deep 1.703→1.723%. The
count-9 initial estimate averages 7.59–7.73
(`eval/pointer_diagnostics/controller-remaining-seed61/`). The no-update audit
(`d79af01`, ran 2026-10-08, `adf40e4`) found a train/validation gap of only about
1–2.2 pp and remaining-work gradients at 9–24% of the stop gradient; no count-9
initial estimate came within half a step.

**2026-10-08 – Controller repair** (`475f652`, `41a4cb7`, `a3c3042`, `11a7458`).
User authorized "work until repaired". Weight-12 initialization loss: count 9
still 0%, deep 1.80%. Negative. GRU with broader counts (58 values, ≤12-loop
prefixes, 6,000 updates): 60.65% on short seen counts and 5.63% on unseen deep
counts. Negative. A ridge probe of the P summary has about 1.5-step error, while
least squares on the last eight token embeddings recovers counts 1–64 to 7.82e-14.
Decision: switch to a learned suffix reader plus scalar affine cell, an explicit
counting bias. Result (`ff319ee`, `1044aee`, `a36d569`): seeds 83/89/97 each get
2,048/2,048 validation and 1,664/1,664 deep exact stops, with 132/132 native
checks. Validation joint is 98.73% because of the executor's 26 failures
(`eval/pointer_diagnostics/controller-affine-repeats/`).

**2026-10-08 – Frozen benchmark: controller fails** (`e71f516`, `d44f602`, `2171b8f`).
Protocol committed before results. The seed-61 test passes 1,536/1,536. On fresh
seeds 211/223/227 (1,350 graphs × counts 1–256) the executor has 96.96% complete
trajectories through 256 (1,309/1,350). The positional reader is exact at 1–69 and
stops early at every count ≥70: count 70 reads as 49.26, because digit 7 never
appeared in that suffix slot. Overall joint success 26.46%. Gate failed;
result kept (`eval/pointer_benchmark/frozen-20261008/`).

**2026-10-08 – Shared number reader and precision refit** (`03cbc13`, `f79c762`,
`704bfb8`, `e08cab2`, `b7bb062`). Same 58 labels; one shared projection plus a
learned gain (10). On fresh seeds 281/283/293: 98.02% joint and 97.33% through 256,
all criteria pass. A longer numeric diagnostic exposed timer drift: the first
early stop is at 1,038, and only 1,037/4,096 are exact. Float64 L-BFGS refit of
the two cell parameters only gives 8,192/8,192
(`eval/pointer_benchmark/shared-number-20261008/`, `…-precision-20261008/`).

**2026-10-08 – Final full benchmark** (`f8a14cb`, `e7bebe5`, `f36abe0`). Fresh seeds
307/311/313 with protocol frozen beforehand. Results in the headline table: 97.81%
joint, 96.89% strict through 256, 100% exact stop. 42 graphs fail first at loops
2–30, with none after. 189/189 native calls match replay. 345,600 decisions and
27,309 native transitions independently audited; zero overlap with 38,988 excluded
tables. Paired checkpoint comparison on 27 graphs × 30 counts (`bcbeb73`, `6cd1c68`,
`f80d800`): exact stop is 35.93% for depth-12 LoRA, 23.46% for GRU, 73.33% for
positional affine and 100% for shared reader; all four isolated arms share 148
identical non-controller tensors. Failure matrices: 37/42 first errors have direct
R and C agreeing on the wrong letter, 13 are one transition ahead, and 91.62% of
transitions after an error follow the edge from the wrong letter. All figures
re-audited (`0c9cac4`).

**2026-10-09 – Architecture history and population suite** (`a3c0462`, `b49030e`,
`47c48c8`, `38826ed`, `4d12b22`, `5a7976d`). Six representatives on the 27-graph
panel. At N=9 complete trajectories: CE-only 13/27, joint 0/27, fixed-6 16/27
(`eval/pointer_benchmark/architecture-history-20261009/`). Evidence inventory at
`b49030e` (`eval/pointer_analysis/evidence-20261009/`). Full population: six models
× 1,350 graphs × counts 1–256, 2,073,600 audited decisions. TF32 and merged LoRA
were rejected because they changed intermediate letters; ordered-index pruning
was kept, bitwise-validated. The ordinary-Qwen full-panel baseline was added.

**2026-10-10 – Population review** (`f07e41c`, `5eb1b2d`). Mean forced final
letter: CE 7.32%, joint 6.30%, fixed-6 7.18%, fixed-12 9.08%, isolated R 97.81%.
Strict trajectory at N=12: 1.11/0.37/2.15/93.33%; fixed-12 drops to 0.07% at 32.
Executor snapshots: learning jumps between updates 750 and 1250. Step 2000 (1,315
strict) beats the shipped 2250 (1,308), losing 29 graphs and gaining 22. The
predeclared 2250 was kept. 25 figure families
(`eval/pointer_analysis/paper-20261009/plots/`).

**2026-10-10/11 – Trace baselines** (`8e96ab3`, `e24f19b`). Full SFT of all
494,032,768 parameters on the executor's 36,000 examples, 2,250 updates, batch 16
(`models/trace_sft/seed61-full/best`, final validation CE 1.981e-6). Strict success:
original trace prompting 0.013%; SFT 100% on trained counts, 0.15% on held-out
7/9/11, 0% on 13–256, 3.52% overall. Paired recurrent-minus-SFT: −0.95 pp on trained
counts, +98.1 pp on held-out. Every SFT failure is a wrong length (e.g. 40→4
states, 256→2 or 6). See limitation 7.

**2026-10-11 – Public demo** (committed in `f343561`). HF model `mukunds/loopformer`
(model commit `e29f88c`). ZeroGPU Space `mukunds/loopformer-demo`, pinned to
torch 2.13 because HF caps it; research stays on 2.14. GitHub Pages playground
(≤26 nodes, ≤256 steps). Hosted fidelity check: 280/280 loops and every stop
probability match. Later an owner-token proxy was added, after anonymous calls hit
HF's run limit. Model card: [demo/hf_model_card.md](../demo/hf_model_card.md).

**2026-10-12 – Language probe and MMLU** (committed in `f343561`). "What color is
the sky?" at 1 loop gives "The sky is blue."; at 2 or 4 forced loops the output
degenerates (`artifacts/language_probe/20261012/`). Full MMLU (57 subjects, 14,042
questions) at one loop: 46.92% → 24.90%, agreement 27.06%. The compact scorer
matches HFLM to 1.68e-5; an unadapted control reproduces base logits exactly.
See limitation 6.

**2026-10-29 – Test hardening and restructuring** (`8d4cc19`). Added a T=1 gate
for the executor view, trainable-tensor round-trip tests, and trace cohort deltas
(responses unchanged). Then documentation collapsed into this file plus
[architecture](architecture.md), [usage](usage.md), [setup](setup.md) and
[plan](plan.md). Root launchers moved to `scripts/launch/`.

## Artifacts and where things live

| What | Path |
| --- | --- |
| Final checkpoint (executor + shared reader/precision timer) | `models/stage1_pointer/controller-shared-number-precision-seed83/best` |
| Successful executor alone | `models/stage1_pointer/executor_r-seed61/step-002250` |
| Earlier controllers (GRU, affine, positional/shared reader) | `models/stage1_pointer/controller-*` |
| Early architectures (CE, joint, fixed-6, fixed-12) | `models/stage1_pointer/depth6-*`, `depth12-fixed-prompt-seed47-gaps/` |
| Trace SFT baseline | `models/trace_sft/seed61-full/` |
| Final benchmark (headline numbers, figures, failure matrices) | `eval/pointer_benchmark/final-full-20261008/` |
| Failed positional-reader benchmark; reader and precision repairs | `eval/pointer_benchmark/frozen-20261008/`, `shared-number-20261008/`, `shared-number-precision-20261008/` |
| Architecture comparisons (27-graph panel) | `eval/pointer_benchmark/checkpoint-comparison-20261008/`, `architecture-history-20261009/` |
| Population suite, ordinary-Qwen baseline, 25 figure families | `eval/pointer_analysis/paper-20261009/` |
| Trace baselines | `eval/pointer_traces/seed61-20261010/` |
| MMLU retention | `eval/general/mmlu-retention-20261012/` |
| Controller diagnostics and repairs | `eval/pointer_diagnostics/` |
| Early-architecture evaluations and probes | `eval/pointer_loops/`, `eval/pointer_probes/`, `eval/pointer_mechanism/`, `eval/pointer_rule_edits/` |
| Configs for every run | [configs/](../configs/) |

Weights and feature caches live only on the desktop, not in Git. Launchers moved
from the repository root to [scripts/launch/](../scripts/launch/) on 2026-10-29,
so old reports that cite `bash <name>.sh` refer to `scripts/launch/<name>.sh`.
Intermediate artifacts from before the 30k run were deleted on 2026-09-17; they
are recoverable at commit `1edacda`. The loop-balanced run artifacts were deleted
on 2026-09-29. The old per-topic guides and dated experiment reports were removed
in the 2026-10-29 restructuring. All of them remain in git history at `8d4cc19`
(for example `git show 8d4cc19:docs/experiments/pointer_final_benchmark.md`).
