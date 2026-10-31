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

**2026-10-30 – v2 sandbox** (`58d0dff`). [scripts/sandbox/](../scripts/sandbox/):
systemd user scope (memory without swap, tasks, CPU quota) → `prlimit` → bwrap
(all namespaces, no nested user namespaces, seccomp denylist and socket-family
allowlist, everything read-only except capped tmpfs `/work`, `/tmp`, `/dev/shm`).
`python -m pytest -q tests/test_sandbox.py` on the desktop: 30 passed in 12.1 s.
Removing any one of 19 protections in a scratch copy made the matching test fail.
Found and fixed while testing: AF_VSOCK reached a service on the Windows host
(CID 2, port 50000) from inside the network namespace, so seccomp now allows
only Unix and IP sockets; bwrap's root and `/dev` were writable and uncapped;
stdin was a host file the program could reopen for writing (now a pipe); WSL's
Windows driver store (9p) was visible read-only under `/usr/lib/wsl` (now
covered). Not covered: a kernel exploit through an allowed syscall, since bwrap
shares the WSL kernel.

**2026-10-30 – v2 grader** (`7b6e1b6`). [grader.py](../scripts/sandbox/grader.py):
one sandbox run per test case, comparison on the host, expected values never
in the sandbox. `python -m pytest -q tests/test_grader.py tests/test_sandbox.py`
on the desktop: 44 passed in 23.5 s. The cheats tested all fail: claiming
success, exiting early, always-equal objects, patching the harness's encoder,
writing to the result channel, crash-inducing JSON, and searching memory,
files and `/proc` for the expected value (a control with the value given as
input finds it). Seven naive-grader mutations are each caught; grading with
Python `==` was missed at first, so cases where `==` gives a different verdict
were added. (Corrected the same day: the bool-vs-int cases among them were
removed when the agreement check below showed EvalPlus counts `1 == True` as
a match; the float-tolerance case still catches the mutation.) Throughput on 60 trivial stdio cases (one warmup grade excluded, wall time of `grade`): 65.7 ms per case with 1 worker, 18.4 ms with 4.

**2026-10-30 – MBPP+ baseline, pinned Qwen2.5-Coder-0.5B-Instruct** (`848673a`).
`scripts/launch/mbpp_baseline.sh`: official EvalPlus 0.3.1 codegen with vLLM
0.32 (GPU share 0.3), then official evaluation inside the sandbox. The first
sampled evaluation was OOM-killed at 8 GB with 4 workers, because EvalPlus
lets each sample use 4 GB. It was rerun with 2 workers and 10 GB (the setting
now in the launcher); greedy used 4 workers. Greedy MBPP 51.3%, MBPP+ 43.4%
(378 tasks), against 52.4 / 43.7 in the Qwen2.5-Coder report (Table 16). n=16
at T=0.8 on MBPP+: pass@1 41.1%, pass@16 68.0%; always / sometimes / never
solved 15.3 / 52.6 / 32.0%. That clears the M0a thresholds on MBPP+ only;
HumanEval+ and LiveCodeBench are not run. An earlier EvalPlus HF-backend
attempt was abandoned at about 3 problems per minute (CPU-bound).
`eval/codeprobe/mbpp-coder-0.5b-20261030/`.

**2026-10-30 – MBPP+ in our grader** (`848673a`). `scripts.codetasks.mbppplus`
converts 365 of 378 tasks (39,539 hidden cases). Excluded: 12 special-oracle
tasks and Mbpp/255, whose reference was OOM-killed at 2 GB
(`data/agent/mbppplus-v0.2.0.excluded.json`). Values cross the sandbox as
tagged JSON, so tuples, sets and complex numbers survive. All 365 canonical
solutions pass every hidden case (`eval/codeprobe/mbppplus-canonical-check-20261030/`).
Our grader against EvalPlus on the 365 greedy samples: 365/365 verdicts agree (160 pass, 205 fail). The first
pass gave 364/365; the one disagreement (Mbpp/167, a sample returning `1`
for expected `True`) showed EvalPlus uses Python `==`. The grader now does
too: numbers compare by value, and tagged tuples still differ from lists.
`eval/codeprobe/mbpp-coder-0.5b-20261030/greedy/agreement/`.

**2026-10-30 – Agent tool loop** (`848673a`). [scripts/agent/](../scripts/agent/):
xgrammar-constrained turns (thought line, then `run_python`, `run_public_tests`
or `submit`; only `submit` on the last turn), one KV cache per episode, FLOPs
counted per processed token, and observations tokenized with special tokens
split. A 5-problem smoke run found two format problems, both fixed. A 300-char
thought cap cut sentences, after which the model wrote prose into code. The
grammar banned the fence-plus-newline token, so the model emitted mangled
fences and ran to the token cap. Greedy on the first 20 problems, fp32, before
the grader's `==` fix: single-turn 7/20 solved, 0.26 TFLOPs per episode;
interaction (4 turns) 7/20 solved, 0.63 TFLOPs, 13 submitted, 6 forced, 1
truncated (a repetition loop). Twenty problems cannot separate the conditions.
`eval/agent/smoke-mbppplus20-20261030/`.

**2026-10-30 – LiveCodeBench split: dev before 2024-11-12, confirmation after**
(`7320c7c`). Confirmation = release_v6 (the newest release) problems on or
after 2024-11-12, Coder-0.5B's public release and after the other two models':
266 problems (67 easy / 83 medium / 116 hard; 163 AtCoder, 103 LeetCode;
2024-11-16 to 2025-04-06), IDs frozen with sha256 `7522600f…abf2`. Dev = the 789
problems before the cutoff. Their private tests are pickled; `livecodebench_data`
decodes them with an unpickler that refuses every class, and only for dev
problems, so confirmation tests have never been decoded. LiveCodeBench's own
loader (remote code plus plain unpickling) is never run.
`python -m scripts.codetasks.confirmation --raw ~/src/lcb-data --dataset-revision 0fe84c3 --cutoff 2024-11-12
--output configs/confirmation_livecodebench.json`.

**2026-10-30 – M0a probe complete; the base model is Coder-1.5B** (`7320c7c`). Official EvalPlus 0.3.1 and LiveCodeBench (28fef95) generation with
vLLM, evaluation inside the sandbox, three models.
`.venv-probe/bin/python -m scripts.codeprobe.commands --run`, then `--run --resume` after LiveCodeBench
problem 3411's greedy sample was OOM-killed in its 8 GB sandbox and stopped the
first attempt (evaluation now retries such problems one sample per run and
fails only that sample). Greedy, ours vs published (Qwen2.5-Coder report, Table 16):

| Model | HumanEval / + | MBPP / + | LiveCodeBench 2407–2409 |
| --- | --- | --- | --- |
| Coder-0.5B | 60.4 / 56.1 (61.6 / 57.3) | – / 43.4 (52.4 / 43.7) | 1.3 (2.0) |
| Coder-1.5B | 71.3 / 66.5 (70.7 / 66.5) | 68.3 / 58.5 (69.2 / 59.4) | 5.8 (6.1) |
| Qwen2.5-0.5B | 38.4 / 34.1 | 49.7 / 41.8 | 1.3 |

Headroom at T=0.8, n=16 (pass@1 / pass@16, sometimes-solved), on all 789 dev
problems for LiveCodeBench:

| Model | HumanEval+ | MBPP+ | LiveCodeBench (easy / medium / hard sometimes-solved) |
| --- | --- | --- | --- |
| Coder-0.5B | 46.5 / 78.0, 61.6% | 41.1 / 68.0, 52.6% | 3.2 / 12.5, 12.4% (34.1 / 3.3 / 0.4%) |
| Coder-1.5B | 62.0 / 87.8, 58.5% | 56.5 / 77.8, 57.4% | 11.4 / 28.5, 27.4% (63.1 / 16.7 / 2.1%) |
| Qwen2.5-0.5B | 28.4 / 58.5, 50.6% | 35.2 / 65.3, 56.9% | 2.7 / 11.4, 11.4% (32.2 / 2.7 / 0%) |

Coder-0.5B fails the LiveCodeBench headroom gate (12.4% sometimes-solved, 9.3-point
gap); Coder-1.5B passes it (27.4%, 17.1 points) and matches its published
numbers, so the user chose Coder-1.5B as v2's base. Hard problems give no
signal at either size. On the 156-problem published window alone, 1.5B's
sometimes-solved is 14.1%; the gate is read on all dev problems.
`eval/codeprobe/headroom-v1/` (`summary.json`, `summary-livecodebench-published-window.json`).

**2026-10-30 – Evaluation sets in our grader** (`7320c7c`). A problem
set is now a directory with hidden tests per problem, loaded on demand. HumanEval+:
163 of 164 tasks (`find_zero`, a special oracle, excluded); all 163 canonical
solutions pass, and our grader agrees with EvalPlus on all 163 Coder-0.5B
greedy samples (92 both pass, 71 both fail). Two tasks have results beyond
Python's 4,300-digit limit, so integers from 10^4000 up cross the sandbox as
digit strings. LiveCodeBench dev: 788 of 789 problems (`abc350_c` has no public
example). Against LiveCodeBench's verdicts on the 788 Coder-1.5B greedy samples,
the first agreement was 95.4%: all 36 disagreements were LeetCode samples that
LiveCodeBench passed and we failed with `NameError` on `List`, because its
evaluator runs a fixed import block before every solution. With that block
added (also given to the agent's `run_python`), 99.9%; the last disagreement was
a slow solution over our 4 s limit and inside LiveCodeBench's 6 s. LiveCodeBench
problems now carry its import block and 6 s limit, which is fairer to the model.
Final: 788/788 verdicts agree (103 both pass, 685 both fail).
`eval/codeprobe/humanevalplus-checks-20261030/`, `eval/codeprobe/livecodebench-agreement-20261030/`
(earlier runs in `-no-imports` and `-4s`).

**2026-10-30 – Training pool: 20k problems** (`7320c7c`). Deduplicated
against every eval statement, including confirmation: word 8-gram containment of
the eval statement in the training statement, threshold 0.3. Calibrated on
43,688 candidates: the one true duplicate (a KodCode rewording of MBPP/119)
scored 0.53, and the 0.3–0.5 band was generic phrasing. The filter removed 144
of 225,336 KodCode candidates and none of 23,688 DeepCoder ones. KodCode-V1
(97d81f0, CC-BY-NC-4.0): 15,000 easy/medium function-style problems (11,108 /
3,892); call inputs are the literal calls inside its pytest asserts, and
expected values come from running its reference solution twice in the sandbox.
Exclusions among those tried: under 3 literal inputs 5,992; reference
nondeterministic, failed or OOM 38; reference raised on an assert input 12;
unencodable input (bytes) 1.
Hidden cases min/median/max 3 / 7 / 38, so its tests are weak. DeepCoder
(177913a) TACO and PrimeIntellect: 5,000 stdin problems (1,691 / 3,309) of 6,500
tried, kept when one of the first 3 reference solutions passes every test in our
grader (832 failed that; 650 function-call and 28 non-string problems skipped);
hidden cases 5 / 102 / 1,226. Commands in [usage](usage.md).
`data/agent/train-kodcode-v1-15k/`, `data/agent/train-deepcoder-5k/` (desktop).

**2026-10-30 – M0c: batched engine** (commit c137f05). At batch 1 the agent
loop ran Coder-1.5B at ~50 generated tokens/s (6.8 s per single-turn and 14 s
per 4-turn episode on LiveCodeBench dev), so one expert-iteration round (20k
problems × 8 interaction attempts) would take ~26 days. Replaced with our own
engine (scripts/agent/engine.py: static per-row KV cache over a layer schedule,
CUDA-graph decode), a batched decoder and a runner that refills rows as episodes
end. Profiling the live run found three costs, fixed in turn: grading on threads
starved the decoding thread of the interpreter lock (now in spawned processes);
the thought line's bounded repetition `[^\n]{0,1000}` made each grammar mask
~0.5 ms per row (any bound costs the same, so the thought line is now unbounded,
limited only by the per-turn token cap: a change to what every condition may
emit); and a mask buffer was allocated each step (now one pinned buffer, filled
while the GPU runs the forward). Throughput, 4-turn interaction at T=0.8, 256
episodes: LiveCodeBench dev 54 episodes/min (578 tokens/s) at 32 rows before
the fixes, 123/min (1,474 tokens/s) at 64 rows after; DeepCoder with
first-failure grading 94/min with the decoder never idle, solving 3 of 256. A
round fits in about a day. Batch invariance, greedy single-turn on 64 problems,
batch 1 vs batch 64: fp32 64/64 identical texts; bf16 29/64 identical and 52/64
same verdict (8 vs 9 solved), from bf16 kernels rounding differently by batch
size, so compared conditions must share dtype and batch settings. Training-mode
grading (`--first-failure`) stops hidden grading at the first failing chunk.
`python -m scripts.agent.run --model <Coder-1.5B snapshot> --problems <set> --condition interaction --max-turns 4
--limit 64 --seeds 0 1 2 3 --temperature 0.8 --dtype bfloat16 --rows 64`.
`eval/agent/throughput-coder-1.5b-20261030/` (batch 1), `eval/agent/throughput-coder-1.5b-batched-20261030/`,
`eval/agent/engine-checks-20261030/` (desktop).

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
