# Adaptive inference compute roadmap

**Status: planned research direction.** No adaptive stopping policy, oracle-summary utility, learned halting head, calibration study, or adaptive latency result is implemented. The immediate experiment remains [loop-balanced Stage 1 training](training_pointer.md#loop-balanced-loss-experiment). This roadmap extends the [research plan](project_plan.md), preserving its gates for execution, untreated dynamics, asymmetric training, and transfer.

## Core question

> Can LoopFormer determine how many recurrent steps an individual example needs instead of using one fixed inference depth for every input?

More precisely:

> Can the model learn when additional recurrent computation is useful, unnecessary, or likely to damage an already-correct state?

Recurrent execution → depth generalization → overscaling dynamics → adaptive inference compute / learned stopping is the broader direction. The [fresh 30k result](experiments/stage1_fresh30k.md) motivates investigating depth: execution extends past six trained transitions but degrades at greater task depths. It does not establish post-completion damage or adaptive-policy headroom. Those require separate evidence.

Stopping and asymmetric training address related but distinct questions. Stopping can avoid a dangerous transition; Stage 3 asks whether that transition can become safer while retaining repair. Adaptive success does not pass the asymmetric-training gate, and stability without useful computation remains insufficient.

## Existing substrate and boundaries

Extend the current components rather than building a second evaluation framework:

| Component | Implemented substrate to reuse |
| --- | --- |
| [`loop_test.py`](../scripts/eval/loop_test.py) | Shared full-loop inference; `trajectories.csv`, `examples.csv`, `depth_by_loop.csv`, `summary.json`; deterministic settings, checkpoint/data/source hashes and synchronized aggregate throughput |
| [`loop_metrics.py`](../scripts/eval/loop_metrics.py) | Complete nominal trajectories, first error, correct prefix, first final match, decoded repeats, and depth-by-loop summaries |
| [`overscaling_metrics.py`](../scripts/eval/overscaling_metrics.py) | All four adjacent correctness transitions, repair/damage, net gain, continuously-correct survival with censoring on validated terminal tasks |
| [`recurrent_pointer.py`](../scripts/eval/recurrent_pointer.py) | Synchronized inference timing and executed example-loops/s, including mixed-depth batch padding |

Existing exports contain predictions, labels, and true-target margins. They do not contain full answer distributions, hidden-state features, or per-example latency distributions. Prediction stability and offline labeled analysis can use existing traces without inference; entropy and top-two answer confidence require a small future feature-export extension to the same evaluator. A stopping-aware execution path and actual latency measurements would be later work. Post-processing full traces does not skip model work.

Use only complete runs with a declared task variant, horizon, target basis, and provenance. Preserve historical output schemas and metric definitions; write any derived analysis to a new location with its own config, version, and input hashes.

## Stage A — depth trajectory analysis

Start with saved traces. Report nominal intermediate execution separately from fixed-final-answer observations. Original pointer mappings continue after requested depth `d`: a correct additional lookup can change the answer. Such changes cannot automatically be labeled overthinking damage. For genuine post-completion analysis, first obtain the deferred [absorbing-terminal evidence](overscaling_eval.md), inspecting nominal performance and terminal-identification shortcuts.

Declare an observed horizon `T`, an eligible first loop `L`, and an early cutoff `b` before analysis. For primary terminal dynamics use `L = d` and, initially, `b = d`. Let `c_t` mean allowed-symbol argmax equals the fixed final target; preserve the evaluator's tie convention. Earlier final matches remain shortcut diagnostics. For other tasks, define completion and eligibility explicitly; original-task classifications are observational only.

Use mutually exclusive categories with this precedence over eligible loops `L..T`:

| Category | Definition |
| --- | --- |
| Unsolved | No eligible correct readout within the observed horizon. |
| Unstable | A correct readout is followed by damage and later recovery: a right→wrong→right pattern. May also end wrong. |
| Overthinking | At least one correct readout followed by damage, with no subsequent recovery. |
| Early-stable | First eligible correct loop is at or before `b`, and every later observed readout is correct. |
| Needs-compute | First eligible correct loop is after `b`, and every later observed readout is correct. |

“Stable” means stable within the observed horizon; it is not indefinite safety. Varying `b` changes the category counts and must be reported. With `b=d`, needs-compute measures late solution/recovery, not simply ordinary traversal of a deeper task.

Derive per example:

- First final-correct loop over all observations and first eligible correct loop, separately.
- Whether correctness remains continuous from first eligible solution through `T`.
- First eligible damage event `t→t+1`, labeled by both indices.
- Last continuously correct region (maximal final run of correct readouts), or none if the last readout is wrong.
- Minimum safe stopping depth under the Stage B rule, or an explicit missing value.
- Fixed-depth recurrent compute `T` and observed trace length; retain nominal complete-trajectory correctness, first error, and correct prefix.

Reuse existing diagnostic aggregation and transition semantics. Add synchronized latency measurements only when running inference, with separate loading, encoding, readout, controller, export, and model-compute scopes. Existing aggregate timings cannot reconstruct p50/p95 per-example latency.

## Stage B — oracle compute allocation

Calculate an offline oracle before training a controller. A first conservative stability rule is:

```text
safe(t) = t >= L and all(c_u for u in t..T)
oracle_stop = min(t where safe(t)), otherwise no safe stop
```

This uses labels and future readouts, so it is unavailable to inference. It is a finite-horizon stability oracle, not a learned policy or evidence of indefinite correctness. On examples with no safe stop, use fallback depth `T` and score its actual answer; never drop these examples. Report their fraction and include fallback loops in all compute summaries.

Compare each fixed budget `B` on the same cohort with an oracle computed for that same horizon (`T=B`), holding task variant, eligibility, scoring, and checkpoint fixed. Primary terminal horizons cover every nominal depth. Report accuracy, mean loops, p50/p95 loop count, total loops, and recurrent-loop savings `1 - sum(oracle_loops)/(N*B)`. Plot quality versus compute across declared budgets/rules rather than selecting one favorable point. Also retain depth-specific counts and uncertainty estimates.

The conservative suffix rule identifies unnecessary tail compute but cannot preserve a transient solution that is damaged by `T`. To study avoiding damage, separately predeclare a window rule: require correctness for `w` consecutive observed loops starting at candidate `t`, with `t+w-1 <= T`, then choose the earliest eligible candidate. Include a `w=1` first-correct oracle as an optimistic reference. These oracles read the future, but stop at `t`; they cannot claim safety beyond that window. Missing windows are censored, not successful. Keep results for each rule separate and use the same fallback accounting.

An oracle is an optimistic bound for its declared offline criterion. The suffix oracle is not the maximum achievable stopped-answer accuracy, since it excludes transient correct answers. Oracle loop savings are potential compute allocation, not measured speedups. Readout overhead, controller cost, prelude cost, batching, and idle work may change actual latency substantially.

Pointer prompts already provide requested depth. Include a **stop-at-requested-depth** baseline alongside common fixed depths: variable loop counts from known `d` alone do not establish learned allocation. Eligibility `L=d` also prevents the primary oracle from claiming savings before nominal completion. Any early-stop analysis must be separate and inspect complete intermediate execution and shortcuts.

## Stage C — heuristic stopping baselines

Before learned halting, compare simple causal policies using only information available at loop `t`:

- Predicted-answer margin threshold: top allowed-answer logit minus runner-up, not the saved true-target margin.
- Entropy threshold over a declared allowed-answer distribution.
- Prediction stability for `k` consecutive loops.
- Confidence improvement below a threshold over a declared window, where meaningful; stalled wrong predictions can also look stable.
- Common fixed-depth baselines at matched average compute, plus the requested-depth baseline.

Declare minimum loop, maximum budget, feature vocabulary/normalization, tie handling, and fallback. Thresholds, `k`, and windows must be selected on development data and frozen before confirmation. Calibration fits also use development data only. Seed-17 evaluations already inspected are development diagnostics; seed 29 remains reserved under the [confirmation protocol](depth_generalization.md#reserved-confirmation-set).

Report task accuracy at the stopped readout, mean/p50/p95 loops, synchronized latency, and quality-compute curves. Complete-trajectory accuracy retains its historical definition: all nominal steps must be observed and correct. Early stopping before `d` leaves that metric incomplete; report coverage and a full-trace diagnostic reference separately, never score a correct truncated prefix as a complete trajectory.

Define additional rates with counts and denominators before evaluation:

- Premature stop: stopped answer wrong although a later correct readout exists within the budget, divided by all examples. Also report stops before nominal completion separately.
- Unnecessary compute: examples with a safe oracle point where policy depth exceeds that point, divided by examples with a safe point; report excess loops too.
- Damage exposure: fraction with an eligible right→wrong event before or at the stop. Separately report a correct readout followed by a wrong stopped answer; recovery can distinguish these rates.

A policy may stop incorrectly even when no later loop would solve the task; this is error, but not the defined premature-stop rate. Keep final-answer correctness and intermediate execution distinct.

Offline policy replay on complete traces is useful for scoring and loop accounting. Actual savings require terminating execution, freezing the stopped readout, and counting all executed work. Evaluate that later implementation with the same metrics and fixed-depth path, including mixed-batch padding or compaction costs. Matched average loop count is only a compute proxy; also compare measured latency.

## Stage D — learned halting

Only proceed if the oracle shows meaningful headroom and heuristic baselines establish what a learned controller must improve. Predeclare a practical headroom criterion on development data before evaluating a controller.

Keep the first version simple: freeze the pretrained base and trained recurrent transition, then train a lightweight head from recurrent hidden states at a declared position and/or compact confidence features to output stop/continue probability. Start with supervised oracle labels or cost-sensitive classification. Explain how no-safe-stop examples, fallback, censoring, and oracle-rule choice determine labels. A simple supervised approach should precede full RL.

A later objective may resemble:

```text
utility = task_success - λ * compute
```

This is one candidate, not the only allowed method. Vary costs or thresholds on development data to obtain a quality-compute Pareto curve; freeze the operating points before confirmation. Report calibration and uncertainty, not just stop/continue classification accuracy. A controller trained on one checkpoint's states may fail on another checkpoint, deeper tasks, or a changed prompt distribution.

## Stage E — broader evaluation

Only after adaptive stopping works in the controlled pointer environment should it broaden to tasks with checkable answers, enough evaluation examples, compositional structure, and a meaningful variable-compute interpretation. Candidate directions include held-out symbolic transition families, structured graph/navigation problems, and small verifiable arithmetic or algorithmic tasks. These are candidates, not implemented benchmarks. Preserve the research plan's separate gates for multi-family recurrent training and whole-family transfer.

Test controller transfer across held-out depths, mappings, surface forms, and eventually families, distinguishing controller training exposure from recurrent-model exposure. Success on natural-language reasoning does **not** imply that one loop corresponds to one human reasoning step. Decoding and language understanding introduce additional uncertainties.

Keep 0.5B for cheap iteration. If adaptive allocation succeeds, reproduce the key result on a model around 1.5B–3B after checking architecture compatibility and resource use. Scaling is conditional, not a current implementation task.

## Adaptive-eval reliability requirements

- Save explicit configs, seeds, model/base/tokenizer identities, checkpoint/data/source provenance, input trace hashes, task variant, stopping rule and analysis version.
- Keep machine-readable traces with loop features, stop decisions, fallback reasons, and executed-work counts; generate regression summaries without overwriting historical outputs.
- Separate development and confirmation splits, check overlap, and freeze checkpoint, thresholds, calibration, and claims before confirmation. Reserve a new untouched split if confirmation informs further tuning.
- Keep fixed-depth inference and matched-compute comparisons. Distinguish loop savings, measured latency savings, and quality changes with counts and uncertainty.
- Evaluate confidence/calibration by depth and relevant cohorts, using reliability bins and a declared scoring rule such as Brier score. Raw logits and stable predictions alone are not calibrated correctness probabilities.
- Synchronize CUDA/MPS for timing. Report p50/p95 latency, warmup, sample counts, batch composition, timing boundaries, and controller/readout overhead; do not infer latency quantiles from aggregate throughput.
- Test stopping indices, causality, no label/future leakage, fallback, incomplete trajectories, censoring, tie handling, all-wrong/stable/damage/recovery cases, aggregation, and actual executed-work accounting.

## What would count as a strong result?

An adaptive policy should ideally preserve most of the best fixed-depth quality, materially reduce average recurrent loops, avoid some later-loop damage, expose useful compute-quality operating points, and transfer at least partially beyond its exact training distribution. Set quantitative success thresholds before confirmation. A lower loop count with a large quality loss is not automatically a win; a quality gain at greater compute is a different result.

Useful negative results include little oracle headroom, confidence that is poorly calibrated beyond training depth, heuristics matching a learned head, controller overfitting, savings disappearing after runtime overhead, and failure to transfer. Report these with the same provenance and per-depth diagnostics. They constrain the adaptive-compute hypothesis without erasing the existing recurrent-execution evidence.

## Current implementation decision

This update adds a roadmap only. The retained nominal traces support observational analysis, but no pretrained terminal traces establish the primary damage/safety setting, and the richer confidence and latency fields are absent. A metrics-only oracle utility is deferred until its task semantics and stability rule are fixed for a bounded analysis. It would read existing traces without rerunning inference or changing historical metrics.
