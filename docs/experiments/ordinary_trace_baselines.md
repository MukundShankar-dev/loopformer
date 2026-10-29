# Ordinary trace baseline experiment — 2026-10-10

Status: complete. Full SFT trained to update 2,250, both 345,600-query
evaluations finished, and `trace_analysis` independent audits passed
(`eval/pointer_traces/seed61-20261010/completion.json`). Results are in
[Completed benchmark results](#completed-benchmark-results--2026-10-11). Sections
below that describe launch, stopping and partial progress are a dated log.

## Protocol

See [the owning guide](../trace_baselines.md),
[training config](../../configs/trace_sft.json) and
[inference config](../../configs/pointer_trace_baselines.json). Both arms begin
from Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`.
The SFT arm trains all 494,032,768 parameters on the exact successful executor's
36,000 examples at counts 1–6/8/10/12, one epoch, effective batch 16 and 2,250
updates. Full-vocabulary response CE covers each successive state and EOS.
Both inference arms get the same three-shot trace prompt, one time per query.

Main evaluation: existing 1,350 graph tables, every request 1–256, 345,600 queries
per arm. Additional regression: existing opened 1,536-query seed-61 test.
No new data, longer training counts, model selection on the benchmark, or
modification of the frozen recurrent system. Strict success requires the entire
correct prefix, exactly N emitted states, and EOS. A cyclic final-letter alias
at a wrong length does not pass.

## Preflight observations

Desktop: RTX 5070 Ti 16 GB, Torch 2.14.0+cu130, Transformers 5.17.0, BF16/SDPA.
Full-model ten-update smoke runs kept effective batch 16:

| Microbatch | Accumulation | Gradient checkpointing | Mean update seconds | Peak allocated GiB |
| --- | --- | --- | --- | --- |
| 4 | 4 | yes | 0.851 | 8.61 |
| 8 | 2 | yes | 0.603 | 8.83 |
| 16 | 1 | yes | 0.515 | 8.03 |

Choose batch 16. Different depth order and allocator lifetimes mean peaks need
not be monotonic; these are whole smoke runs, not controlled per-depth memory
benchmarks. Microbatch 16 without checkpointing in the earlier middle-layer draft
saturated memory and was stopped; its timing is not a full-SFT estimate.
All full-SFT smoke checkpoints were initialized independently from original Qwen;
production training also starts fresh, never from a smoke checkpoint.

The complete dataset dry run validated 36,000 training examples, 2,250 updates,
and a maximum encoded sequence length of 439 with no truncation. Focused tests
check full-vocabulary loss/gradient equivalence, full trainability, EOS/state
scoring including cyclic aliases, unchanged old final-only scoring, cached vs
uncached generation, independent score auditing, standard HF checkpoints, and
bitwise CPU optimizer/RNG/cursor resume.

An initial original-Qwen inference profile through batch 128 rejected static
compiled decoding: slower and nonidentical continuations. The production profiler
also considers 256/512, chooses validation throughput only, and records its actual
selection. Production compilation is disabled after the nonidentical/slower
preflight; this preliminary profile is not the frozen production batch policy.

## Exact commands and artifacts

```bash
bash trace_baselines.sh --dry-run
WANDB_MODE=online bash trace_baselines.sh
python -m scripts.eval.trace_status --remote desktop --watch
```

Training output: `models/trace_sft/seed61-full/` with metrics, W&B provenance,
standard-HF `best/` and `last/` plus optimizer/RNG state. Evaluation output:
`eval/pointer_traces/seed61-20261010/` with each arm's frozen identity, profile,
native sensitivity audit, raw chunks, opened-test responses and summaries.
`trace_analysis` independently validates every raw response before producing
outcome matrices, CSV strata, paired graph-cluster differences and four separate
figure families under `plots/`. Existing 25-family plots remain unchanged.

Capacity, objective/prompt, and separate controller count-label exposure are
explicit comparison limitations. Cached BF16 trace-generation timing cannot be
compared to logical R-pass budgets as though they were equal compute units.

## Launch verification and handoff

The persistent job is `WANDB_MODE=online bash trace_baselines.sh`, with its log
at `eval/pointer_traces/seed61-20261010/launch.log`. At the first check it had
completed 40/2,250 optimizer updates, around 0.47 seconds for the latest update.
This is an early progress observation, not a full-run speed or quality result.
[W&B training run](https://wandb.ai/mukunds/loopformer/runs/bew2vujj).

The 32 focused tests passed, including the complete four-family plot renderer
on explicitly synthetic fixtures. All documentation links checked locally.
Preflight records are saved at `eval/pointer_traces/preflight-20261010/`; disposable
smoke model/optimizer tensors are not production inputs. The final code uses a
cache adapter so the historical ordinary evaluator and its completed source
freezes remain byte-identical. No new dependency is required.

The user stopped the production run on 2026-10-10. Verified that no training
or queued benchmark process remains. The saved best checkpoint at update 250
is retained at `models/trace_sft/seed61-full/best/`; later unsaved updates will
be repeated. Neither full benchmark has started. Restart from the desktop
repository with `bash trace_baselines.sh`.

Monitor with `python -m scripts.eval.trace_status --remote desktop --watch`.
The launcher owns an exclusive lock, and will resume saved training state or
unchanged generation chunks if interrupted. Keep the source/configuration stable
while it runs. Do not interpret a partial trace benchmark as the final comparison.

## Completed SFT log inspection — 2026-10-10

The desktop run subsequently resumed from update 250 and completed update 2,250;
its resumed invocation took 1,045.9 seconds for 2,000 updates. The selected best
checkpoint is update 2,250. [Resumed W&B run](https://wandb.ai/mukunds/loopformer/runs/ohjupgd1).
Trained-count validation response CE decreased from 2.706965 at initialization
to 1.980952e-6. All 64 teacher-forced validation responses per trained count
are correct at the final checkpoint.

The free-generation development panel (four queries per count) achieves 100%
strict success at all trained counts and 0% at held-out counts 7, 9 and 11.
At count 11 all four generated prefixes through N are correct, but none stop
exactly at N. This establishes a failure on the small held-out-count panel;
it does not locate every failure or establish large-benchmark execution quality.
Held-out teacher-forced CE at counts 7/9/11 is 1.321/1.050/0.696 respectively.
Training optimization succeeded; count generalization is not demonstrated.

Small logs and summaries were copied to the Mac at `models/trace_sft/seed61-full/`.
A training-log preview was rendered at the time, but it is not retained; the
audited training figure is `plots/04_sft_training`. Original-Qwen trace
evaluation was running at 18,574/345,600 questions at inspection; SFT evaluation
had not started. No run was launched or modified during this inspection.

## Completed benchmark results — 2026-10-11

Both arms completed all 345,600 queries (1,350 graphs × counts 1–256) and passed
the independent audit, which re-validates every raw response against the rules.
Rates are graph-cluster bootstrap 95% intervals from each arm's
`cluster_intervals.json`; the recurrent column is the frozen final system from
[the population analysis](pointer_population_analysis.md), implied by
`paired_deltas.csv`.

| Strict success (exact states, exactly N, EOS) | Counts | Original Qwen trace prompt | Full trace SFT | Final recurrent |
| --- | --- | --- | --- | --- |
| Trained | 1–6, 8, 10, 12 | 0.37% (0.26–0.49) | 100% (100–100) | 99.05% |
| Held out | 7, 9, 11 | 0% | 0.15% (0.05–0.27) | 98.30% |
| Beyond SFT training | 13–256 | 0% | 0% | 96.93% |
| All | 1–256 | 0.013% | 3.52% | 97.02% |

Paired recurrent-minus-SFT strict-success differences (`paired_deltas.csv`):
trained −0.95 pp (−1.35 to −0.58), held out +98.1 pp (97.5–98.8), counts 13–63
+97.1 pp (96.2–97.9), counts 64–256 +96.9 pp (96.0–97.8).

Interpretation:

- **SFT executes correctly but cannot stop at an unseen length.** In `sft/strata.csv`
  every SFT failure is a termination error; no response continues past a wrong
  state. Emitted state counts in the raw `sft/chunks/` (1,350 responses each):
  count 7 → 6 states (1,344) or 7 (6); 9 → 8 (all); 11 → 12 (all, the first 11
  correct); 13 → 12/11/10 (652/516/182); 40 → 4 (all); 256 → 2 (1,182) or 6 (168).
  In range SFT snaps to a neighbouring trained length. For multi-digit counts it
  appears to key on individual digits (40 → 4, 256 → 2 or 6) rather than the
  number. It learned per-count lengths, not to read N.
- **Inside the trained counts SFT is slightly better** than the recurrent system
  (about 1 pp, with an interval that excludes zero), so the recurrent advantage
  is entirely in counts SFT did not see, including the held-out in-range counts.
- **The comparison is not exposure-matched for stopping.** The final controller
  was trained on counts 1–63 and uses an engineered count reader and countdown;
  SFT saw only the nine trained counts. The 13–63 cohort therefore compares
  trained recurrent stopping with untrained SFT stopping. Counts 7/9/11 and
  64–256 are unseen by both systems' stopping, and the recurrent system still
  leads by about 97–98 pp there.
- Original Qwen with the three-shot trace prompt almost never produces a complete
  correct trace (0.013%): 90% of responses stop early and 6% are malformed.

The cohort rows beyond `all` and `extrapolation` were added on 2026-10-29 by
re-running `python -m scripts.eval.trace_analysis` on the unchanged saved
responses. Outcomes, strata and intervals were byte-identical; only
`paired_deltas.csv`, the recorded analysis-code hashes and `plots/provenance.json`
changed.
