# Ordinary trace baseline experiment — 2026-10-10

Status: full-model production SFT launched and verified on the desktop. Both
complete evaluations and independent audits/plots are queued automatically.
No full-training or full-benchmark quality claim yet.

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
The training-log preview is `eval/pointer_traces/seed61-20261010/previews/sft_learning.png`
and `.pdf`, separate from the queued audited benchmark figures. Original-Qwen
trace evaluation was running at 18,574/345,600 questions at inspection; SFT
evaluation had not started. No run was launched or modified during this inspection.
