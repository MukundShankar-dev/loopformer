# Independent controller initialization supervision — 2026-10-08

The controlled intervention added a weight-12 loop-zero numerical loss to the
seed-83 remaining-work recipe, preserving its frozen executor, GRU, graph/count
panel, optimizer, 3,000 updates and threshold. It ran from commit `475f652` on the
RTX 5070 Ti desktop via `bash repair_controller.sh`.

Training/export completed in 88.55 seconds. Selection remained familiar-count
validation exact timing, then BCE. Selected update 2,800 achieved 95.92% familiar
stopping versus retained control 95.31%. Held-out 7/9/11 fell to 38.02%; count nine
was still 0%. Familiar initialization MAE was 0.212 and count-nine MAE 1.164, with
no count-nine example within half a step. Deep exact stopping was 1.80% overall.
Final-head familiar stopping was 94.10%, held-out 42.71%, deep 1.50%.

Stronger initialization supervision improves a familiar-range readout but does
not repair requested-count generalization. Loss dilution is insufficient as an
explanation; this single coefficient/seed does not rule out all objective changes
or prove the GRU incapable of counting.

Artifacts:

- Training: `models/stage1_pointer/controller-initialization-seed83/`.
- Evaluation: `eval/pointer_diagnostics/controller-initialization-seed83-20261008T223020.339488Z/`.
- [Training W&B](https://wandb.ai/mukunds/loopformer/runs/v3s2j5j1).
- [Evaluation W&B](https://wandb.ai/mukunds/loopformer/runs/069szusv).

Both selected and final heads use existing cached replay and first-crossing
metrics. A repeated correct letter cannot count as exact stopping. Pretrained
native inference was not rerun for this pilot; portable export and native/replay
contracts passed tiny-model tests. Reserved test and seed 29 were not read.
Local validation: 19 remaining-work/controller tests passed; two focused repair
objective/CLI/evaluation checks then passed. Shell syntax, whitespace and 253 local
file links passed before deployment. Tensor binaries and feature caches stay on
the desktop; copied metrics exclude weights and W&B SDK directories.
