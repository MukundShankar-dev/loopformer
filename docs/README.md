# Documentation

Start with the [research plan](project_plan.md) for the questions, stage gates, and claim hierarchy, then read [current status](status.md) for what has actually run and the next bounded experiment. The plan is the research source of truth; status supersedes dated setup language in the guides.

| Need | Canonical document |
| --- | --- |
| Recurrent model, parameter sharing, gradient behavior, Stage 0 validation protocol | [Architecture](architecture.md) |
| Pointer data, training, checkpoints, resume, and historical depth-6 comparison commands | [Pointer execution and training](training_pointer.md) |
| Metrics, ordinary baseline, full-loop evaluation, and deferred terminal overscaling | [Evaluation](evaluation.md) |
| Adaptive analysis, oracle/heuristic stopping, implemented interfaces, and remaining gates | [Adaptive inference compute](adaptive_compute.md) |
| Prompt-only completion, fixed prompt memory, and learned-stop evaluation | [Learned loop completion](learned_loop_completion.md) |
| Python environment and pretrained Stage 0 checks | [Setup](setup.md) |
| Windows/WSL2/CUDA installation and artifact migration | [Desktop setup](windows_cuda_setup.md) |
| Failure diagnostics and performance profiling proposal | [Diagnostics and performance](diagnostics_and_performance.md) |
| Resolved and open design choices | [Decisions](decisions.md) |

The adaptive-compute guide is the **single source for that extension**. The research plan states its place in the stage sequence; evaluation documents shared metric conventions; status records current evidence. Stage 3 asymmetric dynamics and Stages 4–6 transfer remain planned in the research plan. No separate phase-guide directory is needed.

## Evidence, not instructions

The dated reports in [`experiments/`](experiments/) preserve experiment configuration, observations, and limitations. They are not current runbooks. For the latest pretrained depth results, see the [fresh 30k report](experiments/stage1_fresh30k.md); the earlier [CUDA run](experiments/stage1_cuda_5k.md) and [Stage 0 validation](experiments/stage0_validation.md) provide historical evidence. The [adaptive nominal-trace analysis](experiments/adaptive_nominal_analysis.md) is observational and is not a pretrained terminal or adaptive-policy result.

The [completed loop-balanced ablation](experiments/stage1_loopbalanced.md) did not extend the frontier. The [completed baseline diagnostics](experiments/baseline2500_diagnostics.md) support investigating recurrent-history sensitivity and measured execution cost; [the guide](diagnostics_and_performance.md) describes the interfaces. Pretrained terminal overscaling and confirmation evaluation remain deferred. [AGENTS.md](../AGENTS.md) contains repository conventions.

The [offline failure review](experiments/baseline2500_offline_review.md) compares first-error confidence, restarted lookups, and matched trajectories across checkpoints 2500/3250/3750, with a recurrence/supervision audit.

The [controlled restart report](experiments/stage1_controlled_restarts.md) shows that retaining the original displayed Steps does not remove the suffix-restart benefit, while directly splicing fresh Rules-prefix representations into a loop-six state is strongly harmful.

The [frozen rule-edit report](experiments/stage1_rule_edit_probe.md) compares relevant-edge, irrelevant-edge, and early-loop controls on the retained checkpoint. It finds reliable early rule response and weaker, partly prompt-sensitive late response; the mechanism of late failure remains unresolved.

The [integrated mechanism report](experiments/stage1_integrated_mechanism.md) audits the completed three-checkpoint, same-mapping horizon and rule-edit diagnostic. It separates the strong local transition behavior from the later recurrent-history failure and records why the internal cause remains unresolved.

The [learned-completion ablation](experiments/stage1_learned_completion.md) records the joint stop-head training and full-loop evaluation. Stop timing does not extend beyond trained depths, and forced-depth pointer generalization is worse than the matched CE-only checkpoint.
