# Documentation

Start with the [research plan](project_plan.md) for the questions, stage gates, and claim hierarchy, then read [current status](status.md) for what has actually run and the next bounded experiment. The plan is the research source of truth; status supersedes dated setup language in the guides.

| Need | Canonical document |
| --- | --- |
| Shared number reader with unchanged training depth/count exposure | [Number-reader repair](number_reader_repair.md) |
| Frozen reserved-test and larger pointer benchmark | [Benchmark protocol](pointer_benchmark.md) |
| Current controller repair and acceptance criteria | [Controller repair](controller_repair.md) |
| Current executor/controller implementation checklist | [Pipeline upgrade](pipeline_upgrade.md) |
| Recurrent model, parameter sharing, gradient behavior, Stage 0 validation protocol | [Architecture](architecture.md) |
| Pointer data, training, checkpoints, resume, and historical depth-6 comparison commands | [Pointer execution and training](training_pointer.md) |
| Metrics, ordinary baseline, full-loop evaluation, and deferred terminal overscaling | [Evaluation](evaluation.md) |
| Adaptive analysis, oracle/heuristic stopping, implemented interfaces, and remaining gates | [Adaptive inference compute](adaptive_compute.md) |
| Prompt-only completion, fixed prompt memory, and learned-stop evaluation | [Learned loop completion](learned_loop_completion.md) |
| Python environment and pretrained Stage 0 checks | [Setup](setup.md) |
| Windows/WSL2/CUDA installation and artifact migration | [Desktop setup](windows_cuda_setup.md) |
| Current no-update controller training audit and historical performance investigations | [Diagnostics and performance](diagnostics_and_performance.md) |
| Resolved and open design choices | [Decisions](decisions.md) |

The adaptive-compute guide is the **single source for that extension**. The research plan states its place in the stage sequence; evaluation documents shared metric conventions; status records current evidence. Stage 3 asymmetric dynamics and Stages 4–6 transfer remain planned in the research plan. No separate phase-guide directory is needed.

## Evidence, not instructions

The dated reports in [`experiments/`](experiments/) preserve experiment configuration, observations, and limitations. They are not current runbooks. For the latest pretrained result, see the [isolated executor audit](experiments/stage1_executor_seed61.md). The [fresh 30k report](experiments/stage1_fresh30k.md), the earlier [CUDA run](experiments/stage1_cuda_5k.md) and [Stage 0 validation](experiments/stage0_validation.md) provide historical evidence. The [adaptive nominal-trace analysis](experiments/adaptive_nominal_analysis.md) is observational and is not a pretrained terminal or adaptive-policy result.

The [completed loop-balanced ablation](experiments/stage1_loopbalanced.md) did not extend the frontier. The [completed baseline diagnostics](experiments/baseline2500_diagnostics.md) support investigating recurrent-history sensitivity and measured execution cost; [the guide](diagnostics_and_performance.md) describes the interfaces. Pretrained terminal overscaling and confirmation evaluation remain deferred. [AGENTS.md](../AGENTS.md) contains repository conventions.

The [offline failure review](experiments/baseline2500_offline_review.md) compares first-error confidence, restarted lookups, and matched trajectories across checkpoints 2500/3250/3750, with a recurrence/supervision audit.

The [controlled restart report](experiments/stage1_controlled_restarts.md) shows that retaining the original displayed Steps does not remove the suffix-restart benefit, while directly splicing fresh Rules-prefix representations into a loop-six state is strongly harmful.

The [frozen rule-edit report](experiments/stage1_rule_edit_probe.md) compares relevant-edge, irrelevant-edge, and early-loop controls on the retained checkpoint. It finds reliable early rule response and weaker, partly prompt-sensitive late response; the mechanism of late failure remains unresolved.

The [integrated mechanism report](experiments/stage1_integrated_mechanism.md) audits the completed three-checkpoint, same-mapping horizon and rule-edit diagnostic. It separates the strong local transition behavior from the later recurrent-history failure and records why the internal cause remains unresolved.

The [learned-completion ablation](experiments/stage1_learned_completion.md) records the joint stop-head training and full-loop evaluation. Stop timing does not extend beyond trained depths, and forced-depth pointer generalization is worse than the matched CE-only checkpoint.

The [fixed-prompt failure investigation](experiments/stage1_fixed_prompt_review.md) audits the new pretrained run, locates the stopping supervision gap, measures conditional transition failure and threshold limits, and relates the findings to primary literature.

The [paired-Steps probe](experiments/stage1_paired_steps.md) controls mapping/start and confirms requested-count effects on late execution and failure to extend stop timing; it also records working-state drift without claiming a hidden-state fixed point.

The [depth-12 checkpoint progression](experiments/stage1_depth12_progression.md) finds regression after step 5,000 and motivates a matched-count probe before further training.

The [depth-12 paired-count probe](experiments/stage1_depth12_paired_counts.md) demonstrates late execution sensitivity to requested count on identical mappings, alongside persistent later-loop failure.

The [end-to-end reassessment](experiments/pipeline_reassessment.md) separates established implementation evidence from missing architecture, objective, data and optimization controls. The user subsequently authorized its [implementation checklist](pipeline_upgrade.md); the report retains the pre-change evidence.


The [frozen-controller diagnostic](experiments/controller_seed61_diagnostic.md)
finds accessible initial count information, poor later count decoding, and perfect
tiny-set fitting with limited held-out timing generalization. It motivates a
separate controller-training phase with the successful executor frozen.

The [controller-only training report](experiments/controller_seed61_training.md)
records substantially improved familiar-count stopping and memory decodability,
but count-specific interpolation failures and no exact stopping beyond count 13.

The [completed remaining-work comparison](experiments/controller_remaining_seed61.md)
finds better numerical readout but essentially unchanged stopping across three
paired seeds. Initial unfamiliar-count estimates are already wrong before
recurrence. The [training audit](diagnostics_and_performance.md#controller-training-audit--current-desktop-command) checks fitting and objective gradients before another architecture or recipe change.


The [completed controller training audit](experiments/controller_training_audit_seed61.md)
finds modest training/validation graph gaps, residual fitting error, mixed
objective-gradient alignment, and persistent count-nine initialization errors.
The [desktop remote-operations guide](windows_cuda_setup.md#remote-operations-from-the-mac)
records the verified SSH and repository Git-authentication setup.

The [initialization-only repair pilot](experiments/controller_initialization_seed61.md)
improves familiar stopping slightly but leaves count nine and deep timing broken.
The [completed affine-controller repair](experiments/controller_affine_seed61.md)
passes timing through 64 across three optimizer seeds, with bounded native checks.
The [repair protocol](controller_repair.md) owns reproduction, scientific scope
and the distinction between seen numeric values and unseen rollout lengths.


The [frozen larger benchmark](experiments/pointer_frozen_benchmark.md) confirms a
perfect current reserved test and strong executor trajectories through 256 on
1,350 new graphs, but exposes controller number-reading failures starting at 70.
All native/reuse checks and independent reference audits pass. This report
supersedes broader interpretations of the smaller controller repair panel.

The [shared reader report](experiments/controller_number_reader.md) records the
authorized reader-only intervention with unchanged count/depth exposure and a
fresh graph confirmation panel; consult its status before treating it as verified.
