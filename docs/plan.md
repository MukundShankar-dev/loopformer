# Plan — v2: how should a small coding agent spend test-time compute?

Status: draft research design, under review. Nothing below is implemented
except where marked **ready**. v1 is summarized in [history](history.md).

## Question

A small model can spend extra inference compute in three ways:

| Axis | Mechanism | What it buys |
| --- | --- | --- |
| **Depth** | Loop the shared block T times per token | More computation per token, with no new information |
| **Breadth** | Draw k independent attempts and pick one | Diversity |
| **Interaction** | Run its code, read the result, revise, over several turns | New information from execution |

We build a looped coding agent on Qwen2.5-Coder-0.5B-Instruct. Its
architecture is identical to v1's base (Qwen2, 24 layers, hidden 896, GQA
14/2, tied embeddings), so v1's looping code applies. We then ask:

- **RQ1:** At matched compute, which axis gives the most correct solutions on
  held-out problems?
- **RQ2:** Do the axes complement each other, for example does depth help
  more on turns after an error?
- **RQ3:** Can a trained policy allocate compute across axes (loops per
  token, whether to run code, when to submit) better than the best fixed
  allocation at equal mean compute?
- **Engineering (not a research question):** exact KV-cached looped decoding,
  and self-speculative decoding that drafts with fewer loops.

Out of scope: tasks other than self-contained programming problems, and
natural-language transfer claims.

## Experimental design

### Task and information

- **Episode:** one programming problem.
- **What the agent sees:** the statement plus **public** tests (the examples).
- **Success:** the submitted code passes the **hidden** tests, which the
  agent never sees.
- **Same information in every condition:** each gets the statement and the
  public tests; they differ only in how compute is spent.
  - Breadth selects among its k attempts by public-test results; ties are
    broken by the first attempt.
  - Interaction may run the public tests and arbitrary code between turns.
  - Depth-only gets no execution.
- **Selection rule, stated plainly:** oracle pass@k (choosing using hidden
  tests) is a headroom diagnostic, never a deployable baseline.

### Conditions

Inference uses a full grid on the same trained checkpoint:

- T ∈ {1, 2, 4} loops;
- k ∈ {1, 4, 16} attempts;
- turns ∈ {1, 2, 4} (one turn means no execution feedback).

Training conditions are fewer, because each costs a run (see Training).

### Compute accounting

- **Primary:** model FLOPs per episode, from layer counts.
  - Prefill counts only new tokens, since the cache holds the rest.
  - Decode costs P + T·R + C layers per token.
  - Each condition's FLOPs are logged, not estimated afterwards.
- **Secondary:** end-to-end wall-clock time on the desktop, split into model
  time and sandbox time, with synchronized timing and warmup excluded.
- **Comparison method:**
  - Each condition is a point (FLOPs, success rate). Each axis traces a curve.
  - Comparisons are made at three FLOP budgets fixed before results, chosen
    from the M0 cost measurements to span the grid.
  - Success at a budget is interpolated only between measured points, and the
    interpolation is drawn as such.

### Statistics and pre-registration

- **Unit:** the problem. All comparisons are paired on the same problems.
- **Intervals:** bootstrap 95% intervals over problems.
- **Seeds:** at least three training seeds; the spread across seeds is
  reported separately.
- **Primary comparisons, fixed before confirmation:**
  - **P1:** at each budget, best depth-only vs best breadth-only vs best
    interaction-only.
  - **P2:** best combined allocation vs best single axis.
  - **P3:** learned allocation vs best fixed allocation, at equal mean
    compute.
  - Everything else is exploratory and labelled so.
- **Minimum detectable effect:** computed from M0's per-problem variance and
  the confirmation-set size. If the set can't detect, say, 5 points, the
  claims are scoped to what it can detect.
- **Freezing:** primary metric, comparisons, budgets, confirmation problem IDs
  and thresholds are committed (hash recorded) before M4 starts. The
  confirmation set is opened once per final checkpoint.

### Threats to validity

- **Contamination:** headline results come only from LiveCodeBench problems
  published after the base model's release. Deduplicate training against all
  evaluation sets.
- **Weak tests:** use EvalPlus "+" tests where they exist, and report
  public-pass-but-hidden-fail separately.
- **Reward hacking:** hidden tests live outside the agent's workspace.
  - Log sandbox access attempts.
  - Flag solutions that special-case test inputs.
- **Scaffold confound:** one prompt and tool format for all conditions.
  Single-turn uses the same format with execution disabled.
- **Training distribution:** a model trained multi-turn may underrate
  single-turn. Mitigate by training on mixed turn budgets, and report
  single-turn-trained controls.
- **Distillation confound:** avoided in the main line, which is teacher-free.
  The optional teacher-hybrid arm is reported separately.
- **Constrained decoding:** it changes what the model can emit, so it applies
  identically to every condition, both baselines and trained models.
- **Nondeterminism:** fixed seeds, fp32 evaluation, and recorded package and
  hardware versions.

## Evaluation suite

Every episode records structured outcomes, not just pass or fail.

**Per submission:**
- outcome class: `syntax_error / import_error / runtime_error:<type> /
  timeout / memory / wrong_answer / pass`, with the result of each test;
- hidden-test pass fraction, public-test pass fraction, and
  public-pass-but-hidden-fail.

**Per episode (process):**
- turns used; executions; tokens; loops per token; FLOPs; model vs sandbox
  time;
- submitted vs budget exhausted; invalid tool calls and format errors.

**Interaction dynamics:**
- First-turn vs final outcome.
- Per-turn **repair rate** (fail → pass) and **damage rate** (pass → fail),
  the turn-level form of v1's unanswered repair/damage question.
- How error types change across turns, for example syntax → runtime → wrong
  answer → pass.
- Whether a revision touches the code the feedback implicated.

**Depth dynamics:**
- Success vs T per problem, showing which problems flip either way.
- Per-loop convergence: KL divergence between the loop-t and loop-T
  next-token distributions.
- Whether the effect of depth differs on turns that follow error feedback.

**Diagnostics:**
- **CRUXEval** (predict a program's output) vs T: can it trace code,
  separately from writing it?
- A perturbed copy of the dev problems (renamed identifiers, paraphrased
  statements), to measure memorization.

**Integrity:**
- sandbox violation attempts;
- special-cased test inputs;
- reads of paths outside the workspace.

**Training reward** is the hidden-test pass fraction on training problems.
The training problems' hidden tests are still kept away from the agent.
All-pass (binary) reward is compared as an ablation. The richer outcome classes
are used for analysis, not reward shaping, unless an ablation justifies it.

## Agent and sandbox

**Tools** (minimal, single-file tasks):
- `run_python(code, stdin)` returns stdout, stderr, exit status and time, all
  truncated;
- `run_public_tests(code)` returns per-test results;
- `submit(code)`.

Help text and standard-library source are available inside `run_python`.
Library documentation is mounted read-only if a task needs it.

**Format:** each turn is an optional short diagnosis line, capped in length
and counted in FLOPs, followed by exactly one tool call. On the last turn only
`submit` is allowed, so every untruncated episode ends with a submission.

**Constrained decoding:** generation is restricted to this grammar, so an
invalid tool call is impossible rather than unlikely. This removes the need
for a teacher to teach the format.

**Sandbox:** bubblewrap behind our own wrapper.
- No network.
- Only these are visible: the interpreter and installed libraries
  (read-only), documentation (read-only), public tests (read-only), and a
  writable per-episode workspace.
- Never visible: `/mnt/c`, the WSL interop launcher, home directories, hidden
  tests.
- Limits: syscall filter, no-new-privileges, CPU time, wall time, memory,
  process count, output size. Cgroup limits via `systemd-run --user`.
- **Grader:** a separate sandbox run on a fresh copy of the submission, given
  hidden-test inputs only. Expected outputs never enter any sandbox; the
  verdict is computed on the host from the program's output, so a program
  cannot pass by exiting early, patching the harness or printing "pass". GPU
  processes never execute model code.
- **Sandbox output is untrusted text on the host:** never `eval`ed, executed,
  passed to a shell or parsed as Rich markup, and stripped of terminal control
  sequences before printing.
- **Adversarial test suite, required to pass before any model code runs:**
  fork bomb, memory hog, infinite loop, network access, writes outside the
  workspace, reads of `/mnt/c` and hidden tests, attempts to launch Windows
  executables.
- A separate throwaway WSL distro remains an optional extra layer.

## Data

| Use | Source | Notes |
| --- | --- | --- |
| Confirmation | LiveCodeBench problems after the cutoff (hashed ID list) | Public examples plus hidden tests; opened once per final checkpoint |
| Development | LiveCodeBench problems before the cutoff; HumanEval+ and MBPP+ (1–2 tests made public, the rest hidden) | Tune anything here |
| Training | Older LiveCodeBench problems, plus easy tiers of KodCode, TACO and APPS | Needs at least 3 hidden tests per problem and a reference solution that passes in our sandbox; deduplicated against every eval set. Tests must be input/expected pairs (stdin/stdout or call arguments/return value), because expected values never enter the sandbox; KodCode's pytest asserts embed them and need converting first |
| Diagnostic | CRUXEval | Not trained on |

## Training

The main line is **teacher-free**. All knowledge comes from the model itself
plus execution feedback, so gains cannot be attributed to distilling a larger
model.

1. **Surgery.** Loop R. At T=1 the model must equal the base model exactly
   (gate).
   - At T>1 it is untrained and expected to be worse: the middle layers never
     saw their own outputs as input.
   - Code has no per-loop targets, so depth is trained only through the final
     output. The risk is that extra loops become inert rather than useful;
     the depth gates detect this.
2. **Expert-iteration SFT.** This installs looping and the agent behaviour.
   1. Run the model in the scaffold on training problems, many attempts each.
   2. Keep episodes whose submission passes the hidden tests, including ones
      that failed and then repaired their own code.
   3. Fine-tune on the assistant tokens, with T sampled per batch from
      {1, 2, 4} and turn budgets mixed in {1, 2, 4}.
   4. Repeat for one or two rounds as the success rate rises.
   - Ablate the training scope here: R only, LoRA on R, or the full model.
   - If successes are too rare, go easier problems first, then more attempts
     per problem during collection only, then Coder-1.5B.
3. **RL.** A minimal multi-turn GRPO-style loop in this repo. vLLM-based
   rollout engines cannot run the looped model; reuse their loss definitions.
   Log-probabilities come from the same T-loop forward pass, and gradients
   flow through all loops.
   - Stage A: fixed T ∈ {1, 4}.
   - Stage B: the policy also chooses T per turn from a menu. The choice's
     log-probability enters the policy gradient, and the reward is reduced by
     λ·FLOPs.
   - Stage C: per-token depth, only if Stage B beats fixed allocations.
4. **Controls:**
   - the same pipeline with looping off (T=1 throughout);
   - a single-turn-only trained model;
   - a re-initialized R at matched parameters;
   - evaluation after SFT and after RL, so the RL gain is separable.
5. **Teacher-hybrid ablation** (optional): run only if the teacher-free line
   stalls, or as a deliberate comparison.
   - Our model attempts; when it fails, a teacher writes the repair turn
     given the actual code and error; keep episodes that pass hidden tests.
   - It answers how much teacher help with repairs speeds up learning to
     interact.
   - Record the teacher's exact model ID and date. Never mix its data into
     the main line.

## Milestones and gates

Each milestone ends with a gate. Do not start the next until it passes or the
user changes scope.

| Milestone | Content | Gate |
| --- | --- | --- |
| **M0a — Official-tool probe (ready, not run)** | Base Coder-0.5B, Coder-1.5B and Qwen2.5-0.5B on EvalPlus and LiveCodeBench: greedy plus 16 samples. Prepared: `configs/probe_headroom.json`, `scripts/codeprobe/`, `requirements-probe.txt`. | Greedy HumanEval+/MBPP+ within a few points of published numbers. Headroom: sometimes-solved ≥ 20% and pass@16 − pass@1 ≥ 10 points on the dev pool. |
| **M0b — Sandbox and harness** | bubblewrap wrapper, grader, tool loop with constrained decoding, outcome classes, FLOP logging. Base model at T=1 runs the breadth and interaction grid. | The adversarial suite passes. Harness single-turn results match M0a. Interaction headroom is measured. The base success rate in the scaffold on the training pool is high enough for expert iteration; otherwise apply the fallbacks in Training step 2. |
| **M1 — KV cache** | One cache per loop for R (exact). Variants: last-loop-only cache, and variable depth per token. | Greedy cached output is token-identical to uncached for T ∈ {1, 2, 4} on at least 50 prompts. At T=1 it matches HF `generate`. |
| **M2 — Generation API and speed** | Batched generation; self-speculative decoding (draft with T_d < T, verify with T); remove host syncs; CUDA graphs. | Speculative output is identical to plain decoding. Speedups are synchronized and warmup is excluded. |
| **M3 — Pre-registration** | Freeze budgets, primary comparisons, confirmation IDs and minimum detectable effect from M0 data. | Committed and hashed. |
| **M4 — Expert-iteration SFT** | Teacher-free self-training, as above. | Multi-turn ≥ single-turn on dev. pass@1 at T>1 ≥ at T=1 (otherwise depth is inert; report it). T=1 retention margin holds. Three seeds. |
| **M5 — RL** | Stages A–C. | Primary comparisons on confirmation, with three seeds. Negative results are reported. |

## Open decisions

1. **Teacher for the optional hybrid ablation:** a local model (free,
   reproducible) or an API model (stronger repairs, external version
   dependence). Only needed if that arm runs.
2. **The three FLOP budgets.** Proposed: chosen after M0b's cost
   measurements, before any training.
3. **Fallback to Coder-1.5B** if 0.5B fails the M0 headroom gate.
4. **Confirmation-set cutoff date and size,** set from the minimum detectable
   effect.
5. Whether v2 stays in this repo.
