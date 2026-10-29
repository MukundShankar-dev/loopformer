# Plan — v2

Status: draft. Nothing below is implemented except where marked **ready**.
v1 is finished and summarized in [history](history.md); its results are not
re-validated against v2 code.

## Goal

Post-train a looped small code model with execution-based rewards. Ask whether
extra recurrent depth raises the pass rate more cheaply than the alternatives:
more samples, or written reasoning before the code.

1. **Q1 — Depth as test-time compute.** At matched inference FLOPs, does T
   loops beat T=1 with best-of-k, and T=1 with written reasoning?
2. **Q2 — Learned depth.** Can RL with a compute penalty learn when to loop
   more (per problem, then per token), and beat fixed T at equal mean compute?
   Unlike v1, the right depth cannot be read off the prompt.
3. **Q3 — Inference.** Can drafts made with fewer loops of the same weights
   speed up exact full-depth decoding?

Out of scope: non-code tasks, natural-language transfer claims.

## Benchmark selection rule

Use benchmarks where **computation, not knowledge, is the bottleneck**, that
are **recent enough to limit contamination**, and where the base model has
**measured headroom**. For RL, headroom means problems solved sometimes but not
always in n samples. Never-solved problems give no gradient, and always-solved
ones have nothing left to gain.

| Benchmark | Role | Why |
| --- | --- | --- |
| LiveCodeBench, problems after the base model's release | Main dev and confirmation eval | Algorithmic, dated, contamination-controlled. Small windows mean wide intervals, about ±4–5 points. |
| HumanEval+ / MBPP+ (EvalPlus) | Sanity and regression only | Old and likely in pretraining; gains there persuade no one |
| CRUXEval | Diagnostic: does depth help simulate code? | The natural successor to v1's pointer tracing |
| BigCodeBench-Hard | Probe only, if cheap | Mostly tests library knowledge, which loops can't add. Heavy environment. Small models are likely near the floor. |
| Training pool: LiveCodeBench problems before the cutoff, plus easy tiers of KodCode, TACO or APPS | RL and SFT data | Must have executable tests that pass in our sandbox; deduplicate against every eval set |

## Milestones

Each milestone ends with a gate. Do not start the next until it passes or the
user changes scope.

### M0 — Headroom probe (ready, not run)

Before any model work, measure whether the benchmarks above have headroom.

- **Models:**
  - Qwen2.5-Coder-0.5B-Instruct (default base);
  - Qwen2.5-Coder-1.5B-Instruct (fallback if 0.5B is at the floor);
  - Qwen2.5-0.5B-Instruct (v1's base, as a control).
- **Runs:** the official tools generate and grade, so numbers are comparable
  with published ones. Each model gets greedy pass@1 plus 16 samples at
  temperature 0.8.
- **Output:** per benchmark and difficulty, the always/sometimes/never-solved
  fractions and unbiased pass@1/4/16.
- **Prepared:**
  - `configs/probe_headroom.json`;
  - `scripts/codeprobe/commands.py`, which prints or runs the steps and refuses
    to run with unpinned revisions, no sandbox, or an existing output;
  - `scripts/codeprobe/headroom.py`, the summary;
  - `requirements-probe.txt`, for a separate `.venv-probe`;
  - tests in `tests/test_codeprobe.py`.
- **Before running, check:**
  - pin the two Coder revisions;
  - choose the sandbox and set `sandbox_prefix`;
  - check every tool flag in the `--dry-run` output against the installed
    versions. The flags and result layouts follow tool docs and are unverified;
  - confirm the latest LiveCodeBench `release_version`, and that the runner
    accepts these model IDs (it may need them registered);
  - LiveCodeBench may write to its own `output/` directory;
  - on Blackwell GPUs, check whether the runner needs vLLM.
- **Gate:**
  - Base-model HumanEval+ and MBPP+ greedy scores are within a few points of
    the Qwen2.5-Coder report. Otherwise the harness is wrong.
  - Choose the model and benchmarks from the measured headroom. Proposed
    threshold, to be decided by the user before seeing results: a sometimes-solved
    fraction of at least 20%, and pass@16 − pass@1 of at least 10 points.

### M1 — KV-cached recurrent generation

v1 runs with `use_cache=False`, which costs O(sequence² × loops) per generated
sequence and blocks multi-token work.

- **(a) One cache per loop for R's layers.** Exact. Memory grows with T for R
  only. Ship this first, as the reference.
- **(b) Last-loop-only cache.** Cheaper and approximate; needs training.
- **(c) Variable depth per token**, reading the last available loop's entries
  for skipped loops. Needed by M5 Stage C.

Gate:
- Greedy cached generation is token-identical to uncached generation for
  T ∈ {1, 2, 4} on at least 50 prompts in fp32.
- At T=1 it matches Hugging Face `generate`.
- Synchronized throughput and peak memory are recorded.

### M2 — Generation API and inference optimizations

- **Batched `generate(prompt, loops, sampling)`.**
- **Self-speculative decoding:** draft k tokens with T_d < T loops, then verify
  them in one T-loop pass. Greedy verification is exact; a token-identity test
  proves it. Report acceptance rate by T_d, k and dataset, and wall-clock speedup.
- **Remove host syncs inside the loop.** v1 has `.tolist()` and tensor rebuilds
  per layer per loop in `prefix.py` and `model.py`.
- **CUDA-graph the T-loop decode step.**

Gate: speculative output is identical to plain decoding; speedups are
synchronized, warmup is excluded, and the timing scope is stated.

### M3 — Evaluation harness for the looped model

The official tools can't run a custom architecture, so we generate with our
`generate` and grade with the official graders.

- **Sandbox:** no network, read-only root, temporary working directory,
  CPU/wall/memory/process limits. Outcomes are
  `pass / fail / timeout / runtime_error / syntax_error / sandbox_error`, and
  sandbox errors are never counted as fails. An adversarial suite (fork bomb,
  infinite loop, memory hog, file and socket access) must be contained.
- **Metrics:**
  - pass@1 greedy, and pass@k from n ≥ 2k samples;
  - compute per result: tokens, loops per token, FLOPs derived from layer
    counts;
  - main plot: pass rate vs FLOPs for fixed T, T=1 best-of-k, and T=1 with
    written reasoning;
  - paired bootstrap over problems, with seed spread reported separately.
- **Splits:**
  - dev: EvalPlus sets plus a held-out training slice;
  - confirmation: a hashed list of post-cutoff LiveCodeBench problem IDs,
    frozen before M4 and opened once per final checkpoint.

Gate: the looped model at T=1, before training, matches the M0 base-model
numbers through our generation path.

### M4 — Looped SFT warm start

- Verified solutions, deduplicated against all eval sets. T is sampled per
  batch from {1, 2, 4}, and only code tokens are supervised.
- Ablate the training scope in this milestone: R only vs LoRA on R vs full
  model.
- Retention: dev pass@1 at T=1 may not drop more than a pre-declared margin.
  Add a short non-code check whose answer format avoids tokens the training
  rewires (v1's MMLU lesson).

Gate: pass@1 at T > 1 is at least pass@1 at T=1, the retention margin holds,
and there are three seeds.

### M5 — RL with test rewards

A minimal GRPO-style loop in this repo. Standard trainers' rollout engines
(vLLM) cannot run the looped model; reuse their loss definitions.

- **Reward:** all-pass or fraction-passed, chosen on dev.
- **Stage A:** fixed T=1 and T=4. Do RL gains grow with depth?
- **Stage B:** the policy also chooses T from a small menu, and the reward is
  reduced by λ·compute. Compare at equal mean compute.
- **Stage C:** per-token depth. Only if Stage B beats fixed T.
- **Resources:** measure 0.5B full-parameter RL with group size 8 on the 16 GB
  GPU, with gradient checkpointing. Reserve CPU cores for sandbox workers.

Gate: three seeds, and a confirmation-set comparison at equal mean compute.
Negative results are reported.

## Baselines

1. Base model at T=1: greedy and best-of-k.
2. Base model with written reasoning, at matched FLOPs.
3. SFT without looping, on M4's data and steps.
4. Looped model at fixed T (M4, M5 Stage A).
5. Re-initialized R at matched parameters, to test whether pretraining matters.
6. The non-Coder Qwen2.5-0.5B-Instruct.

## Open decisions

1. Sandbox on WSL2: Docker with `--network none`, or `bubblewrap`.
2. M0 headroom thresholds, fixed before the probe runs.
3. LiveCodeBench date cutoff and confirmation-set size.
4. Whether written-reasoning baselines may use a different prompt format.
5. Same repo or a fresh one that vendors `scripts/recurrent_qwen`.
