# Shared number reader — 2026-10-08

The user requested fixing generalization without increasing training counts or
rollout lengths. The larger benchmark located the failure before recurrence:
the flattened suffix reader has independent weights for each token position.
Replace only that reader; retain the selected seed-83 executor, bridge, affine
countdown and stop readout bitwise. No new recurrent optimization is selected.

## Intervention and supervision

Use one shared linear token-to-scalar projection and one learned scalar
accumulation gain. Scan the tokens of the raw Steps field from left to right:
`s = gain * s + projection(token_embedding)`, starting from zero. The same
parameters operate at every digit position and for every number length. This
explicit positional-composition bias replaces the independent suffix slots.
The tokenizer must represent each decimal character as one token; fail rather
than silently accepting an incompatible tokenizer. Field routing extracts text
and tokenizes it, without converting its numeric value. Padding masks contain
token validity only. Neither a numeric lookup table nor a loop clock is supplied
at inference. The configured limit is eight characters, not arbitrary integers.

Fit on **exactly** the prior 58 training values: 1–63 excluding 9/17/29/41/53.
The executor's original training ceiling is unchanged; the controller's previous
free-running supervision ceiling remains 12. No new graph or numeric label enters
fitting. Existing count labels supervise the initializer, as in the old reader.
No digit-value labels are added. First learn the gain by least squares from
two-token training strings whose individual token strings also occur as
one-token training examples. Then fit the shared projection against all training
count labels. Neither the decimal radix nor a decrement is installed by code.
The trained countdown and threshold 0.5 are copied unchanged.

This is an engineered counting architecture with supervised numerical labels;
it is not evidence for an emergent general reasoning controller or an execution
correctness detector. A shared recurrence can extrapolate format length, but
floating-point drift and finite precision still bound reliability.

## Evaluation declared before fitting

Keep the old failed benchmark as evidence. Freeze the new checkpoint after its
training-label fit. Use the shared benchmark evaluator on fresh seeds
**281/283/293**, 150 graphs per seed/mode, all counts 1–256, and the unchanged
quality thresholds and safety cap 272. Reject overlap with every seed-61 split
and the opened benchmark graph tables. Native fidelity includes counts around
69/70 and 99/100 and the largest request. Report every count, all three graph
modes, nominal executor trajectory, exact stopping and joint success. Correct
letters at wrong loops remain failures. No threshold or checkpoint selection
uses these results. Existing seed-61 test is already opened, so a rerun would
be a regression check, not fresh confirmation.

Reader-only integer requests through 10,000 and controller-only rollouts through
4,096 are additional diagnostics, without training or retuning. Distinguish
reading error from countdown drift. These do not establish executor quality at
those depths. Do not claim indefinite extrapolation or extend a favorable range
after looking at results.

## Commands and implementation

```bash
bash repair_number_reader.sh --dry-run
bash repair_number_reader.sh
```

Run on the desktop with the complete old checkpoint, training dataset, old run
metadata/train panel and opened benchmark graph file. The launcher fits only the
reader, exports the normal full checkpoint, freezes inference hashes, runs the
fresh benchmark, numeric-only diagnostics, then an independent saved-result audit.
The audit traverses raw graphs and recomputes every flag/aggregate without the
inference scorer or its reference interpreter. It refuses existing outputs.
W&B stays disabled; artifacts are saved under
`eval/pointer_benchmark/shared-number-20261008/`. Model files stay under
`models/stage1_pointer/controller-shared-number-seed83/` and are ignored by Git.
The shared `loop_test` and `naive_test` loaders support the new checkpoint.
The latter still forces the requested depth and does not test stopping.

Model mechanics are in `scripts/recurrent_qwen/number_reader.py` and the normal
controller factory. Fitting is in `scripts/training/number_reader.py`, composed
by `repair_number_reader`. The existing frozen benchmark is reused, including
native checks, graph clustering and cyclic coincidence exclusions. Numeric-only
diagnostics never run R and must not be confused with longer pointer evaluation.
All old artifacts remain preserved. Results are recorded in the
[experiment report](experiments/controller_number_reader.md).
