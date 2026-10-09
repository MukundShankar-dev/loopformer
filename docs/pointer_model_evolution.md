# How the pointer models evolved

This explains the actual architectures used in the
[paired historical comparison](experiments/pointer_architecture_history.md).
There are three early configurations, a broader-training variant, and two examples
from the later isolated-executor regime. Calling every later reader repair a new
transformer architecture obscures the important changes.

## Common task and model backbone

The public input is one text prompt containing Rules, Start and Steps. A rule
`(A,C)` means that one transition from A reaches C. For requested count N, correct
execution must produce the exact reference state at every loop t through N, and
an autonomous solver must first stop at N. A correct letter at the wrong loop
fails completion, even if a cycle makes that letter equal the final answer.

Every Qwen model uses the pinned Qwen2.5-0.5B-Instruct checkpoint, 24 transformer
layers, hidden width 896, and the same partition:

```text
P: layers 0–5     R: layers 6–17     C: layers 18–23
```

P prepares an initial representation once. The same R parameters are reused on
all loops. C, the final normalization and Qwen's tied LM head produce a letter
readout after each R loop. **The next R loop receives the continuous R state,
not C's output and not the predicted letter.** Readouts are restricted to the
validated 26 symbol tokens. This is latent recurrence, not autoregressive generation
of a written reasoning chain.

Training uses supervised reference execution. At loop t, cross-entropy targets
the symbol after exactly t transitions. Loss masks remove loops after an example's
requested N. Earlier hidden states remain in the autograd graph; later losses
can train earlier R operations. The training interpreter knows t/N and reference
states for labels, but does not feed those labels back into the model.

The ordinary unmodified Qwen generation baseline is different: it consumes the
three-shot task prompt and generates a final letter. Its 6% result is historical
context, not one of the recurrent architectures or a directly matched inference
procedure in these plots.

## 1. Full-sequence recurrence with pointer CE only

**Technical account.** The complete `[batch, sequence, 896]` P output initializes
`h_0`. Every loop applies `h_t = R(h_(t-1))` to all token positions. Consequently,
representations of Rules, Start, Steps and the answer position all evolve together.
The original causal mask and RoPE positions are reused. P and C are frozen, as
are original R weights; rank-8 q/v LoRA in R supplies 270,336 trainable parameters.
There is no re-entry bridge, direct R readout or stopping head.

The run uses 30,000 seed-37 tables conditioned to have nonrepeating nominal paths,
requested depths 1–6, fresh adapters and training seed 17. Its recipe is FP32,
batch 4/accumulation 2, one epoch/3,750 updates, learning rate 0.0002 after ten
warmup updates, and per-example mean intermediate CE. The comparison uses update
3,250, the trained-depth validation-CE selection. The historically stronger
step-2,500 extrapolation checkpoint is documented separately; the comparison does
not retrospectively select it using new outcomes.

Inference requires an externally chosen loop budget. For an execution test,
force N loops and read C at N. This measures whether R can carry out N transitions;
it provides **no evidence of learned stopping**. Stopping cells are therefore N/A,
not zero or 100%.

**Plain-language account.** Imagine a person repeatedly updating the entire page:
the question, lookup table and their working notes can all change internally.
We teach the next correct answer at each turn, but we tell them when the turns
end. They have not been taught a separate decision to finish.

**What motivated the next change.** The model learned familiar transitions but
lost reliability shortly beyond trained depths. We wanted prompt-only completion,
where the model itself decides when the requested work is done. That motivated a
learned stopping readout; it did not establish that missing stopping caused R's
execution failure.

An intervening [equal-loop loss ablation](experiments/stage1_loopbalanced.md)
gave later supervised loops more balanced dataset-level weight, without changing
this architecture or training ceiling. It did not extend the observed frontier.
Its binaries were retired at the user's request, so it is documented historical
evidence rather than another checkpoint in the new paired plots.

## 2. Full-sequence recurrence with a jointly trained completion head

**Technical account.** The same full-sequence recurrence and LoRA are retained.
A completion MLP reads R's answer-position vector `w_t`: LayerNorm, a
896→128 linear layer, GELU, and a 128→1 linear layer. Sigmoid gives a stop
probability. The head has no separate recurrent memory and receives neither t
nor a parsed N. To track progress it must decode information in R's working state,
which also depends on the full prompt, including Steps.

The first comparison holds the data, adapter initialization seed, optimizer,
3,750-update budget and pointer-CE selection rule fixed. The head adds balanced
continue/stop BCE with coefficient 0.1: continue at t<N, stop at t=N, and no loss
after N. Both pointer loss and stop loss backpropagate into R's LoRA. This adds
116,737 head parameters, for 387,073 trainable parameters in total. A classifier
outside the block is therefore not isolated from executor learning.

Forced inference ignores the stop decision and exposes R's trajectory. Autonomous
inference stops at the **first** probability ≥0.5 under a count-independent cap,
then returns C's current letter. Training always executes the supervised nominal
rollout even if the provisional head would stop early.

**Plain-language account.** We keep the same changing page and add someone who
looks at the current notes and says “continue” or “finished.” This person has no
private notebook. Their training also changes how the original worker writes its
notes, so learning to finish can alter learning to do the lookup.

**Observed limitation and next rationale.** Historical development evaluation
found poorer forced-depth extension than matched CE-only training and premature
stops beyond six. This rules out early stopping as the whole explanation, because
R still failed when the stop head was ignored. The next architecture separates
read-only question information from writable working state. These observations
do not identify gradient interference or erased rule memory as a proven cause.

## 3. Fixed prompt memory with one recurrent working position

**Technical account.** P still encodes the full prompt, including Steps. The final
unmasked answer position becomes the writable 896-dimensional working vector.
All preceding positions are fixed, layer-specific prompt memory. Each R layer
reads its original first-pass prefix activations and the current working vector;
only the working output advances. C similarly reads fixed layer-appropriate memory.

Conceptually:

```text
w_t = R(w_(t-1), fixed prompt memory)
letter_t = C(w_t, fixed prompt memory)
stop_t = MLP(w_t)
```

Memory is fixed within a forward, but its LoRA-dependent activations remain
connected to gradients and are recomputed after updates. The original version
executes dense layers and restores fixed prefixes; it is not a KV-cache design.
There is still no bridge, explicit clock, decoded-state feedback or separate
controller memory. The same completion MLP and coupled BCE gradients remain.

The depth-6 run preserves the preceding joint run's seed-37 data, 1–6 requested
counts, 3,750-update recipe and CE-selected update 3,250. Trainable parameter
count is unchanged. Inference has the same forced and first-threshold-stop modes.
The changed component is where recurrence may write and what prompt context each
layer sees on later loops.

**Plain-language account.** We pin the question and lookup table to the desk and
let the worker update one working note. The worker can consult the same original
information on every turn. The “finished?” reader still looks at that working
note and is still trained together with the worker.

**Observed limitation and next rationale.** Historical results improve over the
joint full-sequence variant, but still collapse at later loops and stop early
on unfamiliar counts. Fixed memory alone did not solve execution or completion.
It also changes the size of writable state, so any benefit cannot be assigned
uniquely to preserving the rule table.

## 3b. The depth-12 run is a training change, not a new architecture

This keeps fixed memory, the single working position, LoRA and the coupled MLP.
It trains fresh on 30,000 seed-47 examples at requested counts 1–6/8/10/12,
holding out requested counts 7/9/11. Intermediate positions 7/9/11 still receive
supervision inside longer tasks; they are held-out **requested values**, not
unseen recurrent ages.

The recipe switches to BF16/SDPA, batch 4 without accumulation and depth-grouped
updates; one epoch is 7,500 updates. Learning rate remains 0.0002 after ten warmup
updates. Trained-count CE selected step 5,000. Historical checkpoint comparisons
showed extension improved relative to depth-6 training but degraded later in the
run; increasing training length did not produce stable long-range execution.

In plain language: the worker and notes are unchanged; we gave them longer
practice problems. That moves the failure boundary and changes numerical exposure,
but it is not evidence that the underlying problem was solved.

The [same-map count probe](experiments/stage1_depth12_paired_counts.md) also found
that changing only Steps from 18 to 20 changed correct loop-14 readouts from
30/32 to 8/32. The state that should be reached at loop 14 is identical for both
requests. This directly motivated removing requested count from the executor's
view; it does not explain all later failures or prove a unique internal mechanism.

## 4. Isolated executor, full R training and a separate GRU controller

**Technical account.** The public prompt is routed into two views. The executor
receives Rules/Start/Answer with the Steps line removed. Compacted attention and
RoPE positions prevent digit count or original suffix length from becoming an
executor side channel. The GRU receives a frozen full-prompt P feature including
Steps, and detached actual R working vectors. Requested count is thus available
to control while deliberately absent from R.

Fixed prompt memory remains. Before loops two and later, a learned re-entry bridge
normalizes the working vector by its RMS, restores the initial P working vector's
RMS scale, then applies a learned per-channel gain and 896×896 linear projection.
It does not decode a state or insert a reference answer. The first loop bypasses
it to preserve the startup one-loop gate.

All original R weights are trainable instead of q/v LoRA alone. P, C, embeddings
and LM head remain frozen. C's intermediate CE is supplemented by a learned
896→26 direct R readout with CE coefficient 0.25. That direct prediction is
another supervised observation, never the next-loop input. The seed-61 executor
run trains one epoch/2,250 updates on depth-independent graph sampling with cycles,
counts 1–6/8/10/12, batch 4/accumulation 4, BF16/SDPA, learning rate 0.00002,
100-update warmup, cosine decay and weight decay 0.01. It has about 180.1 million
trainable parameters including its interfaces/controller. This is full **R** SFT,
not full-model SFT or random initialization.

The controller initializes 128-dimensional memory from a LayerNorm/linear/tanh
projection of the full-prompt P feature. At each executor loop, a second projection
of detached R state feeds a shared 128-wide GRUCell; a linear readout emits stop.
Controller gradients cannot reach R, and its state/output never feeds back into R.
Its private memory can represent progress without requiring R to represent it.

The plotted checkpoint is the later `controller-prefix-seed83/best` pilot, not the
original joint executor-run controller. With the successful step-2,250 executor
frozen, it fits only controller parameters for 6,000 updates, batch 256, using
256 training graphs and the same 58 requested values through 63 excluding
9/17/29/41/53. Supervised rollouts cover only min(N,12) loops. It adds numerical
remaining-work labels N−t, including initialization, and weighted stop BCE.
These are targets, never inputs or teacher-forced memory. Trained-count development
selection chose update 5,800. Despite broader requested-value exposure and stronger
supervision, timing remained poor.

**Plain-language account.** The worker sees the lookup table and start, and is
asked to do one more lookup on each turn. A separate manager sees the full request
and keeps a private notebook. The manager can observe the worker but cannot change
its notes or training. We also give the worker much more freedom to learn: the
entire middle block can adapt, and it is taught intermediate states directly.

**What this does and does not establish.** Execution becomes much stronger while
control can still fail. Routing, memory, bridge, full-R capacity, losses, graph
sampling and optimizer changed together, so this comparison cannot assign sole
credit to full SFT or the bridge. It does establish a structural separation that
the earlier stateless head did not provide.

## 5. The same frozen executor with a learned scalar timer

**Technical account.** R, bridge, direct readout, P and C are bitwise identical to
model 4. Only the controller changes. The final reader routes the raw Steps digit
tokens without converting them into an integer. Frozen digit-token embeddings
feed a shared learned linear scalar projection. A learned accumulation gain
combines valid tokens left to right, conceptually `m = gain*m + token_projection`.
The gain and projection are fitted on the existing 58 numeric labels; the observed
gain near ten is learned rather than assigned. Eight characters and finite
precision bound this mechanism.

A learned shared scalar cell `m_(t+1) = a*m_t + b` updates once per executor loop.
A separate affine readout determines the first stop. It ignores R's observation,
so it measures requested work, not correctness or uncertainty. Numerical N−t
supervision fits the cell on free-running nominal rollouts through min(N,12).
The original affine controller used AdamW; the final precision repair fits only
a/b with float64 L-BFGS on those same short labels. Exported countdown arithmetic
is float32; reader accumulation uses float64 before returning the scalar dtype.
There is no supplied per-loop counter, assigned decrement or gold memory re-entry.
This is strong task-specific counting structure plus numerical supervision.

Autonomous inference reads Steps once, then alternates one R transition and one
learned timer update until its stop threshold is crossed, or the safety cap
forces a fallback. The executor cannot know how much requested work remains,
and the timer cannot detect an incorrect pointer state.

**Plain-language account.** We keep the successful worker and replace the manager's
flexible notebook with a deliberately simple numerical timer. We teach it how to
read the requested number and how its remaining-work value should change each
turn. It decides when time is up; it does not check whether the worker is right.
That specialization makes counting much easier to extend, at the cost of being
a task-specific controller rather than a general reasoning evaluator.

## Reading the comparison without overclaiming

Forced final-letter accuracy, strict all-step accuracy, learned exact stopping,
and correct-answer-plus-exact-stop success answer different questions. Strict
prefix success never recovers from an earlier error; final-letter accuracy can
recover or benefit from a cycle. No learned-stop metric exists for model 1.

The new comparison keeps graph/count queries identical, but it is an opened
27-graph diagnostic, not the 1,350-graph population benchmark. Older training
required nonrepeating paths; the new benchmark allows cycles and is therefore a
changed graph distribution for those models. Counts and training budgets also
differ across later versions. Successful execution at 256 on 26-state graphs is
finite cyclic-task evidence, not proof of 256 distinct, nonrepeating states or
transfer to another task family. The controller's separate 8,192-request numerical
test does not execute R and does not establish pointer quality at that depth.

The [architecture guide](architecture.md) owns exact implementation contracts;
the [historical comparison report](experiments/pointer_architecture_history.md)
owns checkpoint identities, new results, figures, commands and verification.

## Implementation history

The source changes corroborate the saved checkpoint metadata:

| Git commit | Change |
| --- | --- |
| `a6de686` | Initial shared full-sequence Qwen recurrence and validation |
| `6112b70` | Joint hidden-state completion MLP and losses |
| `e58c3fb` | Fixed layer-specific prompt memory and writable answer position |
| `9b3154d`, `2a9296c` | Sparse depth-12 training recipe and batch-four launcher |
| `d2eae3a` | Isolated controller, re-entry bridge, direct R supervision and full-R training |
| `11a7458`, `03cbc13`, `704bfb8` | Learned affine controller, shared number reading and scalar precision refinement |

These commits record implementation changes, not their empirical success. The
linked experiment reports record what actually ran and what failed.
