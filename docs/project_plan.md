# Qwen2.5-0.5B Recurrent Reasoning Project

## Research and implementation plan

> **Core question:** Can a small looped language model learn a reusable state-transition mechanism whose recurrent dynamics remain useful when overscaled, and how far does that mechanism transfer across different in-context algorithms?

This project has **two separate research questions**. They should not be mixed in the first experiment.

1.  **Dynamics:** Can recurrent Qwen learn a useful update rule where extra loops can repair wrong states, while already-correct states become hard to damage?
2.  **Generality:** Once that recurrent update rule works, is it specific to one synthetic task, or does it behave more like a reusable in-context state-transition mechanism across different algorithms?

The first question is the core project. The second question is the natural expansion if the first succeeds.

---

# 1\. Research thesis

We are **not** trying to prove that Qwen has learned a general reasoning step.

The testable thesis is:

> A middle block of a pretrained Qwen can be reused recurrently and post-trained so that:
> 
> *   unsolved states remain capable of useful progress and repair;
> *   solved states are much less likely to be damaged by extra loops;
> *   the recurrent hidden state does not need to freeze;
> *   later, the same recurrent update can be tested for transfer across multiple in-context transition systems.

The important distinction is:

```
stable decoded solution !== frozen hidden state
```

A solved problem does not need one special hidden vector. Hidden states may continue moving. What matters is that they remain in a region that still decodes safely to the correct answer.

---

# 2\. Model choice

Start from:

```
Qwen/Qwen2.5-0.5B-Instruct
```

Use:

```
PyTorch
Hugging Face Transformers
PEFT / LoRA
```

Reproduce the recurrent mechanism in PyTorch/Hugging Face, where it is easy to inspect and debug.

## Default recurrent split

Use a configurable split, with the initial default:

```
layers 0-5      frozen prelude/front
layers 6-17     recurrent middle block
layers 18-23    frozen coda/back
```

Conceptually:

```
prompt
  ↓
frozen prelude P
  ↓
recurrent middle R × T
  ↓
frozen coda C
  ↓
answer logits
```

The **same recurrent module** must be reused at every loop.

Do not instantiate separate copies for loop 1, loop 2, loop 3, etc.

Formally:

\[  
h\_0 = P(x)  
\]

\[  
h\_{t+1} = R(h\_t)  
\]

\[  
\\text{logits}\_t = C(h\_t)  
\]

A bridge / re-entry module may be added if needed, but the architecture must preserve the interpretation that the same recurrent transition rule is being reused.

---

# 3\. What gets trained

Initially freeze the original Qwen parameters.

Train only:

*   LoRA adapters inside the recurrent middle block;
*   a small bridge / re-entry module if required.

Keep frozen initially:

*   token embeddings;
*   prelude/front;
*   original recurrent-block weights;
*   coda/back;
*   final norm;
*   LM head.

This is **post-training a recurrent update rule on top of pretrained Qwen**, not training Qwen from scratch.

Later ablations may unfreeze the recurrent block, but that is not the first experiment.

---

# 4\. Stage 0 — recurrent architecture validation

Before doing any research training, prove that the surgery itself is correct.

## Required behavior

Before any LoRA or bridge training:

```
num_loops = 1
```

should reproduce the ordinary Qwen forward pass to numerical tolerance.

This is a hard acceptance criterion.

## Required tests

Implement tests for:

**T=1 equivalence**

*   recurrent wrapper matches base Qwen logits before training.

**Weight sharing**

*   all recurrent iterations use the same module object / parameter set.

**Gradient scope**

*   only LoRA + bridge parameters receive gradients.

**Intermediate observability**

*   the model can return:
    *   recurrent hidden state `h_t` after every loop;
    *   logits after every loop;
    *   answer margins after every loop when labels are provided.

**Configurable recurrent depth**

*   evaluation can request arbitrary `T`.

Initially use:

```python
use_cache = False
```

Do not optimize KV caching until recurrent behavior is correct.

## Stage 0 gate

Do not proceed until:

```
T=1 equivalence passes
shared-weights test passes
gradient-scope test passes
```

---

# 5\. Project Question 1 — can we learn good recurrent dynamics?

The first experiment should stay narrow.

Use **one clean synthetic task family** where one recurrent loop has an unambiguous meaning.

## Stage 1 — install one-step recurrence with pointer chasing

Example:

```
Rules:
A -> F
F -> C
C -> Q

Start: A
```

Desired recurrent behavior:

```
loop 0: A
loop 1: F
loop 2: C
loop 3: Q
```

The purpose of this task is not to claim generality.

The purpose is to prove:

```
1 loop ≈ 1 computational transition
```

and to make recurrent depth interpretable.

## Data requirements

For every example:

*   generate a fresh random mapping;
*   use a fixed pool of tokenizer symbols that are guaranteed to be single tokens;
*   provide the correct intermediate state after every step;
*   vary task depth;
*   hold out deeper compositions for evaluation;
*   avoid fixed symbol-to-symbol mappings that the weights could simply memorize.

The rule should live in the **prompt**, not in the model weights.

## Stage 1 training objective

If the task requires:

```
A -> F -> C -> Q
```

then supervise:

```
loop 1 -> F
loop 2 -> C
loop 3 -> Q
```

Do **not** supervise every loop directly toward the final answer.

Otherwise the model may learn:

```
A -> Q
```

as a shortcut instead of learning iterative execution.

## Stage 1 success criteria

We should see:

*   deeper tasks generally require more loops;
*   intermediate-step accuracy is meaningful;
*   test-time extra loops can extend computation;
*   depth generalization can be measured cleanly.

If depth-2 and depth-8 tasks are both solved at the same shallow recurrent depth, treat that as possible shortcut behavior.

---

# 6\. Stage 2 — establish overthinking before fixing it

Once the recurrent mechanism works, stop training and deliberately run it too long.

For a task of nominal depth `d`, evaluate with:

```
T = 1, 2, 3, 4, 6, 8, 16, 32, 64
```

Track every transition:

```
wrong -> wrong
wrong -> right
right -> right
right -> wrong
```

The untreated recurrent model must ideally exhibit **both**:

```
wrong -> right
```

and:

```
right -> wrong
```

If there is no useful repair and no real overthinking, there is nothing interesting to fix yet.

## Core metrics

At recurrent step `t`, define:

### Repair rate

\[  
R\_t = P(\\text{correct at }t+1 \\mid \\text{wrong at }t)  
\]

### Damage rate

\[  
H\_t = P(\\text{wrong at }t+1 \\mid \\text{correct at }t)  
\]

### Accuracy

\[  
A\_t  
\]

### Net recurrent gain

\[  
G\_t = (1-A\_t)R\_t - A\_t H\_t  
\]

Positive `G_t` means another recurrent step helps overall.

This metric matters because when accuracy is already high, even a small damage rate can outweigh a much larger repair rate.

---

# 7\. Central figure — problem depth × recurrent depth

Treat **problem depth** and **recurrent depth** as separate variables.

Build a heatmap like:

```
                recurrent loops
problem depth   1   2   4   8   16   32   64
------------------------------------------------
1
2
4
8
12
16
24
32
```

This should become one of the main project figures.

It answers:

### Shallow task + huge loop count

Does a solved example survive massive overthinking?

### Deep task + too few loops

Does the model genuinely need more computation?

### Deep task + enough loops

Does test-time recurrent compute improve accuracy?

### Task depth beyond training

Did the model learn an iterative mechanism that extrapolates?

Train initially around:

```
task depth <= 8
recurrent depth <= 8
```

and test both axes beyond the training horizon.

---

# 8\. Answer margins and solved-state regions

For recurrent state `h_t`, pass it through the frozen coda/back.

Restrict evaluation to the allowed symbolic answer set.

Define:

\[  
margin\_t =  
score(correct)  
\-  
\\max(score(any\\ wrong\\ allowed\\ answer))  
\]

Choose configurable `gamma > 0`.

Classify:

## Wrong

\[  
margin\_t \\le 0  
\]

## Fragile correct

\[  
0 \< margin\_t \< \\gamma  
\]

## Robust correct

\[  
margin\_t \\ge \\gamma  
\]

The goal is **not**:

\[  
h\_{t+1} \\approx h\_t  
\]

The goal is:

```
if solved robustly:
    stay in a region that still decodes safely
```

Hidden states are allowed to continue moving.

---

# 9\. Stage 3 — asymmetric recurrent-dynamics training

Only after Stage 2 proves the failure mode exists should we add the proposed fix.

This stage tests the main dynamics hypothesis:

> Can the transition itself become less dangerous without killing useful repair?

## Detached rollout

Generate a sampled number of recurrent steps with gradients disabled:

```
no_grad:
h_0 -> h_1 -> ... -> h_t
```

Detach `h_t`.

Then recompute one additional recurrent step with gradients enabled:

```
with_grad:
h_t -> h_{t+1}
```

Train only that next transition.

Interpretation:

> Put the model into a state generated by its own recurrent dynamics, then teach it what one good next move from that state should do.

This directly trains the transition rule on the states the recurrent model actually visits.

---

# 10\. Stage 3 loss

## Case A — current state wrong

If:

\[  
margin\_t \\le 0  
\]

and the nominal program has not finished:

*   train the next loop toward the known **next intermediate state**;
*   preserve the one-step algorithm.

Use:

\[  
L\_{wrong} =  
CE(\\text{next-loop prediction}, \\text{next intermediate target})  
\]

If the model has already gone beyond nominal task depth and has drifted wrong:

*   use the final answer as the recovery target.

## Case B — fragile correct

If:

\[  
0 \< margin\_t \< \\gamma  
\]

push the next state farther into the safe region:

\[  
L\_{fragile}  
\=  
\\max(0,\\gamma-margin\_{t+1})  
\]

## Case C — robust correct

If:

\[  
margin\_t \\ge \\gamma  
\]

do **not** keep sharpening confidence forever.

Only penalize if the next state leaves the safe region:

\[  
L\_{retain}  
\=  
\\max(0,\\gamma-margin\_{t+1})  
\]

This is a dead-zone loss.

Once:

\[  
margin\_{t+1} \\ge \\gamma  
\]

the retention loss is zero.

The hidden state is free to keep moving.

---

# 11\. No-op failure mode

A trivial way to reduce damage is to make recurrence inert:

\[  
R(h) \\approx h  
\]

That is **not** success.

A no-op block can achieve:

```
right -> right
```

while also producing:

```
wrong -> wrong
```

Therefore the desired result is:

\[  
H \\downarrow  
\]

while:

\[  
R  
\]

remains useful.

We should also preserve meaningful intermediate-step execution.

A useful result looks like:

```
baseline:
repair = 25%
damage = 8%

asymmetric:
repair = 22-25%
damage = 1-3%
```

A bad result looks like:

```
repair = 2%
damage = 0%
```

if that stability comes from doing almost nothing.

---

# 12\. Stage 3 baselines / ablations

Compare at least:

| Version | Training state | Step / repair loss | Retention loss |
| --- | --- | --- | --- |
| A | normal differentiable unroll | yes | no |
| B | detached model-generated state | yes | no |
| C | detached model-generated state | ordinary task loss | yes |
| D | detached model-generated state | yes | yes |

Version D is the full asymmetric method.

Also include inference-time halting baselines:

*   fixed maximum loop count;
*   confidence / margin stopping;
*   prediction-stability stopping;
*   later, optionally probe-based stopping.

Conceptual comparison:

```
Halting:
    avoid dangerous transitions

Our method:
    make the transitions themselves less dangerous
```

---

# 13\. Stage 3 success condition

The main dynamics result is:

```
repair remains useful
damage falls sharply
net recurrent gain stays non-negative farther beyond training depth
```

especially at:

```
16 / 32 / 64 loops
```

The model must also beat a trivial no-op recurrent block.

At this stage we are still **not claiming general recurrent reasoning**.

If Stage 3 succeeds only on pointer chasing, that is still a valid result about recurrent dynamics.

---

# 14\. Project Question 2 — how general is the recurrent operation?

Only after the single-task recurrent mechanism and asymmetric dynamics work should we broaden the task distribution.

Do **not** start multi-family training and asymmetric training simultaneously.

Otherwise failure becomes impossible to diagnose.

The next goal is to test whether one shared recurrent block can behave roughly like:

\[  
(\\text{rule in prompt}, \\text{current state})  
\\rightarrow  
\\text{next state}  
\]

across superficially different algorithms.

This is the beginning of the **in-context interpreter** hypothesis.

Do not call it a general interpreter. Treat that as an empirical direction.

---

# 15\. Stage 4 — multi-family recurrent execution

Train the **same LoRA + bridge** across multiple task families.

Do not create task-specific adapters.

Each family should expose the same interface.

Suggested Python abstraction:

```python
class TransitionTask:
    family_name: str
    task_depth: int
    initial_state: object
    intermediate_states: list
    final_state: object

    def render_prompt(self) -> str:
        ...

    @classmethod
    def generate_example(cls, rng, config):
        ...
```

Every task must provide exact intermediate targets.

---

# 16\. Multi-family task suite

## Family A — pointer lookup

```
A -> D
D -> G

state = A
next = D
```

## Family B — random permutations

Generate a fresh bijection over the state vocabulary.

One loop applies the permutation once.

## Family C — finite-state machines

```
(S2, B) -> S5
```

The next state depends on:

```
current state + current input
```

## Family D — symbolic conditional transitions

```
(A, left)  -> C
(A, right) -> F
(C, left)  -> B
```

## Family E — graph or grid navigation

Example:

```
J north -> P
J east  -> C
P north -> Q
```

One loop executes one edge traversal.

A coordinate version is also possible:

```
state = (2,4)
rule = RIGHT
next = (3,4)
```

## Family F — modular / simple state updates

```
rule: x -> (x + 3) mod 16
state: 7
next: 10
```

If tokenizer behavior makes numeric values awkward, encode numeric states using the same validated single-token symbolic vocabulary.

---

# 17\. Data design for generality

The training data should make memorization as unattractive as possible.

Randomize per example:

*   symbols;
*   transition tables;
*   graph structure;
*   operation labels where possible;
*   start states;
*   task depth;
*   prompt formatting.

The rule should live in the prompt.

The weights should ideally learn something closer to:

```
read active rule
read current state
execute one transition
```

rather than:

```
memorize a fixed procedure
```

---

# 18\. Stage 4 training

Train one shared recurrent adapter over a mixture of task families.

Example configuration:

```
family_mix:
  pointer: 0.25
  permutation: 0.20
  fsm: 0.20
  symbolic_transition: 0.20
  modular: 0.15
```

The exact weights are tunable.

Important:

```
one recurrent parameter set
one bridge
one LoRA configuration
all families
```

No per-family recurrent weights.

Continue intermediate supervision so one loop remains tied to one transition.

---

# 19\. Stage 4b — ordinary-Qwen anchor loss

Do **not** add this during the very first pointer experiment.

First establish the recurrent phenomenon with as few moving parts as possible.

Once synthetic training becomes broader, periodically sample ordinary text/prompts and compare:

```
frozen original Qwen
vs.
adapted recurrent Qwen at normal one-pass behavior
```

Optional loss:

\[  
L\_{anchor}  
\=  
D\_{KL}(p\_{base} \\parallel p\_{adapted})  
\]

Purpose:

> Learn the recurrent capability without gratuitously changing ordinary pretrained behavior.

Track this separately as:

```
ordinary-Qwen behavior degradation
```

The anchor loss is a preservation tool, not part of the core recurrent-dynamics claim.

---

# 20\. Stage 5 — whole-family holdout

This is the real generality test.

Example:

Train on:

```
pointer lookup
modular update
symbolic transition
finite-state machines
```

Never recurrently train on:

```
graph / grid navigation
```

Then evaluate graph navigation zero-shot.

Ask whether:

```
loop 1 = one valid move
loop 2 = second valid move
loop 3 = third valid move
```

emerges without recurrent training on that family.

If it does, that is evidence for **cross-algorithm transfer**.

This is much stronger than unseen pointer instances or greater pointer depth.

---

# 21\. Generalization hierarchy

Do not jump levels when describing results.

| Test | Claim supported |
| --- | --- |
| unseen random mappings in a trained family | instance generalization |
| depth 32 after training depth \<= 8 | depth generalization |
| several trained task families | multi-task recurrent execution |
| entirely held-out task family | cross-algorithm transfer |
| unseen natural-language task descriptions | instruction / surface-form transfer |
| GSM8K or real reasoning benchmark | real-world reasoning transfer |

A model that passes the first three does **not** automatically justify the last two claims.

---

# 22\. Stage 6 — natural-language transfer

Only attempt this after the synthetic recurrent mechanism is understood.

Possible progression:

```
synthetic formal rules
    ↓
same rules described in natural language
    ↓
natural-language algorithm instructions
    ↓
small structured reasoning tasks
    ↓
GSM8K / other real benchmarks
```

Do not jump directly from pointer chasing to GSM8K and treat failure as evidence that recurrence itself failed.

Natural-language reasoning introduces many extra unknowns:

*   one human reasoning step may not align with one recurrent loop;
*   intermediate states may be ambiguous;
*   output decoding may be the bottleneck;
*   arithmetic and language understanding can interfere with the recurrent mechanism.

Stage 6 is aspirational.

It is **not** required for the core project to succeed.

---

# 23\. Evaluation suite

Track the following for every stage where applicable.

## Core dynamics

*   accuracy at each loop;
*   repair rate `R_t`;
*   damage rate `H_t`;
*   net recurrent gain `G_t`;
*   first-correct loop;
*   survival after first solution;
*   answer margin trajectories.

## Computational behavior

*   intermediate-step accuracy;
*   task depth vs. first-correct recurrent depth;
*   task-depth × recurrent-depth heatmap;
*   extrapolation beyond trained loop count;
*   extrapolation beyond trained task depth.

## Generality

*   trained-family performance;
*   held-out instance performance;
*   held-out depth;
*   held-out compositions;
*   alternate surface forms;
*   entire held-out task families.

## Preservation

*   ordinary-Qwen behavior before and after recurrent training;
*   anchor-loss effect;
*   one-pass degradation.

## Failure diagnostics

*   no-op tendency;
*   representation collapse;
*   runaway confidence sharpening;
*   loop-to-loop oscillation;
*   family-specific specialization.

---

# 24\. Survival-after-solution analysis

For each example define:

```
T* = first loop at which the answer becomes correct
```

Then measure:

```
how many extra loops remain correct after T*
```

Plot:

\[  
P(\\text{still correct after } k \\text{ extra loops})  
\]

This directly measures whether the solved region has become robust under repeated recurrence.

This should be one of the central figures alongside the depth × loops heatmap.

---

# 25\. Probe experiments

Treat probes as diagnostics, not required training machinery.

For each hidden state `h_t`, test whether a simple predictor can answer:

1.  **Am I correct now?**
2.  **Would another loop help?**
3.  **Would another loop hurt?**
4.  **Which task family / transition type am I executing?**

Suggested probes:

*   output entropy / margin;
*   linear probe on `h_t`;
*   small MLP probe on `h_t`.

The fourth question is mechanistically interesting.

If task family is trivially separable and each family appears to occupy a distinct “mode,” the model may have learned several task-specific mechanisms.

If states instead organize around more generic factors such as current state, active rule, and transition progress, that would be more consistent with the interpreter story.

Do not force this representation during training. Use it as evidence.

---

# 26\. Repo structure

Use a research-code layout, not a monolithic notebook.

```
tasks/
    base.py
    symbols.py
    pointer.py
    permutation.py
    fsm.py
    symbolic_transition.py
    graph.py
    modular.py

training/
    stage1_pointer.py
    detached_rollout.py
    asymmetric_loss.py
    multifamily.py
    anchor_loss.py

eval/
    transitions.py
    metrics.py
    depth_loop_sweep.py
    survival.py
    family_transfer.py
    probes.py
    plots.py

tests/
    test_t1_equivalence.py
    test_shared_weights.py
    test_gradient_scope.py
    test_task_generators.py
    test_intermediate_targets.py
    test_margin.py

configs/
    stage0.yaml
    stage1_pointer.yaml
    stage2_overthinking.yaml
    stage3_asymmetric.yaml
    stage4_multifamily.yaml
    stage5_holdout.yaml

scripts/
    recurrent_qwen/
        model.py
        bridge.py
        lora_utils.py
        outputs.py
        validation.py
    validate_stage0.py
    train_pointer.py
    evaluate_overthinking.py
    train_asymmetric.py
    train_multifamily.py
    evaluate_holdout_family.py
    evaluate_all.py
```

---

# 27\. Logging requirements

Every run should save enough information to reconstruct recurrent transitions later.

For each evaluated example record at minimum:

```
example_id
seed
family
task_depth
nominal_final_state
loop_index
current_prediction
correct_target_for_this_loop
final_answer
margin
is_correct
next_prediction
transition_type
```

Where `transition_type` is one of:

```
wrong->wrong
wrong->right
right->right
right->wrong
```

Also save:

*   model checkpoint hash;
*   task generator config;
*   recurrent depth config;
*   LoRA config;
*   gamma;
*   random seeds;
*   package versions.

Reproducibility matters more than throughput.

---

# 28\. MacBook constraints

Target hardware:

```
32 GB Apple Silicon MacBook Pro
```

Design around that constraint.

Use:

*   short prompts;
*   small symbolic vocabularies;
*   tiny batches;
*   gradient accumulation;
*   recurrent training depths around 4-8 initially;
*   16/32/64 loops mostly for `no_grad` evaluation;
*   frozen base weights;
*   lightweight LoRA / bridge training.

Remember:

```
small number of trainable parameters != small compute
```

Every extra loop still runs the reused middle chunk of Qwen.

---

# 29\. Stage gates

Do not move forward just because the code runs.

## Gate 0 — architecture

Required:

*   T=1 equivalence;
*   shared recurrent weights;
*   correct gradient scope.

## Gate 1 — stepwise mechanism

Required:

*   one loop corresponds meaningfully to one pointer transition;
*   deeper pointer tasks generally need more loops;
*   intermediate supervision is working.

## Gate 2 — failure exists

Required:

*   some wrong->right repair under extra loops;
*   some right->wrong damage under overscaling.

If damage never appears, do not add a retention fix yet.

## Gate 3 — asymmetric dynamics

Required:

*   damage decreases substantially;
*   repair remains useful;
*   model does not collapse to a no-op;
*   net recurrent gain remains useful farther past training depth.

## Gate 4 — multi-family execution

Required:

*   one shared recurrent parameter set handles several trained task families;
*   one-loop ≈ one-transition behavior remains interpretable.

## Gate 5 — cross-family transfer

Required for the stronger generality claim:

*   a wholly held-out family exhibits meaningful zero-shot recurrent progression.

## Gate 6 — natural-language transfer

Exploratory only.

---

# 30\. Claim hierarchy

Use disciplined language.

## Level 1 — recurrent dynamics result

> Asymmetric post-training reduces damage from excess recurrent computation while preserving useful repair on a controlled iterative task.

## Level 2 — multi-task recurrent execution

> One shared recurrent update rule can execute several trained in-context transition systems.

## Level 3 — cross-algorithm transfer

> The learned recurrent transition mechanism transfers to an entire held-out transition family.

## Level 4 — instruction / natural-language transfer

> The recurrent mechanism survives a substantial change in how rules are represented.

## Level 5 — real reasoning transfer

> Extra latent recurrent compute helps on natural-language reasoning tasks not used to train the recurrent mechanism.

Do not claim a higher level from evidence that only supports a lower one.

---

# 31\. What would count as a good result?

## Good dynamics result

Compared with detached-rollout training without retention:

*   repair remains similar;
*   right->wrong damage falls sharply at 16-64 loops;
*   net recurrent gain stays non-negative farther beyond training depth;
*   the recurrent block still performs useful computation;
*   a no-op baseline is worse.

## Good generality result

After multi-family training:

*   one shared adapter handles multiple task families;
*   recurrent depth tracks computational depth;
*   at least one completely held-out family shows non-trivial zero-shot recurrent progression.

## Useful negative results

Any of these are informative:

*   stability only appears when the recurrent block becomes inert;
*   halting performs as well as or better than asymmetric training;
*   multi-family training destroys clean stepwise recurrence;
*   transfer works across depth but not across task family;
*   held-out-family transfer is absent;
*   the recurrent adapter strongly damages ordinary Qwen behavior;
*   synthetic transition training does not transfer to natural-language task descriptions.

These outcomes should be reported as failures of stronger hypotheses, not hidden by aggregate accuracy.

---

# 32\. Immediate implementation order for Codex

Do **not** implement the entire project at once.

Work in this order:

## First task

1.  inspect the local repository and environment;
2.  inspect the Hugging Face Qwen2.5-0.5B-Instruct model structure;
3.  identify the exact classes and forward arguments used by its decoder layers;
4.  propose the recurrent wrapper architecture;
5.  implement only Stage 0;
6.  write the Stage 0 tests;
7.  run them;
8.  stop and report the result.

Do not proceed to pointer training until Stage 0 is verified.

## Second task

Implement the symbolic vocabulary and pointer-chasing generator.

Add tests for:

*   single-token symbols;
*   mapping validity;
*   exact intermediate targets;
*   train/test depth splits.

Stop again before training.

## Third task

Implement Stage 1 pointer training and the depth × loop evaluator.

Only after that works should the project move to overthinking, asymmetric training, or multi-family tasks.

---

# 33\. Codex operating instructions

While implementing this project:

*   do not silently simplify the research design;
*   do not replace intermediate supervision with final-answer-only training;
*   do not instantiate separate weights per recurrent loop;
*   do not add multi-family training before the single-family mechanism is validated;
*   do not add asymmetric retention before overthinking has been empirically observed;
*   do not add the anchor loss in the first pointer experiment unless a concrete failure makes it necessary;
*   do not optimize KV caching before correctness;
*   do not claim general reasoning from synthetic success;
*   if a technical constraint forces a design change, explain the scientific consequence before changing it.

The priority order is:

```
scientific interpretability
> correctness
> reproducibility
> optimization
```

---

# 34\. One-line summary

> **Teach a small recurrent Qwen to execute one useful in-context state transition per loop, first study and stabilize its recurrent dynamics on a clean task, then test how far that same update mechanism transfers across different algorithms without becoming a no-op or destroying ordinary model behavior.**