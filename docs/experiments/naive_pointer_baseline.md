# Ordinary Qwen pointer baseline — 2026-09-10

## Result and scope

The user completed a full 1,000-example test-set evaluation of ordinary Qwen2.5-0.5B-Instruct with the three-shot final-answer prompt. Accuracy was **60/1,000 (6.00%)**. A subsequent read-only audit verified the saved targets, scoring flags, prompts, coverage, counts, and artifact hashes; the assistant did not rerun model inference.

Accuracy is concentrated at depth 1: **22/125 (17.60%)**. Across depths 2–8 it is **38/875 (4.34%)**, close numerically to the 3.85% expectation for uniform guessing over 26 symbols. This is evidence of weak ordinary-model task performance under this particular prompt, not reliable multi-step execution. It does not test the recurrent wrapper, learned intermediate transitions, repair, damage, or transfer.

## Configuration and command

Executed by the user from the repository root with the local venv active:

```bash
python -m scripts.eval.naive_test \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, resolved revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dataset: `data/pointer/seed-17/test.jsonl`, all 1,000 examples, 125 at each depth 1–8; dataset seed 17.
- Prompt: `prompts/pointer_task.txt`, with independent depth-1/2/3 demonstrations; one user message wrapped with Qwen's chat template. The complete prompt text and chat template are preserved in the run summary.
- Inference: CPU, float32, eager attention, batch 1, greedy decoding, one beam, eight new tokens maximum, `use_cache=False`, seed 17, four CPU threads, strict deterministic algorithms. Cached model/tokenizer files; no downloads.
- Scoring: exact final uppercase symbol after decoding and whitespace stripping; unrestricted generation. Invalid responses remain in the denominator.
- Runtime: Python 3.11.8, torch 2.14.0, transformers 5.17.0, tokenizers 0.23.2, huggingface-hub 1.31.0, rich 15.0.0.
- Recorded Git HEAD: `c3cb2493c4a29f4cfbce4151dd95366dbd10d3fa`; the run summary also preserves worktree status and source-file hashes, since HEAD alone may not identify the executed code.
- Tokenizer source matches the model's pinned source. `resolved_tokenizer_revision` is null in the saved metadata; the serialized tokenizer backend hash is recorded instead of claiming a separately resolved tokenizer commit.

## Artifacts

Run directory:

```text
eval/pointer_task/20260910T172432.074426Z-Qwen-Qwen2.5-0.5B-Instruct/
```

`predictions.csv` contains all task/model inputs, responses, token IDs, targets, and scores. `summary.json` is marked `complete` and contains configuration, provenance, timing, and per-depth results. Generated artifacts are excluded from Git; retain the run directory separately when sharing results.

| Item | SHA-256 |
| --- | --- |
| `predictions.csv` | `4d82eecd681273b2129da9e3dd0b2697fd5546acadeb15c694a4100cde3e9eb1` |
| `summary.json` | `20ae1dd94f02442160a48e101e8611f07536e0558cf75153a85bb2d00768f495` |
| Dataset | `a0d70ada0374b3e0b84af5c65c8b14f6f745b04add7c5fe2fab14e76e5ecfac5` |
| Prompt template | `b21beb6d6de45972329143ce2d7dcabe409da97641e691ff9984d6cc474e353f` |
| Tokenizer backend | `e360e6f6266884e6667874f3a7d158ca20c4639e2e1dc80427e1edc2c69a38c7` |

## Accuracy and response diagnostics

The start-copy and first-lookup columns compare the final decoded prediction with the exact start state and first intermediate target respectively. These are output matches, not observations of internal computation. At depth 1, first-lookup matches are exactly the correct answers.

| Requested steps | Correct / 125 | Accuracy | Copied start | Matched first lookup |
| --- | ---: | ---: | ---: | ---: |
| 1 | 22 | 17.60% | 59 | 22 |
| 2 | 6 | 4.80% | 21 | 36 |
| 3 | 5 | 4.00% | 28 | 27 |
| 4 | 5 | 4.00% | 21 | 29 |
| 5 | 5 | 4.00% | 25 | 22 |
| 6 | 7 | 5.60% | 26 | 18 |
| 7 | 5 | 4.00% | 28 | 15 |
| 8 | 5 | 4.00% | 24 | 16 |
| All | 60 / 1,000 | 6.00% | 232 | 185 |

- Start copied on **232/1,000 (23.2%)** examples. This strategy is always wrong in this dataset because the nominal path does not repeat a state.
- First lookup matched on **185/1,000 (18.5%)** examples. For depths 2–8 specifically, **163/875 (18.63%)** responses match the first intermediate state, while only 38 match the requested final state. This suggests a tendency toward one lookup without reliably completing the requested composition, but symbol biases mean individual matches must not be treated as proof of an internal algorithm.
- Strong output-symbol bias: **Z appears 218 times and T 195 times**, together **41.3%** of responses, although they are correct targets only 40 and 39 times respectively. The model is not behaving like a uniform random guesser; the 1/26 reference is a scale comparison, not a fitted behavioral model or formal significance test.
- **997/1,000** responses satisfy the requested single-symbol format. The three invalid responses are truncated apologies, all at the eight-token budget (depths 3, 4, and 8). Thus format failures account for only three of the 940 errors.

All 1,000 dataset IDs occur once in the CSV. Audit recomputation followed the rule tables to verify every intermediate/final label, then independently compared each stripped response with its final target and checked validity, saved prediction, and correct/incorrect flags. All per-depth totals agree with the summary. Every rendered task prompt matches the saved template and that example's inputs; each appears in the saved full model input. Prompt/data hashes and generated-token counts also agree.

## Resource measurements

- Evaluation-loop wall time: **1,046.48 seconds**, approximately **17 minutes 26 seconds**.
- Synchronized generation time: **1,045.12 seconds**.
- Throughput: **0.956 questions/s**, **1.931 generated tokens/s**.
- Prompt tokens: **420,000 total**, 420 per example.
- Generated tokens: **2,018**, including EOS and excluding padding after EOS.
- Model/tokenizer loading: **0.495 seconds** as measured by the evaluator; this is a cached local run, not a download benchmark.

Generation throughput includes prefill and uses cache-disabled inference. Memory was not measured. These numbers do not establish a training or accelerator throughput budget. The console warning about sampling parameters being ignored was observed; this run intentionally used greedy decoding.

## Interpretation and next boundary

The checkpoint has some one-step lookup signal, substantial start-copying and symbol biases, and little evidence of useful multi-step final-answer execution in this setting. The initial three-example failure was insufficient to characterize the full distribution.

This is a single dataset seed, model, prompt, and decoding configuration. The prompt was revised after inspecting the first three test examples, so this is an exploratory baseline rather than an untouched confirmatory test. No full zero-shot run was collected for a paired zero-shot/three-shot comparison. The result does not show that the demonstrations improved accuracy, that textual intermediate reasoning would fail, or that recurrent training cannot learn the task.

Preserve this run as the ordinary-model three-shot baseline. The planned next research milestone remains Stage 1 training with intermediate supervision and evaluation of one-loop/one-transition behavior. No new training or subsequent phase was launched during this audit.
