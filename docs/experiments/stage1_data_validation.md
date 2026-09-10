# Stage 1 pointer data validation — 2026-09-10

## Scope and result

The data-only milestone is complete. Generated and verified 13,000 seeded pointer examples, with exact intermediate states, single-token answer IDs, balanced depth splits, and no repeated whole mapping table within/across splits. No model loading, optimizer step, training, or model-output evaluation was performed. Gate 1 (learned stepwise execution) remains untested.

The [Stage 1 guide](../phases/stage1_pointer.md) owns commands, APIs, schema, and sampling details; [decisions](../decisions.md#stage-1-data-resolutions--2026-09-10) records their interpretation.

## Configuration and artifacts

- Location: `data/pointer/seed-17/` (generated artifacts, excluded from Git).
- Files: `train.jsonl`, `validation.jsonl`, `test.jsonl`, `depth_test.jsonl`, and `manifest.json`.
- Master seed: 17, with independently derived per-example seeds recorded in every row.
- Vocabulary: A–Z, 26 full-table rules per prompt, space-prefixed symbolic tokens.
- Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`, loaded from cache only; raw text, no chat template or added special tokens.
- Runtime: Python 3.11.8; Transformers 5.17.0; tokenizers 0.23.2; huggingface-hub 1.31.0; Rich 15.0.0. No new dependencies.
- Source provenance: base Git HEAD `a6de68653d8ff58d1eecb26b5e70fb2fe138992d` plus uncommitted implementation. The manifest records source SHA-256 hashes, worktree status, exact command, and package versions; HEAD alone does not identify this change.

| Split | Count | Depths | Examples per depth | Prompt token count |
| --- | ---: | --- | ---: | --- |
| Train | 10,000 | 1–8 | 1,250 | 143 |
| Validation | 1,000 | 1–8 | 125 | 143 |
| Test | 1,000 | 1–8 | 125 | 143 |
| Depth test | 1,000 | 9–16 | 125 | 143–144 |

Table size stays fixed across depths. The extra token in some deep prompts comes from the two-digit requested step count. All nominal trajectories have d+1 distinct states including the start. Unused edges are random and may cycle outside the nominal path.

## Exact commands and checks

Originally executed from the repository root, before packaging the generator in `scripts/dataset/` (these historical commands match the preserved manifest; the current entry point is `python -m scripts.dataset`):

```bash
.venv/bin/python -m scripts.generate_pointer --dry-run
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.generate_pointer --seed 17
.venv/bin/python -m scripts.generate_pointer --verify data/pointer/seed-17
```

Results:

- **59 tests passed in 8.80 seconds**, with no skips: 36 existing architecture tests plus 23 pointer-data cases. The pretrained Stage 0 CPU/MPS CLI was not rerun; model mechanics were unchanged.
- Both five- and ten-example CLI dry runs were tested via subprocesses. Each printed the requested number of examples and left its temporary output location untouched.
- Exact interpreter check: `(A,C) (C,D) (D, E)`, start A, two steps produces `[C, D]`; per-step prediction checking accepts those answers and rejects incorrect symbols or prose.
- Mapping tests cover full source coverage, in-vocabulary destinations, missing rules, duplicate sources, malformed syntax, cycles in the general interpreter, and non-repeating generated paths at depths 1, 2, 8, 16, and 25.
- Tokenizer validation checks all 676 pair contexts and all start/answer symbols. Every saved prompt is checked for exact symbol/token spans; every saved target ID and readout position is verified.
- Generated JSONL files were read back after writing. Validation recomputed targets by parsing prompt rules, checked split counts/depths, enforced whole-table uniqueness, and replayed every record from the master/derived seed configuration.
- Small-dataset tests establish byte-identical JSONL regeneration, stable existing prefixes when counts grow, split independence, and changed mappings under a different master seed.
- Corruption tests reject changed file bytes and incorrect labels even when the file checksum is updated. Existing output directories are refused.

Default artifact SHA-256 hashes (also in the manifest):

| File | SHA-256 |
| --- | --- |
| `train.jsonl` | `1c360049a6fcd0d2afa0ea8ce5d809c0a45735339790f450e6af053c2c0f690e` |
| `validation.jsonl` | `d5925d830c330808bc5df636018ef0728e39b376ca63fa2df2b8b6fce1355b74` |
| `test.jsonl` | `a0d70ada0374b3e0b84af5c65c8b14f6f745b04add7c5fe2fab14e76e5ecfac5` |
| `depth_test.jsonl` | `20c92ce8159e895cd98e7cee2a2da46e01e0ecdf07457e58d91ec31134ead4c9` |

## Packaging and exact reproduction check — 2026-09-10

Moved all dataset modules and the CLI to `scripts/dataset/`, with entry point `python -m scripts.dataset`. The [README](../../README.md#pointer-dataset-preview-and-reproduce) now lists every seed-17 generation setting, runtime/tokenizer requirements, distribution, preview commands, and file comparison instructions.

The original five-example preview showed depths 1, 1, 1, 9, and 2 because it interleaved the beginning of each split. This was a preview-selection artifact. The current five-example preview selects real dataset records at depths 1, 4, 8, 12, and 16; both preview sizes show the complete configured distribution separately. Full-dataset generation and JSONL serialization remain unchanged.

Validation after packaging: **63 tests passed in 10.63 seconds**, with no skips. Added coverage for distinct preview records spanning the depth range, tiny split counts, and packaged CLI generation with correctly located source provenance. The CLI integration test wrote only a small temporary dataset.

The following additional read-only check replayed the explicit README configuration and compared the serialized result with the existing files. All 13,000 records and all four JSONL byte streams matched; SHA-256 values remain those recorded above. No full dataset was written again. The original `manifest.json` also remained unchanged, with SHA-256 `5f2ed9c2c824ceb4739038140ecfdda523c81010ee30c7255336e7282a80138a`.

```bash
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
from transformers import AutoTokenizer
from scripts.dataset.dataset import DatasetConfig, SPLITS, encode_records, generate_dataset, verify_dataset
from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION, validate_symbols
output = Path('data/pointer/seed-17')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
manifest = verify_dataset(output, tokenizer)
config = DatasetConfig(seed=17, train_count=10000, validation_count=1000, test_count=1000, depth_test_count=1000, min_depth=1, max_train_depth=8, max_eval_depth=16)
regenerated = generate_dataset(config, tokenizer, validate_symbols(tokenizer))
for split in SPLITS:
    payload = encode_records(regenerated[split])
    assert payload == (output / f'{split}.jsonl').read_bytes(), split
    print(f'PASS {split}: {len(regenerated[split])} rows, byte-identical, SHA-256 {hashlib.sha256(payload).hexdigest()}')
assert hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest() == '5f2ed9c2c824ceb4739038140ecfdda523c81010ee30c7255336e7282a80138a'
print('PASS original manifest unchanged; no dataset files written')
PY
```

New runs correctly produce a new manifest with current source paths/hashes and command provenance. Exact dataset reproduction refers to the four JSONL files; it does not fabricate the original run's manifest provenance.

## Interpretation and limits

This validates the data and reference targets, not learned execution. Dataset size is a configurable starting budget; training sufficiency, model accuracy, gate thresholds, and training memory remain unmeasured. JSONL order cycles through depths; the future trainer must choose explicit seeded shuffling and mixed-depth masking.

The nominal path is conditioned not to repeat, so this does not test arbitrary cyclic execution. The maximum supported unique-path depth is 25. Whole tables are disjoint across splits, but individual edges/subpaths can recur; the deeper test measures held-out depth within the pointer family, not wholly novel subpaths or cross-family transfer.

Completion means executing exactly the requested number of lookups. There are no targets beyond nominal depth or labels requiring h_0 to decode to the start. Continuing the mapping after completion can legitimately leave the final symbol; Stage 2/3 must define post-completion behavior before treating that as damage or adding retention training.

Reproduction requires the recorded code/configuration and compatible tokenizer/runtime, not just a seed. Manifest provenance can differ across output locations or Git states even when generated JSONL bytes match. The current tool does not promise cross-version byte stability or provide a streaming large-scale data pipeline.
