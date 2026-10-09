"""One explicit provenance migration for bitwise-equivalent active-row indexing.

Run only after stopping the old extractor. Retain old freeze/source and chunk
hashes; permit exactly the tested indexing substitution, no other source,
checkpoint, data, dtype or batch change. Normal resume checks remain strict.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


EVIDENCE = Path('eval/pointer_analysis/performance-20261009')
CHANGED_FILE = 'scripts/recurrent_qwen/partitioned.py'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_substitution(original: str, current: str) -> None:
    """Reject every AST change except the independently profiled row selection."""
    marker = '            if not need.any():'
    if original.count(marker) != 1:
        raise ValueError('Original active-row branch differs')
    expected = original.replace(marker, '            kept = need.nonzero().flatten()\n            if not kept.numel():')
    expected = expected.replace('[need]', '.index_select(0, kept)')
    if ast.dump(ast.parse(expected)) != ast.dump(ast.parse(current)):
        raise ValueError('Source change exceeds the approved indexing substitution')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('eval/pointer_analysis/paper-20261009'))
    args = parser.parse_args()
    running = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
    if any((line.split() and Path(line.split()[0]).name.startswith('python') and
            ('-m scripts.eval.paper_suite' in line or '-m scripts.eval.paper_snapshots' in line))
           and str(args.input) in line for line in running):
        raise ValueError('Stop the old extractor before migrating its resume identity')
    original, current = EVIDENCE/'original_partitioned.py', Path(CHANGED_FILE)
    validate_substitution(original.read_text(), current.read_text())
    proof = json.loads((EVIDENCE/'indexing.json').read_text())
    if {r['model'] for r in proof} != {'ce_full', 'fixed6'} or not all(
            r['bitwise_scores_equal'] and r['bitwise_stops_equal'] for r in proof):
        raise ValueError('Required full-horizon CUDA bitwise checks did not pass')
    config = json.loads(Path('configs/pointer_analysis_protocol.json').read_text())
    planned = []
    for arm in config['models']:
        directory = args.input/arm['name']; freeze = directory/'freeze.json'
        if not freeze.exists():
            continue
        frozen = json.loads(freeze.read_text())
        if frozen['code_sha256'][CHANGED_FILE] == digest(current):
            continue
        if frozen['protocol'] != config or frozen['graph_batch'] != 4 or frozen['dtype'] != 'float32' or frozen['attention'] != 'sdpa':
            raise ValueError('Protocol, batch, precision or attention identity differs')
        if frozen['graph_sha256'] != digest(Path(config['graphs'])):
            raise ValueError('Graph bytes differ')
        for path, expected in frozen['checkpoint_sha256'].items():
            if digest(Path(arm['path'])/path) != expected:
                raise ValueError('Checkpoint bytes differ')
        for path, expected in frozen['code_sha256'].items():
            actual = digest(original if path == CHANGED_FILE else Path(path))
            if actual != expected:
                raise ValueError(f'Unexpected prior source identity: {path}')
        archive = directory/'freeze-pre-single-index.json'
        if archive.exists():
            raise ValueError('An earlier migration archive already exists')
        chunks = {str(p.relative_to(args.input)): digest(p) for p in sorted(directory.glob('graphs-*.npz'))
                  if '.tmp.' not in p.name}
        planned.append((freeze, archive, frozen, chunks))
    records = []
    for freeze, archive, frozen, chunks in planned:
        shutil.copyfile(freeze, archive)
        frozen['code_sha256'][CHANGED_FILE] = digest(current)
        temporary = freeze.with_suffix('.migration.json')
        temporary.write_text(json.dumps(frozen, indent=2)+'\n'); temporary.replace(freeze)
        records.append(dict(model=freeze.parent.name, original_freeze=str(archive.relative_to(args.input)),
                            original_freeze_sha256=digest(archive), new_resume_freeze_sha256=digest(freeze),
                            retained_chunk_sha256=chunks))
        for partial in freeze.parent.glob('graphs-*.tmp.npz'):
            interrupted = freeze.parent/'interrupted-chunks'; interrupted.mkdir(exist_ok=True)
            partial.rename(interrupted/partial.name)
    if records:
        report = args.input/'indexing_migration.json'
        report.write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            reason='Discover ordered active rows once; bitwise same FP32 scores/stops in both tested recurrence modes',
            old_source_sha256=digest(original), new_source_sha256=digest(current), proof_sha256=digest(EVIDENCE/'indexing.json'),
            precision_and_batch_unchanged=True, archived_chunks_produced_by_original_source=True, records=records), indent=2)+'\n')
    print(f'Migrated {len(records)} resume identities; retained old provenance and every completed chunk')


if __name__ == '__main__':
    main()
