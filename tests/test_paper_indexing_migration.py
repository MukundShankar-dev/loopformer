import json
from pathlib import Path
import sys

import pytest

from scripts.eval import migrate_paper_indexing as migration


def test_substitution_rejects_arithmetic_changes():
    original = (migration.EVIDENCE/'original_partitioned.py').read_text()
    current = Path(migration.CHANGED_FILE).read_text()
    migration.validate_substitution(original, current)
    with pytest.raises(ValueError, match='exceeds'):
        migration.validate_substitution(original, current.replace('dropout=0.0', 'dropout=0.1'))


def migration_fixture(tmp_path, monkeypatch):
    original = (migration.EVIDENCE/'original_partitioned.py').read_bytes()
    current = Path(migration.CHANGED_FILE).read_bytes()
    monkeypatch.chdir(tmp_path)
    evidence = tmp_path/'evidence'; evidence.mkdir()
    (evidence/'original_partitioned.py').write_bytes(original)
    (evidence/'indexing.json').write_text(json.dumps([
        {'model': n, 'bitwise_scores_equal': True, 'bitwise_stops_equal': True}
        for n in ('ce_full', 'fixed6')]))
    monkeypatch.setattr(migration, 'EVIDENCE', evidence)
    monkeypatch.setattr(migration.subprocess, 'check_output', lambda *a, **k: '')
    source = Path(migration.CHANGED_FILE); source.parent.mkdir(parents=True); source.write_bytes(current)
    graph = tmp_path/'graphs.jsonl'; graph.write_text('graph bytes')
    weights = tmp_path/'model'; weights.mkdir(); (weights/'weights.pt').write_bytes(b'original weights')
    config = {'models': [{'name': 'ce_full', 'path': str(weights)}], 'graphs': str(graph)}
    Path('configs').mkdir(); Path('configs/pointer_analysis_protocol.json').write_text(json.dumps(config))
    root = tmp_path/'out'; directory = root/'ce_full'; directory.mkdir(parents=True)
    chunk = directory/'graphs-0000-0004.npz'; chunk.write_bytes(b'completed chunk')
    frozen = {'protocol': config, 'graph_batch': 4, 'dtype': 'float32', 'attention': 'sdpa',
              'graph_sha256': migration.digest(graph),
              'checkpoint_sha256': {'weights.pt': migration.digest(weights/'weights.pt')},
              'code_sha256': {migration.CHANGED_FILE: migration.digest(evidence/'original_partitioned.py')}}
    (directory/'freeze.json').write_text(json.dumps(frozen))
    monkeypatch.setattr(sys, 'argv', ['migrate', '--input', str(root)])
    return root, directory, chunk, weights


def test_migration_preserves_source_provenance_and_completed_chunks(tmp_path, monkeypatch):
    root, directory, chunk, weights = migration_fixture(tmp_path, monkeypatch)
    old = (directory/'freeze.json').read_bytes(); chunk_hash = migration.digest(chunk)
    migration.main(); migration.main()
    assert (directory/'freeze-pre-single-index.json').read_bytes() == old
    assert migration.digest(chunk) == chunk_hash
    report = json.loads((root/'indexing_migration.json').read_text())
    assert report['records'][0]['retained_chunk_sha256']['ce_full/graphs-0000-0004.npz'] == chunk_hash
    assert (weights/'weights.pt').read_bytes() == b'original weights'


def test_migration_refuses_changed_weights_before_writing(tmp_path, monkeypatch):
    root, directory, chunk, weights = migration_fixture(tmp_path, monkeypatch)
    old = (directory/'freeze.json').read_bytes()
    (weights/'weights.pt').write_bytes(b'different weights')
    with pytest.raises(ValueError, match='Checkpoint'):
        migration.main()
    assert (directory/'freeze.json').read_bytes() == old
    assert not (directory/'freeze-pre-single-index.json').exists()
