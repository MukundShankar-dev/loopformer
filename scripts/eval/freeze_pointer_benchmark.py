"""Freeze inference identities before opening a reserved pointer benchmark."""
import argparse
import json
from pathlib import Path

from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.eval.pointer_task import sha256_file


def verify_reserved_result(summary: dict, frozen: dict, expected_count: int) -> None:
    """Refuse reuse of a partial test or a result from another frozen policy/model."""
    if (summary.get('status') != 'complete' or summary.get('local_checkpoint_sha256') != frozen['inference_sha256']
            or summary.get('data_sha256') != frozen['test_data_sha256']
            or summary.get('stop_policy') != 'completion' or summary.get('stop_threshold') != frozen['threshold']
            or summary.get('loops') != frozen['safety_cap'] or summary.get('selected_examples') != expected_count
            or summary.get('limit') is not None or summary.get('test_mode')):
        raise ValueError('Reserved result is incomplete or differs from the frozen full-test protocol')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pointer_benchmark.json'))
    parser.add_argument('--output', type=Path, default=Path('eval/pointer_benchmark/frozen-20261008/freeze.json'))
    parser.add_argument('--verify-result', type=Path, help='Verify a completed full reserved-test summary before reuse')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frozen = {'checkpoint': config['model'], 'inference_sha256': checkpoint_hashes(Path(config['model'])),
        'threshold': config['threshold'], 'safety_cap': config['safety_cap'],
        'selection': 'First predeclared repair seed 83; frozen before reserved test',
        'test_data_sha256': sha256_file(Path(config['existing_data']) / 'test.jsonl')}
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if (any(existing.get(k) != v for k,v in frozen.items()) or
                ('protocol_sha256' in existing and existing['protocol_sha256'] != sha256_file(args.config))):
            raise ValueError('Existing freeze differs; cannot change model or test within this benchmark')
        if args.verify_result:
            manifest = json.loads((Path(config['existing_data']) / 'manifest.json').read_text())
            verify_reserved_result(json.loads(args.verify_result.read_text()), existing, manifest['splits']['test']['count'])
        print('Existing freeze/result verified; no changes.'); return
    if args.verify_result:
        raise ValueError('Need an existing pre-test freeze before reusing a reserved result')
    frozen['protocol_sha256'] = sha256_file(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2) + '\n')
    print(f'Frozen inference identity: {args.output}')


if __name__ == '__main__':
    main()
