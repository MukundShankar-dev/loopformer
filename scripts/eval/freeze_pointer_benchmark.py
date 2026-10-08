"""Freeze inference identities before opening a reserved pointer benchmark."""
import argparse
import json
from pathlib import Path

from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.eval.pointer_task import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pointer_benchmark.json'))
    parser.add_argument('--output', type=Path, default=Path('eval/pointer_benchmark/frozen-20261008/freeze.json'))
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
        print('Existing freeze verified; no changes.'); return
    frozen['protocol_sha256'] = sha256_file(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2) + '\n')
    print(f'Frozen inference identity: {args.output}')


if __name__ == '__main__':
    main()
