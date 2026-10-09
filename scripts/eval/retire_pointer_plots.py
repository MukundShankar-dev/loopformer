"""Retire obsolete render exports only after the complete replacement passes.

Raw CSV/JSON/NPZ evidence, model weights, logs and old numeric audits are retained.
Git history retains prior render exports. Never remove arbitrary user directories.
"""
import argparse
import json
from pathlib import Path

from scripts.eval.pointer_task import sha256_file

OLD_DIRECTORIES=(
    'eval/pointer_benchmark/frozen-20261008/independent/plots',
    'eval/pointer_benchmark/architecture-history-20261009/plots',
    'eval/pointer_benchmark/final-full-20261008/independent/plots',
    'eval/pointer_benchmark/shared-number-20261008/independent/plots',
    'eval/pointer_benchmark/final-full-20261008/independent/failure_matrices',
)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'))
    args=parser.parse_args();root=args.input
    if not json.loads((root/'independent_audit.json').read_text())['passed']:raise ValueError('Replacement decision audit failed')
    manifest=json.loads((root/'plots/manifest.json').read_text())
    if manifest['status']!='complete' or len(manifest['figures'])!=15:raise ValueError('Replacement figure set incomplete')
    for figure in manifest['figures']:
        for export in figure['exports'].values():
            if sha256_file(root/'plots'/export['path'])!=export['sha256']:raise ValueError('Replacement export hash differs')
    removed=[]
    for directory in OLD_DIRECTORIES:
        for path in sorted(Path(directory).iterdir()) if Path(directory).exists() else []:
            if path.is_file() and path.suffix in ('.png','.pdf','.svg'):
                removed.append(dict(path=str(path),sha256=sha256_file(path)));path.unlink()
    report=root/'retired_plot_exports.json'
    previous=json.loads(report.read_text()) if report.exists() else []
    report.write_text(json.dumps(previous+removed,indent=2)+'\n')
    print(f'Retired {len(removed)} obsolete render exports; raw evidence preserved.')


if __name__=='__main__':main()
