"""Verify original inference implementation and frozen source before trajectory reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.eval.pointer_task import sha256_file


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'));args=parser.parse_args()
    protocol=json.loads(Path('configs/pointer_analysis_protocol.json').read_text())
    source=Path(protocol['source_results']);summary=json.loads((source/'summary.json').read_text())
    checkpoint=Path(protocol['models'][-1]['path'])
    if checkpoint_hashes(checkpoint)!=summary['checkpoint_sha256']:raise ValueError('Original source checkpoint changed')
    if summary['dtype']!='float32' or summary['attention']!='sdpa' or summary['batch_size']!=64 or summary['torch']!=torch.__version__:
        raise ValueError('Source inference precision/library/batch differs')
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',summary['git_head'],'scripts/recurrent_qwen'],text=True).splitlines()
    hashes={}
    for name in paths:
        if not name.endswith('.py'):continue
        original=subprocess.check_output(['git','show',f'{summary["git_head"]}:{name}'])
        expected=hashlib.sha256(original).hexdigest()
        if not Path(name).exists() or sha256_file(Path(name))!=expected:raise ValueError(f'Original recurrent implementation changed: {name}')
        hashes[name]=expected
    manifest=json.loads((source/'dataset_manifest.json').read_text())
    if manifest['graphs_sha256']!=sha256_file(Path(protocol['graphs'])):raise ValueError('Source graph bytes differ')
    result=dict(passed=True,original_recurrent_code_sha256=hashes,source_summary_sha256=sha256_file(source/'summary.json'),
                source_graphs_csv_sha256=sha256_file(source/'graphs.csv'),source_decisions_sha256=sha256_file(source/'decisions.csv.gz'),
                original_checkpoint_sha256=summary['checkpoint_sha256'],new_partitioned_module='New historical optimization; not used by isolated source or replay')
    args.input.mkdir(parents=True,exist_ok=True);(args.input/'source_reuse_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Original source and {len(hashes)} original recurrent modules verified unchanged.')


if __name__=='__main__':main()
