import json
from pathlib import Path
import sys

import pytest

from scripts.eval import retire_pointer_plots
from scripts.eval.pointer_task import sha256_file


def test_retirement_requires_verified_replacement_and_preserves_raw_evidence(tmp_path,monkeypatch):
    root=tmp_path/'new';(root/'plots').mkdir(parents=True)
    old=tmp_path/'old';old.mkdir()
    (old/'old.png').write_bytes(b'old render');(old/'raw.csv').write_text('raw evidence')
    (old/'weights.pt').write_bytes(b'model bytes')
    monkeypatch.setattr(retire_pointer_plots,'OLD_DIRECTORIES',(str(old),))
    monkeypatch.setattr(sys,'argv',['retire','--input',str(root)])
    (root/'independent_audit.json').write_text(json.dumps({'passed':True}))
    (root/'plots/manifest.json').write_text(json.dumps({'status':'incomplete','figures':[]}))
    with pytest.raises(ValueError):retire_pointer_plots.main()
    assert (old/'old.png').exists()
    entries=[]
    for i in range(21):
        p=root/'plots'/f'{i}.png';p.write_bytes(str(i).encode())
        entries.append({'exports':{'png':{'path':p.name,'sha256':sha256_file(p)}}})
    (root/'plots/manifest.json').write_text(json.dumps({'status':'complete','figures':entries}))
    retire_pointer_plots.main();retire_pointer_plots.main()
    assert not (old/'old.png').exists()
    assert (old/'raw.csv').read_text()=='raw evidence' and (old/'weights.pt').read_bytes()==b'model bytes'
    assert len(json.loads((root/'retired_plot_exports.json').read_text()))==1


def test_duplicate_curves_are_named_once_without_coordinate_jitter(monkeypatch,tmp_path):
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.eval.plot_paper_analysis import grouped_lines
    fig,ax=plt.subplots();x=np.arange(1,4);y=np.array([1.,.5,.5])
    grouped_lines(ax,x,[('ce_full',y,'CE'),('joint_full',y.copy(),'Joint')])
    assert len(ax.lines)==1 and 'CE / Joint (identical)' in ax.lines[0].get_label()
    assert np.array_equal(ax.lines[0].get_xdata(),x) and np.array_equal(ax.lines[0].get_ydata(),y)
    plt.close(fig)


def test_original_code_guard_allows_unused_new_module_and_rejects_original_edits(tmp_path,monkeypatch):
    from scripts.eval import checkpoint_comparison
    directory=tmp_path/'scripts/recurrent_qwen';directory.mkdir(parents=True)
    original=directory/'model.py';original.write_bytes(b'original inference')
    (directory/'new_unused.py').write_text('new helper')
    monkeypatch.chdir(tmp_path)
    def git_read(args,**kwargs):
        return 'scripts/recurrent_qwen/model.py\n' if args[1]=='ls-tree' else b'original inference'
    monkeypatch.setattr(checkpoint_comparison.subprocess,'check_output',git_read)
    assert set(checkpoint_comparison.verify_recurrent_code('snapshot'))=={'scripts/recurrent_qwen/model.py'}
    original.write_text('changed inference')
    with pytest.raises(ValueError,match='implementation changed'):
        checkpoint_comparison.verify_recurrent_code('snapshot')
