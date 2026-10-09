"""Publication-style figures from audited full-panel outcomes; no model loading.

Metric panels and explicit shared-executor labels prevent hidden overlapping
series. Every categorical color has a legend or colorbar. Exports retain the
exact numerical arrays plotted, captions, input hashes and output hashes.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import textwrap

import numpy as np

from scripts.eval.pointer_task import sha256_file
from scripts.eval.pointer_failure_metrics import CATEGORIES
from scripts.eval.plot_pointer_benchmark import save_figure
from scripts.eval.benchmark_metrics import wilson_interval

NAMES={'ce_full':'Full sequence · CE (2500)','joint_full':'Full sequence · CE + stop',
       'fixed6':'Fixed memory · depth 6','fixed12':'Fixed memory · depth 12',
       'gru':'Isolated R + GRU','final':'Isolated R + numeric timer'}
COLORS={'ce_full':'#626e7c','joint_full':'#ce483c','fixed6':'#bd7b16','fixed12':'#8060b0','gru':'#187e81','final':'#2467af'}
STYLES={'ce_full':('-', 'o'),'joint_full':('--','s'),'fixed6':('-.','^'),'fixed12':(':','D'),'gru':('--','v'),'final':('-','P')}
MODES=('random_function','permutation','full_cycle')
MODE_NAMES={'random_function':'Random function','permutation':'Permutation','full_cycle':'26-state cycle'}


def read(path:Path)->list[dict]:
    with path.open() as handle:return list(csv.DictReader(handle))


class Figures:
    def __init__(self,root:Path):
        self.root=root;self.output=root/'plots';self.output.mkdir(exist_ok=True)
        self.numeric=self.output/'numeric';self.numeric.mkdir(exist_ok=True)
        self.entries=[]

    def save(self,fig,group:str,name:str,title:str,caption:str,arrays:dict)->None:
        import matplotlib.pyplot as plt
        directory=self.output/group;directory.mkdir(exist_ok=True)
        fig.suptitle(title,fontsize=15,fontweight='bold',x=.04,y=.99,ha='left')
        fig.text(.04,.015,caption,fontsize=8,color='#45505c',va='bottom')
        fig.tight_layout(rect=(.01,.09,.99,.93))
        save_figure(fig,directory,name);plt.close(fig)
        numeric=self.numeric/f'{name}.npz';np.savez_compressed(numeric,**arrays)
        self.entries.append(dict(group=group,name=name,title=title,caption=caption,
            numeric=str(numeric.relative_to(self.output)),numeric_sha256=sha256_file(numeric),
            exports={ext:dict(path=str((directory/f'{name}.{ext}').relative_to(self.output)),
                             sha256=sha256_file(directory/f'{name}.{ext}')) for ext in ('png','pdf','svg')}))

    def finish(self)->None:
        hashes={p.name:sha256_file(p) for p in (self.root/'reference_audit.json',self.root/'quality_by_depth.csv',self.root/'outcomes.npz',self.root/'fixed_request_loops.csv',self.root/'first_errors.csv',self.root/'structure_failure_rates.csv')}
        (self.output/'manifest.json').write_text(json.dumps(dict(status='complete',figures=self.entries,
            input_sha256=hashes,plot_code_sha256=sha256_file(Path(__file__)),
            overlap_policy='Identical executor curves grouped; separate metric panels; staggered marker locations only, no value jitter',
            coverage='1350 graph panel, all integer requests 1–256 for every architecture'),indent=2)+'\n')
        lines=['# Pointer analysis figures','','All figures use the full 1,350-graph opened panel. Every architecture sees all requested depths 1–256.','','PNG previews, editable SVG and publication PDF have identical numerical sources. `numeric/` and `manifest.json` preserve arrays, captions and hashes.','']
        for e in self.entries:
            lines.extend([f"## {e['title']}",'',e['caption'],'',f"![{e['title']}]({e['group']}/{e['name']}.png)",'',f"[PDF]({e['group']}/{e['name']}.pdf) · [SVG]({e['group']}/{e['name']}.svg)",''])
        (self.output/'index.md').write_text('\n'.join(lines))


def line(ax,x,y,name:str,index:int,label:str|None=None)->None:
    style,marker=STYLES[name]
    # Offset only marker placement, never coordinates or success rates.
    ax.plot(x,y,color=COLORS[name],linestyle=style,marker=marker,markevery=(index*4,31),
            markersize=4,linewidth=1.7,label=label or NAMES[name])


def grouped_lines(ax,x,series:list)->None:
    """Draw exact duplicate series once and name every represented model."""
    groups=[]
    for name,y,label in series:
        match=next((item for item in groups if np.array_equal(item[1],y,equal_nan=True)),None)
        if match is None: groups.append([name,y,[label]])
        else: match[2].append(label)
    for i,(name,y,labels) in enumerate(groups):
        label=textwrap.fill(' / '.join(labels)+(' (identical)' if len(labels)>1 else ''),width=38)
        line(ax,x,y,name,i,label)


def axis(ax,ylabel:str='Graphs (%)')->None:
    ax.set(xlabel='Requested transitions N',ylabel=ylabel,xlim=(1,256),ylim=(-2,102))
    ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)


def quality_figures(figures,arrays,names,g)->None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,LogNorm
    from matplotlib.patches import Patch
    data=lambda name,metric: arrays[f"{name}__{metric}"]
    x=np.arange(1,257)
    curves={f'{name}__{metric}':data(name,metric).mean(0) for name in names for metric in ('nominal_final_correct','complete_trajectory')}
    # Isolated R is genuinely identical, so display one explicitly shared executor curve.
    if not np.array_equal(data('gru','complete_trajectory'),data('final','complete_trajectory')) or not np.array_equal(data('gru','nominal_final_correct'),data('final','nominal_final_correct')):
        raise ValueError('Shared isolated executor results differ')
    fig,axes=plt.subplots(1,3,figsize=(17,5.8))
    for col,metric in enumerate(('nominal_final_correct','complete_trajectory','joint_success')):
        series=[]
        for name in names:
            if metric!='joint_success' and name=='gru':continue
            if f'{name}__{metric}' not in arrays:continue
            label='Shared isolated R (GRU & timer)' if name=='final' and metric!='joint_success' else NAMES[name]
            series.append((name,100*data(name,metric).mean(0),label))
        grouped_lines(axes[col],x,series)
        axis(axes[col]);axes[col].set_title(('Correct forced final letter','Every intermediate letter correct','Correct letter AND exact first stop')[col])
        axes[col].legend(fontsize=7,loc='center right')
    figures.save(fig,'01_quality','architecture_quality','Execution and complete-solver quality',
        'Each depth has 1,350 graphs. Headless CE has no learned-stop score. Shared-executor curves are drawn once.\n'
        'CE uses selected step 2500; architecture, data, capacity and recipes differ, so these are descriptive comparisons.',curves)
    fig,axes=plt.subplots(2,2,figsize=(15,7.5));matrix_arrays={}
    for ax,metric,title in zip(axes.flat,('nominal_final_correct','complete_trajectory','exact_stop','joint_success'),
                              ('Forced final letter','Strict complete trajectory','Exact first stop at N','Final letter + exact first stop')):
        matrix=np.array([data(name,metric).mean(0) if f'{name}__{metric}' in arrays else np.full(256,np.nan) for name in names])
        im=ax.imshow(np.ma.masked_invalid(100*matrix),aspect='auto',vmin=0,vmax=100,cmap='viridis',extent=(.5,256.5,5.5,-.5))
        ax.set_yticks(range(6),[NAMES[n] for n in names],fontsize=8);ax.set(xlabel='Requested N',title=title)
        if np.isnan(matrix[0]).all():ax.text(128,0,'Not applicable: no controller',ha='center',va='center',color='#555')
        fig.colorbar(im,ax=ax,label='Graphs correct (%)',shrink=.85);matrix_arrays[metric]=matrix
    figures.save(fig,'01_quality','quality_matrices','Quality at every requested depth',
        'All integer depths 1–256 are shown. Blank cells mean undefined metrics, not zero accuracy.\nStrict trajectory cannot recover after an earlier error; final-letter accuracy can.',matrix_arrays)


def execution_figures(figures,config,arrays,names,g,loops,first)->None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,LogNorm
    from matplotlib.patches import Patch
    data=lambda name,metric: arrays[f"{name}__{metric}"]
    x=np.arange(1,257)
    fig,axes=plt.subplots(2,2,figsize=(14,9));survival_arrays={}
    for ax,n in zip(axes.flat,config['fixed_prompt_failure_requests']):
        series=[]
        for i,name in enumerate(names):
            if name=='gru':continue
            rows=sorted((r for r in loops if r['model']==name and int(r['requested_depth'])==n),key=lambda r:int(r['loop']))
            xx=np.array([int(r['loop']) for r in rows]);yy=np.array([float(r['survival']) for r in rows])
            series.append((name,100*yy,'Shared isolated R (GRU & timer)' if name=='final' else NAMES[name]))
            survival_arrays[f'{name}__N{n}']=yy
        grouped_lines(ax,np.arange(1,n+1),series)
        ax.set(xlabel='Actual recurrent loop t',ylabel='Graphs still fully correct (%)',xlim=(1,n),ylim=(-2,102),title=f'One fixed prompt: Steps = {n}')
        ax.grid(axis='y',alpha=.22);ax.legend(fontsize=7)
    figures.save(fig,'02_execution','first_error_survival','Where execution first fails, with the prompt held fixed',
        'Survival denominator is always all 1,350 graphs for each fixed N. A first error ends survival permanently.\nThese curves do not compare different requested counts as though they were one rollout.',survival_arrays)
    fig,axes=plt.subplots(2,3,figsize=(16,7));hazards={}
    for ax,name in zip(axes.flat,names):
        matrix=np.full((4,256),np.nan)
        for j,n in enumerate(config['fixed_prompt_failure_requests']):
            rows=[r for r in loops if r['model']==name and int(r['requested_depth'])==n]
            for r in rows:matrix[j,int(r['loop'])-1]=float(r['hazard']) if r['hazard'] else np.nan
        im=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',vmin=0,vmax=1,cmap='magma',extent=(.5,256.5,3.5,-.5))
        ax.set_yticks(range(4),['N=6','N=12','N=32','N=256']);ax.set(title=NAMES[name],xlabel='Actual loop t')
        fig.colorbar(im,ax=ax,label='First failures / graphs still at risk',shrink=.8);hazards[name]=matrix
    figures.save(fig,'02_execution','first_error_hazard','Conditional first-error hazard',
        'Only graphs correct before loop t enter its risk set. Blank: beyond requested N or an empty risk set.\nZero hazard remains a measured zero; raw at-risk denominators are in fixed_request_loops.csv.',hazards)
    fig,ax=plt.subplots(figsize=(11,10));taxonomy=np.zeros((24,4),dtype=int);labels=[]
    for i,name in enumerate(names):
        for j,n in enumerate(config['fixed_prompt_failure_requests']):
            rows=[r for r in first if r['model']==name and int(r['requested_depth'])==n]
            counts=Counter(r['category'] for r in rows);taxonomy[i*4+j]=[counts[c] for c in CATEGORIES]
            labels.append(f'{NAMES[name]} · N={n} · {len(rows)}/{g} failed')
    im=ax.imshow(taxonomy,aspect='auto',cmap='Blues',vmin=0)
    ax.set_yticks(range(24),labels,fontsize=7);ax.set_xticks(range(4),['Previous state','Earlier visited','Later in orbit','Outside orbit'],fontsize=8)
    for i in range(24):
        for j in range(4):ax.text(j,i,str(taxonomy[i,j]),ha='center',va='center',fontsize=7,color='white' if taxonomy[i,j]>taxonomy.max()*.55 else '#222')
    fig.colorbar(im,ax=ax,label='Graphs at their first incorrect transition')
    figures.save(fig,'02_execution','first_error_taxonomy','Mutually exclusive first-error types',
        'Each failed graph contributes once per fixed requested depth. Priority: previous → earlier visited → later reachable → outside orbit.\nThese describe decoded letters, not a causal explanation of latent-state mechanics.',{'counts':taxonomy})


def structure_figures(figures,config,arrays,names,g,metadata,structure)->None:
    root=figures.root
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,LogNorm
    from matplotlib.patches import Patch
    data=lambda name,metric: arrays[f"{name}__{metric}"]
    x=np.arange(1,257)
    fig,axes=plt.subplots(2,2,figsize=(17,12));structure_arrays={}
    bins=[str(a) if a==b else f'{a}–{b}' for a,b in config['cycle_bins']]
    labels=[f'{NAMES[name]} / {MODE_NAMES[mode]}' for name in names for mode in MODES]
    for ax,n in zip(axes.flat,config['fixed_prompt_failure_requests']):
        matrix=np.full((18,5),np.nan);denom=np.zeros((18,5),dtype=int)
        for i,name in enumerate(names):
            for m,mode in enumerate(MODES):
                for j,binlabel in enumerate(bins):
                    selected=[r for r in structure if r['model']==name and int(r['requested_depth'])==n and r['feature']=='cycle_bin' and r['graph_mode']==mode and r['value']==binlabel]
                    if selected:matrix[i*3+m,j]=float(selected[0]['failure_rate']);denom[i*3+m,j]=int(selected[0]['graphs'])
        im=ax.imshow(np.ma.masked_invalid(matrix*100),aspect='auto',vmin=0,vmax=100,cmap='YlOrRd')
        ax.set_yticks(range(18),labels,fontsize=6.5);ax.set_xticks(range(5),bins);ax.set(title=f'Steps = {n}',xlabel='Cycle period (within graph mode)')
        for i in range(18):
            for j in range(5):
                if denom[i,j]:ax.text(j,i,f'{100*matrix[i,j]:.0f}%\nn={denom[i,j]}',ha='center',va='center',fontsize=6,color='white' if matrix[i,j]>.65 else '#222')
                else:ax.text(j,i,'N/A',ha='center',va='center',fontsize=6,color='#888')
        fig.colorbar(im,ax=ax,label='Graphs with ≥1 execution error (%)',shrink=.75)
        structure_arrays[f'N{n}_failure_rate']=matrix;structure_arrays[f'N{n}_denominator']=denom
    figures.save(fig,'03_structure','cycle_strata','Which graph structures break each executor?',
        'Rates and graph denominators appear in every observed cell. Empty bins are N/A. Full cycles have period 26 by construction.\nCycle period is compared within graph mode; these associations cannot isolate a causal graph feature.',structure_arrays)
    fig,axes=plt.subplots(1,3,figsize=(15,5.5));stratum_arrays={}
    final=data('final','complete_trajectory')[:,255]
    for ax,field,title in zip(axes,('graph_mode','dataset_seed','transient_length'),('Graph mode','Dataset seed','Transient length')):
        values=sorted({m[field] for m in metadata},key=str)
        if field=='transient_length':values=[(a,b) for a,b in config['transient_bins']]
        means=[];los=[];his=[];den=[]
        for value in values:
            ids=[i for i,m in enumerate(metadata) if value[0]<=m[field]<=value[1]] if field=='transient_length' else [i for i,m in enumerate(metadata) if m[field]==value]
            count=len(ids);failed=int((~final[ids]).sum());lo,hi=wilson_interval(failed,count) if count else (np.nan,np.nan)
            means.append(failed/count if count else np.nan);los.append(lo);his.append(hi);den.append(count)
        xx=np.arange(len(values));ax.errorbar(xx,100*np.array(means),yerr=100*np.array([np.array(means)-los,np.array(his)-means]),fmt='o',capsize=4,color=COLORS['final'],label='Failure rate · graph-level Wilson 95%')
        texts=[MODE_NAMES[v] if field=='graph_mode' else f'{v[0]}–{v[1]}' if field=='transient_length' else str(v) for v in values]
        ax.set_xticks(xx,[f'{label}\nn={count}' for label,count in zip(texts,den)],fontsize=8);ax.set(title=title,ylabel='Graphs failing by loop 256 (%)',ylim=(-.3,max(8,float(np.nanmax(his))*100+2)));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
        stratum_arrays[field]=np.array([means,los,his,den])
    figures.save(fig,'03_structure','final_strata','Final executor failures by mode, seed and transient',
        'All 1,350 graphs, one binary trajectory outcome per graph at N=256. Pointwise graph-level intervals.\nTransient bins mix graph modes and are descriptive; within-mode strata and secondary risk exposures are saved as tables.',stratum_arrays)
    # Every failed final graph, no selection of unusually interesting examples.
    traces=np.load(root/'final/diagnostic_traces.npz',allow_pickle=False)['predictions'][:,3,:256]
    hit=traces==arrays['targets'][:,:256];failed=[i for i in range(g) if not hit[i].all()]
    failed.sort(key=lambda i:(metadata[i]['graph_mode'],int(np.flatnonzero(~hit[i])[0]),i))
    fig,ax=plt.subplots(figsize=(15,10))
    ax.imshow(hit[failed].astype(int),aspect='auto',interpolation='nearest',cmap=ListedColormap(['#c24c40','#d7eae6']),vmin=0,vmax=1,extent=(.5,256.5,len(failed)-.5,-.5))
    ax.set_yticks(range(len(failed)),[f"g{i} · {metadata[i]['graph_mode']} · λ={metadata[i]['cycle_period']} μ={metadata[i]['transient_length']}" for i in failed],fontsize=6.5)
    ax.set(xlabel='Actual recurrent loop',ylabel='All failed graphs, grouped by mode and first error')
    ax.legend(handles=[Patch(color='#c24c40',label='Wrong C letter'),Patch(color='#d7eae6',label='Correct C letter')],loc='upper center',bbox_to_anchor=(.5,1.10),ncol=2)
    figures.save(fig,'03_structure','final_failed_trajectories','All final-executor failing trajectories',
        f'{len(failed)} graphs fail; {g-len(failed):,} graphs correct throughout are omitted. λ = cycle period; μ = transient length.\nA later correct letter can recover final-answer accuracy, but cannot erase an earlier trajectory failure.',{'graph_indices':np.array(failed),'correct':hit[failed]})


def stopping_figures(figures,arrays,names,g)->None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,LogNorm
    from matplotlib.patches import Patch
    data=lambda name,metric: arrays[f"{name}__{metric}"]
    x=np.arange(1,257)
    fig,axes=plt.subplots(2,3,figsize=(16,10));stop_matrices={}
    for ax,name in zip(axes.flat,names):
        if name=='ce_full':
            ax.axis('off');ax.text(.5,.55,'Full sequence · CE\nNo learned controller\nStopping is not applicable',ha='center',va='center',transform=ax.transAxes,fontsize=13);continue
        stop=data(name,'stops');matrix=np.zeros((256,273),dtype=int)
        for j in range(256):
            actual=stop[:,j];columns=np.where(actual>0,actual-1,272)
            matrix[j]=np.bincount(columns,minlength=273)
        im=ax.imshow(np.ma.masked_where(matrix.T==0,matrix.T/g),aspect='auto',origin='lower',cmap='viridis',norm=LogNorm(vmin=1/g,vmax=1),extent=(.5,256.5,.5,273.5))
        ax.set(title=NAMES[name],xlabel='Requested N',ylabel='Actual first stop t');ax.set_yticks([1,64,128,192,256,273],['1','64','128','192','256','Missing'])
        fig.colorbar(im,ax=ax,label='Fraction of graphs for that N (log scale)',shrink=.75);stop_matrices[name]=matrix
    figures.save(fig,'04_stopping','requested_actual_stops','Requested versus actual stopping loops',
        'Every requested row contains 1,350 graphs. Missing means no crossing by cap 272; it is a separate bin.\nThe diagonal corresponds to exact completion. Blank cells contain zero observations.',stop_matrices)
    timing_colors={'exact_stop':'#298476','early_stop':'#ce6a38','late_stop':'#8060b0','missing_stop':'#72808c'}
    fig,axes=plt.subplots(2,3,figsize=(16,8.5));timing={}
    for ax,name in zip(axes.flat,names):
        if name=='ce_full':ax.axis('off');ax.text(.5,.5,'CE: no learned stopping',ha='center',transform=ax.transAxes);continue
        matrix=np.array([data(name,key).mean(0) for key in timing_colors]);timing[name]=matrix
        ax.stackplot(x,matrix*100,colors=list(timing_colors.values()),labels=['Exact t=N','Early t<N','Late t>N','Missing at cap'],alpha=.9)
        axis(ax);ax.set_title(NAMES[name]);ax.legend(fontsize=7,loc='upper right')
    figures.save(fig,'04_stopping','timing_failure_modes','How learned stopping fails',
        'Exact, early, late and missing are mutually exclusive and sum to 100% at each N.\nA correct letter at the wrong loop is still a timing failure, including cyclic coincidences.',timing)
    fig,axes=plt.subplots(2,3,figsize=(16,8.5));decomp={}
    for ax,name in zip(axes.flat,names):
        if name=='ce_full':ax.axis('off');ax.text(.5,.5,'CE: execution baseline only',ha='center',transform=ax.transAxes);continue
        execution=data(name,'nominal_final_correct');timed=data(name,'exact_stop')
        groups=np.array([(execution&timed).mean(0),(~execution&timed).mean(0),(execution&~timed).mean(0),(~execution&~timed).mean(0)])
        ax.stackplot(x,groups*100,labels=['Correct answer + exact stop','Execution only fails','Timing only fails','Both fail'],colors=['#298476','#ce6a38','#547bab','#8060b0'])
        axis(ax);ax.set_title(NAMES[name]);ax.legend(fontsize=7,loc='upper right');decomp[name]=groups
    figures.save(fig,'04_stopping','execution_timing_decomposition','Execution and timing failures are separate problems',
        'Execution here means the forced letter at requested N, not strict trajectory correctness.\nTiming is the first crossing at N. These four exclusive outcomes distinguish execution failure from controller failure.',decomp)
    fig,axes=plt.subplots(1,2,figsize=(15,5.5));residuals={}
    residual_series=[];coincidence_series=[]
    for i,name in enumerate(names):
        if name=='ce_full':continue
        stop=data(name,'stops');signed=np.where(stop>0,stop-x[None],np.nan)
        median=np.nanmedian(signed,0);low=np.nanquantile(signed,.1,0);high=np.nanquantile(signed,.9,0)
        residual_series.append((name,median,NAMES[name]));axes[0].fill_between(x,low,high,color=COLORS[name],alpha=.08)
        coincidence_series.append((name,100*data(name,'correct_letter_wrong_time').mean(0),NAMES[name]))
        residuals[name]=np.array([median,low,high])
    grouped_lines(axes[0],x,residual_series);grouped_lines(axes[1],x,coincidence_series)
    axes[0].set(xlabel='Requested N',ylabel='Actual first stop minus N',title='Signed timing residual · median and 10–90%');axes[0].grid(axis='y',alpha=.2);axes[0].legend(fontsize=7)
    axis(axes[1]);axes[1].set_title('Correct final letter at a wrong or missing stop');axes[1].legend(fontsize=7)
    figures.save(fig,'04_stopping','stop_residuals_and_coincidences','Timing errors and misleading correct letters',
        'Missing crossings are excluded from residual quantiles and retained in the timing-category plot.\nRight panel uses all graphs: cyclic coincidences never count as joint success.',residuals)


def learning_figures(figures,config,arrays,g,root)->None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,LogNorm
    from matplotlib.patches import Patch
    data=lambda name,metric: arrays[f"{name}__{metric}"]
    x=np.arange(1,257)
    events=[json.loads(line) for line in Path('models/stage1_pointer/executor_r-seed61/metrics.jsonl').open()]
    validation=[r for r in events if r['event']=='validation'];steps=np.array([r['step'] for r in validation])
    fig,axes=plt.subplots(1,2,figsize=(14,5.5));trajectory=np.array([r['validation']['trajectory_accuracy'] for r in validation])
    accuracy=np.array([r['validation']['intermediate_accuracy'] for r in validation]);loss=np.array([[r['validation']['per_loop'][str(t)]['loss'] for r in validation] for t in range(1,13)])
    axes[0].plot(steps,100*trajectory,'o-',label='Strict trajectory',color='#298476');axes[0].plot(steps,100*accuracy,'s--',label='Intermediate symbol',color='#2467af');axes[0].set(xlabel='Executor optimizer update',ylabel='Fixed validation accuracy (%)',ylim=(-2,102));axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    im=axes[1].imshow(loss,aspect='auto',cmap='magma_r');axes[1].set_xticks(range(len(steps)),steps,rotation=45);axes[1].set_yticks(range(12),range(1,13));axes[1].set(xlabel='Executor optimizer update',ylabel='Actual supervised loop');fig.colorbar(im,ax=axes[1],label='Mean symbol cross-entropy')
    figures.save(fig,'05_learning','executor_validation_learning','How the successful executor learned',
        'Fixed 768-query development monitoring from the original run. Each loop has its own eligible denominator.\nStep 1000 has monitoring records but no retained weights. This is executor development, not a solver checkpoint-selection curve.',{'steps':steps,'trajectory':trajectory,'intermediate':accuracy,'per_loop_loss':loss})
    snapshot_root=root/'executor_snapshots';snapshot_steps=config['final_executor_history']['retained_weight_steps'];snapdata=[]
    for step in snapshot_steps:
        directory=snapshot_root/f'step-{step:06d}'
        if not (directory/'summary.json').exists():raise ValueError(f'Snapshot {step} missing: do not publish partial progression')
        chunks=[np.load(p,allow_pickle=False) for p in sorted(directory.glob('graphs-*.npz'))]
        ids=np.concatenate([z['graph_indices'] for z in chunks]);p=np.concatenate([z['predictions'] for z in chunks])
        if not np.array_equal(ids,np.arange(g)):raise ValueError('Snapshot coverage differs')
        hits=p[:,:256]==arrays['targets'][:,:256]
        snapdata.append(np.array([[hits[:,n-1].mean(),hits[:,:n].all(-1).mean()] for n in config['fixed_prompt_failure_requests']]))
    snapdata=np.stack(snapdata);fig,axes=plt.subplots(2,2,figsize=(13,8))
    for j,(ax,n) in enumerate(zip(axes.flat,config['fixed_prompt_failure_requests'])):
        if np.array_equal(snapdata[:,j,0],snapdata[:,j,1]):
            ax.plot(snapshot_steps,snapdata[:,j,0]*100,'o-',color='#2467af',label='Final letter & strict trajectory (identical)')
        else:
            ax.plot(snapshot_steps,snapdata[:,j,0]*100,'o-',color='#2467af',label='Correct letter at N');ax.plot(snapshot_steps,snapdata[:,j,1]*100,'s--',color='#298476',label='Every step through N correct')
        ax.set(title=f'Fixed requested depth N={n}',xlabel='Executor training update',ylabel='Graphs correct (%)',ylim=(-2,102));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    figures.save(fig,'05_learning','executor_population_progression','Long-range execution across retained successful-architecture snapshots',
        'Same 1,350 graphs, frozen executor snapshots; N is hidden from R. Final exact timer composition would inherit final-letter quality.\nThese are diagnostic compositions, not historical autonomous solvers. Missing step-1000 weights are not interpolated.',{'steps':np.array(snapshot_steps),'quality':snapdata})
    history=read(Path('models/stage1_pointer/controller-prefix-seed83/validation_history.csv'))
    cohorts=[c for c in ('trained','interpolation','unseen_requested_count','seen_count_long_rollout') if any(r['cohort']==c for r in history)];fig,axes=plt.subplots(1,2,figsize=(14,5.5));controller_arrays={}
    cohort_series=[]
    for i,cohort in enumerate(cohorts):
        rows=sorted([r for r in history if r['cohort']==cohort],key=lambda r:int(r['step']))
        xx=np.array([int(r['step']) for r in rows]);exact=np.array([float(r['exact_stop']) for r in rows]);mae=np.array([float(r['remaining_initial_mae']) for r in rows])
        style=['-','--',':','-.'][i%4];marker=['o','s','^','D'][i%4];color=['#2467af','#ce6a38','#298476','#8060b0'][i%4]
        cohort_series.append((['final','joint_full','fixed6','fixed12'][i],xx,exact,mae,cohort.replace('_',' ')))
        controller_arrays[cohort]=np.array([xx,exact,mae])
    grouped_lines(axes[0],cohort_series[0][1],[(name,100*exact,label) for name,xx,exact,mae,label in cohort_series])
    grouped_lines(axes[1],cohort_series[0][1],[(name,mae,label) for name,xx,exact,mae,label in cohort_series])
    axes[0].set(xlabel='GRU controller fitting update',ylabel='Exact first stop (%)',ylim=(-2,102));axes[1].set(xlabel='GRU controller fitting update',ylabel='Initial remaining-step MAE')
    for ax in axes:ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    figures.save(fig,'05_learning','controller_development','Frozen-executor GRU controller development',
        'Original controller-development cohorts, not the 1,350-graph paper panel. Cohort definitions are in controller_repair.md.\nReader and scalar-cell repair are later distinct fits; their numeric validation is tabulated separately, never spliced into this learning curve.',controller_arrays)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'))
    parser.add_argument('--replace',action='store_true',help='Replace this suite’s exports after source audits pass')
    args=parser.parse_args();root=args.input
    if not json.loads((root/'reference_audit.json').read_text())['passed']:raise ValueError('Reference audit failed')
    if not json.loads((root/'independent_audit.json').read_text())['passed']:raise ValueError('Independent audit failed')
    if (root/'plots').exists():
        if not args.replace:raise ValueError('Plot output exists; use --replace')
        shutil.rmtree(root/'plots')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,LogNorm
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'loopformer-paper-20261009','legend.frameon':False})
    config=json.loads((root/'protocol.json').read_text());names=[a['name'] for a in config['models']]
    metadata=json.loads((root/'graph_metadata.json').read_text());g=len(metadata)
    quality=read(root/'quality_by_depth.csv');loops=read(root/'fixed_request_loops.csv');first=read(root/'first_errors.csv')
    structure=read(root/'structure_failure_rates.csv');x=np.arange(1,257)
    arrays=np.load(root/'outcomes.npz',allow_pickle=False);figures=Figures(root)
    data=lambda name,metric: arrays[f'{name}__{metric}']
    quality_figures(figures,arrays,names,g)
    execution_figures(figures,config,arrays,names,g,loops,first)
    structure_figures(figures,config,arrays,names,g,metadata,structure)
    stopping_figures(figures,arrays,names,g)
    learning_figures(figures,config,arrays,g,root)
    figures.finish()
    print(f'Wrote {len(figures.entries)} audited figure families to {figures.output}')


if __name__=='__main__':main()
