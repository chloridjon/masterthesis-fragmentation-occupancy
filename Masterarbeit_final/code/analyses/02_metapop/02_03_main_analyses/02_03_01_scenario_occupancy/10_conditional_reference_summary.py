"""Audit and report the newly selected reference without modifying the former reference."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import argparse,hashlib,importlib,json,os,sys,warnings
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-reference-256-summary-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent.parent/'02_03_02_species_dependence'));sys.path.insert(0,str(FINAL/'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
REF=importlib.import_module('11_conditional_reference_update');BOOT=REF.BOOT;R=REF.R
BIRD=importlib.import_module('11_bird_parameter_surfaces')
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,FOCAL_SCENARIOS


def describe(v):
    count=np.isfinite(v).sum(axis=0)
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);q=np.nanquantile(v,[.05,.5,.95],axis=0)
    q[:,count<3]=np.nan
    return count,q


def audit_one(sid,rep,ctx):
    c,worker,base,lc,item,bench,geo,gfp,data,out,fp,source,sourcefp=ctx
    file=data/'trajectories'/f'{sid}__rep{rep:03d}.csv';receipt=json.loads(file.with_suffix('.json').read_text())
    assert receipt['producer_fingerprint']==fp and R.sha(file)==receipt['sha256'] and receipt['complete']
    input_hashes={str(file.relative_to(FINAL)):receipt['sha256'],str(file.with_suffix('.json').relative_to(FINAL)):R.sha(file.with_suffix('.json'))}
    t=pd.read_csv(file,float_precision='round_trip');assert len(t)==88 and not t.H_target.duplicated().any()
    scenario=next(s for s in lc['scenarios'] if s['scenario_id']==sid);frame=bench[bench.scenario_id.eq(sid)&bench.replicate.eq(rep)]
    grids,hs=R.grids(scenario,rep,base,lc,frame,geo,gfp);valid=excluded=0;maxres=meanerr=0.
    for grid,h in zip(grids,hs):
        row=t[t.H_target.eq(h)].iloc[0];land=frame[frame.H_target.eq(h)].iloc[0];path=data/'snapshots'/f'{sid}__rep{rep:03d}__h{h:.3f}.csv';meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['producer_fingerprint']==fp
        snapshot=pd.read_csv(path,float_precision='round_trip');assert len(snapshot)==1
        for field in ['H_target','parameter_id','r','half_distance','alpha','q','grid_sha256','n_patches','status','occupancy']:
            observed=snapshot[field].iloc[0];expected=row[field]
            assert (pd.isna(observed) and pd.isna(expected)) or observed==expected,(path,field)
        input_hashes[str(path.with_suffix('.json').relative_to(FINAL))]=R.sha(path.with_suffix('.json'))
        for suffix,digest in meta['hashes'].items():
            assert R.sha(path.with_suffix(suffix))==digest
            input_hashes[str(path.with_suffix(suffix).relative_to(FINAL))]=digest
        assert row.producer_fingerprint==fp and row.scenario_id==sid and row.replicate==rep and row.seed==lc['base_seed']+rep-1
        assert row.parameter_id==c['selected_parameter_id'] and row.r==item.r.iloc[0] and row.half_distance==item.half_distance.iloc[0] and row.alpha==item.alpha.iloc[0] and row.q==0
        assert row.grid_sha256==land.grid_sha256==hashlib.sha256(grid.tobytes()).hexdigest() and row.n_patches==land.n_clusters
        with np.load(path.with_suffix('.npz')) as z:p=z['probabilities']
        if rep<=5:
            original=source/'snapshots'/path.name;source_meta=json.loads(original.with_suffix('.json').read_text())
            assert source_meta['producer_fingerprint']==sourcefp
            for suffix,digest in source_meta['hashes'].items():assert R.sha(original.with_suffix(suffix))==digest
            source_rows=pd.read_csv(original,usecols=['parameter_id']);ix=int(np.flatnonzero(source_rows.parameter_id.eq(c['selected_parameter_id']))[0])
            with np.load(original.with_suffix('.npz')) as z:expected=z['probabilities'][z['offsets'][ix]:z['offsets'][ix+1]]
            assert np.array_equal(p,expected),'Reused initial vector differs bitwise'
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        if row.n_patches<2:
            assert len(p)==0 and pd.isna(row.occupancy) and row.status=='excluded_fewer_than_two_patches';excluded+=1;continue
        assert len(p)==row.n_patches and row.status=='valid' and row.converged
        a,d=R.geometry(grid,land,f'{sid}__rep{rep:03d}',geo,gfp);k=np.exp(-row.alpha*d);np.fill_diagonal(k,0.)
        incoming=k@(a*p);target=incoming/(incoming+row.r/np.sqrt(a));res=float(abs(target-p).max());err=abs(p.mean()-row.occupancy)
        assert res<=1.1*row.primary_tolerance_used and err<1e-14 and abs(res-row.fixed_point_residual)<1e-13
        maxres=max(maxres,res);meanerr=max(meanerr,float(err));valid+=1
    return t,dict(trajectory=file.stem,valid_vectors=valid,undefined=excluded,maximum_residual=maxres,maximum_mean_error=meanerr,input_sha256=input_hashes)


def plots(summary,cis,out,c,item):
    apply_final_style();stems=[]
    def export(fig,name):
        stem=out/'figures'/name;export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig);stems.append(stem)
    fig,ax=plt.subplots(figsize=figure_size(110));fig.subplots_adjust(left=.13,right=.98,bottom=.31,top=.97)
    for s in FOCAL_SCENARIOS:
        t=summary[summary.scenario_id.eq(s['scenario_id'])].sort_values('H');ax.plot(t.H,t['median'],color=s['color'],ls=s['linestyle'],label=s['label'])
    ax.set_xlim(1,0);ax.set_ylim(0,1);ax.set_xlabel(r'Habitat proportion $H$');ax.set_ylabel(r'Median equilibrium patch occupancy $\bar p$');ax.spines[['top','right']].set_visible(False)
    fig.legend(loc='lower center',frameon=False,ncol=2);export(fig,'10_updated_reference_occupancy')
    fig,axes=plt.subplots(4,2,figsize=figure_size(180),sharex=True,sharey=True);fig.subplots_adjust(left=.1,right=.98,bottom=.15,top=.97,hspace=.5,wspace=.22)
    for i,(ax,s) in enumerate(zip(axes.flat,FOCAL_SCENARIOS)):
        t=summary[summary.scenario_id.eq(s['scenario_id'])].sort_values('H');ax.fill_between(t.H,t.q05,t.q95,color=s['color'],alpha=.2,lw=0);ax.plot(t.H,t['median'],color=s['color'],ls=s['linestyle']);ax.set_xlim(1,0);ax.set_ylim(0,1);ax.set_xticks([1,.5,0]);ax.set_yticks([0,.5,1]);ax.text(0,1.06,f'({chr(97+i)}) '+s['label'],transform=ax.transAxes,fontsize=9.5)
    axes.flat[-1].set_visible(False);fig.supxlabel(r'Habitat proportion $H$');fig.supylabel(r'Equilibrium patch occupancy $\bar p$');fig.legend(handles=[Line2D([],[],color='#555555',label='Median'),Patch(facecolor='#777777',alpha=.2,label='5th–95th percentiles across landscapes')],loc='lower center',frameon=False,ncol=1);export(fig,'10_updated_reference_landscape_variation')
    bound=max(.1,float(np.ceil(np.nanmax(abs(cis[['contrast','low','high']].to_numpy()))*10)/10))
    fig,axes=plt.subplots(3,2,figsize=figure_size(155),sharex=True,sharey=True);fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.96,hspace=.55,wspace=.24)
    for i,(ax,s) in enumerate(zip(axes.flat,FOCAL_SCENARIOS[1:])):
        t=cis[cis.scenario_id.eq(s['scenario_id'])].sort_values('H');ax.fill_between(t.H,t.low,t.high,color=s['color'],alpha=.2,lw=0);ax.plot(t.H,t.contrast,color=s['color'],ls=s['linestyle']);ax.axhline(0,color='#555555',lw=.6);ax.set_xlim(1,0);ax.set_ylim(-bound,bound);ax.set_xticks([1,.5,0]);ax.text(0,1.06,f'({chr(97+i)}) '+s['label'],transform=ax.transAxes,fontsize=9.5)
    fig.supxlabel(r'Habitat proportion $H$');fig.supylabel(r'Median occupancy contrast $\Delta\bar p$');fig.legend(handles=[Line2D([],[],color='#555555',label='Difference from Only-Random'),Patch(facecolor='#777777',alpha=.2,label='95% paired whole-seed bootstrap CI')],loc='lower center',frameon=False,ncol=1);export(fig,'10_updated_reference_contrasts')
    return stems



def run(wait_for_compute=False,audit_workers=None):
    ctx=REF.setup();c,worker,base,lc,item,bench,geo,gfp,data,out,fp,source,sourcefp=ctx;n=c['n_reference_seeds'];hs=np.array(base['h_targets']);ids=[s['scenario_id'] for s in FOCAL_SCENARIOS]
    if not wait_for_compute:assert len(list((data/'trajectories').glob('*.json')))==n*7
    values=np.full((n,7,88),np.nan);checks=[];frames=[];vector_input_hashes={}
    import time
    deadline=time.monotonic()+7200;queued=set();tasks={}
    with ProcessPoolExecutor(max_workers=audit_workers or c['workers']) as pool:
        while len(checks)<n*7:
            for rep in range(1,n+1):
                for sid in ids:
                    key=(sid,rep)
                    if key not in queued and (data/'trajectories'/f'{sid}__rep{rep:03d}.json').exists():
                        tasks[pool.submit(audit_one,sid,rep,ctx)]=key;queued.add(key)
            if tasks:
                finished,_=wait(tasks,timeout=10,return_when=FIRST_COMPLETED)
                for f in finished:
                    t,check=f.result();sid,rep=tasks.pop(f);values[rep-1,ids.index(sid)]=t.set_index('H_target').occupancy.reindex(hs);frames.append(t);vector_input_hashes.update(check.pop('input_sha256'));checks.append(check)
                    R.save_json(out/'logs/10_audit_progress.json',dict(audited_trajectories=len(checks),expected=n*7,queued=len(queued)))
                    print(f'Audited {len(checks)}/{n*7}: {sid} rep{rep:03d}',flush=True)
            else:time.sleep(5)
            assert time.monotonic()<deadline,'Two-hour wait/audit ceiling reached; data remain preserved'
    count,q=describe(values);median=q[1];contrast=median[1:]-median[0]
    raw=pd.concat(frames,ignore_index=True);raw.to_csv(out/'tables/10_reference_response_rows.csv',index=False)
    ind=raw.groupby(['scenario_id','H_target']).occupancy.quantile([.05,.5,.95]).unstack();independent_error=0.
    rows=[]
    for j,sid in enumerate(ids):
        for k,h in enumerate(hs):
            if count[j,k]>=3:independent_error=max(independent_error,float(abs(ind.loc[(sid,h)].to_numpy()-q[:,j,k]).max()))
            rows.append(dict(scenario_id=sid,H=h,n_seed_blocks=n,n_valid=int(count[j,k]),n_excluded=n-int(count[j,k]),median=median[j,k],q05=q[0,j,k],q95=q[2,j,k],contrast_vs_only_random=median[j,k]-median[0,k]))
    assert independent_error<1e-14
    summary=pd.DataFrame(rows);summary.to_csv(out/'tables/10_response_summaries.csv',index=False)
    boot=BOOT.paired_contrast_bootstrap(values,c['bootstrap'],minimum_seeds=3,minimum_valid_fraction=c['minimum_valid_bootstrap_fraction']);pd.DataFrame(boot['history']).to_csv(out/'tables/10_bootstrap_history.csv',index=False)
    cis=pd.DataFrame([dict(scenario_id=sid,H=h,contrast=contrast[j,k],low=boot['low'][j,k],high=boot['high'][j,k],bootstrap_status=boot['status'][j,k],draws=int(boot['draws_used'][j,k])) for j,sid in enumerate(ids[1:]) for k,h in enumerate(hs)])
    cis.to_csv(out/'tables/10_pointwise_contrasts.csv',index=False)
    # Independently reconstruct twelve frozen cells from the original RNG stream.
    rng=np.random.default_rng(c['bootstrap']['seed']);indices=rng.integers(0,n,size=(boot['total_draws'],n));cells=np.argwhere(boot['converged']);boot_error=0.
    for j,k in cells[np.linspace(0,len(cells)-1,min(12,len(cells))).astype(int)]:
        draws=int(boot['draws_used'][j,k]);ix=indices[:draws]
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);a=np.nanmedian(values[ix,j+1,k],axis=1)-np.nanmedian(values[ix,0,k],axis=1);expected=np.nanquantile(a,[.025,.975])
        boot_error=max(boot_error,float(abs(expected-[boot['low'][j,k],boot['high'][j,k]]).max()))
    assert boot_error<1e-14
    stability=[]
    for omit in range(n):
        co,other=describe(np.delete(values,omit,axis=0));dm=abs(other[1]-median);dc=abs((other[1,1:]-other[1,0])-contrast)
        stability.append(dict(omitted_seed=omit+1,maximum_median_change=float(np.nanmax(dm)),maximum_contrast_change=float(np.nanmax(dc)),median_fraction_above_05=float((dm[np.isfinite(dm)]>.05).mean()),contrast_fraction_above_05=float((dc[np.isfinite(dc)]>.05).mean())))
    pd.DataFrame(stability).to_csv(out/'tables/10_seed_omission_stability.csv',index=False)
    count5,q5=describe(values[:5]);contrast5=q5[1,1:]-q5[1,0]
    expansion=pd.DataFrame([dict(scenario_id=sid,H=h,n_valid_initial5=int(count5[j,k]),n_valid_final20=int(count[j,k]),initial5_median=q5[1,j,k],final20_median=median[j,k],median_change=median[j,k]-q5[1,j,k],contrast_change=(contrast[j-1,k]-contrast5[j-1,k]) if j else np.nan) for j,sid in enumerate(ids) for k,h in enumerate(hs)])
    expansion.to_csv(out/'tables/10_reference_5_to_20_expansion.csv',index=False)
    metrics,h=REF.criteria(values[:,:,:,None],hs,item.reset_index(drop=True));metrics.to_csv(out/'tables/10_selected_reference_response_metrics.csv',index=False)
    # Same first20 seeds for comparison with the earlier numerical reference.
    old=FINAL/'data/processed/02_metapop/02_03_main_analyses/02_03_01_scenario_occupancy/reference_1000_16';old_values=np.full_like(values,np.nan)
    for rep in range(1,n+1):
        for j,sid in enumerate(ids):
            p=old/'trajectories'/f'{sid}__rep{rep:03d}.csv';meta=json.loads(p.with_suffix('.json').read_text());assert R.sha(p)==meta['hashes']['.csv'];t=pd.read_csv(p,float_precision='round_trip');old_values[rep-1,j]=t.set_index('H_target').occupancy.reindex(hs)
    _,oldq=describe(old_values)
    pd.DataFrame([dict(scenario_id=sid,H=h,old_reference_median=oldq[1,j,k],new_reference_median=median[j,k],difference=median[j,k]-oldq[1,j,k]) for j,sid in enumerate(ids) for k,h in enumerate(hs)]).to_csv(out/'tables/10_old_new_reference_same20.csv',index=False)
    np.savez_compressed(out/'tables/10_reference_arrays.npz',values=values,quantiles=q,old_reference_values=old_values)
    stems=plots(summary,cis,out,c,item);exports=BIRD.export_checks(stems)
    selected=item.iloc[0];st=pd.DataFrame(stability)
    report=f'''# Updated illustrative Result B reference

The new reference is {selected.parameter_id}: r={selected.r:.12g}, dispersal
half-distance={selected.half_distance:.12g} raster cells, q=0, b=1, x=.5.
It is a response-informed Pareto compromise from the new 256-point design,
not a unique mathematical optimum or a typical biological species. Candidate
tradeoffs and seed-omission selection diagnostics are saved. {c['selection_rule']}
The exact full-precision sampled profile was used without rounding or geometric
substitution. The previous r=1000 / half-distance 16 reference and its 200-seed
report remain unchanged. Old/new reference curves are also compared on the
same first 20 seeds rather than conflating replication and parameter changes.

The new reference was evaluated over {n} matched seeds ×7 scenarios ×88 H
levels. The initial five exact vectors were reused from the audited 256-profile
run, and 15 additional matched seed blocks were calculated using the same
verified landscapes, full kernels, fixed areas/distances and primary 1e-12
tolerance. Every saved vector, grid/seed/parameter identity, probability mean
and original-equation residual was checked independently. Numerical checks
do not by themselves establish sampling precision.

On the common at-least-three-valid-seed domain, the minimum occupancy span
across seven scenario medians is {metrics.minimum_scenario_occupancy_span.iloc[0]:.3f},
minimum total H-width in the .1–.9 occupancy band is
{metrics.minimum_intermediate_H_width.iloc[0]:.3f}, and mean pairwise scenario
separation is {metrics.mean_pairwise_scenario_separation.iloc[0]:.3f}.
All curves reach .1 and .9: {bool(metrics.all_scenarios_reach_10_and_90_percent.iloc[0])}.
These selection diagnostics are distinct from a primary integrated effect;
main results remain pointwise contrasts at constant H. Along-H curves combine
habitat amount and scenario-generated configuration.

Figures show medians, separate landscape 5th–95th percentile bands, and same-H
differences from Only-Random with pointwise 95% percentile-bootstrap CIs. Whole
matched seed blocks were resampled together across scenarios/H; settings were
fixed before uncertainty calculation: 1000 initial draws, 1000 per batch,
maximum 20000, three stable batches, endpoint/centre tolerance .002,
seed {c['bootstrap']['seed']}. At least three original valid seeds and 95% defined
bootstrap contrasts are required. Stable intervals: {int(boot['converged'].sum())}
of {int(boot['eligible'].sum())} eligible; unresolved intervals are withheld.
These are pointwise, small-sample, post-selection illustrative intervals, not
simultaneous bands or confidence intervals for an average species.

Maximum single-seed median/contrast changes were
{st.maximum_median_change.max():.3f}/{st.maximum_contrast_change.max():.3f}.
The five-to-20-seed median/contrast changes reach
{expansion.median_change.abs().max():.3f}/{expansion.contrast_change.abs().max():.3f}
on cells evaluable at both sizes; counts and any changing evaluability are
retained separately. Counts and exclusions are reported by H/scenario. Fewer-than-two-patch
landscapes remain undefined, not extinct. High-H responses describe the
remaining fragmented subset and are not representative of all landscapes.
No primary AUC, biological calibration or bird-informed reference selection
was used. Standalone visual review is pending explicit confirmation;
manuscript-size review remains pending.
'''
    (out/'README.md').write_text(report)
    inputs={str((data/'manifest.json').relative_to(FINAL)):R.sha(data/'manifest.json'),str(Path(__file__).relative_to(FINAL)):R.sha(Path(__file__))}
    inputs.update(vector_input_hashes)
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='10_reference_update_audit.json']
    record=dict(status='numerically_checked_visual_pending',selected_parameter_id=selected.parameter_id,r=float(selected.r),half_distance=float(selected.half_distance),n_matched_seeds=n,n_trajectories=n*7,n_rows=len(raw),valid_vectors=sum(a['valid_vectors'] for a in checks),undefined=sum(a['undefined'] for a in checks),maximum_residual=max(a['maximum_residual'] for a in checks),maximum_independent_summary_error=independent_error,bootstrap_converged=int(boot['converged'].sum()),bootstrap_eligible=int(boot['eligible'].sum()),bootstrap_maximum_draws=boot['total_draws'],maximum_independent_bootstrap_error=boot_error,n_independent_bootstrap_cells=min(12,len(cells)),maximum_seed_omission_contrast=float(st.maximum_contrast_change.max()),visual_review_confirmed=False,input_sha256=inputs,output_sha256={str(p.relative_to(FINAL)):R.sha(p) for p in files},export_checks=exports)
    R.save_json(out/'logs/10_reference_update_audit.json',record);print(json.dumps({k:v for k,v in record.items() if k not in ['input_sha256','output_sha256','export_checks']},indent=2))


def confirm():
    c=json.loads(REF.CONFIG.read_text());p=FINAL/c['results_directory']/'logs/10_reference_update_audit.json';a=json.loads(p.read_text())
    for name,h in {**a['input_sha256'],**a['output_sha256']}.items():assert R.sha(FINAL/name)==h,name
    a.update(status='numerically_and_visually_checked_preliminary',visual_review_confirmed=True,visual_review_at_utc=datetime.now(timezone.utc).isoformat());R.save_json(p,a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--confirm-visual-review',action='store_true');p.add_argument('--wait-for-compute',action='store_true');p.add_argument('--audit-workers',type=int,choices=range(1,7));a=p.parse_args()
    confirm() if a.confirm_visual_review else run(a.wait_for_compute,a.audit_workers)
