"""Independent vector audit and descriptive whole-seed summaries for fixed profiles."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-confirmed-summary-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'));sys.path.insert(0,str(Path(__file__).parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from src.tools.final_plot_style import apply_final_style,figure_size,export_figure,FOCAL_SCENARIOS,MISSING
M=importlib.import_module('11_confirmed_domain_compute');V=M.V;R=V.R


def audit_trajectory(sid,rep,ctx):
    c,base,lc,design,benchmark,geo,gfp,data,results,fp=ctx
    identity=f'{sid}__rep{rep:03d}';file=data/'trajectories'/f'{identity}.csv'
    meta=json.loads(file.with_suffix('.json').read_text());assert meta['producer_fingerprint']==fp and meta['complete']
    assert R.sha(file)==meta['sha256']
    full=pd.read_csv(file,float_precision='round_trip');assert len(full)==88*256
    assert not full.duplicated(['H_target','parameter_id']).any()
    reference=benchmark[benchmark.scenario_id.eq(sid)&benchmark.replicate.eq(rep)]
    scenario=next(x for x in lc['scenarios'] if x['scenario_id']==sid)
    grids,hs=R.grids(scenario,rep,base,lc,reference,geo,gfp)
    valid=excluded=checks=0;residual=mean_error=monotonic=0.
    coor_more=(design.half_distance.to_numpy()[:,None]>=design.half_distance.to_numpy()[None,:])&(design.r.to_numpy()[:,None]<=design.r.to_numpy()[None,:])
    for grid,h in zip(grids,hs):
        h=float(h);land=reference[reference.H_target.eq(h)].iloc[0]
        digest=__import__('hashlib').sha256(grid.tobytes()).hexdigest();assert digest==land.grid_sha256
        pth=data/'snapshots'/f'{identity}__h{h:.3f}.csv';receipt=json.loads(pth.with_suffix('.json').read_text())
        assert receipt['producer_fingerprint']==fp and receipt['grid_sha256']==digest
        for suffix,digest_value in receipt['hashes'].items():assert R.sha(pth.with_suffix(suffix))==digest_value
        table=pd.read_csv(pth,float_precision='round_trip')
        # Concatenation adds missing optional scalar/capacity diagnostic
        # columns as NaN; compare their values, independent of dtype promotion.
        assert set(table.columns)<=set(full.columns)
        pd.testing.assert_frame_equal(full[full.H_target.eq(h)].reset_index(drop=True),
            table.reindex(columns=full.columns),check_exact=True,check_dtype=False)
        assert table.parameter_id.tolist()==design.parameter_id.tolist()
        assert table.producer_fingerprint.eq(fp).all() and table.grid_sha256.eq(digest).all()
        assert table.replicate.eq(rep).all() and table.scenario_id.eq(sid).all() and table.H_target.eq(h).all()
        assert table.seed.eq(lc['base_seed']+rep-1).all() and table.q.eq(0.).all()
        for name in ['log_r','r','half_distance','alpha']:
            np.testing.assert_array_equal(table[name],design[name])
        with np.load(pth.with_suffix('.npz')) as z:packed=z['probabilities'];offsets=z['offsets']
        n=int(land.n_clusters);assert len(offsets)==257 and offsets[0]==0 and offsets[-1]==len(packed)
        assert np.isfinite(packed).all() and (packed>=0).all() and (packed<=1).all()
        if n<2:
            assert (offsets==0).all() and table.n_patches.eq(n).all()
            assert len(packed)==0 and table.occupancy.isna().all() and table.status.eq('excluded_fewer_than_two_patches').all()
            excluded+=256;continue
        assert np.array_equal(offsets,np.arange(257)*n)
        assert table.n_patches.eq(n).all() and table.status.eq('valid').all() and table.converged.all()
        values=packed.reshape(256,n).T
        mean_error=max(mean_error,float(np.max(np.abs(values.mean(axis=0)-table.occupancy.to_numpy()))))
        a,d=R.geometry(grid,land,identity,geo,gfp)
        for group_id,items in design.groupby('dispersion_group',sort=False):
            indices=items.index.to_numpy();p=values[:,indices]
            k=np.exp(-float(items.alpha.iloc[0])*d);np.fill_diagonal(k,0.)
            incoming=k@(a[:,None]*p);extinction=items.r.to_numpy()[None,:]/np.sqrt(a[:,None])
            target=incoming/(incoming+extinction);res=np.max(np.abs(target-p),axis=0)
            residual=max(residual,float(res.max()))
            limits=table.iloc[indices].primary_tolerance_used.to_numpy()*1.1
            assert (res<=limits).all()
            np.testing.assert_allclose(res,table.iloc[indices].fixed_point_residual,atol=1e-13,rtol=0)
            for j,index in enumerate(indices):
                row=table.iloc[index]
                if not p[:,j].any():
                    if row.solver=='row_sum_verified_empty_equilibrium':
                        assert row.r>float(((k@a)*np.sqrt(a)).max())
                    elif 'capacity_verified_empty' in row.solver:
                        assert np.isfinite(row.capacity_upper) and row.r>row.capacity_upper
                if row.independent_check_requested:
                    assert row.independent_check_converged and row.independent_probability_difference<=2e-7
                    checks+=1
        med=table.occupancy.to_numpy()
        error=np.maximum(med[None,:]-med[:,None],0)
        monotonic=max(monotonic,float(error[coor_more].max()));assert monotonic<=1e-8
        valid+=256
    assert mean_error<=1e-14
    return full,dict(trajectory=identity,valid_vectors=valid,exclusions=excluded,maximum_residual=residual,
        maximum_mean_error=mean_error,maximum_parameter_monotonicity_violation=monotonic,
        independent_scaled_checks=checks,trajectory_sha256=meta['sha256'])


def statistics(values,minimum=3):
    n,s,h,p=values.shape;flat=values.reshape(n,s*h,p)
    mask=np.isfinite(flat[:,:,0]).T
    assert np.array_equal(np.isfinite(flat),np.broadcast_to(mask.T[:,:,None],flat.shape))
    count=mask.sum(axis=1);q=np.full((3,s*h,p),np.nan)
    patterns,inverse=np.unique(mask,axis=0,return_inverse=True)
    for i,seeds in enumerate(patterns):
        if seeds.sum()<minimum:continue
        cells=np.flatnonzero(inverse==i)
        q[:,cells]=np.quantile(flat[seeds][:,cells],[.05,.5,.95],axis=0,method='linear')
    return count.reshape(s,h),q.reshape(3,s,h,p)


def run(n_seeds,workers):
    ctx=M.inputs();c,base,lc,design,benchmark,geo,gfp,data,results,fp=ctx
    assert M.matched_prefix(c,lc,data,fp)>=n_seeds>=5
    output=results/f'snapshot_{n_seeds:03d}';(output/'tables').mkdir(parents=True,exist_ok=True);(output/'figures').mkdir(exist_ok=True)
    marker=output/'11_numerical_audit.json'
    if marker.exists():
        a=json.loads(marker.read_text())
        for name,digest in a['input_sha256'].items():assert R.sha(FINAL/name)==digest
        for name,digest in a['output_sha256'].items():assert R.sha(FINAL/name)==digest
        print(f'Previously audited prefix {n_seeds} retained.',flush=True);return
    scenarios=[s['scenario_id'] for s in FOCAL_SCENARIOS];hs=base['h_targets'];ids=design.parameter_id.tolist()
    values=np.full((n_seeds,7,88,256),np.nan);audit_rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(audit_trajectory,sid,rep,ctx):(sid,rep) for rep in range(1,n_seeds+1) for sid in scenarios}
        for f in as_completed(tasks):
            full,a=f.result();sid,rep=tasks[f]
            values[rep-1,scenarios.index(sid)]=full.pivot(index='H_target',columns='parameter_id',values='occupancy').reindex(index=hs,columns=ids).to_numpy()
            audit_rows.append(a);print(f'Audited {a["trajectory"]}: {a["valid_vectors"]} vectors',flush=True)
    count,q=statistics(values,c['summary']['minimum_valid_seeds']);med=q[1];contrast=med-med[0:1]
    pairs=(np.isfinite(values)&np.isfinite(values[:,0:1])).sum(axis=0)
    keys=pd.MultiIndex.from_product([scenarios,hs,ids],names=['scenario_id','H_target','parameter_id']).to_frame(index=False)
    keys['n_valid']=np.repeat(count[:,:,None],256,axis=2).ravel();keys['n_excluded']=n_seeds-keys.n_valid
    keys['q05']=q[0].ravel();keys['median']=med.ravel();keys['q95']=q[2].ravel()
    keys['contrast_vs_only_random']=contrast.ravel();keys['n_paired_valid']=pairs.ravel()
    independent_error=0.
    for sid in [0,2,6]:
        for hi in [3,40,85]:
            for pi in [0,63,127,191,255]:
                finite=np.sort(values[:,sid,hi,pi][np.isfinite(values[:,sid,hi,pi])])
                if len(finite)<3:assert np.isnan(q[:,sid,hi,pi]).all();continue
                expected=[]
                for level in [.05,.5,.95]:
                    rank=level*(len(finite)-1);a,b=int(np.floor(rank)),int(np.ceil(rank))
                    expected.append(finite[a]+(rank-a)*(finite[b]-finite[a]))
                independent_error=max(independent_error,float(np.max(np.abs(expected-q[:,sid,hi,pi]))))
    assert independent_error<=1e-14
    keys.to_csv(output/'tables/11_response_summaries.csv',index=False)
    stability=[];max_loo=0.
    for omitted in range(n_seeds):
        co,other=statistics(np.delete(values,omitted,axis=0),3);delta=other[1]-other[1,0:1]
        dm=np.abs(other[1]-med);dc=np.abs(delta-contrast)
        finite=np.isfinite(dm);finite_c=np.isfinite(dc[1:])
        max_loo=max(max_loo,float(np.nanmax(dm)),float(np.nanmax(dc[1:])))
        stability.append(dict(omitted_replicate=omitted+1,maximum_median_change=float(np.nanmax(dm)),
            maximum_contrast_change=float(np.nanmax(dc[1:])),median_fraction_above_05=float((dm[finite]>.05).mean()),
            contrast_fraction_above_05=float((dc[1:][finite_c]>.05).mean()),
            median_cells_losing_evaluability=int((np.isfinite(med)&~np.isfinite(other[1])).sum())))
    pd.DataFrame(stability).to_csv(output/'tables/11_seed_omission_stability.csv',index=False)
    exclusions=pd.DataFrame(dict(scenario_id=np.repeat(scenarios,88),H_target=np.tile(hs,7),
        n_valid_landscapes=count.ravel(),n_excluded_landscapes=(n_seeds-count).ravel()))
    exclusions.to_csv(output/'tables/11_landscape_exclusions.csv',index=False)
    span=np.nanmax(med.reshape(-1,256),axis=0)-np.nanmin(med.reshape(-1,256),axis=0)
    profile_table=design.copy();profile_table['median_response_span']=span
    profile_table['min_landscape_q05']=np.nanmin(q[0].reshape(-1,256),axis=0)
    profile_table['max_landscape_q95']=np.nanmax(q[2].reshape(-1,256),axis=0)
    profile_table.to_csv(output/'tables/11_profile_response_diagnostics.csv',index=False)
    np.savez_compressed(output/'tables/11_response_arrays.npz',values=values,quantiles=q,valid_counts=count)
    pd.DataFrame(audit_rows).to_csv(output/'tables/11_vector_audit_by_trajectory.csv',index=False)
    # Parameter variation is separate from landscape variation and CIs.
    ensemble=np.full((3,7,88),np.nan)
    for sid in range(7):
        for h in range(88):
            finite=med[sid,h][np.isfinite(med[sid,h])]
            if len(finite):ensemble[:,sid,h]=np.quantile(finite,[.05,.5,.95],method='linear')
    ensemble_keys=pd.MultiIndex.from_product([scenarios,hs],names=['scenario_id','H_target']).to_frame(index=False)
    for name,arr in zip(['parameter_q05','parameter_median','parameter_q95'],ensemble):ensemble_keys[name]=arr.ravel()
    ensemble_keys.to_csv(output/'tables/11_parameter_ensemble_curves.csv',index=False)
    render(design,span,ensemble,hs,output,n_seeds)
    note=output/'11_preliminary_results_and_methods.md'
    total=n_seeds*7*88*256;valid=sum(x['valid_vectors'] for x in audit_rows);excluded=sum(x['exclusions'] for x in audit_rows)
    assert valid+excluded==total
    note.write_text(f'''# Preliminary conditional-domain results

We computed 256 fixed parameter profiles on {n_seeds} complete matched seed
blocks, seven focal scenarios and all 88 habitat levels: {total:,} response
rows, {valid:,} valid patch-probability vectors and {excluded:,} exclusions.
The main design is uniform in logarithmic area within the certified inner
response region, with 64 marginal-area strata and four conditional rate strata.
Every point receives weight 1/256. Adaptive boundary queries and the earlier
rectangular LHS were not pooled into these summaries. Half-distances remain
within 1–512 cells and log10(r) within -2.5–5.75; the expanded boundary search
was a diagnostic rather than an automatically adopted main distribution.

The independent audit checked every vector, original-model residual, raw
patch mean, grid/seed/parameter identity, and parameter-order comparison.
Maximum original-equation residual was {max(x['maximum_residual'] for x in audit_rows):.4g};
independently calculated summary error was {independent_error:.4g}.
Median response spans across eligible scenario–H cells ranged from
{span.min():.3f} to {span.max():.3f}. The shared selection criterion concerns
landscape-quantile envelopes; it does not require a complete median 1-to-0
response. A reduced seed prefix can therefore show saturation even for points
whose 20-seed pilot envelope certified eligibility.

Whole-seed omission changed an individual median or contrast by as much as
{max_loo:.3f}. Detailed sensitive fractions and changes in evaluable cells are
saved separately. Small residuals establish numerical consistency, not adequate
landscape replication in every cell. These results are preliminary; no
bootstrap confidence intervals, biological parameter calibration or completed
Result-D inference are claimed. Contrasts are differences of scenario medians
against Only-Random at the same H and parameter combination. At high H,
available multi-patch landscapes form a conditional subset; excluded landscapes
are never recoded as extinction. Habitat curves reflect joint changes in amount
and configuration, whereas pointwise contrasts compare scenarios at fixed H.

## Methods paragraph

We sampled 256 parameter profiles from the confirmed interior of the response
region, using a uniform measure in logarithmic parameter area. An inverse-area
mapping placed one randomly jittered dispersal coordinate in each of 64
equal-probability strata. At each coordinate, we sampled four randomly jittered
strata of the conditional logarithmic extinction-to-colonization ratio. The
four ratios shared the same full dispersal kernel for computation, but each
equilibrium retained its own convergence and original-equation residual check.
The model used q=0, b=1 and x=0.5, with raw habitat-cell areas and full
Voronoi-hybrid distances. We reused the verified original landscapes and
calculated all habitat levels for {n_seeds} matched seed blocks. Landscape
variation was summarised by linear 5th and 95th percentiles; medians and
scenario contrasts were recomputed from saved rows. We assessed Monte Carlo
sensitivity by omitting entire matched seed blocks. The design remained fixed
during any replication extension.

## Figure captions

Parameter diagnostic: 256 sampled profiles, coloured by the span of median
patch occupancy across eligible scenario–habitat cells. Each median uses up
to {n_seeds} matched landscapes, with at least three required for evaluation.
No parameter-surface interpolation is used. Both parameter axes are logarithmic;
half-distance is in raster-cell coordinates, without physical calibration.

Parameter-ensemble curves: separate panels show the seven fixed scenarios.
Lines show the median across the 256 landscape-median response curves and
ribbons their 5th–95th percentiles across parameter profiles. These ribbons
represent parameter variation, not landscape variation or confidence intervals.
Profiles have equal weights under the stated conditional logarithmic-area
design. Landscape medians require at least three valid multi-patch landscapes;
missing endpoints and insufficient-count cells remain unavailable. H decreases
left to right. Actual standalone visual inspection remains required before
final use, and manuscript-size inspection has not been performed.
''')
    inputs_hashes={str((data/'trajectories'/f'{sid}__rep{rep:03d}.csv').relative_to(FINAL)):R.sha(data/'trajectories'/f'{sid}__rep{rep:03d}.csv') for rep in range(1,n_seeds+1) for sid in scenarios}
    inputs_hashes[str((data/'manifest.json').relative_to(FINAL))]=R.sha(data/'manifest.json')
    output_hashes={str(p.relative_to(FINAL)):R.sha(p) for p in output.rglob('*') if p.is_file()}
    result=dict(status='independently_audited_exports_visual_review_pending',n_parameters=256,n_seed_blocks=n_seeds,
        n_response_rows=total,n_valid_vectors=valid,n_exclusions=excluded,
        maximum_original_equation_residual=max(x['maximum_residual'] for x in audit_rows),
        maximum_vector_mean_error=max(x['maximum_mean_error'] for x in audit_rows),
        maximum_parameter_monotonicity_violation=max(x['maximum_parameter_monotonicity_violation'] for x in audit_rows),
        independent_scaled_checks=sum(x['independent_scaled_checks'] for x in audit_rows),
        maximum_independent_summary_error=independent_error,maximum_seed_omission_change=max_loo,
        input_sha256=inputs_hashes,output_sha256=output_hashes,summary_source_sha256=R.sha(Path(__file__)),
        visual_review_confirmed=False,bootstrap_intervals_calculated=False,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    R.save_json(marker,result)
    print(json.dumps({k:result[k] for k in ['status','n_parameters','n_seed_blocks','n_valid_vectors','n_exclusions','maximum_seed_omission_change']},indent=2))


def render(design,span,ensemble,hs,output,n_seeds):
    apply_final_style();fig,ax=plt.subplots(figsize=figure_size(102))
    points=ax.scatter(design.half_distance,design.r,c=span,cmap='cividis',vmin=0,vmax=1,s=15,edgecolors='white',lw=.25)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1,512);ax.set_ylim(10**-2.5,10**5.75)
    ax.set_xlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    ax.set_ylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    ax.set_xticks([1,4,16,64,256],['1','4','16','64','256']);ax.minorticks_off()
    fig.colorbar(points,ax=ax,label='Median occupancy response span (0–1)')
    stem=output/'figures/11_sampled_profile_response_span';export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=figure_size(182),sharex=True,sharey=True)
    fig.subplots_adjust(left=.1,right=.98,bottom=.15,top=.98,hspace=.55,wspace=.2)
    for i,(ax,scenario) in enumerate(zip(axes.flat,FOCAL_SCENARIOS)):
        ax.fill_between(hs,ensemble[0,i],ensemble[2,i],color=scenario['color'],alpha=.18,lw=0)
        ax.plot(hs,ensemble[1,i],color=scenario['color'],ls=scenario['linestyle'],lw=1.25)
        ax.set_xlim(1,0);ax.set_ylim(0,1);ax.set_xticks([1,.5,0]);ax.set_yticks([0,.5,1])
        ax.text(0,1.07,f'({chr(97+i)}) '+scenario['label'],transform=ax.transAxes,fontsize=9.5)
    axes.flat[-1].set_visible(False)
    fig.supxlabel(r'Habitat proportion $H$');fig.supylabel('Mean equilibrium patch occupancy')
    fig.legend(handles=[Line2D([],[],color='#555555',label='Median across parameter profiles'),
        Patch(facecolor='#DDDDDD',label='5th–95th percentiles across parameter profiles')],loc='lower center',frameon=False,ncol=1)
    stem2=output/'figures/11_parameter_ensemble_curves';export_figure(fig,stem2);fig.savefig(stem2.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    for s in [stem,stem2]:
        fonts=subprocess.run(['pdffonts',str(s.with_suffix('.pdf'))],capture_output=True,text=True,check=True).stdout
        assert 'LMRoman8' in fonts
        svg=s.with_suffix('.svg').read_text();assert '@font-face' in svg and '<text' in svg


def confirm(n):
    c=json.loads(M.DEFAULT.read_text());output=FINAL/c['results_directory']/f'snapshot_{n:03d}'
    path=output/'11_numerical_audit.json';a=json.loads(path.read_text())
    for name,digest in {**a['input_sha256'],**a['output_sha256']}.items():assert R.sha(FINAL/name)==digest
    a.update(status='independently_and_visually_audited',visual_review_confirmed=True,
        visual_review_at_utc=datetime.now(timezone.utc).isoformat())
    R.save_json(path,a);print(a['status'])


def audit_completed(identity):
    """Check one completed immutable trajectory while other jobs continue."""
    sid,rep=identity.rsplit('__rep',1);rep=int(rep)
    ctx=M.inputs();full,answer=audit_trajectory(sid,rep,ctx)
    answer.update(status='completed_trajectory_independently_audited',n_rows=len(full),
        producer_fingerprint=ctx[-1],audit_source_sha256=R.sha(Path(__file__)),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        complete_matched_seed_summary=False)
    destination=ctx[-2]/'logs'/'partial_audits'/f'{identity}.json'
    R.save_json(destination,answer);print(json.dumps(answer,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--n-seeds',type=int);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--audit-trajectory')
    p.add_argument('--confirm-visual-review',action='store_true');args=p.parse_args()
    if args.audit_trajectory:audit_completed(args.audit_trajectory)
    else:
        assert args.n_seeds is not None,'--n-seeds required for matched-prefix summaries'
        confirm(args.n_seeds) if args.confirm_visual_review else run(args.n_seeds,args.workers)
