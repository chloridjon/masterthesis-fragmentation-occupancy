"""Independent saved-vector audit and exports for expanded boundary queries."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-boundary-audit-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,MISSING,NEUTRAL
DEFAULT=FINAL/'config/02_metapop/02_02_species_parameter_selection/response_boundary_refinement/09_response_boundary_refinement.json'
COLORS=['#0072B2','#009E73','#E69F00','#F7F7F7',MISSING]
PALE=['#CFDFEB','#BDE3D7','#F2DEAD','#F7F7F7',MISSING]
LABELS=['Near-one throughout','Retained response domain','Near-zero throughout','Bounds unresolved','Outside sampled support']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def independent_certificate(lo,hi):
    Llo,Lhi=np.min(lo[0]),np.min(hi[0]);Ulo,Uhi=np.max(lo[1]),np.max(hi[1])
    if Llo>=.95:return 0
    if Uhi<=.05:return 2
    if Lhi<.95 and Ulo>.05:return 1
    return 3


def sorted_percentile(values,fraction):
    x=np.sort(np.asarray(values));rank=fraction*(len(x)-1)
    a,b=int(np.floor(rank)),int(np.ceil(rank))
    return float(x[a]+(rank-a)*(x[b]-x[a]))


def independent_intervals(e,r,lo,hi,ell,lr,D):
    at_zero=np.asarray(r)-lr
    at_D=at_zero+np.log(2)*D*(1/np.asarray(e)-1/ell)
    same=(np.asarray(e)==ell)&(np.asarray(r)==lr)
    more=(np.minimum(at_zero,at_D)>=1e-12)|same
    less=(np.maximum(at_zero,at_D)<=-1e-12)|same
    low=lo[more].max(axis=0) if more.any() else np.zeros(lo.shape[1:])
    high=hi[less].min(axis=0) if less.any() else np.ones(hi.shape[1:])
    assert (low<=high+1e-9).all()
    return np.minimum(low,high),np.maximum(low,high)


def audit(path):
    c=json.loads(path.read_text());data=FINAL/c['data_directory'];results=FINAL/c['results_directory']
    manifest=json.loads((data/'manifest.json').read_text());receipt=json.loads((results/'logs/09_compute_receipt.json').read_text())
    assert receipt['status']=='classified_boundary_queries_complete' and receipt['all_rate_endpoints_closed']
    fp=manifest['producer_fingerprint'];assert receipt['producer_fingerprint']==fp
    for name,digest in {**manifest['source_sha256'],**receipt['outputs_sha256']}.items():assert sha(FINAL/name)==digest,name
    original_manifest=FINAL/'data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds/manifest.json'
    original_hashes=json.loads(original_manifest.read_text())['source_hashes']
    original_scientific={name:digest for name,digest in original_hashes.items()
        if '/src/landscape/' in name or '/src/metapop/' in name or name.endswith('11_lhs_workers.py')}
    for name,digest in original_scientific.items():assert sha(FINAL/name)==digest,name
    base=json.loads((FINAL/c['baseline_configuration']).read_text())
    old=pd.read_csv(FINAL/base['lhs_design'],float_precision='round_trip')
    source=pd.read_csv(FINAL/c['baseline_summary'],float_precision='round_trip')
    with np.load(data/'09_combined_quantile_bounds.npz') as z:
        saved_lo,saved_hi=z['lower'].copy(),z['upper'].copy()
        all_e,all_r,all_categories=z['half_distance'].copy(),z['log_r'].copy(),z['category'].copy()
        cells=pd.DataFrame(dict(scenario_id=z['scenario_id'],H_target=z['H']))
    keys=pd.MultiIndex.from_frame(cells)
    old_q=np.stack([source.pivot(index=['scenario_id','H_target'],columns='parameter_id',values=v).reindex(index=keys,columns=old.parameter_id).to_numpy().T for v in ['q05','q95']],axis=1)
    np.testing.assert_allclose(saved_lo[:128],old_q,atol=1e-15,rtol=0)
    np.testing.assert_allclose(saved_hi[:128],old_q,atol=1e-15,rtol=0)
    benchmark=pd.read_csv(FINAL/base['landscape_data']).set_index(['scenario_id','replicate','H_target'])
    queried=pd.read_csv(results/'tables/09_refinement_queries.csv',float_precision='round_trip')
    geometry_cache=OrderedDict();verified_geometries=set();verified_grids=set()
    valid=excluded=0;max_residual=mean_error=quantile_error=interval_error=0.
    scalar_checks=0;max_scalar_difference=0.;margin_min=1.;precision_vectors=0
    for number,row in enumerate(queried.itertuples(index=False),128):
        record=json.loads((data/'queries'/f'{row.query_id}.json').read_text())
        assert record['producer_fingerprint']==fp and record['query_id']==row.query_id
        ell,lr=record['half_distance'],record['log_r']
        assert all_e[number]==ell and all_r[number]==lr
        lo,hi=independent_intervals(all_e[:number],all_r[:number],saved_lo[:number],saved_hi[:number],ell,lr,c['distance_upper_bound'])
        if lr>np.log(c['r_zero_certificate_bound']):lo=np.zeros_like(lo);hi=lo.copy()
        if lr<=-np.log(2)*c['distance_upper_bound']/ell+np.log(.025):lo=np.maximum(lo,.975)
        before=independent_certificate(lo,hi)
        needed=np.zeros(len(cells),bool)
        if before==3:
            if hi[0].min()>=.95:needed|=(lo[0]<.95)&(hi[0]>=.95)
            if lo[1].max()<=.05:needed|=(lo[1]<=.05)&(hi[1]>.05)
        array=data/'queries'/f'{row.query_id}.npz';assert sha(array)==record['interval_sha256']
        with np.load(array) as z:
            np.testing.assert_array_equal(z['direct_cells'],needed)
            np.testing.assert_array_equal(z['lower'],saved_lo[number]);np.testing.assert_array_equal(z['upper'],saved_hi[number])
        frames=[]
        for name,digest in record['raw_sha256'].items():assert sha(FINAL/name)==digest,name
        for file in sorted((data/'responses'/row.query_id).glob('*.csv')):
            metadata=json.loads(file.with_suffix('.json').read_text())
            if 'precision_source_sha256' in metadata:
                helper=Path(__file__).with_name('09_response_boundary_recovery.py')
                assert sha(helper)==metadata['precision_source_sha256']
                assert (metadata['declared_query'],metadata['declared_trajectory'])==(row.query_id,file.stem)
                diag=metadata['precision_diagnosis']
                assert diag['old_probability_difference']>2e-7 and diag['tighter_probability_difference']<=2e-7
                assert metadata['primary_tolerance_used']==1e-12 and diag['unchanged_probability_tolerance']==2e-7
            table=pd.read_csv(file,float_precision='round_trip');frames.append(table)
            assert table.producer_fingerprint.eq(fp).all() and table.query_id.eq(row.query_id).all()
            assert table.half_distance.eq(ell).all() and table.log_r.eq(lr).all()
            with np.load(file.with_suffix('.npz')) as z:p=z['probabilities'];offsets=z['offsets']
            assert len(offsets)==len(table)+1 and offsets[0]==0 and offsets[-1]==len(p)
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
            for i,answer in enumerate(table.itertuples(index=False)):
                v=p[offsets[i]:offsets[i+1]];key=(answer.scenario_id,answer.replicate,answer.H_target)
                land=benchmark.loc[key]
                assert answer.grid_sha256==land.grid_sha256 and answer.n_patches==land.n_clusters
                if answer.n_patches<2:
                    assert len(v)==0 and np.isnan(answer.occupancy) and answer.status=='excluded_fewer_than_two_patches'
                    excluded+=1;continue
                assert len(v)==answer.n_patches and answer.status=='valid' and answer.converged
                valid+=1;mean_error=max(mean_error,abs(float(v.mean())-answer.occupancy))
                identity=f'{answer.scenario_id}__rep{answer.replicate:03d}'
                if identity not in verified_grids:
                    gf=data/'landscapes'/f'{identity}.npz';gm=json.loads(gf.with_suffix('.json').read_text())
                    assert gm['producer_fingerprint']==fp and sha(gf)==gm['sha256'] and gm['full_event_trace_verified']
                    with np.load(gf) as g:
                        assert len(g['H'])==88
                        for H,grid in zip(g['H'],g['grids']):
                            original=benchmark.loc[(answer.scenario_id,answer.replicate,float(H))]
                            assert hashlib.sha256(grid.tobytes()).hexdigest()==original.grid_sha256
                            assert grid.sum()==round(40000*H)
                    verified_grids.add(identity)
                gf=data/'geometry'/f'{identity}__h{answer.H_target:.3f}.npz'
                if str(gf) in geometry_cache:
                    a,d=geometry_cache.pop(str(gf));geometry_cache[str(gf)]=(a,d)
                else:
                    gm=json.loads(gf.with_suffix('.json').read_text())
                    assert gm['producer_fingerprint']==fp and sha(gf)==gm['sha256']
                    with np.load(gf) as g:
                        assert str(g['grid_sha256'])==land.grid_sha256
                        a,d=g['areas'].copy(),g['distances'].copy()
                    assert len(a)==land.n_clusters and a.sum()==round(40000*answer.H_target)
                    assert d.shape==(len(a),len(a)) and np.array_equal(d,d.T) and d.max()<=c['distance_upper_bound']
                    geometry_cache[str(gf)]=(a,d)
                    while len(geometry_cache)>4:geometry_cache.popitem(last=False)
                    verified_geometries.add(str(gf))
                k=np.exp(-np.log(2)/ell*d);np.fill_diagonal(k,0.)
                incoming=k@(a*v);rate=np.exp(lr)/np.sqrt(a)
                target=incoming/(incoming+rate)
                residual=float(np.max(np.abs(target-v)))
                max_residual=max(max_residual,residual)
                assert residual<=1.1e-10 and abs(residual-answer.fixed_point_residual)<1e-13
                if 'precision_source_sha256' in metadata:
                    assert answer.primary_tolerance_used==1e-12 and residual<=1.1e-12
                    precision_vectors+=1
                if np.isfinite(answer.independent_probability_difference):
                    scalar_checks+=1;max_scalar_difference=max(max_scalar_difference,answer.independent_probability_difference)
                    assert answer.independent_probability_difference<=2e-7
        if needed.any():
            raw=pd.concat(frames,ignore_index=True)
            assert len(raw)==int(needed.sum())*20 and not raw.duplicated(['scenario_id','replicate','H_target']).any()
            for cell_id in np.flatnonzero(needed):
                cell=cells.iloc[cell_id]
                sub=raw[raw.scenario_id.eq(cell.scenario_id)&raw.H_target.eq(cell.H_target)]
                assert sorted(sub.replicate)==list(range(1,21))
                values=sub.occupancy.dropna().to_numpy();assert len(values)>=3
                quantiles=np.array([sorted_percentile(values,.05),sorted_percentile(values,.95)])
                quantile_error=max(quantile_error,float(np.max(np.abs(quantiles-saved_lo[number,:,cell_id]))))
                lo[:,cell_id]=quantiles;hi[:,cell_id]=quantiles
                margin_min=min(margin_min,abs(quantiles[0]-.95),abs(quantiles[1]-.05))
        assert int(needed.sum())==record['n_direct_cells']
        interval_error=max(interval_error,float(np.max(np.abs(lo-saved_lo[number]))),float(np.max(np.abs(hi-saved_hi[number]))))
        assert interval_error<=1e-14 and independent_certificate(lo,hi)==record['category']==all_categories[number]
        print(f'Audited {row.query_id}: {int(needed.sum())} direct cells',flush=True)
    assert mean_error<=1e-14 and quantile_error<=1e-14
    assert valid==receipt['n_valid_vectors'] and excluded==receipt['n_exclusions']
    after=pd.read_csv(results/'tables/09_refined_boundary_brackets.csv')
    for row in after.itertuples(index=False):
        if row.boundary=='lower':mask_a=(all_e<=row.half_distance)&(all_categories==0);mask_b=(all_e>=row.half_distance)&np.isin(all_categories,[1,2])
        else:mask_a=(all_e<=row.half_distance)&np.isin(all_categories,[0,1]);mask_b=(all_e>=row.half_distance)&(all_categories==2)
        assert mask_a.any() and mask_b.any()
        a,b=all_r[mask_a].max()/np.log(10),all_r[mask_b].min()/np.log(10)
        np.testing.assert_allclose([a,b],[row.log10_r_low,row.log10_r_high],atol=1e-13,rtol=0)
    answer=dict(status='independently_audited',created_at_utc=datetime.now(timezone.utc).isoformat(),
        producer_fingerprint=fp,n_queries=len(queried),valid_vectors_checked=valid,exclusions_checked=excluded,
        geometry_files_checked=len(verified_geometries),full_grid_replays_checked=len(verified_grids),
        max_original_equation_residual=max_residual,max_vector_mean_error=mean_error,
        max_independent_quantile_error=quantile_error,max_certificate_interval_error=interval_error,
        independent_scaled_checks=scalar_checks,max_independent_scaled_difference=max_scalar_difference,
        precision_refined_vectors=precision_vectors,
        minimum_direct_quantile_distance_to_threshold=margin_min,n_resolved_brackets=int(after.resolution_reached.sum()),
        n_remaining_wide_brackets=int((~after.resolution_reached).sum()),all_rate_endpoints_closed=True,
        baseline_scientific_source_hashes_unchanged=True,
        compute_receipt_sha256=sha(results/'logs/09_compute_receipt.json'),
        audit_source_sha256=sha(Path(__file__)),visual_review_confirmed=False)
    dump(results/'logs/09_independent_boundary_audit.json',answer)
    return c,data,results,all_e,all_r,all_categories,receipt,answer


def display_grid(e,r,categories,window,nx=260,ny=260):
    x=np.linspace(0,np.log10(4096),nx);y=np.linspace(*window,ny)
    xx,yy=np.meshgrid(x,y);target=np.column_stack([xx.ravel(),yy.ravel()])
    points=np.column_stack([np.log10(e),r/np.log(10)])
    more=(target[:,0,None]>=points[:,0])&(target[:,1,None]<=points[:,1])
    less=(target[:,0,None]<=points[:,0])&(target[:,1,None]>=points[:,1])
    one=(more&(categories==0)).any(axis=1);zero=(less&(categories==2)).any(axis=1)
    retained=(less&np.isin(categories,[1,2])).any(axis=1)&(more&np.isin(categories,[0,1])).any(axis=1)
    assert not ((one&zero)|(one&retained)|(zero&retained)).any()
    z=np.full(len(target),3);z[one]=0;z[zero]=2;z[retained]=1
    scaled=points/np.array([np.log10(4096),window[1]-window[0]])
    ts=target/np.array([np.log10(4096),window[1]-window[0]])
    z[Delaunay(scaled).find_simplex(ts)<0]=4
    return 10**x,10**y,z.reshape(ny,nx)


def plot_and_note(ctx):
    c,data,results,e,r,categories,receipt,audit_result=ctx
    apply_final_style();fig,axes=plt.subplots(1,2,figsize=figure_size(108),sharex=True)
    fig.subplots_adjust(left=.13,right=.98,bottom=.29,top=.93,wspace=.27)
    brackets=pd.read_csv(results/'tables/09_refined_boundary_brackets.csv')
    # The full panel retains the rare isolated-patch tail. Resolve the main
    # sampling window separately instead of repeating all 51 decades.
    detail_low,detail_high=-3.5,6.5
    windows=[receipt['final_log10_r_window'],[detail_low,detail_high]]
    fractions=[]
    for i,(ax,window) in enumerate(zip(axes,windows)):
        ell,ratio,z=display_grid(e,r,categories,window)
        before=display_grid(e[:128],r[:128],categories[:128],window)[2]
        fractions.append(dict(panel=i,old_unresolved_or_unavailable_fraction=float(np.isin(before,[3,4]).mean()),
            new_unresolved_fraction=float((z==3).mean()),new_unavailable_fraction=float((z==4).mean())))
        np.savez_compressed(data/f'09_boundary_display_panel{i+1}.npz',half_distance=ell,r=ratio,classification=z)
        ax.pcolormesh(ell,ratio,z,cmap=ListedColormap(PALE),norm=BoundaryNorm(np.arange(-.5,5.5),5),shading='nearest',rasterized=True)
        ax.contourf(ell,ratio,np.where(z==3,1.,np.nan),levels=[.5,1.5],colors='none',hatches=['////'])
        for category in [0,1,2]:
            for idx,marker,size in [(np.arange(128),'o',9),(np.arange(128,len(e)),'D',12)]:
                sel=idx[categories[idx]==category]
                ax.scatter(e[sel],np.exp(r[sel]),s=size,marker=marker,color=COLORS[category],edgecolors='white',lw=.25,zorder=4)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1,4096);ax.set_ylim(10**window[0],10**window[1])
        ax.set_xticks([1,4,16,64,256,1024,4096],['1','4','16','64','256','1024','4096'])
        ax.minorticks_off()
        ax.text(0,1.055,['(a) Expanded search window','(b) Boundary detail'][i],transform=ax.transAxes)
        ax.add_patch(Rectangle((1,10**-2.5),511,10**5.75-10**-2.5,fill=False,color=NEUTRAL,lw=.7,zorder=3))
    full_ticks=[v for v in [-44,-36,-28,-20,-12,-4,4,7] if windows[0][0]-1e-9<=v<=windows[0][1]+1e-9]
    axes[0].set_yticks([10.**v for v in full_ticks],[f'$10^{{{v}}}$' for v in full_ticks])
    detail_ticks=[v for v in range(-96,9,2) if detail_low<=v<=detail_high]
    axes[1].set_yticks([10.**v for v in detail_ticks],[f'$10^{{{v}}}$' for v in detail_ticks])
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    handles=[Patch(facecolor=PALE[k],label=LABELS[k]) for k in [0,1,2]]
    handles+=[Patch(facecolor='white',edgecolor=NEUTRAL,hatch='////',label='Bounds unresolved (not a CI)'),
        Line2D([],[],color=NEUTRAL,marker='o',ls='none',ms=3,label='Original LHS'),
        Line2D([],[],color=NEUTRAL,marker='D',ls='none',ms=3,label='Refinement certificates'),
        Patch(facecolor='none',edgecolor=NEUTRAL,label='Original search rectangle'),
        Patch(facecolor=MISSING,label='Outside sampled support')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,columnspacing=1.0,handlelength=1.4)
    stem=results/'figures/09_expanded_response_boundaries';export_figure(fig,stem)
    fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    fonts=subprocess.run(['pdffonts',str(stem.with_suffix('.pdf'))],capture_output=True,text=True,check=True).stdout
    assert 'LMRoman8' in fonts and 'LatinModernMath' in fonts
    svg=stem.with_suffix('.svg').read_text();assert '@font-face' in svg and '<text' in svg
    queried=pd.read_csv(results/'tables/09_refinement_queries.csv')
    maximum=float(brackets.bracket_width_decades.max())
    tail=brackets[brackets.half_distance>=512].copy()
    tail.to_csv(results/'tables/09_long_distance_boundary_tail.csv',index=False)
    note=f'''# Completed first boundary refinement

The expanded pilot classified {len(queried)} additional parameter queries at
half-distances 1–4096 cells, with log10(r) endpoints {receipt['final_log10_r_window']}.
The same seven scenarios, 20 matched seed blocks and 601 eligible scenario–H
cells were retained. Both rate endpoints were certified; the original 128-point
LHS and its response dataset remain unchanged. {int(queried.n_direct_cells.sum())}
query/cell combinations required direct quantiles, based on {receipt['n_valid_vectors']}
valid patch-probability vectors and {receipt['n_exclusions']} exclusions.
Other cells were bounded by verified entrywise kernel/r comparisons; their
complete occupancy curves were not simulated or interpolated.

The prespecified four bisection rounds resolved {audit_result['n_resolved_brackets']}
of 26 vertical brackets to the 0.25-decade target. {audit_result['n_remaining_wide_brackets']}
remain wider, up to {maximum:.3f} decades. Rate coverage, vertical bracket
precision and precision between dispersal anchors are separate issues.
Hatching in the figure includes uncertainty between anchors; it is not a
bootstrap confidence interval. A long-distance tail remains an exploratory
diagnostic, not evidence that arbitrarily large dispersal changes occupancy.

The independent audit reloaded all direct vectors and checked means and
original-model residuals (maximum {audit_result['max_original_equation_residual']:.4g}),
grid and geometry identities, quantiles, conservative interval certificates
and boundary witnesses. No bird RMSE or correlation-based selection was used.
These queries are not a random/LHS sample and do not increase the main
inferential sample size. A denser main-domain sample still requires its own
target-density decision.

## Methods paragraph for later manuscript integration

We refined the boundaries of the parameter region excluded by nearly uniform
low or high equilibrium occupancy. The search covered dispersal half-distances
from one to 4096 raster cells. At each of 13 doubling half-distances, we
queried geometric midpoints of rate brackets delimiting globally near-one
and near-zero responses. Eligibility was defined using the minimum landscape
5th percentile and maximum 95th percentile across seven scenarios and eligible
habitat levels, retaining profiles whose two summaries were below 0.95 and
above 0.05, respectively. We used the same 20 matched landscape seeds as in
the initial Latin-hypercube survey. Conservative entrywise comparisons of
the dispersal kernel divided by the extinction-to-colonization ratio supplied
bounds on occupancy and landscape quantiles. Where these bounds did not
determine a profile's classification, we recomputed the quantiles from all
matched seeds in the required scenario–habitat cells. The first refinement
used at most four rounds and a target bracket width of 0.25 logarithmic
decades; wider remaining brackets were reported. The adaptive boundary points
were kept separate from the main parameter sample.

## Figure caption

Expanded boundary search for the q=0 model. (a) Full search window.
(b) Detail spanning log10(r)={detail_low:g} to {detail_high:g}, resolving the
main sampling window. The rare low-rate tail remains visible in panel (a).
Circles show the original 128 LHS profiles;
diamonds mark {len(queried)} additional query certificates, which may use directly
evaluated quantiles, verified monotone bounds or universal endpoint bounds.
Blue denotes globally near-one response, orange globally near-zero response,
and green profiles retained under the L<.95/U>.05 rule. Classification uses
landscape 5th–95th percentiles, seven scenarios and 20 matched seeds; only cells
with at least three valid multi-patch landscapes are eligible. Pale areas
show coordinate-order implications of these certificates; hatched gaps
remain unresolved and are not confidence intervals. The outlined rectangle
is the original main search window. Both axes are logarithmic; half-distance
is measured in raster-cell coordinates and is not calibrated to physical units.
Standalone numerical/export review is complete; manuscript-size review and
final sampling-domain adoption remain pending.
'''
    note_path=results/'09_refinement_results_and_methods.md';note_path.write_text(note)
    outputs=[stem.with_suffix(s) for s in ['.svg','.pdf','.png']]+[note_path]+list(data.glob('09_boundary_display_panel*.npz'))
    metadata=dict(status='exports_checked_visual_review_pending',font_check=fonts,
        audit_sha256=sha(results/'logs/09_independent_boundary_audit.json'),display_fractions=fractions,
        output_sha256={str(p.relative_to(FINAL)):sha(p) for p in outputs},visual_review_confirmed=False)
    dump(results/'logs/09_boundary_export_audit.json',metadata)
    print(json.dumps({**audit_result,'display_fractions':fractions},indent=2),flush=True)


def confirm(path):
    c=json.loads(path.read_text());results=FINAL/c['results_directory'];file=results/'logs/09_boundary_export_audit.json'
    e=json.loads(file.read_text());assert sha(results/'logs/09_independent_boundary_audit.json')==e['audit_sha256']
    for name,digest in e['output_sha256'].items():assert sha(FINAL/name)==digest
    e.update(status='independently_and_visually_checked',visual_review_confirmed=True,
        visual_review_at_utc=datetime.now(timezone.utc).isoformat())
    dump(file,e);print(e['status'])


def checked_plot_inputs(path):
    c=json.loads(path.read_text());data=FINAL/c['data_directory'];results=FINAL/c['results_directory']
    a=json.loads((results/'logs/09_independent_boundary_audit.json').read_text())
    receipt_path=results/'logs/09_compute_receipt.json';receipt=json.loads(receipt_path.read_text())
    assert a['status']=='independently_audited' and sha(receipt_path)==a['compute_receipt_sha256']
    for name,digest in receipt['outputs_sha256'].items():assert sha(FINAL/name)==digest,name
    with np.load(data/'09_combined_quantile_bounds.npz') as z:e,r,categories=z['half_distance'].copy(),z['log_r'].copy(),z['category'].copy()
    return c,data,results,e,r,categories,receipt,a


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','audit','plot','confirm'],default='all');args=parser.parse_args()
    if args.stage=='confirm':confirm(args.config.resolve())
    elif args.stage=='audit':audit(args.config.resolve())
    elif args.stage=='plot':plot_and_note(checked_plot_inputs(args.config.resolve()))
    else:plot_and_note(audit(args.config.resolve()))
