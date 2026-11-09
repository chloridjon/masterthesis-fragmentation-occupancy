"""Result D: scenario-balanced structural-metric associations at fixed H.

Run from the repository root via final-result-d-metric-associations with stages
prepare, bootstrap, export, confirm. Reads only previously audited responses.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import warnings
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-result-d-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
sys.path.insert(0,str(Path(__file__).parent.parent/'02_03_02_species_dependence'))
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from src.tools.weighted_rank_associations import balanced_spearman
from src.tools.cohort_rank_bootstrap import two_cohort_indices, frequencies
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS, MISSING, NEUTRAL
B=importlib.import_module('11_bird_parameter_surfaces')
U=importlib.import_module('11_conditional_bird_update')
REL='02_metapop/02_03_main_analyses/02_03_03_metric_associations'
CONFIG=FINAL/f'config/{REL}/12_result_d.json'
LABELS=['Edge density','Largest cluster fraction',r'$\chi$','Mean NN distance','Mean inter-patch distance',r'Complete-MST $D_{50}$']
GROUPS=np.repeat(np.arange(7),16)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def key(p):return str(Path(p).relative_to(FINAL,walk_up=True))
def write(p,a):p.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
def paths():
    c=json.loads(CONFIG.read_text());out,data=(FINAL/c[k] for k in ['results_directory','data_directory'])
    for p in [out/'tables',out/'logs',out/'figures',out/'review',data,data/'bootstrap_blocks']:p.mkdir(parents=True,exist_ok=True)
    return c,out,data


def checked(receipt):
    for k,v in {**receipt.get('input_sha256',{}),**receipt.get('output_sha256',{})}.items():
        assert sha(FINAL/k)==v,k


def associations(x,y,c,f=None,threshold=None):
    """x[H,metric,landscape], y[H,profile,landscape]."""
    if f is not None:f=np.broadcast_to(f,y.shape)
    return np.stack([balanced_spearman(x[:,m,None,:],y,GROUPS,f,
        minimum_per_group=c['minimum_valid_per_scenario'],
        minimum_response_range=c['minimum_response_range'] if threshold is None else threshold)
        for m in range(x.shape[1])],axis=1)


def independent(x,y,f,c):
    """Scalar reference using explicit pairwise mid-distribution sums."""
    ok=np.isfinite(x)&np.isfinite(y)&(f>0);x=x[ok];y=y[ok];g=GROUPS[ok];f=f[ok]
    if len(set(g))!=7 or any(f[g==j].sum()<c['minimum_valid_per_scenario'] for j in range(7)) or np.ptp(y)<=c['minimum_response_range'] or np.ptp(x)==0:return np.nan
    w=np.array([f[i]/f[g==g[i]].sum()/7 for i in range(len(x))])
    a=np.array([w[x<v].sum()+.5*w[x==v].sum() for v in x])
    b=np.array([w[y<v].sum()+.5*w[y==v].sum() for v in y])
    a-=sum(w*a);b-=sum(w*b)
    den=np.sqrt(sum(w*a*a)*sum(w*b*b))
    return sum(w*a*b)/den if den>1e-28 else np.nan


def prepare():
    c,out,data=paths();receipt=out/'logs/12_preparation.json';assert not receipt.exists(),'Use saved preparation; no overwrite'
    snap=FINAL/c['source_snapshot'];source=FINAL/c['source_data']
    audit=json.loads((snap/'11_numerical_audit.json').read_text());visual=json.loads((snap/'11_export_audit.json').read_text())
    assert audit['n_trajectories']==104 and audit['n_parameters']==256 and visual['visual_review_confirmed']
    assert visual['numeric_audit_sha256']==sha(snap/'11_numerical_audit.json')
    inputs=[CONFIG,Path(__file__),FINAL/'code/base/src/tools/weighted_rank_associations.py',
        FINAL/'code/base/src/tools/cohort_rank_bootstrap.py',snap/'11_numerical_audit.json',snap/'11_export_audit.json',
        source/'manifest.json',FINAL/c['landscape_data'],FINAL/c['domain_strips'],FINAL/c['bird_matches']]
    manifest=json.loads((source/'manifest.json').read_text())
    assert sha(FINAL/c['landscape_data'])==manifest['source_sha256'][c['landscape_data']]
    for name,digest in manifest['source_sha256'].items():assert sha(FINAL/name)==digest,name
    table_paths=[snap/'tables'/name for name in ['11_response_arrays.npz','11_response_summaries.csv','11_profile_response_diagnostics.csv']]
    for p in table_paths:assert sha(p)==audit['output_sha256'][key(p)]
    inputs+=table_paths
    design=pd.read_csv(table_paths[2],float_precision='round_trip')
    df=pd.read_csv(table_paths[1],float_precision='round_trip');h=np.sort(df.H_target.unique())
    z=np.load(table_paths[0]);raw=z['values'];planned=z['planned_seed_mask'];count=z['valid_counts']
    assert raw.shape==(16,7,88,256) and planned.sum()==104
    assert planned[:8].all() and not planned[8:,0].any() and planned[8:,1:].all()
    assert not np.isfinite(raw[~planned]).any()
    assert np.all(np.isfinite(raw)==np.isfinite(raw[...,0,None]))
    land=pd.read_csv(FINAL/c['landscape_data'],float_precision='round_trip')
    land=land[land.replicate.le(16)&(land.scenario_id.ne('only_random_rho0')|land.replicate.le(8))].copy()
    assert len(land)==9152 and not land.duplicated(['scenario_id','replicate','H_target']).any()
    assert np.array_equal(land.H_real.to_numpy(),land.H_target.to_numpy())
    x=np.full((88,6,112),np.nan);y=raw.transpose(2,3,1,0).reshape(88,256,112)
    domain=[];trajectory_error=0.;n_hashes=0
    for j,s in enumerate(FOCAL_SCENARIOS):
        n=8 if j==0 else 16
        for rep in range(1,n+1):
            identity=f"{s['scenario_id']}__rep{rep:03d}"
            a=land[land.scenario_id.eq(s['scenario_id'])&land.replicate.eq(rep)].set_index('H_target').reindex(h)
            assert len(a)==88 and a.seed.eq(20260924+rep-1).all()
            valid=a.n_clusters.to_numpy()>=2
            np.testing.assert_array_equal(valid,np.isfinite(raw[rep-1,j,:,0]))
            x[:,:,j*16+rep-1]=a[c['metrics']].to_numpy()
            x[~valid,:,j*16+rep-1]=np.nan
            p=source/'trajectories'/f'{identity}.csv';meta=p.with_suffix('.json')
            assert sha(p)==audit['input_sha256'][key(p)] and sha(meta)==audit['input_sha256'][key(meta)]
            t=pd.read_csv(p,float_precision='round_trip')
            saved=t.pivot(index='H_target',columns='parameter_id',values='occupancy').reindex(index=h,columns=design.parameter_id).to_numpy()
            np.testing.assert_allclose(saved,raw[rep-1,j],atol=0,rtol=0,equal_nan=True)
            hashes=t[['H_target','grid_sha256','n_patches']].drop_duplicates().set_index('H_target').reindex(h)
            assert hashes.grid_sha256.tolist()==a.grid_sha256.tolist();np.testing.assert_array_equal(hashes.n_patches,a.n_clusters);n_hashes+=88
            inputs += [p,meta]
        for k,hh in enumerate(h):domain.append(dict(scenario_id=s['scenario_id'],H=hh,n_planned=n,n_valid=int(count[j,k]),n_excluded=n-int(count[j,k]),unplanned_seeds=8 if j==0 else 0))
    # ED's saved global positive normalisation factor cannot alter any ranks;
    # all associations use the immutable source values and report no ED units.
    assert np.isfinite(x).all(axis=1).sum()==np.isfinite(y[:,0]).sum()
    start=time.perf_counter();rho=associations(x,y,c)
    print(f'Original {np.isfinite(rho).sum()} coefficients: {time.perf_counter()-start:.1f}s',flush=True)
    # Unweighted pool is a diagnostic only; shared-eight still balances scenarios.
    rounded=associations(x,np.round(y/c['tie_sensitivity_rounding'])*c['tie_sensitivity_rounding'],c)
    shared_mask=np.tile(np.r_[np.ones(8),np.zeros(8)],7)
    initial_mask=np.tile(np.r_[np.ones(5),np.zeros(11)],7)
    rho8=associations(x,y,c,shared_mask);rho5=associations(x,y,c,initial_mask)
    omitted=[]
    for seed in range(16):
        f=np.ones(112);f[np.arange(7)*16+seed]=0
        omitted.append(associations(x,y,c,f));print(f'Whole-seed omission {seed+1}/16 checked',flush=True)
    omitted=np.array(omitted)
    # Independently check all reference coefficients and a spread of profiles.
    err=0.;checks=0
    selected=sorted(set([0,63,127,130,191,255]))
    for k in range(88):
        for m in range(6):
            for p in selected:
                expected=independent(x[k,m],y[k,p],np.ones(112),c)
                np.testing.assert_allclose(expected,rho[k,m,p],atol=1e-13,rtol=0,equal_nan=True)
                if np.isfinite(expected):err=max(err,abs(expected-rho[k,m,p]));checks+=1
    # Balanced complete H levels agree with ordinary Spearman for shared8.
    scipyerr=0.
    for k in np.flatnonzero((count>=8).all(axis=0)):
        ix=np.flatnonzero(shared_mask)
        for m in range(6):
            v=y[k,130,ix];xx=x[k,m,ix]
            if np.ptp(v)>c['minimum_response_range'] and np.ptp(xx)>0:
                scipyerr=max(scipyerr,abs(spearmanr(xx,v).statistic-rho8[k,m,130]))
    assert scipyerr<1e-13
    birds=pd.read_csv(FINAL/c['bird_matches']);example=birds.best_parameter_id.value_counts().rename_axis('parameter_id').reset_index(name='count').sort_values(['count','parameter_id'],ascending=[False,True]).iloc[0].parameter_id
    shared,extra=two_cohort_indices(c['bootstrap']['maximum_draws'],c['bootstrap']['seed'])
    np.savez_compressed(data/'12_prepared_arrays.npz',H=h,x=x,y=y,rho=rho,rho_matched8=rho8,rho_initial5=rho5,rho_rounded=rounded,omitted=omitted,valid_counts=count,planned=planned,parameter_ids=design.parameter_id.to_numpy(str),r=design.r,half_distance=design.half_distance,shared_indices=shared,extra_indices=extra)
    pd.DataFrame(domain).to_csv(out/'tables/12_domain_and_exclusions.csv',index=False)
    design.to_csv(data/'12_parameter_design.csv',index=False)
    preparation=dict(status='source_rows_hashes_and_associations_verified',created_utc=datetime.now(timezone.utc).isoformat(),n_trajectories=104,n_snapshots=n_hashes,n_profiles=256,n_H=88,n_eligible_coefficients=int(np.isfinite(rho).sum()),independent_checks=checks,maximum_independent_coefficient_error=float(err),maximum_shared8_scipy_error=float(scipyerr),empirical_example=example,empirical_example_species=birds.loc[birds.best_parameter_id.eq(example),'species_as_published'].tolist(),parameter_ranges=dict(r=[float(design.r.min()),float(design.r.max())],half_distance=[float(design.half_distance.min()),float(design.half_distance.max())]),input_sha256={key(p):sha(p) for p in inputs},output_sha256={key(p):sha(p) for p in [data/'12_prepared_arrays.npz',data/'12_parameter_design.csv',out/'tables/12_domain_and_exclusions.csv']})
    write(receipt,preparation);print(json.dumps({k:preparation[k] for k in ['n_eligible_coefficients','maximum_independent_coefficient_error','empirical_example','parameter_ranges']}),flush=True)


def bootstrap_block(x,y,shared,extra,c):
    rows=[]
    for start in range(0,len(shared),c['bootstrap']['processing_chunk']):
        a=frequencies(shared[start:start+c['bootstrap']['processing_chunk']]);b=frequencies(extra[start:start+c['bootstrap']['processing_chunk']])
        # Scenario-major layout; baseline extra slots have zero multiplicity.
        f=np.concatenate([np.c_[a,np.zeros_like(a)]]+[np.c_[a,b]]*6,axis=1)
        rows.append(np.stack([balanced_spearman(x[None,:,m,:],y[None],GROUPS,f[:,None,:],
            minimum_per_group=c['minimum_valid_per_scenario'],minimum_response_range=c['minimum_response_range']) for m in range(6)],axis=2))
    return np.concatenate(rows)


def bootstrap():
    c,out,data=paths();prep=json.loads((out/'logs/12_preparation.json').read_text());checked(prep)
    assert not (out/'logs/12_bootstrap_audit.json').exists(),'Completed bootstrap retained'
    z=np.load(data/'12_prepared_arrays.npz');ix=list(z['parameter_ids']).index(c['numerical_reference']);x=z['x'];y=z['y'][:,ix];original=z['rho'][:,:,ix];s=c['bootstrap']
    eligible=np.isfinite(original);done=np.zeros_like(eligible);steady=np.zeros_like(eligible,int)
    limits=np.full((3,88,6),np.nan);previous=limits.copy();used=np.zeros_like(eligible,int);fraction=np.full_like(original,np.nan);blocks=[];history=[];total=0
    while total<s['maximum_draws'] and (eligible&~done).any():
        n=min(s['initial_draws'] if not total else s['batch_draws'],s['maximum_draws']-total)
        p=data/f'bootstrap_blocks/12_draws_{total+1:05d}_{total+n:05d}.npz';binding=p.with_suffix('.json')
        if p.exists():
            a=json.loads(binding.read_text());assert a['sha256']==sha(p) and a['preparation_sha256']==sha(out/'logs/12_preparation.json');block=np.load(p)['rho']
        else:
            started=time.perf_counter();block=bootstrap_block(x,y,z['shared_indices'][total:total+n],z['extra_indices'][total:total+n],c)
            np.savez_compressed(p,rho=block);write(binding,dict(sha256=sha(p),preparation_sha256=sha(out/'logs/12_preparation.json'),seconds=time.perf_counter()-started))
        blocks.append(block);total+=n;samples=np.concatenate(blocks)
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);q=np.nanquantile(samples,[.025,.5,.975],axis=0,method='linear')
        valid=np.isfinite(samples).mean(axis=0);change=abs(q-previous);active=eligible&~done
        stable=eligible&(valid>=s['minimum_valid_fraction'])&np.isfinite(change).all(axis=0)&(change[1]<=s['central_tolerance'])&(np.maximum(change[0],change[2])<=s['endpoint_tolerance'])
        steady[active]=np.where(stable[active],steady[active]+1,0);new=active&(steady>=s['consecutive_stable_batches'])
        limits[:,new]=q[:,new];used[new]=total;fraction[new]=valid[new];done[new]=True
        for k,m in np.argwhere(active):history.append(dict(H=z['H'][k],metric=c['metrics'][m],draws=total,valid_fraction=valid[k,m],low=q[0,k,m],centre=q[1,k,m],high=q[2,k,m],low_change=change[0,k,m],centre_change=change[1,k,m],high_change=change[2,k,m],stable_batches=steady[k,m],converged=bool(done[k,m])))
        pd.DataFrame(history).to_csv(out/'tables/12_bootstrap_convergence.csv',index=False)
        write(out/'logs/12_bootstrap_progress.json',dict(draws=total,eligible=int(eligible.sum()),converged=int(done.sum())))
        print(f'Bootstrap {total}: {done.sum()}/{eligible.sum()} reference intervals stable',flush=True);previous=q
    unresolved=eligible&~done;used[unresolved]=total;fraction[unresolved]=valid[unresolved]
    np.savez_compressed(data/'12_bootstrap_results.npz',limits=limits,draws_used=used,valid_fraction=fraction,converged=done,eligible=eligible,total_draws=total)
    # Literal seed repetition and independent scalar weighted ranks. Reproduce
    # every draw for three full interval cells, not just selected quantiles.
    candidates=np.argwhere(done);selected=candidates[np.linspace(0,len(candidates)-1,min(3,len(candidates)),dtype=int)] if len(candidates) else []
    verification=[];error=0.
    for k,m in selected:
        n=int(used[k,m]);rr=[]
        for draw in range(n):
            a=z['shared_indices'][draw].astype(int);b=z['extra_indices'][draw].astype(int)+8
            literal=np.concatenate([a]+[np.r_[a,b]+16*j for j in range(1,7)])
            # independent() expects112 slots: integer multiplicities are exactly
            # the literal repeated-index sample; np.add.at makes this explicit.
            f=np.zeros(112);np.add.at(f,literal,1)
            rr.append(independent(x[k,m],y[k],f,c))
        rr=np.array(rr);np.testing.assert_allclose(rr,samples[:n,k,m],atol=1e-13,rtol=0,equal_nan=True)
        qq=np.nanquantile(rr,[.025,.5,.975],method='linear');e=float(abs(qq-limits[:,k,m]).max());assert e<1e-13;error=max(error,e)
        verification.append(dict(H=float(z['H'][k]),metric=c['metrics'][m],draws=n,maximum_quantile_error=e))
    pd.DataFrame(verification).to_csv(out/'tables/12_independent_interval_verification.csv',index=False)
    write(out/'logs/12_bootstrap_audit.json',dict(status='adaptive_reference_bootstrap_verified',total_draws=total,eligible=int(eligible.sum()),converged=int(done.sum()),withheld=int(unresolved.sum()),independent_full_intervals=len(verification),maximum_quantile_error=error,input_sha256={key(out/'logs/12_preparation.json'):sha(out/'logs/12_preparation.json')},output_sha256={key(p):sha(p) for p in [data/'12_bootstrap_results.npz',out/'tables/12_bootstrap_convergence.csv',out/'tables/12_independent_interval_verification.csv']}))


def summaries(rho,mask=None):
    a=np.where(mask[:,None,:],rho,np.nan) if mask is not None else rho
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        strength=np.nanmedian(abs(a),axis=0);signed=np.nanmedian(a,axis=0)
    n=np.isfinite(a).sum(axis=0);pos=(a>0).sum(axis=0);neg=(a<0).sum(axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):consistency=np.maximum(pos,neg)/n
    return strength,signed,consistency,n


def draw_curves(c,out,z,b,profile,ci):
    p=list(z['parameter_ids']).index(profile);apply_final_style()
    fig,axes=plt.subplots(3,2,figsize=figure_size(175),sharex=True,sharey=True)
    fig.subplots_adjust(left=.13,right=.98,top=.96,bottom=.19,hspace=.40,wspace=.22)
    for m,ax in enumerate(axes.flat):
        ax.axhline(0,color='#999999',lw=.6)
        if ci:ax.fill_between(z['H'],b['limits'][0,:,m],b['limits'][2,:,m],color='#377A9F',alpha=.22,lw=0)
        ax.plot(z['H'],z['rho'][:,m,p],color='#377A9F',lw=1.4)
        ax.set_xlim(1,0);ax.set_ylim(-1,1);ax.set_xticks([1,.5,0]);ax.set_yticks([-1,-.5,0,.5,1]);ax.spines[['top','right']].set_visible(False)
        ax.text(0,1.06,f'({chr(97+m)}) '+LABELS[m],transform=ax.transAxes,fontsize=9.5)
    fig.supxlabel(r'Habitat proportion $H$');fig.supylabel(r'Scenario-balanced Spearman $r_s$')
    handles=[Line2D([],[],color='#377A9F',label='Equal-scenario weighted Spearman')]
    if ci:handles.append(Patch(facecolor='#377A9F',alpha=.22,label='95% pointwise whole-seed bootstrap CI'))
    handles.append(Line2D([],[],ls='none',label=rf'$r={z["r"][p]:.3g}$, $\ell_{{1/2}}={z["half_distance"][p]:.3g}$ cells'))
    fig.legend(handles=handles,loc='lower center',frameon=False,ncol=1)
    stem=out/f'figures/12_curves_{profile}';export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig);return stem


def draw_surface(c,out,design,ell,rr,fields,kind,diagnostic=False):
    signed=kind=='median_signed';apply_final_style()
    fig,axes=plt.subplots(3,2,figsize=figure_size(190),sharex=True,sharey=True)
    fig.subplots_adjust(left=.14,right=.80,top=.96,bottom=.17,hspace=.40,wspace=.20)
    cmap=plt.get_cmap('RdBu_r' if signed else 'cividis').copy();cmap.set_bad(MISSING)
    norm=plt.Normalize(-1 if signed else 0,1)
    for m,ax in enumerate(axes.flat):
        ax.set_facecolor(MISSING);ax.pcolormesh(ell,rr,np.ma.masked_invalid(fields[m]),cmap=cmap,norm=norm,shading='nearest',rasterized=True)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1,512);ax.set_ylim(10**-2.5,10**5.75)
        ax.set_xticks([1,4,16,64,256],['1','4','16','64','256']);ax.set_yticks([.01,1,100,10000],[r'$10^{-2}$',r'$10^0$',r'$10^2$',r'$10^4$']);ax.minorticks_off()
        ax.text(0,1.06,f'({chr(97+m)}) '+LABELS[m],transform=ax.transAxes,fontsize=9.5)
        if diagnostic:ax.scatter(design.half_distance,design.r,s=6,facecolors='none',edgecolors=NEUTRAL,linewidths=.35)
    labels=dict(median_absolute=r'Median $|r_s|$ across informative $H$',median_signed=r'Median signed $r_s$ across informative $H$',sign_consistency='Dominant-sign fraction across informative H',informative_fraction='Fraction of 88 H levels with defined association')
    cax=fig.add_axes([.86,.21,.025,.69]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-1,-.5,0,.5,1] if signed else [0,.25,.5,.75,1],label=labels[kind])
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)');fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    handles=[Patch(facecolor=MISSING,label='Outside sampled support or undefined summary')]
    if diagnostic:handles.append(Line2D([],[],ls='none',marker='o',ms=3,color=NEUTRAL,markerfacecolor='none',label='Sampled profile'))
    fig.legend(handles=handles,loc='lower center',frameon=False,ncol=1)
    stem=out/f'{"review" if diagnostic else "figures"}/12_{kind}{"_sample_locations" if diagnostic else ""}'
    if not diagnostic:export_figure(fig,stem)
    else:
        from src.tools.final_plot_style import compact_shared_labels
        compact_shared_labels(fig)
    fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig);return stem


def export():
    c,out,data=paths();prep=json.loads((out/'logs/12_preparation.json').read_text());boot=json.loads((out/'logs/12_bootstrap_audit.json').read_text());checked(prep);checked(boot)
    assert not (out/'logs/12_final_audit.json').exists(),'Final exports already exist'
    z=dict(np.load(data/'12_prepared_arrays.npz'));b=dict(np.load(data/'12_bootstrap_results.npz'));design=pd.read_csv(data/'12_parameter_design.csv',float_precision='round_trip')
    rho=z['rho'];strength,signed,consistency,n=summaries(rho)
    span=np.max(np.where(np.isfinite(z['y']),z['y'],-np.inf),axis=-1)-np.min(np.where(np.isfinite(z['y']),z['y'],np.inf),axis=-1)
    sensitivity=summaries(rho,span>=c['response_range_sensitivity'])
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);loo=np.nanmax(abs(z['omitted']-rho),axis=0)
    tie=abs(rho-z['rho_rounded']);match=abs(rho-z['rho_matched8']);initial=abs(rho-z['rho_initial5'])
    curves=[];summary=[];omission=[];reference=list(z['parameter_ids']).index(c['numerical_reference'])
    for k,h in enumerate(z['H']):
        for m,metric in enumerate(c['metrics']):
            for p,pid in enumerate(z['parameter_ids']):
                row=dict(H=h,metric=metric,parameter_id=pid,spearman=rho[k,m,p],occupancy_span=span[k,p] if np.isfinite(span[k,p]) else np.nan,maximum_seed_omission_change=loo[k,m,p],rho_matched8=z['rho_matched8'][k,m,p],rho_initial5=z['rho_initial5'][k,m,p],rho_rounded=z['rho_rounded'][k,m,p],omission_lost_estimate_count=int((np.isfinite(rho[k,m,p])&~np.isfinite(z['omitted'][:,k,m,p])).sum()))
                if p==reference:row.update(ci_low=b['limits'][0,k,m],ci_high=b['limits'][2,k,m],bootstrap_median=b['limits'][1,k,m],bootstrap_draws=int(b['draws_used'][k,m]),bootstrap_valid_fraction=b['valid_fraction'][k,m],bootstrap_converged=bool(b['converged'][k,m]))
                curves.append(row)
    for m,metric in enumerate(c['metrics']):
        for p,row in design.iterrows():summary.append(dict(parameter_id=row.parameter_id,r=row.r,half_distance=row.half_distance,metric=metric,n_informative_H=int(n[m,p]),median_absolute_spearman=strength[m,p],median_signed_spearman=signed[m,p],sign_consistency=consistency[m,p],n_positive_H=int((rho[:,m,p]>0).sum()),n_negative_H=int((rho[:,m,p]<0).sum()),n_zero_H=int((rho[:,m,p]==0).sum()),n_span_at_least_005_H=int(sensitivity[3][m,p]),span005_median_absolute=sensitivity[0][m,p],span005_median_signed=sensitivity[1][m,p],span005_sign_consistency=sensitivity[2][m,p]))
    for seed in range(16):
        for k,h in enumerate(z['H']):
            for m,metric in enumerate(c['metrics']):omission.append(dict(seed=seed+1,cohort='shared' if seed<8 else 'directed_only',H=h,metric=metric,reference_spearman=z['omitted'][seed,k,m,reference],maximum_change_across_profiles=float(np.nanmax(abs(z['omitted'][seed,k,m]-rho[k,m]))) if np.isfinite(z['omitted'][seed,k,m]).any() else np.nan))
    pd.DataFrame(curves).to_csv(out/'tables/12_within_H_associations.csv',index=False);pd.DataFrame(summary).to_csv(out/'tables/12_parameter_metric_summaries.csv',index=False);pd.DataFrame(omission).to_csv(out/'tables/12_whole_seed_omission.csv',index=False)
    metric_report=[]
    for m,metric in enumerate(c['metrics']):
        a=strength[m];metric_report.append(dict(metric=metric,n_profiles=int(np.isfinite(a).sum()),median_across_profiles_of_median_absolute=float(np.nanmedian(a)),q05_across_profiles=float(np.nanquantile(a,.05)),q95_across_profiles=float(np.nanquantile(a,.95)),median_sign_consistency=float(np.nanmedian(consistency[m])),reference_median_absolute=float(strength[m,reference]),reference_median_signed=float(signed[m,reference]),reference_sign_consistency=float(consistency[m,reference]),reference_informative_H=int(n[m,reference])))
    report=pd.DataFrame(metric_report);report.to_csv(out/'tables/12_metric_overview.csv',index=False)
    points,low,high=B.normalized_coordinates(design,dict(half_distance_min=1,half_distance_max=512,log10_r_min=-2.5,log10_r_max=5.75));tri=Delaunay(points)
    gx,gy=np.linspace(0,1,130),np.linspace(0,1,160);xx,yy=np.meshgrid(gx,gy);targets=np.c_[xx.ravel(),yy.ravel()]
    ell=10**(low[0]+gx*(high[0]-low[0]));rr=10**(low[1]+gy*(high[1]-low[1]));ee,rs=np.meshgrid(ell,rr)
    strips=pd.read_csv(FINAL/c['domain_strips'],float_precision='round_trip');inside=U.inner_mask(np.log(ee),np.log(rs),strips)
    stems=[draw_curves(c,out,z,b,c['numerical_reference'],True)]
    if prep['empirical_example']!=c['numerical_reference']:stems.append(draw_curves(c,out,z,b,prep['empirical_example'],False))
    fields={};node_error=0.
    for kind,values in [('median_absolute',strength),('median_signed',signed),('sign_consistency',consistency),('informative_fraction',n/88)]:
        interpolator=LinearNDInterpolator(tri,values.T);field=interpolator(targets).T.reshape(6,len(gy),len(gx));field[:,~inside]=np.nan
        error=float(np.nanmax(abs(interpolator(points).T-values)));node_error=max(node_error,error)
        assert error<1e-12 and np.nanmin(field)>=(-1 if kind=='median_signed' else 0)-1e-12 and np.nanmax(field)<=1+1e-12
        fields[kind]=field;stems.append(draw_surface(c,out,design,ell,rr,field,kind));draw_surface(c,out,design,ell,rr,field,kind,True)
    np.savez_compressed(data/'12_parameter_surfaces.npz',grid_half_distance=ell,grid_r=rr,domain_mask=inside,**fields)
    fontchecks=B.export_checks(stems)
    maxloo=float(np.nanmax(loo));sensitive=int((loo>c['seed_omission_tolerance']).sum());eligible=int(np.isfinite(rho).sum());lost=int((np.isfinite(rho)[None]&~np.isfinite(z['omitted'])).sum())
    text=f'''# Result D: structural associations at fixed habitat proportion

We related each of the six retained structural metrics to mean equilibrium patch occupancy separately at every stored habitat proportion and for every one of the 256 fixed numerical parameter profiles. The source comprises 104 complete trajectories (eight Only-Random and 16 for each directed scenario), 88 habitat levels and 9,152 snapshots. No landscape, parameter profile or equilibrium was newly simulated. All 104 saved occupancy trajectories were hash-checked and reproduced the consumed response array exactly. Snapshot grid hashes and patch counts matched the structural data at all 9,152 identities. Producer sources and the immutable landscape table were checked against the production manifest.

## Equal-scenario estimand and analysis domain

The author confirmed equal scenario weighting on 15 October 2026. Within an H/profile/metric cell, each of the seven scenarios receives mass 1/7, shared equally among its valid paired landscapes. For a value v, its weighted rank is F(v-) + one half the mass tied at v. We take the weighted Pearson correlation of these two weighted mid-distribution ranks. Both the ranks and their correlation use scenario weights. This is a weighted Spearman association under an equal seven-scenario mixture; ordinary unweighted pooling of 8/16 observations would target a different mixture. Ties are retained with average mass ranks, without jitter. The shared-eight sensitivity converges to ordinary Spearman where all eight landscapes in each scenario are valid; independent SciPy checks had maximum error {prep['maximum_shared8_scipy_error']:.3g}.

Only landscapes with at least two patches and finite paired quantities are included. To retain the same seven-scenario target, a coefficient additionally requires at least three valid paired landscapes per scenario. H=0 and1 are excluded; H=.975 is withheld because Only-Random supplies only two valid landscapes. The primary domain is H=.005–.925. At H=.925, Only-Random supplies five of eight landscapes, so interpretation is conditional on the remaining fragmented landscapes. Missing or unplanned trajectories are never coded as extinction. The three-observation rule screens missingness and does not establish precision. Structural metrics at excluded landscapes remain missing even when a stored convention supplies a number.

A response range at or below 1e-10 occupancy units, or a constant metric, makes an association undefined. We retain all 256 parameter profiles, including the nine with narrow overall median response spans. Floor and ceiling regions can give poorly informative or undefined associations; their absence is not evidence that spatial configuration is irrelevant. Tables record the within-cell occupancy span, all defined-H counts, and a separate sensitivity restricting summaries to spans at least .05. Rounding responses to1e-8 is a numerical tie sensitivity only (maximum comparable coefficient change {np.nanmax(tie):.6f}); it does not replace exact saved-value ranks. The saved edge-density normalisation differs from a cell-area convention by a global positive factor on this fixed raster; rank correlations are invariant to that factor and no absolute edge-density values are reported here.

## Summaries and species dependence

For each profile and metric, we give the median absolute correlation, median signed correlation and dominant-sign fraction across defined H levels. Every defined sampled level has equal weight; these are not H integrals or prevalence estimates. The dominant-sign fraction is max(number positive, number negative)/number defined; exact zeros remain in the denominator. Positive/negative/zero counts and summary availability accompany the values. Summaries of strength must be read with sign and H curves. The source parameter distribution is the adopted stratified conditional log-area sample, not a rectangular LHS or a population distribution of biological species. Each profile retains design weight1/256. Observed bounds: r={prep['parameter_ranges']['r'][0]:.8g}–{prep['parameter_ranges']['r'][1]:.8g}, dispersal half-distance={prep['parameter_ranges']['half_distance'][0]:.8g}–{prep['parameter_ranges']['half_distance'][1]:.8g} cell-coordinate units. No physical calibration is implied.

Parameter maps use piecewise-linear Delaunay interpolation in normalized log10(r), log10(half-distance), with the adopted exact stepwise domain and the sampled convex hull both masking support. We never extrapolate. All sampled summaries are supplied in the CSV, and all maps have diagnostic versions showing sample locations. Maximum source-node recovery error was {node_error:.3g}. Interpolated sign fractions and counts describe displays, not new observations. The numerical reference is domain131, as selected in Result B. The optional empirical illustration is {prep['empirical_example']}, the most frequent best sampled profile in the existing Result-C comparison ({', '.join(prep['empirical_example_species'])}); this display rule uses no Result-D coefficient. It is an illustrative numerical match, not an inferred demographic parameter or universally preferred bird profile. Only the numerical reference receives bootstrap intervals.

## Pointwise bootstrap and seed influence

The preconfigured bootstrap independently resamples two size-eight cohorts: shared seeds1–8 jointly across all seven scenarios, all H and metric values; directed-only seeds9–16 jointly across the six directed scenarios. No parameter profile is resampled. Each draw recomputes scenario weights, both weighted ranks and their correlation; missing rows stay in their original clusters. At least three valid observations counting bootstrap multiplicities per scenario are required. Baseline9–16 trajectories are never created. Saved index matrices use seed2026101504. Percentile limits use linear .025/.975 quantiles of defined draws, conditional on estimability. The intervals are pointwise, not simultaneous.

The initial/increment batches are1000, maximum20000, endpoint tolerance.01, bootstrap-median tolerance.005, with three consecutive stable comparisons and at least95% defined draws. Each eligible cell freezes at its first stable batch. {boot['converged']}/{boot['eligible']} reference intervals converged within {boot['total_draws']} draws; {boot['withheld']} were withheld. Three full interval cells were independently reconstructed using literal seed indices and scalar pairwise rank sums; maximum quantile error {boot['maximum_quantile_error']:.3g}. Nonconverged limits remain missing. Numerical bootstrap convergence does not establish sufficient landscape replication.

Across all profiles, the largest whole-seed omission change was {maxloo:.6f}; {sensitive}/{eligible} coefficients exceeded the prespecified .05 influence tolerance under at least one omission. Omission made an originally defined coefficient unavailable in {lost} omission/cell combinations, recorded separately. Maximum shared-eight to8/16 change: {np.nanmax(match):.6f}; initial-five to8/16: {np.nanmax(initial):.6f}. Complete-cohort sensitivities are available alongside every original coefficient; these are descriptive checks, not additional uncertainty intervals.

## Descriptive findings

The overview table reports association strength across all profiles as well as the numerical reference. This comparison is conditional on the examined H raster and parameter distribution. Pooled associations can reflect both between-scenario differences and within-scenario variation. Equal weighting does not control for scenario membership. No p-values, causal metric effects, universal metric ranking or out-of-sample predictive performance are claimed. Metapopulation capacity is omitted because it is an optional species-dependent diagnostic and adds no required output here.

{report.to_csv(index=False)}

## Figure captions

**Reference curves.** Scenario-balanced weighted Spearman associations of structural metrics with mean equilibrium patch occupancy at each habitat proportion, for numerical reference domain131. Each panel names one retained metric. NN denotes nearest-neighbour distance. Lines show original estimates; ribbons show95% pointwise percentile intervals from the two-cohort whole-seed bootstrap above. The source has eight Only-Random and16 trajectories per other scenario. Only valid multi-patch landscapes contribute, every scenario has equal total mass, and at least three valid paired observations per scenario are required. H decreases from1 to0; undefined estimates and withheld intervals remain gaps. Both rank variables and the Pearson correlation use the same weights.

**Empirically informed curves.** The same structural associations for the most frequently selected numerical match in the existing15-bird comparison. Lines show weighted Spearman estimates without uncertainty intervals. The profile is an illustration from the fixed numerical sample; its selection does not estimate biological demographic parameters. Domain, weights and exclusions follow the reference figure.

**Parameter maps.** Panels show the statistic named on the shared colour bar for each structural metric across the256-profile conditional numerical design. The strength map gives median absolute within-H association, the signed map its median direction, the consistency map its dominant-sign fraction, and the availability map its defined-H fraction out of88 stored levels. H summaries weight defined sampled levels equally. Absolute strengths and fractions use0–1, signed correlations use-1–1. Values use linear interpolation in the two log coordinates within the adopted sampled support; grey indicates unavailable support or an undefined statistic. Diagnostic PNGs mark all sampled parameter locations. Read strength together with sign, availability and the underlying H curves.

## Reproduction and review

Use `.venv/bin/python -m src.tools.codex_run final-result-d-metric-associations --stage prepare`, followed by bootstrap and export. Bootstrap blocks have immutable source/configuration receipts. After all six PDFs and four sample-location diagnostics have actually been rendered and inspected, save review/12_actual_visual_review.json and run stage confirm. Manuscript integration and review at its final embedded size remain subsequent editorial steps.
'''
    (out/'12_methods_results_and_captions.md').write_text(text)
    outputs=list((out/'tables').glob('*.csv'))+list((out/'figures').glob('*'))+[out/'12_methods_results_and_captions.md',data/'12_parameter_surfaces.npz']
    write(out/'logs/12_final_audit.json',dict(status='numerically_checked_visual_review_pending',visual_review_confirmed=False,n_figures=len(stems),maximum_interpolation_error=node_error,maximum_seed_omission_change=maxloo,n_seed_sensitive_coefficients=sensitive,n_eligible_coefficients=eligible,omission_unavailable_combinations=lost,bootstrap=boot,font_checks=fontchecks,input_sha256={key(p):sha(p) for p in [CONFIG,Path(__file__),out/'logs/12_preparation.json',out/'logs/12_bootstrap_audit.json',FINAL/'code/base/src/tools/final_plot_style.py']},output_sha256={key(p):sha(p) for p in outputs}))
    print(report.to_string(index=False),flush=True)


def confirm():
    c,out,data=paths();p=out/'logs/12_final_audit.json';a=json.loads(p.read_text());checked(a)
    review=out/'review/12_actual_visual_review.json';r=json.loads(review.read_text());assert r['all_six_pdfs_rendered_and_viewed'] and r['all_four_sample_location_diagnostics_viewed']
    a.update(status='numerically_and_visually_checked',visual_review_confirmed=True,review_sha256=sha(review));write(p,a)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','bootstrap','export','confirm'],required=True)
    globals()[parser.parse_args().stage]()
