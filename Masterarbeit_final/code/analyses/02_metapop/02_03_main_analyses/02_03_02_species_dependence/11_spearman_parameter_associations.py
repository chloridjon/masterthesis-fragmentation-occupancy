"""Fixed-design parameter Spearman curves with two-cohort whole-seed CIs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import importlib
import json
import sys
import time
import warnings

FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent))
M=importlib.import_module('11_integrated_occupancy_contrast');B=M.B
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
from src.tools import cohort_rank_bootstrap as R
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,FOCAL_SCENARIOS,NEUTRAL
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REL='02_metapop/02_03_main_analyses/02_03_02_species_dependence'
CONFIG=FINAL/f'config/{REL}/spearman_parameter_associations/11_spearman.json'
PARAMETERS=['log10_r','log10_half_distance']


def provenance_key(path):
    return str(Path(path).relative_to(FINAL,walk_up=True))


def paths():
    c=json.loads(CONFIG.read_text());out,data=(FINAL/c[k] for k in ['results_directory','data_directory'])
    for p in [out/'logs',out/'tables',out/'figures',out/'review',data,data/'bootstrap_blocks']:p.mkdir(parents=True,exist_ok=True)
    return c,out,data


def check_hashes(a):
    for name,digest in {**a.get('input_sha256',{}),**a.get('output_sha256',{})}.items():assert B.sha(FINAL/name)==digest,name


def bootstrap_chunk(values,predictors,shared,extra,settings,c):
    """One index matrix per cohort, reused across every H and profile."""
    base=R.PreparedSeedMedian(values[:8,0])
    directed=[R.PreparedSeedMedian(values[:,j]) for j in range(1,7)]
    output=[]
    for start in range(0,len(shared),settings['processing_chunk']):
        stop=min(start+settings['processing_chunk'],len(shared))
        w=R.frequencies(shared[start:stop]);v=R.frequencies(extra[start:stop]);full=np.c_[w,v]
        bm,_=base.evaluate(w,c['minimum_valid_landscape_seeds'])
        med=np.stack([p.evaluate(full,c['minimum_valid_landscape_seeds'])[0] for p in directed],axis=1)
        output.append(R.fixed_design_spearman(med-bm[:,None],predictors,c['minimum_contrast_range']))
    return np.concatenate(output)


def prepare():
    c,out,data=paths();receipt=out/'logs/11_preparation.json'
    assert not receipt.exists(),'Preparation already exists; use bootstrap/export stages'
    snap=FINAL/c['source_snapshot'];a=json.loads((snap/'11_numerical_audit.json').read_text());v=json.loads((snap/'11_export_audit.json').read_text())
    assert a['n_trajectories']==104 and a['n_parameters']==256 and v['visual_review_confirmed']
    assert v['numeric_audit_sha256']==B.sha(snap/'11_numerical_audit.json')
    inputs=[snap/'tables'/f for f in ['11_response_arrays.npz','11_response_summaries.csv','11_profile_response_diagnostics.csv']]
    for p in inputs:assert B.sha(p)==a['output_sha256'][str(p.relative_to(FINAL))],p
    df=pd.read_csv(inputs[1],float_precision='round_trip');design=pd.read_csv(inputs[2],float_precision='round_trip');h=df.H_target.drop_duplicates().to_numpy()
    with np.load(inputs[0]) as z:values=z['values'];med=z['quantiles'][1];counts=z['valid_counts'];planned=z['planned_seed_mask']
    assert values.shape==(16,7,88,256) and planned.sum()==104
    assert planned[:8].all() and not planned[8:,0].any() and planned[8:,1:].all() and not np.isfinite(values[~planned]).any()
    np.testing.assert_allclose(M.seed_medians(values),med,atol=1e-14,rtol=0)
    error=0.
    for j,s in enumerate(FOCAL_SCENARIOS):
        saved=df[df.scenario_id==s['scenario_id']].pivot(index='H_target',columns='parameter_id',values='median').reindex(index=h,columns=design.parameter_id).to_numpy()
        np.testing.assert_allclose(saved,med[j],atol=1e-14,rtol=0,equal_nan=True)
    predictors=np.array([np.log10(design.r.to_numpy()),np.log10(design.half_distance.to_numpy())])
    delta=med[1:]-med[0];estimate=R.fixed_design_spearman(delta,predictors,c['minimum_contrast_range'])
    for j,k,p in np.argwhere(np.isfinite(estimate)):
        expected=spearmanr(predictors[p],delta[j,k]).statistic;error=max(error,abs(float(expected-estimate[j,k,p])))
    assert error<1e-14
    rounded=R.fixed_design_spearman(np.round(delta/c['numerical_tie_sensitivity_rounding'])*c['numerical_tie_sensitivity_rounding'],predictors,c['minimum_contrast_range'])
    eight=M.seed_medians(values[:8]);five=M.seed_medians(values[:5])
    rho8=R.fixed_design_spearman(eight[1:]-eight[0],predictors,c['minimum_contrast_range']);rho5=R.fixed_design_spearman(five[1:]-five[0],predictors,c['minimum_contrast_range'])
    omitted=[];rows=[]
    for seed in range(16):
        m=M.seed_medians(np.delete(values,seed,axis=0))
        if seed>=8:np.testing.assert_array_equal(m[0],med[0])
        r=R.fixed_design_spearman(m[1:]-m[0],predictors,c['minimum_contrast_range']);omitted.append(r)
        for j,s in enumerate(FOCAL_SCENARIOS[1:]):
            for k,hh in enumerate(h):
                for p,param in enumerate(PARAMETERS):rows.append(dict(omitted_seed=seed+1,cohort='shared_1_to_8' if seed<8 else 'directed_only_9_to_16',scenario_id=s['scenario_id'],H=hh,parameter=param,spearman=r[j,k,p],absolute_change=abs(r[j,k,p]-estimate[j,k,p])))
    pd.DataFrame(rows).to_csv(out/'tables/11_whole_seed_omission.csv',index=False)
    omitted=np.array(omitted)
    shared,extra=R.two_cohort_indices(c['bootstrap']['maximum_draws'],c['bootstrap']['seed'])
    np.savez_compressed(data/'11_prepared_arrays.npz',H=h,values=values,medians=med,valid_counts=counts,predictors=predictors,estimates=estimate,
        parameter_ids=design.parameter_id.to_numpy(str),rho_initial5=rho5,rho_matched8=rho8,rho_tie_sensitivity=rounded,omitted_seed_estimates=omitted)
    np.savez_compressed(data/'11_bootstrap_indices.npz',shared=shared,extra=extra)
    # Timing pilot uses the exact configured index prefix, later reused verbatim.
    start=time.perf_counter();pilot=bootstrap_chunk(values,predictors,shared[:32],extra[:32],c['bootstrap'],c);elapsed=time.perf_counter()-start
    np.savez_compressed(data/'bootstrap_blocks/11_timing_prefix_00032.npz',estimates=pilot)
    inputs += [CONFIG,Path(__file__),Path(M.__file__),Path(R.__file__),snap/'11_numerical_audit.json',snap/'11_export_audit.json']
    outputs=[data/'11_prepared_arrays.npz',data/'11_bootstrap_indices.npz',data/'bootstrap_blocks/11_timing_prefix_00032.npz',out/'tables/11_whole_seed_omission.csv']
    prep=dict(status='prepared_and_independently_checked',created_utc=datetime.now(timezone.utc).isoformat(),n_parameters=256,n_H=88,n_eligible_coefficients=int(np.isfinite(estimate).sum()),
        maximum_scipy_coefficient_error=error,parameter_design_spearman=float(spearmanr(*predictors).statistic),timing_pilot_draws=32,timing_pilot_seconds=elapsed,
        input_sha256={provenance_key(p):B.sha(p) for p in inputs},output_sha256={provenance_key(p):B.sha(p) for p in outputs})
    B.write_json(receipt,prep);print(json.dumps({k:prep[k] for k in ['n_eligible_coefficients','maximum_scipy_coefficient_error','parameter_design_spearman','timing_pilot_seconds']}),flush=True)


def bootstrap():
    c,out,data=paths();prep=json.loads((out/'logs/11_preparation.json').read_text());check_hashes(prep)
    assert not (out/'logs/11_bootstrap_audit.json').exists(),'Bootstrap already complete'
    z=np.load(data/'11_prepared_arrays.npz');values,predictors,estimate=z['values'],z['predictors'],z['estimates'];h=z['H']
    ix=np.load(data/'11_bootstrap_indices.npz');shared,extra=ix['shared'],ix['extra'];settings=c['bootstrap']
    eligible=np.isfinite(estimate);done=np.zeros_like(eligible);steady=np.zeros_like(eligible,int)
    low=np.full(estimate.shape,np.nan);high=low.copy();centre=low.copy();draws=np.zeros(estimate.shape,int);fraction=low.copy()
    previous=np.full((3,*estimate.shape),np.nan);blocks=[];history=[];total=0
    while total<settings['maximum_draws'] and np.any(eligible&~done):
        n=min(settings['initial_draws'] if total==0 else settings['batch_draws'],settings['maximum_draws']-total)
        path=data/f'bootstrap_blocks/11_draws_{total+1:05d}_{total+n:05d}.npz';binding=path.with_suffix('.json')
        if path.exists():
            a=json.loads(binding.read_text());assert a['sha256']==B.sha(path) and a['preparation_sha256']==B.sha(out/'logs/11_preparation.json')
            block=np.load(path)['estimates']
        else:
            start=time.perf_counter();block=bootstrap_chunk(values,predictors,shared[total:total+n],extra[total:total+n],settings,c)
            if total==0:np.testing.assert_allclose(block[:32],np.load(data/'bootstrap_blocks/11_timing_prefix_00032.npz')['estimates'],atol=0,rtol=0,equal_nan=True)
            np.savez_compressed(path,estimates=block)
            B.write_json(binding,dict(sha256=B.sha(path),preparation_sha256=B.sha(out/'logs/11_preparation.json'),seconds=time.perf_counter()-start))
        assert block.shape==(n,6,88,2);blocks.append(block);total+=n;samples=np.concatenate(blocks)
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);q=np.nanquantile(samples,[.025,.5,.975],axis=0,method='linear')
        valid=np.isfinite(samples).mean(axis=0);change=abs(q-previous)
        stable=eligible&(valid>=settings['minimum_valid_fraction'])&np.isfinite(change).all(axis=0)
        stable&=(change[1]<=settings['central_tolerance'])&(np.maximum(change[0],change[2])<=settings['endpoint_tolerance'])
        active=eligible&~done;steady[active]=np.where(stable[active],steady[active]+1,0)
        new=active&(steady>=settings['consecutive_stable_batches'])
        low[new]=q[0][new];centre[new]=q[1][new];high[new]=q[2][new];draws[new]=total;fraction[new]=valid[new];done[new]=True
        for j,k,p in np.argwhere(active):history.append(dict(scenario_id=FOCAL_SCENARIOS[j+1]['scenario_id'],H=h[k],parameter=PARAMETERS[p],draws=total,valid_fraction=valid[j,k,p],raw_low=q[0,j,k,p],raw_centre=q[1,j,k,p],raw_high=q[2,j,k,p],low_change=change[0,j,k,p],central_change=change[1,j,k,p],high_change=change[2,j,k,p],stable_batches=steady[j,k,p],converged=done[j,k,p]))
        pd.DataFrame(history).to_csv(out/'tables/11_bootstrap_convergence.csv',index=False)
        B.write_json(out/'logs/11_bootstrap_progress.json',dict(draws=total,eligible=int(eligible.sum()),converged=int(done.sum()),last_completed_block=str(path.relative_to(FINAL))))
        print(f'Bootstrap {total}: {done.sum()}/{eligible.sum()} pointwise intervals stable.',flush=True);previous=q
    unresolved=eligible&~done;draws[unresolved]=total;fraction[unresolved]=valid[unresolved]
    status=np.full(estimate.shape,'undefined_original_association',dtype='U48');status[eligible]='budget_exhausted';status[done]='converged'
    status[unresolved&(fraction<settings['minimum_valid_fraction'])]='insufficient_bootstrap_availability'
    np.savez_compressed(data/'11_bootstrap_results.npz',low=low,high=high,centre=centre,draws_used=draws,valid_fraction=fraction,converged=done,eligible=eligible,status=status,total_draws=total)
    # Independent literal-index resampling reproduces12 complete interval cells.
    cells=np.argwhere(done);chosen=cells[np.linspace(0,len(cells)-1,min(12,len(cells)),dtype=int)] if len(cells) else []
    full_error=0.;quantile_error=0.;verification=[]
    for j,k,p in chosen:
        n=int(draws[j,k,p]);direct_ix=np.c_[shared[:n],extra[:n]+8];direct=values[:,j+1,k];base=values[:8,0,k];result=[]
        for start in range(0,n,128):
            aa=direct[direct_ix[start:start+128]];bb=base[shared[start:min(start+128,n)]]
            with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);am=np.nanmedian(aa,axis=1);bm=np.nanmedian(bb,axis=1)
            am[np.isfinite(aa).sum(axis=1)<3]=np.nan;bm[np.isfinite(bb).sum(axis=1)<3]=np.nan;delta=am-bm
            ranks=rankdata(delta,axis=1,method='average',nan_policy='propagate');xr=rankdata(predictors[p]);xr-=xr.mean();ranks-=ranks.mean(axis=1,keepdims=True)
            with np.errstate(divide='ignore',invalid='ignore'):rr=(ranks*xr).sum(axis=1)/np.sqrt((ranks*ranks).sum(axis=1)*(xr*xr).sum())
            rr[~np.isfinite(delta).all(axis=1)|(np.ptp(delta,axis=1)<=c['minimum_contrast_range'])]=np.nan;result.extend(rr)
        result=np.array(result);np.testing.assert_allclose(result,samples[:n,j,k,p],atol=1e-14,rtol=0,equal_nan=True)
        full_error=max(full_error,float(np.nanmax(abs(result-samples[:n,j,k,p]))));qq=np.nanquantile(result,[.025,.5,.975],method='linear')
        difference=float(abs(qq-[low[j,k,p],centre[j,k,p],high[j,k,p]]).max());quantile_error=max(quantile_error,difference)
        assert difference<1e-14;verification.append(dict(scenario_id=FOCAL_SCENARIOS[j+1]['scenario_id'],H=h[k],parameter=PARAMETERS[p],draws=n,maximum_quantile_error=difference))
    pd.DataFrame(verification).to_csv(out/'tables/11_independent_bootstrap_verification.csv',index=False)
    B.write_json(out/'logs/11_bootstrap_audit.json',dict(status='adaptive_bootstrap_checked',total_draws=total,eligible=int(eligible.sum()),converged=int(done.sum()),withheld=int((eligible&~done).sum()),n_independent_intervals=len(verification),maximum_independent_draw_error=full_error,maximum_independent_quantile_error=quantile_error,
        input_sha256={str((out/'logs/11_preparation.json').relative_to(FINAL)):B.sha(out/'logs/11_preparation.json')},output_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in [data/'11_bootstrap_results.npz',out/'tables/11_bootstrap_convergence.csv',out/'tables/11_independent_bootstrap_verification.csv']}))


def export():
    c,out,data=paths();prep=json.loads((out/'logs/11_preparation.json').read_text());boot=json.loads((out/'logs/11_bootstrap_audit.json').read_text());check_hashes(prep);check_hashes(boot)
    marker=out/'logs/11_spearman_audit.json'
    if marker.exists():assert not json.loads(marker.read_text())['visual_review_confirmed']
    z=np.load(data/'11_prepared_arrays.npz');b=np.load(data/'11_bootstrap_results.npz');h=z['H'];rho=z['estimates'];counts=z['valid_counts'];omitted=z['omitted_seed_estimates']
    rows=[];maxloo=np.nanmax(abs(omitted-rho),axis=0)
    for j,s in enumerate(FOCAL_SCENARIOS[1:]):
        for k,hh in enumerate(h):
            for p,param in enumerate(PARAMETERS):rows.append(dict(scenario_id=s['scenario_id'],H=hh,parameter=param,spearman=rho[j,k,p],ci_low=b['low'][j,k,p],ci_high=b['high'][j,k,p],bootstrap_centre=b['centre'][j,k,p],bootstrap_draws=int(b['draws_used'][j,k,p]),bootstrap_valid_fraction=b['valid_fraction'][j,k,p],bootstrap_status=b['status'][j,k,p],n_parameter_profiles=256 if np.isfinite(z['medians'][[0,j+1],k]).all() else 0,n_baseline_valid=int(counts[0,k]),n_scenario_valid=int(counts[j+1,k]),maximum_seed_omission_change=maxloo[j,k,p],seed_sensitive=bool(maxloo[j,k,p]>c['seed_omission_tolerance']),rho_initial5=z['rho_initial5'][j,k,p],rho_matched8=z['rho_matched8'][j,k,p],rho_rounding_sensitivity=z['rho_tie_sensitivity'][j,k,p]))
    table=pd.DataFrame(rows);table.to_csv(out/'tables/11_spearman_associations.csv',index=False)
    domain=pd.DataFrame([dict(scenario_id=s['scenario_id'],H=hh,n_planned=8 if j==0 else 16,n_valid=int(counts[j,k]),n_excluded=(8 if j==0 else 16)-int(counts[j,k]),structurally_absent=8 if j==0 else 0) for j,s in enumerate(FOCAL_SCENARIOS) for k,hh in enumerate(h)])
    domain.to_csv(out/'tables/11_H_domain_and_exclusions.csv',index=False);stems=[]
    for p,param in enumerate(PARAMETERS):
        apply_final_style();fig,axes=plt.subplots(3,2,figsize=figure_size(190),sharex=True,sharey=True)
        fig.subplots_adjust(left=.13,right=.98,top=.96,bottom=.20,hspace=.42,wspace=.22)
        for j,(ax,s) in enumerate(zip(axes.flat,FOCAL_SCENARIOS[1:])):
            ax.axhline(0,color='#999999',lw=.6)
            ax.fill_between(h,b['low'][j,:,p],b['high'][j,:,p],color=s['color'],alpha=.22,lw=0)
            ax.plot(h,rho[j,:,p],color=s['color'],ls=s['linestyle'],lw=1.4)
            missing=np.isfinite(rho[j,:,p])&~b['converged'][j,:,p]
            ax.scatter(h[missing],rho[j,missing,p],marker='x',color=NEUTRAL,s=9,zorder=3)
            ax.set_xlim(1,0);ax.set_ylim(-1,1);ax.set_xticks([1,.5,0]);ax.set_yticks([-1,-.5,0,.5,1]);ax.spines[['top','right']].set_visible(False)
            ax.text(0,1.06,f'({chr(97+j)}) '+s['label'],transform=ax.transAxes,fontsize=9.5)
        fig.supxlabel(r'Habitat proportion $H$');fig.supylabel(r'Spearman $r_s$ with '+(r'$\log_{10}r$' if p==0 else r'$\log_{10}\ell_{1/2}$'))
        handles=[Line2D([],[],color=NEUTRAL,label='Spearman estimate; scenario line style'),Patch(facecolor='#777777',alpha=.22,label='95% pointwise stratified whole-seed bootstrap CI')]
        if np.any(np.isfinite(rho[:,:,p])&~b['converged'][:,:,p]):handles.append(Line2D([],[],ls='none',marker='x',color=NEUTRAL,label='CI withheld'))
        fig.legend(handles=handles,loc='lower center',frameon=False,ncol=1)
        stem=out/f'figures/11_spearman_{param}';export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig);stems.append(stem)
    fonts=B.export_checks(stems)
    finite=table.spearman.notna();tie=abs(table.spearman-table.rho_rounding_sensitivity);exp8=abs(table.spearman-table.rho_matched8)
    note=f'''# Parameter associations with occupancy contrasts

At each of88 sampled H levels and for each of the six directed scenarios, we calculated the Spearman association across the fixed256 numerical parameter profiles between the profile-specific median occupancy contrast against Only-Random and either log10(r) or log10(dispersal half-distance). The contrast is a difference of scenario medians, with8 Only-Random and16 directed-scenario landscape seeds. Ordinary Spearman ranks correspond to equal profile design weights1/256. Average ranks retain exact ties, including the four profiles at each shared dispersal coordinate. No parameter profile was selected or resampled. Responses missing for any profile invalidate that complete association; contrast ranges at or below1e-10 occupancy units are treated as numerically constant. No p-values were calculated. Rank calculations were checked against [SciPy's spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

These are marginal associations under the adopted joint conditional log-area design, not independent parameter effects. The design correlation between the two parameter coordinates is{prep['parameter_design_spearman']:.6f}; a correlation with one parameter does not control for the other. A small coefficient can also describe a nonmonotonic response. Floor/ceiling profiles remain in the fixed sample, and their absolute occupancy maps should accompany interpretation.

## Bootstrap design fixed before computation

We resampled the shared seed cohort1-8 jointly across all seven scenarios, H levels and profiles. Independently, we resampled cohort9-16 jointly across the six directed scenarios only. Each draw thus contains8 baseline and16 directed seed observations; baseline seeds9-16 are never invented. Both cohort index matrices are saved and remain unchanged with processing chunk size. Every draw recomputes landscape medians, their contrasts and profile ranks. Missing responses remain in their seed blocks. At least3 resampled valid landscape observations (counting multiplicities) are required for a median. Exact frequency-weighted order statistics were verified against literal repeated-index nanmedians; they accelerate computation without changing the statistic.

The configured seed is2026101503, initial batch1000, increments1000, maximum20000, endpoint tolerance.01 and central-bootstrap-median tolerance.005 in Spearman units, with3 consecutive stable comparisons. Percentile limits use linear quantiles at.025 and.975. At least95% of draws must yield a defined coefficient. Each cell freezes at its first qualifying stable batch, and all nonconverged or insufficiently available intervals are withheld. Percentiles use the defined draws, so sparse-H intervals are conditional on bootstrap estimability; this qualification remains relevant even when numerical convergence succeeds. The intervals are pointwise, not simultaneous bands. They describe landscape-seed sampling conditional on the fixed numerical parameter design, not uncertainty about a biological species distribution.

## Checks and limitations

Original medians and saved-row contrasts were source/hash checked; all finite source coefficients agree with independent SciPy calculations to{prep['maximum_scipy_coefficient_error']:.3g}. Numerical tolerance/missingness/ties/cohort-index fixtures passed. Bootstrap: {boot['converged']}/{boot['eligible']} eligible pointwise intervals converged within{boot['total_draws']} draws; {boot['withheld']} were withheld. Independently repeated literal-index calculations reproduced{boot['n_independent_intervals']} complete interval cells, maximum quantile error{boot['maximum_independent_quantile_error']:.3g}. Source H=0 and1 are undefined; at H=.975 Only-Random has2/8 valid landscapes and no coefficient is estimated. H=.925 has5/8 and is conditional on the remaining fragmented subset. No missing response was recoded as extinction.

The largest whole-seed omission change in an original coefficient is{np.nanmax(maxloo):.6f}; {int((table.maximum_seed_omission_change>c['seed_omission_tolerance']).sum())}/{int(finite.sum())} coefficients exceed the predeclared.05 sensitivity tolerance under at least one omission. The largest matched8-to-8/16 coefficient change is{exp8.max():.6f}. Rounding contrasts to1e-8 as a numerical tie-sensitivity check changes a finite coefficient by at most{tie.max():.6f}. These diagnostics do not establish precision in every cell. Eight baseline seeds limit inference even where bootstrap numerical stability is achieved. Full tables retain these checks alongside every coefficient. No simulations, biological calibration or Result-D analysis were started.

## Figure captions

Spearman associations of scenario-minus-Only-Random median equilibrium patch-occupancy contrasts with the parameter shown on the y-axis label, across256 fixed numerical parameter profiles at each H. Panels show Edge-Random (left) and Access-based (right) at rho=.9, .99 and.999 from top to bottom; colour encodes rho and dashed curves indicate Access-based. Lines are original Spearman estimates, shaded intervals are95% pointwise percentile intervals from stratified whole-seed resampling of the shared1-8 and directed-only9-16 cohorts. Missing or nonconverged intervals are withheld; any marked crosses indicate available original coefficients without a reported CI. H decreases from1 to0; undefined original coefficients remain gaps. The parameter design is held fixed, and correlations are marginal under its nonrectangular joint distribution. Interval construction and limited high-H baseline availability are detailed above.

## Reproduction

`final-spearman-parameter-associations --stage prepare`, then `--stage bootstrap`, then `--stage export` through the stable project runner. Bootstrap blocks have immutable receipts and can be resumed with the same sources/configuration. Read and follow the PDF workflow before the first export; after both PDFs are actually rendered and viewed, use `--stage confirm`.
'''
    (out/'11_methods_results_and_captions.md').write_text(note)
    inputs=[CONFIG,Path(__file__),Path(R.__file__),out/'logs/11_preparation.json',out/'logs/11_bootstrap_audit.json',FINAL/'code/base/src/tools/final_plot_style.py']
    outputs=list((out/'figures').glob('*'))+list((out/'tables').glob('*.csv'))+[out/'11_methods_results_and_captions.md']
    B.write_json(marker,dict(status='numerically_checked_visual_review_pending',visual_review_confirmed=False,bootstrap_summary=boot,maximum_seed_omission_change=float(np.nanmax(maxloo)),maximum_tie_sensitivity_change=float(tie.max()),maximum_matched8_to_full_change=float(exp8.max()),export_checks=fonts,input_sha256={provenance_key(p):B.sha(p) for p in inputs},output_sha256={provenance_key(p):B.sha(p) for p in outputs}))


def confirm():
    c,out,data=paths();p=out/'logs/11_spearman_audit.json';a=json.loads(p.read_text());check_hashes(a)
    review=out/'review/11_actual_visual_review.json';assert json.loads(review.read_text())['both_pdfs_rendered_and_viewed']
    a.update(status='numerically_and_visually_checked_preliminary',visual_review_confirmed=True,visual_review_receipt_sha256=B.sha(review));B.write_json(p,a)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','bootstrap','export','confirm'],required=True)
    {'prepare':prepare,'bootstrap':bootstrap,'export':export,'confirm':confirm}[parser.parse_args().stage]()
