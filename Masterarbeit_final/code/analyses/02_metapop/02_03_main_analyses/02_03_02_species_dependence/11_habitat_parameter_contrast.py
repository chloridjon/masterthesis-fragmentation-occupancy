"""Conditional parameter-bin summaries of pointwise occupancy contrasts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib
import json
import math
import os
import sys

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-habitat-parameter-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(FINAL/'code/base'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
M = importlib.import_module('11_integrated_occupancy_contrast')
B = M.B
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS, MISSING
REL = '02_metapop/02_03_main_analyses/02_03_02_species_dependence'
CONFIG = FINAL/f'config/{REL}/habitat_parameter_contrast/11_habitat_parameter_contrast.json'


def assign_bins(coordinate, edges):
    """Left-closed bins, except for an inclusive final upper edge."""
    coordinate, edges = np.asarray(coordinate), np.asarray(edges)
    if not np.isfinite(coordinate).all() or not np.all(np.diff(edges) > 0):
        raise ValueError('Finite coordinates and increasing edges required')
    if np.any(coordinate < edges[0]) or np.any(coordinate > edges[-1]):
        raise ValueError('Coordinate outside fixed bin range')
    return np.minimum(np.searchsorted(edges,coordinate,side='right')-1,len(edges)-2)


def conditional_means(contrasts, assignments, weights, n_bins):
    """Weighted means of profile-level contrasts, never pooled seed medians."""
    contrasts, assignments, weights = np.asarray(contrasts), np.asarray(assignments), np.asarray(weights)
    if contrasts.shape[-1] != len(weights) or assignments.shape != weights.shape:
        raise ValueError('Profile identities must align')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Positive finite design weights required')
    result = np.full((*contrasts.shape[:-1], n_bins), np.nan)
    for k in range(n_bins):
        take = assignments == k
        if take.any():
            # A missing member invalidates the cell; no H-dependent profile changes.
            result[...,k] = np.sum(contrasts[...,take]*weights[take],axis=-1)/weights[take].sum()
    return result


def draw(h, edges, fields, parameter, stem):
    apply_final_style()
    fig, axes = plt.subplots(3,2,figsize=figure_size(190),sharex=True,sharey=True)
    fig.subplots_adjust(left=.14,right=.80,top=.96,bottom=.18,hspace=.40,wspace=.18)
    # Midpoint H cells are clipped to the common observed domain, with no endpoint extrapolation.
    he = np.r_[h[0],(h[:-1]+h[1:])/2,h[-1]]
    cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad(MISSING)
    norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    for j, ax in enumerate(axes.flat):
        ax.set_facecolor(MISSING)
        ax.pcolormesh(he,10**edges,np.ma.masked_invalid(fields[j].T),shading='flat',cmap=cmap,norm=norm,rasterized=True)
        ax.set_xlim(1,0); ax.set_xticks([1,.8,.6,.4,.2,0],['1','0.8','0.6','0.4','0.2','0'])
        ax.set_yscale('log'); ax.set_ylim(10**edges[0],10**edges[-1]); ax.minorticks_off()
        if parameter == 'r':
            ax.set_yticks([.01,1,100,10000],[r'$10^{-2}$',r'$10^0$',r'$10^2$',r'$10^4$'])
        else:
            ax.set_yticks([1,4,16,64,256],['1','4','16','64','256'])
        rho=[.9,.99,.999][j//2]; family='Edge-Random' if j%2==0 else 'Access-based'
        ax.text(0,1.04,f'({chr(97+j)}) {family}, '+rf'$\rho={rho}$',transform=ax.transAxes,fontsize=9.5)
    cax=fig.add_axes([.86,.23,.025,.66])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-1,-.5,0,.5,1],
        label='Mean profile contrast (scenario '+r'$-$'+' Only-Random)')
    fig.supxlabel(r'Habitat proportion $H$')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)' if parameter=='r' else r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.legend(handles=[Patch(facecolor=MISSING,label='Outside common evaluable habitat range')],loc='lower center',frameon=False)
    export_figure(fig,stem)
    fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def run():
    c=json.loads(CONFIG.read_text()); out,data,snap=(FINAL/c[k] for k in ['results_directory','data_directory','source_snapshot'])
    marker=out/'logs/11_habitat_parameter_audit.json'
    if marker.exists():
        assert json.loads(marker.read_text())['status']=='numerically_checked_visual_review_pending'
    for p in [out/'logs',out/'tables',out/'figures',out/'review',data]:p.mkdir(parents=True,exist_ok=True)
    numeric=json.loads((snap/'11_numerical_audit.json').read_text())
    visual=json.loads((snap/'11_export_audit.json').read_text())
    assert numeric['n_trajectories']==104 and numeric['n_parameters']==256
    assert visual['visual_review_confirmed'] and visual['numeric_audit_sha256']==B.sha(snap/'11_numerical_audit.json')
    consumed=[snap/'tables'/name for name in ['11_response_arrays.npz','11_response_summaries.csv','11_profile_response_diagnostics.csv']]
    for p in consumed:assert B.sha(p)==numeric['output_sha256'][str(p.relative_to(FINAL))],p
    df=pd.read_csv(consumed[1],float_precision='round_trip')
    design=pd.read_csv(consumed[2],float_precision='round_trip')
    h=df.H_target.drop_duplicates().to_numpy()
    with np.load(consumed[0]) as z:
        values,med,counts,planned=z['values'],z['quantiles'][1],z['valid_counts'],z['planned_seed_mask']
    assert values.shape==(16,7,88,256) and planned.sum()==104
    assert planned[:8].all() and not planned[8:,0].any() and planned[8:,1:].all()
    assert not np.isfinite(values[~planned]).any()
    np.testing.assert_allclose(M.seed_medians(values),med,atol=1e-14,rtol=0)
    ids=design.parameter_id.to_list()
    # All saved CSV contrasts must align with the same scenario/H/profile array order.
    delta=med[1:]-med[0]
    for j,scenario in enumerate(FOCAL_SCENARIOS[1:]):
        saved=df[df.scenario_id==scenario['scenario_id']].pivot(index='H_target',columns='parameter_id',values='contrast_vs_only_random').reindex(index=h,columns=ids).to_numpy()
        np.testing.assert_allclose(saved,delta[j],atol=1e-14,rtol=0,equal_nan=True)
    included=np.isfinite(med).all(axis=(0,2)); hi=h[included]
    assert len(hi)==85 and hi[0]==.005 and hi[-1]==.925 and np.all(np.diff(np.flatnonzero(included))==1)
    contrasts=delta[:,included,:]
    weights=design.parameter_design_weight.to_numpy()
    np.testing.assert_array_equal(weights,np.full(256,1/256))
    all_rows,counts_rows,membership,summary,omission_rows=[],[],[],{},[]
    arrays={'H':hi,'parameter_ids':np.array(ids),'profile_contrasts':contrasts}
    stems=[]; independent_error=0.; mass_error=0.
    for parameter,bounds in [('r',(-2.5,5.75)),('half_distance',(0,math.log10(512)))]:
        edges=np.linspace(*bounds,c['n_log_bins']+1)
        coord=np.log10(design[parameter].to_numpy()); assignments=assign_bins(coord,edges)
        fields=conditional_means(contrasts,assignments,weights,c['n_log_bins'])
        count=np.bincount(assignments,minlength=c['n_log_bins'])
        assert count.sum()==256 and count.min()>=4
        independent=np.empty_like(fields)
        for k in range(c['n_log_bins']):
            take=np.flatnonzero(assignments==k)
            for j in range(6):
                for t in range(len(hi)):
                    independent[j,t,k]=math.fsum(float(contrasts[j,t,i]) for i in take)/len(take)
        independent_error=max(independent_error,float(abs(independent-fields).max()))
        mass_error=max(mass_error,float(abs((fields*count).sum(axis=-1)/256-contrasts.mean(axis=-1)).max()))
        assert independent_error<1e-14 and mass_error<1e-14
        eight=M.seed_medians(values[:8])[:,included,:]
        eight_fields=conditional_means(eight[1:]-eight[0],assignments,weights,c['n_log_bins'])
        max_omit=0.
        for seed in range(16):
            other=M.seed_medians(np.delete(values,seed,axis=0))[:,included,:]
            if seed>=8:np.testing.assert_array_equal(other[0],med[0,included,:])
            shifted=conditional_means(other[1:]-other[0],assignments,weights,c['n_log_bins'])
            for j,s in enumerate(FOCAL_SCENARIOS[1:]):
                change=float(abs(shifted[j]-fields[j]).max());max_omit=max(max_omit,change)
                omission_rows.append(dict(parameter=parameter,omitted_seed=seed+1,cohort='shared_1_to_8' if seed<8 else 'directed_only_9_to_16',scenario_id=s['scenario_id'],max_absolute_change=change))
        summary[parameter]=dict(bin_counts=count.tolist(),minimum_profile_count=int(count.min()),maximum_profile_count=int(count.max()),
            minimum_contrast=float(fields.min()),maximum_contrast=float(fields.max()),maximum_seed_omission_change=max_omit,
            maximum_matched8_to_full_change=float(abs(eight_fields-fields).max()))
        arrays.update({parameter+'_log_bin_edges':edges,parameter+'_profile_assignments':assignments,parameter+'_mean_contrast':fields,parameter+'_matched8_mean_contrast':eight_fields})
        for i,p in design.iterrows():membership.append(dict(parameter=parameter,parameter_id=p.parameter_id,bin_index=int(assignments[i]),design_weight=weights[i],r=p.r,half_distance=p.half_distance))
        for k in range(c['n_log_bins']):
            counts_rows.append(dict(parameter=parameter,bin_index=k,lower_bound=10**edges[k],upper_bound=10**edges[k+1],n_profiles=int(count[k])))
            for j,s in enumerate(FOCAL_SCENARIOS[1:]):
                for t,hh in enumerate(hi):
                    all_rows.append(dict(parameter=parameter,bin_index=k,lower_bound=10**edges[k],upper_bound=10**edges[k+1],n_profiles=int(count[k]),scenario_id=s['scenario_id'],H=float(hh),mean_profile_contrast=fields[j,t,k],n_baseline_valid=int(counts[0,included][t]),n_scenario_valid=int(counts[j+1,included][t]),matched8_mean_profile_contrast=eight_fields[j,t,k]))
        stem=out/f'figures/11_contrast_H_by_{parameter}';draw(hi,edges,fields,parameter,stem);stems.append(stem)
    pd.DataFrame(all_rows).to_csv(data/'11_habitat_parameter_contrasts.csv',index=False)
    pd.DataFrame(membership).to_csv(data/'11_profile_bin_membership.csv',index=False)
    pd.DataFrame(counts_rows).to_csv(out/'tables/11_profile_counts.csv',index=False)
    pd.DataFrame(omission_rows).to_csv(out/'tables/11_cohort_seed_omission.csv',index=False)
    domain=df[['scenario_id','H_target','n_planned','n_valid','n_excluded','n_structurally_absent']].drop_duplicates().copy()
    domain['displayed']=domain.H_target.isin(hi);domain.to_csv(out/'tables/11_H_domain_and_exclusions.csv',index=False)
    np.savez_compressed(data/'11_habitat_parameter_arrays.npz',**arrays)
    checks=B.export_checks(stems)
    note=f'''# Habitat proportion and species-parameter contrast maps

These author-requested descriptive maps use the same audited256-profile conditional log-area design and8 Only-Random/16 directed-scenario seeds. No new simulations, bootstrap intervals or biological calibration were performed. First, for each profile and habitat proportion, we calculated the difference of landscape medians, directed scenario minus Only-Random. We then calculated the equal-design-weight arithmetic mean of these contrasts across profiles in each of16 fixed equal-width logarithmic parameter classes. This is a mean of profile-specific median differences, not a pooled landscape median or an effect at a fixed value of the omitted parameter.

For the r map, bin edges span log10(r)=-2.5 to5.75. For the dispersal map, bin edges span half-distance1 to512 cells. Every profile has weight1/256 in the original design; weights are renormalised within each class. Every one of the256 profiles belongs to one bin in each figure, with no curve-shape or contrast-based selection. Bin counts range{summary['r']['minimum_profile_count']}-{summary['r']['maximum_profile_count']} for r and{summary['half_distance']['minimum_profile_count']}-{summary['half_distance']['maximum_profile_count']} for half-distance; counts and memberships are saved. Four profiles in the sparsest r class provide limited parameter coverage. Shared dispersal groups are numerical design locations, not independent landscape replicates.

The other parameter varies over the portion of the adopted nonrectangular domain represented by the profiles in each class. Its conditional distribution therefore changes with the displayed parameter. Patterns describe the chosen joint design and cannot isolate the causal effect of changing r or dispersal while holding the other fixed. Averaging may also hide opposing profile-level contrasts.

All panels use the same85 evaluable H levels, .005-.925, requiring at least3 valid landscapes per scenario/H/profile before aggregation. The baseline has5/8 valid landscapes at H=.925; H=.975 has only2/8 and is withheld. H=0 and H=1 have undefined responses. Exclusions are not zero occupancy. Grey regions mark H outside the common domain. H decreases from1 to0. Display cells are centred on sampled H, bounded by adjacent midpoints and clipped to .005-.925; H spacing is not treated as uniform. Parameter classes are displayed at their exact logarithmic edges. No interpolation or smoothing is applied.

Both figures use a common signed colour scale -1 to+1 in occupancy units, with red positive, blue negative and neutral zero. The statistic remains pointwise in H; no integration is performed here. These colours encode no confidence intervals or significance test.

Independent scalar means reconstructed every plotted cell with maximum error{independent_error:.3g}; profile-count-weighted class means recover the overall mean with error{mass_error:.3g}. The CSV contrasts were separately checked against all source median-array differences, and landscape medians were recomputed from the audited saved response array. Cohort-aware whole-seed omissions preserve shared seeds1-8 and directed-only seeds9-16. Maximum omission changes are{summary['r']['maximum_seed_omission_change']:.6f} for the r map and{summary['half_distance']['maximum_seed_omission_change']:.6f} for the dispersal map. Matched8-to-8/16 maximum changes are{summary['r']['maximum_matched8_to_full_change']:.6f} and{summary['half_distance']['maximum_matched8_to_full_change']:.6f}, respectively. These diagnostics do not establish universal stability or inferential coverage.

## Figure captions

**Extinction-to-colonization ratio.** Pointwise occupancy contrasts against Only-Random across habitat proportion and16 logarithmic classes of the ratio r. Each cell is the arithmetic mean across the sampled parameter profiles in that class of the difference between scenario and Only-Random median equilibrium patch occupancy. Profile medians use16 seeds for directed scenarios and8 for Only-Random, restricted to valid multi-patch landscapes. The other parameter, dispersal half-distance, varies among the sampled profiles within each class. Panels compare Edge-Random (left) and Access-based (right) at rho=.9, .99 and .999 (top to bottom). Red indicates higher and blue lower occupancy than Only-Random; grey marks the unevaluable H domain. The common colour scale is -1 to+1 occupancy units. The85 included H levels span .005-.925, shown in the direction of habitat loss; no smoothing or confidence interval is encoded. Counts range4-31 profiles per parameter class. Patterns are conditional on the adopted joint parameter distribution.

**Dispersal half-distance.** As above, with16 logarithmic classes of dispersal half-distance1-512 cells on the y-axis. The extinction-to-colonization ratio varies among the profiles represented within each class. Classes contain12-24 profiles; no value of r is held fixed. All remaining definitions, scenario panels and colour limits are identical.

## Reproduction

`.venv/bin/python -m src.tools.codex_run final-habitat-parameter-contrast`

After both PDFs are rendered and actually viewed, confirm with `--confirm-visual-review`.
'''
    (out/'11_methods_and_captions.md').write_text(note)
    consumed += [CONFIG,Path(__file__),Path(M.__file__),Path(B.__file__),snap/'11_numerical_audit.json',snap/'11_export_audit.json',FINAL/'code/base/src/tools/final_plot_style.py']
    outputs=[p for folder in [data,out/'tables',out/'figures'] for p in folder.rglob('*') if p.is_file()]+[out/'11_methods_and_captions.md']
    a=dict(status='numerically_checked_visual_review_pending',visual_review_confirmed=False,created_utc=datetime.now(timezone.utc).isoformat(),
        n_profiles=256,n_H=len(hi),n_bins=c['n_log_bins'],H_min=float(hi[0]),H_max=float(hi[-1]),color_limits=[-1,1],
        maximum_independent_mean_error=independent_error,maximum_weighted_recovery_error=mass_error,parameter_summaries=summary,
        bootstrap_intervals_calculated=False,export_checks=checks,input_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in consumed},output_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in outputs})
    B.write_json(marker,a);print(json.dumps(summary))


def confirm():
    c=json.loads(CONFIG.read_text());out=FINAL/c['results_directory'];p=out/'logs/11_habitat_parameter_audit.json'
    a=json.loads(p.read_text())
    for name,digest in {**a['input_sha256'],**a['output_sha256']}.items():assert B.sha(FINAL/name)==digest,name
    review=out/'review/11_actual_visual_review.json';assert json.loads(review.read_text())['both_pdfs_rendered_and_viewed']
    a.update(status='numerically_and_visually_checked_descriptive',visual_review_confirmed=True,visual_review_receipt_sha256=B.sha(review),visual_review_utc=datetime.now(timezone.utc).isoformat());B.write_json(p,a)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--confirm-visual-review',action='store_true')
    confirm() if parser.parse_args().confirm_visual_review else run()
