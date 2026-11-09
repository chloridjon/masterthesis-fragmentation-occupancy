"""Joint species-parameter occupancy and contrast maps at fixed H levels."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib
import json
import sys
import subprocess

FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent))
M=importlib.import_module('11_integrated_occupancy_contrast')
B,U=M.B,M.U
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,FOCAL_SCENARIOS,MISSING,NEUTRAL

REL='02_metapop/02_03_main_analyses/02_03_02_species_dependence'
CONFIG=FINAL/f'config/{REL}/fixed_H_parameter_surfaces/11_fixed_H_surfaces.json'


def draw(design,ell,rr,fields,h,kind,stem,diagnostic=False):
    apply_final_style()
    absolute=kind=='absolute'
    fig,axes=plt.subplots(4 if absolute else 3,2,figsize=figure_size(235 if absolute else 190),sharex=True,sharey=True)
    fig.subplots_adjust(left=.14,right=.80,top=.96,bottom=.15 if absolute else .18,hspace=.40,wspace=.18)
    cmap=plt.get_cmap('cividis' if absolute else 'RdBu_r').copy();cmap.set_bad(MISSING)
    norm=Normalize(0,1) if absolute else TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    locations=[0,2,3,4,5,6,7] if absolute else list(range(6))
    scenarios=FOCAL_SCENARIOS if absolute else FOCAL_SCENARIOS[1:]
    if absolute:axes.flat[1].set_visible(False)
    for j,(location,scenario) in enumerate(zip(locations,scenarios)):
        ax=axes.flat[location];ax.set_facecolor(MISSING)
        ax.pcolormesh(ell,rr,np.ma.masked_invalid(fields[j]),shading='nearest',cmap=cmap,norm=norm,rasterized=True)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlim(1,512);ax.set_ylim(10**-2.5,10**5.75)
        ax.set_xticks([1,4,16,64,256],['1','4','16','64','256'])
        ax.set_yticks([.01,1,100,10000],[r'$10^{-2}$',r'$10^0$',r'$10^2$',r'$10^4$']);ax.minorticks_off()
        ax.text(0,1.04,f'({chr(97+j)}) '+scenario['label'],transform=ax.transAxes,fontsize=9.5)
        if diagnostic:
            ax.scatter(design.half_distance,design.r,s=6,facecolors='none',edgecolors=NEUTRAL,linewidths=.35)
    cax=fig.add_axes([.86,.21,.025,.69])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[0,.25,.5,.75,1] if absolute else [-1,-.5,0,.5,1],
        label='Median equilibrium patch occupancy' if absolute else 'Median occupancy contrast (scenario '+r'$-$'+' Only-Random)')
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    handles=[Line2D([],[],ls='none',label=rf'$H={h:g}$'+'; 8 Only-Random / 16 other-scenario seeds'),Patch(facecolor=MISSING,label='Outside sampled parameter support')]
    if diagnostic:handles.append(Line2D([],[],ls='none',marker='o',markerfacecolor='none',color=NEUTRAL,ms=3,label='Sampled profile'))
    fig.legend(handles=handles,loc='lower center',frameon=False,ncol=1)
    if diagnostic:
        # Internal sampling-support diagnostics; primary figures have vector exports.
        from src.tools.final_plot_style import compact_shared_labels
        compact_shared_labels(fig)
    else:export_figure(fig,stem)
    fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)


def run():
    c=json.loads(CONFIG.read_text());snap,out,data=(FINAL/c[k] for k in ['source_snapshot','results_directory','data_directory'])
    marker=out/'logs/11_fixed_H_audit.json'
    if marker.exists():assert not json.loads(marker.read_text())['visual_review_confirmed']
    for p in [out/'figures',out/'tables',out/'logs',out/'review',data]:p.mkdir(parents=True,exist_ok=True)
    numeric=json.loads((snap/'11_numerical_audit.json').read_text());visual=json.loads((snap/'11_export_audit.json').read_text())
    assert numeric['n_trajectories']==104 and numeric['n_parameters']==256
    assert visual['visual_review_confirmed'] and visual['numeric_audit_sha256']==B.sha(snap/'11_numerical_audit.json')
    consumed=[snap/'tables'/name for name in ['11_response_arrays.npz','11_response_summaries.csv','11_profile_response_diagnostics.csv']]
    for p in consumed:assert B.sha(p)==numeric['output_sha256'][str(p.relative_to(FINAL))],p
    df=pd.read_csv(consumed[1],float_precision='round_trip');design=pd.read_csv(consumed[2],float_precision='round_trip')
    hs=df.H_target.drop_duplicates().to_numpy();selected=np.array(c['selected_H'])
    for nominal,chosen in zip(c['nominal_H'],selected):
        distance=abs(hs-nominal);assert max(hs[np.isclose(distance,distance.min(),atol=1e-12,rtol=0)])==chosen
    indices=[list(hs).index(h) for h in selected];ids=design.parameter_id.tolist()
    with np.load(consumed[0]) as z:
        values=z['values'][:,:,indices,:];median=z['quantiles'][1][:,indices,:];counts=z['valid_counts'][:,indices];planned=z['planned_seed_mask']
    assert values.shape==(16,7,4,256) and planned.sum()==104
    assert planned[:8].all() and not planned[8:,0].any() and planned[8:,1:].all()
    assert not np.isfinite(values[~planned]).any()
    np.testing.assert_allclose(M.seed_medians(values),median,atol=1e-14,rtol=0)
    assert np.isfinite(median).all() and (counts>=c['minimum_valid_seeds']).all()
    independent_error=0.
    for j,s in enumerate(FOCAL_SCENARIOS):
        saved=df[df.scenario_id==s['scenario_id']].pivot(index='H_target',columns='parameter_id',values='median').reindex(index=selected,columns=ids).to_numpy()
        independent_error=max(independent_error,float(abs(saved-median[j]).max()))
    assert independent_error<1e-14
    contrasts=median[1:]-median[0]
    point_rows=[];descriptive=[];stability=[]
    for t,h in enumerate(selected):
        for j,s in enumerate(FOCAL_SCENARIOS):
            d=median[j,t]-median[0,t]
            descriptive.append(dict(H=h,scenario_id=s['scenario_id'],n_valid=int(counts[j,t]),n_planned=8 if j==0 else 16,
                fraction_profiles_at_floor=float(np.mean(median[j,t]<=.05)),fraction_profiles_at_ceiling=float(np.mean(median[j,t]>=.95)),
                minimum_sampled_median=float(median[j,t].min()),maximum_sampled_median=float(median[j,t].max()),
                minimum_sampled_contrast=float(d.min()),maximum_sampled_contrast=float(d.max())))
            for i,p in design.iterrows():point_rows.append(dict(H=h,scenario_id=s['scenario_id'],parameter_id=p.parameter_id,r=p.r,half_distance=p.half_distance,
                n_valid=int(counts[j,t]),n_baseline_valid=int(counts[0,t]),median_occupancy=median[j,t,i],contrast_vs_only_random=d[i]))
    for seed in range(16):
        other=M.seed_medians(np.delete(values,seed,axis=0));diff=other[1:]-other[0]
        if seed>=8:np.testing.assert_array_equal(other[0],median[0])
        for t,h in enumerate(selected):
            for j,s in enumerate(FOCAL_SCENARIOS[1:]):stability.append(dict(H=h,scenario_id=s['scenario_id'],omitted_seed=seed+1,
                cohort='shared_1_to_8' if seed<8 else 'directed_only_9_to_16',maximum_median_change=float(abs(other[j+1,t]-median[j+1,t]).max()),
                maximum_contrast_change=float(abs(diff[j,t]-contrasts[j,t]).max())))
    pd.DataFrame(point_rows).to_csv(data/'11_fixed_H_sampled_responses.csv',index=False)
    pd.DataFrame(descriptive).to_csv(out/'tables/11_floor_ceiling_and_valid_counts.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'tables/11_cohort_seed_omission.csv',index=False)
    domain=df[['scenario_id','H_target','n_planned','n_valid','n_excluded','n_structurally_absent']].drop_duplicates().copy()
    domain['selected_for_display']=domain.H_target.isin(selected);domain.to_csv(out/'tables/11_H_domain_and_exclusions.csv',index=False)
    points,low,high=B.normalized_coordinates(design,dict(half_distance_min=1,half_distance_max=512,log10_r_min=-2.5,log10_r_max=5.75))
    x,y=np.linspace(0,1,c['grid_n_half_distance']),np.linspace(0,1,c['grid_n_r'])
    xx,yy=np.meshgrid(x,y);targets=np.column_stack([xx.ravel(),yy.ravel()])
    ell,rr=10**(low[0]+x*(high[0]-low[0])),10**(low[1]+y*(high[1]-low[1]))
    ee,rs=np.meshgrid(ell,rr);strips=pd.read_csv(FINAL/c['domain_strips'],float_precision='round_trip')
    inside=U.inner_mask(np.log(ee),np.log(rs),strips)
    assert U.inner_mask(np.log(design.half_distance.to_numpy()),design.log_r.to_numpy(),strips).all()
    absolute_fields=[];contrast_fields=[];stems={'absolute':[],'contrast':[]};node_error=0.;linearity_error=0.
    for t,h in enumerate(selected):
        field,_,tri=B.interpolate_scenarios(points,median[:,t],targets)
        node_error=max(node_error,float(abs(LinearNDInterpolator(tri,median[:,t].T)(points).T-median[:,t]).max()))
        delta,_,_=B.interpolate_scenarios(points,contrasts[:,t],targets)
        field=field.reshape(7,len(y),len(x));delta=delta.reshape(6,len(y),len(x))
        field[:,~inside]=np.nan;delta[:,~inside]=np.nan
        assert np.array_equal(np.isnan(delta),np.isnan(field[1:]-field[0]))
        linearity_error=max(linearity_error,float(np.nanmax(abs(delta-(field[1:]-field[0])))))
        assert np.nanmin(field)>=-1e-14 and np.nanmax(field)<=1+1e-14
        absolute_fields.append(field);contrast_fields.append(delta)
        htag=f'{h:.3f}'.replace('.','p')
        for kind,fields in [('absolute',field),('contrast',delta)]:
            stem=out/f'figures/11_{kind}_H{htag}'
            draw(design,ell,rr,fields,h,kind,stem);stems[kind].append(stem)
            draw(design,ell,rr,fields,h,kind,out/f'review/11_{kind}_H{htag}_sample_locations',diagnostic=True)
    assert node_error<1e-12 and linearity_error<1e-12
    np.savez_compressed(data/'11_fixed_H_parameter_surfaces.npz',H=selected,parameter_ids=np.array(ids),sampled_medians=median,sampled_contrasts=contrasts,
        grid_half_distance=ell,grid_r=rr,absolute_fields=np.array(absolute_fields),contrast_fields=np.array(contrast_fields),adopted_domain_mask=inside,
        sampled_half_distance=design.half_distance.to_numpy(),sampled_r=design.r.to_numpy())
    fontchecks=B.export_checks(stems['absolute']+stems['contrast'])
    for kind in stems:
        subprocess.run(['pdfunite',*[str(s.with_suffix('.pdf')) for s in stems[kind]],str(out/f'11_{kind}_fixed_H_collection.pdf')],check=True,capture_output=True)
    max_omission=max(row['maximum_contrast_change'] for row in stability)
    note=f'''# Joint parameter surfaces at fixed habitat proportions

These descriptive Results-C surfaces show both species parameters simultaneously, without marginal averaging, parameter classes or selection of a single fitted profile. The fixed 256-profile conditional log-area design, q=0, b=1, x=.5 and the seven focal scenarios are unchanged. The data use 8 Only-Random and 16 seeds per directed scenario. No simulations or confidence intervals were added.

## Habitat-level selection

Before constructing the surfaces, the display configuration fixed nominal H=.8, .6, .4 and .2 as a broad, equally spaced overview. Each nominal value was matched to the nearest stored H level, with ties within1e-12 resolved towards the larger H. The resulting levels are .825, .6, .4 and .2. No occupancy maximum, scenario-contrast maximum or bird-fit score entered this selection. The overview is not a search for the strongest effects and does not replace the complete H-dependent results.

## Statistics and display

Each sampled point represents a median equilibrium mean patch occupancy over valid multi-patch landscapes for that scenario and H. Each contrast is the scenario median minus the Only-Random median at the same H and exact parameter combination. These are differences of medians, not medians of paired seed differences. At least three valid landscapes are required; counts and exclusions for every H/scenario are saved. Missing landscapes are never assigned zero occupancy. Endpoint and sparse high-H limitations remain in the source data, although the selected H levels avoid the sparse .925/.975 baseline region.

Absolute occupancy uses a common cividis scale from0 to1. Contrasts use the same diverging scale from-1 to+1 on every page, with red positive, blue negative and neutral zero. Seven absolute panels include Only-Random and the six directed scenarios; six contrast panels compare the latter with Only-Random. H and planned seed counts are labelled within each figure. Small contrasts must be read together with absolute occupancy: all scenarios can occupy a floor or ceiling region.

Both plot axes are logarithmic: dispersal half-distance1-512 cells horizontally and r=10^-2.5 to10^5.75 vertically. Piecewise-linear Delaunay interpolation uses unit-normalised log10 coordinates and is masked to the intersection of the sampled convex hull and the adopted exact stepwise inner domain. No extrapolation fills unsupported locations; these are grey. The same geometry and mask are used for absolute values and contrasts. Separate PNG diagnostics mark all256 sampled locations. Interpolated structure is descriptive and does not add parameter resolution or establish biological species archetypes.

## Verification and scope

Source medians were independently reconstructed from the checked saved response array and compared with saved-row CSV medians (maximum error{independent_error:.3g}). Interpolation recovers source nodes to{node_error:.3g}, and interpolating pointwise contrasts agrees with subtracting interpolated medians to{linearity_error:.3g}. No absolute surface leaves the0-1 range beyond floating-point tolerance. Whole-seed omissions preserve the shared cohort1-8 and directed-only cohort9-16. The largest sampled pointwise contrast change at the selected H levels is{max_omission:.6f}; the full per-H/scenario table is retained. These are descriptive sensitivity checks, not CIs or a claim that all cells are stable.

## Figure captions

**Absolute occupancy.** Median equilibrium patch occupancy across the joint species-parameter design at the H value shown in each figure. Panel(a) is Only-Random; subsequent paired rows compare Edge-Random (left) with Access-based (right) at rho=.9, .99 and .999. Medians use eight Only-Random landscape seeds and sixteen seeds per other scenario, restricted to valid multi-patch landscapes. Colour shows occupancy on the common0-1 scale. Surfaces use piecewise-linear interpolation in normalised logarithmic parameter coordinates; grey indicates locations outside the supported inner parameter domain. Sampling-location diagnostics show all256 input profiles. The four pages show H=.825, .6, .4 and .2, selected as an overview before surface construction. No interval or inferential test is encoded.

**Pointwise contrast.** Differences between each directed scenario's median equilibrium patch occupancy and the Only-Random median at the same H and parameter combination. Paired columns show Edge-Random (left) and Access-based (right), with rho=.9, .99 and .999 from top to bottom. Red indicates higher and blue lower occupancy than Only-Random; all pages share the-1 to+1 occupancy-unit colour scale. The same H levels, landscape counts, parameter design, interpolation and unsupported-domain mask as in the absolute-occupancy collection apply. These contrasts should be interpreted alongside absolute occupancy to identify floor and ceiling regions. No confidence interval or significance claim is shown.

## Reproduction

`.venv/bin/python -m src.tools.codex_run final-fixed-H-parameter-surfaces`

After all eight PDF pages are rendered and actually inspected, confirm with `--confirm-visual-review`. Primary collection pages run from higher to lower H. Individual PDF/SVG/PNG exports are included for manuscript assembly.
'''
    (out/'11_methods_and_captions.md').write_text(note)
    index=['# Fixed-H parameter surfaces','', '[Absolute occupancy: four-page collection](11_absolute_fixed_H_collection.pdf)', '[Pointwise contrasts: four-page collection](11_contrast_fixed_H_collection.pdf)','', '| H | Absolute occupancy | Contrast |','|---|---|---|']
    for t,h in enumerate(selected):index.append(f'| {h:g} | [PDF]({stems["absolute"][t].relative_to(out)}.pdf) | [PDF]({stems["contrast"][t].relative_to(out)}.pdf) |')
    index+=['','[Methods and captions](11_methods_and_captions.md)','[Valid counts and floor/ceiling diagnostics](tables/11_floor_ceiling_and_valid_counts.csv)']
    (out/'11_plot_index.md').write_text('\n'.join(index)+'\n')
    consumed += [CONFIG,Path(__file__),Path(M.__file__),Path(B.__file__),Path(U.__file__),FINAL/c['domain_strips'],snap/'11_numerical_audit.json',snap/'11_export_audit.json',FINAL/'code/base/src/tools/final_plot_style.py']
    outputs=[p for folder in [data,out/'figures',out/'tables'] for p in folder.rglob('*') if p.is_file()]+list(out.glob('*.pdf'))+[out/'11_methods_and_captions.md',out/'11_plot_index.md']+list((out/'review').glob('*sample_locations.png'))
    B.write_json(marker,dict(status='numerically_and_export_checked_visual_review_pending',visual_review_confirmed=False,created_utc=datetime.now(timezone.utc).isoformat(),
        selected_H=selected.tolist(),valid_landscape_counts=counts.tolist(),n_profiles=256,bootstrap_intervals_calculated=False,
        maximum_independent_median_error=independent_error,maximum_interpolation_node_error=node_error,maximum_interpolation_linearity_error=linearity_error,
        maximum_seed_omission_contrast_change=max_omission,export_checks=fontchecks,
        input_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in consumed},output_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in outputs}))
    print(json.dumps(dict(selected_H=selected.tolist(),valid_counts=counts.tolist(),node_error=node_error,linearity_error=linearity_error,maximum_seed_omission_contrast_change=max_omission)))


def confirm():
    c=json.loads(CONFIG.read_text());out=FINAL/c['results_directory'];p=out/'logs/11_fixed_H_audit.json';a=json.loads(p.read_text())
    for name,digest in {**a['input_sha256'],**a['output_sha256']}.items():assert B.sha(FINAL/name)==digest,name
    review=out/'review/11_actual_visual_review.json';r=json.loads(review.read_text())
    assert r['all_eight_pdf_pages_rendered_and_viewed'] and r['all_eight_sampling_diagnostics_viewed']
    a.update(status='numerically_and_visually_checked_descriptive',visual_review_confirmed=True,visual_review_utc=datetime.now(timezone.utc).isoformat(),visual_review_receipt_sha256=B.sha(review));B.write_json(p,a)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--confirm-visual-review',action='store_true')
    confirm() if parser.parse_args().confirm_visual_review else run()
