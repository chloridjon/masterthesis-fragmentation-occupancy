"""Repeat the bird-response comparison on the checked conditional 256-point design."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib,json,math,os,sys,warnings
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-bird-256-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(FINAL/'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
BASE=importlib.import_module('11_bird_parameter_surfaces')
M=importlib.import_module('11_confirmed_domain_compute')
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,FOCAL_SCENARIOS
CONFIG=FINAL/'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_parameter_surfaces/conditional_256_5/11_bird_update.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def inner_mask(logell,logr,strips):
    mask=np.zeros(np.shape(logell),bool)
    for row in strips.itertuples(index=False):mask|=(logell>=row.log_ell_low-1e-12)&(logell<=row.log_ell_high+1e-12)&(logr>=row.log_r_low-1e-12)&(logr<=row.log_r_high+1e-12)
    return mask


def overlays(coeff,birds,common,predictions,scores,bands,out):
    apply_final_style();fig,axes=plt.subplots(5,3,figsize=figure_size(245),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.98,top=.97,bottom=.14,wspace=.28,hspace=.55)
    for b,ax in enumerate(axes.flat):
        species=coeff.iloc[b];sid,p=np.unravel_index(scores[b].argmin(),scores[b].shape);style=FOCAL_SCENARIOS[sid]
        z=birds[birds.species_id.eq(species.species_id)].set_index('H').reindex(common)
        band=bands[bands.species_id.eq(species.species_id)].set_index('H').reindex(common)
        ax.fill_between(common,band.psi_ci_lower_digitized,band.psi_ci_upper_digitized,color='#777777',alpha=.2,lw=0)
        ax.plot(common,z.psi_reconstructed,color='#222222',lw=1.)
        ax.plot(common,predictions[sid,p],color=style['color'],ls=style['linestyle'],lw=1.1)
        ax.set_xlim(1,0);ax.set_ylim(0,1);ax.set_xticks([1,.5,0]);ax.set_yticks([0,.5,1]);ax.spines[['top','right']].set_visible(False)
        name=species.species_as_published.split();short=name[0][0]+'. '+' '.join(name[1:])
        ax.text(0,1.05,f'({chr(97+b)}) {short}',transform=ax.transAxes,fontstyle='italic',fontsize=9.5)
        ax.text(.03,.04,f'RMSE {scores[b,sid,p]:.3f}',transform=ax.transAxes,fontsize=8.5,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5))
    fig.supxlabel(r'Habitat proportion $H$');fig.supylabel('Occupancy (0–1)')
    fig.legend(handles=[Line2D([],[],color='#222222',label='Reconstructed bird curve (Table 1)'),Line2D([],[],color='#377A9F',label='Best sampled model median; scenario style'),Patch(facecolor='#777777',alpha=.2,label='Digitized bird band (assumed 95% CI)')],loc='lower center',ncol=1,frameon=False)
    stem=out/'figures/11_bird_best_curve_overlays';export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    return stem


def run():
    c=json.loads(CONFIG.read_text());ctx=M.inputs();mc,base,lc,design,bench,geo,gfp,source,_,fp=ctx
    snap=FINAL/c['source_snapshot'];audit=json.loads((snap/'11_numerical_audit.json').read_text());assert audit['visual_review_confirmed'] and audit['n_seed_blocks']==5
    hashes={}
    for p,h in {**audit['input_sha256'],**audit['output_sha256']}.items():assert sha(FINAL/p)==h,p;hashes[p]=h
    bird=FINAL/c['bird_data'];manifest=json.loads((bird/'manifest.json').read_text());digitized=json.loads((bird/'figure2_digitization_manifest.json').read_text())
    for p,h in {**manifest['output_sha256'],**digitized['output_sha256']}.items():assert sha(FINAL.parent/p)==h,p
    birds=pd.read_csv(bird/'11_bird_curves_main_H_grid.csv',float_precision='round_trip');coeff=pd.read_csv(bird/'11_bird_curve_coefficients.csv',float_precision='round_trip');bands=pd.read_csv(bird/'11_bird_curves_main_H_grid_with_figure2_bands.csv',float_precision='round_trip')
    with np.load(snap/'tables/11_response_arrays.npz') as z:values=z['values'].copy();quantiles=z['quantiles'].copy();counts=z['valid_counts'].copy()
    assert values.shape==(5,7,88,256) and len(coeff)==15
    hs=np.array(base['h_targets']);common=[h for h in hs if h in set(birds.H) and np.isfinite(quantiles[1,:,list(hs).index(h),:]).all()]
    indices=[list(hs).index(h) for h in common];assert len(common)>20
    observed=birds.pivot(index='species_id',columns='H',values='psi_reconstructed').reindex(index=coeff.species_id,columns=common).to_numpy()
    alternative=bands.pivot(index='species_id',columns='H',values='psi_figure2_curve').reindex(index=coeff.species_id,columns=common).to_numpy()
    predictions=quantiles[1][:,indices,:].transpose(0,2,1);scores=BASE.curve_rmse(predictions[None],observed[:,None,None]);alt=BASE.curve_rmse(predictions[None],alternative[:,None,None])
    error=0.
    for b in range(15):
        for sid in [0,3,6]:
            for p in [0,130,255]:error=max(error,abs(scores[b,sid,p]-math.sqrt(math.fsum((float(x)-float(y))**2 for x,y in zip(predictions[sid,p],observed[b]))/len(common))))
    assert error<1e-14
    data=FINAL/c['data_directory'];out=FINAL/c['results_directory']
    for p in [data,out/'tables',out/'figures',out/'logs']:p.mkdir(parents=True,exist_ok=True)
    points,low,high=BASE.normalized_coordinates(design,dict(half_distance_min=1,half_distance_max=512,log10_r_min=-2.5,log10_r_max=5.75))
    x=np.linspace(0,1,c['grid_n_half_distance']);y=np.linspace(0,1,c['grid_n_r']);xx,yy=np.meshgrid(x,y);targets=np.column_stack([xx.ravel(),yy.ravel()])
    ell=10**(low[0]+x*(high[0]-low[0]));rr=10**(low[1]+y*(high[1]-low[1]));ee,rs=np.meshgrid(ell,rr)
    strips=pd.read_csv(FINAL/c['domain_strips'],float_precision='round_trip');inside=inner_mask(np.log(ee),np.log(rs),strips)
    assert inner_mask(np.log(design.half_distance.to_numpy()),design.log_r.to_numpy(),strips).all()
    surfaces=[];interpolation_error=0.;linearity_error=0.
    for b in range(15):
        fields,minimum,tri=BASE.interpolate_scenarios(points,scores[b],targets)
        recovered=LinearNDInterpolator(tri,scores[b].T)(points).T;interpolation_error=max(interpolation_error,float(abs(recovered-scores[b]).max()))
        fields=fields.reshape(7,len(y),len(x));fields[:,~inside]=np.nan
        summed=np.sum(fields,axis=0);check=LinearNDInterpolator(tri,scores[b].sum(axis=0))(targets).reshape(len(y),len(x));check[~inside]=np.nan
        assert np.array_equal(np.isnan(summed),np.isnan(check));linearity_error=max(linearity_error,float(np.nanmax(abs(summed-check))))
        surfaces.append(fields)
    assert interpolation_error<1e-12 and linearity_error<1e-12
    surfaces=np.array(surfaces);minimum=surfaces.min(axis=1);summed=surfaces.sum(axis=1)
    rows=[];aggregate=[];best=[]
    for b,species in enumerate(coeff.species_id):
        minsite=scores[b].min(axis=0);sums=scores[b].sum(axis=0);j,p=np.unravel_index(scores[b].argmin(),scores[b].shape);sumidx=int(sums.argmin())
        for k,item in design.iterrows():
            aggregate.append(dict(species_id=species,parameter_id=item.parameter_id,r=item.r,half_distance=item.half_distance,minimum_scenario_rmse=float(minsite[k]),excess_minimum_rmse=float(minsite[k]-minsite.min()),sum_scenario_rmse=float(sums[k]),mean_scenario_rmse=float(sums[k]/7),excess_summed_rmse=float(sums[k]-sums.min())))
            for s,scenario in enumerate(FOCAL_SCENARIOS):rows.append(dict(species_id=species,scenario_id=scenario['scenario_id'],parameter_id=item.parameter_id,r=item.r,half_distance=item.half_distance,rmse=float(scores[b,s,k]),rmse_figure2_central_sensitivity=float(alt[b,s,k]),n_H=len(common)))
        best.append(dict(species_id=species,species_as_published=coeff.iloc[b].species_as_published,trophic_guild=coeff.iloc[b].trophic_guild,minimum_rmse=float(scores[b,j,p]),best_scenario=FOCAL_SCENARIOS[j]['scenario_id'],best_parameter_id=design.parameter_id.iloc[p],best_r=design.r.iloc[p],best_half_distance=design.half_distance.iloc[p],minimum_summed_rmse=float(sums[sumidx]),summed_best_parameter_id=design.parameter_id.iloc[sumidx],maximum_digitized_central_rmse_change=float(abs(scores[b]-alt[b]).max())))
    stability=[]
    for omit in range(5):
        v=np.delete(values[:,:,indices,:],omit,axis=0);n=np.isfinite(v).sum(axis=0)
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(v,axis=0)
        med[n<3]=np.nan;assert np.isfinite(med).all(),'Whole-curve omission became undefined; report missingness explicitly'
        other=BASE.curve_rmse(med.transpose(0,2,1)[None],observed[:,None,None])
        for b,species in enumerate(coeff.species_id):stability.append(dict(species_id=species,omitted_seed=omit+1,maximum_scenario_rmse_change=float(abs(other[b]-scores[b]).max()),maximum_minimum_scenario_rmse_change=float(abs(other[b].min(axis=0)-scores[b].min(axis=0)).max()),maximum_sum_rmse_change=float(abs(other[b].sum(axis=0)-scores[b].sum(axis=0)).max())))
    pd.DataFrame(rows).to_csv(data/'11_bird_scenario_parameter_rmse.csv',index=False);pd.DataFrame(aggregate).to_csv(data/'11_bird_aggregated_rmse.csv',index=False);pd.DataFrame(best).to_csv(out/'tables/11_best_sampled_matches.csv',index=False);pd.DataFrame(stability).to_csv(out/'tables/11_leave_one_seed_out_rmse.csv',index=False)
    domain=pd.DataFrame([dict(scenario_id=s['scenario_id'],H=float(h),n_valid=int(counts[j,i]),n_excluded=5-int(counts[j,i]),used_in_rmse=h in common) for j,s in enumerate(FOCAL_SCENARIOS) for i,h in enumerate(hs)]);domain.to_csv(out/'tables/11_domain_and_exclusions.csv',index=False)
    np.savez_compressed(data/'11_bird_rmse_display_surfaces.npz',species_ids=coeff.species_id.to_numpy(str),grid_half_distance=ell,grid_r=rr,scenario_rmse=surfaces,minimum_scenario_rmse=minimum,summed_scenario_rmse=summed,adopted_domain_mask=inside,H_comparison=np.array(common),sampled_scores=scores,figure2_scores=alt)
    # Preserve the previous sample; compare it on the SAME five seeds and H levels.
    old_config=json.loads((FINAL/c['old_lhs_configuration']).read_text());old_design=pd.read_csv(FINAL/old_config['lhs_design']);old_data=FINAL/c['old_lhs_data'];old_values=np.full((5,7,88,128),np.nan)
    for rep in range(1,6):
        for j,s in enumerate(FOCAL_SCENARIOS):
            path=old_data/'trajectories'/f'{s["scenario_id"]}__rep{rep:03d}.csv';t=pd.read_csv(path,usecols=['H_target','parameter_id','occupancy'],float_precision='round_trip');hashes[str(path.relative_to(FINAL))]=sha(path)
            old_values[rep-1,j]=t.pivot(index='H_target',columns='parameter_id',values='occupancy').reindex(index=hs,columns=old_design.parameter_id).to_numpy()
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);old_median=np.nanmedian(old_values[:,:,indices,:],axis=0)
    assert (np.isfinite(old_values[:,:,indices,0]).sum(axis=0)>=3).all()
    old_scores=BASE.curve_rmse(old_median.transpose(0,2,1)[None],observed[:,None,None])
    old_twenty=pd.read_csv(FINAL/c['old_score_table'],float_precision='round_trip');comparison=pd.DataFrame(best)
    comparison['old_128_first5_best_rmse']=old_scores.min(axis=(1,2));comparison['old_128_first5_best_summed_rmse']=old_scores.sum(axis=1).min(axis=1)
    comparison['old_128_twenty_seed_best_rmse']=old_twenty.groupby('species_id').rmse.min().reindex(coeff.species_id).to_numpy();comparison.to_csv(out/'tables/11_old_new_comparison.csv',index=False)
    stems=[];plot_c={**c,'layout':dict(width_mm=160,height_mm=250),'rmse_colormap':'cividis_r'}
    names=[('11_bird_minimum_rmse',minimum,'Minimum curve RMSE',1,None,None),('11_bird_excess_minimum_rmse',minimum-scores.min(axis=(1,2))[:,None,None],'Excess RMSE above sampled best',.5,None,None),('11_bird_minimum_sample_locations',minimum,'Minimum curve RMSE',1,design,None)]
    names +=[(f'11_bird_{s["scenario_id"]}',surfaces[:,j],'Curve RMSE',1,None,s) for j,s in enumerate(FOCAL_SCENARIOS)]
    names +=[('11_bird_summed_rmse',summed,'Sum of seven curve RMSEs',7,None,None),('11_bird_excess_summed_rmse',summed-scores.sum(axis=1).min(axis=1)[:,None,None],'Excess sum above sampled best',7,None,None),('11_bird_sum_sample_locations',summed,'Sum of seven curve RMSEs',7,design,None)]
    for name,field,label,vmax,sites,scenario in names:
        stem=out/'figures'/name;BASE.make_figure(plot_c,coeff,ell,rr,field,stem,label,vmax,sites,scenario);stems.append(stem);print(f'Exported {name}',flush=True)
    stems.append(overlays(coeff,birds,common,predictions,scores,bands,out));exports=BASE.export_checks(stems)
    note=f'''# Bird comparison on the conditional 256-profile sample

The 15 species were compared with 256 parameter profiles, seven fixed scenarios
and five matched seeds. RMSE uses {len(common)} common, equally weighted H levels
from {min(common):g} to {max(common):g}. Curves and scores are recomputed from the
verified new dataset. Table 1 central curves are the primary empirical target;
digitized Figure 2 central curves are a separate sensitivity check. Source data
were preserved. No empirical uncertainty band or landscape percentile is used
as an RMSE weight. Missing landscapes are never recoded as extinction.

The minimum-scenario overview selects one fixed scenario for the whole curve,
not a different scenario at each H. The sum overview literally adds the seven
scenario-specific whole-curve RMSEs with equal weights; dividing by seven gives
the corresponding mean. Low error is bright yellow (reversed cividis); all
species share fixed scales, 0–1 for absolute RMSE, 0–.5 for excess minimum,
and 0–7 for sum/excess sum. Continuous similarly matching regions are shown;
no isolated optimum, binary good-fit boundary or formal confidence region is
plotted. Best sampled combinations are listed as descriptive matches only.

Piecewise-linear Delaunay interpolation is performed separately for each
scenario in normalized logarithmic coordinates. Display support is the
intersection of the sampled convex hull with the exact conservative adopted
inner region. The latter mask prevents bridging excluded parts of the
nonrectangular domain. Grey is unavailable support, not zero error. White
circles in diagnostic versions mark the actual 256 model profiles. Interpolated
values add no simulations; linear interpolation artifacts can remain.

The 15-panel curve overlay shows the reconstructed Table 1 bird response in
black and the best sampled model median in its fixed scenario colour and
line style (Access dashed). The grey empirical band was digitized from Figure 2
and is treated as an assumed pointwise 95% CI. Its coverage and construction
are unverified in the source; unreadable limits remain missing. These bands
are approximate illustration, not recovered joint curve uncertainty. Model
medians describe patches across a dimensionless simulated landscape, while
the empirical response describes sampled forest sites and forest cover within
600m. Similar shape does not identify biological parameters or validate all
mechanisms. Five seeds limit precision. Seed omission and empirical-central-
source sensitivity are descriptive, not inferential intervals.

The old/new comparison table separately recomputes the former 128 profiles
using their first five matched seeds on the identical H domain. This separates
parameter-design differences from the previous 20-versus-five-seed comparison.
It does not turn the two different designs into the same target distribution.
Earlier 128/20 surfaces and the 200-seed reference are preserved.

All figures are SVG/PDF/300-dpi PNG. Actual standalone visual review is pending
until the explicit confirmation stage; manuscript-size review remains pending.
Formal identifiability regions and biological archetypes are not inferred.
'''
    (out/'README.md').write_text(note)
    inputs=[CONFIG,Path(__file__),Path(BASE.__file__),snap/'11_numerical_audit.json',FINAL/c['domain_strips'],bird/'manifest.json',bird/'figure2_digitization_manifest.json',bird/'11_bird_curves_main_H_grid.csv',bird/'11_bird_curve_coefficients.csv',bird/'11_bird_curves_main_H_grid_with_figure2_bands.csv',FINAL/c['old_score_table']]
    hashes.update({str(p.relative_to(FINAL)):sha(p) for p in inputs})
    outputs=[p for p in list(data.rglob('*'))+list(out.rglob('*')) if p.is_file()]
    record=dict(status='numeric_and_export_checks_passed_visual_pending',n_species=15,n_profiles=256,n_seed_blocks=5,n_scenarios=7,n_H=len(common),H_min=min(common),H_max=max(common),n_score_rows=15*7*256,maximum_scalar_rmse_error=error,maximum_interpolation_recovery_error=interpolation_error,maximum_sum_linearity_error=linearity_error,best_rmse_range=[float(scores.min(axis=(1,2)).min()),float(scores.min(axis=(1,2)).max())],maximum_seed_omission_rmse_change=float(pd.DataFrame(stability).maximum_scenario_rmse_change.max()),visual_review_confirmed=False,export_checks=exports,input_sha256=hashes,output_sha256={str(p.relative_to(FINAL)):sha(p) for p in outputs},old_results_preserved=True)
    BASE.write_json(out/'logs/11_bird_update_audit.json',record);print(json.dumps({k:record[k] for k in ['status','n_H','best_rmse_range','maximum_seed_omission_rmse_change']},indent=2))


def confirm():
    c=json.loads(CONFIG.read_text());p=FINAL/c['results_directory']/'logs/11_bird_update_audit.json';a=json.loads(p.read_text())
    for name,h in {**a['input_sha256'],**a['output_sha256']}.items():assert sha(FINAL/name)==h,name
    a.update(status='numerically_and_visually_checked_preliminary',visual_review_confirmed=True,visual_review_at_utc=datetime.now(timezone.utc).isoformat());BASE.write_json(p,a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--confirm-visual-review',action='store_true');a=p.parse_args()
    confirm() if a.confirm_visual_review else run()
