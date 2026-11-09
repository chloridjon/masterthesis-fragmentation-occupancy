"""Assess the proposed response-domain rule on the 128-profile existing LHS."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-lhs-domain-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from src.tools.final_plot_style import apply_final_style,figure_size,export_figure,MISSING,NEUTRAL,FOCAL_SCENARIOS

DEFAULT_CONFIG=FINAL/'config/02_metapop/02_02_species_parameter_selection/lhs_128_response_domain/09_lhs_response_domain.json'
LABELS={0:'Near-one throughout',1:'Retained response domain',2:'Near-zero throughout',3:'Boundary unresolved',4:'Outside sampled support'}
COLORS={0:'#0072B2',1:'#009E73',2:'#E69F00',3:'#F7F7F7',4:MISSING}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def classify(lower,upper,epsilon):
    lower,upper=np.asarray(lower),np.asarray(upper)
    if not (0<epsilon<.5):raise ValueError('Saturation threshold must lie between 0 and .5')
    result=np.full(lower.shape,3,dtype=int)
    finite=np.isfinite(lower)&np.isfinite(upper)
    result[finite&(lower>=1-epsilon)]=0
    result[finite&(upper<=epsilon)]=2
    result[finite&(lower<1-epsilon)&(upper>epsilon)]=1
    return result


def quantiles_by_valid_pattern(values,minimum):
    """Quantile across full seed blocks; keep landscape eligibility distinct."""
    n,s,h,p=values.shape
    flat=values.reshape(n,s*h,p)
    mask=np.isfinite(flat[:,:,0]).T
    assert np.array_equal(np.isfinite(flat),np.broadcast_to(mask.T[:,:,None],flat.shape))
    patterns,inverse=np.unique(mask,axis=0,return_inverse=True)
    result=np.full((2,s*h,p),np.nan)
    count=mask.sum(axis=1)
    for i,seeds in enumerate(patterns):
        if seeds.sum()<minimum:continue
        cells=np.flatnonzero(inverse==i)
        result[:,cells]=np.quantile(flat[seeds][:,cells,:],[.05,.95],axis=0,method='linear')
    return count.reshape(s,h),result.reshape(2,s,h,p)


def monotone_classes(points,lower,upper,targets,epsilon):
    """Infer only what the model's coordinate-wise monotonicity implies."""
    # Targets have x=log half-distance and y=log r, both unit-normalised.
    more=(targets[:,0,None]>=points[:,0])&(targets[:,1,None]<=points[:,1])
    less=(targets[:,0,None]<=points[:,0])&(targets[:,1,None]>=points[:,1])
    near_one=(more&(lower>=1-epsilon)).any(axis=1)
    near_zero=(less&(upper<=epsilon)).any(axis=1)
    not_one=(less&(lower<1-epsilon)).any(axis=1)
    not_zero=(more&(upper>epsilon)).any(axis=1)
    core=not_one&not_zero
    if ((near_one&near_zero)|(near_one&core)|(near_zero&core)).any():
        raise ValueError('Contradictory monotonic bounds; inspect source responses')
    result=np.full(len(targets),3,dtype=int)
    result[near_one]=0;result[core]=1;result[near_zero]=2
    return result


def boundary_brackets(points,lower,upper,x,epsilon,lo,hi):
    rows=[]
    def endpoint(mask,kind):
        if not mask.any():return np.nan
        return float((np.max if kind=='max' else np.min)(points[mask,1]))
    for along in x:
        # Each bracket uses a dominating actual observation, not a fitted line.
        low_a=endpoint((points[:,0]<=along)&(lower>=1-epsilon),'max')
        low_b=endpoint((points[:,0]>=along)&(lower<1-epsilon),'min')
        high_a=endpoint((points[:,0]<=along)&(upper>epsilon),'max')
        high_b=endpoint((points[:,0]>=along)&(upper<=epsilon),'min')
        row={'half_distance':float(10**(lo[0]+along*(hi[0]-lo[0])))}
        for name,v in [('lower_boundary_bracket_low',low_a),('lower_boundary_bracket_high',low_b),
                       ('upper_boundary_bracket_low',high_a),('upper_boundary_bracket_high',high_b)]:
            row[name]=float(10**(lo[1]+v*(hi[1]-lo[1]))) if np.isfinite(v) else np.nan
        row['lower_boundary_bracket_complete']=np.isfinite([low_a,low_b]).all()
        row['upper_boundary_bracket_complete']=np.isfinite([high_a,high_b]).all()
        rows.append(row)
    return pd.DataFrame(rows)


def axes_style(ax):
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlim(1,512);ax.set_ylim(10**-2.5,10**5.75)
    ax.set_xticks([1,4,16,64,256],['1','4','16','64','256'])
    ax.set_yticks([.01,1,100,10000],[r'$10^{-2}$',r'$10^0$',r'$10^2$',r'$10^4$'])
    ax.minorticks_off()


def plot_main(profiles,ell,r,region,results,epsilon):
    apply_final_style()
    fig,axes=plt.subplots(1,2,figsize=figure_size(108),sharex=True,sharey=True)
    fig.subplots_adjust(left=.13,right=.98,bottom=.30,top=.93,wspace=.27)
    for ax in axes:axes_style(ax)
    for category in (0,1,2):
        part=profiles[profiles.category==category]
        axes[0].scatter(part.half_distance,part.r,s=16,color=COLORS[category],edgecolors='white',lw=.25,zorder=3)
    axes[0].text(0,1.055,'(a) Computed LHS profiles',transform=axes[0].transAxes)
    axes[1].text(0,1.055,'(b) Monotonicity bounds',transform=axes[1].transAxes)
    pale=['#CFDFEB','#BDE3D7','#F2DEAD',COLORS[3],MISSING]
    axes[1].pcolormesh(ell,r,region,cmap=ListedColormap(pale),norm=BoundaryNorm(np.arange(-.5,5.5),5),shading='nearest',rasterized=True)
    axes[1].contourf(ell,r,np.where(region==3,1.,np.nan),levels=[.5,1.5],colors='none',hatches=['////'])
    for category in (0,1,2):
        part=profiles[profiles.category==category]
        axes[1].scatter(part.half_distance,part.r,s=8,color=COLORS[category],edgecolors='white',lw=.2,zorder=4)
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    handles=[Line2D([],[],linestyle='none',marker='o',ms=4,color=COLORS[i],label=f'{LABELS[i]} ({int((profiles.category==i).sum())})') for i in (0,1,2)]
    handles += [Patch(facecolor='white',edgecolor=NEUTRAL,hatch='////',label='Bounds unresolved (not a CI)'),Patch(facecolor=MISSING,label=LABELS[4])]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    stem=results/'figures/09_lhs_response_domain'
    export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    return stem


def plot_sensitivity(profiles,categories,results):
    apply_final_style()
    fig,axes=plt.subplots(1,3,figsize=figure_size(86),sharex=True,sharey=True)
    fig.subplots_adjust(left=.13,right=.98,bottom=.35,top=.91,wspace=.30)
    for i,(epsilon,category) in enumerate(categories.items()):
        ax=axes[i];axes_style(ax)
        for kind in (0,1,2):
            part=profiles[category==kind]
            ax.scatter(part.half_distance,part.r,s=12,color=COLORS[kind],edgecolors='white',lw=.25)
        ax.text(0,1.055,f'({chr(97+i)}) {epsilon:g}–{1-epsilon:g}',transform=ax.transAxes)
        ax.set_xlabel(f'Retained: {int((category==1).sum())}/128',fontsize=8.5,labelpad=5)
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    fig.legend(handles=[Line2D([],[],ls='none',marker='o',ms=4,color=COLORS[i],label=LABELS[i]) for i in (0,1,2)],loc='lower center',ncol=2,frameon=False)
    stem=results/'figures/09_lhs_domain_threshold_sensitivity'
    export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    return stem


def run(config_path):
    c=json.loads(config_path.read_text())
    source_audit=json.loads((FINAL/c['lhs_audit']).read_text())
    assert source_audit['status']=='passed' and source_audit['source_hashes_verified']
    plausibility=json.loads((FINAL/c['lhs_plausibility']).read_text())
    assert plausibility['status']=='complete_initial_batch_plausibility_passed'
    lc=json.loads((FINAL/c['lhs_configuration']).read_text())
    assert lc['q']==0 and len(lc['replicates'])==20
    manifest=json.loads((FINAL/c['lhs_data']/'manifest.json').read_text())
    design_path=FINAL/lc['lhs_design'];assert sha(design_path)==manifest['source_hashes'][lc['lhs_design']]
    design=pd.read_csv(design_path,float_precision='round_trip')
    hs=lc['h_targets'];ids=design.parameter_id.tolist();scenarios=[s['scenario_id'] for s in FOCAL_SCENARIOS]
    values=np.full((20,7,len(hs),128),np.nan);source_hashes={}
    for i,rep in enumerate(lc['replicates']):
        for j,sid in enumerate(scenarios):
            path=FINAL/c['lhs_data']/'trajectories'/f'{sid}__rep{rep:03d}.csv'
            part=pd.read_csv(path,usecols=['H_target','parameter_id','occupancy','producer_fingerprint','status'],float_precision='round_trip')
            assert len(part)==len(hs)*128 and not part.duplicated(['H_target','parameter_id']).any()
            assert part.producer_fingerprint.eq(source_audit['producer_fingerprint']).all()
            assert not part.status.eq('nonconverged').any()
            assert part.loc[part.status.eq('valid'),'occupancy'].between(0,1).all()
            assert part.loc[~part.status.eq('valid'),'occupancy'].isna().all()
            values[i,j]=part.pivot(index='H_target',columns='parameter_id',values='occupancy').reindex(index=hs,columns=ids).to_numpy()
            source_hashes[str(path.relative_to(FINAL))]=sha(path)
    count,q=quantiles_by_valid_pattern(values,c['minimum_valid_seeds'])
    saved=pd.read_csv(FINAL/c['lhs_summary'],float_precision='round_trip')
    keys=pd.MultiIndex.from_product([scenarios,hs,ids],names=['scenario_id','H_target','parameter_id']).to_frame(index=False)
    keys['q05_recomputed']=q[0].ravel();keys['q95_recomputed']=q[1].ravel()
    aligned=saved.merge(keys,on=['scenario_id','H_target','parameter_id'],validate='one_to_one')
    error=0.
    for name in ['q05','q95']:
        assert np.array_equal(aligned[name].isna(),aligned[name+'_recomputed'].isna())
        error=max(error,float((aligned[name]-aligned[name+'_recomputed']).abs().max()))
    assert error<1e-14
    lower=np.nanmin(q[0].reshape(-1,128),axis=0);upper=np.nanmax(q[1].reshape(-1,128),axis=0)
    epsilon=c['saturation_threshold'];category=classify(lower,upper,epsilon)
    profiles=design.copy();profiles['min_landscape_q05']=lower;profiles['max_landscape_q95']=upper
    profiles['category']=category;profiles['classification']=[LABELS[i] for i in category]
    # Min/max witnesses explain which model H levels set the global envelope.
    for field,flat,index in [('lower',q[0].reshape(-1,128),np.nanargmin(q[0].reshape(-1,128),axis=0)),
                             ('upper',q[1].reshape(-1,128),np.nanargmax(q[1].reshape(-1,128),axis=0))]:
        profiles[field+'_witness_scenario']=[scenarios[i//len(hs)] for i in index]
        profiles[field+'_witness_H']=[hs[i%len(hs)] for i in index]
    decision=json.loads((FINAL/c['parameter_decision']).read_text())
    low=np.array([np.log10(decision['half_distance_min']),decision['log10_r_min']])
    high=np.array([np.log10(decision['half_distance_max']),decision['log10_r_max']])
    points=(np.column_stack([np.log10(design.half_distance),np.log10(design.r)])-low)/(high-low)
    increasing=(points[:,0,None]>=points[None,:,0])&(points[:,1,None]<=points[None,:,1])
    max_violation=max(float(np.maximum(lower[None,:]-lower[:,None],0)[increasing].max()),
                      float(np.maximum(upper[None,:]-upper[:,None],0)[increasing].max()))
    assert max_violation<1e-9
    assert np.array_equal(monotone_classes(points,lower,upper,points,epsilon),category)
    ny,nx=c['grid_shape'];x=np.linspace(0,1,nx);y=np.linspace(0,1,ny)
    xx,yy=np.meshgrid(x,y);targets=np.column_stack([xx.ravel(),yy.ravel()])
    region=monotone_classes(points,lower,upper,targets,epsilon)
    inside=Delaunay(points).find_simplex(targets)>=0
    region[~inside]=4;region=region.reshape(ny,nx)
    ell=10**(low[0]+x*(high[0]-low[0]));r=10**(low[1]+y*(high[1]-low[1]))
    brackets=boundary_brackets(points,lower,upper,x,epsilon,low,high)
    threshold_records=[];categories={}
    for threshold in c['sensitivity_thresholds']:
        labels=classify(lower,upper,threshold);categories[threshold]=labels
        threshold_records.append({'threshold':threshold,'retained':int((labels==1).sum()),
            'near_one':int((labels==0).sum()),'near_zero':int((labels==2).sum())})
    omit=[];flips=np.zeros(128,dtype=int)
    for i,rep in enumerate(lc['replicates']):
        _,other=quantiles_by_valid_pattern(np.delete(values,i,axis=0),c['minimum_valid_seeds'])
        lo=np.nanmin(other[0].reshape(-1,128),axis=0);hi=np.nanmax(other[1].reshape(-1,128),axis=0)
        labels=classify(lo,hi,epsilon);different=labels!=category;flips+=different
        omit.append({'omitted_seed':rep,'retained':int((labels==1).sum()),'changed_profiles':int(different.sum()),
            'changed_parameter_ids':';'.join(np.array(ids)[different])})
    profiles['seed_omission_classification_changes']=flips
    data,results=FINAL/c['data_directory'],FINAL/c['results_directory']
    for p in [data,results/'figures',results/'logs']:p.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for filename,table in [('09_lhs_domain_classification.csv',profiles),('09_monotonic_boundary_brackets.csv',brackets),
        ('09_threshold_sensitivity.csv',pd.DataFrame(threshold_records)),('09_seed_omission_sensitivity.csv',pd.DataFrame(omit))]:
        path=data/filename;table.to_csv(path,index=False);outputs.append(path)
    path=data/'09_monotonic_domain_display.npz';np.savez_compressed(path,half_distance=ell,r=r,classification=region);outputs.append(path)
    stems=[plot_main(profiles,ell,r,region,results,epsilon),plot_sensitivity(profiles,categories,results)]
    fonts=[]
    for stem in stems:
        listing=subprocess.run(['pdffonts',str(stem.with_suffix('.pdf'))],capture_output=True,text=True,check=True).stdout
        assert 'LMRoman8' in listing and 'LatinModernMath' in listing
        svg=stem.with_suffix('.svg').read_text();assert '<text' in svg and '@font-face' in svg
        fonts.append({'stem':stem.name,'pdf_fonts':listing,'editable_SVG_text':True})
        outputs += [stem.with_suffix(s) for s in ['.svg','.pdf','.png']]
    counts={LABELS[i]:int((category==i).sum()) for i in (0,1,2)}
    fractions={LABELS[i]:float(((region==i)&inside.reshape(ny,nx)).sum()/inside.sum()) for i in (0,1,2,3)}
    note=f'''# Candidate response domain from the existing 128-profile LHS

Panel (a) shows actual computed profiles; green denotes {counts[LABELS[1]]}
retained profiles, blue {counts[LABELS[0]]} uniformly near-one profiles, and
orange {counts[LABELS[2]]} uniformly near-zero profiles. Each profile uses 20
matched landscape seed blocks and all seven fixed scenarios. The baseline
model remains q=0, x=.5, b=1. For each eligible scenario-H cell (at least three
valid landscapes with at least two patches), compute the 5th and 95th
landscape percentiles of mean equilibrium patch occupancy. Across those cells,
let L be the minimum q05 and U the maximum q95. Retain L<.95 and U>.05.
This retains partial/steep transitions and scenario or landscape differences;
it does not require a complete near-one to near-zero habitat response.
It excludes the two globally saturated extremes rather than guaranteeing
habitat-dependent variation: a flat intermediate response also qualifies.
Percentiles describe landscape variation, not a confidence interval.

All eligible model H levels (.005–.975) are used, with 601 eligible
scenario-H cells per profile. H=0 and H=1 are undefined in the q=0 multi-patch
domain; Only-Random H=.975 has too few valid landscapes. No unavailable
response is recoded as zero. The empirical bird H range and bird RMSE do not
enter this domain criterion. Scope: the assessed existing rectangle only,
half-distance 1–512 cells and r from 10^-2.5 to 10^5.75.

Panel (b) uses model monotonicity, not linear surface interpolation: occupancy
increases with half-distance and decreases with r. A computed near-one
profile supports near-one classification at greater half-distance and lower r;
a near-zero profile supports near-zero at shorter half-distance and higher r.
The complementary implications give a retained region only where both
conditions follow from actual sampled points. Pale blue/green/orange show
these classifications; hatching marks unbracketed gaps between sampled points,
not a bootstrap confidence interval. Grey is outside the sampled convex hull,
where no plot classification is made. The stepwise edges are partial-order
limits, not simulated or smoothly fitted boundary curves. Crossings lacking
one bracket side remain explicitly missing in the boundary CSV.

This envelope criterion yields a broad band: {counts[LABELS[1]]}/128 sampled
profiles are retained. It should not be mistaken for the narrower bird-fit
regions. The class grid implies a retained fraction of {fractions[LABELS[1]]:.3f}
within the convex hull in logarithmic coordinates, with unresolved fraction
{fractions[LABELS[3]]:.3f}; these are display-grid fractions, not species
frequencies. Parameter coverage and boundary resolution are distinct from
occupancy solver convergence. The alternative-threshold figure reports
.01–.99 and .10–.90 rules alongside the .05–.95 rule; seed-omission tables
recompute full quantiles by omitting matched blocks, never single snapshots.

The outputs are a candidate-domain diagnostic for discussion. No new LHS,
no new landscape/equilibrium calculation and no final domain adoption were
performed. Result D still checks actual variation at each H within retained
profiles. Actual-size manuscript review remains pending; no LaTeX edits.

Rebuild: `.venv/bin/python -m src.tools.codex_run final-lhs-response-domain`.
'''
    path=results/'09_candidate_domain_note.md';path.write_text(note);outputs.append(path)
    sources=[config_path,Path(__file__),FINAL/c['lhs_summary'],FINAL/c['lhs_audit'],FINAL/c['lhs_plausibility'],
        FINAL/c['lhs_configuration'],design_path,FINAL/c['parameter_decision'],FINAL/'code/base/src/tools/final_plot_style.py']
    source_hashes.update({str(p.relative_to(FINAL)):sha(p) for p in sources})
    audit={'status':'numerically_and_export_checked_visual_review_pending','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'configuration':c,'counts':counts,'threshold_sensitivity':threshold_records,'n_eligible_cells_per_profile':int((count>=3).sum()),
        'maximum_independent_quantile_error':error,'maximum_monotonicity_violation':max_violation,
        'seed_omission_retained_range':[int(min(o['retained'] for o in omit)),int(max(o['retained'] for o in omit))],
        'n_profiles_with_any_seed_omission_change':int((flips>0).sum()),'display_log_area_fractions_within_hull':fractions,
        'complete_lower_boundary_bracket_columns':int(brackets.lower_boundary_bracket_complete.sum()),
        'complete_upper_boundary_bracket_columns':int(brackets.upper_boundary_bracket_complete.sum()),
        'total_boundary_columns':len(brackets),'input_sha256':source_hashes,
        'output_sha256':{str(p.relative_to(FINAL)):sha(p) for p in outputs},'font_checks':fonts,'visual_review_confirmed':False,
        'final_domain_adopted':False,'new_model_runs':0}
    save_json(results/'logs/09_lhs_response_domain_audit.json',audit)
    save_json(data/'manifest.json',{'configuration':c,'input_sha256':source_hashes,'output_sha256':audit['output_sha256'],
        'audit_path':str((results/'logs/09_lhs_response_domain_audit.json').relative_to(FINAL))})
    print(json.dumps({k:audit[k] for k in ['status','counts','threshold_sensitivity','seed_omission_retained_range',
        'n_profiles_with_any_seed_omission_change','complete_lower_boundary_bracket_columns','complete_upper_boundary_bracket_columns']},indent=2))


def confirm(config_path):
    c=json.loads(config_path.read_text());p=FINAL/c['results_directory']/'logs/09_lhs_response_domain_audit.json'
    audit=json.loads(p.read_text())
    for name,digest in {**audit['input_sha256'],**audit['output_sha256']}.items():assert sha(FINAL/name)==digest,name
    audit.update(status='numerically_and_visually_checked_candidate',visual_review_confirmed=True,
        visual_review_at_utc=datetime.now(timezone.utc).isoformat(),visual_review_scope='Both PDF figures rendered and inspected')
    save_json(p,audit);print(audit['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--confirm-visual-review',action='store_true')
    args=parser.parse_args()
    confirm(args.config.resolve()) if args.confirm_visual_review else run(args.config.resolve())
