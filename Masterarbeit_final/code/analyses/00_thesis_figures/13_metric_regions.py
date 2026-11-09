"""Section5.3 review figures and descriptive A/B/C checks from audited data."""
from pathlib import Path
import os
import sys
import json
import hashlib
import warnings
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-metric-regions-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,MISSING

LABELS=['Edge density','Largest cluster\nfraction',r'$\chi$','Mean NN distance','Mean inter-patch\ndistance',r'Complete-MST $D_{50}$']
METRICS=['edge_density_total','largest_cluster_fraction_forest','chi','mean_nearest_neighbor_distance','mean_pairwise_distance','complete_mst_d50']
INPUTS={}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def checked(p,expected=None):
    h=sha(p)
    if expected is not None:assert h==expected,p
    INPUTS[str(p.relative_to(FINAL))]=h
    return p
def read(p):return pd.read_csv(checked(p),float_precision='round_trip')
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def emit(fig,out,name):
    stem=out/'figures'/name;export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def region_mask(values,half_distance,rule):
    result=(values<rule['threshold']) if rule['operator']=='lt' else (values>=rule['threshold'])
    if 'max_half_distance' in rule:result=result&(half_distance<=rule['max_half_distance'])
    return np.isfinite(values)&result

def map_axes(ax):
    ax.set(xscale='log',yscale='log',xlim=(1,512),ylim=(10**-2.5,10**5.75),facecolor=MISSING)
    ax.set_xticks([1,16,256],['1','16','256']);ax.set_yticks([.01,1,100,10000],[r'$10^{-2}$',r'$10^0$',r'$10^2$',r'$10^4$']);ax.minorticks_off()

def maps(out,surf,indices,name,c,members=None):
    fig,axes=plt.subplots(1,3,figsize=figure_size(95),sharex=True,sharey=True)
    fig.subplots_adjust(left=.13,right=.86,top=.88,bottom=.25,wspace=.22)
    for letter,(ax,m) in enumerate(zip(axes,indices)):
        im=ax.pcolormesh(surf['grid_half_distance'],surf['grid_r'],np.ma.masked_invalid(surf['median_signed'][m]),cmap='RdBu_r',vmin=-1,vmax=1,shading='nearest',rasterized=True)
        map_axes(ax);ax.text(0,1.06,f'({chr(97+letter)}) '+LABELS[m],transform=ax.transAxes)
        if members is not None:
            ax.scatter(members.half_distance,members.r,s=4,c='#222222',alpha=.35,zorder=5)
            for reg,rule in c['regions'].items():
                field=surf['median_signed'][METRICS.index(rule['metric'])]
                mask=region_mask(field,surf['grid_half_distance'][None,:],rule)
                drawn=np.where(np.isfinite(field),mask.astype(float),np.nan)
                ax.contour(surf['grid_half_distance'],surf['grid_r'],drawn,levels=[.5],colors='#222222',linewidths=1,linestyles='--',zorder=6)
                points=members[members[f'in_region_{reg}']]
                cx=10**np.median(np.log10(points.half_distance));cy=10**np.median(np.log10(points.r))
                ax.text(cx,cy,reg,ha='center',va='center',weight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2),zorder=7)
    cb=fig.add_axes([.90,.28,.018,.55]);fig.colorbar(im,cax=cb,ticks=[-1,0,1],label=r'Median signed $r_s$')
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)');fig.supylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
    handles=[Patch(facecolor=MISSING,label='Outside sampled support or undefined')]
    if members is not None:handles.append(Line2D([],[],color='#222222',ls='--',label='Exploratory regions A/B/C; dots: sampled profiles'))
    fig.legend(handles=handles,loc='lower center',ncol=1,frameon=False)
    emit(fig,out,name)

def main():
    cp=FINAL/'config/00_thesis_figures/13_metric_regions_20261102.json';c=json.loads(checked(cp).read_text());checked(Path(__file__))
    source=FINAL/'results'/c['source'];data=FINAL/'data/processed'/c['source'];out=FINAL/c['output']
    for sub in ['figures','tables','logs']:(out/sub).mkdir(parents=True,exist_ok=True)
    audit=json.loads(checked(source/'logs/12_final_audit.json').read_text())
    for name in ['12_within_H_associations.csv','12_parameter_metric_summaries.csv']:
        path=source/'tables'/name;checked(path,audit['output_sha256'][str(path.relative_to(FINAL))])
    prep=json.loads(checked(source/'logs/12_preparation.json',audit['input_sha256'][str((source/'logs/12_preparation.json').relative_to(FINAL))]).read_text())
    for name in ['12_prepared_arrays.npz','12_parameter_design.csv']:
        path=data/name;checked(path,prep['output_sha256'][str(path.relative_to(FINAL))])
    surf_path=checked(data/'12_parameter_surfaces.npz',audit['output_sha256'][str((data/'12_parameter_surfaces.npz').relative_to(FINAL))])
    z=dict(np.load(data/'12_prepared_arrays.npz',allow_pickle=False));surf=dict(np.load(surf_path,allow_pickle=False))
    design=read(data/'12_parameter_design.csv');table=read(source/'tables/12_within_H_associations.csv');summary=read(source/'tables/12_parameter_metric_summaries.csv')
    config=json.loads((FINAL/'config/02_metapop/02_03_main_analyses/02_03_03_metric_associations/12_result_d.json').read_text());assert config['metrics']==METRICS
    assert design.parameter_id.tolist()==z['parameter_ids'].tolist() and z['rho'].shape==(88,6,256)
    # Numeric display matches saved rows, including all undefined coefficients.
    for m,metric in enumerate(METRICS):
        arr=table[table.metric.eq(metric)].pivot(index='H',columns='parameter_id',values='spearman').reindex(index=z['H'],columns=z['parameter_ids']).to_numpy()
        np.testing.assert_allclose(arr,z['rho'][:,m,:],atol=0,rtol=0,equal_nan=True)
    xy=np.column_stack([np.log10(design.half_distance),np.log10(design.r)])
    sampled=summary.pivot(index='parameter_id',columns='metric',values='median_signed_spearman').reindex(design.parameter_id)
    for reg,rule in c['regions'].items():
        design[f'in_region_{reg}']=region_mask(sampled[rule['metric']].to_numpy(),design.half_distance.to_numpy(),rule)
    bits=design[[f'in_region_{reg}' for reg in c['regions']]].to_numpy()
    design['region']=[';'.join(reg for reg,yes in zip(c['regions'],row) if yes) or 'unassigned' for row in bits]
    counts={reg:int(design[f'in_region_{reg}'].sum()) for reg in c['regions']}
    counts.update(unassigned=int((bits.sum(axis=1)==0).sum()),overlap=int((bits.sum(axis=1)>1).sum()))
    memberships=pd.DataFrame([dict(parameter_id=row.parameter_id,region=reg) for row in design.itertuples() for reg in c['regions'] if getattr(row,f'in_region_{reg}')])
    design.to_csv(out/'tables/13_profile_memberships.csv',index=False)
    memberships.to_csv(out/'tables/13_region_memberships_long.csv',index=False)
    assert all(counts[reg]>0 for reg in c['regions'])
    representatives={}
    for reg in c['regions']:
        pp=np.flatnonzero(design[f'in_region_{reg}']);center=np.median(xy[pp],axis=0)
        representatives[reg]=int(pp[np.argmin(np.sum(((xy[pp]-center)/[np.log10(512),8.25])**2,axis=1))])
    design.iloc[list(representatives.values())].to_csv(out/'tables/13_representative_profiles.csv',index=False)
    save(out/'logs/13_preflight.json',dict(status='passed',n_profiles=256,profile_counts=counts,r_range=[float(design.r.min()),float(design.r.max())],half_distance_range=[float(design.half_distance.min()),float(design.half_distance.max())],region_scope=c['region_coordinate_system']))
    joined=summary.merge(memberships,on='parameter_id',validate='many_to_many');joined.to_csv(out/'tables/13_per_profile_diagnostics.csv',index=False)
    apply_final_style()
    # Audited reference confidence intervals, reused without new resampling.
    ref=table[table.parameter_id.eq(c['reference'])]
    fig,axes=plt.subplots(1,2,figsize=figure_size(78),sharex=True,sharey=True);fig.subplots_adjust(left=.13,right=.98,bottom=.30,top=.88,wspace=.18)
    for i,(ax,m) in enumerate(zip(axes,[0,1])):
        g=ref[ref.metric.eq(METRICS[m])].sort_values('H');ax.axhline(0,color='#999999',lw=.6)
        assert g.loc[g.ci_low.notna(),'bootstrap_converged'].all()
        ax.fill_between(g.H,g.ci_low,g.ci_high,color='#377A9F',alpha=.22,lw=0)
        ax.plot(g.H,g.spearman,color='#377A9F');ax.set(xlim=(1,0),ylim=(-1,1));ax.set_xticks([1,.5,0]);ax.set_yticks([-1,0,1]);ax.text(0,1.07,f'({chr(97+i)}) '+LABELS[m],transform=ax.transAxes)
    fig.supxlabel(r'Forest proportion $H$');fig.supylabel(r'Scenario-balanced Spearman $r_s$')
    fig.legend(handles=[Line2D([],[],color='#377A9F',label='Reference profile'),Patch(facecolor='#377A9F',alpha=.22,label='95% pointwise whole-run bootstrap CI')],loc='lower center',ncol=1,frameon=False)
    emit(fig,out,'13_reference_ED_LCF')
    maps(out,surf,[0,1,2],'13_structure_parameter_maps',c)
    maps(out,surf,[3,4,5],'13_connectivity_parameter_maps',c)
    maps(out,surf,[3,1,2],'13_provisional_regions',c,design)
    curves=[];regional=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);span=np.nanmax(z['y'],axis=2)-np.nanmin(z['y'],axis=2)
    for indices,name in [([0,1,2],'13_structure_regions_H'),([3,4,5],'13_connectivity_regions_H')]:
        fig,axes=plt.subplots(3,3,figsize=figure_size(178),sharex=True,sharey=True);fig.subplots_adjust(left=.18,right=.98,bottom=.22,top=.94,hspace=.37,wspace=.24)
        for row,reg in enumerate(c['regions']):
            pp=np.flatnonzero(design[f'in_region_{reg}']);n=len(pp)
            axes[row,0].text(-.32,.5,f'Region\n{reg}',transform=axes[row,0].transAxes,ha='center',va='center',fontsize=12,fontweight='bold')
            for col,m in enumerate(indices):
                ax=axes[row,col];values=z['rho'][:,m,pp];filtered=np.where(span[:,pp]>=c['response_range_sensitivity'],values,np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning);lo,mid,hi=np.nanquantile(values,[.05,.5,.95],axis=1);restricted=np.nanmedian(filtered,axis=1)
                ax.axhline(0,color='#999999',lw=.5);ax.fill_between(z['H'],lo,hi,color='#377A9F',alpha=.18,lw=0);ax.plot(z['H'],mid,color='#377A9F',lw=1.6);ax.plot(z['H'],restricted,color='#CC5500',ls='--',lw=1.3)
                ax.plot(z['H'],z['rho'][:,m,representatives[reg]],color='#333333',ls=':',lw=1.1)
                ax.set(xlim=(1,0),ylim=(-1,1));ax.set_xticks([1,.5,0]);ax.set_yticks([-1,0,1]);ax.text(0,1.06,f'({chr(97+row*3+col)}) '+LABELS[m],transform=ax.transAxes,fontsize=9.5)
                for k,h in enumerate(z['H']):
                    curves.append(dict(region=reg,metric=METRICS[m],H=float(h),n_profiles=n,n_defined=int(np.isfinite(values[k]).sum()),n_span005=int(np.isfinite(filtered[k]).sum()),q05=lo[k],median=mid[k],q95=hi[k],span005_median=restricted[k]))
                g=joined[joined.region.eq(reg)&joined.metric.eq(METRICS[m])]
                regional.append(dict(region=reg,metric=METRICS[m],n_profiles=n,median_of_signed_medians=float(g.median_signed_spearman.median()),median_of_absolute_medians=float(g.median_absolute_spearman.median()),median_sign_consistency=float(g.sign_consistency.median()),median_n_informative_H=float(g.n_informative_H.median()),median_n_span005_H=float(g.n_span_at_least_005_H.median())))
        fig.supxlabel(r'Forest proportion $H$');fig.supylabel(r'Scenario-balanced Spearman $r_s$')
        legend_counts='; '.join(f'{reg}: {int(design[f'in_region_{reg}'].sum())}' for reg in c['regions'])
        fig.legend(handles=[Line2D([],[],color='#377A9F',label='Median across defined profiles in region'),Patch(facecolor='#377A9F',alpha=.18,label='5–95% profile range (not a confidence interval)'),Line2D([],[],color='#CC5500',ls='--',label=r'Median restricted to occupancy range $\geq 0.05$'),Line2D([],[],color='#333333',ls=':',label='Fixed representative profile in each region'),Line2D([],[],ls='none',label='Sampled profiles: '+legend_counts)],loc='lower center',ncol=1,frameon=False)
        emit(fig,out,name)
    pd.DataFrame(curves).to_csv(out/'tables/13_region_by_H.csv',index=False);pd.DataFrame(regional).to_csv(out/'tables/13_region_summary.csv',index=False)
    save(out/'logs/13_audit.json',dict(status='numeric_checks_passed_visual_pending',input_sha256=INPUTS,n_profiles=256,n_H=88,n_defined_H_levels=int(np.isfinite(z['rho']).any(axis=(1,2)).sum()),profile_counts=counts,all_saved_coefficients_recovered=True,reference_intervals_reused=True,limitations=[c['interpretation'],'Blue regional curves and orange restricted curves may summarise different profiles at each H. Counts saved; no confidence inference for regional summaries.','Regions were chosen visually from signed maps and the author sketch; findings are exploratory and not an independent test.']))
    print(json.dumps({'profile_counts':counts,'figures':6,'output':str(out)},indent=2))

if __name__=='__main__':main()
