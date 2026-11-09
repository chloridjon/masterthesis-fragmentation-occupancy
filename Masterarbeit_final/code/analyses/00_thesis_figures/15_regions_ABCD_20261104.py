"""Adopt the author's four regions using the full, unchanged Result-D ensemble."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-regions-abcd-mpl')
import sys, json, hashlib, importlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,MISSING
BASE=importlib.import_module('13_metric_regions')
INPUTS={}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def checked(p,expected=None):
    digest=sha(p)
    if expected is not None: assert digest==expected,p
    INPUTS[str(p.relative_to(FINAL))]=digest
    return p
def read(p,expected=None): return pd.read_csv(checked(p,expected),float_precision='round_trip')
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def emit(fig,out,name):
    stem=out/'figures'/name;export_figure(fig,stem)
    fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    cp=FINAL/'config/00_thesis_figures/15_regions_ABCD_20261104.json';c=json.loads(checked(cp).read_text());checked(Path(__file__))
    out=FINAL/c['output'];old=FINAL/c['previous_regions'];src=FINAL/'results'/c['source'];data=FINAL/'data/processed'/c['source']
    for sub in ['tables','figures','logs']:(out/sub).mkdir(parents=True,exist_ok=True)
    audit=json.loads(checked(src/'logs/12_final_audit.json').read_text())
    prep=json.loads(checked(src/'logs/12_preparation.json').read_text())
    sp=data/'12_parameter_surfaces.npz';checked(sp,audit['output_sha256'][str(sp.relative_to(FINAL))]);surf=dict(np.load(sp,allow_pickle=False))
    zp=data/'12_prepared_arrays.npz';checked(zp,prep['output_sha256'][str(zp.relative_to(FINAL))])
    with np.load(zp,allow_pickle=False) as f:z={k:f[k] for k in ['H','rho','y','parameter_ids','r','half_distance']}
    summary=read(src/'tables/12_parameter_metric_summaries.csv',audit['output_sha256'][str((src/'tables/12_parameter_metric_summaries.csv').relative_to(FINAL))])
    table=read(src/'tables/12_within_H_associations.csv',audit['output_sha256'][str((src/'tables/12_within_H_associations.csv').relative_to(FINAL))])
    design=read(data/'12_parameter_design.csv');previous=read(old/'tables/13_profile_memberships.csv')
    boundary=read(FINAL/c['C_boundary']);sampled=summary.pivot(index='parameter_id',columns='metric',values='median_signed_spearman').reindex(design.parameter_id)
    assert design.parameter_id.tolist()==z['parameter_ids'].tolist()==previous.parameter_id.tolist()
    cut=np.exp(np.interp(np.log(design.half_distance),np.log(boundary.half_distance),np.log(boundary.proposed_C_lower_r)))
    for reg in ['A','B']:
        rule=c['regions'][reg]
        design[f'in_region_{reg}']=BASE.region_mask(sampled[rule['metric']].to_numpy(),design.half_distance.to_numpy(),rule)
        np.testing.assert_array_equal(design[f'in_region_{reg}'],previous[f'in_region_{reg}'])
    design['C_boundary_r']=cut
    design['in_region_C']=(design.r>cut)&np.isfinite(sampled.largest_cluster_fraction_forest.to_numpy())
    design['in_region_D']=(design.half_distance>16)&(sampled.mean_nearest_neighbor_distance.to_numpy()<=-.3)&(design.r<cut)
    bits=design[[f'in_region_{reg}' for reg in c['region_order']]].to_numpy()
    assert np.all(bits.sum(axis=1)<=1)
    assert np.all(design.loc[previous.in_region_C,'in_region_C'])
    design['region']=[';'.join(r for r,yes in zip(c['region_order'],row) if yes) or 'unassigned' for row in bits]
    counts={r:int(design[f'in_region_{r}'].sum()) for r in c['region_order']};counts['unassigned']=int((bits.sum(axis=1)==0).sum())
    assert counts==dict(A=61,B=124,C=49,D=6,unassigned=16),counts
    design.to_csv(out/'tables/15_profile_memberships.csv',index=False)
    members=pd.DataFrame([dict(parameter_id=row.parameter_id,region=r) for row in design.itertuples() for r in c['region_order'] if getattr(row,f'in_region_{r}')])
    members.to_csv(out/'tables/15_region_memberships_long.csv',index=False)
    joined=summary.merge(members,on='parameter_id',validate='many_to_many');joined.to_csv(out/'tables/15_per_profile_diagnostics.csv',index=False)
    oldreps=read(old/'tables/13_representative_profiles.csv')
    xy=np.column_stack([np.log10(design.half_distance),np.log10(design.r)])
    reps={}
    for reg in c['region_order']:
        pp=np.flatnonzero(design[f'in_region_{reg}']);center=np.median(xy[pp],axis=0)
        reps[reg]=int(pp[np.argmin(np.sum(((xy[pp]-center)/[np.log10(512),8.25])**2,axis=1))])
        if reg in ['A','B']:
            oldid=oldreps.loc[oldreps[f'in_region_{reg}'],'parameter_id'].iloc[0]
            assert design.iloc[reps[reg]].parameter_id==oldid
    design.iloc[list(reps.values())].to_csv(out/'tables/15_representative_profiles.csv',index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);span=np.nanmax(z['y'],axis=2)-np.nanmin(z['y'],axis=2)
    apply_final_style();curves=[]
    # Same map fields and adopted boundaries as the selected preview.
    ell=surf['grid_half_distance'];rr=surf['grid_r'];field=surf['median_signed'];cut_grid=boundary.proposed_C_lower_r.to_numpy()
    np.testing.assert_array_equal(ell,boundary.half_distance)
    fig,axes=plt.subplots(1,3,figsize=figure_size(95),sharex=True,sharey=True);fig.subplots_adjust(left=.13,right=.86,top=.88,bottom=.25,wspace=.22)
    for i,(ax,m) in enumerate(zip(axes,[3,1,2])):
        im=ax.pcolormesh(ell,rr,np.ma.masked_invalid(field[m]),cmap='RdBu_r',vmin=-1,vmax=1,shading='nearest',rasterized=True)
        BASE.map_axes(ax);ax.text(0,1.06,f'({chr(97+i)}) '+BASE.LABELS[m],transform=ax.transAxes)
        ax.scatter(design.half_distance,design.r,s=4,c='#222222',alpha=.35,zorder=5)
        for reg in ['A','B']:
            rule=c['regions'][reg];f=field[BASE.METRICS.index(rule['metric'])];mask=BASE.region_mask(f,ell[None,:],rule)
            ax.contour(ell,rr,np.where(np.isfinite(f),mask.astype(float),np.nan),levels=[.5],colors='#222222',linewidths=1,linestyles='--',zorder=6)
        ci=np.argmin(abs(rr[:,None]-cut_grid[None,:]),axis=0);supported=np.isfinite(field[1])[ci,np.arange(len(ell))]
        ax.plot(ell,np.where(supported,cut_grid,np.nan),color='#222222',lw=1,ls='--',zorder=7)
        nn=np.where((ell[None,:]>16)&(rr[:,None]<cut_grid[None,:]),field[3],np.nan)
        ax.contour(ell,rr,nn,levels=[-.3],colors='#222222',linewidths=1,linestyles='--',zorder=7)
        for reg,(x,y) in dict(A=(3.6,.36),B=(21,550),C=(13,85000),D=(400,80)).items():
            ax.text(x,y,reg,ha='center',va='center',weight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2),zorder=8)
    cb=fig.add_axes([.90,.28,.018,.55]);fig.colorbar(im,cax=cb,ticks=[-1,0,1],label=r'Median signed $r_s$')
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)');fig.supylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
    fig.legend(handles=[Patch(facecolor=MISSING,label='Outside sampled support or undefined'),Line2D([],[],color='#222222',ls='--',label='Exploratory regions A/B/C/D; dots: sampled profiles')],loc='lower center',frameon=False,ncol=1)
    emit(fig,out,'15_regions_ABCD')
    for indices,name in [([0,1,2],'15_structure_regions_H'),([3,4,5],'15_connectivity_regions_H')]:
        fig,axes=plt.subplots(4,3,figsize=figure_size(200),sharex=True,sharey=True)
        fig.subplots_adjust(left=.18,right=.98,bottom=.19,top=.96,hspace=.42,wspace=.24)
        for row,reg in enumerate(c['region_order']):
            pp=np.flatnonzero(design[f'in_region_{reg}']);n=len(pp)
            axes[row,0].text(-.32,.5,f'Region\n{reg}',transform=axes[row,0].transAxes,ha='center',va='center',fontsize=11,fontweight='bold')
            for col,m in enumerate(indices):
                ax=axes[row,col];values=z['rho'][:,m,pp];filtered=np.where(span[:,pp]>=.05,values,np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning);lo,mid,hi=np.nanquantile(values,[.05,.5,.95],axis=1);restricted=np.nanmedian(filtered,axis=1)
                ax.axhline(0,color='#999999',lw=.5);ax.fill_between(z['H'],lo,hi,color='#377A9F',alpha=.18,lw=0)
                ax.plot(z['H'],mid,color='#377A9F',lw=1.6)
                ax.plot(z['H'],restricted,color='#CC5500',lw=1.3)
                ax.plot(z['H'],z['rho'][:,m,reps[reg]],color='#333333',lw=.7,alpha=.8,marker='.',markevery=6,ms=2)
                title=r'Complete-MST'+'\n'+r'$D_{50}$' if m==5 else BASE.LABELS[m]
                ax.set(xlim=(1,0),ylim=(-1,1),xticks=[1,.5,0],yticks=[-1,0,1]);ax.text(0,1.06,f'({chr(97+row*3+col)}) '+title,transform=ax.transAxes,fontsize=9.5)
                for k,h in enumerate(z['H']):
                    curves.append(dict(region=reg,metric=BASE.METRICS[m],H=float(h),n_profiles=n,n_defined=int(np.isfinite(values[k]).sum()),n_span005=int(np.isfinite(filtered[k]).sum()),q05=lo[k],median=mid[k],q95=hi[k],span005_median=restricted[k]))
        fig.supxlabel(r'Forest proportion $H$');fig.supylabel(r'Scenario-balanced Spearman $r_s$')
        counttext='; '.join(f'{r}: {counts[r]}' for r in c['region_order'])
        fig.legend(handles=[Line2D([],[],color='#377A9F',label='Median across defined profiles in region'),Patch(facecolor='#377A9F',alpha=.18,label='5–95% profile range (not a confidence interval)'),Line2D([],[],color='#CC5500',label=r'Median restricted to occupancy range $\geq 0.05$'),Line2D([],[],color='#333333',lw=.7,marker='.',ms=2,label='Fixed representative profile in each region'),Line2D([],[],ls='none',label='Sampled profiles: '+counttext)],loc='lower center',ncol=1,frameon=False)
        emit(fig,out,name)
    actual=pd.DataFrame(curves);actual.to_csv(out/'tables/15_region_by_H.csv',index=False)
    # Independent reconstruction from the original CSV, not the prepared array.
    original=table.merge(members,on='parameter_id',validate='many_to_many');group=original.groupby(['region','metric','H']).spearman
    q=group.quantile([.05,.5,.95]).unstack();q.columns=['q05','median','q95'];q['n_defined']=group.count()
    f=original[original.occupancy_span>=.05].groupby(['region','metric','H']).spearman
    q['span005_median']=f.median();q['n_span005']=f.count();q['n_span005']=q.n_span005.fillna(0)
    idx=actual.set_index(['region','metric','H']).sort_index();q=q.reindex(idx.index)
    for col in ['q05','median','q95','n_defined','n_span005','span005_median']:np.testing.assert_allclose(idx[col],q[col],atol=2e-15,rtol=0,equal_nan=True)
    oldcurves=read(old/'tables/13_region_by_H.csv').set_index(['region','metric','H']).sort_index()
    for reg in ['A','B']:
        for col in ['q05','median','q95','n_defined','n_span005','span005_median']:np.testing.assert_allclose(idx.loc[reg,col],oldcurves.loc[reg,col].reindex(idx.loc[reg].index),atol=2e-15,rtol=0,equal_nan=True)
    # Independent scalar membership checks against original summaries.
    raw=summary.set_index(['parameter_id','metric'])
    for row in design.itertuples():
        lo=row.C_boundary_r
        nn=raw.loc[(row.parameter_id,'mean_nearest_neighbor_distance'),'median_signed_spearman']
        lcf=raw.loc[(row.parameter_id,'largest_cluster_fraction_forest'),'median_signed_spearman']
        assert row.in_region_C==bool(row.r>lo and np.isfinite(lcf))
        assert row.in_region_D==bool(row.half_distance>16 and nn<=-.3 and row.r<lo)
    joined.groupby(['region','metric']).agg(n_profiles=('parameter_id','size'),median_signed=('median_signed_spearman','median'),median_absolute=('median_absolute_spearman','median')).reset_index().to_csv(out/'tables/15_region_summary.csv',index=False)
    save(out/'logs/15_numeric_audit.json',dict(status='passed_visual_review_pending',input_sha256=INPUTS,profile_counts=counts,n_profiles=256,n_H=88,n_region_metric_H_rows_checked=len(actual),old_C_profiles_retained=18,unchanged_A_B_membership_and_curves=True,all_percentiles_independently_reproduced=True,new_landscape_simulations=0,new_equilibrium_solves=0,new_bootstrap_draws=0,scope=c['boundary_scope'],style='Regional summary and restricted median are solid blue/orange; fixed example is thin grey with occasional dots. Dashed lines reserved for access-based response curves; map boundaries remain dashed.'))
    print(json.dumps(dict(profile_counts=counts,output=str(out)),indent=2))

if __name__=='__main__':main()
