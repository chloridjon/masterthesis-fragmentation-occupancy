"""Figures 5.11--5.13 restricted to H<=0.6, retaining adopted A--D boundaries."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-low-cover-maps-mpl')
import sys,json,hashlib,importlib,warnings
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,MISSING
BASE=importlib.import_module('13_metric_regions')
REL='02_metapop/02_03_main_analyses/02_03_03_metric_associations/conditional_256_8_16'
OUT=FINAL/'results/00_thesis_figures/16_low_cover_metric_maps_20261104'
INPUTS={}

def checked(p,expected=None):
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if expected is not None:assert digest==expected,p
    INPUTS[str(p.relative_to(FINAL))]=digest
    return p

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for s in ['figures','tables','logs']:(OUT/s).mkdir(exist_ok=True)
    data=FINAL/'data/processed'/REL;source=FINAL/'results'/REL
    audit=json.loads(checked(source/'logs/12_final_audit.json').read_text());prep=json.loads(checked(source/'logs/12_preparation.json').read_text())
    p=data/'12_prepared_arrays.npz';checked(p,prep['output_sha256'][str(p.relative_to(FINAL))])
    with np.load(p,allow_pickle=False) as f:z={k:f[k] for k in ['H','rho','r','half_distance','parameter_ids']}
    p=data/'12_parameter_surfaces.npz';checked(p,audit['output_sha256'][str(p.relative_to(FINAL))]);full=dict(np.load(p,allow_pickle=False))
    c=json.loads(checked(FINAL/'config/00_thesis_figures/15_regions_ABCD_20261104.json').read_text())
    boundary=pd.read_csv(checked(FINAL/c['C_boundary']),float_precision='round_trip')
    members=pd.read_csv(checked(FINAL/'results/00_thesis_figures/15_regions_ABCD_20261104/tables/15_profile_memberships.csv'),float_precision='round_trip')
    assert z['parameter_ids'].tolist()==members.parameter_id.tolist()
    mask=z['H']<=.6+1e-12
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);values=np.nanmedian(z['rho'][mask],axis=0);fullvalues=np.nanmedian(z['rho'],axis=0)
    n=np.isfinite(z['rho'][mask]).sum(axis=0)
    points=np.column_stack([np.log10(z['half_distance'])/np.log10(512),(np.log10(z['r'])+2.5)/8.25]);tri=Delaunay(points)
    gx=np.linspace(0,1,130);gy=np.linspace(0,1,160);xx,yy=np.meshgrid(gx,gy);targets=np.c_[xx.ravel(),yy.ravel()]
    def interpolate(v):
        f=LinearNDInterpolator(tri,v.T);result=f(targets).T.reshape(6,160,130);result[:,~full['domain_mask']]=np.nan
        np.testing.assert_allclose(f(points).T,v,atol=1e-12,rtol=0,equal_nan=True)
        return result
    np.testing.assert_allclose(interpolate(fullvalues),full['median_signed'],atol=1e-12,rtol=0,equal_nan=True)
    fields=interpolate(values)
    assert np.nanmin(fields)>=-1 and np.nanmax(fields)<=1
    rows=[]
    for m,metric in enumerate(BASE.METRICS):
        for i,pid in enumerate(z['parameter_ids']):rows.append(dict(parameter_id=pid,half_distance=float(z['half_distance'][i]),r=float(z['r'][i]),metric=metric,median_signed_spearman=values[m,i],n_defined_H=int(n[m,i]),median_signed_full_H=fullvalues[m,i]))
    table=pd.DataFrame(rows);table.to_csv(OUT/'tables/16_Hle060_profile_summaries.csv',index=False)
    # Independent medians from the authoritative CSV, not the prepared array.
    p=source/'tables/12_within_H_associations.csv';raw=pd.read_csv(checked(p,audit['output_sha256'][str(p.relative_to(FINAL))]),float_precision='round_trip')
    q=raw[raw.H<=.6+1e-12].groupby(['parameter_id','metric']).spearman.agg(['median','count'])
    actual=table.set_index(['parameter_id','metric']).sort_index();q=q.reindex(actual.index)
    np.testing.assert_allclose(actual.median_signed_spearman,q['median'],atol=1e-15,rtol=0,equal_nan=True);np.testing.assert_array_equal(actual.n_defined_H,q['count'])
    np.savez_compressed(OUT/'16_Hle060_parameter_surfaces.npz',grid_half_distance=full['grid_half_distance'],grid_r=full['grid_r'],domain_mask=full['domain_mask'],median_signed=fields,H=z['H'][mask])
    ell=full['grid_half_distance'];rr=full['grid_r'];cut=boundary.proposed_C_lower_r.to_numpy();np.testing.assert_array_equal(ell,boundary.half_distance)
    apply_final_style()
    combined=OUT/'Metric_Maps_5_11_to_5_13_Hle060.pdf'
    with PdfPages(combined) as pdf:
        for indices,name,regions in [([0,1,2],'16_structure_parameter_maps_Hle060',False),([3,4,5],'16_connectivity_parameter_maps_Hle060',False),([3,1,2],'16_regions_Hle060',True)]:
            fig,axes=plt.subplots(1,3,figsize=figure_size(95),sharex=True,sharey=True);fig.subplots_adjust(left=.13,right=.86,top=.88,bottom=.25,wspace=.22)
            for letter,(ax,m) in enumerate(zip(axes,indices)):
                im=ax.pcolormesh(ell,rr,np.ma.masked_invalid(fields[m]),cmap='RdBu_r',vmin=-1,vmax=1,shading='nearest',rasterized=True)
                BASE.map_axes(ax);ax.text(0,1.06,f'({chr(97+letter)}) '+BASE.LABELS[m],transform=ax.transAxes)
                if regions:
                    ax.scatter(members.half_distance,members.r,s=4,c='#222222',alpha=.35,zorder=5)
                    for reg in ['A','B']:
                        rule=c['regions'][reg];f=full['median_signed'][BASE.METRICS.index(rule['metric'])];inside=BASE.region_mask(f,ell[None,:],rule)
                        ax.contour(ell,rr,np.where(np.isfinite(f),inside.astype(float),np.nan),levels=[.5],colors='#222222',linewidths=1,linestyles='--',zorder=6)
                    ci=np.argmin(abs(rr[:,None]-cut[None,:]),axis=0);supported=np.isfinite(full['median_signed'][1])[ci,np.arange(len(ell))]
                    ax.plot(ell,np.where(supported,cut,np.nan),color='#222222',lw=1,ls='--',zorder=7)
                    nn=np.where((ell[None,:]>16)&(rr[:,None]<cut[None,:]),full['median_signed'][3],np.nan)
                    ax.contour(ell,rr,nn,levels=[-.3],colors='#222222',linewidths=1,linestyles='--',zorder=7)
                    for reg,(x,y) in dict(A=(3.6,.36),B=(21,550),C=(13,85000),D=(400,80)).items():ax.text(x,y,reg,ha='center',va='center',weight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2),zorder=8)
            cb=fig.add_axes([.90,.28,.018,.55]);fig.colorbar(im,cax=cb,ticks=[-1,0,1],label=r'Median signed $r_s$')
            fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)');fig.supylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
            handles=[Line2D([],[],ls='none',label=r'Median over defined forest proportions $H\leq0.6$'),Patch(facecolor=MISSING,label='Outside sampled support or undefined')]
            if regions:handles.append(Line2D([],[],color='#222222',ls='--',label='Original A/B/C/D boundaries retained; dots: sampled profiles'))
            fig.legend(handles=handles,loc='lower center',ncol=1,frameon=False)
            stem=OUT/'figures'/name;export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white');pdf.savefig(fig,bbox_inches='tight',facecolor='white');plt.close(fig)
    checked(Path(__file__))
    (OUT/'logs/16_numeric_audit.json').write_text(json.dumps(dict(status='passed_visual_pending',input_sha256=INPUTS,H_max=.6,H_values=z['H'][mask].tolist(),n_selected_H_levels=int(mask.sum()),profiles=256,metrics=6,profile_summaries_independently_checked=len(table),new_equilibria=0,new_bootstrap_draws=0,boundaries='Original full-H adopted A/B/C/D contours and memberships retained; no reclassification',interpolation='Same normalized-log Delaunay triangulation and exact domain mask as original; full-H fields reproduced before restricting H; undefined summaries remain missing'),indent=2)+'\n')
    (OUT/'README.md').write_text('''# Metric maps restricted to H <= 0.6

Separate comparison exports corresponding to Figures 5.11, 5.12 and 5.13.
At each of the 256 fixed profiles, take the median of defined signed within-H
Spearman coefficients only over stored H <= 0.6. Every defined H level has
equal weight. The original full seven-scenario, 8/16-run coefficients,
scenario weighting, interpolation coordinates, colour scale and domain are
unchanged. Grey also marks unavailable restricted summaries. The region panel
retains the adopted full-H A/B/C/D boundaries and all sample positions for
comparison; no new region classification is implied. No model, bootstrap or
landscape run. No active thesis figures or prose are replaced by these exports.
''')
    print(json.dumps(dict(n_selected_H=int(mask.sum()),defined_profiles_per_metric=np.isfinite(values).sum(axis=1).tolist(),combined_pdf=str(combined)),indent=2))

if __name__=='__main__':main()
