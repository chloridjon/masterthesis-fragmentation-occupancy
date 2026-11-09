"""Separate visual proposal: extend C leftward and add D right of B.

Reuses immutable colour fields and A/B contours. No new profile assignment,
model run, statistical summary, or manuscript edit.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-region-proposal-mpl')
import sys
import json
import hashlib
import importlib
import argparse
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
OUT=FINAL/'results/00_thesis_figures/14_region_proposal_20261104'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--focused-d',action='store_true',help='Separate revision: D follows Mean NN <= -0.3 at long dispersal.')
    parser.add_argument('--d-threshold',type=float,default=-.3)
    args=parser.parse_args()
    out=OUT/('revision_03_broader_D' if args.d_threshold==-.2 else 'revision_02_focused_D') if args.focused_d else OUT
    source=FINAL/'data/processed/02_metapop/02_03_main_analyses/02_03_03_metric_associations/conditional_256_8_16/12_parameter_surfaces.npz'
    original=FINAL/'results/00_thesis_figures/13_metric_regions_20261102'
    cp=FINAL/'config/00_thesis_figures/13_metric_regions_20261102.json'
    c=json.loads(cp.read_text());mp=original/'tables/13_profile_memberships.csv'
    members=pd.read_csv(mp);z=dict(np.load(source,allow_pickle=False))
    audit=json.loads((FINAL/'results/02_metapop/02_03_main_analyses/02_03_03_metric_associations/conditional_256_8_16/logs/12_final_audit.json').read_text())
    assert sha(source)==audit['output_sha256'][str(source.relative_to(FINAL))]
    before={str(p.relative_to(FINAL)):sha(p) for p in [cp,mp,original/'figures/13_provisional_regions.pdf',original/'figures/13_provisional_regions.png']}
    ell=z['grid_half_distance'];rr=z['grid_r'];field=z['median_signed'];valid=np.isfinite(field[1])
    b=field[1]>=.5
    top=np.array([np.flatnonzero(b[:,j])[-1] if b[:,j].any() else -1 for j in range(len(ell))])
    # The left continuation follows the existing upper B contour. On the right,
    # keep the lower boundary of the former C core (chi >= .7). This is a visual
    # proposal only; it is not used to classify sampled profiles.
    cut=np.full(len(ell),np.nan)
    for j,e in enumerate(ell):
        chi=np.flatnonzero(field[2,:,j]>=.7)
        if e>=128 and len(chi):
            i=chi[0];cut[j]=(rr[max(i-1,0)]+rr[i])/2
        elif top[j]>=0:
            i=top[j];cut[j]=(rr[i]+rr[min(i+1,len(rr)-1)])/2
    good=np.isfinite(cut)
    cut=np.exp(np.interp(np.log(ell),np.log(ell[good]),np.log(cut[good])))
    # D is shown to the right of the existing rightmost B contour at each r,
    # below C. Its left separator therefore reuses the unchanged B contour.
    right_b=np.array([ell[np.flatnonzero(row)[-1]] if row.any() else np.nan for row in b])
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'half_distance':ell,'proposed_C_lower_r':cut}).to_csv(out/'14_visual_C_boundary.csv',index=False)
    pd.DataFrame({'r':rr,'existing_B_rightmost_half_distance':right_b}).to_csv(out/'14_visual_D_reference.csv',index=False)
    apply_final_style()
    fig,axes=plt.subplots(1,3,figsize=figure_size(95),sharex=True,sharey=True)
    fig.subplots_adjust(left=.13,right=.86,top=.88,bottom=.25,wspace=.22)
    coords={'A':(3.6,.36),'B':(21,550),'C':(13,85000),'D':(350,180)}
    if args.focused_d: coords['D']=(400,80)
    for k,(ax,m) in enumerate(zip(axes,[3,1,2])):
        im=ax.pcolormesh(ell,rr,np.ma.masked_invalid(field[m]),cmap='RdBu_r',vmin=-1,vmax=1,shading='nearest',rasterized=True)
        BASE.map_axes(ax)
        ax.text(0,1.06,f'({chr(97+k)}) '+BASE.LABELS[m],transform=ax.transAxes)
        ax.scatter(members.half_distance,members.r,s=4,c='#222222',alpha=.35,zorder=5)
        for reg in ['A','B']:
            rule=c['regions'][reg];f=field[BASE.METRICS.index(rule['metric'])]
            mask=BASE.region_mask(f,ell[None,:],rule)
            ax.contour(ell,rr,np.where(np.isfinite(f),mask.astype(float),np.nan),levels=[.5],colors='#222222',linewidths=1,linestyles='--',zorder=6)
        # Draw the entire proposed lower C boundary, clipped to finite support.
        ci=np.argmin(abs(rr[:,None]-cut[None,:]),axis=0)
        supported=valid[ci,np.arange(len(ell))]
        ax.plot(ell,np.where(supported,cut,np.nan),color='#222222',lw=1,ls='--',zorder=7)
        if args.focused_d:
            # A separate, narrower D boundary; leave the intervening area
            # unassigned rather than moving the existing B boundary.
            nn=np.where((ell[None,:]>16)&(rr[:,None]<cut[None,:]),field[3],np.nan)
            ax.contour(ell,rr,nn,levels=[args.d_threshold],colors='#222222',linewidths=1,linestyles='--',zorder=7)
        for reg,(x,y) in coords.items():
            ax.text(x,y,reg,ha='center',va='center',weight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2),zorder=8)
    cb=fig.add_axes([.90,.28,.018,.55]);fig.colorbar(im,cax=cb,ticks=[-1,0,1],label=r'Median signed $r_s$')
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
    handles=[Patch(facecolor=MISSING,label='Outside sampled support or undefined'),Line2D([],[],color='#222222',ls='--',label='Proposed regions A/B/C/D; dots: sampled profiles')]
    fig.legend(handles=handles,loc='lower center',ncol=1,frameon=False)
    stem=out/'14_proposed_regions_ABCD'
    export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
    assert all(sha(FINAL/p)==v for p,v in before.items())
    receipt=dict(status='visual_proposal_only',input_sha256={str(source.relative_to(FINAL)):sha(source),str(mp.relative_to(FINAL)):sha(mp),str(cp.relative_to(FINAL)):sha(cp)},protected_files_sha256=before,script_sha256=sha(Path(__file__)),changes=['C extended over the full dispersal range above B; right-hand C core retained','D indicated on the large-dispersal side of B below C'],unchanged=['all colour fields and masks','all 256 sampled positions','A and B contour rules','manuscript and all existing figures'],scope='No new profile membership, counts, H curves or biological interpretation. C continuation and D are a visual suggestion pending author review.')
    if args.focused_d:
        receipt['changes']=['A/B/C identical to previous visual proposal',f'D lower-right area guided by Mean NN median signed coefficient <= {args.d_threshold}; intervening area remains unassigned']
        receipt['proposed_D_threshold']=args.d_threshold
        receipt['scope']='Visual boundary suggestion only; the correlation level is a provisional guide, not an inferred ecological threshold. No new memberships or summaries.'
        assert sha(out/'14_visual_C_boundary.csv')==sha(OUT/'14_visual_C_boundary.csv')
    (out/'14_proposal_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'README.md').write_text('''# Separate A/B/C/D visual proposal

Author request, 4 November 2026: extend region C leftward across the dispersal
range and introduce region D to the right of B at large dispersal. The source
maps, colour scale, support, sample positions and A/B contour rules are reused
unchanged. The lower boundary of C follows the upper B contour on the left
and the previous chi >= .7 core boundary on the right. D is labelled in the
right-hand area below C, with the existing right B contour as its left guide.
This is an approximate visual proposal, not a new quantitative classification.
No memberships, region counts, H summaries, thesis prose or prior figure change.
''')
    if args.focused_d:
        (out/'README.md').write_text('''# Revision 2: narrower lower-right D

Only D changes relative to the previous visual proposal. Its new guide is
the median signed Mean NN association contour at -0.3, restricted to long
dispersal below C. The label moves from (350,180) to (400,80). The region
between the original B edge and the proposed D edge remains unassigned.
The -0.3 level is a provisional visual choice for author review, not a
biological threshold. A/B/C, source fields, support and sample positions are
unchanged. No memberships, summaries, model results or manuscript are changed.
''')
        if args.d_threshold==-.2:
            (out/'README.md').write_text('''# Revision 3: moderately enlarged D

Author requests D slightly larger than revision 2. Use the Mean NN median
signed coefficient contour at -0.2 instead of -0.3 as a provisional visual
guide. D remains in the lower-right area, distinct from B. All A/B/C lines,
fields, support, sample positions and label positions are unchanged relative
to revision 2. No quantitative memberships or manuscript changes.
''')
    print(stem.with_suffix('.png'))

if __name__=='__main__':main()
