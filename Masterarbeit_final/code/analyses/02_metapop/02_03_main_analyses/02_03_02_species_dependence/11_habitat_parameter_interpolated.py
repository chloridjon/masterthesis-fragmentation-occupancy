"""Linear display interpolation of checked H-by-parameter class means."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib
import json
import sys

FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent))
P=importlib.import_module('11_habitat_parameter_contrast')
B=P.B
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,MISSING

CONFIG=FINAL/f'config/{P.REL}/habitat_parameter_contrast/11_interpolated_display.json'


def draw(h,logedges,fields,parameter,stem):
    apply_final_style()
    fig,axes=plt.subplots(3,2,figsize=figure_size(190),sharex=True,sharey=True)
    fig.subplots_adjust(left=.14,right=.80,top=.96,bottom=.18,hspace=.40,wspace=.18)
    he=np.r_[h[0],(h[:-1]+h[1:])/2,h[-1]]
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad(MISSING)
    norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    for j,ax in enumerate(axes.flat):
        ax.set_facecolor(MISSING)
        ax.pcolormesh(he,10**logedges,np.ma.masked_invalid(fields[j].T),shading='flat',cmap=cmap,norm=norm,rasterized=True)
        ax.set_xlim(1,0);ax.set_xticks([1,.8,.6,.4,.2,0],['1','0.8','0.6','0.4','0.2','0'])
        ax.set_yscale('log');ax.minorticks_off()
        if parameter=='r':ax.set_yticks([.01,1,100,10000],[r'$10^{-2}$',r'$10^0$',r'$10^2$',r'$10^4$'])
        else:ax.set_yticks([1,4,16,64,256],['1','4','16','64','256'])
        ax.set_ylim(10**logedges[0],10**logedges[-1])
        rho=[.9,.99,.999][j//2];family='Edge-Random' if j%2==0 else 'Access-based'
        ax.text(0,1.04,f'({chr(97+j)}) {family}, '+rf'$\rho={rho}$',transform=ax.transAxes,fontsize=9.5)
    cax=fig.add_axes([.86,.23,.025,.66])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-1,-.5,0,.5,1],
        label='Mean profile contrast (scenario '+r'$-$'+' Only-Random)')
    fig.supxlabel(r'Habitat proportion $H$')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)' if parameter=='r' else r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.legend(handles=[Patch(facecolor=MISSING,label='Outside interpolation support')],loc='lower center',frameon=False)
    export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)


def run():
    c=json.loads(CONFIG.read_text());source=FINAL/c['source_results'];out=source/'interpolated';data=FINAL/c['data_directory']
    auditpath=source/'logs/11_habitat_parameter_audit.json';a=json.loads(auditpath.read_text())
    assert a['visual_review_confirmed']
    for name,digest in {**a['input_sha256'],**a['output_sha256']}.items():assert B.sha(FINAL/name)==digest,name
    marker=out/'logs/11_interpolation_audit.json'
    if marker.exists():assert not json.loads(marker.read_text())['visual_review_confirmed']
    for p in [out/'figures',out/'logs',out/'review',data]:p.mkdir(parents=True,exist_ok=True)
    source_npz=FINAL/c['source_arrays'];z=np.load(source_npz);h=z['H']
    hdense=np.linspace(h[0],h[-1],c['n_display_H']);arrays={'H_display':hdense,'H_source':h};stems=[];checks={}
    rng=np.random.default_rng(c['verification_seed'])
    for parameter in ['r','half_distance']:
        edges=z[parameter+'_log_bin_edges'];centers=(edges[:-1]+edges[1:])/2
        values=z[parameter+'_mean_contrast'];assert values.shape==(6,85,16) and np.isfinite(values).all()
        # Query at log-cell centres, with edges exactly at the support boundary.
        # Separate masked outer strips preserve the original full axis range.
        fineedges=np.unique(np.r_[np.linspace(edges[0],edges[-1],c['n_display_parameter']+1),centers[0],centers[-1]])
        ydense=(fineedges[:-1]+fineedges[1:])/2
        hh,yy=np.meshgrid(hdense,ydense,indexing='ij');targets=np.column_stack([hh.ravel(),yy.ravel()])
        interpolator=RegularGridInterpolator((h,centers),values.transpose(1,2,0),method='linear',bounds_error=False,fill_value=np.nan)
        fields=interpolator(targets).reshape(len(hdense),len(ydense),6).transpose(2,0,1)
        sh,sy=np.meshgrid(h,centers,indexing='ij');recovered=interpolator(np.column_stack([sh.ravel(),sy.ravel()])).reshape(85,16,6).transpose(2,0,1)
        recovery=float(abs(recovered-values).max());assert recovery<1e-14
        # Independent four-corner bilinear calculations at interior locations.
        error=0.
        for _ in range(100):
            i=int(rng.integers(0,len(h)-1));k=int(rng.integers(0,len(centers)-1));u,v=rng.random(2)
            point=[h[i]+u*(h[i+1]-h[i]),centers[k]+v*(centers[k+1]-centers[k])]
            scalar=(1-u)*(1-v)*values[:,i,k]+u*(1-v)*values[:,i+1,k]+(1-u)*v*values[:,i,k+1]+u*v*values[:,i+1,k+1]
            error=max(error,float(abs(interpolator([point])[0]-scalar).max()))
        assert error<1e-14
        outside=(targets[:,1]<centers[0])|(targets[:,1]>centers[-1])
        assert np.isnan(interpolator(targets[outside])).all()
        assert np.isnan(interpolator([[h[0]-.001,centers[3]],[h[-1]+.001,centers[3]]])).all()
        for j in range(6):
            assert np.nanmin(fields[j])>=values[j].min()-1e-14 and np.nanmax(fields[j])<=values[j].max()+1e-14
        checks[parameter]=dict(maximum_node_recovery_error=recovery,maximum_independent_bilinear_error=error,no_extrapolation=True,no_new_extrema=True,
            minimum_parameter_support=float(10**centers[0]),maximum_parameter_support=float(10**centers[-1]))
        arrays.update({parameter+'_log_source_centers':centers,parameter+'_source_values':values,parameter+'_log_display_edges':fineedges,parameter+'_interpolated_contrast':fields})
        stem=out/f'figures/11_contrast_H_by_{parameter}_interpolated';draw(hdense,fineedges,fields,parameter,stem);stems.append(stem)
    np.savez_compressed(data/'11_interpolated_display_arrays.npz',**arrays)
    fonts=B.export_checks(stems)
    note='''# Interpolated habitat-by-parameter contrast maps

These figures display the unchanged conditional means from the checked16-class habitat-by-parameter maps. They use bilinear interpolation: linear in the actual habitat proportion H and linear in log10(r) or log10(dispersal half-distance). Each original parameter-class mean is anchored at its logarithmic class midpoint. The85 H values and16 class means are unchanged; interpolation recovers every original node and cannot exceed the local corner range. No higher-order spline, model fit, new simulation, seed resampling or new confidence interval was used.

Display interpolation is limited to H=.005-.925 and to the first through last logarithmic class midpoints. The outer half-class strips are grey because interpolating beyond the midpoint range would require extrapolation. Both axes retain their previous full limits for comparison. Grey also marks the unavailable H region, including the sparse baseline at H=.975. The denser display grid adds no independent information or sampling resolution. Original un-interpolated figures and their audits remain preserved.

## Figure captions

**Ratio r.** Interpolated pointwise occupancy contrast against Only-Random across habitat proportion and extinction-to-colonization ratio. Panels compare Edge-Random (left) and Access-based (right), with rho=.9, .99 and .999 from top to bottom. Underlying nodes are arithmetic means across profiles within16 logarithmic r classes of their scenario-minus-Only-Random median occupancy, based on16 and8 landscape seeds, respectively. Dispersal half-distance varies within each class under the adopted joint sampling design. Bilinear display interpolation is linear in H and log10(r) between the85 observed H levels and16 class midpoints. Red indicates positive and blue negative contrasts on the unchanged-1 to+1 occupancy-unit colour scale. Grey indicates locations outside interpolation support. No inferential interval is shown. Averaging remains conditional on the nonrectangular parameter domain and can conceal profile differences.

**Dispersal half-distance.** As above, using logarithmic half-distance classes, with r varying among profiles in each class. Interpolation is linear in H and log10(half-distance), restricted to the first and last class midpoints. All other statistical definitions, scenario panels and colour limits are unchanged.

## Reproduction

`.venv/bin/python -m src.tools.codex_run final-habitat-parameter-interpolated`

After actual PDF rendering and viewing, use `--confirm-visual-review`.
'''
    (out/'11_interpolation_methods_and_captions.md').write_text(note)
    inputs=[CONFIG,Path(__file__),Path(P.__file__),Path(B.__file__),source_npz,auditpath,FINAL/'code/base/src/tools/final_plot_style.py']
    outputs=[p for folder in [data,out/'figures'] for p in folder.rglob('*') if p.is_file()]+[out/'11_interpolation_methods_and_captions.md']
    B.write_json(marker,dict(status='interpolation_checked_visual_review_pending',visual_review_confirmed=False,created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
        retained_source_statistics_sha256=B.sha(source_npz),export_checks=fonts,input_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in inputs},output_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in outputs}))
    print(json.dumps(checks))


def confirm():
    c=json.loads(CONFIG.read_text());out=FINAL/c['source_results']/'interpolated';p=out/'logs/11_interpolation_audit.json';a=json.loads(p.read_text())
    for name,digest in {**a['input_sha256'],**a['output_sha256']}.items():assert B.sha(FINAL/name)==digest,name
    review=out/'review/11_actual_visual_review.json';assert json.loads(review.read_text())['both_pdfs_rendered_and_viewed']
    a.update(status='interpolation_and_visual_review_checked',visual_review_confirmed=True,visual_review_receipt_sha256=B.sha(review));B.write_json(p,a)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--confirm-visual-review',action='store_true')
    confirm() if parser.parse_args().confirm_visual_review else run()
