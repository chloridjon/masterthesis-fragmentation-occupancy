"""Export the fixed conditional design without using unfinished responses."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-confirmed-design-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from src.tools.final_plot_style import apply_final_style, figure_size, export_figure
RESULTS=FINAL/'results/02_metapop/02_02_species_parameter_selection/confirmed_response_sampling'
DATA=FINAL/'data/processed/02_metapop/02_02_species_parameter_selection/confirmed_response_sampling'
SAMPLE_BLUE = '#0072B2'
DOMAIN_BLUE = '#CCE3F0'
EXCLUDED_GREY = '#F2F2F2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(confirm=False):
    audit=RESULTS/'logs/09_sampling_figure_audit.json'
    if confirm:
        a=json.loads(audit.read_text())
        for n,h in {**a['input_sha256'],**a['output_sha256']}.items():assert sha(FINAL/n)==h
        a.update(status='design_figure_visually_checked',visual_review_confirmed=True,
            visual_review_at_utc=datetime.now(timezone.utc).isoformat())
        audit.write_text(json.dumps(a,indent=2)+'\n');print(a['status']);return
    preflight=RESULTS/'logs/09_design_preflight.json';a=json.loads(preflight.read_text())
    assert a['status']=='design_verified'
    for n,h in a['output_sha256'].items():assert sha(FINAL/n)==h
    design=pd.read_csv(DATA/'09_confirmed_response_parameters.csv',float_precision='round_trip')
    strips=pd.read_csv(DATA/'09_confirmed_log_area_strips.csv',float_precision='round_trip')
    assert len(design)==256 and (design.retention_L_upper<.95).all() and (design.retention_U_lower>.05).all()
    apply_final_style();fig,ax=plt.subplots(figsize=figure_size(115))
    fig.subplots_adjust(left=.15,right=.98,top=.98,bottom=.22)
    ax.set_facecolor(EXCLUDED_GREY)
    for row in strips.itertuples(index=False):
        xl,xh,yl,yh=np.exp([row.log_ell_low,row.log_ell_high,row.log_r_low,row.log_r_high])
        ax.add_patch(Rectangle((xl,yl),xh-xl,yh-yl,facecolor=DOMAIN_BLUE,edgecolor='none'))
    ax.scatter(design.half_distance,design.r,s=12,color=SAMPLE_BLUE,edgecolors='white',linewidths=.25,zorder=3)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1,512);ax.set_ylim(10**-2.5,10**5.75)
    ax.set_xticks([1,4,16,64,256,512],['1','4','16','64','256','512']);ax.minorticks_off()
    ax.set_xlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    ax.set_ylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    fig.legend(handles=[Patch(facecolor=DOMAIN_BLUE,label='Certified inner sampling region'),
        Line2D([],[],color=SAMPLE_BLUE,ls='none',marker='o',ms=3,label='256 profiles: 64 dispersal groups × 4 ratios'),
        Patch(facecolor=EXCLUDED_GREY,label='Outside the adopted inner region')],
        loc='lower center',ncol=1,frameon=False)
    stem=RESULTS/'figures/09_conditional_parameter_design';stem.parent.mkdir(exist_ok=True)
    export_figure(fig,stem);fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    fonts=subprocess.run(['pdffonts',str(stem.with_suffix('.pdf'))],check=True,capture_output=True,text=True).stdout
    assert 'LMRoman8' in fonts and 'LatinModernMath' in fonts
    svg=stem.with_suffix('.svg').read_text();assert '<text' in svg and '@font-face' in svg
    caption=RESULTS/'09_sampling_figure_caption.md'
    caption.write_text('''# Conditional parameter design

The 256 points were sampled from the light-blue certified inner region within
half-distances of 1–512 cells and log10(r) of −2.5–5.75. Grey areas fall outside
this adopted interior; they are not all classified as saturated or extinct.
The region uses conservative monotonicity certificates from the 128-point,
20-landscape-seed pilot and 83 boundary queries. No surface interpolation is
used. The expanded diagnostic window up to 4096 cells and down to r=10^-44
is shown separately and was not adopted as the main sampling distribution.

One randomly jittered dispersal position was placed in each of 64 strata
with equal probability under logarithmic area; four stratified conditional
log-r values were drawn per position. All points receive weight 1/256.
Sharing a dispersal coordinate enables kernel reuse and does not change
the target measure. The adopted region covers 69.29% of the original log
rectangle, increasing average point density by a factor of 2.89 relative
to the original 128-point design. The model remains dimensionless and
half-distance is in raster-cell coordinates. This is a sampling-design
figure and contains no results from the unfinished new response calculation.
''')
    inputs={str(preflight.relative_to(FINAL)):sha(preflight),**a['output_sha256']}
    outputs={str(p.relative_to(FINAL)):sha(p) for p in [caption,*[stem.with_suffix(s) for s in ['.pdf','.svg','.png']]]}
    audit.write_text(json.dumps(dict(status='design_export_checked_visual_review_pending',
        input_sha256=inputs,output_sha256=outputs,font_check=fonts,plot_source_sha256=sha(Path(__file__)),
        visual_review_confirmed=False),indent=2)+'\n')
    print('Design figure exported; visual review pending.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--confirm-visual-review',action='store_true')
    main(p.parse_args().confirm_visual_review)
