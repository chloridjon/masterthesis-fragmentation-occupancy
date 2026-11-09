"""Revise Chapter3 Figures3.2/3.4: irregular forest clusters without IDs.

3.2 is a constructed raster with independently checked83-cell centre distance
and4-cell boundary-cell distance. 3.4 retains the complete saved landscape,
Voronoi regions and all15 connection segments; only labels change.
"""
from pathlib import Path
import os,sys,json,hashlib,importlib.util
os.environ.setdefault('MPLCONFIGDIR','/tmp/chapter3-cluster-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Patch,FancyArrowPatch,Rectangle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import label,binary_erosion
from scipy.spatial.distance import cdist
ROOT=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(ROOT/'code/base'))
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,LANDSCAPE_COLORS,CLUSTER_PALETTE_RGB
OUT=ROOT/'results/00_thesis_figures/23_chapter3_cluster_illustrations_20261107'
FIG=OUT/'figures';TABLE=OUT/'tables'
for p in [FIG,TABLE]:p.mkdir(parents=True,exist_ok=True)
source=ROOT/'results/00_thesis_figures/21_voronoi_hybrid_schematic/tables/landscape.npz'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
helper=ROOT/'code/analyses/00_thesis_figures/21_voronoi_hybrid_schematic.py'
spec=importlib.util.spec_from_file_location('hybrid23',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
GREEN=LANDSCAPE_COLORS['habitat'];GREY=LANDSCAPE_COLORS['non_habitat']
BLUE,ORANGE=[matplotlib.colors.to_hex(np.asarray(c)/255) for c in CLUSTER_PALETTE_RGB[:2]]
AUDIT={}
def save(fig,stem):
    export_figure(fig,FIG/stem);fig.savefig(FIG/f'{stem}.png',dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)

def distance_figure():
    yy,xx=np.indices((200,200));left=np.zeros((200,200),dtype=bool)
    ellipses=[(49.5,100,28,12),(24.5,95,15,9),(74.5,105,15,9),(45.5,86,9,6),(53.5,114,9,6)]
    for x,y,rx,ry in ellipses:left|=((xx-x)/rx)**2+((yy-y)/ry)**2<=1
    right=np.zeros_like(left)
    for y,x in np.argwhere(left):right[200-y,x+83]=True
    assert not np.any(left&right)
    masks=[left,right];grid=left|right;assert label(grid)[1]==2
    centres=np.array([np.argwhere(m)[:,::-1].mean(axis=0) for m in masks]);centre_distance=float(np.linalg.norm(centres[1]-centres[0]))
    boundaries=[np.argwhere(m&~binary_erosion(m))[:,::-1] for m in masks];distance=cdist(*boundaries);i,j=np.unravel_index(distance.argmin(),distance.shape);edge_points=np.array([boundaries[0][i],boundaries[1][j]]);edge_distance=float(distance[i,j])
    np.testing.assert_allclose(centre_distance,83);np.testing.assert_allclose(edge_distance,4)
    fig,ax=plt.subplots(figsize=figure_size(90));fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98)
    ax.set(xlim=(-.5,183.5),ylim=(155.5,45.5),aspect='equal');ax.set_facecolor(GREY);ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
    for mask in masks:
        for vertices in mod.cell_polygons(mask):ax.add_patch(Polygon(vertices,facecolor=GREEN,edgecolor='none',zorder=2))
    for x,y in centres:ax.plot([x,x],[70,y],color=ORANGE,lw=.8,zorder=3);ax.scatter(x,y,s=36,facecolor='white',edgecolor=ORANGE,linewidth=1.5,zorder=5)
    for x,y in edge_points:ax.plot([x,x],[y,134],color=BLUE,lw=.8,zorder=3);ax.scatter(x,y,s=16,facecolor='white',edgecolor=BLUE,linewidth=1,zorder=5)
    for xs,y,colour in [(centres[:,0],70,ORANGE),(edge_points[:,0],134,BLUE)]:ax.add_patch(FancyArrowPatch((xs[0],y),(xs[1],y),arrowstyle='|-|',mutation_scale=8,shrinkA=0,shrinkB=0,lw=1.6,color=colour,zorder=3))
    middle=centres[:,0].mean();ax.text(middle,52,'Center-to-center distance',ha='center',va='center');ax.text(middle,61,'83 cells between cluster centres',ha='center',va='center',fontweight='bold')
    ax.text(middle,143,'Edge-to-edge distance',ha='center',va='center');ax.text(middle,151,'4 cells between cluster edges',ha='center',va='center',fontweight='bold')
    save(fig,'3_02_distance_representation');np.savez_compressed(TABLE/'distance_geometry.npz',grid=grid,masks=np.array(masks),centres_xy=centres,edge_points_xy=edge_points)
    AUDIT['3.2']={'constructed_example':True,'n_forest_clusters':2,'grid_shape':[200,200],'geometry':'unions of overlapping ellipses with exact stepped cell-side outlines','cluster_ids_displayed':False,'centre_distance_cells':centre_distance,'edge_distance_cells':edge_distance,'distances_preserved_from_prior_example':True}

def hybrid_figure():
    z=np.load(source);grid=z['grid'];labels=z['labels'];ids=z['patch_ids'];centres=z['centers_xy'];regions=z['regions'];edge_segments=z['edge_to_edge_segments'];centre_segments=z['center_to_center_segments']
    assert len(ids)==6 and len(edge_segments)==9 and len(centre_segments)==6
    fig=plt.figure(figsize=figure_size(125));ax=fig.add_axes([2/160,14.5/125,156/160,(156*92/132)/125]);ax.set(aspect='equal',xlim=(-.5,grid.shape[1]-.5),ylim=(grid.shape[0]-.5,-.5));ax.set_facecolor(GREY);ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
    y,x=np.nonzero(regions[:,:-1]!=regions[:,1:]);vertical=np.stack((np.column_stack((x+.5,y-.5)),np.column_stack((x+.5,y+.5))),axis=1)
    y,x=np.nonzero(regions[:-1]!=regions[1:]);horizontal=np.stack((np.column_stack((x-.5,y+.5)),np.column_stack((x+.5,y+.5))),axis=1)
    ax.add_collection(LineCollection(np.concatenate((vertical,horizontal)),colors='#B5B5B5',linewidths=.45,zorder=1))
    for identifier in ids:
        for vertices in mod.cell_polygons(labels==identifier):ax.add_patch(Polygon(vertices,facecolor=GREEN,edgecolor='none',zorder=2))
    ax.add_collection(LineCollection(centre_segments,colors=ORANGE,linewidths=1.25,zorder=3));ax.add_collection(LineCollection(edge_segments,colors=BLUE,linewidths=1.45,zorder=4));endpoints=edge_segments.reshape(-1,2)
    ax.scatter(endpoints[:,0],endpoints[:,1],s=10,c=BLUE,edgecolors='white',linewidths=.35,zorder=5);ax.scatter(centres[:,0],centres[:,1],s=19,c=ORANGE,edgecolors='white',linewidths=.6,zorder=6)
    fig.legend(handles=[Line2D([],[],color=BLUE,linewidth=1.45,marker='o',markersize=3,label='Voronoi neighbours: edge-to-edge'),Patch(facecolor=GREEN,edgecolor='none',label='Forest cluster'),Line2D([],[],color=ORANGE,linewidth=1.25,marker='o',markersize=3.8,label='All other pairs: center-to-center'),Line2D([],[],color='#B5B5B5',linewidth=.65,label='Voronoi boundary')],loc='upper center',bbox_to_anchor=(.5,12.5/125),frameon=False,ncols=2,handlelength=2.3,labelspacing=.45)
    save(fig,'3_04_voronoi_hybrid');assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    AUDIT['3.4']={'source':str(source.relative_to(ROOT)),'source_sha256':source_hash,'source_unchanged':True,'all_geometry_centres_regions_and_15_segments_preserved':True,'cluster_ids_displayed':False,'legend':'Forest cluster'}

def remaining_terminology():
    import pandas as pd
    path=ROOT/'results/00_thesis_figures/03_method_decisions/tables/access_effect_share_summary.csv'
    summary=pd.read_csv(path);assert len(summary)==6
    fig,axes=plt.subplots(1,2,figsize=figure_size(68),sharex=True,sharey=True)
    fig.subplots_adjust(left=.125,right=.985,bottom=.25,top=.87,wspace=.17)
    styles=[('eta','#0072B2',r'$\eta$'),('k','#E69F00',r'$k$'),('lambda_','#009E73',r'$\lambda$'),('interactions','#B9B9B9','Interactions')]
    for ax,metric,name in zip(axes,['edge_density_total','largest_cluster_fraction_forest'],['(a) Edge density ED','(b) Largest cluster fraction LCF']):
        part=summary[summary.metric.eq(metric)].reset_index(drop=True);left=np.zeros(3)
        for column,colour,label_text in styles:ax.barh(np.arange(3),part[column],left=left,height=.54,color=colour);left+=part[column]
        for i,row in part.iterrows():
            if row.n_sensitive_H:ax.scatter(1.018,i,marker='*',color='#404040',s=24,clip_on=False)
        ax.set(yticks=[0,1,2],yticklabels=['0.9','0.99','0.999'],xlim=(0,1),ylim=(2.6,-.6));ax.set_xticks([0,.25,.5,.75,1]);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.text(0,1.04,name,transform=ax.transAxes,va='bottom')
    axes[0].set_ylabel(r'Edge-removal probability $\rho$');fig.supxlabel('Mean share of parameter-setting variation')
    handles=[Patch(facecolor=c,label=t) for _,c,t in styles]
    if summary.n_sensitive_H.gt(0).any():handles.append(Line2D([],[],marker='*',color='#404040',linestyle='none',label='Sensitive to run omission'))
    fig.legend(handles=handles,loc='lower center',ncols=3,frameon=False);save(fig,'3_01_access_effects')
    AUDIT['3.1']={'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'saved_summaries_unchanged':True}

    base=ROOT/'data/processed/02_metapop/02_02_species_parameter_selection/confirmed_response_sampling'
    design=pd.read_csv(base/'09_confirmed_response_parameters.csv',float_precision='round_trip');strips=pd.read_csv(base/'09_confirmed_log_area_strips.csv',float_precision='round_trip');assert len(design)==256
    fig,ax=plt.subplots(figsize=figure_size(115));fig.subplots_adjust(left=.15,right=.98,top=.98,bottom=.22);ax.set_facecolor('#F2F2F2')
    for row in strips.itertuples(index=False):
        xl,xh,yl,yh=np.exp([row.log_ell_low,row.log_ell_high,row.log_r_low,row.log_r_high]);ax.add_patch(Rectangle((xl,yl),xh-xl,yh-yl,facecolor='#CCE3F0',edgecolor='none'))
    ax.scatter(design.half_distance,design.r,s=12,color='#0072B2',edgecolors='white',linewidths=.25,zorder=3);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1,512);ax.set_ylim(10**-2.5,10**5.75);ax.set_xticks([1,4,16,64,256,512],['1','4','16','64','256','512']);ax.minorticks_off()
    ax.set_xlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)');ax.set_ylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
    fig.legend(handles=[Patch(facecolor='#CCE3F0',label='Selected parameter region'),Line2D([],[],color='#0072B2',ls='none',marker='o',ms=3,label='256 parameter combinations'),Patch(facecolor='#F2F2F2',label='Excluded region')],loc='lower center',ncols=1,frameon=False);save(fig,'3_11_parameter_design')
    AUDIT['3.11']={'saved_points_and_region_unchanged':True,'n_combinations':256,'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'09_confirmed_response_parameters.csv',base/'09_confirmed_log_area_strips.csv']}}

def main():
    apply_final_style();distance_figure();hybrid_figure();remaining_terminology();(OUT/'audit.json').write_text(json.dumps(AUDIT,indent=2)+'\n');print('Figures3.2/3.4 updated;83/4 distances and15 links preserved. Terminology in3.1/3.11 aligned.')
if __name__=='__main__':main()
