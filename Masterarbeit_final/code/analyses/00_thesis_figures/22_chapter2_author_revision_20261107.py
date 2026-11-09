"""Author-approved Chapter 2 illustration revision; cached results stay unchanged.

Figures2.1--2.6/2.8 reuse existing geometry, grids and numerical outputs.
Figure2.7 is a new deterministic teaching landscape, with stepped cell-side
outlines like Figure3.4. No production trajectories/equilibria are rerun.
"""
from pathlib import Path
import sys,importlib.util,json,hashlib,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Polygon,Rectangle,FancyArrowPatch
from scipy.ndimage import label as component_label
from scipy.spatial.distance import cdist
ROOT=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(ROOT/'code/base'))
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,LANDSCAPE_COLORS,VORONOI_NETWORK_COLORS
OUT=ROOT/'results/00_thesis_figures/22_chapter2_author_revision_20261107'
FIG=OUT/'figures';TABLE=OUT/'tables'
for directory in [FIG,TABLE]:directory.mkdir(parents=True,exist_ok=True)
INPUTS={};AUDIT={}
GREEN=LANDSCAPE_COLORS['habitat'];GREY=LANDSCAPE_COLORS['non_habitat'];BLUE='#0072B2';ORANGE='#E69F00'
METRICS=Path('01_landscape/01_01_analysis_framework/01_01_02_landscape_metrics')
EXAMPLES=Path('01_landscape/01_01_analysis_framework/01_01_01_landscape_examples')
MODEL=Path('02_metapop/02_01_model_framework')
def source(relative):
    p=ROOT/relative;INPUTS[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest();return p

def loadmod(name,relative):
    path=source(relative);spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def relabel(fig):
    from matplotlib.text import Text
    mapping={'(b) Structured patch landscape':'(b) Forest-cluster landscape','Non-habitat outside rings':'Non-forest outside rings','(a) Connected forest patch':'(a) Connected forest cluster'}
    for t in fig.findobj(Text):
        if t.get_text() in mapping:t.set_text(mapping[t.get_text()])

def save(fig,name):
    relabel(fig);export_figure(fig,FIG/name);fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)


def rings_and_configuration():
    rings=loadmod('rings22','code/analyses/01_landscape/01_01_analysis_framework/01_01_01_access_rings_schematic.py')
    rings.OUTPUT_STEM=FIG/'2_01_access_rings'
    rings.export_figure=lambda fig,stem:(relabel(fig),export_figure(fig,stem))
    rings.main()
    config=loadmod('config22','code/analyses/00_thesis_figures/06_supervisor_revision_figures.py')
    config.OUT=OUT;config.save=lambda fig,name:save(fig,'2_04_configuration')
    config.configuration()


def scenario_label(ax,letter,name,rho):
    y=1.025;t=ax.text(0,y,f'({letter}) ',transform=ax.transAxes,va='bottom')
    fig=ax.figure;fig.canvas.draw();x=t.get_window_extent().width/ax.get_window_extent().width
    t=ax.text(x,y,name,transform=ax.transAxes,va='bottom',fontstyle='italic')
    fig.canvas.draw();x+=t.get_window_extent().width/ax.get_window_extent().width+.02
    if rho:ax.text(x,y,fr'($\rho={rho}$)',transform=ax.transAxes,va='bottom')


def landscape_figures():
    for oldstem,name,rows in [('deforestation_landscapes_h030','2_02_forest_removal',2),('edge_access_landscapes_rho099_h030','2_03_edge_access',1)]:
        with np.load(source(Path('data/processed')/EXAMPLES/f'{oldstem}.npz')) as z:grids=z['grids'];ids=z['scenario_ids'].tolist();h=float(z['H'])
        assert np.all(grids.sum(axis=(1,2))==12000) and h==.3
        fig,axes=plt.subplots(rows,2,figsize=figure_size(174 if rows==2 else 90));fig.subplots_adjust(left=.025,right=.975,top=.95 if rows==2 else .91,bottom=.075 if rows==2 else .14,wspace=.12,hspace=.19)
        for i,(ax,grid,identifier) in enumerate(zip(axes.flat,grids,ids)):
            ax.imshow(grid,cmap=ListedColormap([GREY,GREEN]),vmin=0,vmax=1,interpolation='none',resample=False,aspect='equal');ax.set_axis_off()
            name_s='only random' if identifier.startswith('only_random') else ('edge random' if identifier.startswith('edge_random') else 'access based')
            rho=identifier.rsplit('rho',1)[-1] if 'rho' in identifier else ''
            rho={'09':'0.9','099':'0.99','0999':'0.999','0':'0'}.get(rho,rho)
            scenario_label(ax,chr(97+i),name_s,'' if name_s=='only random' else rho)
        fig.legend(handles=[Patch(facecolor=GREEN,label='Forest'),Patch(facecolor=GREY,label='Non-forest')],loc='lower center',bbox_to_anchor=(.5,.01),ncols=2,frameon=False)
        save(fig,name);AUDIT[name]={'saved_grids_unchanged':True,'forest_cover':h,'scenario_names_italic':True}


def voronoi_figure():
    z=np.load(source(Path('data/processed')/METRICS/'access_voronoi_rho099_h030.npz'));grid=z['grid'];labels=z['labels'];regions=z['regions'];colours=z['cluster_colors']
    mod=loadmod('regions22','code/analyses/01_landscape/01_01_analysis_framework/01_01_02_voronoi_regions_illustration.py')
    rgb=np.full((*grid.shape,3),238,dtype=np.uint8);rgb[grid==1]=colours[labels[grid==1]]
    fig=plt.figure(figsize=figure_size(86))
    for left,raster,title in [(.01,rgb,'(a) Forest clusters'),(.55,colours[regions],'(b) Voronoi regions')]:
        ax=fig.add_axes([left,.15,.44,160*.44/86]);ax.imshow(raster,interpolation='none',resample=False,aspect='equal')
        if left>.5:ax.add_collection(LineCollection(mod.boundary_segments(regions),colors='#777777',linewidth=.3,alpha=.85))
        ax.set(xlim=(-.5,199.5),ylim=(199.5,-.5));ax.set_axis_off();ax.text(0,1.025,title,transform=ax.transAxes,va='bottom')
    fig.add_artist(FancyArrowPatch((.475,.56),(.525,.56),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=10,color='#404040',linewidth=1))
    fig.text(.5,.035,'Same colours identify forest clusters and their Voronoi regions',ha='center',fontsize=8.5)
    save(fig,'2_05_voronoi_regions');AUDIT['2_05_voronoi_regions']={'source_geometry_unchanged':True,'forest_cover':float(grid.mean())}


def overlap(a,b):
    width=min(a.x1,b.x1)-max(a.x0,b.x0);height=min(a.y1,b.y1)-max(a.y0,b.y0)
    return max(0,width)*max(0,height)


def mst_figure():
    z=np.load(source(Path('data/processed')/METRICS/'small_patch_voronoi_mst.npz'));centres=z['centroids_xy'];areas=z['patch_areas'];colours=z['colors_rgb'];pairs=z['neighbor_pairs']-1;weights=z['edge_weights'];retained=z['mst_edge_mask'];grid=z['grid']
    meta=json.loads(source(Path('results')/METRICS/'logs/small_patch_voronoi_mst_manifest.json').read_text());d50=meta['complete_mst_d50']
    radii=3.7+(np.sqrt(areas)-np.sqrt(areas.min()))/(np.sqrt(areas.max())-np.sqrt(areas.min()))*1.6
    fig=plt.figure(figsize=figure_size(110));axes=[fig.add_axes([left,.31,.44,160*.44/110]) for left in [.01,.55]]
    segments=centres[pairs];padding=7.;diameter=2*radii*(160*.44/(80+2*padding))*72/25.4
    for ax in axes:ax.set(xlim=(-.5-padding,79.5+padding),ylim=(79.5+padding,-.5-padding),aspect='equal');ax.set_axis_off()
    axes[0].add_collection(LineCollection(segments,colors=VORONOI_NETWORK_COLORS['edge'],linewidths=.9,alpha=.65,zorder=1))
    axes[1].add_collection(LineCollection(segments[~retained],colors=VORONOI_NETWORK_COLORS['non_tree'],linewidths=.75,zorder=1))
    axes[1].add_collection(LineCollection(segments[retained],colors=VORONOI_NETWORK_COLORS['mst'],linewidths=1.5,zorder=2))
    axes[1].add_collection(LineCollection(segments[retained & (weights<=d50+1e-12)],colors='#D55E00',linewidths=2.1,zorder=2.5))
    for ax in axes:
        ax.scatter(centres[:,0],centres[:,1],s=diameter**2,c=colours,edgecolors='white',linewidths=.4,zorder=3)
        for (x,y),area,rgb in zip(centres,areas,colours):
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);colour='#202020' if linear@np.array([.2126,.7152,.0722])>.179 else 'white'
            ax.text(x,y,str(int(area)),ha='center',va='center',color=colour,fontsize=8.5,zorder=4)
    from matplotlib.transforms import Bbox
    fig.canvas.draw();ax=axes[1];renderer=fig.canvas.get_renderer();node_boxes=[]
    for point,d in zip(centres,diameter):
        x,y=ax.transData.transform(point);rad=d/2*fig.dpi/72+2;node_boxes.append(Bbox.from_extents(x-rad,y-rad,x+rad,y+rad))
    used=[];label_positions=[]
    for (i,j),weight in zip(pairs[retained],weights[retained]):
        a,b=centres[[i,j]];v=b-a;normal=np.array([-v[1],v[0]])/np.linalg.norm(v);candidates=[]
        text=ax.text(0,0,f'{weight:.2f}'.rstrip('0').rstrip('.'),ha='center',va='center',fontsize=8.5,zorder=6,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
        for fraction in [.5,.4,.6]:
            for offset in [3.2,-3.2,5.,-5.,7.,-7.]:
                point=a+fraction*v+offset*normal;text.set_position(point);box=text.get_window_extent(renderer).expanded(1.08,1.14)
                score=30*sum(overlap(box,node) for node in node_boxes)+sum(overlap(box,prior) for prior in used)+.01*abs(offset)+.01*abs(fraction-.5)
                if not ax.get_window_extent().contains(box.x0,box.y0) or not ax.get_window_extent().contains(box.x1,box.y1):score+=10000
                candidates.append((score,point.copy(),box))
        _,point,box=min(candidates,key=lambda row:row[0]);text.set_position(point);used.append(box);label_positions.append(point.tolist())
    for ax,text in zip(axes,['(a) Voronoi network','(b) Minimum spanning tree']):ax.text(0,1.035,text,transform=ax.transAxes,va='bottom')
    mid=.31+160*.44/110/2;fig.add_artist(FancyArrowPatch((.475,mid),(.525,mid),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=9,color='#404040'))
    fig.text(.5,.235,fr'$\mathrm{{MNN}}={meta["mean_nearest_neighbor_distance"]:.2f}$ cells; $\mathrm{{MIP}}={meta["mean_pairwise_distance"]:.2f}$ cells; $\mathrm{{D50}}={d50:.2f}$ cells',ha='center',fontsize=8.5)
    fig.text(.5,.175,'Node numbers: forest-cluster area (cells); MST edge numbers: distance (cells)',ha='center',fontsize=8.5)
    fig.text(.5,.12,'Orange links connect at least half the forest area',ha='center',fontsize=8.5)
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=colours[0],markersize=7,label='Forest cluster'),Line2D([],[],color=VORONOI_NETWORK_COLORS['edge'],label='Voronoi link'),Line2D([],[],color=VORONOI_NETWORK_COLORS['mst'],label='MST link'),Line2D([],[],color=VORONOI_NETWORK_COLORS['non_tree'],label='Discarded link')],loc='lower center',bbox_to_anchor=(.5,.025),ncols=4,frameon=False)
    save(fig,'2_06_voronoi_mst');assert retained.sum()==11
    AUDIT['2_06_voronoi_mst']={'all_original_geometry_areas_and_weights_preserved':True,'areas_inside_nodes':areas.tolist(),'mst_distance_labels':weights[retained].tolist(),'mst_label_positions':label_positions,'cluster_ids_removed':True,'MNN':meta['mean_nearest_neighbor_distance'],'MIP':meta['mean_pairwise_distance'],'D50':d50}


def connectivity_figure():
    old=loadmod('concept22','code/analyses/02_metapop/02_01_model_framework/02_01_conceptual_figures.py')
    shapes=loadmod('shape22','code/analyses/00_thesis_figures/21_voronoi_hybrid_schematic.py')
    yy,xx=np.indices((56,80))
    ellipses=[[(18,40,8,5),(22,42,4,4),(13,37,4,3)],[(12,10,4,3),(14,11,2,2)],[(63,41,7,5),(67,43,3,4),(59,38,3,3)],[(64,10,5,3.5),(67,12,3,3),(61,8,2,2)],[(38,25,7,5),(42,26,4,4),(35,21,3,3)]]
    masks=[];polygons=[];centres=[]
    for group in ellipses:
        mask=np.zeros_like(xx,dtype=bool)
        for x,y,rx,ry in group:mask|=((xx-x)/rx)**2+((yy-y)/ry)**2<=1
        assert component_label(mask)[1]==1
        outlines=shapes.cell_polygons(mask);assert len(outlines)==1
        masks.append(mask);polygons.append(outlines[0]);centres.append(np.argwhere(mask)[:,::-1].mean(axis=0))
    grid=np.stack(masks).sum(axis=0);assert grid.max()==1 and component_label(grid)[1]==5
    recipient=np.argwhere(masks[-1])[:,::-1];probabilities=[.9,.9,.2,.6];alpha=.08
    records=[];segments=[]
    for i,(mask,p) in enumerate(zip(masks,probabilities),1):
        points=np.argwhere(mask)[:,::-1];d=cdist(points,recipient);a,b=np.unravel_index(d.argmin(),d.shape);distance=float(d[a,b]);area=int(mask.sum());contribution=np.exp(-alpha*distance)*area*p
        records.append({'source':i,'area':area,'occupancy':p,'distance':distance,'contribution':contribution});segments.append(np.array([points[a],recipient[b]]))
    weights=np.array([row['contribution'] for row in records]);total=float(weights.sum());fig=plt.figure(figsize=figure_size(140));ax=fig.add_axes([.025,.32,.95,.66]);ax.set(xlim=(-3,82),ylim=(-3,57),aspect='equal');ax.set_facecolor('#F4F4F4');ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
    for record,outline,centre,p,segment in zip(records,polygons,centres,probabilities,segments):
        width=5.*record['contribution']/weights.max();ax.add_patch(FancyArrowPatch(*segment,arrowstyle='-|>',mutation_scale=8+width,linewidth=width,color=BLUE,zorder=2))
        fill=old.occupancy_fill(outline,p);np.testing.assert_allclose(old.polygon_area(outline),record['area']);np.testing.assert_allclose(old.polygon_area(fill),p*record['area'])
        ax.add_patch(Polygon(outline,facecolor='white',edgecolor='none',zorder=3));ax.add_patch(Polygon(fill,facecolor=GREEN,edgecolor='none',zorder=4));ax.add_patch(Polygon(outline,facecolor='none',edgecolor=GREEN,linewidth=.8,zorder=5))
        ax.text(centre[0],centre[1]-.6,f'{p:.0%}',ha='center',va='center',color='white' if p>=.5 else GREEN,fontsize=9.5,zorder=6)
        ax.text(centre[0],outline[:,1].max()+1.7,f'Forest cluster {record["source"]}',ha='center',va='bottom',fontsize=8.5,zorder=6)
    focal=polygons[-1];centre=centres[-1];ax.add_patch(Polygon(focal,facecolor='white',edgecolor=BLUE,linewidth=1.8,zorder=3));ax.text(*centre,'Recipient',color=BLUE,ha='center',va='center',fontsize=9.5,zorder=4);ax.text(centre[0],focal[:,1].max()+1.7,r'Forest cluster $i$',ha='center',va='bottom',fontsize=8.5)
    fig.text(.5,.275,r'Green fraction $=p_j$; arrow width $\propto K_{ij} A_j^b p_j$',ha='center')
    fig.text(.5,.22,r'Sum of source contributions $=S_i$',ha='center')
    bar=fig.add_axes([.10,.135,.80,.045]);bar.set(xlim=(0,total),ylim=(0,1));bar.set_axis_off();offset=0
    for record,weight in zip(records,weights):
        bar.add_patch(Rectangle((offset,0),weight,1,facecolor=BLUE,edgecolor='white',linewidth=.8));bar.text(offset+weight/2,.5,str(record['source']),ha='center',va='center',color='white',fontsize=8.5);offset+=weight
    fig.text(.5,.07,fr'Constructed forest landscape; forest cover $H={grid.mean():.3f}$',ha='center',fontsize=8.5)
    save(fig,'2_07_colonisation_contributions');np.savez_compressed(TABLE/'2_07_landscape.npz',grid=grid,masks=np.array(masks),source_probabilities=probabilities)
    with (TABLE/'2_07_contributions.csv').open('w') as f:writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    AUDIT['2_07_colonisation_contributions']={'constructed_example':True,'simulation_replicate':False,'n_clusters':5,'forest_cover':float(grid.mean()),'source_area_exponent':1,'kernel_alpha':alpha,'distance':'exact boundary-cell-centre distance','source_records':records,'total_connectivity':total,'green_fraction_matches_probability':True,'arrows_and_bar_share_weights':True,'raster_grid_shape':list(grid.shape)}


def capacity_figure():
    data={name:np.load(source(Path('data/processed')/MODEL/f'capacity_{name}.npz')) for name in ['dispersed','clustered']}
    with source(Path('data/processed')/MODEL/'capacity_equilibria.csv').open() as f:rows=list(csv.DictReader(f))
    fig=plt.figure(figsize=figure_size(125));colours={'clustered':BLUE,'dispersed':ORANGE};titles={'clustered':'Grouped forest clusters','dispersed':'Dispersed forest clusters'}
    for name,left,letter in [('dispersed',.17,'a'),('clustered',.62,'b')]:
        ax=fig.add_axes([left,.73,.21,.269]);ax.imshow(data[name]['grid'],cmap=ListedColormap([GREY,GREEN]),vmin=0,vmax=1,interpolation='none',resample=False);ax.set_axis_off();ax.text(.5,1.04,f'({letter}) '+titles[name],ha='center',transform=ax.transAxes,color=colours[name],fontsize=8.5)
    fig.text(.5,.69,fr'Same five forest clusters and forest cover $H={data["clustered"]["grid"].mean():.3f}$; different arrangement',ha='center',fontsize=8.5)
    ax=fig.add_axes([.14,.18,.83,.445]);ax.text(0,1.055,'(c) Equilibrium occupancy and persistence thresholds',transform=ax.transAxes)
    maximum=0
    for name in ['clustered','dispersed']:
        subset=[row for row in rows if row['landscape']==name];rr=np.array([float(row['r']) for row in subset]);means=np.array([float(row['mean_occupancy']) for row in subset]);cap=float(data[name]['capacity']);maximum=max(maximum,rr.max());assert np.all(np.diff(means)<=1e-9)
        ax.plot(rr,means,color=colours[name],linewidth=1.7,label=titles[name]);ax.plot(cap,0,'o',color=colours[name],markersize=4,clip_on=False);ax.annotate(fr'$\lambda_M={cap:.1f}$',xy=(cap,.015),xytext=(cap,.23),ha='center',color=colours[name],fontsize=9.5,arrowprops=dict(arrowstyle='->',color=colours[name],linewidth=.8))
    ax.set(xlim=(0,maximum),ylim=(0,1.02),yticks=[0,.25,.5,.75,1]);ax.set_yticklabels(['0','0.25','0.5','0.75','1']);ax.set_xlabel(r'Extinction-to-colonisation ratio $r=e_0/c$');ax.set_ylabel(r'Mean equilibrium patch occupancy $\bar p$');ax.spines[['top','right']].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.02),ncols=2,frameon=False);save(fig,'2_08_capacity')
    AUDIT['2_08_capacity']={'cached_grids_equilibria_capacities_unchanged':True,'total_forest_cells':113,'forest_cover':float(data['clustered']['grid'].mean()),'capacities':{name:float(d['capacity']) for name,d in data.items()}}


def main():
    apply_final_style();rings_and_configuration();landscape_figures();voronoi_figure();mst_figure();connectivity_figure();capacity_figure()
    for path,digest in INPUTS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    (OUT/'audit.json').write_text(json.dumps({'input_hashes':INPUTS,'inputs_unchanged':True,'analysis_reruns':False,'figure_audits':AUDIT},indent=2)+'\n')
    print('Eight Chapter2 figures exported; original result inputs unchanged.',flush=True)
if __name__=='__main__':main()
