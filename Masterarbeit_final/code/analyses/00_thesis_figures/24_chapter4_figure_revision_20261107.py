"""Presentation-only Chapter4 revision from existing tables and vector artwork.
Preserves curves, intervals, examples and source data. No simulation runs.
"""
from pathlib import Path
import sys,os,json,hashlib,importlib.util,re,textwrap,ctypes,xml.etree.ElementTree as ET
os.environ.setdefault('MPLCONFIGDIR','/tmp/chapter4-mpl')
ROOT=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(ROOT/'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from src.tools.final_plot_style import apply_final_style,export_figure,roman_font_files
from src.tools.final_figure_data import read_metric_plot_table
OUT=ROOT/'results/00_thesis_figures/24_chapter4_figure_revision_20261107'
FIG=OUT/'figures';TABLE=OUT/'tables'
for p in [FIG,TABLE]:p.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'code/analyses/01_landscape/01_03_result_a'
STRUCT=ROOT/'results/01_landscape/01_03_result_a/01_03_01_structural_scenario_responses'
COVER=ROOT/'results/01_landscape/01_03_result_a/01_03_02_real_landscape_coverage'
METRICS=['edge_density_total','largest_cluster_fraction_forest','chi','mean_nearest_neighbor_distance','mean_pairwise_distance','complete_mst_d50']
LABELS=['Edge density ED','Largest cluster\nfraction LCF',r'Susceptibility $\chi$ (forest cells)','Mean nearest-neighbour\ndistance MNN (cells)','Mean inter-patch\ndistance MIP (cells)','Forest-connectivity distance\nthreshold D50 (cells)']
SOURCES={};AUDIT={}
def tracked(path):
    SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest();return path

def module(name,path):
    tracked(path);spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);m.export_figure=save;return m

def words(s):
    for a,b in [('Only-Random','only random'),('Edge-Random','edge random'),('Access-based','access based'),('Habitat proportion','Forest cover'),('Forest proportion','Forest cover'),('habitat cells','forest cells'),('habitat area','forest area'),('patches','clusters'),('Patches','Clusters'),('each patch','each cluster'),('nearest patch','nearest cluster'),('Patch nearest-neighbour','Cluster nearest-neighbour'),('mean NN','MNN'),('Whole-landscape NN','Landscape MNN'),('Selected exact NN links','Nearest-neighbour links'),(' km windows',' km landscapes'),(' km window',' km landscape')]:s=s.replace(a,b)
    s=re.sub(r'(?<![A-Za-z])\$H=',r'Forest cover $H=',s)
    s=s.replace('clusters; mean =','forest clusters; MIP =')
    return s

def wrap_y(label,height,fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    if label.get_text() and '\n' not in label.get_text() and label.get_window_extent(renderer).height>height:
        s=label.get_text();parts=s.split(' ');best=None
        # Keep math expressions intact; current long y-labels contain plain text.
        for i in range(1,len(parts)):
            if parts[:i].count('$')%2:continue
            a,b=' '.join(parts[:i]),' '.join(parts[i:]);score=max(len(a),len(b))
            if best is None or score<best[0]:best=(score,a+'\n'+b)
        if best:label.set_text(best[1])

def save(fig,stem):
    name=Path(stem).name
    for ax in fig.axes:
        ticks=ax.get_yticklabels()
        if any(any(n in t.get_text() for n in ['Only-Random','Edge-Random','Access-based']) for t in ticks):
            ax.set_yticks(ax.get_yticks(),[words(t.get_text()) for t in ticks])
    for t in fig.findobj(Text):
        t.set_text(words(t.get_text()))
        if any(n in t.get_text() for n in ['only random','edge random','access based']):t.set_fontstyle('italic')
    axes=[a for a in fig.axes if a.get_label()!='<colorbar>' and a.get_visible()]
    if name=='10_chi_nonlargest_cluster_size_cdfs':
        fig.subplots_adjust(wspace=.40,hspace=.48)
        for a in axes:
            a.set_ylabel('')
            for t in a.texts:t.set_text(t.get_text().replace('; ','\n'))
        for col,txt in enumerate(['Mean cumulative fraction\nof forest clusters','Mean cumulative fraction of\nremaining forest area']):
            group=axes[col::2];x=min(a.get_position().x0 for a in group)-.077;y=(min(a.get_position().y0 for a in group)+max(a.get_position().y1 for a in group))/2
            fig.text(x,y,txt,ha='center',va='center',rotation=90,fontsize=9.5)
    if name=='09_nn_only_random_access_area_contrast':axes[0].set_ylabel('Mean cumulative fraction\nof forest clusters')
    for a in axes:
        wrap_y(a.yaxis.label,a.bbox.height,fig)
        fig.canvas.draw()
        if a.xaxis.label.get_text() and '\n' not in a.xaxis.label.get_text() and a.xaxis.label.get_window_extent(fig.canvas.get_renderer()).width>a.bbox.width:
            a.xaxis.label.set_text(textwrap.fill(a.xaxis.label.get_text(),width=max(18,len(a.xaxis.label.get_text())//2+2)))
    sy=getattr(fig,'_supylabel',None)
    center=None
    if sy is not None:
        low=min(a.get_position().y0 for a in axes);high=max(a.get_position().y1 for a in axes);center=(low+high)/2
        sy.set_y(center);wrap_y(sy,(high-low)*fig.bbox.height,fig)
    export_figure(fig,FIG/name)
    fig.savefig(FIG/f'{name}.png',dpi=160,bbox_inches='tight',facecolor='white')
    AUDIT[name]={'shared_ylabel_y':None if sy is None else sy.get_position()[1],'axes_vertical_center':center,'shared_ylabel':None if sy is None else sy.get_text(),'all_ylabels':[a.yaxis.label.get_text() for a in axes if a.yaxis.label.get_text()]}
    if sy is not None:assert abs(sy.get_position()[1]-center)<1e-12
    # Several legacy renderers also save a PNG after export; prevent old-file writes.
    fig.savefig=lambda *a,**kw:None
    print('Saved',name,flush=True)

def response_figures():
    m=module('response24',BASE/'01_03_01_structural_scenario_responses/07_structural_scenario_summary.py')
    m.METRIC_LABELS.update(dict(zip(METRICS,LABELS)))
    sources=json.loads(tracked(ROOT/'config/01_landscape/01_03_result_a/figure_sources.json').read_text())
    d=read_metric_plot_table(tracked(STRUCT/'tables/07_response_by_scenario_h.csv'),ROOT,sources,('median','q05','q95'))
    apply_final_style()
    for metric in METRICS:m.plot_response(d,metric,FIG,200)

def main_coverage():
    path=tracked(ROOT/'code/analyses/00_thesis_figures/09_results_v3_figures.py');src=path.read_text()
    # Run this existing table-only renderer in an isolated namespace and redirect every output.
    src=src.replace("OUT = FINAL / 'results/00_thesis_figures/09_results_draft_v3_20261101/figures'",f'OUT = Path({str(FIG)!r})')
    src=src.replace("labels = ['Edge density','Largest cluster fraction',r'Susceptibility $\\chi$','Mean nearest-neighbour\\ndistance','Mean inter-patch distance',r'Complete-MST $D_{50}$']", "labels = ['Edge density ED','Largest cluster fraction LCF',r'Susceptibility $\\chi$','Mean nearest-neighbour\\ndistance MNN','Mean inter-patch distance MIP','Forest-connectivity distance\\nthreshold D50']")
    src=src.replace("ROOT = FINAL",'export_figure = revision_export\nROOT = FINAL')
    src=src.replace("'edge_density','Edge density'","'edge_density','Edge density ED'").replace("'lcf','Largest cluster fraction'","'lcf','Largest cluster fraction LCF'")
    for p in [COVER/'tables/08_overall_coverage.csv',*(COVER/'figures/five_panel_plots'/f'08_five_panel_{n}.csv' for n in ['observations','real_medians','simulation_medians'])]:tracked(p)
    exec(compile(src,str(path),'exec'),{'__file__':str(path),'__name__':'coverage_revision24','revision_export':save})

def appendix():
    m=module('coverage24',BASE/'01_03_02_real_landscape_coverage/08_real_landscape_figures.py')
    c=json.loads(tracked(m.CONFIG).read_text());d=pd.read_csv(tracked(COVER/'tables/08_coverage_by_bin.csv'));apply_final_style()
    for metric in METRICS:m.coverage_heatmap(d,metric,c)
    m=module('map24',BASE/'01_03_02_real_landscape_coverage/08_lcf_window_examples_figure.py');m.main()
    for p in m.DATA.iterdir():
        if p.is_file():tracked(p)
    m=module('chi24',BASE/'01_03_01_structural_scenario_responses/10_chi_cluster_size_diagnostic.py');m.render(pd.read_csv(tracked(m.CURVE_TABLE)))
    m=module('nn24',BASE/'01_03_01_structural_scenario_responses/07_nn_explanatory_schematic.py');m.main()
    m=module('area24',BASE/'01_03_01_structural_scenario_responses/09_nn_scenario_area_contrast.py')
    for p in [m.INPUT_PATCHES,m.INPUT_LANDSCAPES,m.ARCHIVED]:tracked(p)
    m.INTERVAL_TABLE=TABLE/m.INTERVAL_TABLE.name;m.SUMMARY_TABLE=TABLE/m.SUMMARY_TABLE.name;m.LOG=OUT/m.LOG.name;m.main()
    old=pd.read_csv(STRUCT/'tables/09_nn_scenario_area_summary.csv');new=pd.read_csv(m.SUMMARY_TABLE);pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-12,atol=1e-12)
    m=module('mip24',BASE/'01_03_01_structural_scenario_responses/12_interpatch_geometry_diagnostic.py');m.LABELS=('only random','edge random','access based');rep=m.render(pd.read_csv(tracked(m.OUT/'12_landscape_geometry.csv')))
    assert rep==json.loads((m.OUT/'12_audit.json').read_text())['figure_replicate']
    for sid in m.SCENARIOS:tracked(m.OUT/'grids'/f'{sid}_rep{rep:03d}.npz')
    zoom_from_svg()

def zoom_from_svg():
    # Retain all exact selected raster images and link coordinates from the vector original.
    path=tracked(STRUCT/'figures/07_low_h_nn_zoom_examples.svg');tree=ET.parse(path);root=tree.getroot();ns={'s':'http://www.w3.org/2000/svg'}
    for t in root.findall('.//s:text',ns):
        joined=''.join(n.text or '' for n in t.iter()).replace('\n','').strip()
        for oldname,newname,letter in [('Edge-Random','edge random','b'),('Access-based','access based','c')]:
            if oldname in joined:
                for child in list(t):t.remove(child)
                t.text=f'({letter}) {newname}, ρ = 0.999'
                t.set('x','0');t.set('y','-.5625');t.set('style',"font-size:9.5px;font-family:'Latin Modern Roman 8';font-style:italic")
        for node in t.iter():
            if node.text:node.text=words(node.text)
        if any(s in ''.join(t.itertext()) for s in ['only random','edge random','access based']):
            t.set('style',t.get('style','')+';font-style:italic')
            for child in t.iter():
                if child is not t:child.set('style',child.get('style','')+';font-style:italic')
    for t in root.findall('.//s:text',ns):
        for node in t.iter():
            if 'style' in node.attrib:node.set('style',node.get('style').replace("'Latin Modern Roman 8'","'LM Roman 8', 'Latin Modern Math'"))
    target=FIG/'07_low_h_nn_zoom_examples.svg';tree.write(target,encoding='utf-8',xml_declaration=True)
    # Register the actual Roman8 family with the SVG renderer as well as Matplotlib.
    fc=ctypes.CDLL('libfontconfig.so.1');fc.FcConfigGetCurrent.restype=ctypes.c_void_p
    fc.FcConfigAppFontAddFile.argtypes=[ctypes.c_void_p,ctypes.c_char_p];fc.FcConfigAppFontAddFile.restype=ctypes.c_int
    config=fc.FcConfigGetCurrent()
    fonts=list(roman_font_files().values())+[Path('/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf')]
    for font in fonts:assert font.exists() and fc.FcConfigAppFontAddFile(config,str(font).encode())

    # librsvg/Cairo vector conversion avoids regenerating any forest-removal trajectory.
    svg=ctypes.CDLL('librsvg-2.so.2');cairo=ctypes.CDLL('libcairo.so.2');gobj=ctypes.CDLL('libgobject-2.0.so.0')
    svg.rsvg_handle_new_from_file.argtypes=[ctypes.c_char_p,ctypes.c_void_p];svg.rsvg_handle_new_from_file.restype=ctypes.c_void_p
    cairo.cairo_pdf_surface_create.argtypes=[ctypes.c_char_p,ctypes.c_double,ctypes.c_double];cairo.cairo_pdf_surface_create.restype=ctypes.c_void_p
    cairo.cairo_create.argtypes=[ctypes.c_void_p];cairo.cairo_create.restype=ctypes.c_void_p
    cairo.cairo_scale.argtypes=[ctypes.c_void_p,ctypes.c_double,ctypes.c_double]
    svg.rsvg_handle_render_cairo.argtypes=[ctypes.c_void_p,ctypes.c_void_p];svg.rsvg_handle_render_cairo.restype=ctypes.c_int
    cairo.cairo_destroy.argtypes=[ctypes.c_void_p];cairo.cairo_surface_finish.argtypes=[ctypes.c_void_p];cairo.cairo_surface_destroy.argtypes=[ctypes.c_void_p];gobj.g_object_unref.argtypes=[ctypes.c_void_p]
    width=float(root.get('width').removesuffix('pt'));height=float(root.get('height').removesuffix('pt'))
    handle=svg.rsvg_handle_new_from_file(str(target).encode(),None);assert handle
    surface=cairo.cairo_pdf_surface_create(str(target.with_suffix('.pdf')).encode(),width,height);cr=cairo.cairo_create(surface);cairo.cairo_scale(cr,.75,.75)
    assert svg.rsvg_handle_render_cairo(handle,cr);cairo.cairo_destroy(cr);cairo.cairo_surface_finish(surface);cairo.cairo_surface_destroy(surface);gobj.g_object_unref(handle)
    AUDIT[target.stem]={'original_raster_images_and_all_selected_link_coordinates_preserved':True,'method':'SVG text edit and vector PDF conversion; no trajectories regenerated'}
    print('Saved',target.stem,flush=True)

if __name__=='__main__':
    response_figures();main_coverage();appendix()
    for path,sha in SOURCES.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    assert len(AUDIT)==21,len(AUDIT)
    (OUT/'revision_audit.json').write_text(json.dumps({'scope':'Chapter4 and its12 supplementary figures; presentation only','source_sha256':SOURCES,'figures':AUDIT,'fonts':{str(k):str(v) for k,v in roman_font_files().items()}},indent=2)+'\n')
