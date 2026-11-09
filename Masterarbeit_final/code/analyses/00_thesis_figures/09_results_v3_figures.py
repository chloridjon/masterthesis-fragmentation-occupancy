"""Review-v3 displays from existing, verified Result A figure tables only.
No simulations, metric recomputation, refitting, or raw-data modifications.
"""
from pathlib import Path
import sys, json, hashlib
FINAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FINAL / 'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS, roman_font_files
ROOT = FINAL / 'results/01_landscape/01_03_result_a/01_03_02_real_landscape_coverage'
OUT = FINAL / 'results/00_thesis_figures/09_results_draft_v3_20261101/figures'
OUT.mkdir(parents=True, exist_ok=True)
apply_final_style()
inputs = [ROOT/'tables/08_overall_coverage.csv'] + [ROOT/'figures/five_panel_plots'/f'08_five_panel_{name}.csv' for name in ['observations','real_medians','simulation_medians']]
coverage, real, bins, sim = [pd.read_csv(p, float_precision='round_trip') for p in inputs]
metrics = ['edge_density_total','largest_cluster_fraction_forest','chi','mean_nearest_neighbor_distance','mean_pairwise_distance','complete_mst_d50']
labels = ['Edge density','Largest cluster fraction',r'Susceptibility $\chi$','Mean nearest-neighbour\ndistance','Mean inter-patch distance',r'Complete-MST $D_{50}$']
sizes = [2000,4000,8000,16000,32000]
cov = coverage[coverage.scenario_id.eq('any_scenario')].pivot(index='metric',columns='extent_m',values='mean_equal_bin_coverage').loc[metrics,sizes]
expected=np.array([[.624,.630,.481,.356,.308],[.765,.796,.800,.776,.757],[.731,.776,.790,.793,.794],[.284,.394,.637,.798,.864],[.621,.769,.882,.922,.938],[.948,.948,.950,.952,.964]])
assert np.array_equal(np.round(cov.to_numpy(),3),expected), 'Heatmap must preserve all original displayed table values'
cov.to_csv(OUT/'09_coverage_heatmap_data.csv')
fig,ax=plt.subplots(figsize=figure_size(85))
fig.subplots_adjust(left=.38,right=.88,bottom=.17,top=.98)
im=ax.imshow(cov,aspect='auto',vmin=0,vmax=1,cmap='cividis')
ax.set_yticks(range(6),labels);ax.set_xticks(range(5),['2','4','8','16','32'])
ax.set_xlabel('Real landscape size (km)')
ax.tick_params(length=0,pad=6)
for i in range(6):
 for j in range(5): ax.text(j,i,f'{expected[i,j]:.3f}',ha='center',va='center',fontsize=8.5,color='white' if cov.iloc[i,j]<.5 else '#171717')
for spine in ax.spines.values(): spine.set_visible(False)
cbar=fig.colorbar(im,ax=ax,fraction=.06,pad=.06);cbar.set_label('Coverage proportion');cbar.set_ticks([0,.5,1])
export_figure(fig,OUT/'09_coverage_heatmap');fig.savefig(OUT/'09_coverage_heatmap.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Reuse the established scenario palette and line styles, updating display names only.
for metric,short,ylabel in [('edge_density_total','edge_density','Edge density'),('largest_cluster_fraction_forest','lcf','Largest cluster fraction')]:
 fig,axes=plt.subplots(1,2,sharex=True,sharey=True,figsize=figure_size(100))
 fig.subplots_adjust(left=.12,right=.98,top=.90,bottom=.39,wspace=.12)
 for i,(ax,size) in enumerate(zip(axes,[2000,32000])):
  points=real.loc[real.extent_m.eq(size)&np.isfinite(real[metric])]
  ax.scatter(points.H,points[metric],s=3,color='#777777',alpha=.22,linewidths=0,zorder=1)
  for item in FOCAL_SCENARIOS:
   curve=sim.loc[sim.metric.eq(metric)&sim.scenario_id.eq(item['scenario_id'])].sort_values('H_target')
   ax.plot(curve.H_target,curve['median'],color=item['color'],ls=item['linestyle'],lw=1.25,zorder=3)
  obs=bins.loc[bins.metric.eq(metric)&bins.extent_m.eq(size)].sort_values('H_bin_midpoint')
  ax.plot(obs.H_bin_midpoint,obs.real_median,color='#303030',lw=1.8,zorder=4)
  ax.set_xlim(1,0);ax.set_ylim(0,1.05 if short=='edge_density' else 1.02);ax.set_xticks([1,.5,0]);ax.set_yticks([0,.25,.5,.75,1]);ax.spines[['top','right']].set_visible(False)
  ax.text(0,1.05,f'({chr(97+i)}) {size//1000} km landscapes',transform=ax.transAxes,fontsize=9.5)
 axes[0].set_ylabel(ylabel)
 fig.supxlabel(r'Forest proportion $H$')
 handles=[]
 for item in FOCAL_SCENARIOS:
  name='only random' if item['scenario_id'].startswith('only') else 'edge random' if item['scenario_id'].startswith('edge') else 'access based'
  rho=item['label'].split('(')[1]
  handles.append(Line2D([0],[0],color=item['color'],ls=item['linestyle'],label=name+' ('+rho))
 handles.extend([Line2D([0],[0],color='#303030',lw=1.8,label='Real median (0.05-wide bins)'),Line2D([0],[0],marker='o',ls='none',color='#777777',ms=3,label='Real landscapes')])
 leg=fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,columnspacing=1.2)
 for t in leg.get_texts():
  if t.get_text().startswith(('only random','edge random','access based')): t.set_fontstyle('italic')
 export_figure(fig,OUT/f'09_{short}_2km_32km');fig.savefig(OUT/f'09_{short}_2km_32km.png',dpi=300,bbox_inches='tight');plt.close(fig)
 real.loc[real.extent_m.isin([2000,32000]),['candidate_id','extent_m','H',metric]].to_csv(OUT/f'09_{short}_observations.csv',index=False)
 bins.loc[bins.extent_m.isin([2000,32000])&bins.metric.eq(metric)].to_csv(OUT/f'09_{short}_real_medians.csv',index=False)
 sim.loc[sim.metric.eq(metric)].to_csv(OUT/f'09_{short}_simulation_medians.csv',index=False)
(OUT/'09_source_audit.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(FINAL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'heatmap_all_30_values_equal_v2_table':True,'fonts':{str(k):str(v) for k,v in roman_font_files().items()},'scope':'Existing-table visualisation only; no analyses rerun.'},indent=2))
print('Created three figures; all 30 heatmap values match the original table.')
