"""Independently inspect saved pilot vectors, geometry, summaries, and exports."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-self-pilot-mpl')
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('self_pilot',HERE/'13_self_colonization_pilot.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from src.landscape.landscape import Landscape
from src.landscape.generation import run_sequential_deforestation
from src.tools.final_plot_style import apply_final_style,figure_size,export_figure,FOCAL_SCENARIOS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LABELS=['Edge density','Largest cluster fraction',r'$\chi$','Nearest-neighbour distance','Mean inter-patch distance',r'Complete-MST $D_{50}$']
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00']

def audit(c,design,land,out,data):
    assert len(list((data/'snapshots').glob('*.csv')))==len(land),'Complete pilot required before final audit'
    assert (out/'tables/13_D_metric_associations.csv').exists(),'Summary required before final audit'
    max_residual=0.;max_mean=0.;n_vectors=0;max_base_residual=0.;source_hashes={};recomputed_geometry=0
    for path in sorted((data/'snapshots').glob('*.csv')):
        marker=json.loads(path.with_suffix('.json').read_text())
        assert P.sha(path)==marker['csv_sha256'] and P.sha(path.with_suffix('.npz'))==marker['vectors_sha256']
        frame=pd.read_csv(path,float_precision='round_trip')
        identity=f'{frame.scenario_id.iloc[0]}__rep{int(frame.replicate.iloc[0]):03d}'
        h=float(frame.H_target.iloc[0]);n=int(frame.n_patches.iloc[0])
        with np.load(path.with_suffix('.npz')) as z:
            v,offsets=z['probabilities'],z['offsets'];assert len(offsets)==len(frame)+1 and offsets[-1]==len(v)
            vectors=[v[offsets[i]:offsets[i+1]] for i in range(len(frame))]
        assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
        if n<2:
            assert len(v)==0 and frame.occupancy.isna().all();continue
        geo=P.FINAL/c['geometry_cache']/'geometry'/path.with_suffix('.npz').name
        with np.load(geo) as z:a,d=z['areas'],z['distances']
        for i,row in frame.iterrows():
            p=vectors[i];assert len(p)==n
            error=abs(p.mean()-row.occupancy);max_mean=max(max_mean,float(error));assert error<1e-14
        # Recompute the original equation in batches from stored vectors.
        for ell,items in frame.groupby('half_distance'):
            values=np.stack([vectors[i] for i in items.index],axis=1)
            k=np.exp(-np.log(2)/ell*d);np.fill_diagonal(k,1)
            sources=k@(a[:,None]*values);rates=a[:,None]**(-.5)*items.r.to_numpy()
            residual=float(np.max(np.abs(sources/(sources+rates)-values)))
            max_residual=max(max_residual,residual);assert residual<=1.1*c['tol']
        n_vectors+=len(frame)
        # Stored baseline vectors: independently re-evaluate the reference on every snapshot.
        bp=P.FINAL/c['baseline_data']/'snapshots'/path.name
        bm=json.loads(bp.with_suffix('.json').read_text())
        assert P.sha(bp)==bm['hashes']['.csv'] and P.sha(bp.with_suffix('.npz'))==bm['hashes']['.npz']
        bf=pd.read_csv(bp,float_precision='round_trip');index=int(np.flatnonzero(bf.parameter_id.eq(c['reference']))[0])
        with np.load(bp.with_suffix('.npz')) as z:bv=z['probabilities'][z['offsets'][index]:z['offsets'][index+1]]
        reference=design[design.parameter_id.eq(c['reference'])].iloc[0]
        k=np.exp(-reference.alpha*d);np.fill_diagonal(k,0)
        source=k@(a*bv);residual=float(np.max(np.abs(source/(source+reference.r*a**(-.5))-bv)))
        max_base_residual=max(max_base_residual,residual);assert residual<1.1e-12
        assert abs(bv.mean()-float(frame.loc[frame.parameter_id.eq(c['reference']),'baseline_occupancy'].iloc[0]))<1e-14
        # Recompute geometry from saved raster: every selected H of seed1/Only-Random,
        # plus H=.4 for seed1 of all remaining scenarios.
        if int(frame.replicate.iloc[0])==1 and (identity.startswith('only_random') or h==.4):
            cache=P.FINAL/c['geometry_cache']/'landscapes'/f'{identity}.npz'
            receipt=json.loads(cache.with_suffix('.json').read_text());assert P.sha(cache)==receipt['sha256']
            with np.load(cache) as z:grid=z['grids'][np.flatnonzero(z['H']==h)[0]]
            assert hashlib.sha256(grid.tobytes()).hexdigest()==frame.grid_sha256.iloc[0]
            landscape=Landscape(grid);landscape.ensure_clusters(edge_depth=0,connectivity='4',boundary_mode='open',distance_mode='voronoi_dt',voronoi_depth=1)
            np.testing.assert_array_equal(landscape.cluster_areas,a)
            np.testing.assert_allclose(landscape.cluster_dist_matrix,d,rtol=0,atol=1e-12)
            source_hashes[str(cache.relative_to(P.FINAL))]=P.sha(cache);recomputed_geometry+=1
    # Fresh full event trace under the current frozen generator.
    bc=json.loads((P.FINAL/'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds/11_lhs_occupancy.json').read_text())
    lc=json.loads((P.FINAL/bc['landscape_config']).read_text());pars=lc['access_reference']
    seed=int(land.seed.min());run=run_sequential_deforestation(np.ones((200,200),dtype=np.int8),np.random.default_rng(seed),
        rho=.99,directed_mode='access_weighted',target_habitat_fraction=0,generation_boundary=lc['generation_boundary'],
        eta=pars['eta'],k=pars['k'],lambda_decay=pars['lambda'],save_habitat_fractions=c['h_targets'],seed=seed)
    assert run.n_events==40000 and run.n_random_forced_initial==1 and run.n_random_total+run.n_directed==40000
    counts=np.r_[0,np.cumsum(np.array(run.event_modes)=='random')]
    for snap in run.snapshots:
        h=float(snap.requested_habitat_fractions[0]);row=land[(land.scenario_id=='access_weighted_rho099')&(land.replicate==1)&land.H_target.eq(h)].iloc[0]
        assert hashlib.sha256(snap.grid.tobytes()).hexdigest()==row.grid_sha256
        assert counts[snap.event_count]==row.n_random_cumulative
    # Scalar, literal ranks independently reproduce a representative equal-weight D cell.
    dd=pd.read_csv(out/'tables/13_D_metric_associations.csv')
    values=[]
    for path in sorted((data/'snapshots').glob('*__h0.400.csv')):
        f=pd.read_csv(path);f=f[f.parameter_id.eq(c['reference'])];values.append(f)
    g=pd.concat(values).merge(land,on=['scenario_id','replicate','H_target','seed','grid_sha256'],suffixes=('','_land'))
    x=rankdata(g.edge_density_total);y=rankdata(g.occupancy);expected=np.corrcoef(x,y)[0,1]
    observed=float(dd[(dd.q==1)&dd.parameter_id.eq(c['reference'])&dd.H_target.eq(.4)&dd.metric.eq('edge_density_total')].coefficient.iloc[0])
    assert abs(expected-observed)<1e-13
    save=P.save;save(out/'logs/13_independent_audit.json',dict(status='passed',n_saved_vectors=n_vectors,
        maximum_recomputed_residual=max_residual,maximum_recomputed_baseline_reference_residual=max_base_residual,
        maximum_mean_error=max_mean,n_independent_geometry_recomputations=recomputed_geometry,
        fresh_access_event_trace=True,literal_D_coefficient_error=abs(expected-observed),landscape_cache_sha256=source_hashes,
        review_source_sha256=P.sha(Path(__file__))))

def panel(ax,text):ax.text(0,1.04,text,transform=ax.transAxes,ha='left',va='bottom')

def finish(fig,stem,out):
    export_figure(fig,out/'figures'/stem)
    fig.savefig(out/'figures'/f'{stem}.png',dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def export(c,design,out):
    apply_final_style()
    s=pd.read_csv(out/'tables/13_B_reference.csv');handles=[Line2D([],[],color=x['color'],ls=x['linestyle'],label=x['label']) for x in FOCAL_SCENARIOS]
    fig,axes=plt.subplots(2,2,figsize=figure_size(158));fig.subplots_adjust(bottom=.30,hspace=.32,wspace=.28)
    for q in [0,1]:
        for scenario in FOCAL_SCENARIOS:
            g=s[(s.q==q)&s.scenario_id.eq(scenario['scenario_id'])].sort_values('H_target')
            for ax,column in [(axes[0,q],'median'),(axes[1,q],'contrast')]:
                ax.plot(g.H_target,g[column],color=scenario['color'],ls=scenario['linestyle'])
            axes[0,q].fill_between(g.H_target,g.q05,g.q95,color=scenario['color'],alpha=.12,lw=0)
        axes[0,q].set_ylim(0,1);axes[1,q].set_ylim(-1,1);axes[1,q].axhline(0,color='.7',lw=.6)
        for ax in axes[:,q]:ax.set_xlim(1,0)
        panel(axes[0,q],f'({"ab"[q]}) Occupancy, $q={q}$');panel(axes[1,q],f'({"cd"[q]}) Contrast, $q={q}$')
    axes[0,0].set_ylabel('Mean patch occupancy');axes[1,0].set_ylabel('Scenario minus Only-Random')
    fig.supxlabel('Habitat proportion, $H$')
    fig.legend(handles=handles+[Line2D([],[],color='.3',label='Median'),Patch(facecolor='.7',alpha=.3,label='5th-95th percentiles\nacross landscapes')],loc='lower center',ncol=2,frameon=False)
    finish(fig,'13_B_reference_comparison',out)
    ca=pd.read_csv(out/'tables/13_C_parameter_associations.csv')
    fig,axes=plt.subplots(2,2,figsize=figure_size(147));fig.subplots_adjust(bottom=.27,hspace=.35,wspace=.25)
    for row,parameter in enumerate(['r','half_distance']):
        for q in [0,1]:
            ax=axes[row,q]
            for scenario in FOCAL_SCENARIOS[1:]:
                g=ca[(ca.q==q)&ca.parameter.eq(parameter)&ca.scenario_id.eq(scenario['scenario_id'])].sort_values('H_target')
                ax.plot(g.H_target,g.coefficient,color=scenario['color'],ls=scenario['linestyle'])
            ax.set_ylim(-1,1);ax.set_xlim(1,0);ax.axhline(0,color='.7',lw=.6)
            symbol=r'$\log_{10}r$' if row==0 else r'$\log_{10}\ell_{1/2}$'
            panel(ax,f'({"abcd"[row*2+q]}) {symbol}, $q={q}$')
    fig.supylabel('Spearman association with occupancy contrast');fig.supxlabel('Habitat proportion, $H$')
    fig.legend(handles=handles[1:],loc='lower center',ncol=2,frameon=False)
    finish(fig,'13_C_parameter_comparison',out)
    dd=pd.read_csv(out/'tables/13_D_metric_associations.csv')
    fig,axes=plt.subplots(1,2,figsize=figure_size(90));fig.subplots_adjust(bottom=.40,wspace=.22)
    for q,ax in enumerate(axes):
        for metric,label,color in zip(P.METRICS,LABELS,COLORS):
            g=dd[(dd.q==q)&dd.parameter_id.eq(c['reference'])&dd.metric.eq(metric)].sort_values('H_target')
            ax.plot(g.H_target,g.coefficient,color=color,label=label)
        ax.set_ylim(-1,1);ax.set_xlim(1,0);ax.axhline(0,color='.7',lw=.6);panel(ax,f'({"ab"[q]}) Reference, $q={q}$')
    axes[0].set_ylabel('Weighted Spearman correlation');fig.supxlabel('Habitat proportion, $H$')
    fig.legend(handles=axes[0].get_legend_handles_labels()[0],loc='lower center',ncol=2,frameon=False)
    finish(fig,'13_D_reference_comparison',out)
    e=pd.read_csv(out/'tables/13_extension_survey.csv')
    fig,axes=plt.subplots(1,3,figsize=figure_size(90));fig.subplots_adjust(bottom=.40,wspace=.23)
    # Same H across all scenarios for an interpretable coarse dispersal comparison.
    for i,(ell,ax) in enumerate(zip(c['extension_half_distances'],axes)):
        for scenario in FOCAL_SCENARIOS:
            g=e[(e.half_distance==ell)&e.H_target.eq(.4)&e.scenario_id.eq(scenario['scenario_id'])].sort_values('r')
            ax.plot(g.r,g['median'],color=scenario['color'],ls=scenario['linestyle'])
        ax.set_xscale('log');ax.set_xlim(.01,1e7);ax.set_xticks([.01,1e2,1e6]);ax.set_ylim(0,1)
        panel(ax,rf'({"abc"[i]}) $\ell_{{1/2}}={ell}$')
    axes[0].set_ylabel('Median patch occupancy');fig.supxlabel('Extinction-to-colonization ratio, $r$ (log scale)')
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    finish(fig,'13_extension_transition_survey',out)
    P.save(out/'logs/13_export.json',dict(status='exported_visual_review_pending',
        files={str(p.relative_to(out)):P.sha(p) for p in (out/'figures').iterdir()},source_sha256=P.sha(Path(__file__))))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','export'],required=True)
    args=parser.parse_args();c,design,ext,land,out,data,fp=P.inputs(P.DEFAULT)
    if args.stage=='audit':audit(c,design,land,out,data)
    else:export(c,design,out)

if __name__=='__main__':main()
