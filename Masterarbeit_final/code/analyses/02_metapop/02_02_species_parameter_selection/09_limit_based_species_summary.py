"""English diagnostics, exports and audit summaries for the limit-based pilot."""
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,FOCAL_SCENARIOS,MISSING,NEUTRAL
from src.metapop.domain_selection import scientific_ratio,half_distance_bounds


def cells_table(data,c):
    rows=[]
    for key,part in data[data.phase.eq('pilot')].groupby(['q','scenario_id','H_target','parameter_id','log_r','half_distance']):
        values=part.loc[part.status.eq('valid'),'occupancy'].to_numpy()
        quantiles=np.quantile(values,[.05,.5,.95]) if len(values) else [np.nan]*3
        rows.append(dict(zip(['q','scenario_id','H_target','parameter_id','log_r','half_distance'],key))|dict(
            n_valid=len(values),n_excluded=int(part.n_patches.lt(2).sum()),n_nonconverged=int(part.status.eq('nonconverged').sum()),
            q05=quantiles[0],median=quantiles[1],q95=quantiles[2],
            floor_fraction=float(np.mean(values<=c['pilot_grid']['floor'])) if len(values) else np.nan,
            ceiling_fraction=float(np.mean(values>=c['pilot_grid']['ceiling'])) if len(values) else np.nan,
            intermediate_fraction=float(np.mean((values>c['pilot_grid']['floor'])&(values<c['pilot_grid']['ceiling']))) if len(values) else np.nan))
    return pd.DataFrame(rows)


def log_edges(values):
    v=np.asarray(values,float)
    return np.r_[v[0]-(v[1]-v[0])/2,(v[:-1]+v[1:])/2,v[-1]+(v[-1]-v[-2])/2]


def math_number(value,precision=3):
    label=f'{value:.{precision}g}'
    if 'e' in label:
        mantissa,exponent=label.split('e')
        return rf'{mantissa}\times10^{{{int(exponent)}}}'
    return label


def figures(data,c,cells,domains,results):
    apply_final_style(); profile_tables=[]
    for domain in domains:
        q=int(domain['q']); eligible=cells[cells.q.eq(q)&cells.n_valid.ge(c['pilot_grid']['minimum_valid_seeds'])]
        fractions=eligible.groupby(['log_r','half_distance'],as_index=False).agg(
            intermediate_fraction=('intermediate_fraction','mean'),floor_fraction=('floor_fraction','mean'),ceiling_fraction=('ceiling_fraction','mean'))
        fractions.to_csv(results/'tables'/f'09_q{q}_response_category_fractions.csv',index=False)
        lr=np.sort(fractions.log_r.unique()); ell=np.sort(fractions.half_distance.unique())
        fig,axes=plt.subplots(1,3,figsize=figure_size(76))
        fig.subplots_adjust(left=.12,right=.87,bottom=.24,top=.88,wspace=.28)
        for ax,field,descriptor in zip(axes,['intermediate_fraction','floor_fraction','ceiling_fraction'],['(a) Intermediate','(b) Near zero','(c) Near one']):
            matrix=fractions.pivot(index='log_r',columns='half_distance',values=field).reindex(index=lr,columns=ell).to_numpy()
            cmap=plt.get_cmap('cividis').copy(); cmap.set_bad(MISSING)
            mesh=ax.pcolormesh(np.exp(log_edges(np.log(ell))),log_edges(lr/np.log(10)),matrix,cmap=cmap,vmin=0,vmax=1,shading='flat',rasterized=True)
            ax.set_xscale('log'); ax.set_xticks([ell[0],10.,ell[-1]])
            ax.set_xticklabels([f'{ell[0]:.3g}','10',rf'${math_number(ell[-1],2)}$'])
            ax.get_xticklabels()[0].set_horizontalalignment('left')
            ax.get_xticklabels()[-1].set_horizontalalignment('right')
            ax.set_xlim(ell[0],ell[-1]); ax.set_ylim(lr[0]/np.log(10),lr[-1]/np.log(10))
            ticks=np.linspace(lr[0]/np.log(10),lr[-1]/np.log(10),4)
            ax.set_yticks(ticks); ax.set_yticklabels([rf'$10^{{{v:.1f}}}$' for v in ticks])
            if ax is not axes[0]: ax.tick_params(labelleft=False)
            ax.text(0,1.03,descriptor,transform=ax.transAxes,fontsize=9.5)
        cax=fig.add_axes([.90,.26,.025,.61]); bar=fig.colorbar(mesh,cax=cax,ticks=[0,.5,1]); bar.set_label('Fraction of valid landscapes')
        fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
        fig.supylabel(r'Extinction-to-colonization ratio $r$'+'\n'+rf'($q={q}$; log scale)')
        stem=results/'figures'/f'09_q{q}_parameter_response_categories'
        export_figure(fig,stem); fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=figure_size(155)); fig.subplots_adjust(left=.10,right=.98,bottom=.28,top=.94,hspace=.5,wspace=.25)
        for number,profile in enumerate(domain['profiles']):
            ax=axes.flat[number]; part=data[data.q.eq(q)&data.phase.eq('profiles')&data.parameter_id.eq(profile['parameter_id'])]
            label=rf'({chr(97+number)}) $\log_{{10}}r={profile["log_r"]/np.log(10):.2f}$, $\ell_{{1/2}}={math_number(profile["half_distance"])}$'
            ax.text(0,1.06,label,transform=ax.transAxes,fontsize=9.5)
            for scenario in FOCAL_SCENARIOS:
                rows=[]
                for h,group in part[part.scenario_id.eq(scenario['scenario_id'])].groupby('H_target'):
                    v=group.loc[group.status.eq('valid'),'occupancy'].to_numpy(); quantile=np.quantile(v,[.05,.5,.95]) if len(v) else [np.nan]*3
                    rows.append(dict(q=q,profile=profile['parameter_id'],scenario_id=scenario['scenario_id'],H_target=h,n_valid=len(v),q05=quantile[0],median=quantile[1],q95=quantile[2]))
                table=pd.DataFrame(rows); table['plotted_median']=table['median'].where(table.n_valid.ge(c['pilot_grid']['minimum_valid_seeds']))
                profile_tables.append(table)
                ax.plot(table.H_target,table.plotted_median,color=scenario['color'],linestyle=scenario['linestyle'],lw=1.5)
            ax.set_xlim(1,0); ax.set_ylim(0,1); ax.set_xticks([1,.5,0]); ax.set_yticks([0,.5,1])
            ax.spines[['top','right']].set_visible(False)
            if number%2: ax.tick_params(labelleft=False)
        axes.flat[-1].axis('off'); fig.supxlabel(r'Habitat proportion $H$'); fig.supylabel(r'Mean equilibrium patch occupancy $\bar p$'+rf' ($q={q}$)')
        handles=[Line2D([],[],color=s['color'],linestyle=s['linestyle'],label=s['label']) for s in FOCAL_SCENARIOS]
        handles.append(Line2D([],[],color=NEUTRAL,label=r'Median ($\geq3$ valid landscapes)'))
        fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
        stem=results/'figures'/f'09_q{q}_generic_profile_diagnostics'
        export_figure(fig,stem); fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)
    pd.concat(profile_tables,ignore_index=True).to_csv(results/'tables/09_generic_profile_responses.csv',index=False)


def seed_diagnostics(g,boundaries,data,c,results):
    rows=[]; geometry_rows=[]
    for seed in sorted(g.seed.unique()):
        retained=g[g.seed.ne(seed)]
        ell=half_distance_bounds(retained.d_min_positive.min(),retained.d_max.max(),c['kernel_tolerance'])
        geometry_rows.append(dict(omitted_seed=int(seed),half_distance_min=ell[0],half_distance_max=ell[1]))
    pd.DataFrame(geometry_rows).to_csv(results/'tables/09_geometry_seed_influence.csv',index=False)
    keys=['scenario_id','H_target','parameter_id']
    for q in c['model_formulations']:
        b=boundaries[boundaries.q.eq(q)]; p=data[data.q.eq(q)&data.phase.eq('pilot')&data.status.eq('valid')]
        full=p.groupby(keys).occupancy.agg(['median','count']); full.loc[full['count'].lt(3),'median']=np.nan
        for check,seed,seeds in [('leave_one_out',int(s),[x for x in sorted(g.seed.unique()) if x!=s]) for s in sorted(g.seed.unique())]+[('first_three',-1,sorted(g.seed.unique())[:3])]:
            retained=b[b.seed.isin(seeds)]; sample=p[p.seed.isin(seeds)]
            med=sample.groupby(keys).occupancy.agg(['median','count']); med.loc[med['count'].lt(3),'median']=np.nan
            changes=(med['median']-full['median']).abs().dropna()
            rows.append(dict(q=q,check=check,omitted_seed=seed,n_seed_blocks=len(seeds),
                log_r_min_fixed_dispersal=float(retained.log_r_feasible.min()),
                change_log10_r_min=float((retained.log_r_feasible.min()-b.log_r_feasible.min())/np.log(10)),
                r_max_fixed_dispersal=float(retained.capacity_upper.max()*(1+c['upper_capacity_margin'])),
                max_median_change=float(changes.max()),fraction_median_changes_above_tolerance=float((changes>c['seed_influence']['occupancy_tolerance']).mean()),
                n_comparable_cells=len(changes)))
    table=pd.DataFrame(rows)
    table['sensitive']=table.fraction_median_changes_above_tolerance.gt(c['seed_influence']['maximum_sensitive_fraction'])
    table.to_csv(results/'tables/09_seed_influence.csv',index=False)
    return table


def transition_and_interaction_tables(cells,c,results):
    """Observed adjacent crossings and mixed differences; no gap bridging."""
    from importlib import import_module
    crossing=import_module('09_species_parameter_summary').first_downward_crossing
    usable=cells.copy()
    usable.loc[usable.n_valid.lt(c['pilot_grid']['minimum_valid_seeds']),'median']=np.nan
    habitat=[]; parameter=[]; interaction=[]
    for key,part in usable.groupby(['q','scenario_id','log_r','half_distance']):
        row=dict(zip(['q','scenario_id','log_r','half_distance'],key))
        for label,t in [('H05',.05),('H50',.5),('H95',.95)]:
            point,n=crossing(part.H_target.to_numpy(),part['median'].to_numpy(),t)
            row[label]=point; row[label+'_n_downward_crossings']=n
        row['transition_width_H95_minus_H05']=row['H95']-row['H05']
        habitat.append(row)
    for key,part in usable.groupby(['q','scenario_id','H_target','half_distance']):
        part=part.sort_values('log_r'); lr=part.log_r.to_numpy(); y=part['median'].to_numpy()
        row=dict(zip(['q','scenario_id','H_target','half_distance'],key))
        for label,t in [('log10_r_at_p05',.05),('log10_r_at_p50',.5),('log10_r_at_p95',.95)]:
            point,n=crossing(-lr,y,t)
            row[label]=-point/np.log(10); row[label+'_n_crossings']=n
        row['transition_width_log10_r']=row['log10_r_at_p05']-row['log10_r_at_p95']
        row['n_intermediate_grid_values']=int(np.sum((y>.05)&(y<.95)))
        row['maximum_adjacent_occupancy_jump']=float(np.nanmax(np.abs(np.diff(y)))) if np.isfinite(y).sum()>1 else np.nan
        parameter.append(row)
    for key,part in usable.groupby(['q','scenario_id','H_target']):
        matrix=part.pivot(index='log_r',columns='half_distance',values='median').sort_index().sort_index(axis=1)
        lr=matrix.index.to_numpy(); le=np.log(matrix.columns.to_numpy()); m=matrix.to_numpy()
        for i in range(len(lr)-1):
            for j in range(len(le)-1):
                change=m[i+1,j+1]-m[i+1,j]-m[i,j+1]+m[i,j]
                interaction.append(dict(zip(['q','scenario_id','H_target'],key))|dict(log_r_low=lr[i],log_r_high=lr[i+1],half_distance_low=np.exp(le[j]),half_distance_high=np.exp(le[j+1]),mixed_occupancy_difference=change,mixed_difference_per_log_coordinate=change/((lr[i+1]-lr[i])*(le[j+1]-le[j]))))
    pd.DataFrame(habitat).to_csv(results/'tables/09_habitat_transition_locations.csv',index=False)
    pd.DataFrame(parameter).to_csv(results/'tables/09_parameter_transition_locations.csv',index=False)
    pd.DataFrame(interaction).to_csv(results/'tables/09_parameter_interactions.csv',index=False)


def summarise(g,boundaries,data,c,value,results,processed):
    cells=cells_table(data,c); cells.to_csv(results/'tables/09_parameter_scenario_h_diagnostics.csv',index=False)
    transition_and_interaction_tables(cells,c,results)
    exclusion=g.groupby(['scenario_id','H_target'],as_index=False).agg(n_total=('seed','size'),n_excluded=('n_patches',lambda x:int((x<2).sum())))
    exclusion['excluded_fraction']=exclusion.n_excluded/exclusion.n_total
    exclusion.to_csv(results/'tables/09_analysis_domain_exclusions.csv',index=False)
    boundaries.to_csv(results/'tables/09_boundary_landscapes.csv',index=False)
    audit=seed_diagnostics(g,boundaries,data,c,results)
    figures(data,c,cells,value['domains'],results)
    valid=data[data.status.eq('valid')]; failed=int(data.status.eq('nonconverged').sum())
    validation=[]
    for d in value['domains']:
        q=d['q']; part=valid[valid.q.eq(q)&valid.phase.eq('pilot')]
        low=part[np.isclose(part.log_r,d['log_r_min'],rtol=0,atol=1e-10)]
        high=part[np.isclose(part.log_r,d['log_r_max'],rtol=0,atol=1e-10)]
        boundarylow=low[np.isclose(low.half_distance,d['half_distance_values'][0])]
        validation.append(dict(q=q,n_lower_endpoint_rows=len(low),min_mean_occupancy_at_lower_r=float(low.occupancy.min()),
            min_mean_occupancy_at_shortest_dispersal=float(boundarylow.occupancy.min()),
            n_upper_endpoint_rows=len(high),max_mean_occupancy_at_upper_r=float(high.occupancy.max()),
            reference_ceiling_fraction=float(valid[valid.q.eq(q)&valid.phase.eq('profiles')&valid.parameter_id.eq('Reference')].occupancy.ge(.95).mean())))
    endpoint=pd.DataFrame(validation); endpoint.to_csv(results/'tables/09_endpoint_validation.csv',index=False)
    endpoint_passed=bool(endpoint.min_mean_occupancy_at_lower_r.ge(.99-1e-9).all() and endpoint.max_mean_occupancy_at_upper_r.le(1e-8).all())
    value['status']='validated_on_pilot_ensemble' if failed==0 and endpoint_passed else 'provisional_numerical_failure'
    value['endpoint_validation']=validation; value['n_nonconverged_rows']=failed
    value['lhs_response_status']='not_computed'
    value['result_b_status']='not_computed'
    value['profile_response_status']=[dict(q=d['q'],
        all_prescribed_profiles_near_one=bool(valid[valid.q.eq(d['q'])&valid.phase.eq('profiles')].occupancy.ge(.95).all())) for d in value['domains']]
    (results/'logs/09_parameter_domains.json').write_text(json.dumps(value,indent=2)+'\n')
    summary=dict(status=value['status'],n_landscapes=len(g),n_excluded_landscapes=int(g.n_patches.lt(2).sum()),
        n_parameter_rows=len(data),n_valid_rows=len(valid),n_nonconverged_rows=failed,
        max_iterations=int(valid.n_iter.max()),n_retried=int(data.retried.sum()),
        max_fixed_point_residual=float(valid.fixed_point_residual.max()),endpoint_validation=validation,
        seed_influence_occupancy_tolerance=c['seed_influence']['occupancy_tolerance'],
        seed_influence_sensitive_fraction_threshold=c['seed_influence']['maximum_sensitive_fraction'],
        n_sensitive_seed_checks=int(audit.sensitive.sum()),
        source_producer_fingerprint=json.loads((processed/'response_manifest.json').read_text())['producer_fingerprint'])
    (results/'logs/09_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table='\n'.join(f"| {int(d['q'])} | {d['r_min']} | {d['r_max']} |" for d in value['domains'])
    lo,hi=value['half_distance_bounds']; minchange=float(audit.change_log10_r_min.max())
    note=f'''# Limit-based species-parameter preliminary analysis

The revised analysis examines possible model responses systematically. The
shared dispersal half-distance interval is {lo:.10g}–{hi:.10g} cell-coordinate
units, derived from kernel weights of 0.01 at the shortest positive distance
and 0.99 at the largest distance. The intervals approximate kernel limits;
they do not imply a one-percent error bound on occupancy. No off-diagonal
zero distances were found in this ensemble.

| Self-colonization q | Lower r | Upper r |
|---|---:|---:|
{table}

Each lower ratio is the largest numerically bracketed value achieving mean
patch occupancy at least 0.99 in every valid pilot landscape at the shortest
half-distance. Feasible endpoints are retained, with a logarithmic bracket
width at most {c['boundary_log_relative_tolerance']:g}. The upper endpoint is
1.01 times the largest eigenvalue upper bracket at the longest half-distance.
The one-percent upper margin protects the strict extinction-boundary inequality;
it is distinct from both the kernel tolerance and the occupancy target.

Five complete matched seed blocks generated seven focal scenarios and ten H
levels. All 350 rasters were replayed and compared with their Result A hashes
and recomputed patch counts. The same 74 zero- or one-patch landscapes are
excluded for q=0 and q=1. The rectangular pilot uses 25 logarithmic ratios and
17 logarithmic half-distances per formulation. It retains saturated responses.
No marginal intermediate-response filter or scenario-effect optimization was
used. Profiles occupy the specified quarter and midpoint positions. Separate
independent 128-point logarithmic Latin hypercubes were generated after the
boundary calculation; their occupancy responses have not been evaluated.

Areas remain raw habitat-cell counts and distances have no physical calibration.
The full depth-1 Voronoi hybrid matrix is retained. The kernel is unnormalised
and untruncated, with b=1 and x=0.5. Self-colonization adds q A_i^1.5 to the
capacity-matrix diagonal. At short dispersal the self term remains active.
The same half-distance interval is shared, but r endpoints are independently
determined. Equal relative positions in the two rectangles are not identical
species parameters; their comparison addresses conclusions across model
formulations rather than the isolated effect of changing q at fixed r.

Numerical equilibria were iterated from full occupancy. Algebraic row scaling
uses B_ij=M_ij/Q_i and retains log(Q_i), where Q_i=sum_j M_ij. The map is
(Bp)_i/((Bp)_i+r/Q_i), exactly the original unnormalised model in exact arithmetic.
This prevents entire weak-source rows from disappearing through underflow.
It does not replace the model with a row-normalised dispersal kernel. Ratios
are stored as log(r) and scientific strings, with float values only where
representable. Relative weights below machine range may still round to zero;
absolute probability bounds, residuals and independent comparisons are audited.
All probability vectors and row offsets are retained. Nonconverged rows remain
missing in summaries. {failed} response rows remained nonconverged.

The domain is conditional on the pilot ensemble and its extremes. Leave-one-
seed-out lower-bound changes at fixed dispersal endpoints reached
{minchange:.3f} logarithmic decades. Separate geometry diagnostics show how
omitting seeds changes the dispersal extrema; the fixed-dispersal boundary
checks do not recompute all bounds under those altered extrema. These checks
are not a proof that five seeds capture every possible landscape. Endpoint
coverage must be rechecked when adding landscapes. The wide log-r interval
may allocate many equally spaced pilot or LHS points to saturated responses;
category fractions and profile responses are retained rather than selecting
a different domain to make scenario effects larger.

Transition tables report adjacent-level linear crossings in H or log(r),
without connecting missing values. Multiple crossings are counted. The broad
pilot grid may jump across a transition without sampling any intermediate
response, particularly for q=0. Interpolated widths are descriptive grid
diagnostics, not resolved ecological thresholds. Mixed finite differences in
log(r) and log(half-distance) describe interactions on the sampled grid.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ufz-limit-mpl .venv/bin/python -m src.tools.codex_run final-species-parameter-limits --workers 4
```

The configuration, geometry and numerical-source fingerprints, kernel extrema,
individual lower-bound brackets and capacity brackets are saved with the run.
Original and expanded historical pilot outputs are preserved separately.
'''
    (results/'logs/09_methods_and_decision.md').write_text(note)
    (results/'README.md').write_text('''# Limit-based species-parameter preliminary analysis

See [methods and domains](logs/09_methods_and_decision.md),
[machine-readable domains](logs/09_parameter_domains.json), and
[summary and endpoint checks](logs/09_summary.json).

The q=0 and q=1 heatmaps and five-profile curves are preliminary diagnostics.
Complete per-landscape boundary brackets, kernel extrema, exclusions, numerical
convergence, endpoint coverage and seed influence are retained in tables.
Independent LHS samples are generated but their responses and Result B are not
computed. Historical pilots remain preserved in their original directories.
''')
    captions='''# Preliminary figures

For each q formulation, panels (a)–(c) of the parameter-response heatmap show
intermediate (0.05 < mean occupancy < 0.95), near-zero (<=0.05), and near-one
(>=0.95) response fractions. Fractions are computed within scenario–H cells
and averaged with equal weight across cells with at least three converged
multi-patch landscapes. Five matched seeds span seven focal scenarios and ten
H levels. Excluded zero- and one-patch landscapes remain missing. The axes are
logarithmic; cells display computed grid values without interpolation. Colours
share a 0–1 scale. The entire plotted grid is the limit-based investigation
domain; there is no occupancy-selected rectangle. The two q formulations have
different r domains, so equal vertical positions need not be equal ratios.

The five-profile figures show medians of mean equilibrium patch occupancy
across the five matched seeds, using only converged multi-patch landscapes.
Panels give log10(r) and half-distance values; panel (a) is the geometric-
midpoint reference and panels (b)–(e) are the four quarter-position profiles.
Colours encode rho and dashed lines identify Access-based. H decreases in the
direction of habitat loss. Cells with fewer than three valid seeds and missing
adjacent responses are not connected. Occupancy uses the original 0–1 scale.
No uncertainty interval is plotted; descriptive 5th–95th percentiles and valid
counts are saved. Five seeds do not provide a precise estimate of distribution
tails. Units are cell coordinates without physical calibration. The full
Voronoi-hybrid kernel is used, with b=1, x=0.5, and q as indicated by the file.
'''
    (results/'figures/09_captions.md').write_text(captions)
    (results/'logs/09_run_metadata.json').write_text(json.dumps(dict(end_time_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),driver_source_sha256=hashlib.sha256(Path(__file__).with_name('09_limit_based_species_parameters.py').read_bytes()).hexdigest()),indent=2)+'\n')
