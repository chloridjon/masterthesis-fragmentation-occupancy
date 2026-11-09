"""Selection, sensitivity, publication figures and frozen transition-domain decision."""
from __future__ import annotations
import hashlib,importlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from src.metapop.transition_domain import transition_crossing,select_transition_rectangle
from src.metapop.domain_selection import scientific_ratio
SUMMARY=importlib.import_module('09_limit_based_species_summary')


def json_save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def cell_summaries(data,c):
    """Same cell statistics as the broad-survey helper, grouped in vector form."""
    keys=['q','scenario_id','H_target','parameter_id','log_r','half_distance']
    pilot=data[data.phase.eq('pilot')].copy()
    pilot['excluded']=pilot.n_patches.lt(2)
    pilot['failed']=pilot.status.eq('nonconverged')
    total=pilot.groupby(keys).agg(n_excluded=('excluded','sum'),n_nonconverged=('failed','sum'))
    valid=pilot[pilot.status.eq('valid')].copy()
    valid['floor_fraction']=valid.occupancy.le(c['pilot_grid']['floor'])
    valid['ceiling_fraction']=valid.occupancy.ge(c['pilot_grid']['ceiling'])
    valid['intermediate_fraction']=valid.occupancy.between(c['pilot_grid']['floor'],c['pilot_grid']['ceiling'],inclusive='neither')
    group=valid.groupby(keys)
    stats=group.agg(n_valid=('occupancy','size'),median=('occupancy','median'),
        floor_fraction=('floor_fraction','mean'),ceiling_fraction=('ceiling_fraction','mean'),intermediate_fraction=('intermediate_fraction','mean'))
    percentiles=group.occupancy.quantile([.05,.95]).unstack().rename(columns={.05:'q05',.95:'q95'})
    result=total.join(stats).join(percentiles).reset_index()
    result['n_valid']=result.n_valid.fillna(0).astype(int)
    return result[keys+['n_valid','n_excluded','n_nonconverged','q05','median','q95','floor_fraction','ceiling_fraction','intermediate_fraction']]


def transition_table(cells,minimum=3):
    rows=[]
    for key,part in cells.groupby(['scenario_id','H_target','half_distance']):
        if part.n_valid.min()<minimum: continue
        part=part.sort_values('log_r'); row=dict(zip(['scenario_id','H_target','half_distance'],key))
        for name,target in [('p05',.05),('p50',.5),('p95',.95)]:
            result=transition_crossing(part.log_r.to_numpy()/np.log(10),part['median'].to_numpy(),target)
            row.update({name:result['value'],name+'_low':result['low'],name+'_high':result['high'],name+'_status':result['status']})
        row['n_intermediate_grid_levels']=int(part['median'].between(.05,.95,inclusive='neither').sum())
        rows.append(row)
    return pd.DataFrame(rows)


def select(data,c,results):
    cells=cell_summaries(data,c)
    cells.to_csv(results/'tables/09_parameter_scenario_h_diagnostics.csv',index=False)
    t=transition_table(cells,c['pilot_grid']['minimum_valid_seeds'])
    t.to_csv(results/'tables/09_resolved_transitions.csv',index=False)
    selection=select_transition_rectangle(t,**c['selection'])
    pd.DataFrame(selection['plateau_steps']).to_csv(results/'tables/09_plateau_diagnostics.csv',index=False)
    pd.DataFrame(selection['transition_envelopes']).to_csv(results/'tables/09_transition_envelopes.csv',index=False)
    sensitivity=[]
    variants=[('plateau_tolerance',v) for v in c['selection_sensitivity']['plateau_tolerances']]
    variants += [('margin_log10_r',v) for v in c['selection_sensitivity']['margins_log10_r']]
    variants += [('lower_half_distance',v) for v in c['selection_sensitivity']['lower_half_distances']]
    for setting,value in variants:
        settings=c['selection']|{setting:value}
        try:
            alternative=select_transition_rectangle(t,**settings)
            sensitivity.append(dict(check=setting,value=value,status='selected',**{k:alternative[k] for k in ['half_distance_min','half_distance_max','log10_r_min','log10_r_max']}))
        except ValueError as error:
            sensitivity.append(dict(check=setting,value=value,status=str(error)))
    pd.DataFrame(sensitivity).to_csv(results/'tables/09_selection_convention_sensitivity.csv',index=False)
    seed=[]
    fullkeys=cells[cells.n_valid.ge(3)].set_index(['scenario_id','H_target','parameter_id'])['median']
    for block in sorted(data.seed.unique()):
        sample=data[data.seed.ne(block)]
        leavecells=cell_summaries(sample,c)
        leave=transition_table(leavecells,3)
        alt=select_transition_rectangle(leave,**c['selection'])
        values=leavecells[leavecells.n_valid.ge(3)].set_index(['scenario_id','H_target','parameter_id'])['median']
        difference=(values-fullkeys).abs().dropna()
        low=alt['log10_r_min']-selection['log10_r_min']; high=alt['log10_r_max']-selection['log10_r_max']
        seed.append(dict(omitted_seed=int(block),half_distance_max=alt['half_distance_max'],
            log10_r_min=alt['log10_r_min'],log10_r_max=alt['log10_r_max'],
            change_log10_r_min=low,change_log10_r_max=high,
            endpoint_sensitive=bool(max(abs(low),abs(high))>c['selection_sensitivity']['seed_endpoint_tolerance_log10']),
            half_distance_endpoint_changed=bool(alt['half_distance_max']!=selection['half_distance_max']),
            maximum_median_occupancy_change=float(difference.max()),
            fraction_median_changes_over_005=float((difference>.05).mean()),n_comparable_cells=len(difference)))
    pd.DataFrame(seed).to_csv(results/'tables/09_seed_influence.csv',index=False)
    selection.update(q=0.,r_min=scientific_ratio(selection['log10_r_min']*np.log(10)),
        r_max=scientific_ratio(selection['log10_r_max']*np.log(10)),
        log_r_min=selection['log10_r_min']*np.log(10),log_r_max=selection['log10_r_max']*np.log(10),
        status='domain_selected_profiles_pending',selection_settings=c['selection'],
        n_sensitive_seed_endpoints=sum(s['endpoint_sensitive'] for s in seed),
        n_seed_checks_changing_half_distance=sum(s['half_distance_endpoint_changed'] for s in seed),
        result_b_status='not_computed',lhs_response_status='not_computed')
    # Save all finite tail crossings as well as selected central-envelope coverage.
    lo=selection['log10_r_min']; hi=selection['log10_r_max']
    coverage=t[t.half_distance.between(selection['half_distance_min'],selection['half_distance_max'])].copy()
    coverage['lower_transition_inside']=coverage.p95_low.ge(lo)
    coverage['upper_transition_inside']=coverage.p05_high.le(hi)
    coverage['whole_transition_inside']=coverage.lower_transition_inside&coverage.upper_transition_inside
    coverage.to_csv(results/'tables/09_transition_coverage_and_tails.csv',index=False)
    selection['retained_transition_bracket_coverage_fraction']=float(coverage.whole_transition_inside.mean())
    selection['n_censored_p95_crossings']=int(t.p95_status.ne('resolved').sum())
    selection['n_censored_p50_crossings']=int(t.p50_status.ne('resolved').sum())
    selection['n_censored_p05_crossings']=int(t.p05_status.ne('resolved').sum())
    return selection


def profile_domain(decision):
    lo=decision['half_distance_min']; hi=decision['half_distance_max']
    ell=[float(np.exp(np.log(lo)+position*np.log(hi/lo))) for position in [.25,.5,.75]]
    profiles=[]
    for label,ri,ei in [('Reference',.5,1),('Low r, short dispersal',.25,0),('Low r, long dispersal',.25,2),('High r, short dispersal',.75,0),('High r, long dispersal',.75,2)]:
        profiles.append(dict(parameter_id=label,ell_id=ei,
            log_r=decision['log_r_min']+ri*(decision['log_r_max']-decision['log_r_min']),half_distance=ell[ei]))
    return dict(q=0.,grid=[],half_distance_values=ell,profiles=profiles)


def export(fig,stem):
    from src.tools.final_plot_style import export_figure
    export_figure(fig,stem)
    fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white')


def transition_figure(t,decision,results):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Patch
    from matplotlib.lines import Line2D
    from src.tools.final_plot_style import apply_final_style,figure_size,FOCAL_SCENARIOS,NEUTRAL
    apply_final_style()
    fig,axes=plt.subplots(1,2,figsize=figure_size(85)); fig.subplots_adjust(left=.12,right=.98,bottom=.28,top=.9,wspace=.3)
    for ax in axes:
        for scenario in FOCAL_SCENARIOS:
            part=t[t.scenario_id.eq(scenario['scenario_id'])]
            for _,line in part.groupby('H_target'):
                ax.plot(line.half_distance,line.p50,color=scenario['color'],linestyle=scenario['linestyle'],lw=.8,alpha=.65)
        rectangle=Rectangle((decision['half_distance_min'],decision['log10_r_min']),decision['half_distance_max']-decision['half_distance_min'],decision['log10_r_max']-decision['log10_r_min'],facecolor=NEUTRAL,alpha=.12,edgecolor=NEUTRAL,lw=1.5)
        ax.add_patch(rectangle); ax.set_xscale('log'); ax.set_ylim(decision['log10_r_min']-.5,decision['log10_r_max']+.5)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlim(1,max(t.half_distance)); axes[1].set_xlim(decision['half_distance_min'],decision['half_distance_max'])
    axes[0].text(0,1.04,'(a) Broad dispersal range',transform=axes[0].transAxes)
    axes[1].text(0,1.04,'(b) Selected main range',transform=axes[1].transAxes)
    axes[1].tick_params(labelleft=False)
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Median-occupancy transition $\log_{10}r_{50}$')
    handles=[Line2D([],[],color=s['color'],linestyle=s['linestyle'],label=s['label']) for s in FOCAL_SCENARIOS]
    handles.append(Patch(facecolor=NEUTRAL,alpha=.12,label='Selected parameter rectangle'))
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    export(fig,results/'figures/09_transition_domain_selection'); plt.close(fig)


def finish(g,data,profiles,c,decision,results,processed):
    cells=pd.read_csv(results/'tables/09_parameter_scenario_h_diagnostics.csv')
    t=pd.read_csv(results/'tables/09_resolved_transitions.csv')
    domain=profile_domain(decision); domain['grid']=[]
    # Shared thesis helper renders categories from the actual pilot and profiles
    # from the newly computed profile rows. It does not choose any main bounds.
    plotted=domain|{'half_distance_values':c['pilot_grid']['half_distances']}
    eligible=cells[cells.n_valid.ge(3)]
    eligible.groupby(['log_r','half_distance'],as_index=False)[['floor_fraction','intermediate_fraction','ceiling_fraction']].mean().to_csv(results/'tables/09_full_refinement_category_fractions.csv',index=False)
    maincells=cells[cells.half_distance.between(decision['half_distance_min'],decision['half_distance_max'])&cells.log_r.between(decision['log_r_min']-1e-10,decision['log_r_max']+1e-10)]
    SUMMARY.figures(pd.concat([data,profiles],ignore_index=True),c,maincells,[plotted],results)
    transition_figure(t,decision,results)
    valid=data[data.status.eq('valid')]; reference=profiles[profiles.parameter_id.eq('Reference')&profiles.status.eq('valid')]
    profile_summary=pd.read_csv(results/'tables/09_generic_profile_responses.csv')
    refcells=profile_summary[profile_summary.profile.eq('Reference')&profile_summary.n_valid.ge(3)]
    n_intermediate=int(refcells['median'].between(.05,.95,inclusive='neither').sum())
    reference_pass=bool(n_intermediate>0 and reference.occupancy.max()-reference.occupancy.min()>.05)
    decision.update(status='selected_main_domain_independent_audit_pending' if reference_pass else 'selected_domain_reference_review_required',
        reference_communication_check_passed=reference_pass,n_reference_intermediate_cells=n_intermediate,
        reference_occupancy_min=float(reference.occupancy.min()),reference_occupancy_max=float(reference.occupancy.max()),
        reference_ceiling_fraction=float(reference.occupancy.ge(.95).mean()),
        profiles=domain['profiles'],n_valid_pilot_responses=len(valid),n_valid_profile_responses=int(profiles.status.eq('valid').sum()),
        n_nonconverged_responses=int(pd.concat([data,profiles]).status.eq('nonconverged').sum()),
        max_fixed_point_residual=float(max(data.fixed_point_residual.max(),profiles.fixed_point_residual.max())))
    table=pd.DataFrame(domain['profiles']); table['r_float']=np.exp(table.log_r); table['alpha']=np.log(2)/table.half_distance
    table.to_csv(results/'tables/09_q0_generic_profiles.csv',index=False)
    unit=qmc.LatinHypercube(d=2,seed=c['final_lhs']['seed']).random(c['final_lhs']['n_samples'])
    lr=decision['log_r_min']+unit[:,0]*(decision['log_r_max']-decision['log_r_min'])
    ell=np.exp(np.log(decision['half_distance_min'])+unit[:,1]*np.log(decision['half_distance_max']/decision['half_distance_min']))
    lhs=pd.DataFrame(dict(parameter_id=[f'lhs{i:03d}' for i in range(1,len(lr)+1)],q=0.,log_r=lr,log10_r=lr/np.log(10),r=np.exp(lr),half_distance=ell,alpha=np.log(2)/ell,unit_log_r=unit[:,0],unit_log_half_distance=unit[:,1]))
    lhs.to_csv(results/'tables/09_q0_independent_main_lhs.csv',index=False)
    decision['final_lhs']=dict(n_samples=len(lhs),seed=c['final_lhs']['seed'],response_status='not_computed')
    continuation=results/'logs/09_equilibrium_continuations.json'
    continued=json.loads(continuation.read_text())['continued_rows'] if continuation.exists() else []
    decision['n_continued_equilibria']=len(continued)
    empty_checks=results/'logs/09_continued_equilibrium_capacity_checks.json'
    decision['n_capacity_verified_empty_equilibria']=sum(check['verified_empty'] for check in json.loads(empty_checks.read_text())['checks']) if empty_checks.exists() else 0
    decision['maximum_total_iterations']=int(max(data.n_iter.max(),profiles.n_iter.max()))
    seed_table=pd.read_csv(results/'tables/09_seed_influence.csv')
    decision['maximum_seed_endpoint_change_log10_r']=float(seed_table[['change_log10_r_min','change_log10_r_max']].abs().max().max())
    decision['n_seed_checks_changing_lower_r']=int(seed_table.change_log10_r_min.abs().gt(1e-10).sum())
    decision['ready_for_result_b']=False
    decision['numerical_profile_checks_passed']=bool(reference_pass and decision['n_nonconverged_responses']==0)
    json_save(results/'logs/09_main_parameter_decision.json',decision)
    exclusion=g.groupby(['scenario_id','H_target'],as_index=False).agg(n_total=('seed','size'),n_excluded=('n_patches',lambda x:int((x<2).sum())))
    exclusion.to_csv(results/'tables/09_analysis_domain_exclusions.csv',index=False)
    categories=pd.read_csv(results/'tables/09_q0_response_category_fractions.csv')
    selected=categories
    means=selected[['floor_fraction','intermediate_fraction','ceiling_fraction']].mean().to_dict()
    decision['selected_pilot_response_fractions']=means
    ref=next(p for p in domain['profiles'] if p['parameter_id']=='Reference')
    continuation_note=('One saved near-critical iterate required continuation' if len(continued)==1 else f'{len(continued)} saved near-critical iterates required continuation')
    capacity_note=('In one near-critical case, a verified capacity bound established' if decision['n_capacity_verified_empty_equilibria']==1 else f'In {decision["n_capacity_verified_empty_equilibria"]} near-critical cases, verified capacity bounds established')
    note=f'''# Baseline species-parameter selection: refined transitions

The main parameter domain was selected to investigate occupancy transitions,
following inspection of the broad limit-based survey. A new baseline pilot
used 949 parameter combinations on the same 350 matched snapshots. The
same 74 zero- and one-patch landscapes were excluded. All {len(valid):,}
valid pilot responses and {decision['n_valid_profile_responses']:,} valid
profile responses converged. {continuation_note}
beyond the initial iteration budget at unchanged
convergence tolerance; these are documented separately. The ratio grid used
0.25-decade steps. {capacity_note} that the only
equilibrium was exactly zero; small residual iterates were retained as
numerical diagnostics rather than interpreted as positive equilibria.

The selected main range is $r={SUMMARY.math_number(np.exp(decision['log_r_min']),3)}$ to
${SUMMARY.math_number(np.exp(decision['log_r_max']),3)}$ and dispersal half-distance
${decision['half_distance_min']:g}$ to ${decision['half_distance_max']:g}$
cell-coordinate units. The lower dispersal bound is a one-cell resolution
convention. The upper bound is the onset of a plateau in median-occupancy
transition locations under the stated 0.05-decade criterion. The central
90% transition-bracket envelope and an outward margin of 0.25 decades set
the ratio bounds. These choices were recorded before computing this pilot,
but after earlier exploratory results; they are numerical design conventions,
not empirical species limits. All eligible scenario–habitat cells receive
equal weight. The range was not selected to maximise scenario differences.

The rectangle covers {decision['retained_transition_bracket_coverage_fraction']:.1%}
of complete transition brackets across retained half-distances and eligible
scenario–habitat cells. Individual landscape responses and excluded extreme
transitions remain available. Response fractions averaged over sampled points
within the selected rectangle were {means['floor_fraction']:.1%} near zero,
{means['intermediate_fraction']:.1%} intermediate and {means['ceiling_fraction']:.1%}
near one. These describe the pilot grid, not species prevalence or future LHS
response frequencies. Low and high occupancy in some corners are expected.

The reference uses the geometric midpoint of both selected axes:
$r={SUMMARY.math_number(np.exp(ref['log_r']),4)}$ and $\\ell_{{1/2}}={ref['half_distance']:.5g}$.
It has intermediate median occupancy in {n_intermediate} of {len(refcells)}
eligible scenario–habitat cells. Across valid pilot landscapes its occupancy
ranges from {reference.occupancy.min():.6g} to {reference.occupancy.max():.6g};
{decision['reference_ceiling_fraction']:.1%} of individual responses remain
near one. The communication check {'passed' if reference_pass else 'failed; reference review remains required'}.
The four quarter-position profiles were evaluated independently at their
actual parameter values. They are illustrative profiles, not discrete species.

Omitting complete seed blocks changed the lower ratio endpoint in
{decision['n_seed_checks_changing_lower_r']} of five checks. Its largest change
was {decision['maximum_seed_endpoint_change_log10_r']:.2f} decades, a factor
of {10**decision['maximum_seed_endpoint_change_log10_r']:.2f}. Changes exceeded
the 0.25-decade reporting tolerance in {decision['n_sensitive_seed_endpoints']}
checks. The selected upper half-distance changed in
{decision['n_seed_checks_changing_half_distance']} checks. Full endpoint
changes, occupancy changes and alternative numerical conventions are retained
in the sensitivity tables. Five pilot seeds do not establish adequacy for
all landscapes. Plateau locations depend on the sampled doubling steps and
interpolated crossings; sampled brackets retain the grid's precision limits.

A new independent 128-point logarithmic Latin hypercube was generated with
seed {c['final_lhs']['seed']}. Its occupancy responses and Result B have not
been computed. The numerical profile check {'passed' if decision['numerical_profile_checks_passed'] else 'failed'}.
Independent verification is required before downstream adoption and is
reported separately below.
The broad-survey rectangles, profiles and samples remain diagnostic records.
Self-colonization will first be compared at the same main parameter values.
'''
    (results/'09_results_draft.md').write_text(note)
    (results/'README.md').write_text('''# Baseline transition-domain selection

See [verified findings and selected bounds](09_results_draft.md),
[selection protocol](09_selection_protocol.md), and
[machine-readable main decision](logs/09_main_parameter_decision.json).
The independent main sample is in `tables/09_q0_independent_main_lhs.csv`.
The broad limit-based survey remains supplementary diagnostics.
Figures show transition locations, response categories and five new profiles.
Numerical audits are recorded separately before downstream adoption.
''')
    (results/'figures/09_captions.md').write_text('''# Figure captions

**Transition-domain selection.** Each curve shows the extinction-to-colonization
ratio at which median mean equilibrium patch occupancy crosses 0.50 within a
focal scenario and habitat-proportion cell. Medians use at least three of five
independent matched landscape replicates; 55 cells are eligible. Colours
identify the seven scenarios, with Access-based curves dashed. Panel (a) shows
the full sampled dispersal range and panel (b) the selected main interval.
Shading identifies the selected parameter rectangle. Crossings use linear
interpolation in log10(r) between sampled levels 0.25 decades apart; sampled
half-distances differ by factors of two. Curves connect evaluated crossings,
without extrapolation. These curves describe ensemble responses, not confidence
intervals or empirically calibrated species. Numerical onset of the plateau
and central-envelope selection are defined in the selection protocol.

**Baseline parameter-response categories.** Panels (a), (b) and (c) show the
fractions of intermediate (0.05 < mean occupancy < 0.95), near-zero (at most
0.05) and near-one (at least 0.95) responses. Fractions are calculated across
valid independent seeds within scenario–habitat cells and averaged equally
across cells with at least three valid seeds. The heatmaps show the sampled
subset inside the selected main rectangle from the 949-point refinement
grid, which used 73 logarithmic r values and 13 half-distances. The full
refinement including the plateau tail is retained in a separate table.
All 276 multi-patch
landscapes are retained; 74 zero- or one-patch snapshots are excluded. The
common sequential colour scale ranges from zero to one. Cells display sampled
values without smoothing or interpolation; both axes are logarithmic.

**Generic baseline profiles.** Panels show the geometric midpoint reference
and four quarter-position combinations within the selected logarithmic
rectangle. Lines represent scenario-specific median mean equilibrium patch
occupancy, shown only with at least three valid landscape replicates. The
pilot has five matched seed blocks; zero- and one-patch snapshots remain
missing. Colours and Access-based dashed lines use the fixed scenario mapping.
Habitat proportion decreases from left to right. Panels report actual profile
parameter values. The 5th–95th landscape percentiles are saved in the underlying
table; no confidence intervals or variation ribbons are displayed here.
''')
    manifests={phase:json.loads((processed/phase/'manifest.json').read_text())['producer_fingerprint'] for phase in ['pilot','profiles']}
    import scipy,matplotlib
    final=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
    sources=[Path(__file__),Path(SUMMARY.__file__),Path(__file__).with_name('09_transition_species_parameters.py'),
        final/'code/base/src/metapop/transition_domain.py',final/'code/base/src/tools/final_plot_style.py']
    json_save(results/'logs/09_summary_provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        summary_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),response_fingerprints=manifests,
        source_sha256={str(p.relative_to(final)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        configuration=c,decision=decision))
    json_save(results/'logs/09_main_parameter_decision.json',decision)
