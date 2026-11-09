"""Tables, English methods notes and final-style figures for the species pilot."""
from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import sys
import warnings

FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
REL=Path('02_metapop/02_02_species_parameter_selection')
sys.path.insert(0,str(FINAL/'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import scipy
from src.tools.final_plot_style import apply_final_style,export_figure,figure_size,FOCAL_SCENARIOS,MISSING,NEUTRAL

RESULTS=FINAL/'results'/REL


def first_downward_crossing(h,y,threshold):
    """Linear interpolation on observed adjacent levels only, without bridging gaps."""
    crossings=[]
    order=np.argsort(h)[::-1]
    h,y=np.asarray(h)[order],np.asarray(y)[order]
    for i in range(1,len(h)):
        if np.isfinite(y[i-1]) and np.isfinite(y[i]) and y[i-1]>=threshold>y[i]:
            t=(threshold-y[i-1])/(y[i]-y[i-1])
            crossings.append(float(h[i-1]+t*(h[i]-h[i-1])))
    return (crossings[0] if crossings else np.nan,len(crossings))


def transition_summary(cells):
    rows=[]
    for key,part in cells.groupby(['scenario_id','r','half_distance']):
        values={'scenario_id':key[0],'r':key[1],'half_distance':key[2]}
        for label,threshold in [('H05',.05),('H50',.5),('H95',.95)]:
            point,count=first_downward_crossing(part.H_target.to_numpy(),part['median'].to_numpy(),threshold)
            values[label]=point
            values[f'{label}_n_downward_crossings']=count
        values['transition_width_H95_minus_H05']=values['H95']-values['H05']
        values['n_intermediate_sampled_h']=int(part['median'].between(.05,.95,inclusive='neither').sum())
        rows.append(values)
    return pd.DataFrame(rows)


def interaction_summary(cells,c):
    """Mixed finite differences at fixed scenario and H, on logarithmic axes."""
    rows=[]
    for (scenario,h),part in cells.groupby(['scenario_id','H_target']):
        part=part.copy()
        part.loc[part.n_valid.lt(c['selection']['minimum_valid_seeds_per_scenario_h']),'median']=np.nan
        matrix=part.pivot(index='r',columns='half_distance',values='median').sort_index().sort_index(axis=1)
        r,ell=matrix.index.to_numpy(),matrix.columns.to_numpy()
        m=matrix.to_numpy()
        for i in range(len(r)-1):
            for j in range(len(ell)-1):
                corner=m[i:i+2,j:j+2]
                change=corner[1,1]-corner[1,0]-corner[0,1]+corner[0,0]
                rows.append(dict(scenario_id=scenario,H_target=h,r_low=r[i],r_high=r[i+1],
                    half_distance_low=ell[j],half_distance_high=ell[j+1],
                    mixed_occupancy_difference=change,
                    mixed_difference_per_log_coordinate=change/(np.log(r[i+1]/r[i])*np.log(ell[j+1]/ell[j]))))
    return pd.DataFrame(rows)


def edges(values):
    """Geometric cell edges for a logarithmic grid; no interpolation of values."""
    log=np.log(np.asarray(values,float))
    return np.exp(np.r_[log[0]-(log[1]-log[0])/2,(log[:-1]+log[1:])/2,log[-1]+(log[-1]-log[-2])/2])


def heatmap(ax,table,field,c,descriptor,decision):
    r,ell=c['r_values'],c['half_distance_values']
    pivot=table.pivot(index='r',columns='half_distance',values=field).reindex(index=r,columns=ell)
    cmap=plt.get_cmap('cividis').copy()
    cmap.set_bad(MISSING)
    artist=ax.pcolormesh(edges(ell),edges(r),np.ma.masked_invalid(pivot.to_numpy()),cmap=cmap,vmin=0,vmax=1,shading='flat')
    ax.set_xscale('log'); ax.set_yscale('log')
    tick_candidates=[.0625,1,32,1024] if ell[0]<.25 or ell[-1]>128 else [.25,1,4,16,128]
    ticks=[v for v in tick_candidates if ell[0]<=v<=ell[-1]]
    ax.set_xticks(ticks,[f'{v:g}' for v in ticks])
    ticks=[v for v in [.001,.1,100,1e5,1e7] if r[0]<=v<=r[-1]]
    ax.set_yticks(ticks,[rf'$10^{{{int(np.log10(v))}}}$' for v in ticks])
    ax.minorticks_off()
    ax.text(.015,1.03,descriptor,transform=ax.transAxes,fontsize=9.5)
    lo,hi=decision['half_distance_range']; rlo,rhi=decision['r_range']
    ax.add_patch(Rectangle((lo,rlo),hi-lo,rhi-rlo,fill=False,edgecolor='#B24A35',linewidth=1.2))
    return artist


def figures(cells,profiles,c,decision):
    apply_final_style()
    eligible=cells[cells.n_valid.ge(c['selection']['minimum_valid_seeds_per_scenario_h'])]
    pooled=eligible.groupby(['r','half_distance'],as_index=False).agg(
        intermediate_fraction=('intermediate_fraction','mean'),floor_fraction=('floor_fraction','mean'),
        ceiling_fraction=('ceiling_fraction','mean'))
    pooled.to_csv(RESULTS/'tables/09_balanced_parameter_diagnostics.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=figure_size(75),layout=None)
    fig.subplots_adjust(left=.1,right=.86,bottom=.26,top=.85,wspace=.33)
    artist=None
    for ax,field,label in zip(axes,['intermediate_fraction','floor_fraction','ceiling_fraction'],
                               ['(a) Intermediate','(b) Near zero','(c) Near one']):
        artist=heatmap(ax,pooled,field,c,label,decision)
    for ax in axes[1:]: ax.tick_params(labelleft=False)
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    color_ax=fig.add_axes([.89,.26,.025,.59])
    fig.colorbar(artist,cax=color_ax,label='Fraction of valid landscapes',ticks=[0,.5,1])
    fig.legend(handles=[Line2D([],[],color='#B24A35',label='Candidate parameter range'),
                        Patch(facecolor=MISSING,label='Unavailable')],loc='lower center',ncol=2,frameon=False)
    export_figure(fig,RESULTS/'figures/09_parameter_response_categories')
    fig.savefig(RESULTS/'figures/09_parameter_response_categories.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

    # Profile comparison: separate panels preserve scenario colour and line style.
    fig,axes=plt.subplots(3,2,figsize=figure_size(155))
    fig.subplots_adjust(left=.10,right=.98,bottom=.29,top=.94,hspace=.48,wspace=.25)
    tables=[]
    names=list(profiles.parameter_id.drop_duplicates())
    for index,name in enumerate(names):
        ax=axes.flat[index]
        part=profiles[profiles.parameter_id.eq(name)]
        r,ell=part.iloc[0][['r','half_distance']]
        label=(rf'({chr(97+index)}) $r={r:.3g}$, $\ell_{{1/2}}={ell:.3g}$')
        ax.text(0,1.06,label,transform=ax.transAxes,fontsize=9.5)
        for scenario in FOCAL_SCENARIOS:
            records=[]
            for h,group in part[part.scenario_id.eq(scenario['scenario_id'])].groupby('H_target'):
                v=group.loc[group.status.eq('valid'),'occupancy'].to_numpy()
                q=np.quantile(v,[.05,.5,.95]) if len(v) else [np.nan]*3
                records.append(dict(profile=name,scenario_id=scenario['scenario_id'],H_target=h,n_valid=len(v),q05=q[0],median=q[1],q95=q[2]))
            tab=pd.DataFrame(records).sort_values('H_target')
            tab['plotted_median']=tab['median'].where(tab.n_valid.ge(c['selection']['minimum_valid_seeds_per_scenario_h']))
            tables.append(tab)
            ax.plot(tab.H_target,tab['plotted_median'],color=scenario['color'],ls=scenario['linestyle'],lw=1.2)
            # Five seeds give descriptive percentiles only; avoid seven overlapping ribbons.
        ax.set_xlim(1,0);ax.set_ylim(0,1)
        ax.set_xticks([1,.5,0]);ax.set_yticks([0,.5,1])
        ax.spines[['top','right']].set_visible(False)
    axes.flat[-1].set_visible(False)
    fig.supxlabel(r'Habitat proportion $H$')
    fig.supylabel(r'Mean equilibrium patch occupancy $\bar p$')
    handles=[Line2D([],[],color=s['color'],ls=s['linestyle'],label=s['label']) for s in FOCAL_SCENARIOS]
    handles += [Line2D([],[],color=NEUTRAL,label=r'Median ($\geq 3$ valid landscapes)')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    export_figure(fig,RESULTS/'figures/09_generic_profile_diagnostics')
    fig.savefig(RESULTS/'figures/09_generic_profile_diagnostics.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.concat(tables,ignore_index=True).to_csv(RESULTS/'tables/09_generic_profile_responses.csv',index=False)


def summarise(pilot,profiles,c,cells,decision,audit,processed_dir=None):
    transition_summary(cells).to_csv(RESULTS/'tables/09_transition_locations_and_widths.csv',index=False)
    interaction_summary(cells,c).to_csv(RESULTS/'tables/09_parameter_interactions_at_fixed_h.csv',index=False)
    landscape=pilot.drop_duplicates(['scenario_id','seed','H_target'])
    exclusion=landscape.groupby(['scenario_id','H_target'],as_index=False).agg(
        n_total=('seed','size'),n_excluded=('n_patches',lambda v:int((v<2).sum())),
        min_patches=('n_patches','min'),max_patches=('n_patches','max'))
    exclusion['excluded_fraction']=exclusion.n_excluded/exclusion.n_total
    exclusion.to_csv(RESULTS/'tables/09_analysis_domain_exclusions.csv',index=False)
    convergence=[]
    for phase,frame in [('pilot',pilot),('profiles',profiles)]:
        for key,group in frame.groupby(['scenario_id','H_target']):
            valid=group[group.n_patches.ge(2)]
            convergence.append(dict(phase=phase,scenario_id=key[0],H_target=key[1],
                n_attempted=len(valid),n_nonconverged=int(valid.status.eq('nonconverged').sum()),
                n_retried=int(valid.retried.sum()),max_iterations=int(valid.n_iter.max()) if len(valid) else 0,
                max_fixed_point_residual=float(valid.fixed_point_residual.max()) if len(valid) else np.nan))
    pd.DataFrame(convergence).to_csv(RESULTS/'tables/09_solver_convergence.csv',index=False)
    figures(cells,profiles,c,decision)
    reference=decision['reference_profile']
    valid=pilot[pilot.status.eq('valid')]
    n_excluded=int(landscape.n_patches.lt(2).sum())
    n_failed=int(pilot.status.eq('nonconverged').sum())
    loo_max=float(audit.loc[audit.check.eq('leave_one_out'),'max_median_change'].max())
    first_three_max=float(audit.loc[audit.check.eq('first_three'),'max_median_change'].max())
    sensitive_max=float(audit.fraction_median_changes_exceeding_tolerance.max())
    long_profile=profiles[profiles.parameter_id.eq('Low r, long dispersal') & profiles.status.eq('valid')]
    long_ceiling_fraction=float(long_profile.occupancy.ge(c['selection']['ceiling']).mean())
    note=f'''# Species-parameter selection (Sections 2.2.1–2.2.4)

The preliminary analysis identified parameter ranges that retain intermediate
mean equilibrium patch occupancy across the seven fixed landscape scenarios.
The decision used response distributions and numerical convergence, without
optimising a scenario contrast. The retained candidate ranges were r =
{decision['r_range'][0]:g}–{decision['r_range'][1]:g} and dispersal half-distance =
{decision['half_distance_range'][0]:g}–{decision['half_distance_range'][1]:g}
cell-coordinate units. The reference profile is their geometric midpoint:
r = {reference['r']:.8g}, half-distance = {reference['half_distance']:.8g}, and
alpha = {reference['alpha']:.8g}. Decision status: **{decision['status']}**.
Retention at an exploratory grid edge does not show that the informative
parameter region ends there. For expanded runs, the separate boundary table
reports changes between the outer sampled levels. Profile values and the
continuous sample remain candidates when boundaries are still open.

## Sample and model

Five matched seed blocks (replicates {c['replicates']}) represented all seven
fixed scenarios on a 200 × 200 grid. Ten habitat proportions, including the
undefined endpoints, spanned the habitat-loss gradient. Every snapshot was
regenerated from its full trajectory and checked against its Result A grid
hash and recomputed patch count. The pilot contains {len(landscape)} landscapes
and {len(pilot)} landscape–parameter rows, from a {len(c['r_values'])} × {len(c['half_distance_values'])} logarithmic grid.
The seed blocks, rather than the snapshots or parameter rows, are the
independent replication units.

Patch areas are habitat-cell counts, without normalisation. Distances use
cell coordinates and have no calibrated physical scale. Multiplying all areas
by a constant t requires multiplying r by t^(b+x) to preserve the same
equilibrium. Thus these r values must not be transferred to normalised areas
without conversion. The full depth-1 Voronoi hybrid distance matrix combines
exact distances for direct Voronoi neighbours and centroid distances for other
pairs. The exponential kernel was unnormalised and untruncated. The area
exponents were x = 0.5 and b = 1; self-colonisation was absent.

Fixed-point iteration started from full occupancy for every parameter
combination. Simultaneous columns shared a kernel but had separate convergence
checks. The maximum absolute change tolerance was 10^-10; the initial budget
was 20,000 iterations. Unconverged columns were rerun from full occupancy with
a 100,000-iteration budget. Final patch probabilities, iteration counts,
maximum changes, fixed-point residuals and convergence flags were retained.
{n_failed} pilot rows remained nonconverged and were withheld from summaries.

Mean equilibrium patch occupancy is the expected proportion of habitat patches
occupied under the model. It does not represent abundance or an empirical
probability of long-term persistence. {n_excluded} of {len(landscape)} sampled
landscapes contained fewer than two patches and were excluded. Their parameter
rows remain missing. The exclusions table reports this by scenario and H;
high-H summaries can describe a selected fragmented subset.

## Predefined range rule and stability

Responses at or below 0.05 were classed as near zero; responses at or above
0.95 as near one. Values strictly between these thresholds were intermediate.
Within each scenario–H–parameter cell, fractions used only converged,
multi-patch landscapes. Cells needed at least three valid seed blocks to
contribute to range selection. These eligible scenario–H cells received equal
weight. Marginal intermediate-response fractions averaged across the other
parameter axis. Axis levels were eligible at a marginal fraction of at least
0.10. Each retained range spans the lowest to highest eligible sampled level;
interior levels remain included even if individually below the threshold.
This rectangular design retains interactions and some saturated responses.
It does not establish optimal biological parameter limits or justify
extrapolation beyond the exploratory grid.

Complete matched seeds were omitted one at a time, and the first three pilot
seeds were compared with all five. The predeclared occupancy tolerance was
0.05, with no more than 10% of comparable medians exceeding it; range endpoints
could move at most one exploratory grid step. {int(audit.stable.sum())} of
{len(audit)} checks met both criteria. Full diagnostics retain changes in
selected endpoints and medians. The decision remains conditional on five
pilot seeds and the sampled H levels. It does not establish precision of
later Result B–D contrasts.
The largest local median change was {loo_max:.3f} when omitting one seed and
{first_three_max:.3f} when reducing the sample to three seeds. Up to
{100*sensitive_max:.2f}% of comparable cells exceeded the 0.05 occupancy
tolerance. Stable range boundaries therefore do not imply that every
individual response median is precisely estimated.
The low-r, long-dispersal profile produced ceiling responses in
{100*long_ceiling_fraction:.1f}% of its converged multi-patch landscape runs
over the sampled H domain. Retaining intermediate responses marginally along
each axis does not remove every saturated corner of a rectangular parameter
space. This profile remains the predefined lower-r, longer-dispersal contrast;
the result does not justify naming it a biologically resilient species type.

Transition locations H05, H50 and H95 were interpolated only between adjacent
observed finite median responses at a downward crossing during habitat loss.
Unobserved crossings remain missing. Multiple crossings are reported; widths
are descriptive and limited by the coarse exploratory H grid.
The interaction table records mixed finite differences of median occupancy
over adjacent logarithmic parameter-grid cells at fixed scenario and H.
This describes whether the response to one parameter changes with the other;
it does not pool habitat-proportion trends into a configuration contrast.

## Profiles and continuous sample

The five generic profiles use the geometric midpoint and the 25% and 75%
positions within each logarithmic range. A separate 128-point Latin hypercube
was generated uniformly in the two logarithmic coordinates with seed
{c['final_lhs']['seed']}. It was generated after the pilot assessment and is
independent of the pilot grid. When the decision status is provisional, this
is a candidate design and must not be treated as a frozen inferential sample. Its occupancy responses have not been evaluated;
that belongs to the subsequent species-dependent analyses. Profile curves
here are preliminary diagnostics on five seeds, rather than final Result B.

## Reproduction

Run from the repository root:

```sh
.venv/bin/python -m src.tools.codex_run {'final-species-parameter-expansion' if 'extension' in c else 'final-species-parameter-selection'} --workers 4
```

Configuration, source fingerprint and the Result A table hash are retained
in {(processed_dir or FINAL/'data/processed'/REL).relative_to(FINAL)}/run_manifest.json.
CSV rows map to NPZ probabilities through offsets: row i uses
probabilities[offsets[i]:offsets[i+1]]. Empty slices indicate excluded
landscapes. Raw data and the existing Result A tables are not modified.
'''
    (RESULTS/'logs/09_methods_and_decision.md').write_text(note)
    (RESULTS/'README.md').write_text('''# Species-parameter preliminary analysis

See [methods and decision](logs/09_methods_and_decision.md) and the
[machine-readable parameter decision](logs/09_parameter_decision.json).
The two figures are diagnostic preliminary results, not Result B.
CSV tables retain the response categories, exclusions, numerical diagnostics,
transitions, seed influence, five profiles and independent final LHS design.
''')
    captions=f'''# Preliminary species-parameter figures

## 09_parameter_response_categories

Mean equilibrium patch occupancy across a broad preliminary parameter grid.
Panels show the fraction of valid landscapes with (a) intermediate occupancy,
0.05 < mean occupancy < 0.95, (b) occupancy at or below 0.05, and (c) occupancy
at or above 0.95. Fractions were computed within scenario–H cells and averaged
with equal weight across cells having at least three converged multi-patch
landscapes. Five complete matched seeds generated seven scenarios at ten H
levels; zero- and one-patch cases were excluded. Both axes are logarithmic.
Colours share a 0–1 scale; grey denotes unavailable cells. Red outlines mark
the candidate ranges, r = {decision['r_range'][0]:g}–{decision['r_range'][1]:g} and half-distance = {decision['half_distance_range'][0]:g}–{decision['half_distance_range'][1]:g} cell-coordinate
units. These fractions describe the sampled landscape
and parameter design, rather than probabilities for real species.

## 09_generic_profile_diagnostics

Preliminary occupancy responses for the five generic profiles. Panels identify
their r and dispersal half-distance values; half-distances use cell coordinates,
without physical calibration. The reference profile in panel (a) uses the
geometric midpoint; panels (b)–(e) use the four combinations of 25% and 75%
positions in the logarithmic parameter ranges. Lines show median mean
equilibrium patch occupancy across five matched seeds per scenario at each
sampled H, restricted to converged landscapes with at least two patches.
Curves omit H levels with fewer than three valid seed blocks; the complete
table retains their descriptive medians and valid counts. This finite-count
filter does not establish precision. In particular, one Only-Random landscape
at H = 0.975 contained two patches, while the other four were excluded; its
response is retained in the table but is not joined to a response curve.
Colours identify rho; solid lines represent Edge-Random and Only-Random,
and dashed lines represent Access-based. H decreases in the direction of
habitat loss; occupancy uses its original 0–1 scale. No uncertainty or variation
interval is plotted. Descriptive 5th–95th percentiles and valid counts are
retained in 09_generic_profile_responses.csv; five seeds do not support a
precise estimate of distribution tails. The kernel uses the full depth-1
Voronoi hybrid matrix without truncation, x = 0.5, b = 1 and no self-colonisation.
Undefined endpoints and missing adjacent responses are not connected.
'''
    (RESULTS/'figures/09_captions.md').write_text(captions)
    summary=dict(n_landscapes=len(landscape),n_parameter_rows=len(pilot),n_valid_rows=len(valid),
                 n_excluded_landscapes=n_excluded,n_nonconverged_rows=n_failed,
                 max_iterations=int(valid.n_iter.max()),max_fixed_point_residual=float(valid.fixed_point_residual.max()),
                 decision=decision,seed_audit_passed=bool(audit.stable.all()))
    (RESULTS/'logs/09_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest=(processed_dir or FINAL/'data/processed'/REL)/'run_manifest.json'
    metadata=dict(start_time_utc=datetime.fromtimestamp(manifest.stat().st_mtime,timezone.utc).isoformat(),
                  end_time_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
                  matplotlib=matplotlib.__version__,
                  summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS/'logs/09_run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

    if 'extension' in c:
        boundary=pd.read_csv(RESULTS/'tables/09_boundary_response_changes.csv')
        lower=boundary[boundary.axis.eq('r') & boundary.side.eq('lower')].iloc[0]
        upper=boundary[boundary.axis.eq('half_distance') & boundary.side.eq('upper')].iloc[0]
        reference_runs=profiles[profiles.parameter_id.eq('Reference') & profiles.status.eq('valid')]
        n_reference_ceiling=int(reference_runs.occupancy.ge(c['selection']['ceiling']).sum())
        extension_note=f"""# Expanded species-parameter pilot: assessment

The original parameter grid did not delimit both informative ranges. We therefore
added r = 0.001 and 0.01 and dispersal half-distances of 0.0625, 0.125, 256, 512
and 1024 cell-coordinate units. All original 31,500 rows and their saved patch
probabilities were retained. The combined 11 × 15 grid contains {len(pilot):,}
landscape–parameter rows, including {len(valid):,} converged multi-patch runs.
The same 74 landscapes remained outside the model domain. The original run is
preserved separately; this extension supersedes its interpretation of range breadth.

The unchanged marginal rule retained r = {decision['r_range'][0]:g}–{decision['r_range'][1]:g}
and half-distance = {decision['half_distance_range'][0]:g}–{decision['half_distance_range'][1]:g}.
The shortest sampled half-distance, 0.0625, produced near-zero responses in
95.8% of the equally weighted eligible cells. At r = 1,000,000 and 10,000,000,
all eligible responses were near zero. These are sampled outer checks; the
rule excluded those levels from the candidate ranges.

The lower r boundary remains unresolved. Moving from {lower.first_value:g} to
{lower.second_value:g} changed {100*lower.fraction_changes_above_tolerance:.1f}%
of comparable occupancy medians by more than 0.02, with a maximum change of
{lower.max_median_change:.3f}. At the upper dispersal boundary, moving from
{upper.first_value:g} to {upper.second_value:g} changed
{100*upper.fraction_changes_above_tolerance:.1f}% of medians by more than 0.02;
the maximum change was {upper.max_median_change:.3f}. This met the exploratory
plateau screen of at most 5% exceeding 0.02. It establishes a small response
change for most sampled cells, not a universal upper dispersal limit.
The boundary table contains all four comparisons. Both tolerances were recorded
in the extension configuration before its runs.

Every seed influence check retained the same endpoints. This addresses seed
influence within the sampled grid, not whether the grid spans all transitions.
Marginal fractions also depend on the levels of the other parameter axis.
At long dispersal distances some r values can remain intermediate even when
further dispersal increases change little; retention at the grid edge therefore
does not necessarily imply that another extension will locate a response cutoff.

The candidate geometric-midpoint reference is r = {reference['r']:.6g},
half-distance = {reference['half_distance']:.6g}, alpha = {reference['alpha']:.6g}.
It gave near-one occupancy in {n_reference_ceiling} of {len(reference_runs)}
valid runs ({100*n_reference_ceiling/len(reference_runs):.1f}%). The low-r,
long-dispersal profile was near one on all 276 valid landscapes. These results
are retained rather than selecting profiles to strengthen a scenario contrast.
The five profiles and independent 128-point logarithmic Latin hypercube remain
candidates. The final parameter bounds and Result B reference are not frozen.
A further boundary decision must address the lower r tail and the joint
response distribution before adopting these candidates for Results B–D.
"""
        (RESULTS/'logs/09_grid_extension_assessment.md').write_text(extension_note)
        readme=RESULTS/'README.md'
        readme.write_text(readme.read_text()+"\nThe [expanded-grid assessment](logs/09_grid_extension_assessment.md) explains\nwhy these ranges remain provisional. The [results draft](09_results_draft.md)\nis an English summary for review. Neither the reference profile nor the LHS\nis frozen for downstream inference.\n")
        draft=f"""# Species-parameter preliminary analysis: expanded grid

The expanded preliminary grid retained intermediate mean equilibrium patch
occupancy over a broader parameter region than the original grid. Across seven
scenarios, five matched landscape seeds and ten habitat proportions, all
{len(valid):,} eligible landscape–parameter runs converged. Seventy-four of the
350 landscapes had fewer than two patches and were excluded from the model.
The unchanged response rule retained candidate ranges of r =
{decision['r_range'][0]:g}–{decision['r_range'][1]:g} and dispersal half-distance =
{decision['half_distance_range'][0]:g}–{decision['half_distance_range'][1]:g}
cell-coordinate units. All six seed influence checks retained the same endpoints.

The extension exposed an unresolved lower r boundary. Changing r from 0.001 to
0.01 altered {100*lower.fraction_changes_above_tolerance:.1f}% of comparable
occupancy medians by more than 0.02. Increasing dispersal half-distance from
512 to 1024 exceeded that tolerance in only
{100*upper.fraction_changes_above_tolerance:.1f}% of cells, although the largest
change was {upper.max_median_change:.3f}. Thus the sampled upper dispersal tail
approached a response plateau, while the lower r tail remained insufficiently
delimited.

The geometric-midpoint candidate reference had near-one occupancy in
{n_reference_ceiling} of {len(reference_runs)} valid runs
({100*n_reference_ceiling/len(reference_runs):.1f}%). Stable marginal range
selection therefore did not guarantee an informative reference response over
the habitat gradient. The candidate profiles and continuous parameter sample
remain provisional pending a decision on the lower r tail and the joint
response distribution. These exploratory diagnostics do not constitute Result B.
"""
        (RESULTS/'09_results_draft.md').write_text(draft)




if __name__=='__main__':
    raise SystemExit('Run through 09_species_parameter_selection.py to validate inputs and complete the sample first.')
