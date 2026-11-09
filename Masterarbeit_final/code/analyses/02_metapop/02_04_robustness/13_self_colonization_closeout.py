"""Save paired pilot findings and provisional q=1 transition brackets."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('self_pilot',HERE/'13_self_colonization_pilot.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def main():
    c,design,ext,land,out,data,fp=P.inputs(P.DEFAULT)
    audit=json.loads((out/'logs/13_numeric_audit.json').read_text())
    independent=json.loads((out/'logs/13_independent_audit.json').read_text());assert independent['status']=='passed'
    s=pd.read_csv(out/'tables/13_B_C_medians_contrasts.csv')
    contrast=s.pivot(index=['parameter_id','scenario_id','H_target'],columns='q',values='contrast').dropna()
    contrast=contrast[contrast.index.get_level_values('scenario_id')!='only_random_rho0']
    contrast['change']=contrast[1]-contrast[0]
    contrast['opposite_sign_above_005']=(contrast[0]*contrast[1]<0)&(contrast[0].abs()>.05)&(contrast[1].abs()>.05)
    contrast.to_csv(out/'tables/13_paired_contrast_changes.csv')
    b=contrast.loc[c['reference']]
    ca=pd.read_csv(out/'tables/13_C_parameter_associations.csv')
    cp=ca.pivot(index=['scenario_id','H_target','parameter'],columns='q',values='coefficient').dropna()
    cp['change']=cp[1]-cp[0];cp.to_csv(out/'tables/13_paired_C_associations.csv')
    dd=pd.read_csv(out/'tables/13_D_metric_associations.csv')
    dp=dd.pivot(index=['parameter_id','H_target','metric'],columns='q',values='coefficient').dropna()
    dr=dp.loc[c['reference']].groupby('metric').agg(n_common_H=(0,'count'),q0_median_absolute=(0,lambda x:x.abs().median()),q1_median_absolute=(1,lambda x:x.abs().median()),q0_median_signed=(0,'median'),q1_median_signed=(1,'median')).reset_index()
    dr.to_csv(out/'tables/13_D_reference_common_H.csv',index=False)
    raw=pd.concat([pd.read_csv(p) for p in (data/'snapshots').glob('*.csv')],ignore_index=True)
    matched=raw[raw['sample'].eq('matched')]
    spans=matched.groupby(['parameter_id','H_target']).agg(
        q0_span=('baseline_occupancy',lambda x:x.max()-x.min()),q1_span=('occupancy',lambda x:x.max()-x.min()))
    dp_with_span=dp.join(spans,on=['parameter_id','H_target'])
    informative=dp_with_span[(dp_with_span.q0_span>.05)&(dp_with_span.q1_span>.05)]
    d_sensitivity=dict(minimum_response_range_in_both_models=.05,n_paired_coefficients=len(informative),
        median_absolute_change=float((informative[1]-informative[0]).abs().median()),
        opposite_sign_above_01=int(((informative[0]*informative[1]<0)&(informative[0].abs()>.1)&(informative[1].abs()>.1)).sum()))
    P.save(out/'logs/13_D_response_range_sensitivity.json',d_sensitivity)
    extension=pd.read_csv(out/'tables/13_extension_survey.csv')
    brackets=[]
    for (scenario,h,ell),g in extension.groupby(['scenario_id','H_target','half_distance']):
        g=g.dropna(subset=['median']).sort_values('r');r=g.r.to_numpy();p=g['median'].to_numpy()
        if not len(p):continue
        assert np.all(np.diff(p)<=1e-8)
        for target in [.95,.5,.05]:
            if p[0]>=target and p[-1]<=target:
                j=int(np.flatnonzero(p<=target)[0]);i=max(j-1,0)
                if i==j:crossing=r[j]
                else:crossing=10**np.interp(target,[p[j],p[i]],[np.log10(r[j]),np.log10(r[i])])
                brackets.append(dict(scenario_id=scenario,H_target=h,half_distance=ell,target=target,
                    r_lower=r[i],r_upper=r[j],r_interpolated=crossing))
    brackets=pd.DataFrame(brackets);brackets.to_csv(out/'tables/13_q1_transition_brackets.csv',index=False)
    envelopes=[]
    for ell,g in brackets.groupby('half_distance'):
        low=g[g.target==.95].r_interpolated;high=g[g.target==.05].r_interpolated
        envelopes.append(dict(half_distance=ell,n_scenario_H_cells=len(low),
            minimum_r_095=float(low.min()),median_r_095=float(low.median()),maximum_r_005=float(high.max()),
            central_90_lower_r=float(10**np.quantile(np.log10(low),.05)),
            central_90_upper_r=float(10**np.quantile(np.log10(high),.95))))
    envelopes=pd.DataFrame(envelopes);envelopes.to_csv(out/'tables/13_q1_transition_envelopes.csv',index=False)
    lower=float(brackets[brackets.target==.95].r_lower.min());upper=float(brackets[brackets.target==.05].r_upper.max())
    candidate=dict(status='provisional_coarse_pilot_only',q=1,
        minimum_r_095_bracket_lower=lower,maximum_r_005_bracket_upper=upper,
        candidate_ratio_window_with_half_decade_margin=[max(.01,lower/np.sqrt(10)),min(1e7,upper*np.sqrt(10))],
        half_distance_decision='Unresolved: three levels cannot establish a dispersal plateau or a nonrectangular domain.',
        selection_scope='95%-to-5% scenario-H median transitions at five H and three half-distances; not biological calibration; no baseline decision changed.')
    P.save(out/'logs/13_parameter_range_candidate.json',candidate)
    metrics=dict(B_n_comparable=len(b),B_maximum_absolute_contrast_change=float(b.change.abs().max()),
        B_maximum_contrast_change_H_at_most_0825=float(b.loc[b.index.get_level_values('H_target')<=.825,'change'].abs().max()),
        B_opposite_sign_above_005=int(b.opposite_sign_above_005.sum()),
        C_n_paired_contrasts=len(contrast),C_maximum_absolute_contrast_change=float(contrast.change.abs().max()),
        C_opposite_sign_above_005=int(contrast.opposite_sign_above_005.sum()),
        C_n_paired_associations=len(cp),C_median_absolute_association_change=float(cp.change.abs().median()),
        C_maximum_absolute_association_change=float(cp.change.abs().max()),
        C_opposite_association_sign_above_01=int(((cp[0]*cp[1]<0)&(cp[0].abs()>.1)&(cp[1].abs()>.1)).sum()),
        D_reference_common_H=dr.to_dict(orient='records'),D_response_range_sensitivity=d_sensitivity,parameter_candidate=candidate)
    P.save(out/'logs/13_findings.json',metrics)
    coverage=pd.DataFrame(audit['response_coverage'])
    text=f'''# Self-colonization pilot for Results B-D

This exploratory comparison used three complete shared seed blocks, all seven
fixed deforestation scenarios and 13 habitat proportions. The 65 profiles are
the four conditional ratio samples in 16 systematically spaced dispersal groups,
plus the unchanged numerical reference domain131 (r=858.7775251431638,
half-distance=12.946219067264217 cell-coordinate units). The subset is a
screening design, not a new independent Latin hypercube. No bird profiles were
refitted. Baseline production results remain unchanged.

The matched comparison held species parameters, landscape identity, b=1,
x=0.5 and the unnormalised hybrid-distance kernel constant. Self-colonization
q=1 entered once through the kernel diagonal. Both models excluded the same
{audit['n_excluded_landscapes']} of {audit['n_landscapes']} landscapes with fewer
than two patches. Medians and contrasts require three valid seeds. Missing
high-H scenario medians and all corresponding contrasts remain missing.
The seed/H/scenario exclusion table is saved with the results.

## Paired occupancy and Result B

Across {audit['n_valid_pairs']} valid profile-landscape pairs, occupancy increased
by a median {audit['occupancy_increase_median']:.6f} and at most
{audit['occupancy_increase_maximum']:.6f}. This median gives each retained
profile-landscape row equal weight; it is descriptive of this pilot subset.
At the unchanged B reference, the maximum absolute change in a pointwise
scenario-minus-Only-Random contrast was
{metrics['B_maximum_absolute_contrast_change']:.6f} occupancy units.
Of {len(b)} paired reference contrasts,
{metrics['B_opposite_sign_above_005']} changed sign with both absolute
contrasts exceeding 0.05. Changes in absolute occupancy and in contrasts are
different quantities; an increase in both scenario and baseline occupancy can
reduce, preserve or reverse their difference.

The largest change occurred at H=0.925. All three sampled Only-Random
landscapes have exactly two patches there. The response averages patches
equally, so retaining an isolated small patch through self-colonization can
shift the landscape mean strongly. This conditional high-H comparison needs
particular caution. Over sampled H no greater than 0.825, the largest reference
contrast change was {metrics['B_maximum_contrast_change_H_at_most_0825']:.6f}
occupancy units.

## Result C

Across the retained parameter profiles, {metrics['C_opposite_sign_above_005']}
of {len(contrast)} comparable pointwise contrasts changed sign with both
magnitudes above 0.05. The largest absolute contrast change was
{metrics['C_maximum_absolute_contrast_change']:.6f}. The median absolute change
in the {len(cp)} defined paired Spearman coefficients relating contrasts to
log-r or log-half-distance was
{metrics['C_median_absolute_association_change']:.6f}; the maximum was
{metrics['C_maximum_absolute_association_change']:.6f}. There were
{metrics['C_opposite_association_sign_above_01']} sign reversals with both
coefficient magnitudes above 0.1. These associations are marginal under the
same joint screening design and do not isolate independent parameter effects.

Equal-weight scenario-profile-H median cells fell into floor/intermediate/ceiling
classes (thresholds 0.05 and 0.95) as follows:

{coverage.to_csv(index=False)}

These are cell fractions conditional on the sampled profiles and H levels,
not parameter-space volume estimates. A low occupancy contrast in a saturated
cell does not establish weak configuration dependence elsewhere.

## Result D

All six retained structural metrics were assessed within H across the seven
scenarios, with equal total scenario mass after paired exclusions. Every
scenario required three valid paired landscapes. Numerically constant response
ranges no greater than 1e-10 remained undefined. Across
{audit['D_n_paired_coefficients']} paired coefficients, the median absolute
change was {audit['D_median_absolute_change']:.6f}, and the largest was
{audit['D_maximum_absolute_change']:.6f}. There were
{audit['D_opposite_sign_count']} sign reversals with both magnitudes above 0.1.
The reference summary below uses only H levels defined under both models:

{dr.to_csv(index=False)}

These associations describe the generated configuration spectrum; pooling does
not control for scenario membership. Coefficients in very narrow response
ranges require particular caution. Three seeds cannot establish precise
rankings or general robustness.

Requiring an occupancy range greater than 0.05 under both models retains
{d_sensitivity['n_paired_coefficients']} paired coefficients. Within this
descriptive sensitivity subset the median absolute coefficient change is
{d_sensitivity['median_absolute_change']:.6f}, and
{d_sensitivity['opposite_sign_above_01']} sign reversals retain magnitudes
above 0.1 in both models. This check leaves the primary coefficient table
unchanged.

## Separate parameter-range survey

A distinct coarse q=1 survey evaluated 19 ratios from 0.01 to 10^7 at
half-distances 1, 16 and 256 and H=0.025, 0.2, 0.4, 0.6 and 0.825.
It does not replace the matched-domain comparison. Scenario-specific median
transitions at occupancy 0.95, 0.5 and 0.05 were bracketed on the log-r grid;
interpolated crossings are descriptive. The outer 0.95-to-0.05 brackets span
r={lower:g} to {upper:g}. Adding a half-decade margin gives the provisional
ratio window {candidate['candidate_ratio_window_with_half_decade_margin']}.
This is a candidate for a finer q=1 domain study. Three dispersal levels and
five H levels cannot validate final joint bounds or demonstrate a dispersal
plateau. Both domain choice and any q=1 reference selection remain open.
The dispersal-specific coarse transition envelopes below illustrate why a
single broad rectangle is insufficient for a final sampling decision. Central
90% envelopes use the fifth percentile of log-r at occupancy 0.95 and the
95th percentile at occupancy 0.05 across eligible scenario-H cells; they have
no added margin and are exploratory.

{envelopes.to_csv(index=False)}

## Verification and uncertainty

All saved q=1 vectors were reopened and checked against the original
equilibrium equation; maximum independently recomputed residual was
{independent['maximum_recomputed_residual']:.4g}. Baseline reference vectors
were independently rechecked at every valid landscape. Stored means,
CSV/vector hashes and landscape identities passed. The geometry was freshly
recomputed from saved rasters on {independent['n_independent_geometry_recomputations']}
fixtures. One complete fresh Access-based event trace matched the recorded
landscape hashes and removal counters. Analytic isolated-patch and explicit
scalar self-term tests passed. Nine focused batch tests and four balanced-rank
tests passed. Four incomplete trajectories were recovered after the original
iteration exceeded its budget at an exact local critical rate
(r=31.622776601683793, half-distance=1). The recovery solved the same local
self-colonization quadratic analytically while iterating external connectivity
from full occupancy. All previously completed snapshot files were preserved
byte-for-byte. The supplemental solver source, new snapshot hashes and
retained-source fingerprints are recorded in 13_critical_recovery.json.

Intervals in the B figure are 5th-95th percentiles across three generated
landscapes, not confidence intervals. No bootstrap confidence intervals were
calculated. The curves and screening changes are preliminary; they do not
establish sample adequacy or a final robustness classification. All coefficients,
denominators, missing cells and transition brackets are saved for inspection.

## Figure captions

**B.** Median equilibrium mean patch occupancy and pointwise scenario contrasts
against Only-Random for the unchanged numerical reference, under q=0 and q=1.
Colours identify the fixed rho values; dashed lines denote Access-based and
solid lines the other scenarios. Ribbons show 5th-95th percentiles across the
three shared landscape seeds. They describe landscape variation, not uncertainty
in the median. Cells with fewer than three multi-patch landscapes remain missing.
Lines connect the sampled H levels. Habitat proportion decreases left to right.

**C.** Spearman associations between pointwise median occupancy contrasts and
log10-r or log10-half-distance across the same 65 profiles, separately for every
scenario and H. Panels compare q=0 and q=1. Scenario colours and line styles
follow B. No parameter-profile resampling, smoothing or confidence intervals
are used. Undefined contrasts and numerically constant contrast ranges remain
missing. Associations are marginal under the nonrectangular joint design.

**D.** Equal-scenario weighted Spearman correlations between six structural
metrics and occupancy within H for the unchanged numerical reference.
Panels compare q=0 and q=1. The three shared seed blocks are the independent
landscape units; each scenario needs three valid paired observations.
Undefined metric/response cells remain missing. Colours identify metrics;
no confidence intervals are shown. Associations pool the generated scenarios
and do not control for their membership.

**Range survey.** Median q=1 occupancy at H=0.4 against extinction-to-colonization
ratio on a logarithmic axis, with panels for half-distances 1, 16 and 256.
All seven scenarios use three shared seeds and the same multi-patch exclusion.
Colours and line styles follow B. Lines connect the coarse sampled ratios;
no smoothing, extrapolation or confidence intervals are used. The complete
survey also includes four other H levels in the saved table.

## Reproduction

From the repository root:

```sh
.venv/bin/python -m src.tools.codex_run final-self-colonization-recovery
.venv/bin/python -m src.tools.codex_run final-self-colonization-review --stage audit
.venv/bin/python -m src.tools.codex_run final-self-colonization-review --stage export
.venv/bin/python -m src.tools.codex_run final-self-colonization-closeout
```

For an already completed dataset, rerun summaries with
`final-self-colonization-pilot --stage summary`; retain the original recovery
receipt. The first raw-iteration producer is retained for provenance.
'''
    (out/'13_pilot_report.md').write_text(text)
    exclusions=land.assign(excluded=land.n_clusters<2).groupby(['scenario_id','H_target']).excluded.agg(n_excluded='sum',n_landscapes='count').reset_index()
    exclusions.to_csv(out/'tables/13_landscape_exclusions.csv',index=False)
    P.save(out/'logs/13_closeout.json',dict(status='numeric_report_complete_visual_receipt_separate',
        report_sha256=P.sha(out/'13_pilot_report.md'),source_sha256=P.sha(Path(__file__))))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
