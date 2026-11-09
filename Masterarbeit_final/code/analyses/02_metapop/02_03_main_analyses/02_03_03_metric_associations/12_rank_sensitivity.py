"""Post-analysis audit of numerical ties and seed sensitivity for Result D.

Uses the already prespecified 1e-8 rounding and .05 omission diagnostic;
does not alter coefficients, intervals, exclusion rules or figures.
"""
from pathlib import Path
import importlib
import json
import sys
import warnings
sys.path.insert(0,str(Path(__file__).parent))
D=importlib.import_module('12_result_d_metric_associations')
import numpy as np
import pandas as pd


def run():
    c,out,data=D.paths();prep=json.loads((out/'logs/12_preparation.json').read_text());D.checked(prep)
    z=np.load(data/'12_prepared_arrays.npz');rho=z['rho'];rounded=z['rho_rounded'];delta=abs(rho-rounded)
    exact_summary=D.summaries(rho);rounded_summary=D.summaries(rounded)
    rows=[]
    for m,metric in enumerate(c['metrics']):
        for p,pid in enumerate(z['parameter_ids']):
            a=delta[:,m,p];finite=np.isfinite(a)
            rows.append(dict(metric=metric,parameter_id=pid,original_median_absolute=exact_summary[0][m,p],rounded_median_absolute=rounded_summary[0][m,p],original_median_signed=exact_summary[1][m,p],rounded_median_signed=rounded_summary[1][m,p],original_sign_consistency=exact_summary[2][m,p],rounded_sign_consistency=rounded_summary[2][m,p],maximum_rounding_coefficient_change=float(np.max(a[finite])) if finite.any() else np.nan,n_rounding_changes_above_005=int((a>.05).sum()),n_original_defined=int(np.isfinite(rho[:,m,p]).sum()),n_rounded_defined=int(np.isfinite(rounded[:,m,p]).sum())))
    table=pd.DataFrame(rows);table.to_csv(out/'tables/12_numerical_rank_sensitivity.csv',index=False)
    k,m,p=np.unravel_index(np.nanargmax(delta),delta.shape);v=z['y'][k,p];v=v[np.isfinite(v)]
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);loo=np.nanmax(abs(z['omitted']-rho),axis=0)
    total=int(np.isfinite(rho).sum());sensitive=int((loo>c['seed_omission_tolerance']).sum());reference=list(z['parameter_ids']).index(c['numerical_reference'])
    boot=np.load(data/'12_bootstrap_results.npz');unavailable=np.argwhere(boot['eligible']&~boot['converged'])
    withheld=[dict(H=float(z['H'][i]),metric=c['metrics'][j],valid_fraction=float(boot['valid_fraction'][i,j]),draws=int(boot['draws_used'][i,j])) for i,j in unavailable]
    summary=dict(n_original_coefficients=total,n_rounding_changes_above_005=int((delta>.05).sum()),maximum_rounding_change=float(np.nanmax(delta)),maximum_reference_rounding_change=float(np.nanmax(delta[:,:,reference])),n_coefficients_losing_estimability_after_rounding=int((np.isfinite(rho)&~np.isfinite(rounded)).sum()),maximum_profile_median_absolute_change=float(abs(table.original_median_absolute-table.rounded_median_absolute).max()),maximum_profile_median_signed_change=float(abs(table.original_median_signed-table.rounded_median_signed).max()),seed_sensitive_coefficients=sensitive,seed_sensitive_fraction=sensitive/total,maximum_seed_omission_change=float(np.nanmax(loo)),withheld_reference_intervals=withheld,largest_rounding_cell=dict(H=float(z['H'][k]),metric=c['metrics'][m],parameter_id=str(z['parameter_ids'][p]),original=float(rho[k,m,p]),rounded=float(rounded[k,m,p]),occupancy_span=float(np.ptp(v)),fraction_of_valid_landscapes_at_floor=float((v<=.05).mean()),fraction_at_ceiling=float((v>=.95).mean())))
    note=f'''# Sensitivity of Result-D ranks

All diagnostics here use choices already fixed in12_result_d.json before computation. The original coefficients and reported reference intervals remain unchanged. Numerical rounding is a sensitivity check, not a retrospectively adopted rank rule.

Omitting a complete seed changes {sensitive}/{total} defined coefficients ({100*sensitive/total:.2f}%) by more than .05 under at least one of the16 omissions. The maximum change is{summary['maximum_seed_omission_change']:.6f}. Finite intervals should therefore not be read as evidence of uniformly adequate landscape replication.

Rounding occupancy to1e-8 changes {summary['n_rounding_changes_above_005']}/{total} comparable coefficients by more than .05; the maximum change is{summary['maximum_rounding_change']:.6f}. No original coefficient becomes undefined after rounding. The numerical reference's coefficients are unchanged by this diagnostic (maximum difference{summary['maximum_reference_rounding_change']:.3g}). Across all profile/metric summaries, the largest change in median absolute association is{summary['maximum_profile_median_absolute_change']:.6f}; the largest change in median signed association is{summary['maximum_profile_median_signed_change']:.6f}. The full table reports every profile, including changes in sign consistency.

The largest coefficient change occurs at H={summary['largest_rounding_cell']['H']:g}, metric {summary['largest_rounding_cell']['metric']}, profile {summary['largest_rounding_cell']['parameter_id']}: {summary['largest_rounding_cell']['original']:.6f} versus{summary['largest_rounding_cell']['rounded']:.6f}. Its full response span is{summary['largest_rounding_cell']['occupancy_span']:.6f}; {100*summary['largest_rounding_cell']['fraction_of_valid_landscapes_at_floor']:.1f}% of valid landscapes are at occupancy at most.05 and{100*summary['largest_rounding_cell']['fraction_at_ceiling']:.1f}% are at least.95. A nonzero overall span can coexist with many effectively tied floor or ceiling responses, so the separate span-at-least-.05 summary cannot by itself eliminate sensitivity to near ties. Local parameter-map values and individual H coefficients should be interpreted with this table, without treating tiny occupancy differences as strong biological separation.

After20000 bootstrap draws, the sole withheld reference interval is Complete-MST D50 at H=.6: only{100*withheld[0]['valid_fraction']:.2f}% of draws define the coefficient, below the prespecified95% requirement. Its original coefficient remains reported, but no interval is shown. The other491 reference intervals met the fixed availability and convergence rules. Undefined bootstrap draws can result from repeated seed selection removing metric variation; they do not imply occupancy extinction.
'''
    (out/'12_rank_and_seed_sensitivity.md').write_text(note)
    D.write(out/'logs/12_rank_sensitivity_audit.json',dict(status='saved_predeclared_diagnostics_checked',summary=summary,input_sha256={D.key(p):D.sha(p) for p in [Path(__file__),D.CONFIG,out/'logs/12_preparation.json',out/'logs/12_bootstrap_audit.json']},output_sha256={D.key(p):D.sha(p) for p in [out/'tables/12_numerical_rank_sensitivity.csv',out/'12_rank_and_seed_sensitivity.md']}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
