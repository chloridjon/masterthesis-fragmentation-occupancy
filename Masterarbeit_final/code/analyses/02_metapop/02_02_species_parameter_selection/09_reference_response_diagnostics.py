"""Describe reference candidates from verified pilot responses; do not select one."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
REL = Path('02_metapop/02_02_species_parameter_selection/transition_focused')
RESULTS = ROOT / 'Masterarbeit_final/results' / REL
PROCESSED = ROOT / 'Masterarbeit_final/data/processed' / REL


def intermediate_width(h, p, low=.1, high=.9):
    """Habitat width inside occupancy limits on a piecewise-linear curve."""
    total = 0.
    for left, right, a, b in zip(h[:-1], h[1:], p[:-1], p[1:]):
        if a == b:
            fraction = float(low < a < high)
        else:
            bounds = sorted(((low-a)/(b-a), (high-a)/(b-a)))
            fraction = max(0., min(1., bounds[1])-max(0., bounds[0]))
        total += (right-left)*fraction
    return total


def main():
    source = RESULTS / 'tables/09_parameter_scenario_h_diagnostics.csv'
    decision_path = RESULTS / 'logs/09_main_parameter_decision.json'
    decision = json.loads(decision_path.read_text())
    assert decision['status'] == 'validated_main_domain'
    assert decision['independent_audit_status'] == 'passed'
    d = pd.read_csv(source)
    assert not d.duplicated(['parameter_id', 'scenario_id', 'H_target']).any()
    d = d[d.n_valid.ge(3) & d.half_distance.between(1, 512)
          & d.log_r.between(decision['log_r_min']-1e-9, decision['log_r_max']+1e-9)]
    h = np.array(sorted(d.groupby('H_target').scenario_id.nunique()
                        .loc[lambda x: x.eq(7)].index))
    assert len(h) >= 3
    # Independently recompute every used median from the original saved rows.
    raw_path = PROCESSED / '09_pilot_occupancy.csv'
    raw = pd.read_csv(raw_path)
    raw = raw[raw.status.eq('valid') & raw.parameter_id.isin(d.parameter_id)
              & raw.H_target.isin(h)]
    medians = raw.groupby(['parameter_id','scenario_id','H_target']).occupancy.median()
    original = d[d.H_target.isin(h)].set_index(['parameter_id','scenario_id','H_target'])['median']
    assert len(original) == len(medians)
    assert np.allclose(original.sort_index(), medians.sort_index(), atol=1e-14, rtol=0)
    rows = []
    for key, part in d.groupby('parameter_id'):
        curves = part.pivot(index='H_target', columns='scenario_id', values='median').loc[h]
        v = curves.to_numpy()
        assert v.shape == (len(h), 7) and np.isfinite(v).all()
        separation = np.mean([np.trapezoid(np.abs(v[:,i]-v[:,j]), h)/(h[-1]-h[0])
                              for i in range(7) for j in range(i+1,7)])
        u = (part.log_r.iloc[0]-decision['log_r_min'])/(decision['log_r_max']-decision['log_r_min'])
        w = np.log(part.half_distance.iloc[0])/np.log(512)
        rows.append(dict(parameter_id=key, r=np.exp(part.log_r.iloc[0]),
                         half_distance=part.half_distance.iloc[0],
                         minimum_scenario_occupancy_span=np.ptp(v, axis=0).min(),
                         largest_low_H_occupancy=v[0].max(),
                         smallest_high_H_occupancy=v[-1].min(),
                         minimum_intermediate_H_width=min(intermediate_width(h,v[:,i]) for i in range(7)),
                         mean_pairwise_scenario_separation=separation,
                         relative_log_r=u, relative_log_half_distance=w,
                         distance_to_log_midpoint=np.hypot(u-.5,w-.5),
                         interior=bool(1e-9<u<1-1e-9 and 1e-9<w<1-1e-9),
                         all_scenarios_reach_10_and_90_percent=bool(v[0].max()<=.1 and v[-1].min()>=.9)))
    out = pd.DataFrame(rows)
    # Three response criteria are retained separately: no hidden weighted score.
    criteria = ['minimum_scenario_occupancy_span', 'minimum_intermediate_H_width',
                'mean_pairwise_scenario_separation']
    out['response_pareto_interior'] = False
    interior = out[out.interior]
    scores = interior[criteria].to_numpy()
    for index, score in zip(interior.index, scores):
        dominated = ((scores >= score).all(axis=1) & (scores > score).any(axis=1)).any()
        out.loc[index, 'response_pareto_interior'] = not dominated
    target = RESULTS / 'tables/09_reference_response_candidates.csv'
    out.sort_values(['half_distance','r']).to_csv(target, index=False)
    complete = out[out.interior & out.all_scenarios_reach_10_and_90_percent]
    for _, row in complete.iterrows():
        part = d[d.parameter_id.eq(row.parameter_id)].sort_values(['H_target','scenario_id'])
        part.to_csv(RESULTS / 'tables' / f'09_reference_candidate_{row.parameter_id}_curves.csv', index=False)
    paths = [source, raw_path, Path(__file__), target]
    log = dict(status='diagnostics_complete_reference_not_selected',
               n_candidates=len(out), n_interior_candidates=len(interior),
               common_H=h.tolist(), n_recomputed_medians=len(original),
               near_endpoint_reporting_limits=[.1,.9],
               interpolation='piecewise linear; no extrapolation',
               scenario_weighting='equal; all 21 pairs for separation',
               H_weighting='habitat interval width for transition and separation',
               boundary_policy='exclude exact domain boundaries; midpoint distance is a soft preference',
               criteria=criteria, n_interior_pareto_candidates=int(out.response_pareto_interior.sum()),
               interior_endpoint_candidates=complete.parameter_id.tolist(),
               hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (RESULTS / 'logs/09_reference_response_diagnostics.json').write_text(json.dumps(log,indent=2)+'\n')
    print(json.dumps({k:v for k,v in log.items() if k!='hashes'},indent=2))
    print(complete.to_string(index=False))


if __name__ == '__main__':
    main()
