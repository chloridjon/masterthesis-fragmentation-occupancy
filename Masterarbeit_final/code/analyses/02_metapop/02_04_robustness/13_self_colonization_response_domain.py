"""Response-conditioned q=1 grid screening for Chapter-5 TODO 5-05.

Run from the repository root with the existing .venv/bin/python. This consumes
audited, immutable numerical responses; it does not rerun the model or change
the baseline design. Selection is independent of scenario contrasts. Retained
grid points are certified for the baseline all-H response-envelope inequalities
using witnesses at five H levels; unretained points remain unresolved for the
all-H rule. Counts are grid diagnostics, not area estimates or species weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-self-response-domain-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(FINAL / 'code/base'))
import numpy as np
import pandas as pd
from src.tools.final_plot_style import FOCAL_SCENARIOS

REL = '02_metapop/02_04_robustness/self_colonization_response_domain_20261102'
CONFIG = FINAL / 'config' / REL / '13_response_domain.json'
INPUTS = {}
KEY = ['parameter_id', 'half_distance', 'log10_r', 'r']


def checked(path, digest=None):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest is not None:
        assert actual == digest, f'Checksum mismatch: {path}'
    INPUTS[str(path.relative_to(FINAL))] = actual
    return path


def read_json(path):
    return json.loads(checked(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(table, c):
    group = table.groupby(KEY + ['scenario_id', 'H_target'], sort=True).occupancy
    s = group.agg(n_valid='count', median='median').reset_index()
    quantiles = group.quantile([.05, .95], interpolation=c['quantile_method']).unstack()
    quantiles.columns = ['q05', 'q95']
    s = s.merge(quantiles.reset_index(), on=KEY + ['scenario_id', 'H_target'], validate='one_to_one')
    s.loc[s.n_valid < c['minimum_valid_seeds'], ['median', 'q05', 'q95']] = np.nan
    envelope = s.groupby(KEY).agg(L=('q05', 'min'), U=('q95', 'max'), eligible_cells=('median', 'count')).reset_index()
    epsilon = c['saturation_threshold']
    envelope['retained'] = (envelope.L < 1 - epsilon) & (envelope.U > epsilon)
    envelope['classification'] = np.where(envelope.retained, 'certified_inner_point', 'unresolved_all_H')
    baseline = s[s.scenario_id.eq(c['baseline_scenario'])][KEY + ['H_target', 'median']].rename(columns={'median': 'baseline_median'})
    s = s.merge(baseline, on=KEY + ['H_target'], how='left', validate='many_to_one')
    s['contrast'] = s['median'] - s.baseline_median
    s = s.merge(envelope[KEY + ['retained']], on=KEY, validate='many_to_one')
    return envelope, s


def minima(summary, c, label, n, omitted=None):
    rows = []
    for (sid, h), g in summary[summary.scenario_id.ne(c['baseline_scenario'])].groupby(['scenario_id', 'H_target']):
        for subset, part in [('rectangle', g), ('response_inner_points', g[g.retained])]:
            finite = part[part.contrast.notna()]
            row = dict(check=label, n_runs=n, omitted_run=omitted, scenario_id=sid, H_target=float(h), subset=subset,
                       n_evaluable=len(finite), n_negative=int((finite.contrast < -1e-9).sum()),
                       n_below_minus_005=int((finite.contrast < c['contrast_threshold']).sum()))
            if len(finite):
                low = finite.loc[finite.contrast.idxmin()]
                row.update(minimum_contrast=float(low.contrast), parameter_id=low.parameter_id,
                           r=float(low.r), half_distance=float(low.half_distance))
            rows.append(row)
    return rows


def main():
    c = read_json(CONFIG)
    checked(Path(__file__))
    audit_path = FINAL / c['source_audit']
    audit = read_json(audit_path)
    assert audit['status'] == 'numerically_and_visually_checked_extension'
    for name, digest in audit['independent_audits'].items():
        checked(audit_path.parent / name, digest)
    decision = read_json(FINAL / c['rectangle_decision'])
    out = FINAL / c['results_directory']
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    ell = 2. ** np.arange(10)
    logr = np.arange(-1.5, 6.75 + .125, .25)
    assert len(ell) == 10 and len(logr) == 34
    assert decision['half_distance_min'] == ell[0] and decision['half_distance_max'] == ell[-1]
    assert decision['log10_r_min'] == logr[0] and decision['log10_r_max'] == logr[-1]
    save(out / 'logs/13_preflight.json', dict(status='passed', q=1, half_distances=ell.tolist(),
         log10_ratios=logr.tolist(), numeric_ratio_endpoints=(10. ** logr[[0, -1]]).tolist(),
         n_rectangle_points=len(ell) * len(logr), n_runs=8, h_targets=c['h_targets'],
         selection_independent_of_contrasts=True, no_new_model_runs=True,
         exclusions='Same multi-patch domain; n_patches<2 excluded; minimum three valid runs per scenario-H cell.',
         limitations=c['h_scope']))
    parts = []
    for scenario in FOCAL_SCENARIOS:
        sid = scenario['scenario_id']
        for rep in c['replicates']:
            for h in c['h_targets']:
                path = FINAL / c['source_directory'] / f'{sid}__rep{rep:03d}__h{h:.3f}.csv'
                receipt = read_json(path.with_suffix('.json'))
                checked(path, receipt['csv_sha256'])
                t = pd.read_csv(path, float_precision='round_trip')
                assert len(t) == 481 and t.parameter_id.nunique() == 481
                assert t.scenario_id.eq(sid).all() and t.replicate.eq(rep).all() and t.H_target.eq(h).all()
                assert t.grid_sha256.eq(receipt['grid_sha256']).all()
                assert t.n_patches.nunique() == 1
                eligible = t.n_patches.ge(2)
                assert np.isfinite(t.loc[eligible, 'occupancy']).all()
                assert t.loc[eligible, 'occupancy'].between(0, 1).all()
                assert t.loc[eligible, 'fixed_point_residual'].le(1.1e-12).all()
                # The raw source caches contain plateau-tail points as well.
                t = t[t.half_distance.isin(ell) & t.log10_r.isin(logr)].copy()
                assert len(t) == 340
                t.loc[t.n_patches.lt(2), 'occupancy'] = np.nan
                parts.append(t[KEY + ['scenario_id', 'replicate', 'H_target', 'occupancy', 'n_patches', 'grid_sha256']])
    table = pd.concat(parts, ignore_index=True)
    assert len(table) == 340 * 7 * 5 * 8
    assert not table.duplicated(['parameter_id', 'scenario_id', 'replicate', 'H_target']).any()
    rows = []
    stages = []
    for n in c['replication_checks']:
        selected = table[table.replicate.le(n)]
        envelope, summary = summarize(selected, c)
        envelope.to_csv(out / f'tables/13_domain_points_{n:03d}.csv', index=False)
        summary.to_csv(out / f'tables/13_contrasts_{n:03d}.csv', index=False)
        rows += minima(summary, c, 'all_runs', n)
        omissions = []
        omission_envelopes = []
        for omitted in range(1, n + 1):
            oe, os = summarize(selected[selected.replicate.ne(omitted)], c)
            rows += minima(os, c, 'complete_run_omission', n - 1, omitted)
            omissions.append(os.assign(omitted_run=omitted))
            omission_envelopes.append(oe.set_index('parameter_id').retained.rename(omitted))
        omission_table = pd.concat(omissions, ignore_index=True)
        omission_table.to_csv(out / f'tables/13_omission_contrasts_{n:03d}.csv', index=False)
        stability = pd.concat(omission_envelopes, axis=1)
        base_retained = envelope.set_index('parameter_id').retained
        main_contrast = summary.set_index(['parameter_id', 'scenario_id', 'H_target']).contrast
        differences = []
        for omitted, os in omission_table.groupby('omitted_run'):
            delta = (os.set_index(['parameter_id', 'scenario_id', 'H_target']).contrast - main_contrast).abs()
            differences.append(float(delta.max()))
        stages.append(dict(n_runs=n, n_retained=int(envelope.retained.sum()),
                           n_unresolved_all_H=int((~envelope.retained).sum()),
                           n_retained_in_every_omission=int(stability.all(axis=1).sum()),
                           n_classification_sensitive=int(stability.ne(base_retained, axis=0).any(axis=1).sum()),
                           maximum_absolute_contrast_change_on_omission=max(differences)))
    extremes = pd.DataFrame(rows)
    extremes.to_csv(out / 'tables/13_minima_and_omissions.csv', index=False)
    result = dict(status='checked_descriptive_grid_screening', source_csv_count=len(parts),
                  source_rows=len(table), stages=stages, q=1, selection_rule=c['selection'],
                  limitations=[c['h_scope'], 'Finite grid and modest replication; no continuous absence proof or confidence intervals.',
                               'Same saturation criterion but no new 256-profile design, bird fit, or one-to-one profile remapping.',
                               'Grid counts do not estimate continuous area or biological species frequencies.'],
                  input_sha256=INPUTS)
    save(out / 'logs/13_audit.json', result)
    focal = extremes[(extremes.check == 'all_runs') & extremes.scenario_id.eq(c['focal_scenario']) & extremes.subset.eq('response_inner_points')]
    print(json.dumps({'stages': stages, 'focal': focal.replace({np.nan: None}).to_dict(orient='records')}, indent=2))


if __name__ == '__main__':
    main()
