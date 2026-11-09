"""Update existing bird plots using the audited, unequal 8/16 replication."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import warnings
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-bird-8-16-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(FINAL / 'code/base'))
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
U = importlib.import_module('11_conditional_bird_update')
B = U.BASE
from src.tools.final_plot_style import FOCAL_SCENARIOS
CONFIG = FINAL / 'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_parameter_surfaces/conditional_256_8_16/11_bird_update.json'


def predictions(values, minimum=3):
    count = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    median[count < minimum] = np.nan
    return median.transpose(0, 2, 1)


def verified(record):
    for name, digest in {**record['input_sha256'], **record['output_sha256']}.items():
        assert B.sha(FINAL / name) == digest, name


def confirm():
    c = json.loads(CONFIG.read_text())
    p = FINAL / c['results_directory'] / 'logs/11_bird_update_audit.json'
    a = json.loads(p.read_text()); verified(a)
    a.update(status='numerically_and_visually_checked_preliminary', visual_review_confirmed=True,
             visual_review_at_utc=datetime.now(timezone.utc).isoformat())
    B.write_json(p, a)


def run():
    c = json.loads(CONFIG.read_text())
    out, data = FINAL / c['results_directory'], FINAL / c['data_directory']
    for p in [data, out / 'tables', out / 'figures', out / 'logs']:
        p.mkdir(parents=True, exist_ok=True)
    marker = out / 'logs/11_bird_update_audit.json'
    assert not marker.exists(), 'Use --confirm-visual-review for existing exports'
    snap = FINAL / c['source_snapshot']
    numeric_path, visual_path = snap / '11_numerical_audit.json', snap / '11_export_audit.json'
    audit = json.loads(numeric_path.read_text()); verified(audit)
    visual = json.loads(visual_path.read_text())
    assert visual['visual_review_confirmed'] and visual['numeric_audit_sha256'] == B.sha(numeric_path)
    assert audit['n_trajectories'] == 104 and audit['n_parameters'] == 256
    for name, digest in visual['output_sha256'].items():
        assert B.sha(FINAL / name) == digest, name
    design = pd.read_csv(snap / 'tables/11_profile_response_diagnostics.csv', float_precision='round_trip')
    source_summary = pd.read_csv(snap / 'tables/11_response_summaries.csv', float_precision='round_trip')
    hs = source_summary.H_target.drop_duplicates().to_list()
    with np.load(snap / 'tables/11_response_arrays.npz') as z:
        values, q, counts, planned = z['values'], z['quantiles'], z['valid_counts'], z['planned_seed_mask']
    assert values.shape == (16, 7, 88, 256) and planned.sum() == 104
    assert planned[:8].all() and not planned[8:, 0].any() and planned[8:, 1:].all()
    assert not np.isfinite(values[~planned]).any()
    assert list(audit['scenario_seed_targets'].values()) == [8, 16, 16, 16, 16, 16, 16]
    bird = FINAL / c['bird_data']
    for manifest in ['manifest.json', 'figure2_digitization_manifest.json']:
        for name, digest in json.loads((bird / manifest).read_text())['output_sha256'].items():
            assert B.sha(FINAL.parent / name) == digest, name
    coeff = pd.read_csv(bird / '11_bird_curve_coefficients.csv', float_precision='round_trip')
    birds = pd.read_csv(bird / '11_bird_curves_main_H_grid.csv', float_precision='round_trip')
    bands = pd.read_csv(bird / '11_bird_curves_main_H_grid_with_figure2_bands.csv', float_precision='round_trip')
    common = [h for h in hs if h in set(birds.H) and np.isfinite(q[1, :, hs.index(h), :]).all()]
    indices = [hs.index(h) for h in common]
    assert len(coeff) == 15 and len(common) == 77 and min(common) == .07 and max(common) == .925
    observed = birds.pivot(index='species_id', columns='H', values='psi_reconstructed').reindex(index=coeff.species_id, columns=common).to_numpy()
    alternate = bands.pivot(index='species_id', columns='H', values='psi_figure2_curve').reindex(index=coeff.species_id, columns=common).to_numpy()
    prediction = q[1][:, indices, :].transpose(0, 2, 1)
    np.testing.assert_allclose(predictions(values[:, :, indices, :]), prediction, atol=1e-14, rtol=0)
    scores = B.curve_rmse(prediction[None], observed[:, None, None])
    alternative = B.curve_rmse(prediction[None], alternate[:, None, None])
    error = 0.
    for b in range(15):
        for j in range(7):
            for i in [0, 130, 255]:
                scalar = math.sqrt(math.fsum((float(x)-float(y))**2 for x, y in zip(prediction[j, i], observed[b])) / len(common))
                error = max(error, abs(scalar-scores[b, j, i]))
    assert error < 1e-14
    five = B.curve_rmse(predictions(values[:5, :, indices, :])[None], observed[:, None, None])
    eight = B.curve_rmse(predictions(values[:8, :, indices, :])[None], observed[:, None, None])
    old_c = json.loads(U.CONFIG.read_text())
    old_npz = FINAL / old_c['data_directory'] / '11_bird_rmse_display_surfaces.npz'
    old_audit_path = FINAL / old_c['results_directory'] / 'logs/11_bird_update_audit.json'
    old_audit = json.loads(old_audit_path.read_text())
    assert old_audit['visual_review_confirmed'] and B.sha(old_npz) == old_audit['output_sha256'][str(old_npz.relative_to(FINAL))]
    with np.load(old_npz) as z:
        np.testing.assert_array_equal(z['H_comparison'], common)
        np.testing.assert_allclose(z['sampled_scores'], five, atol=1e-14, rtol=0)
    print('Checked 8/16 medians, 77 H levels and exact retained-five-score reconstruction.', flush=True)
    # Each omission preserves the two partially paired cohorts and fixed common H grid.
    stability = []
    for omit in range(16):
        other = B.curve_rmse(predictions(np.delete(values[:, :, indices, :], omit, axis=0))[None], observed[:, None, None])
        if omit >= 8:
            np.testing.assert_array_equal(other[:, 0], scores[:, 0])
        for b, species in enumerate(coeff.species_id):
            stability.append(dict(species_id=species, omitted_seed=omit+1,
                cohort='shared_1_to_8' if omit < 8 else 'directed_only_9_to_16',
                maximum_scenario_rmse_change=float(abs(other[b]-scores[b]).max()),
                maximum_minimum_rmse_change=float(abs(other[b].min(axis=0)-scores[b].min(axis=0)).max()),
                maximum_summed_rmse_change=float(abs(other[b].sum(axis=0)-scores[b].sum(axis=0)).max())))
    points, low, high = B.normalized_coordinates(design, dict(half_distance_min=1, half_distance_max=512, log10_r_min=-2.5, log10_r_max=5.75))
    x, y = np.linspace(0, 1, c['grid_n_half_distance']), np.linspace(0, 1, c['grid_n_r'])
    xx, yy = np.meshgrid(x, y); targets = np.column_stack([xx.ravel(), yy.ravel()])
    ell, rr = 10**(low[0]+x*(high[0]-low[0])), 10**(low[1]+y*(high[1]-low[1]))
    ee, rs = np.meshgrid(ell, rr)
    strips = pd.read_csv(FINAL / c['domain_strips'], float_precision='round_trip')
    inside = U.inner_mask(np.log(ee), np.log(rs), strips)
    assert U.inner_mask(np.log(design.half_distance.to_numpy()), design.log_r.to_numpy(), strips).all()
    surfaces = []; interpolation_error = 0.; linearity_error = 0.
    for b in range(15):
        fields, _, tri = B.interpolate_scenarios(points, scores[b], targets)
        recovered = LinearNDInterpolator(tri, scores[b].T)(points).T
        interpolation_error = max(interpolation_error, float(abs(recovered-scores[b]).max()))
        fields = fields.reshape(7, len(y), len(x)); fields[:, ~inside] = np.nan
        check = LinearNDInterpolator(tri, scores[b].sum(axis=0))(targets).reshape(len(y), len(x)); check[~inside] = np.nan
        assert np.array_equal(np.isnan(fields.sum(axis=0)), np.isnan(check))
        linearity_error = max(linearity_error, float(np.nanmax(abs(fields.sum(axis=0)-check))))
        surfaces.append(fields)
    assert interpolation_error < 1e-12 and linearity_error < 1e-12
    surfaces = np.array(surfaces); minimum, summed = surfaces.min(axis=1), surfaces.sum(axis=1)
    rows, aggregate, best = [], [], []
    for b, species in enumerate(coeff.species_id):
        minsite, sums = scores[b].min(axis=0), scores[b].sum(axis=0)
        j, i = np.unravel_index(scores[b].argmin(), scores[b].shape)
        si = int(sums.argmin())
        best.append(dict(species_id=species, species_as_published=coeff.iloc[b].species_as_published,
            trophic_guild=coeff.iloc[b].trophic_guild, minimum_rmse=float(scores[b, j, i]),
            best_scenario=FOCAL_SCENARIOS[j]['scenario_id'], best_parameter_id=design.parameter_id.iloc[i],
            best_r=design.r.iloc[i], best_half_distance=design.half_distance.iloc[i],
            minimum_summed_rmse=float(sums[si]), summed_best_parameter_id=design.parameter_id.iloc[si],
            initial5_best_rmse=float(five[b].min()), matched8_best_rmse=float(eight[b].min()),
            maximum_initial5_score_change=float(abs(scores[b]-five[b]).max()),
            maximum_matched8_score_change=float(abs(scores[b]-eight[b]).max())))
        for i, item in design.iterrows():
            aggregate.append(dict(species_id=species, parameter_id=item.parameter_id, r=item.r, half_distance=item.half_distance,
                minimum_scenario_rmse=minsite[i], excess_minimum_rmse=minsite[i]-minsite.min(),
                sum_scenario_rmse=sums[i], mean_scenario_rmse=sums[i]/7, excess_summed_rmse=sums[i]-sums.min()))
            for j, scenario in enumerate(FOCAL_SCENARIOS):
                rows.append(dict(species_id=species, scenario_id=scenario['scenario_id'], parameter_id=item.parameter_id,
                    r=item.r, half_distance=item.half_distance, n_planned_seeds=8 if j == 0 else 16,
                    n_H=len(common), rmse=scores[b,j,i], rmse_initial5=five[b,j,i], rmse_matched8=eight[b,j,i],
                    rmse_figure2_central_sensitivity=alternative[b,j,i]))
    pd.DataFrame(rows).to_csv(data / '11_bird_scenario_parameter_rmse.csv', index=False)
    pd.DataFrame(aggregate).to_csv(data / '11_bird_aggregated_rmse.csv', index=False)
    pd.DataFrame(best).to_csv(out / 'tables/11_best_sampled_matches.csv', index=False)
    pd.DataFrame(stability).to_csv(out / 'tables/11_cohort_seed_omission_rmse.csv', index=False)
    domain = pd.DataFrame([dict(scenario_id=s['scenario_id'], H=float(h), n_planned=8 if j == 0 else 16,
        n_valid=int(counts[j,k]), n_excluded=(8 if j == 0 else 16)-int(counts[j,k]),
        n_structurally_absent=8 if j == 0 else 0, used_in_rmse=h in common)
        for j, s in enumerate(FOCAL_SCENARIOS) for k, h in enumerate(hs)])
    domain.to_csv(out / 'tables/11_domain_and_exclusions.csv', index=False)
    np.savez_compressed(data / '11_bird_rmse_display_surfaces.npz', species_ids=coeff.species_id.to_numpy(str),
        grid_half_distance=ell, grid_r=rr, scenario_rmse=surfaces, minimum_scenario_rmse=minimum,
        summed_scenario_rmse=summed, adopted_domain_mask=inside, H_comparison=np.array(common),
        sampled_scores=scores, figure2_scores=alternative, sampled_scores_initial5=five, sampled_scores_matched8=eight)
    # Keep the established figure family, limits and reusable plotting functions.
    names = [('11_bird_minimum_rmse', minimum, 'Minimum curve RMSE', 1, None, None),
        ('11_bird_excess_minimum_rmse', minimum-scores.min(axis=(1,2))[:,None,None], 'Excess RMSE above sampled best', .5, None, None),
        ('11_bird_minimum_sample_locations', minimum, 'Minimum curve RMSE', 1, design, None)]
    names += [(f'11_bird_{s["scenario_id"]}', surfaces[:,j], 'Curve RMSE', 1, None, s) for j,s in enumerate(FOCAL_SCENARIOS)]
    names += [('11_bird_summed_rmse', summed, 'Sum of seven curve RMSEs', 7, None, None),
        ('11_bird_excess_summed_rmse', summed-scores.sum(axis=1).min(axis=1)[:,None,None], 'Excess sum above sampled best', 7, None, None),
        ('11_bird_sum_sample_locations', summed, 'Sum of seven curve RMSEs', 7, design, None)]
    stems=[]; plot_c={**c, 'layout':dict(width_mm=160, height_mm=250)}
    for name, fields, label, vmax, sites, scenario in names:
        stem=out / 'figures' / name; B.make_figure(plot_c, coeff, ell, rr, fields, stem, label, vmax, sites, scenario)
        stems.append(stem); print(f'Exported {name}', flush=True)
    stems.append(U.overlays(coeff, birds, common, prediction, scores, bands, out))
    checks = B.export_checks(stems)
    ordered = [stems[0], stems[10], stems[1], stems[11], stems[-1], stems[2], stems[12]] + stems[3:10]
    assert len(ordered) == 14 and len(set(ordered)) == 14
    collection = out / '11_bird_plot_collection_8_16.pdf'
    subprocess.run(['pdfunite', *[str(s.with_suffix('.pdf')) for s in ordered], str(collection)], check=True)
    info = subprocess.run(['pdfinfo', str(collection)], capture_output=True, text=True, check=True).stdout
    assert any(line.split() == ['Pages:', '14'] for line in info.splitlines())
    st, bt = pd.DataFrame(stability), pd.DataFrame(best)
    captions = '''# Updated bird plots: 8/16 seeds

All panels show the same 15 species, 256 fixed profiles and 77 equally weighted
H levels (.07–.925). Landscape medians use 8 Only-Random seeds or 16 seeds in
each other scenario, with at least 3 valid multi-patch landscapes. Baseline
seeds9–16 were not planned. Scenario errors receive equal weight, irrespective
of replicate counts. All profiles remain included; model data are unchanged.

The absolute-error overview takes the minimum over seven separately calculated
whole-curve scenario RMSEs. A fixed scenario is used for a whole curve; scenarios
are not switched independently at each H. The sum overview literally sums
the seven whole-curve RMSEs; its mean is one seventh of that sum. Excess-error
views subtract each species' best sampled value and do not define confidence
regions, biological archetypes or identifiable demographic parameters.

Surfaces use piecewise-linear Delaunay interpolation in normalised log10(r)
and log10(dispersal half-distance). They are masked outside the sampled hull
and the exact adopted nonrectangular inner domain. Grey denotes unsupported
locations. Interpolation adds no simulations and sampling artifacts may remain.
White circles in the diagnostic versions locate the actual sampled profiles.
Low RMSE is yellow (reversed cividis). Fixed scales are0–1 for absolute RMSE,
0–.5 for excess minimum and0–7 for sum/excess sum. Separate scenario maps
retain the same scales. Species names follow the published labels.

The overlay compares the Table1 reconstruction (black) with the best sampled
model median (scenario colour; Access dashed). Its grey bird band is digitised
from Figure2 and treated as an assumed pointwise95% CI with unverified coverage
and construction. Missing/occluded limits remain absent. The band is not a
model CI, RMSE weight or recovered joint uncertainty. Figure2 central curves
are a separate sensitivity calculation. Simulated patch occupancy and empirical
forest-site occupancy within600m have different observation units. Fits are
descriptive in-sample shape comparisons, not biological validation.

The omission table preserves shared seed clusters1–8 across seven scenarios
and directed-only clusters9–16 across six. No new bootstrap CIs are calculated.
The initial5 and matched8 comparison uses identical profiles/H levels; the
initial5 scores independently reproduce the preserved five-seed plot data.
The separate20-seed Results-B reference is preserved and is not reduced to
8/16 or reselected. Standalone PDF visual review precedes confirmation;
manuscript embedding review remains a later task.
'''
    (out / '11_figure_captions.md').write_text(captions)
    links='\n'.join(f'{i}. [{s.name}](figures/{s.name}.pdf)' for i,s in enumerate(ordered,1))
    note=f'''# Bird plots updated from five to8/16 seeds

Only-Random uses8 seeds; each of the other six scenarios uses16. All256 fixed
profiles and15 species remain included. The shared comparison domain contains
77 H levels from.07 to.925. Best sampled whole-curve RMSE ranges from
{bt.minimum_rmse.min():.4f} to{bt.minimum_rmse.max():.4f}. Maximum whole-seed
omission RMSE change is{st.maximum_scenario_rmse_change.max():.4f}. These are
descriptive sample sensitivity checks, not confidence intervals.

Both minimum- and summed-scenario surfaces, excess-error views, sampled-point
diagnostics, all seven scenario maps and the15-species overlay were updated.
Full definitions and limitations are in [the captions](11_figure_captions.md).
The original five-seed figures and separate20-seed Results-B reference remain
preserved. No new landscape simulations, biological calibration or bootstrap
intervals were calculated. Plot confirmation requires actual visual review.

[Complete14-page plot collection](11_bird_plot_collection_8_16.pdf)

## Page index and individual plots

{links}
'''
    (out / 'README.md').write_text(note)
    inputs=[CONFIG, Path(__file__), Path(U.__file__), Path(B.__file__), numeric_path, visual_path,
        snap/'tables/11_response_arrays.npz', snap/'tables/11_response_summaries.csv',
        snap/'tables/11_profile_response_diagnostics.csv', FINAL/c['domain_strips'], old_audit_path, old_npz,
        bird/'manifest.json', bird/'figure2_digitization_manifest.json', bird/'11_bird_curve_coefficients.csv',
        bird/'11_bird_curves_main_H_grid.csv', bird/'11_bird_curves_main_H_grid_with_figure2_bands.csv']
    outputs=[p for root in [data,out] for p in root.rglob('*') if p.is_file() and p!=marker]
    record=dict(status='numeric_and_export_checks_passed_visual_pending', n_species=15, n_profiles=256,
        scenario_seed_targets=audit['scenario_seed_targets'], n_scenarios=7, n_H=77, H_min=.07, H_max=.925,
        n_score_rows=len(rows), maximum_scalar_rmse_error=error, maximum_interpolation_recovery_error=interpolation_error,
        maximum_sum_linearity_error=linearity_error, best_rmse_range=[float(bt.minimum_rmse.min()),float(bt.minimum_rmse.max())],
        maximum_seed_omission_rmse_change=float(st.maximum_scenario_rmse_change.max()),
        n_individual_figures=14, n_collection_pages=14, visual_review_confirmed=False, bootstrap_intervals_calculated=False,
        old_results_preserved=True, export_checks=checks,
        input_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in inputs},
        output_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in outputs})
    B.write_json(marker,record)
    print(json.dumps({k:record[k] for k in ['status','best_rmse_range','maximum_seed_omission_rmse_change']},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--confirm-visual-review',action='store_true')
    args=parser.parse_args();confirm() if args.confirm_visual_review else run()
