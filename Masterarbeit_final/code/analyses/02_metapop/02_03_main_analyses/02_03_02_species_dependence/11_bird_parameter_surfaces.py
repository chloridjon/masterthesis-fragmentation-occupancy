"""Whole-curve bird matching and exploratory 15-species parameter surfaces."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-bird-surfaces-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(FINAL / 'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from src.tools.final_plot_style import (apply_final_style, export_figure, figure_size,
    FOCAL_SCENARIOS, MISSING, NEUTRAL, roman_font_files)

DEFAULT_CONFIG = FINAL / 'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_parameter_surfaces/11_bird_parameter_surfaces.json'


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256')
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def curve_rmse(predictions, observed):
    """One fixed scenario and parameter profile per complete comparison curve."""
    predictions, observed = np.asarray(predictions), np.asarray(observed)
    if not np.isfinite(predictions).all() or not np.isfinite(observed).all():
        raise ValueError('RMSE needs the same finite H domain for every curve')
    return np.sqrt(np.mean((predictions - observed) ** 2, axis=-1))


def normalized_coordinates(design, decision):
    ell = np.log10(design.half_distance.to_numpy())
    r = np.log10(design.r.to_numpy())
    low = np.array([math.log10(decision['half_distance_min']), decision['log10_r_min']])
    high = np.array([math.log10(decision['half_distance_max']), decision['log10_r_max']])
    points = (np.column_stack([ell, r]) - low) / (high - low)
    assert np.isfinite(points).all() and (points >= 0).all() and (points <= 1).all()
    return points, low, high


def interpolate_scenarios(points, scores, targets):
    """Interpolate each whole-curve score first, then form its lower envelope."""
    tri = Delaunay(points)
    values = LinearNDInterpolator(tri, np.asarray(scores).T)(targets).T
    envelope = np.min(values, axis=0)
    return values, envelope, tri


def load_inputs(c):
    config = json.loads((FINAL / c['lhs_configuration']).read_text())
    audit = json.loads((FINAL / c['lhs_audit']).read_text())
    plausibility = json.loads((FINAL / c['lhs_plausibility']).read_text())
    assert audit['status'] == 'passed' and audit['source_hashes_verified']
    assert plausibility['status'] == 'complete_initial_batch_plausibility_passed'
    assert plausibility['n_nonconverged_rows'] == 0
    assert config['q'] == 0 and len(config['replicates']) == 20
    design_path = FINAL / config['lhs_design']
    data = FINAL / c['lhs_data']
    manifest = json.loads((data / 'manifest.json').read_text())
    assert sha(design_path) == manifest['source_hashes'][config['lhs_design']]
    design = pd.read_csv(design_path, float_precision='round_trip')
    birds_path = FINAL / c['bird_data'] / '11_bird_curves_main_H_grid.csv'
    bird_manifest = json.loads((FINAL / c['bird_data'] / 'manifest.json').read_text())
    for rel, digest in bird_manifest['output_sha256'].items():
        assert sha(FINAL.parent / rel) == digest, rel
    # The reconstruction audit and CSVs remain their own source, with no band weighting.
    birds = pd.read_csv(birds_path, float_precision='round_trip')
    coefficients = pd.read_csv(FINAL / c['bird_data'] / '11_bird_curve_coefficients.csv', float_precision='round_trip')
    species = coefficients.species_id.tolist()
    ids, hs = design.parameter_id.tolist(), config['h_targets']
    scenarios = [s['scenario_id'] for s in FOCAL_SCENARIOS]
    assert len(ids) == 128 and len(species) == 15
    assert len(birds) == 1170 and not birds.duplicated(['species_id', 'H']).any()
    values = np.full((20, 7, len(hs), 128), np.nan)
    input_hashes = {}
    exclusions = []
    for k, rep in enumerate(config['replicates']):
        for j, scenario in enumerate(scenarios):
            path = data / 'trajectories' / f'{scenario}__rep{rep:03d}.csv'
            part = pd.read_csv(path, usecols=['scenario_id', 'replicate', 'H_target', 'parameter_id',
                'occupancy', 'status', 'producer_fingerprint'], float_precision='round_trip')
            assert len(part) == len(hs) * 128 and not part.duplicated(['H_target', 'parameter_id']).any()
            assert part.producer_fingerprint.eq(audit['producer_fingerprint']).all()
            assert part.scenario_id.eq(scenario).all() and part.replicate.eq(rep).all()
            assert set(part.H_target) == set(hs) and set(part.parameter_id) == set(ids)
            valid = part.status.eq('valid')
            assert part.loc[valid, 'occupancy'].between(0, 1).all()
            assert part.loc[~valid, 'occupancy'].isna().all() and not part.status.eq('nonconverged').any()
            values[k, j] = part.pivot(index='H_target', columns='parameter_id', values='occupancy').reindex(index=hs, columns=ids).to_numpy()
            input_hashes[str(path.relative_to(FINAL))] = sha(path)
    count = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    median[count < c['minimum_valid_seeds']] = np.nan
    saved = pd.read_csv(FINAL / c['lhs_summaries'], float_precision='round_trip')
    reconstructed = pd.MultiIndex.from_product([scenarios, hs, ids], names=['scenario_id', 'H_target', 'parameter_id']).to_frame(index=False)
    reconstructed['median'] = median.ravel()
    reconstructed['n_valid'] = count.ravel()
    aligned = saved.merge(reconstructed, on=['scenario_id', 'H_target', 'parameter_id'], suffixes=('_saved', '_recomputed'), validate='one_to_one')
    assert len(aligned) == 7 * len(hs) * 128
    assert np.array_equal(aligned.n_valid_saved, aligned.n_valid_recomputed)
    assert np.array_equal(aligned.median_saved.isna(), aligned.median_recomputed.isna())
    error = float((aligned.median_saved - aligned.median_recomputed).abs().max())
    assert error < 1e-14
    empirical_hs = sorted(birds.H.unique())
    common = [h for h in hs if h in empirical_hs and np.isfinite(median[:, hs.index(h)]).all()]
    h_indices = [hs.index(h) for h in common]
    assert len(common) > 20
    for j, scenario in enumerate(scenarios):
        for i, h in enumerate(hs):
            assert np.ptp(count[j, i]) == 0, 'Landscape eligibility must not depend on species parameters'
            exclusions.append(dict(scenario_id=scenario, H=h, n_valid=int(count[j, i, 0]),
                n_excluded=20-int(count[j, i, 0]), inside_empirical_domain=h in empirical_hs,
                used_in_common_rmse_domain=h in common))
    empirical = birds.pivot(index='species_id', columns='H', values='psi_reconstructed').reindex(index=species, columns=common).to_numpy()
    predictions = median[:, h_indices, :].transpose(0, 2, 1)
    scores = curve_rmse(predictions[None, :, :, :], empirical[:, None, None, :])
    assert scores.shape == (15, 7, 128) and np.isfinite(scores).all()
    independent_error = 0.
    for b in (0, 6, 14):
        for j in range(7):
            for p in (0, 42, 127):
                exact = math.sqrt(math.fsum((float(a)-float(v))**2 for a, v in zip(predictions[j, p], empirical[b])) / len(common))
                independent_error = max(independent_error, abs(exact - scores[b, j, p]))
    assert independent_error < 1e-14
    provenance = {'input_hashes': input_hashes, 'maximum_independent_median_error': error,
                  'maximum_scalar_rmse_error': independent_error,
                  'source_reconstruction_manifest': bird_manifest}
    return design, coefficients, values, scores, empirical, common, h_indices, pd.DataFrame(exclusions), provenance


def source_sensitivity(c, coefficients, values, indices, common, scores):
    digitization = json.loads((FINAL / c['bird_data'] / 'figure2_digitization_manifest.json').read_text())
    band_path = FINAL / c['bird_data'] / '11_bird_curves_main_H_grid_with_figure2_bands.csv'
    assert sha(band_path) == digitization['output_sha256'][str(band_path.relative_to(FINAL.parent))]
    bands = pd.read_csv(FINAL / c['bird_data'] / '11_bird_curves_main_H_grid_with_figure2_bands.csv', float_precision='round_trip')
    other = bands.pivot(index='species_id', columns='H', values='psi_figure2_curve').reindex(index=coefficients.species_id, columns=common).to_numpy()
    assert np.isfinite(other).all()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        prediction = np.nanmedian(values[:, :, indices, :], axis=0).transpose(0, 2, 1)
    alternative = curve_rmse(prediction[None], other[:, None, None])
    rows = []
    final = scores.min(axis=1)
    alternative_final = alternative.min(axis=1)
    for b, species in enumerate(coefficients.species_id):
        rows.append(dict(species_id=species, maximum_rmse_change=float(np.max(abs(scores[b]-alternative[b]))),
            maximum_scenario_envelope_change=float(np.max(abs(final[b]-alternative_final[b]))),
            best_sampled_rmse_table1=float(final[b].min()), best_sampled_rmse_figure2=float(alternative_final[b].min())))
    return alternative, pd.DataFrame(rows)


def seed_sensitivity(values, indices, empirical, scores, coefficients):
    rows = []
    for omit in range(20):
        part = np.delete(values[:, :, indices, :], omit, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            prediction = np.nanmedian(part, axis=0).transpose(0, 2, 1)
        assert np.isfinite(prediction).all()
        other = curve_rmse(prediction[None], empirical[:, None, None])
        for b, species in enumerate(coefficients.species_id):
            rows.append(dict(species_id=species, omitted_seed_block=omit+1,
                maximum_scenario_rmse_change=float(abs(other[b]-scores[b]).max()),
                maximum_envelope_rmse_change=float(abs(other[b].min(axis=0)-scores[b].min(axis=0)).max())))
    return pd.DataFrame(rows)


def make_figure(c, coefficients, grid_ell, grid_r, fields, stem, colour_label, vmax, points=None, scenario=None):
    apply_final_style()
    layout = c['layout']
    fig, axes = plt.subplots(5, 3, figsize=figure_size(layout['height_mm'], layout['width_mm']), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.84, bottom=.13, top=.965, hspace=.44, wspace=.28)
    cmap = plt.get_cmap(c.get('rmse_colormap', 'cividis_r')).copy()
    levels = np.linspace(0, vmax, 31)
    norm = BoundaryNorm(levels, cmap.N)
    for b, ax in enumerate(axes.flat):
        ax.set_facecolor(MISSING)
        filled = ax.contourf(grid_ell, grid_r, np.ma.masked_invalid(fields[b]), levels=levels, norm=norm, cmap=cmap,
            extend='max' if vmax < 1 else 'neither', corner_mask=False)
        # Adjacent vector fill paths must not produce white stitching in PDF readers.
        filled.set_edgecolor('face'); filled.set_linewidth(.12)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1, 512); ax.set_ylim(10**-2.5, 10**5.75)
        ax.set_xticks([1, 16, 256], ['1', '16', '256'])
        ax.set_yticks([.01, 1, 100, 10000], [r'$10^{-2}$', r'$10^{0}$', r'$10^{2}$', r'$10^{4}$'])
        ax.minorticks_off()
        name = coefficients.iloc[b].species_as_published.split()
        short = name[0][0] + '. ' + ' '.join(name[1:])
        ax.text(0, 1.045, f'({chr(97+b)}) {short}', transform=ax.transAxes, fontsize=9.5, fontstyle='italic')
        if points is not None:
            ax.scatter(points.half_distance, points.r, s=3.5, facecolors='white', edgecolors=NEUTRAL, linewidths=.22, alpha=.85)
    color_ax = fig.add_axes([.88, .17, .022, .73])
    bar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(bar, cax=color_ax, label=colour_label, ticks=np.linspace(0, vmax, 6), extend='max' if vmax < 1 else 'neither')
    colorbar.solids.set_edgecolor('face')
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    handles = [Patch(facecolor=MISSING, label='Outside sampled parameter support')]
    if scenario is not None:
        handles.insert(0, Line2D([], [], color=scenario['color'], linestyle=scenario['linestyle'], label=scenario['label']))
    if points is not None:
        handles.append(Line2D([], [], ls='none', marker='o', ms=3, color=NEUTRAL, markerfacecolor='white', label='Sampled profile'))
    fig.legend(handles=handles, loc='lower center', frameon=False, ncol=1)
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def export_checks(stems):
    records = []
    for stem in stems:
        fonts = subprocess.run(['pdffonts', str(stem.with_suffix('.pdf'))], text=True, capture_output=True, check=True).stdout
        assert 'LMRoman8' in fonts and 'LatinModernMath' in fonts
        for line in fonts.splitlines()[2:]:
            parts = line.split()
            if len(parts) >= 8: assert parts[-5] == 'yes', line
        svg = stem.with_suffix('.svg').read_text()
        assert '<text' in svg and '@font-face' in svg and 'data:font/otf;base64' in svg
        assert svg.count('species-placeholder') == 0
        records.append({'stem': str(stem.relative_to(FINAL)), 'pdf_fonts': fonts,
            'editable_svg_text': True, 'embedded_svg_fonts': True})
    return records


def run(config_path):
    started = datetime.now(timezone.utc).isoformat()
    c = json.loads(config_path.read_text())
    data, results = FINAL / c['data_directory'], FINAL / c['results_directory']
    for p in (data, results/'tables', results/'figures', results/'logs'): p.mkdir(parents=True, exist_ok=True)
    design, coefficients, values, scores, empirical, common, indices, exclusions, provenance = load_inputs(c)
    print(f'Checked 140 trajectories; shared H domain has {len(common)} levels.', flush=True)
    decision = json.loads((FINAL / c['parameter_decision']).read_text())
    points, low, high = normalized_coordinates(design, decision)
    x = np.linspace(0, 1, c['grid_n_half_distance']); y = np.linspace(0, 1, c['grid_n_r'])
    xx, yy = np.meshgrid(x, y); targets = np.column_stack([xx.ravel(), yy.ravel()])
    grid_ell = 10 ** (low[0] + x * (high[0]-low[0])); grid_r = 10 ** (low[1] + y * (high[1]-low[1]))
    surfaces, envelopes, hull_error = [], [], 0.
    for b in range(15):
        fields, envelope, tri = interpolate_scenarios(points, scores[b], targets)
        at_sites = LinearNDInterpolator(tri, scores[b].T)(points).T
        hull_error = max(hull_error, float(abs(at_sites-scores[b]).max()))
        assert hull_error < 1e-12
        assert np.array_equal(np.isfinite(envelope), tri.find_simplex(targets) >= 0)
        assert np.nanmin(envelope) >= scores[b].min()-1e-12
        surfaces.append(fields.reshape(7, len(y), len(x))); envelopes.append(envelope.reshape(len(y), len(x)))
    surfaces, envelopes = np.asarray(surfaces), np.asarray(envelopes)
    alternative, source_check = source_sensitivity(c, coefficients, values, indices, common, scores)
    seed_check = seed_sensitivity(values, indices, empirical, scores, coefficients)
    rows, overview, summaries = [], [], []
    for b, species in enumerate(coefficients.species_id):
        for j, scenario in enumerate(FOCAL_SCENARIOS):
            for p, profile in design.iterrows():
                rows.append(dict(species_id=species, species_as_published=coefficients.iloc[b].species_as_published,
                    scenario_id=scenario['scenario_id'], parameter_id=profile.parameter_id, r=profile.r,
                    half_distance=profile.half_distance, rmse=float(scores[b,j,p]),
                    rmse_figure2_central_sensitivity=float(alternative[b,j,p]), n_H=len(common)))
        minima = scores[b].min(axis=0)
        for p, profile in design.iterrows():
            winners = [FOCAL_SCENARIOS[j]['scenario_id'] for j in range(7) if abs(scores[b,j,p]-minima[p])<=1e-12]
            overview.append(dict(species_id=species, parameter_id=profile.parameter_id, r=profile.r,
                half_distance=profile.half_distance, minimum_scenario_rmse=float(minima[p]),
                excess_over_species_sampled_minimum=float(minima[p]-minima.min()),
                best_matching_scenarios=';'.join(winners), n_tied_scenarios=len(winners)))
        summaries.append(dict(species_id=species, species_as_published=coefficients.iloc[b].species_as_published,
            trophic_guild=coefficients.iloc[b].trophic_guild, minimum_sampled_rmse=float(minima.min()),
            maximum_sampled_rmse=float(scores[b].max()), n_parameters=128, n_scenarios=7, n_seeds=20,
            n_H=len(common), H_min=min(common), H_max=max(common),
            maximum_leave_one_seed_out_envelope_change=float(seed_check.loc[seed_check.species_id.eq(species),'maximum_envelope_rmse_change'].max()),
            formal_near_optimal_region='Not inferred; continuous absolute and relative RMSE surfaces shown'))
    outputs = []
    for filename, frame in [('11_bird_scenario_parameter_rmse.csv', pd.DataFrame(rows)),
            ('11_bird_parameter_envelope.csv', pd.DataFrame(overview)), ('11_bird_match_summary.csv', pd.DataFrame(summaries)),
            ('11_curve_domain_and_exclusions.csv', exclusions), ('11_empirical_central_source_sensitivity.csv', source_check),
            ('11_leave_one_seed_out_rmse_changes.csv', seed_check)]:
        path = data / filename; frame.to_csv(path, index=False); outputs.append(path)
    grid_path = data / '11_bird_rmse_display_surfaces.npz'
    np.savez_compressed(grid_path, species_ids=coefficients.species_id.to_numpy(dtype=str),
        scenario_ids=np.array([s['scenario_id'] for s in FOCAL_SCENARIOS]), grid_half_distance=grid_ell, grid_r=grid_r,
        scenario_rmse=surfaces, minimum_scenario_rmse=envelopes,
        excess_rmse=envelopes-scores.min(axis=(1,2))[:,None,None], H_comparison=np.array(common))
    outputs.append(grid_path)
    stems = []
    names = [('11_bird_parameter_rmse_15_panels', envelopes, 'Minimum curve RMSE', 1., None, None),
        ('11_bird_parameter_excess_rmse_15_panels', envelopes-scores.min(axis=(1,2))[:,None,None], 'Excess RMSE above sampled best', .5, None, None),
        ('11_bird_parameter_rmse_sampled_locations', envelopes, 'Minimum curve RMSE', 1., design, None)]
    names += [(f'11_bird_rmse_{s["scenario_id"]}_15_panels', surfaces[:,j], 'Curve RMSE', 1., None, s) for j,s in enumerate(FOCAL_SCENARIOS)]
    for name, fields, label, vmax, sample, scenario in names:
        stem = results/'figures'/name
        make_figure(c, coefficients, grid_ell, grid_r, fields, stem, label, vmax, sample, scenario)
        stems.append(stem)
        print(f'Exported {name}', flush=True)
    figure_checks = export_checks(stems)
    caption = f'''# Figure captions and reading guide

**15-species absolute RMSE overview.** Panels (a)–(o) follow the order of the
15 forest-dependent bird species in Morante-Filho et al. (2021), Figure 2 and
Table 1. Bright yellow colours indicate closer agreement between the reconstructed
empirical site-occupancy curve and median simulated mean equilibrium patch
occupancy. At each parameter combination the colour is the lowest whole-curve
RMSE among the seven fixed scenarios. A single scenario is used across all H
within each RMSE calculation; scenarios are not switched independently at each
habitat level. The entire score surface is shown, with no isolated optimum
marker or binary good-fit cutoff. Similar colours describe similarly matching
response patterns, not a confidence region or true demographic parameters.

The simulations have 128 LHS profiles, q=0, x=0.5, b=1 and 20 matched landscape
seed blocks per scenario. Medians include only landscapes with at least two
patches. All curves use the same {len(common)} available H levels, from
{min(common):g} to {max(common):g}, inside the observed empirical range
0.0675–0.9818; H=0.975 is withheld for all comparisons because Only-Random has
fewer than three valid landscapes. H levels are equally weighted as specified
in Analysis Plan 2.3.2, although this inherited grid is irregular. Valid counts
and exclusions by scenario and H are provided in the domain table. The bird
central responses are reconstructed from rounded Table 1 predictions at H=.2
and .4. Figure 2 centre-line matching is reported separately as a sensitivity
check. Neither digitised empirical bands nor simulation percentile bands are
used as RMSE weights, and no inferential confidence intervals are displayed.

The x-axis shows dispersal half-distance in cells and the y-axis the
extinction-to-colonization ratio, both logarithmic. Piecewise-linear Delaunay
interpolation is performed separately for each scenario after normalising both
log10 coordinates by the selected design bounds; the lower envelope is formed
after interpolation. Light grey marks locations outside the convex hull of
the 128 sampled profiles. No scores are extrapolated to the rectangle corners.
The irregular design can leave narrow fit regions unresolved; interpolated
shape is exploratory rather than an additional set of simulated profiles.
Colours use reversed cividis: low RMSE is yellow and high RMSE is dark blue.
The same 0–1 RMSE scale is used in every absolute and scenario panel.
Names retain the source spelling and genus abbreviations.

**15-species excess-RMSE overview.** Layout and domain are the same. Each
species' lowest RMSE among the sampled profiles and scenarios is subtracted
from its whole-curve scenario envelope. Bright yellow regions therefore identify
similarly matching combinations relative to that species' sampled best,
including broad alternatives. They do not establish an adequate absolute fit:
refer to the absolute-RMSE figure and species summary. The common 0–0.5 scale
has an upper extension for larger excess scores. No isolated optimal point or
arbitrary confidence contour is drawn.

**Sample-location diagnostic.** The absolute overview with the 128 genuinely
simulated parameter profiles marked in each panel. White circles with dark
edges identify the sampled locations and distinguish simulation support from
display interpolation. **Seven scenario figures.** Each repeats the 15-species
layout for one named fixed scenario, using that scenario's whole-curve RMSE
and the common absolute colour scale. A shared legend names the fixed scenario
using the established scenario colour and method line style; these keys name
the scenario and do not encode a second surface quantity.

These are preliminary comparison figures, generated and inspected as
standalone exports. Actual-size manuscript inspection remains pending; no
LaTeX manuscript changes were made. No formal adaptive near-optimal region or
four illustrative species profiles have yet been selected.
'''
    caption_path = results/'figures/11_captions.md'; caption_path.write_text(caption); outputs.append(caption_path)
    readme = f'''# Exploratory bird-response parameter surfaces

Rebuild from the repository root:
`.venv/bin/python -m src.tools.codex_run final-bird-parameter-surfaces`.

The absolute 15-panel overview shows the full RMSE pattern, not one optimal
point. The excess-RMSE overview exposes broad similarly matching regions.
Seven additional figures preserve the scenario-specific surfaces, and a
sample-location diagnostic makes the irregular design explicit. Read the
figure captions for interpolation and ecological interpretation limits.

The tables contain 13,440 species/scenario/profile scores and 1,920 sampled
scenario-envelope scores. The NPZ contains every displayed grid in the same
species/scenario order plus its coordinates and common H domain. Source CSVs
are preserved. Medians were independently recomputed from all 140 stored
trajectories. Scalar RMSE fixtures, interpolation-at-sample recovery, outside-
hull missingness and saved-source checks passed. Leave-one-whole-seed-out and
Table-1-versus-Figure-2 central-response sensitivity tables are descriptive;
they are not empirical or bootstrap confidence intervals.

All comparisons use {len(common)} common H levels ({min(common):g}–{max(common):g}).
No missing landscape is recoded as extinction. Whole-curve minimisation over
scenarios is not pointwise minimisation of occupancy residuals across H.
No confidence region, true species parameters or illustrative bird profiles
are inferred from these exploratory displays.
'''
    (results/'README.md').write_text(readme); (data/'README.md').write_text(readme)
    inputs = [config_path, Path(__file__), FINAL/c['lhs_configuration'], FINAL/c['lhs_summaries'], FINAL/c['lhs_audit'],
        FINAL/c['lhs_plausibility'], FINAL/c['parameter_decision'], FINAL/c['bird_data']/'manifest.json',
        FINAL/c['bird_data']/'11_bird_curves_main_H_grid.csv', FINAL/c['bird_data']/'11_bird_curve_coefficients.csv',
        FINAL/c['bird_data']/'11_bird_curves_main_H_grid_with_figure2_bands.csv', FINAL/'code/base/src/tools/final_plot_style.py']
    input_hashes = {str(p.relative_to(FINAL)):sha(p) for p in inputs}
    input_hashes.update(provenance['input_hashes'])
    outputs += [p for stem in stems for p in [stem.with_suffix('.svg'), stem.with_suffix('.pdf'), stem.with_suffix('.png')]]
    record = {'status':'numerically_and_export_checked_visual_review_pending', 'started_at_utc': started,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'configuration': c,
        'n_species':15, 'n_scenarios':7, 'n_profiles':128, 'n_matched_seed_blocks':20, 'n_common_H':len(common),
        'H_min':min(common), 'H_max':max(common), 'H_common':common,
        'minimum_valid_seed_count_in_comparison':int(exclusions.loc[exclusions.used_in_common_rmse_domain,'n_valid'].min()),
        'maximum_independent_median_error':provenance['maximum_independent_median_error'],
        'maximum_scalar_rmse_error':provenance['maximum_scalar_rmse_error'], 'maximum_interpolation_at_samples_error':hull_error,
        'minimum_sampled_rmse_range':[float(scores.min(axis=(1,2)).min()), float(scores.min(axis=(1,2)).max())],
        'maximum_leave_one_seed_out_envelope_change':float(seed_check.maximum_envelope_rmse_change.max()),
        'maximum_empirical_central_source_rmse_change':float(source_check.maximum_rmse_change.max()),
        'input_sha256':input_hashes, 'output_sha256':{str(p.relative_to(FINAL)):sha(p) for p in outputs},
        'font_files':{str(k):str(p) for k,p in roman_font_files().items()}, 'export_checks':figure_checks,
        'visual_review_confirmed':False, 'formal_confidence_region_inferred':False,
        'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}}
    write_json(results/'logs/11_bird_parameter_surfaces_audit.json', record)
    write_json(data/'manifest.json', {'method':c, 'input_sha256':input_hashes, 'output_sha256':record['output_sha256'],
        'audit':str((results/'logs/11_bird_parameter_surfaces_audit.json').relative_to(FINAL))})
    print(json.dumps({k:record[k] for k in ('status','n_common_H','minimum_sampled_rmse_range','maximum_leave_one_seed_out_envelope_change','maximum_empirical_central_source_rmse_change')}, indent=2))


def confirm(config_path):
    c=json.loads(config_path.read_text()); results=FINAL/c['results_directory']
    path=results/'logs/11_bird_parameter_surfaces_audit.json'; audit=json.loads(path.read_text())
    for rel,digest in {**audit['input_sha256'], **audit['output_sha256']}.items():
        assert sha(FINAL/rel)==digest, rel
    audit.update(status='numerically_and_visually_checked_preliminary', visual_review_confirmed=True,
        visual_review_at_utc=datetime.now(timezone.utc).isoformat(),
        visual_review_scope='All ten PDF pages rendered and inspected, with 15-panel overview and excess-RMSE page inspected separately at full resolution',
        manuscript_actual_size_review='Pending; no manuscript edit requested')
    write_json(path,audit)
    print(audit['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--confirm-visual-review',action='store_true')
    args=parser.parse_args()
    confirm(args.config.resolve()) if args.confirm_visual_review else run(args.config.resolve())
