"""Supplementary signed areas between audited median occupancy curves."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib
import json
import math
import os
import sys
import warnings

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-integrated-contrast-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(FINAL / 'code/base'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator

U = importlib.import_module('11_conditional_bird_update')
B = U.BASE
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, MISSING, NEUTRAL, FOCAL_SCENARIOS

REL = '02_metapop/02_03_main_analyses/02_03_02_species_dependence'
CONFIG = FINAL / f'config/{REL}/integrated_occupancy_contrast/11_integrated_contrast.json'


def integrate_curves(h, medians):
    """Trapezoidal areas on one strictly increasing, common finite H domain."""
    h, medians = np.asarray(h), np.asarray(medians)
    if h.ndim != 1 or len(h) < 2 or not np.all(np.diff(h) > 0):
        raise ValueError('H must be strictly increasing')
    if medians.shape[-2] != len(h) or not np.isfinite(medians).all():
        raise ValueError('All curves must be finite on the same H domain')
    return np.sum((medians[..., :-1, :] + medians[..., 1:, :]) * np.diff(h)[:, None] / 2, axis=-2)


def seed_medians(values, minimum=3):
    count = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = np.nanmedian(values, axis=0)
    result[count < minimum] = np.nan
    return result


def signed_parts(h, delta):
    """Positive and negative areas of piecewise-linear contrasts, split at zero."""
    h, delta = np.asarray(h), np.asarray(delta)
    a, b = delta[:, :-1], delta[:, 1:]
    width = np.diff(h)[None, :, None]
    positive = (np.maximum(a, 0) + np.maximum(b, 0)) * width / 2
    negative = (np.minimum(a, 0) + np.minimum(b, 0)) * width / 2
    crossing = a * b < 0
    # Endpoint clipping alone overestimates triangles on zero-crossing segments.
    fraction = np.abs(a) / np.where(crossing, np.abs(a) + np.abs(b), 1)
    crosspos = np.where(a > 0, a * fraction, b * (1-fraction)) * width / 2
    crossneg = np.where(a < 0, a * fraction, b * (1-fraction)) * width / 2
    return np.where(crossing, crosspos, positive).sum(axis=1), np.where(crossing, crossneg, negative).sum(axis=1)


def draw(design, ell, rr, fields, limit, stem, sampled=False):
    apply_final_style()
    fig, axes = plt.subplots(3, 2, figsize=figure_size(190), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.80, top=.96, bottom=.17, hspace=.40, wspace=.18)
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad(MISSING)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for j, ax in enumerate(axes.flat):
        ax.set_facecolor(MISSING)
        ax.pcolormesh(ell, rr, np.ma.masked_invalid(fields[j]), cmap=cmap, norm=norm,
                      shading='nearest', rasterized=True)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1, 512); ax.set_ylim(10**-2.5, 10**5.75)
        ax.set_xticks([1, 4, 16, 64, 256], ['1', '4', '16', '64', '256'])
        ax.set_yticks([.01, 1, 100, 10000], [r'$10^{-2}$', r'$10^0$', r'$10^2$', r'$10^4$'])
        ax.minorticks_off()
        rho = [.9, .99, .999][j//2]
        family = 'Edge-Random' if j % 2 == 0 else 'Access-based'
        ax.text(0, 1.04, f'({chr(97+j)}) {family}, '+rf'$\rho={rho}$', transform=ax.transAxes, fontsize=9.5)
        if sampled:
            ax.scatter(design.half_distance, design.r, s=7, facecolors='none', edgecolors=NEUTRAL, linewidths=.4)
    cax = fig.add_axes([.855, .21, .025, .69])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 ticks=np.linspace(-limit, limit, 5), label='Signed area difference (occupancy '+r'$\times\,H$'+')')
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    handles = [Patch(facecolor=MISSING, label='Outside sampled parameter support')]
    if sampled:
        handles.append(Line2D([], [], ls='none', marker='o', ms=3, color=NEUTRAL, markerfacecolor='none', label='Sampled profile'))
    fig.legend(handles=handles, loc='lower center', frameon=False, ncol=1)
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run():
    c = json.loads(CONFIG.read_text())
    out, data, snap = (FINAL / c[k] for k in ['results_directory', 'data_directory', 'source_snapshot'])
    marker = out / 'logs/11_integrated_contrast_audit.json'
    if marker.exists():
        assert json.loads(marker.read_text())['status'] == 'numerically_checked_visual_review_pending', 'Preserve checked outputs; use --confirm-visual-review'
    for p in [out/'logs', out/'tables', out/'figures', out/'review', data]:
        p.mkdir(parents=True, exist_ok=True)
    numeric = json.loads((snap/'11_numerical_audit.json').read_text())
    visual = json.loads((snap/'11_export_audit.json').read_text())
    assert numeric['n_trajectories'] == 104 and numeric['n_parameters'] == 256
    assert visual['visual_review_confirmed'] and visual['numeric_audit_sha256'] == B.sha(snap/'11_numerical_audit.json')
    consumed = [snap/'tables'/name for name in ['11_response_arrays.npz', '11_response_summaries.csv', '11_profile_response_diagnostics.csv']]
    for p in consumed:
        assert numeric['output_sha256'][str(p.relative_to(FINAL))] == B.sha(p), p
    df = pd.read_csv(consumed[1], float_precision='round_trip')
    design = pd.read_csv(consumed[2], float_precision='round_trip')
    h = df.H_target.drop_duplicates().to_numpy()
    assert np.all(np.diff(h) > 0) and len(h) == 88 and len(design) == 256
    with np.load(consumed[0]) as z:
        values, med, counts, planned = z['values'], z['quantiles'][1], z['valid_counts'], z['planned_seed_mask']
    assert values.shape == (16, 7, 88, 256) and planned.sum() == 104
    assert planned[:8].all() and not planned[8:,0].any() and planned[8:,1:].all()
    assert not np.isfinite(values[~planned]).any()
    np.testing.assert_allclose(seed_medians(values), med, atol=1e-14, rtol=0)
    included = np.isfinite(med).all(axis=(0,2))
    hi = h[included]
    assert len(hi) == 85 and hi[0] == .005 and hi[-1] == .925
    # The selected indices are contiguous: never bridge an internal missing H.
    assert np.all(np.diff(np.flatnonzero(included)) == 1)
    q = med[:,included,:]
    areas = integrate_curves(hi, q)
    contrasts = areas[1:] - areas[0]
    delta = q[1:] - q[0]
    positive, negative = signed_parts(hi, delta)
    np.testing.assert_allclose(positive+negative, contrasts, atol=1e-14, rtol=0)
    np.testing.assert_allclose(integrate_curves(hi, delta), contrasts, atol=1e-14, rtol=0)
    # Independent saved CSV + scalar fsum integration, distinct from array code.
    independent = np.empty_like(areas)
    for j, scenario in enumerate(FOCAL_SCENARIOS):
        a = df[df.scenario_id == scenario['scenario_id']].pivot(index='H_target', columns='parameter_id', values='median').reindex(index=hi, columns=design.parameter_id).to_numpy()
        for i in range(256):
            independent[j,i] = math.fsum(float(hi[k+1]-hi[k]) * (float(a[k,i])+float(a[k+1,i])) / 2 for k in range(len(hi)-1))
    error = float(np.max(abs(independent-areas)))
    assert error < 1e-14
    eight = integrate_curves(hi, seed_medians(values[:8])[:,included,:])
    matched_delta = eight[1:] - eight[0]
    stability = []
    omitted = []
    for seed in range(16):
        m = seed_medians(np.delete(values, seed, axis=0))[:,included,:]
        a = integrate_curves(hi, m); diff = a[1:] - a[0]
        if seed >= 8:
            np.testing.assert_array_equal(a[0], areas[0])
        omitted.append(diff)
        for j in range(6):
            stability.append(dict(omitted_seed=seed+1, cohort='shared_1_to_8' if seed < 8 else 'directed_only_9_to_16',
                scenario_id=FOCAL_SCENARIOS[j+1]['scenario_id'], max_absolute_area_contrast_change=float(abs(diff[j]-contrasts[j]).max())))
    omitted = np.array(omitted)
    rows = []
    for j in range(6):
        for i, p in design.iterrows():
            rows.append(dict(parameter_id=p.parameter_id, r=p.r, half_distance=p.half_distance,
                scenario_id=FOCAL_SCENARIOS[j+1]['scenario_id'], n_scenario_seeds=16, n_baseline_seeds=8,
                H_min=hi[0], H_max=hi[-1], n_H=len(hi), scenario_area=areas[j+1,i], baseline_area=areas[0,i],
                signed_area_difference=contrasts[j,i], mean_contrast_over_H=contrasts[j,i]/(hi[-1]-hi[0]),
                positive_contribution=positive[j,i], negative_contribution=negative[j,i],
                matched8_signed_area_difference=matched_delta[j,i],
                maximum_seed_omission_change=float(abs(omitted[:,j,i]-contrasts[j,i]).max())))
    pd.DataFrame(rows).to_csv(data/'11_integrated_occupancy_contrasts.csv', index=False)
    pd.DataFrame(stability).to_csv(out/'tables/11_cohort_seed_omission.csv', index=False)
    domain = df[['scenario_id','H_target','n_planned','n_valid','n_excluded','n_structurally_absent']].drop_duplicates().copy()
    domain['included_in_integral'] = domain.H_target.isin(hi)
    domain.to_csv(out/'tables/11_H_domain_and_exclusions.csv', index=False)
    points, low, high = B.normalized_coordinates(design, dict(half_distance_min=1, half_distance_max=512, log10_r_min=-2.5, log10_r_max=5.75))
    x, y = np.linspace(0,1,c['grid_n_half_distance']), np.linspace(0,1,c['grid_n_r'])
    xx, yy = np.meshgrid(x,y); targets = np.column_stack([xx.ravel(),yy.ravel()])
    fields, _, tri = B.interpolate_scenarios(points, contrasts, targets)
    interpolation_error = float(abs(LinearNDInterpolator(tri, contrasts.T)(points).T-contrasts).max())
    assert interpolation_error < 1e-12
    ell, rr = 10**(low[0]+x*(high[0]-low[0])), 10**(low[1]+y*(high[1]-low[1]))
    ee, rs = np.meshgrid(ell,rr)
    strips = pd.read_csv(FINAL/c['domain_strips'], float_precision='round_trip')
    inside = U.inner_mask(np.log(ee), np.log(rs), strips)
    assert U.inner_mask(np.log(design.half_distance.to_numpy()), design.log_r.to_numpy(), strips).all()
    fields = fields.reshape(6,len(y),len(x)); fields[:,~inside] = np.nan
    limit = math.ceil(float(abs(contrasts).max())/.05) * .05
    assert np.nanmax(abs(fields)) <= limit
    np.savez_compressed(data/'11_integrated_contrast_surfaces.npz', H=hi, parameter_ids=design.parameter_id.to_numpy(str),
        sampled_areas=areas, sampled_contrasts=contrasts, matched8_contrasts=matched_delta, omitted_seed_contrasts=omitted,
        positive_contributions=positive, negative_contributions=negative, grid_half_distance=ell, grid_r=rr,
        contrast_surfaces=fields, adopted_domain_mask=inside, color_limit=limit)
    stems = [out/'figures/11_integrated_occupancy_contrast', out/'figures/11_integrated_occupancy_contrast_sample_locations']
    for k, stem in enumerate(stems):
        draw(design, ell, rr, fields, limit, stem, sampled=bool(k))
    checks = B.export_checks(stems)
    lo, up = np.unravel_index(contrasts.argmin(), contrasts.shape), np.unravel_index(contrasts.argmax(), contrasts.shape)
    note = f'''# Supplementary integrated occupancy contrasts

This author-requested descriptive heatmap supplements the pointwise contrasts in Results B/C. It does not replace their primary role in the analysis plan. No new simulations, parameter selection or confidence intervals were calculated.

For each of 256 retained parameter profiles and six directed scenarios, we integrated the difference between the scenario median equilibrium patch occupancy and the Only-Random median:

$$A_s(\\theta)=\\int_{{0.005}}^{{0.925}} [m_s(H,\\theta)-m_0(H,\\theta)]\\,dH.$$

Here $m_s$ denotes the median across valid simulated landscapes, using 16 planned seeds per directed scenario and 8 for Only-Random. This is the difference of areas under median curves, not a median of paired seed-level areas. The trapezoidal rule used actual distances between the 85 sampled H values on the common contiguous domain. No equal-weight average over the irregular H grid was substituted. No endpoint extrapolation or internal missing-value bridge was used. H=0 and H=1 have undefined responses, and Only-Random at H=.975 has only 2/8 valid landscapes; these levels were excluded for every scenario/profile. At H=.925 the baseline has 5/8 valid landscapes, limiting high-H interpretation. Missing landscapes were never assigned zero occupancy.

Positive values indicate a larger area for the scenario, negative values a smaller area, in occupancy times habitat-proportion units. Dividing by the domain width .92 would give the mean occupancy difference across this H interval; the figure shows the unnormalised area difference requested. Both parameters and H are dimensionless, with dispersal distance expressed in grid cells. Positive and negative contributions can cancel, so a small net area does not imply similar curves. The saved table separates positive and negative areas, splitting linear segments at their zero crossings.

Surfaces use piecewise-linear Delaunay interpolation in unit-normalised log10 parameter coordinates, with no extrapolation beyond the sampled convex hull or the adopted conservative stepwise domain. All panels share the same diverging scale, -{limit:g} to +{limit:g}, with neutral zero; red means positive and blue negative. Grey marks unsupported parameter locations. The diagnostic export also shows all 256 sampled locations. Interpolation patterns do not add parameter resolution or establish biological archetypes.

Independent scalar integration of saved-row CSV medians agreed with array-based calculations to {error:.3g}. Whole-seed omissions preserved the shared cohort 1-8 and directed-only cohort 9-16; the maximum absolute area-contrast change was {abs(omitted-contrasts).max():.6f}. Comparing 8 matched seeds with all 8/16 seeds changed a sampled area contrast by at most {abs(matched_delta-contrasts).max():.6f}. These descriptive checks do not establish confidence intervals or stability of every cell.

The sampled net areas range from {contrasts[lo]:.6f} ({FOCAL_SCENARIOS[lo[0]+1]['scenario_id']}, {design.parameter_id.iloc[lo[1]]}) to {contrasts[up]:.6f} ({FOCAL_SCENARIOS[up[0]+1]['scenario_id']}, {design.parameter_id.iloc[up[1]]}). These extrema were selected after viewing the full design and are descriptive.

## Figure caption

Signed differences in areas under median equilibrium patch-occupancy curves across the species-parameter sample. Panels compare Edge-Random (left) and Access-based (right) with Only-Random, for rho=.9, .99 and .999 from top to bottom. Each sampled parameter profile uses 16 landscape seeds for the directed scenario and 8 for Only-Random, with at least three valid multi-patch landscapes required at every retained H. Areas cover the same 85 habitat proportions from .005 to .925 and use trapezoidal integration with the observed H spacing. Red indicates greater integrated occupancy than Only-Random, blue lower occupancy and neutral colours small net differences. Opposing pointwise differences may cancel. Grey areas lie outside the supported parameter domain; the diagnostic version marks sampled profiles with open circles. Piecewise-linear interpolation uses normalised log10 coordinates, restricted to the sampled convex hull and adopted inner domain. No uncertainty interval or significance threshold is encoded. This supplementary summary complements pointwise scenario contrasts at constant H.

## Reproduction

`.venv/bin/python -m src.tools.codex_run final-integrated-occupancy-contrast`

After rendering and inspecting both PDFs, confirm with `--confirm-visual-review`. Numerical producers and original checked outputs remain unchanged.
'''
    (out/'11_methods_and_caption.md').write_text(note)
    consumed += [CONFIG, Path(__file__), snap/'11_numerical_audit.json', snap/'11_export_audit.json', FINAL/c['domain_strips'],
        Path(B.__file__), Path(U.__file__), FINAL/'code/base/src/tools/final_plot_style.py']
    outputs = [p for p in data.rglob('*') if p.is_file()] + [p for folder in [out/'tables',out/'figures'] for p in folder.rglob('*') if p.is_file()] + [out/'11_methods_and_caption.md']
    B.write_json(marker, dict(status='numerically_checked_visual_review_pending', visual_review_confirmed=False,
        created_utc=datetime.now(timezone.utc).isoformat(), minimum_valid_seeds=3, n_profiles=256, n_scenarios=6,
        H_min=float(hi[0]), H_max=float(hi[-1]), n_H=len(hi), color_limit=limit,
        minimum_sampled_contrast=float(contrasts.min()), maximum_sampled_contrast=float(contrasts.max()),
        maximum_seed_omission_change=float(abs(omitted-contrasts).max()), maximum_matched8_to_full_change=float(abs(matched_delta-contrasts).max()),
        maximum_independent_integration_error=error, maximum_interpolation_recovery_error=interpolation_error,
        bootstrap_intervals_calculated=False, export_checks=checks,
        input_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in consumed},
        output_sha256={str(p.relative_to(FINAL)):B.sha(p) for p in outputs}))
    print(json.dumps(dict(n_profiles=256,n_H=len(hi),contrast_min=float(contrasts.min()),contrast_max=float(contrasts.max()),independent_error=error,color_limit=limit)))


def confirm():
    c = json.loads(CONFIG.read_text()); out = FINAL/c['results_directory']
    p = out/'logs/11_integrated_contrast_audit.json'; a = json.loads(p.read_text())
    for name, digest in {**a['input_sha256'], **a['output_sha256']}.items():
        assert B.sha(FINAL/name) == digest, name
    review = out/'review/11_actual_visual_review.json'
    receipt = json.loads(review.read_text())
    assert receipt['both_pdfs_rendered_and_viewed']
    a.update(status='numerically_and_visually_checked_supplementary', visual_review_confirmed=True,
        visual_review_receipt_sha256=B.sha(review), visual_review_utc=datetime.now(timezone.utc).isoformat())
    B.write_json(p,a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-visual-review', action='store_true')
    args = parser.parse_args()
    confirm() if args.confirm_visual_review else run()
