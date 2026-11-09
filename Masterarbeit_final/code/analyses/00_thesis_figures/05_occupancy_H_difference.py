"""Preview the difference of Figure 5.4 and 5.3: medians at H=.4 minus H=.6.

No simulations, smoothing, new parameter selection or confidence intervals.
Run from repository root with .venv/bin/python Masterarbeit_final/code/analyses/
00_thesis_figures/05_occupancy_H_difference.py. Inputs are the audited fixed-H
surfaces and sampled medians; all new artifacts remain under results.
"""
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-H-difference-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
REL = '02_metapop/02_03_main_analyses/02_03_02_species_dependence'
sys.path.insert(0, str(FINAL / 'code/analyses' / REL))
F = importlib.import_module('11_fixed_H_parameter_surfaces')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS, MISSING

SOURCE = FINAL / f'data/processed/{REL}/fixed_H_parameter_surfaces/conditional_256_8_16'
AUDIT = FINAL / f'results/{REL}/fixed_H_parameter_surfaces/conditional_256_8_16/logs/11_fixed_H_audit.json'
OUT = FINAL / 'results/00_thesis_figures/05_occupancy_H_difference'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    audit = json.loads(AUDIT.read_text())
    assert audit['visual_review_confirmed']
    files = [SOURCE / '11_fixed_H_parameter_surfaces.npz', SOURCE / '11_fixed_H_sampled_responses.csv']
    for p in files:
        assert sha(p) == audit['output_sha256'][str(p.relative_to(FINAL))]
    with np.load(files[0]) as z:
        high, low = [int(np.flatnonzero(np.isclose(z['H'], h, atol=1e-12, rtol=0))[0]) for h in [.6, .4]]
        delta = z['sampled_medians'][:, low] - z['sampled_medians'][:, high]
        fields = z['absolute_fields'][low] - z['absolute_fields'][high]
        ell, rr = z['grid_half_distance'], z['grid_r']
        design = pd.DataFrame(dict(parameter_id=z['parameter_ids'], half_distance=z['sampled_half_distance'], r=z['sampled_r']))
    assert delta.shape == (7, 256) and np.isfinite(delta).all()
    assert np.nanmin(fields) >= -1 and np.nanmax(fields) <= 1
    source = pd.read_csv(files[1], float_precision='round_trip')
    rows = []
    for j, scenario in enumerate(FOCAL_SCENARIOS):
        saved = source[source.scenario_id == scenario['scenario_id']].pivot(index='H', columns='parameter_id', values='median_occupancy').reindex(columns=design.parameter_id)
        np.testing.assert_allclose(saved.loc[.4] - saved.loc[.6], delta[j], atol=1e-14, rtol=0)
        for i, profile in design.iterrows():
            rows.append(dict(scenario_id=scenario['scenario_id'], parameter_id=profile.parameter_id, half_distance=profile.half_distance, r=profile.r,
                             median_H0p600=float(saved.loc[.6].iloc[i]), median_H0p400=float(saved.loc[.4].iloc[i]), difference=float(delta[j, i])))
    # Independent interpolation of sampled differences must match differences
    # of the two saved surfaces wherever both surfaces are supported.
    points, lo, hi = F.B.normalized_coordinates(design, dict(half_distance_min=1, half_distance_max=512, log10_r_min=-2.5, log10_r_max=5.75))
    xx, yy = np.meshgrid(np.log10(ell), np.log10(rr))
    targets = (np.column_stack([xx.ravel(), yy.ravel()]) - lo) / (hi - lo)
    independent = F.B.interpolate_scenarios(points, delta, targets)[0].reshape(fields.shape)
    finite = np.isfinite(fields)
    np.testing.assert_allclose(independent[finite], fields[finite], atol=1e-12, rtol=0)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / '05_sampled_occupancy_differences.csv', index=False)
    np.savez_compressed(OUT / '05_difference_surfaces.npz', grid_half_distance=ell, grid_r=rr, differences=fields, sampled_differences=delta, H_high=.6, H_low=.4)
    apply_final_style()
    fig, axes = plt.subplots(4, 2, figsize=figure_size(185), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.80, top=.96, bottom=.15, hspace=.40, wspace=.18)
    axes.flat[1].set_visible(False)
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad(MISSING)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for j, (location, scenario) in enumerate(zip([0, 2, 3, 4, 5, 6, 7], FOCAL_SCENARIOS)):
        ax = axes.flat[location]
        ax.set_facecolor(MISSING)
        ax.pcolormesh(ell, rr, np.ma.masked_invalid(fields[j]), shading='nearest', cmap=cmap, norm=norm, rasterized=True)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1, 512); ax.set_ylim(10**-2.5, 10**5.75)
        ax.set_xticks([1, 4, 16, 64, 256], ['1', '4', '16', '64', '256'])
        ax.set_yticks([.01, 1, 100, 10000], [r'$10^{-2}$', r'$10^0$', r'$10^2$', r'$10^4$']); ax.minorticks_off()
        ax.text(0, 1.04, f'({chr(97+j)}) ' + scenario['label'], transform=ax.transAxes, fontsize=9.5)
    cax = fig.add_axes([.86, .21, .025, .69])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-1, -.5, 0, .5, 1],
                 label=r'Occupancy difference: $H=0.4$ minus $H=0.6$')
    fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
    fig.supylabel(r'Extinction-to-colonization ratio $r$ (log scale)')
    fig.legend(handles=[Line2D([], [], ls='none', label='Difference of medians; 8 Only-Random / 16 other-scenario seeds'),
                        Patch(facecolor=MISSING, label='Outside sampled parameter support')], loc='lower center', frameon=False)
    stem = OUT / '05_occupancy_H0p400_minus_H0p600'
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)
    checks = F.B.export_checks([stem])
    (OUT / 'README.md').write_text('''# Occupancy difference between H = 0.6 and H = 0.4

Preview for combining thesis Figures 5.3 and 5.4. Each panel shows the median equilibrium mean patch occupancy at H = 0.4 minus that at H = 0.6, at the same scenario and numerical parameter profile. Blue indicates lower occupancy after habitat loss; red indicates higher occupancy. The common scale is -1 to +1 occupancy units (multiply by 100 for percentage points).

The figure uses all 256 fixed profiles and the verified eight Only-Random and 16 directed-scenario seeds. All selected scenario/H combinations have their full planned seed counts and exceed the three-valid-landscape rule. This is a difference of scenario medians across H, not a median of paired trajectory differences. Landscapes with fewer than two patches remain excluded. No confidence interval is calculated.

Display uses piecewise-linear interpolation in normalised logarithmic parameter coordinates, restricted to the sampled convex hull and adopted inner domain. Grey denotes unsupported parameter locations, not zero. No new simulations or parameter profiles are introduced. A near-zero difference can indicate a common occupancy floor or ceiling; consult the original absolute maps. This compares habitat proportions within a scenario, rather than contrasting removal scenarios at fixed habitat proportion.

The difference plot is included in the thesis main text; the original absolute maps are in Appendix B. PDF, editable SVG, PNG, the 1,792 sampled differences, interpolated grids and an audit accompany this preview. Reproduction is documented in the script header.
''')
    (OUT / '05_audit.json').write_text(json.dumps(dict(input_sha256={str(p.relative_to(FINAL)):sha(p) for p in [*files, AUDIT, Path(__file__)]},
        difference='median(H=0.4) - median(H=0.6)', n_profiles=256, n_scenarios=7, grid_shape=list(fields.shape),
        maximum_interpolation_difference=float(np.max(abs(independent[finite]-fields[finite]))), export_checks=checks,
        sampled_ranges={s['scenario_id']:[float(delta[j].min()),float(delta[j].max())] for j,s in enumerate(FOCAL_SCENARIOS)}), indent=2))
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
