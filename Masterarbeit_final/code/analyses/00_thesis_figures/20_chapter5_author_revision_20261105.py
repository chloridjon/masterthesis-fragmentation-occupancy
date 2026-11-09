"""Presentation changes requested in the annotated Chapter-5 Revision 6.

Reuses saved arrays and summaries. No model solves, correlations, bootstrap
draws, parameter selection or new interpolation are performed.
"""
from pathlib import Path
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-chapter5-author-revision')
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(ROOT / 'code/base'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, MISSING

REGIONS = importlib.import_module('13_metric_regions')
BIRDS = importlib.import_module('04_selected_bird_sum_surfaces')
OUT = ROOT / 'results/00_thesis_figures/20_chapter5_author_revision_20261105'
F = OUT / 'figures'
INPUTS = {}
BLUE = '#377A9F'
REL = '02_metapop/02_03_main_analyses/02_03_03_metric_associations/conditional_256_8_16'


def read(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def emit(fig, name):
    export_figure(fig, F / name)
    fig.savefig(F / (name + '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def regional_curves():
    source = ROOT / 'results/00_thesis_figures/17_thematic_association_draft_20261104'
    curves = pd.read_csv(read(source / 'tables/17_full_regional_curves.csv'), float_precision='round_trip')
    original = pd.read_csv(read(ROOT / 'results/00_thesis_figures/15_regions_ABCD_20261104/tables/15_region_by_H.csv'), float_precision='round_trip')
    pd.testing.assert_frame_equal(curves, original)
    curves.to_csv(OUT / 'tables/20_regional_curves_unchanged.csv', index=False)
    for metrics, name in [([0, 1, 2], '20_structure_regional_curves'), ([3, 4, 5], '20_connectivity_regional_curves')]:
        fig, axes = plt.subplots(4, 3, figsize=figure_size(185), sharex=True, sharey=True)
        fig.subplots_adjust(left=.16, right=.98, top=.96, bottom=.13, hspace=.44, wspace=.24)
        for row, region in enumerate('ABCD'):
            axes[row, 0].text(-.33, .5, f'Region\n{region}', transform=axes[row, 0].transAxes,
                              ha='center', va='center', fontsize=9.5, weight='bold')
            for col, m in enumerate(metrics):
                ax = axes[row, col]
                g = curves[(curves.region == region) & (curves.metric == REGIONS.METRICS[m])].sort_values('H')
                ax.axhline(0, color='#999999', lw=.5)
                ax.fill_between(g.H, g.q05, g.q95, color=BLUE, alpha=.18, lw=0)
                ax.plot(g.H, g['median'], color=BLUE, lw=1.6)
                assert len(ax.lines) == 2  # One median and the zero reference; no extra curves.
                ax.set(xlim=(1, 0), ylim=(-1, 1), xticks=[1, .5, 0], yticks=[-1, 0, 1])
                title = REGIONS.LABELS[m]
                if m == 5:
                    title = 'Complete-MST\n' + r'$D_{50}$'
                ax.text(0, 1.06, f'({chr(97+row*3+col)}) {title}', transform=ax.transAxes, fontsize=9.5)
        fig.supxlabel(r'Forest proportion $H$')
        fig.supylabel(r'Scenario-balanced Spearman $r_s$')
        fig.legend(handles=[Line2D([], [], color=BLUE, label='Median across defined profiles'),
                            Patch(facecolor=BLUE, alpha=.18, label='5th–95th percentiles across profiles')],
                   loc='lower center', ncol=1, frameon=False)
        emit(fig, name)


def parameter_maps():
    source = ROOT / 'results/00_thesis_figures/17_thematic_association_draft_20261104'
    audit = json.loads(read(source / 'logs/17_numeric_audit.json').read_text())
    array_path = ROOT / 'data/processed' / REL / '12_parameter_surfaces.npz'
    read(array_path)
    assert INPUTS[str(array_path.relative_to(ROOT))] == audit['input_sha256'][str(array_path.relative_to(ROOT))]
    with np.load(array_path, allow_pickle=False) as z:
        ell, rr, fields = z['grid_half_distance'], z['grid_r'], z['median_signed']
    config = json.loads(read(ROOT / 'config/00_thesis_figures/15_regions_ABCD_20261104.json').read_text())
    boundary = pd.read_csv(read(ROOT / config['C_boundary']), float_precision='round_trip')
    cut = boundary.proposed_C_lower_r.to_numpy()
    np.testing.assert_array_equal(ell, boundary.half_distance)

    def boundaries(ax):
        for reg in ['A', 'B']:
            rule = config['regions'][reg]
            field = fields[REGIONS.METRICS.index(rule['metric'])]
            mask = REGIONS.region_mask(field, ell[None, :], rule)
            ax.contour(ell, rr, np.where(np.isfinite(field), mask.astype(float), np.nan),
                       levels=[.5], colors='#333333', linewidths=.65, linestyles='--', alpha=.8)
        ci = np.argmin(abs(rr[:, None] - cut[None, :]), axis=0)
        supported = np.isfinite(fields[1])[ci, np.arange(len(ell))]
        ax.plot(ell, np.where(supported, cut, np.nan), color='#333333', lw=.65, ls='--', alpha=.8)
        nn = np.where((ell[None, :] > 16) & (rr[:, None] < cut[None, :]), fields[3], np.nan)
        ax.contour(ell, rr, nn, levels=[-.3], colors='#333333', linewidths=.65, linestyles='--', alpha=.8)
        for reg, (x, y) in dict(A=(3.6, .36), B=(21, 550), C=(13, 85000), D=(385, 80)).items():
            ax.text(x, y, reg, ha='center', va='center', weight='bold', fontsize=8.5,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.5))

    for indices, name in [([0, 1, 2], '20_structure_maps_regions'), ([3, 4, 5], '20_connectivity_maps_regions')]:
        fig, axes = plt.subplots(1, 3, figsize=figure_size(77), sharex=True, sharey=True)
        fig.subplots_adjust(left=.13, right=.86, top=.84, bottom=.30, wspace=.22)
        for letter, (ax, m) in enumerate(zip(axes, indices)):
            im = ax.pcolormesh(ell, rr, np.ma.masked_invalid(fields[m]), cmap='RdBu_r',
                               vmin=-1, vmax=1, shading='nearest', rasterized=True)
            REGIONS.map_axes(ax)
            ax.text(0, 1.08, f'({chr(97+letter)}) ' + REGIONS.LABELS[m], transform=ax.transAxes)
            boundaries(ax)
        cb = fig.add_axes([.90, .34, .018, .46])
        fig.colorbar(im, cax=cb, ticks=[-1, 0, 1], label=r'Median signed $r_s$')
        fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
        fig.supylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
        fig.legend(handles=[Patch(facecolor=MISSING, label='Outside sampled support or undefined'),
                            Line2D([], [], color='#333333', ls='--', lw=.65, label='Exploratory regions A/B/C/D')],
                   loc='lower center', ncol=1, frameon=False)
        emit(fig, name)


def forest_parameter_contrasts():
    path = ROOT / ('data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/'
                   'habitat_parameter_contrast/conditional_256_8_16/interpolated/11_interpolated_display_arrays.npz')
    with np.load(read(path), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    # Same index 5 and display arrays as panels b in the two removed selected grids.
    h = arrays['H_display']
    he = np.r_[h[0], (h[:-1]+h[1:])/2, h[-1]]
    fig, axes = plt.subplots(1, 2, figsize=figure_size(87))
    fig.subplots_adjust(left=.13, right=.83, bottom=.25, top=.87, wspace=.55)
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad(MISSING)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for i, (ax, parameter) in enumerate(zip(axes, ['r', 'half_distance'])):
        field = arrays[f'{parameter}_interpolated_contrast'][5].T
        y = 10**arrays[f'{parameter}_log_display_edges']
        ax.set_facecolor(MISSING)
        ax.pcolormesh(he, y, np.ma.masked_invalid(field), shading='auto', cmap=cmap,
                      norm=norm, rasterized=True)
        ax.set_yscale('log'); ax.set_xlim(1, 0); ax.set_xticks([1, .5, 0])
        if parameter == 'r':
            ax.set_yticks([.01, 1, 100, 10000], [r'$10^{-2}$', r'$10^0$', r'$10^2$', r'$10^4$'])
            ax.set_ylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
            descriptor = 'Extinction-to-colonisation ratio'
        else:
            ax.set_yticks([1, 4, 16, 64, 256], ['1', '4', '16', '64', '256'])
            ax.set_ylabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
            descriptor = 'Dispersal half-distance'
        ax.minorticks_off()
        ax.text(0, 1.06, f'({chr(97+i)}) {descriptor}', transform=ax.transAxes, fontsize=9.5)
    cax = fig.add_axes([.89, .29, .02, .53])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 ticks=[-1, -.5, 0, .5, 1], label='Mean occupancy contrast')
    fig.supxlabel(r'Forest proportion $H$')
    fig.legend(handles=[Patch(facecolor=MISSING, label='Outside sampled support')], loc='lower center', frameon=False)
    emit(fig, '20_access_H_parameter_contrasts')
    shutil.copy2(path, OUT / 'tables/20_unchanged_H_parameter_arrays.npz')


def occupancy_maps():
    source = ROOT / 'results/00_thesis_figures/11_occupancy_self_colonisation_example_20261102'
    fields = {}
    table = pd.read_csv(read(source / '11_patch_probabilities.csv'), float_precision='round_trip')
    for h in [.6, .4]:
        with np.load(read(source / f'11_example_H{h:.1f}.npz'), allow_pickle=False) as z:
            fields[h] = {key: z[key].copy() for key in z.files}
        for q in [0, 1]:
            saved = table[(table.H == h) & (table.q == q)].sort_values('patch_label')
            np.testing.assert_allclose(fields[h][f'q{q}_probabilities'], saved.probability, atol=1e-15, rtol=0)
    vmax = max(fields[h][f'q{q}_probabilities'].max() for h in fields for q in [0, 1])
    fig = plt.figure(figsize=figure_size(160))
    cmap = LinearSegmentedColormap.from_list('patch_occupancy_greens',
                 ['#F7FCF5', '#C7E9C0', '#74C476', '#238B45', '#00441B'])
    cmap.set_bad('#D9D9D9')
    for row, q in enumerate([0, 1]):
        offset = row*.49
        delta = fields[.4][f'q{q}_probabilities'].mean()-fields[.6][f'q{q}_probabilities'].mean()
        title = 'Without self-colonisation' if q == 0 else 'With self-colonisation'
        direction = 'Increase' if delta > 0 else 'Decrease'
        fig.text(.025, .985-offset, title, va='top', fontsize=9.5)
        fig.text(.025, .949-offset, direction+rf' in mean occupancy: $\Delta\bar p={delta:+.3f}$',
                 va='top', fontsize=9.5, color=BLUE if delta > 0 else '#CC5500')
        for col, h in enumerate([.6, .4]):
            state = fields[h]; p = state[f'q{q}_probabilities']
            ax = fig.add_axes([.025+col*.435, .625-offset, .365, .277])
            mesh = ax.imshow(np.ma.masked_invalid(np.r_[np.nan, p][state['labels']]),
                             cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.text(0, 1.035, f'({chr(97+2*row+col)}) '+rf'$H={h}$', transform=ax.transAxes, fontsize=9.5)
            fig.canvas.draw(); pos = ax.get_position()
            fig.text(pos.x0, .556-offset, rf'Mean patch occupancy: {p.mean():.3f}', fontsize=8.5, va='bottom')
            bx = fig.add_axes([pos.x0, .522-offset, pos.width, .019])
            bx.barh([0], [1], color='#E9E9E9', height=.85)
            bx.barh([0], [p.mean()], color=BLUE, height=.85)
            bx.set(xlim=(0, 1), ylim=(-.55, .55), yticks=[], xticks=[0, .5, 1])
            bx.set_xticklabels(['0', '0.5', '1']); bx.tick_params(axis='x', length=2, pad=1)
            bx.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
    cax = fig.add_axes([.905, .23, .023, .58])
    cb = fig.colorbar(mesh, cax=cax, ticks=[0, .25, .5, .75, vmax])
    cb.ax.set_yticklabels(['0', '0.25', '0.50', '0.75', f'{vmax:.3f}'])
    cb.set_label(r'Patch occupancy probability $p_i^*$')
    fig.legend(handles=[Patch(facecolor='#D9D9D9', label='Non-habitat'),
                        Patch(facecolor=BLUE, label='Mean across patches (equal weights)')],
               loc='lower center', bbox_to_anchor=(.48, -.070), ncol=1, frameon=False)
    emit(fig, '20_self_colonisation_maps_and_means')
    table.to_csv(OUT / 'tables/20_unchanged_patch_probabilities.csv', index=False)


def bird_legend():
    for p in [BIRDS.BEST, BIRDS.BIRD_CURVES, BIRDS.BIRD_BANDS, BIRDS.MODEL_CURVES, BIRDS.REVIEWED_BAND]:
        read(p)
    with BIRDS.BEST.open(newline='') as stream:
        best = {row['species_id']: row for row in csv.DictReader(stream)}
    BIRDS.plot_curve_overlays(best, curve_dest=F/'20_selected_bird_curve_overlays',
                              band_label='95% CI (assumed; gaps interpolated)')


def main():
    for name in ['figures', 'tables', 'logs']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    read(Path(__file__))
    read(ROOT/'code/base/src/tools/final_plot_style.py')
    read(Path(BIRDS.__file__))
    apply_final_style()
    regional_curves()
    parameter_maps()
    forest_parameter_contrasts()
    occupancy_maps()
    bird_legend()
    (OUT/'logs/20_presentation_audit.json').write_text(json.dumps(dict(
        status='numeric_passed_visual_pending', input_sha256=INPUTS,
        new_models=0, new_correlations=0, new_bootstrap_draws=0, new_interpolation=0,
        regional_medians_and_percentiles_unchanged=True, regional_main_curves='Median and 5th–95th profile percentiles only',
        removed_patch_area_and_count_labels=True, H_parameter_panels='Existing access based rho=.999 panels only',
        output_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(F.iterdir()) if p.is_file()}), indent=2)+'\n')
    print('Saved seven figures from unchanged source values:', F)


if __name__ == '__main__':
    main()
