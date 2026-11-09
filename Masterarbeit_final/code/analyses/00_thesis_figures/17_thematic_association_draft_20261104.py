"""Rearrange audited Result-D displays for the thematic Chapter-5.3 draft.

No new models, associations, regions, bootstrap draws or example profiles.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-thematic-associations-mpl')
import sys, json, hashlib, importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(ROOT / 'code/base'))
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, MISSING
BASE = importlib.import_module('13_metric_regions')
OUT = ROOT / 'results/00_thesis_figures/17_thematic_association_draft_20261104'
SOURCE = ROOT / 'results/00_thesis_figures/15_regions_ABCD_20261104'
REL = '02_metapop/02_03_main_analyses/02_03_03_metric_associations/conditional_256_8_16'
INPUTS = {}
BLUE, ORANGE = '#377A9F', '#CC5500'


def read(path, expected=None):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, path
    INPUTS[str(path.relative_to(ROOT))] = digest
    return path


def emit(fig, name):
    export_figure(fig, OUT / 'figures' / name)
    fig.savefig(OUT / 'figures' / (name + '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    for name in ['figures', 'tables', 'logs']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    read(Path(__file__))
    audit = json.loads(read(SOURCE / 'logs/15_final_audit.json').read_text())
    assert audit['status'] == 'passed_numeric_and_visual_review'
    config = json.loads(read(ROOT / 'config/00_thesis_figures/15_regions_ABCD_20261104.json', audit['input_sha256']['config/00_thesis_figures/15_regions_ABCD_20261104.json']).read_text())
    curves = pd.read_csv(read(SOURCE / 'tables/15_region_by_H.csv'), float_precision='round_trip')
    members = pd.read_csv(read(SOURCE / 'tables/15_profile_memberships.csv'), float_precision='round_trip')
    rawpath = ROOT / 'results' / REL / 'tables/12_within_H_associations.csv'
    raw = pd.read_csv(read(rawpath, audit['input_sha256'][str(rawpath.relative_to(ROOT))]), float_precision='round_trip')
    joined = raw.merge(members[['parameter_id', 'region']], on='parameter_id', validate='many_to_one')
    joined = joined[joined.region.isin(list('ABCD'))]
    group = joined.groupby(['region', 'metric', 'H']).spearman
    q = group.quantile([.05, .5, .95]).unstack()
    q.columns = ['q05', 'median', 'q95']
    q['n_defined'] = group.count()
    filtered = joined[joined.occupancy_span >= .05].groupby(['region', 'metric', 'H']).spearman
    q['span005_median'] = filtered.median()
    q['n_span005'] = filtered.count()
    q.n_span005 = q.n_span005.fillna(0)
    actual = curves.set_index(['region', 'metric', 'H']).sort_index()
    q = q.reindex(actual.index)
    for name in ['q05', 'median', 'q95', 'n_defined', 'span005_median', 'n_span005']:
        np.testing.assert_allclose(actual[name], q[name], atol=2e-15, rtol=0, equal_nan=True)
    curves.to_csv(OUT / 'tables/17_full_regional_curves.csv', index=False)
    surfacepath = ROOT / 'data/processed' / REL / '12_parameter_surfaces.npz'
    surface = dict(np.load(read(surfacepath, audit['input_sha256'][str(surfacepath.relative_to(ROOT))]), allow_pickle=False))
    boundary = pd.read_csv(read(ROOT / config['C_boundary'], audit['input_sha256'][config['C_boundary']]), float_precision='round_trip')
    ell, rr = surface['grid_half_distance'], surface['grid_r']
    fields, cut = surface['median_signed'], boundary.proposed_C_lower_r.to_numpy()
    np.testing.assert_array_equal(ell, boundary.half_distance)
    counts = members.region.value_counts().to_dict()
    assert all(counts[r] == n for r, n in dict(A=61, B=124, C=49, D=6, unassigned=16).items())
    apply_final_style()

    def boundaries(ax):
        for reg in ['A', 'B']:
            rule = config['regions'][reg]
            field = fields[BASE.METRICS.index(rule['metric'])]
            mask = BASE.region_mask(field, ell[None, :], rule)
            ax.contour(ell, rr, np.where(np.isfinite(field), mask.astype(float), np.nan), levels=[.5], colors='#333333', linewidths=.65, linestyles='--', alpha=.8)
        ci = np.argmin(abs(rr[:, None] - cut[None, :]), axis=0)
        supported = np.isfinite(fields[1])[ci, np.arange(len(ell))]
        ax.plot(ell, np.where(supported, cut, np.nan), color='#333333', lw=.65, ls='--', alpha=.8)
        nn = np.where((ell[None, :] > 16) & (rr[:, None] < cut[None, :]), fields[3], np.nan)
        ax.contour(ell, rr, nn, levels=[-.3], colors='#333333', linewidths=.65, linestyles='--', alpha=.8)
        for reg, (x, y) in dict(A=(3.6, .36), B=(21, 550), C=(13, 85000), D=(385, 80)).items():
            ax.text(x, y, reg, ha='center', va='center', weight='bold', fontsize=8.5,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.5))

    for indices, name in [([0, 1, 2], '17_structure_maps_regions'), ([3, 4, 5], '17_connectivity_maps_regions')]:
        fig, axes = plt.subplots(1, 3, figsize=figure_size(98), sharex=True, sharey=True)
        fig.subplots_adjust(left=.13, right=.86, top=.88, bottom=.27, wspace=.22)
        for letter, (ax, m) in enumerate(zip(axes, indices)):
            im = ax.pcolormesh(ell, rr, np.ma.masked_invalid(fields[m]), cmap='RdBu_r', vmin=-1, vmax=1, shading='nearest', rasterized=True)
            BASE.map_axes(ax)
            ax.text(0, 1.06, f'({chr(97+letter)}) ' + BASE.LABELS[m], transform=ax.transAxes)
            boundaries(ax)
        cb = fig.add_axes([.90, .30, .018, .53])
        fig.colorbar(im, cax=cb, ticks=[-1, 0, 1], label=r'Median signed $r_s$')
        fig.supxlabel(r'Dispersal half-distance $\ell_{1/2}$ (cells; log scale)')
        fig.supylabel(r'Extinction-to-colonisation ratio $r$ (log scale)')
        fig.legend(handles=[Patch(facecolor=MISSING, label='Outside sampled support or undefined'),
                            Line2D([], [], color='#333333', ls='--', lw=.65, label='Exploratory regions A/B/C/D (unchanged)')],
                   loc='lower center', ncol=1, frameon=False)
        emit(fig, name)

    def draw_curve(ax, reg, m, restricted=False):
        g = curves[(curves.region == reg) & (curves.metric == BASE.METRICS[m])].sort_values('H')
        ax.axhline(0, color='#999999', lw=.5)
        ax.fill_between(g.H, g.q05, g.q95, color=BLUE, alpha=.18, lw=0)
        ax.plot(g.H, g['median'], color=BLUE, lw=1.6)
        if restricted:
            ax.plot(g.H, g.span005_median, color=ORANGE, lw=1.3)
        ax.set(xlim=(1, 0), ylim=(-1, 1), xticks=[1, .5, 0], yticks=[-1, 0, 1])
        assert len(ax.lines) == (3 if restricted else 2)
        assert all(line.get_marker() in [None, 'None', '', ' '] for line in ax.lines)
        return g

    selection = [('B', 0), ('B', 1), ('C', 2), ('A', 3), ('B', 3), ('B', 5)]
    fig, axes = plt.subplots(2, 3, figsize=figure_size(132), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.98, top=.91, bottom=.20, hspace=.48, wspace=.25)
    selected = []
    for i, (ax, (reg, m)) in enumerate(zip(axes.flat, selection)):
        selected.append(draw_curve(ax, reg, m).assign(panel=chr(97+i)))
        title = BASE.LABELS[m].replace('\n', ' ')
        if m == 1:
            title = 'LCF'
        ax.text(0, 1.07, f'({chr(97+i)}) {title}\nRegion {reg}', transform=ax.transAxes, fontsize=9.5)
    fig.supxlabel(r'Forest proportion $H$')
    fig.supylabel(r'Scenario-balanced Spearman $r_s$')
    fig.legend(handles=[Line2D([], [], color=BLUE, label='Median across defined profiles'),
                        Patch(facecolor=BLUE, alpha=.18, label='5th–95th percentiles across profiles (not a confidence interval)')],
               loc='lower center', ncol=1, frameon=False)
    emit(fig, '17_selected_regional_curves')
    pd.concat(selected).to_csv(OUT / 'tables/17_main_six_curves.csv', index=False)

    for indices, name in [([0, 1, 2], '17_structure_full_regions'), ([3, 4, 5], '17_connectivity_full_regions')]:
        fig, axes = plt.subplots(4, 3, figsize=figure_size(195), sharex=True, sharey=True)
        fig.subplots_adjust(left=.17, right=.98, top=.95, bottom=.17, hspace=.44, wspace=.24)
        for row, reg in enumerate('ABCD'):
            axes[row, 0].text(-.32, .5, f'Region\n{reg}', transform=axes[row, 0].transAxes,
                              ha='center', va='center', fontsize=10, weight='bold')
            for col, m in enumerate(indices):
                ax = axes[row, col]
                draw_curve(ax, reg, m, restricted=True)
                title = BASE.LABELS[m]
                if m == 5:
                    title = 'Complete-MST\n' + r'$D_{50}$'
                ax.text(0, 1.06, f'({chr(97+row*3+col)}) {title}', transform=ax.transAxes, fontsize=9.5)
        fig.supxlabel(r'Forest proportion $H$')
        fig.supylabel(r'Scenario-balanced Spearman $r_s$')
        fig.legend(handles=[Line2D([], [], color=BLUE, label='Median across defined profiles'),
                            Patch(facecolor=BLUE, alpha=.18, label='5th–95th percentiles across profiles (not a confidence interval)'),
                            Line2D([], [], color=ORANGE, label=r'Median for occupancy range $\geq0.05$'),
                            Line2D([], [], ls='none', label='Profiles: A = 61; B = 124; C = 49; D = 6')],
                   loc='lower center', ncol=1, frameon=False)
        emit(fig, name)
    (OUT / 'logs/17_numeric_audit.json').write_text(json.dumps(dict(
        status='passed_numeric_visual_pending', input_sha256=INPUTS,
        regional_rows_independently_checked=len(curves), original_fields_reused=True,
        region_membership_unchanged=True, profile_counts=counts,
        main_selection=[dict(region=r, metric=BASE.METRICS[m]) for r, m in selection],
        main_lines='Regional medians only; 5th–95th profile-percentile bands',
        appendix_lines='Regional medians and occupancy-range>=0.05 restricted medians; same percentile bands',
        individual_regional_example_profiles=0, new_landscapes=0, new_equilibria=0,
        new_bootstrap_draws=0, new_correlations=0), indent=2) + '\n')
    print('Verified all 2,112 regional rows from original CSV; exported five figures without individual example curves.')


if __name__ == '__main__':
    main()
