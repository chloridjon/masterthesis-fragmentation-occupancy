"""Display-only revision of Figures 5.1 and 5.5 from verified saved values.

No new landscapes, equilibria, parameter selection or uncertainty estimates.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-occupancy-revision-mpl')
import sys, json, hashlib
ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(ROOT / 'code/base'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS

OUT = ROOT / 'results/00_thesis_figures/18_occupancy_revision_20261104'
PAIR = ROOT / 'results/00_thesis_figures/11_occupancy_self_colonisation_example_20261102'
INPUTS = {}


def checked(path, digest=None):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest is not None:
        assert actual == digest, path
    INPUTS[str(path.relative_to(ROOT))] = actual
    return path


def emit(fig, name):
    stem = OUT / 'figures' / name
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    for part in ['figures', 'tables', 'logs']:
        (OUT / part).mkdir(parents=True, exist_ok=True)
    checked(Path(__file__))
    checked(ROOT / 'code/base/src/tools/final_plot_style.py')
    apply_final_style()
    source = ROOT / ('results/02_metapop/02_03_main_analyses/02_03_01_scenario_occupancy/'
                     'reference_conditional_256/tables/10_response_summaries.csv')
    summary = pd.read_csv(checked(source), float_precision='round_trip')
    assert list(summary.scenario_id.drop_duplicates()) == [s['scenario_id'] for s in FOCAL_SCENARIOS]
    assert summary.groupby('scenario_id').size().eq(88).all()
    assert summary.n_seed_blocks.eq(20).all()
    summary.to_csv(OUT / 'tables/18_reference_curves.csv', index=False)
    # The figure is authored at its intended 128 mm print width, keeping labels legible.
    fig, ax = plt.subplots(figsize=figure_size(100, width_mm=128))
    fig.subplots_adjust(left=.15, right=.98, bottom=.36, top=.98)
    for item in FOCAL_SCENARIOS:
        part = summary[summary.scenario_id.eq(item['scenario_id'])].sort_values('H')
        ax.fill_between(part.H, part.q05, part.q95, color=item['color'], alpha=.10, linewidth=0)
        name = 'only random' if item['scenario_id'].startswith('only_random') else (
            'edge random' if item['scenario_id'].startswith('edge_random') else 'access based')
        rho = item['scenario_id'].split('rho')[-1]
        rho = {'0': '0', '09': '0.9', '099': '0.99', '0999': '0.999'}[rho]
        label = r'$\mathit{' + name.replace(' ', r'\ ') + r'}$ ($\rho=' + rho + '$)'
        ax.plot(part.H, part['median'], color=item['color'], ls=item['linestyle'],
                lw=1.45, label=label)
    ax.set(xlim=(1, 0), ylim=(0, 1), xlabel=r'Forest proportion $H$',
           ylabel=r'Mean patch occupancy $\bar p$')
    ax.spines[['top', 'right']].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .052), ncol=2,
               frameon=False, handlelength=2.3, fontsize=8.5)
    fig.legend(handles=[Patch(facecolor='#777777', alpha=.18, edgecolor='none',
                              label='5th–95th percentiles across landscapes')],
               loc='lower center', bbox_to_anchor=(.52, -.004), frameon=False,
               fontsize=8.5)
    emit(fig, '18_reference_occupancy_compact')

    audit = json.loads(checked(PAIR / '11_audit.json').read_text())
    fields, panels, patch_rows = {}, [], []
    saved = pd.read_csv(checked(PAIR / '11_figure_values.csv'), float_precision='round_trip')
    probabilities = pd.read_csv(checked(PAIR / '11_patch_probabilities.csv'), float_precision='round_trip')
    for h in [.6, .4]:
        path = PAIR / f'11_example_H{h:.1f}.npz'
        with np.load(checked(path, audit['output_sha256'][str(path.relative_to(ROOT))])) as z:
            state = {k: z[k].copy() for k in z.files}
        fields[h] = state
        assert state['grid'].shape == state['labels'].shape == (200, 200)
        np.testing.assert_array_equal(state['grid'].astype(bool), state['labels'] > 0)
        np.testing.assert_array_equal(np.bincount(state['labels'].ravel())[1:], state['areas'])
        assert state['areas'].sum() == round(h * 40000)
        for q in [0, 1]:
            p = state[f'q{q}_probabilities']
            part = probabilities[(probabilities.H == h) & (probabilities.q == q)].sort_values('patch_label')
            np.testing.assert_allclose(p, part.probability, atol=1e-15, rtol=0)
            np.testing.assert_array_equal(state['areas'], part.area)
            row = saved[(saved.H == h) & (saved.q == q)].iloc[0]
            np.testing.assert_allclose(p.mean(), row.mean_occupancy, atol=1e-15, rtol=0)
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            panels.append(dict(H=h, q=q, mean_occupancy=float(p.mean()),
                               maximum_patch_occupancy=float(p.max()), n_patches=len(p),
                               largest_area=int(np.sort(state['areas'])[-1]),
                               second_largest_area=int(np.sort(state['areas'])[-2])))
            patch_rows.extend(part.to_dict('records'))
    panel_c_max = float(fields[.6]['q1_probabilities'].max())
    global_max = max(p['maximum_patch_occupancy'] for p in panels)
    vmax = panel_c_max if global_max <= panel_c_max else global_max
    assert all(p['maximum_patch_occupancy'] <= vmax for p in panels)
    pd.DataFrame(panels).to_csv(OUT / 'tables/18_four_panel_values.csv', index=False)
    pd.DataFrame(patch_rows).to_csv(OUT / 'tables/18_patch_probabilities.csv', index=False)

    fig = plt.figure(figsize=figure_size(146))
    cmap = LinearSegmentedColormap.from_list('patch_occupancy_green',
                                            ['#F0F8EC', '#C9E4C1', '#73B96B', '#26773F', '#064520'])
    cmap.set_bad('#D9D9D9')
    for row, q in enumerate([0, 1]):
        delta = fields[.4][f'q{q}_probabilities'].mean() - fields[.6][f'q{q}_probabilities'].mean()
        title = 'Without self-colonisation' if q == 0 else 'With self-colonisation'
        fig.text(.025, .985-row*.49, title + rf': change in $\bar p={delta:+.3f}$', va='top', fontsize=9.5)
        for col, h in enumerate([.6, .4]):
            state = fields[h]; p = state[f'q{q}_probabilities']; a = state['areas']
            ax = fig.add_axes([.025+col*.435, .585-row*.49, .365, .32])
            mesh = ax.imshow(np.ma.masked_invalid(np.r_[np.nan, p][state['labels']]),
                             cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            panel = chr(97+row*2+col)
            ax.text(0, 1.055, f'({panel}) '+rf'$H={h}$; $\bar p={p.mean():.3f}$',
                    transform=ax.transAxes, fontsize=9.5)
            sizes = np.sort(a)[::-1]
            text = f'{len(a)} patches; '+rf'$A_1={sizes[0]:,.0f}$'+'\n'+rf'$A_2={sizes[1]:,.0f}$ cells'
            ax.text(0, -.055, text, transform=ax.transAxes, va='top', fontsize=8.5, linespacing=1.2)
    cax = fig.add_axes([.905, .20, .023, .61])
    ticks = [0, .25, .5, .75, vmax]
    cb = fig.colorbar(mesh, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels(['0', '0.25', '0.50', '0.75', f'{vmax:.3f}'])
    cb.set_label(r'Patch occupancy probability $p_i^*$')
    fig.legend(handles=[Patch(facecolor='#D9D9D9', label='Non-habitat')],
               loc='lower center', bbox_to_anchor=(.46, -.015), frameon=False)
    emit(fig, '18_self_colonisation_green')
    record = dict(status='passed_numeric_visual_pending', no_new_simulations=True,
                  no_new_equilibrium_solves=True, no_new_statistics=True,
                  reference_rows=len(summary), checked_patch_probabilities=len(patch_rows),
                  panel_values=panels, colour_limits=[0, vmax], panel_c_max=panel_c_max,
                  global_patch_max=global_max, clipped_values=0,
                  colour_scale='very light green to dark green, shared across four panels',
                  non_habitat='#D9D9D9', input_sha256=INPUTS)
    (OUT / 'logs/18_figure_audit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ['status', 'reference_rows', 'checked_patch_probabilities',
                                           'colour_limits', 'panel_c_max', 'global_patch_max']}, indent=2))


if __name__ == '__main__':
    main()
