"""One matched landscape pair, with baseline and self-colonisation occupancy.

Retain the inspected baseline example unchanged. Evaluate q=1 at the exact
same profile and geometry using two existing equilibrium implementations.
No new landscape simulations, parameter selection or ensemble summaries.
Run from repository root with .venv/bin/python and this script's path.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-self-colonisation-example-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(FINAL / 'code/base'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from src.metapop.self_equilibrium import self_colonization_sweep
from src.metapop.verified_batch import raw_ratio_group
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size

REL = '02_metapop/02_03_main_analyses/02_03_02_species_dependence'
SOURCE = FINAL / f'results/{REL}/occupancy_increase_review_20261102'
DESIGN = FINAL / 'data/processed/02_metapop/02_02_species_parameter_selection/confirmed_response_sampling/09_confirmed_response_parameters.csv'
OUT = FINAL / 'results/00_thesis_figures/11_occupancy_self_colonisation_example_20261102'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((SOURCE / '11_audit.json').read_text())
    row = pd.read_csv(DESIGN, float_precision='round_trip').set_index('parameter_id').loc['domain196']
    assert np.isclose(row.r, 21136.03366986693, rtol=0, atol=1e-10)
    assert np.isclose(row.half_distance, 70.72697814791127, rtol=0, atol=1e-12)
    inputs = [Path(__file__), DESIGN, SOURCE / '11_audit.json']
    inputs += [FINAL / 'code/base/src/metapop' / f'{name}.py' for name in ['self_equilibrium', 'verified_batch', 'parameter_sweep', 'hanski']]
    inputs += [FINAL / 'code/base/src/tools/final_plot_style.py']
    fields, rows, checks = {}, [], []
    for h in [.6, .4]:
        source = SOURCE / f'11_example_H{h:.1f}.npz'
        assert sha(source) == audit['output_sha256'][str(source.relative_to(FINAL))]
        inputs.append(source)
        with np.load(source) as z:
            grid, labels, a, d, p0 = [z[name].copy() for name in ['grid', 'labels', 'areas', 'distances', 'probabilities']]
        assert len(a) > 1 and a.sum() == round(h * 40000)
        np.testing.assert_array_equal(np.bincount(labels.ravel())[1:], a)
        k = np.exp(-row.alpha * d); np.fill_diagonal(k, 0)
        result = self_colonization_sweep(a, k, [row.r], q=1, b=1, x=.5, tol=1e-13, max_iter=100000)[0]
        assert result.converged
        independent, reports, _ = raw_ratio_group(a, d, row.half_distance, np.array([row.r]), q=1, b=1, x=.5, tol=1e-13, max_iter=100000)
        assert reports[0]['converged'] and not reports[0]['needs_scalar_verification']
        p1 = result.probabilities
        np.testing.assert_allclose(p1, independent[:, 0], atol=2e-12, rtol=0)
        assert np.isfinite(p1).all() and (p1 >= 0).all() and (p1 <= 1).all()
        assert (p1 >= p0 - 2e-12).all()
        support = (k + np.eye(len(a))) @ (a * p1)
        residual = float(np.max(abs(p1 - support / (support + row.r / np.sqrt(a)))))
        assert residual <= 1e-13
        fields[h] = dict(grid=grid, labels=labels, areas=a, p0=p0, p1=p1)
        checks.append(dict(H=h, n_patches=len(a), q1_iterations=result.n_iter, q1_residual=residual, independent_max_difference=float(abs(p1-independent[:, 0]).max())))
        np.savez_compressed(OUT / f'11_example_H{h:.1f}.npz', grid=grid, labels=labels, areas=a, distances=d, q0_probabilities=p0, q1_probabilities=p1, r=row.r, half_distance=row.half_distance)
        for q, p in [(0, p0), (1, p1)]:
            for label in range(len(a)):
                rows.append(dict(scenario_id='access_weighted_rho0999', replicate=1, parameter_id='domain196', r=row.r, half_distance=row.half_distance, H=h, q=q, patch_label=label+1, area=a[label], probability=p[label]))
    pd.DataFrame(rows).to_csv(OUT / '11_patch_probabilities.csv', index=False)
    summary = pd.DataFrame([dict(H=h, q=q, n_patches=len(state['areas']), largest_area=float(np.sort(state['areas'])[-1]), second_largest_area=float(np.sort(state['areas'])[-2]), mean_occupancy=float(state[f'p{q}'].mean())) for h, state in fields.items() for q in [0, 1]])
    summary.to_csv(OUT / '11_figure_values.csv', index=False)
    draw(fields)
    record = dict(status='numerically_checked_figure_visual_review_pending_caption_approval_pending', scenario_id='access_weighted_rho0999', replicate=1, parameter_id='domain196', r=float(row.r), half_distance=float(row.half_distance), b=1, x=.5, q_values=[0, 1], checks=checks, no_new_landscapes=True, new_q1_equilibria=2, baseline_unchanged=True, profile_selected_post_hoc=True, n_independent_trajectories=1, ensemble_claim=False, input_sha256={str(p.relative_to(FINAL)):sha(p) for p in inputs}, output_sha256={str(p.relative_to(FINAL)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='11_audit.json'})
    (OUT / '11_audit.json').write_text(json.dumps(record, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(checks, indent=2))

def draw(fields):
    apply_final_style()
    fig = plt.figure(figsize=figure_size(175))
    cmap = LinearSegmentedColormap.from_list(
        'forest_occupancy', ['#B8CFDF', '#8FBEAE', '#4B9B68', '#146B34'])
    cmap.set_bad('#F0F0F0')
    for row, q in enumerate([0, 1]):
        delta = fields[.4][f'p{q}'].mean() - fields[.6][f'p{q}'].mean()
        title = 'Without self-colonisation' if q == 0 else 'With self-colonisation'
        fig.text(.05, .98-row*.49, title + rf': change in $\bar p={delta:+.3f}$', va='top', fontsize=9.5)
        for column, h in enumerate([.6, .4]):
            state = fields[h]; p = state[f'p{q}']; a = state['areas']
            ax = fig.add_axes([.05+column*.44, .58-row*.49, .35, .32])
            values = np.r_[np.nan, p][state['labels']]
            mesh = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            panel = chr(97+2*row+column)
            ax.text(0, 1.045, f'({panel}) '+rf'$H={h}$; $\bar p={p.mean():.3f}$', transform=ax.transAxes, fontsize=9.5)
            sizes = np.sort(a)[::-1]
            text = f'{len(a)} patches\n'+rf'$A_1={sizes[0]:,.0f}$; $A_2={sizes[1]:,.0f}$ cells'
            ax.text(0, -.055, text, transform=ax.transAxes, va='top', fontsize=8.5, linespacing=1.25)
    cax = fig.add_axes([.89, .18, .022, .65])
    fig.colorbar(mesh, cax=cax, label=r'Patch occupancy $p_i^*$')
    fig.legend(handles=[Patch(facecolor='#F0F0F0', label='Non-habitat')], loc='lower center', bbox_to_anchor=(.46, -.012), frameon=False)
    stem = OUT / '11_occupancy_self_colonisation_example'
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    main()
