"""Deterministic teaching figures and per-H correlation dots for the revision.

No production landscape ensemble or occupancy calculation is rerun.
"""
from pathlib import Path
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import label

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(FINAL / 'code/base'))
from src.tools.final_plot_style import LANDSCAPE_COLORS, apply_final_style, export_figure, figure_size
REL = Path('00_thesis_figures/06_supervisor_revision')
OUT = FINAL / 'results' / REL


def save(fig, name):
    export_figure(fig, OUT / 'figures' / name)
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def configuration():
    before = np.zeros((11, 15), dtype=np.uint8)
    before[2:5, 2:5] = 1
    before[2:5, 6:9] = 1
    before[3, 5] = 1  # sole connecting cell
    before[7:9, 11:13] = 1
    after = before.copy()
    after[3, 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=figure_size(82))
    fig.subplots_adjust(left=.01, right=.99, top=.91, bottom=.28, wspace=.13)
    records = []
    for ax, grid, title in zip(axes, (before, after), ('(a) Connected forest patch', '(b) Connection removed')):
        labels, n = label(grid)
        areas = np.bincount(labels.ravel())[1:]
        edges = np.count_nonzero(grid[1:] != grid[:-1]) + np.count_nonzero(grid[:, 1:] != grid[:, :-1])
        others = np.delete(areas, np.argmax(areas))
        chi = float((others ** 2).sum() / others.sum())
        ed = float(edges / grid.size)
        lcf = float(areas.max() / areas.sum())
        rgb = np.zeros((*grid.shape, 3))
        from matplotlib.colors import to_rgb
        rgb[:] = to_rgb(LANDSCAPE_COLORS['non_habitat'])
        rgb[grid == 1] = to_rgb(LANDSCAPE_COLORS['habitat'])
        ax.imshow(rgb, interpolation='nearest', aspect='equal')
        for i in range(1, n + 1):
            yy, xx = np.nonzero(labels == i)
            ax.text(xx.mean(), yy.mean(), str(areas[i - 1]), ha='center', va='center',
                    color='white', fontsize=9.5,
                    bbox=dict(facecolor=LANDSCAPE_COLORS['habitat'], edgecolor='none', pad=.4))
        ax.add_patch(Rectangle((4.5, 2.5), 1, 1, fill=False, edgecolor='#D55E00', linewidth=1.5))
        ax.set_axis_off()
        ax.text(0, 1.05, title, transform=ax.transAxes, va='bottom')
        ax.text(.5, -.08, fr'$ED={edges}/{grid.size}={ed:.3f}$'+'\n'+
                fr'$LCF={areas.max()}/{areas.sum()}={lcf:.3f}$'+ '\n'+fr'$\chi={chi:.2f}$ cells',
                transform=ax.transAxes, ha='center', va='top', fontsize=9.5)
        records.append(dict(panel=title, habitat_cells=int(areas.sum()), patch_areas=areas.tolist(),
                            edge_count=int(edges), ED=ed, LCF=lcf, chi=chi))
    assert records[0]['patch_areas'] == [19, 4] and records[1]['patch_areas'] == [9, 9, 4]
    np.savez_compressed(OUT/'tables/configuration_example.npz', before=before, after=after)
    (OUT/'tables/configuration_example.json').write_text(json.dumps(records, indent=2)+'\n')
    save(fig, 'configuration_example')


def correlation_dots():
    source = FINAL/'results/01_landscape/01_02_preliminary_analyses/01_02_02_metric_redundancy/n30/tables/03_pairwise_by_h.csv'
    df = pd.read_csv(source, float_precision='round_trip')
    metrics = ['edge_density_total', 'largest_cluster_fraction_forest', 'chi',
               'mean_nearest_neighbor_distance', 'mean_pairwise_distance', 'complete_mst_d50']
    # Match the exact saved metric identifiers before plotting.
    found = set(df.metric_1) | set(df.metric_2)
    if set(metrics) != found:
        raise ValueError(f'Unexpected metric identifiers: {found}')
    fig, ax = plt.subplots(figsize=figure_size(111))
    fig.subplots_adjust(left=.10, right=.81, top=.98, bottom=.20)
    mat = np.full((6, 6), np.nan)
    for i in range(6):
        for j in range(i):
            sub = df[((df.metric_1 == metrics[i]) & (df.metric_2 == metrics[j])) |
                     ((df.metric_1 == metrics[j]) & (df.metric_2 == metrics[i]))]
            vals = sub.absolute_spearman.dropna().to_numpy()
            if len(vals):
                mat[i, j] = np.median(vals)
    cmap = plt.get_cmap('cividis').copy(); cmap.set_bad('white')
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    for i in range(6):
        ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, color='#eeeeee', ec='white'))
        for j in range(i):
            sub = df[((df.metric_1 == metrics[i]) & (df.metric_2 == metrics[j])) |
                     ((df.metric_1 == metrics[j]) & (df.metric_2 == metrics[i]))]
            vals = sub.absolute_spearman.dropna().to_numpy()
            # In-cell horizontal positions have a literal 0..1 coefficient scale.
            ax.plot([j-.38, j+.38], [i+.13, i+.13], color='#888888', lw=.6)
            for k, val in enumerate(vals):
                ax.plot(j-.38+.76*val, i+.13+(k%3-1)*.065, 'o', ms=3,
                        mfc='white', mec='#202020', mew=.5)
            ax.text(j, i-.22, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8.5,
                    color='white' if mat[i,j]<.55 else '#202020')
    labels=['ED', 'LCF', r'$\chi$', 'MNN', 'MIP', 'D50']
    ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    cax=fig.add_axes([.85,.27,.025,.64]); fig.colorbar(im,cax=cax).set_label('Median absolute Spearman correlation')
    fig.text(.48,.10,'Each dot: one forest-cover level; position in each cell: 0 (left) to 1 (right)',ha='center',fontsize=8.5)
    fig.text(.48,.045,'Colour and number: median correlation strength; small offsets separate dots',ha='center',fontsize=8.5)
    df.to_csv(OUT/'tables/correlation_dots_source.csv',index=False)
    save(fig,'metric_redundancy_dots')


def main():
    for directory in ('figures','tables'):
        (OUT/directory).mkdir(parents=True,exist_ok=True)
    apply_final_style()
    configuration()
    correlation_dots()

if __name__ == '__main__':
    main()
