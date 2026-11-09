"""Update Chapter-5 labels from saved figures' inputs, without new analyses.

The source arrays, regional medians, percentile bands and reference intervals
are reused unchanged. Only artist text and figure destinations are changed.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0, str(ROOT / 'code/base'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size

PREVIOUS = importlib.import_module('20_chapter5_author_revision_20261105')
OUT = ROOT / 'results/00_thesis_figures/21_chapter5_annotation_labels_20261107'
FIGURES = OUT / 'figures'
LABELS = {
    r'Forest proportion $H$': r'Forest cover $H$',
    r'Scenario-balanced Spearman $r_s$': r'Spearman rank correlation $r_s$',
    r'Median signed $r_s$': r'Median Spearman $r_s$',
    'Median across defined profiles': 'Median across parameter combinations',
    '5th–95th percentiles across profiles': '5th–95th percentiles across parameter combinations',
    'Outside sampled support or undefined': 'Outside sampled range or undefined',
    'Outside sampled support': 'Outside sampled range',
    'Exploratory regions A/B/C/D': 'Exploratory regions A–D',
    'Mean occupancy contrast': 'Mean occupancy difference',
    'Reference profile': 'Reference parameter combination',
}


def emit(figure, name):
    for artist in figure.findobj(matplotlib.text.Text):
        value = artist.get_text()
        if value in LABELS:
            artist.set_text(LABELS[value])
        elif 'Complete-MST' in value:
            artist.set_text(value.replace('Complete-MST\n' + r'$D_{50}$', r'$\mathrm{D50}$')
                           .replace(r'Complete-MST $D_{50}$', r'$\mathrm{D50}$'))
    export_figure(figure, FIGURES / name)
    figure.savefig(FIGURES / f'{name}.png', dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def reference_curves():
    source = ROOT / 'results' / PREVIOUS.REL / 'tables/12_within_H_associations.csv'
    config_path = ROOT / 'config/00_thesis_figures/13_metric_regions_20261102.json'
    config = json.loads(PREVIOUS.read(config_path).read_text())
    table = pd.read_csv(PREVIOUS.read(source), float_precision='round_trip')
    reference = table[table.parameter_id.eq(config['reference'])]
    figure, axes = plt.subplots(1, 2, figsize=figure_size(78), sharex=True, sharey=True)
    figure.subplots_adjust(left=.13, right=.98, bottom=.30, top=.88, wspace=.18)
    for index, (ax, metric) in enumerate(zip(axes, PREVIOUS.REGIONS.METRICS[:2])):
        group = reference[reference.metric.eq(metric)].sort_values('H')
        assert group.loc[group.ci_low.notna(), 'bootstrap_converged'].all()
        ax.axhline(0, color='#999999', lw=.6)
        ax.fill_between(group.H, group.ci_low, group.ci_high, color='#377A9F', alpha=.22, lw=0)
        ax.plot(group.H, group.spearman, color='#377A9F')
        ax.set(xlim=(1, 0), ylim=(-1, 1), xticks=[1, .5, 0], yticks=[-1, 0, 1])
        ax.text(0, 1.07, f'({chr(97+index)}) ' + PREVIOUS.REGIONS.LABELS[index], transform=ax.transAxes)
    figure.supxlabel(r'Forest cover $H$')
    figure.supylabel(r'Spearman rank correlation $r_s$')
    figure.legend(handles=[Line2D([], [], color='#377A9F', label='Reference parameter combination'),
                           Patch(facecolor='#377A9F', alpha=.22,
                                 label='95% pointwise whole-run bootstrap CI')],
                  loc='lower center', ncol=1, frameon=False)
    emit(figure, '13_reference_ED_LCF')


def main():
    for name in ['figures', 'tables', 'logs']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    PREVIOUS.OUT = OUT
    PREVIOUS.F = FIGURES
    PREVIOUS.emit = emit
    PREVIOUS.REGIONS.LABELS = ['Edge density', 'Largest cluster\nfraction', r'$\chi$',
                               'MNN', 'MIP', r'$\mathrm{D50}$']
    apply_final_style()
    PREVIOUS.read(Path(__file__))
    PREVIOUS.read(Path(PREVIOUS.__file__))
    reference_curves()
    PREVIOUS.regional_curves()
    PREVIOUS.parameter_maps()
    PREVIOUS.forest_parameter_contrasts()
    audit = {
        'purpose': 'Chapter-5 author annotation wording and labels only',
        'new_model_solves': 0, 'new_correlations': 0, 'new_bootstrap_draws': 0,
        'new_interpolation': 0, 'regional_values_unchanged': True,
        'input_sha256': PREVIOUS.INPUTS,
        'output_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(FIGURES.iterdir()) if p.is_file()},
    }
    (OUT / 'logs/21_presentation_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'figures': len(list(FIGURES.glob('*.pdf'))), 'output': str(FIGURES),
                      'new_analysis': False}))


if __name__ == '__main__':
    main()
