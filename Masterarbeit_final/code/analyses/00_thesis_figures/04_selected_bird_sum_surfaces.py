"""Three illustrative bird fit surfaces from the verified 8/16-seed data."""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ufz-selected-bird-mpl")
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from src.tools.final_plot_style import MISSING, FOCAL_SCENARIOS, apply_final_style, export_figure, figure_size

SOURCE = FINAL / "data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_parameter_surfaces/conditional_256_8_16/11_bird_rmse_display_surfaces.npz"
BEST = FINAL / "results/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_parameter_surfaces/conditional_256_8_16/tables/11_best_sampled_matches.csv"
DEST = FINAL / "results/00_thesis_figures/04_selected_bird_sum_surfaces/figures/04_selected_bird_summed_rmse"
CURVE_DEST = DEST.with_name("04_selected_bird_curve_overlays")
BIRD_CURVES = FINAL / "data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_reconstruction/11_bird_curves_main_H_grid.csv"
BIRD_BANDS = FINAL / "data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_reconstruction/11_bird_curves_main_H_grid_with_figure2_bands.csv"
REVIEWED_BAND = FINAL / "data/processed/00_thesis_figures/04_bird_band_repair_20261102/04_G_spirurus_reviewed_native_band.csv"
MODEL_CURVES = FINAL / "results/02_metapop/02_03_main_analyses/02_03_02_species_dependence/confirmed_domain_256/replication_008_016/summary_008_016/tables/11_response_summaries.csv"
# Illustrative species with differing reconstructed occupancy curves and
# summed-error surfaces. They are not biological classes or inferred traits.
SELECTED = (
    ("mf2021_07", "Glyphorynchus spirurus"),
    ("mf2021_08", "Herpsilochmus pileatus"),
    ("mf2021_09", "Lipaugus vociferans"),
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curves-only', action='store_true', help='Update only the selected curve figure.')
    args = parser.parse_args()
    with BEST.open(newline="") as stream:
        best = {row["species_id"]: row for row in csv.DictReader(stream)}
    if args.curves_only:
        apply_final_style()
        plot_curve_overlays(best)
        return
    with np.load(SOURCE) as source:
        ids = source["species_ids"].tolist()
        ell = source["grid_half_distance"]
        ratio = source["grid_r"]
        summed = source["summed_scenario_rmse"]
        sampled = source["sampled_scores"]
    assert len(ids) == 15 and summed.shape[0] == 15
    assert sampled.shape[:2] == (15, 7)
    for species_id, name in SELECTED:
        assert species_id in ids and best[species_id]["species_as_published"] == name
    apply_final_style()
    fig, axes = plt.subplots(1, 3, figsize=figure_size(85), sharex=True, sharey=True)
    fig.subplots_adjust(left=.11, right=.87, bottom=.25, top=.86, wspace=.26)
    cmap = plt.get_cmap("cividis_r").copy()
    levels = np.linspace(0, 7, 36)
    norm = BoundaryNorm(levels, cmap.N)
    for panel, (ax, (species_id, name)) in enumerate(zip(axes, SELECTED)):
        values = np.ma.masked_invalid(summed[ids.index(species_id)])
        ax.set_facecolor(MISSING)
        filled = ax.contourf(ell, ratio, values, levels=levels, norm=norm,
                             cmap=cmap, corner_mask=False)
        filled.set_edgecolor("face")
        filled.set_linewidth(.12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 512)
        ax.set_ylim(10**-2.5, 10**5.75)
        ax.set_xticks([1, 16, 256], ["1", "16", "256"])
        ax.set_yticks([.01, 1, 100, 10000],
                      [r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$"])
        ax.minorticks_off()
        first, *rest = name.split()
        ax.text(0, 1.06, f"({chr(97 + panel)}) {first[0]}. {' '.join(rest)}",
                transform=ax.transAxes, fontsize=9.5, fontstyle="italic")
    bar_ax = fig.add_axes([.90, .30, .018, .53])
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(scalar, cax=bar_ax, ticks=np.arange(0, 8),
                       label="Sum of seven curve RMSEs")
    bar.solids.set_edgecolor("face")
    fig.supxlabel(r"Dispersal half-distance $\ell_{1/2}$ (cells; log scale)")
    fig.supylabel(r"Extinction-to-colonization ratio $r$ (log scale)")
    fig.legend(handles=[Patch(facecolor=MISSING, label="Outside sampled parameter support")],
               loc="lower center", frameon=False)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    export_figure(fig, DEST)
    fig.savefig(DEST.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(DEST.with_suffix(".pdf"))
    plot_curve_overlays(best)


def plot_curve_overlays(best: dict[str, dict[str, str]], *, curve_dest: Path = CURVE_DEST,
                        band_label: str = "Digitized 95% CI (assumed; gaps interpolated)") -> None:
    """Show the selected empirical curves against their best complete model curve."""
    with BIRD_CURVES.open(newline="") as stream:
        observed = list(csv.DictReader(stream))
    with BIRD_BANDS.open(newline="") as stream:
        bands = list(csv.DictReader(stream))
    audit = json.loads((DEST.parent.parent / 'review/20261102_todo5_07/04_band_source_audit.json').read_text())
    assert hashlib.sha256(REVIEWED_BAND.read_bytes()).hexdigest() == audit['corrected_table_sha256']
    assert hashlib.sha256(BIRD_BANDS.read_bytes()).hexdigest() == audit['original_joined_band_sha256']
    raw_trace = BIRD_BANDS.with_name('11_bird_figure2_native_pixel_traces.csv')
    assert hashlib.sha256(raw_trace.read_bytes()).hexdigest() == audit['source_trace_sha256']
    with REVIEWED_BAND.open(newline="") as stream:
        native_band = list(csv.DictReader(stream))
    with MODEL_CURVES.open(newline="") as stream:
        model = {
            (row["scenario_id"], row["parameter_id"], round(float(row["H_target"]), 6)):
            float(row["median"]) if row["median"] else float("nan")
            for row in csv.DictReader(stream)
        }
    band_lookup = {(row["species_id"], round(float(row["H"]), 6)): row for row in bands}
    palette = {row["scenario_id"]: row for row in FOCAL_SCENARIOS}
    fig, axes = plt.subplots(1, 3, figsize=figure_size(91), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.29, top=.83, wspace=.17)
    for panel, (ax, (species_id, name)) in enumerate(zip(axes, SELECTED)):
        rows = sorted((row for row in observed if row["species_id"] == species_id
                       and float(row["H"]) <= .925),
                      key=lambda row: float(row["H"]))
        assert len(rows) == 77
        h = np.array([float(row["H"]) for row in rows])
        empirical = np.array([float(row["psi_reconstructed"]) for row in rows])
        lower = np.array([float(band_lookup[(species_id, round(value, 6))]["psi_ci_lower_digitized"])
                          if band_lookup[(species_id, round(value, 6))]["psi_ci_lower_digitized"]
                          else np.nan for value in h])
        upper = np.array([float(band_lookup[(species_id, round(value, 6))]["psi_ci_upper_digitized"])
                          if band_lookup[(species_id, round(value, 6))]["psi_ci_upper_digitized"]
                          else np.nan for value in h])
        choice = best[species_id]
        scenario = choice["best_scenario"]
        model_y = np.array([model[(scenario, choice["best_parameter_id"], round(value, 6))]
                            for value in h])
        assert np.isfinite(model_y).all()
        rmse = np.sqrt(np.mean((model_y - empirical) ** 2))
        assert np.isclose(rmse, float(choice["minimum_rmse"]), atol=1e-10), (name, rmse)
        if species_id == 'mf2021_07':
            native = sorted((r for r in native_band if r['within_observed_domain'] == 'True'
                             and .07 <= float(r['H_source_pixel']) <= .925),
                            key=lambda r: float(r['H_source_pixel']))
            band_h = np.array([float(r['H_source_pixel']) for r in native])
            band_lower = np.array([float(r['psi_ci_lower_digitized']) if r['lower_available'] == 'True' else np.nan for r in native])
            band_upper = np.array([float(r['psi_ci_upper_digitized']) if r['upper_available'] == 'True' else np.nan for r in native])
            lower_missing, upper_missing = ~np.isfinite(band_lower), ~np.isfinite(band_upper)
            # Author explicitly requested closing the four isolated source gaps
            # for display. Preserve the reviewed source table and fitting inputs.
            assert int((lower_missing | upper_missing).sum()) == 4
            for values in [band_lower, band_upper]:
                valid = np.isfinite(values)
                assert valid[0] and valid[-1], 'No endpoint extrapolation'
                missing = np.flatnonzero(~valid)
                assert all(valid[i-1] and valid[i+1] for i in missing), 'Only isolated gaps'
                values[~valid] = np.interp(band_h[~valid], band_h[valid], values[valid])
            assert np.all(band_lower <= band_upper) and np.all(band_lower >= 0) and np.all(band_upper <= 1)
            display_path = CURVE_DEST.with_name('04_selected_bird_curve_overlays_band_display.csv')
            with display_path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['H_source_pixel','lower_display','upper_display','lower_interpolated','upper_interpolated'])
                writer.writerows(zip(band_h, band_lower, band_upper, lower_missing, upper_missing))
            ax.fill_between(band_h, band_lower, band_upper, color="#D0D0D0", linewidth=0, zorder=1)
        else:
            ax.fill_between(h, lower, upper, color="#D0D0D0", linewidth=0, zorder=1)
        ax.plot(h, empirical, color="#262626", linewidth=1.7, zorder=3)
        style = palette[scenario]
        ax.plot(h, model_y, color=style["color"], linestyle=style["linestyle"],
                linewidth=1.9, zorder=4)
        ax.set_xlim(1, 0)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, .8, .6, .4, .2, 0])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis="y", color="#E5E5E5", linewidth=.4)
        first, *rest = name.split()
        ax.text(0, 1.06, f"({chr(97 + panel)}) {first[0]}. {' '.join(rest)}",
                transform=ax.transAxes, fontsize=9.5, fontstyle="italic")
    fig.supxlabel(r"Forest proportion $H$")
    fig.supylabel("Occupancy")
    handles = [
        Line2D([], [], color="#262626", linewidth=1.7, label="Reconstructed empirical curve"),
        Patch(facecolor="#D0D0D0", edgecolor="none", label=band_label),
        Line2D([], [], color=palette["access_weighted_rho09"]["color"],
               linestyle="--", linewidth=1.9, label=r"Best model: $\it{access\ based}$ ($\rho=0.9$)"),
        Line2D([], [], color=palette["edge_random_rho099"]["color"],
               linewidth=1.9, label=r"Best model: $\it{edge\ random}$ ($\rho=0.99$)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .02),
               ncol=2, frameon=False, columnspacing=1.5, handlelength=2.7)
    export_figure(fig, curve_dest)
    fig.savefig(curve_dest.with_suffix(".png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(curve_dest.with_suffix(".pdf"))

if __name__ == "__main__":
    main()
