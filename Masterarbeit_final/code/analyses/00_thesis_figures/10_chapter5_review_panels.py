"""Compact Chapter-5 review panels from unchanged, saved display arrays.

Run from the repository root with .venv/bin/python and this script's path.
No simulations, fitting, bootstrapping or interpolation are performed here.
Full original scenario grids remain available in Appendix B.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ufz-chapter5-review-mpl")
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS, MISSING

REL = "02_metapop/02_03_main_analyses/02_03_02_species_dependence"
OUT = FINAL / "results/00_thesis_figures/10_chapter5_review_20261102"
SOURCES = []

def read_npz(path):
    SOURCES.append(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

def save(fig, name):
    export_figure(fig, OUT / name)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def label(ax, i, scenario):
    name = r"$\it{edge\ random}$" if scenario["scenario_id"].startswith("edge") else r"$\it{access\ based}$"
    if scenario["scenario_id"].startswith("only_random"):
        name = r"$\it{only\ random}$"
    else:
        rho = re.search(r"rho=([0-9.]+)", scenario["label"]).group(1)
        name += ", " + rf"$\rho={rho}$"
    ax.text(0, 1.06, f"({chr(97+i)}) " + name, transform=ax.transAxes, fontsize=9.5)

def surface(name, x, y, fields, scenarios, limit, xlabel, ylabel, barlabel, habitat=False):
    fig, axes = plt.subplots(1, len(scenarios), figsize=figure_size(84), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=.13, right=.83, bottom=.28, top=.86, wspace=.18)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for i, (ax, field, scenario) in enumerate(zip(axes, fields, scenarios)):
        ax.set_facecolor(MISSING)
        ax.pcolormesh(x, y, np.ma.masked_invalid(field), shading="auto", cmap=cmap, norm=norm, rasterized=True)
        ax.set_yscale("log"); ax.minorticks_off()
        if habitat:
            ax.set_xlim(1, 0); ax.set_xticks([1, .8, .6, .4, .2, 0])
        else:
            ax.set_xscale("log"); ax.set_xlim(1, 512)
            ax.set_xticks([1, 16, 256], ["1", "16", "256"])
        if "ratio" in ylabel:
            ax.set_yticks([.01, 1, 100, 10000], [r"$10^{-2}$", r"$10^0$", r"$10^2$", r"$10^4$"])
        else:
            ax.set_yticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.minorticks_off()
        label(ax, i, scenario)
    cax = fig.add_axes([.88, .30, .02, .52])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=np.linspace(-limit, limit, 5), label=barlabel)
    fig.supxlabel(xlabel); fig.supylabel(ylabel)
    fig.legend(handles=[Patch(facecolor=MISSING, label="Outside sampled support")], loc="lower center", frameon=False)
    save(fig, name)

def main():
    OUT.mkdir(parents=True, exist_ok=True); apply_final_style()
    scenarios = [FOCAL_SCENARIOS[1], FOCAL_SCENARIOS[-1]]
    path = FINAL / "results/02_metapop/02_03_main_analyses/02_03_01_scenario_occupancy/reference_conditional_256/tables/10_pointwise_contrasts.csv"
    SOURCES.append(path); table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=figure_size(79), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.35, top=.84, wspace=.17)
    for i, (ax, s) in enumerate(zip(axes, scenarios)):
        rows = table[table.scenario_id == s["scenario_id"]].sort_values("H")
        ax.fill_between(rows.H, rows.low, rows.high, color=s["color"], alpha=.2, lw=0)
        ax.plot(rows.H, rows.contrast, color=s["color"], linestyle=s["linestyle"])
        ax.axhline(0, color="#555555", lw=.65); ax.set_ylim(-.8, .8); ax.set_xlim(1, 0)
        ax.set_xticks([1, .75, .5, .25, 0]); label(ax, i, s)
    from matplotlib.lines import Line2D
    fig.supxlabel(r"Habitat proportion $H$"); fig.supylabel("Median occupancy difference")
    fig.legend(handles=[Line2D([], [], color="#555555", label="Scenario minus only random"), Patch(facecolor="#777777", alpha=.2, label="95% bootstrap CI for median difference")], loc="lower center", frameon=False)
    save(fig, "10_reference_contrasts_selected")
    z = read_npz(FINAL / f"data/processed/{REL}/integrated_occupancy_contrast/conditional_256_8_16/11_integrated_contrast_surfaces.npz")
    ell_label = r"Dispersal half-distance $\ell_{1/2}$ (cells; log scale)"
    r_label = r"Extinction-to-colonisation ratio $r$ (log scale)"
    surface("10_integrated_selected", z["grid_half_distance"], z["grid_r"], z["contrast_surfaces"][[0, 5]], scenarios, float(z["color_limit"]), ell_label, r_label, r"Signed area (occupancy $\times H$)")
    z = read_npz(FINAL / "results/00_thesis_figures/05_occupancy_H_difference/05_difference_surfaces.npz")
    # The source has seven panels: retain the baseline plus both selected contrasts.
    surface("10_H_difference_selected", z["grid_half_distance"], z["grid_r"], z["differences"][[0, 1, 6]], [FOCAL_SCENARIOS[0], *scenarios], 1, ell_label, r_label, "Occupancy: H=0.4 minus H=0.6")
    z = read_npz(FINAL / f"data/processed/{REL}/habitat_parameter_contrast/conditional_256_8_16/interpolated/11_interpolated_display_arrays.npz")
    for parameter, ylabel in [("r", r_label), ("half_distance", ell_label)]:
        h = z["H_display"]; he = np.r_[h[0], (h[:-1] + h[1:])/2, h[-1]]
        surface(f"10_H_{parameter}_selected", he, 10**z[f"{parameter}_log_display_edges"], z[f"{parameter}_interpolated_contrast"][[0, 5]].transpose(0, 2, 1), scenarios, 1, r"Habitat proportion $H$", ylabel, "Mean occupancy contrast", habitat=True)
    receipt = {"inputs_sha256": {str(p.relative_to(FINAL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}, "six_panel_indices": [0, 5], "seven_panel_indices": [0, 1, 6], "new_simulations": False, "new_interpolation": False, "original_arrays_unchanged": True}
    (OUT / "10_display_audit.json").write_text(json.dumps(receipt, indent=2))
    print(OUT)

if __name__ == "__main__":
    main()
