"""Export Result A coverage, distribution overlays, and per-extent scatter plots.

Reads verified processed observations and saved summaries only. No sampling,
metric computation, simulation, interpolation to observations, or bootstrap
is performed. Per-extent plots are saved for later selection, not inserted
into the thesis automatically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_02_real_landscape_coverage")
CONFIG = FINAL / "config" / REL / "08_real_landscape_coverage.json"
RESULTS = FINAL / "results" / REL
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
REAL_DATA = FINAL / "data/processed" / REL / "08_real_landscape_metrics.csv"
SIM_TABLE = (FINAL / "results/01_landscape/01_03_result_a/"
             "01_03_01_structural_scenario_responses/tables/07_response_by_scenario_h.csv")
SINGLE_FIGURES = FIGURES / "single_plots"
sys.path.insert(0, str(FINAL / "code/base"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS
from src.tools.final_figure_data import read_metric_plot_table

LABELS = {
    "edge_density_total": "Edge density",
    "largest_cluster_fraction_forest": "Largest cluster fraction",
    "chi": r"Susceptibility $\chi$",
    "mean_nearest_neighbor_distance": "Mean nearest-neighbour distance",
    "mean_pairwise_distance": "Mean inter-patch distance",
    "complete_mst_d50": r"Complete-MST $D_{50}$",
}
SCENARIOS = tuple(item["scenario_id"] for item in FOCAL_SCENARIOS)
SHORT = (
    r"Only-Random ($\rho=0$)",
    r"Edge-Random ($\rho=0.9$)",
    r"Access-based ($\rho=0.9$)",
    r"Edge-Random ($\rho=0.99$)",
    r"Access-based ($\rho=0.99$)",
    r"Edge-Random ($\rho=0.999$)",
    r"Access-based ($\rho=0.999$)",
)
OVERLAY = ("only_random_rho0", "edge_random_rho099", "access_weighted_rho099")
LOG_SINGLE_METRICS = {"chi", "mean_nearest_neighbor_distance"}


def coverage_heatmap(coverage: pd.DataFrame, metric: str, c: dict) -> None:
    fig, axes = plt.subplots(5, 1, figsize=figure_size(160), sharex=True)
    fig.subplots_adjust(left=.30, right=.83, top=.965, bottom=.09, hspace=.37)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#E4E4E4")
    image = None
    for index, (ax, extent) in enumerate(zip(axes, c["window_extents_m"], strict=True)):
        part = coverage[(coverage["metric"] == metric) &
                        (coverage["extent_m"] == extent) &
                        (coverage["scenario_id"].isin(SCENARIOS))]
        matrix = (part.pivot(index="scenario_id", columns="h_bin_index", values="coverage_fraction")
                  .reindex(index=SCENARIOS, columns=range(c["n_h_bins"])).to_numpy(float))
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1,
                          cmap=cmap, interpolation="nearest", aspect="auto")
        ax.set_yticks(range(7), labels=SHORT)
        ax.set_xticks([0, 4, 9, 14, 19], labels=["0.025", "0.225", "0.475", "0.725", "0.975"])
        ax.set_xlim(c["n_h_bins"] - .5, -.5)
        ax.tick_params(axis="both", length=2.5, width=.6)
        ax.text(0, 1.08, f"({chr(97 + index)}) {extent // 1000} km windows",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5)
        for y in np.arange(.5, 7, 1):
            ax.axhline(y, color="white", linewidth=.2)
    axes[-1].set_xlabel(r"Habitat proportion $H$ (bin midpoint)")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=.58, pad=.025, fraction=.055)
    colorbar.set_label("Coverage fraction")
    colorbar.set_ticks([0, .25, .5, .75, 1])
    fig.legend(handles=[Patch(facecolor="#E4E4E4", edgecolor="#666666", linewidth=.4,
                              label="Unavailable")], loc="lower right", bbox_to_anchor=(.97, .015),
               frameon=False)
    export_figure(fig, FIGURES / f"08_coverage_{metric}")
    plt.close(fig)


def distribution_overlay(distribution: pd.DataFrame, intervals: pd.DataFrame,
                         metric: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figure_size(143), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.975, top=.97, bottom=.295, hspace=.21)
    colors = {item["scenario_id"]: item["color"] for item in FOCAL_SCENARIOS}
    for index, (ax, extent) in enumerate(zip(axes, (2000, 32000), strict=True)):
        real = distribution[(distribution["metric"] == metric) &
                            (distribution["extent_m"] == extent)].sort_values("h_bin_index")
        h = real["H_bin_midpoint"].to_numpy(float)
        ax.fill_between(h, real["real_q05"].to_numpy(float), real["real_q95"].to_numpy(float),
                        color="#4E4E4E", alpha=.20, linewidth=0)
        ax.plot(h, real["real_median"].to_numpy(float), color="#333333", linewidth=1.7)
        for scenario in OVERLAY:
            sim = intervals[(intervals["metric"] == metric) &
                            (intervals["scenario_id"] == scenario)].sort_values("h_bin_index")
            x = sim["H_bin_midpoint"].to_numpy(float)
            color = colors[scenario]
            ax.fill_between(x, sim["q05"].to_numpy(float), sim["q95"].to_numpy(float),
                            color=color, alpha=.13, linewidth=0)
            ax.plot(x, sim["median"].to_numpy(float), color=color,
                    linestyle="--" if scenario.startswith("access_weighted") else "-",
                    linewidth=1.5)
        ax.text(0, 1.035, f"({chr(97 + index)}) {extent // 1000} km windows",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5)
        ax.set_xlim(1, 0)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=.6)
    axes[-1].set_xticks([0, .25, .5, .75, 1])
    fig.supxlabel(r"Habitat proportion $H$")
    units = " (cells)" if metric in {"mean_nearest_neighbor_distance", "mean_pairwise_distance", "complete_mst_d50"} else ""
    fig.supylabel(LABELS[metric] + units)
    handles = [Line2D([0], [0], color="#333333", linewidth=1.7, label="Real median")]
    for scenario in OVERLAY:
        label = next(item["label"] for item in FOCAL_SCENARIOS if item["scenario_id"] == scenario)
        handles.append(Line2D([0], [0], color=colors[scenario], linewidth=1.5,
                              linestyle="--" if scenario.startswith("access_weighted") else "-",
                              label=label))
    handles += [Patch(facecolor="#4E4E4E", alpha=.30, edgecolor="none",
                      label="Real 5th–95th percentiles across windows"),
                Patch(facecolor=colors["edge_random_rho099"], alpha=.26, edgecolor="none",
                      label="Simulation 5th–95th percentiles across landscapes")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=2, frameon=False, columnspacing=1.0, handletextpad=.6, labelspacing=.45)
    export_figure(fig, FIGURES / f"08_real_simulation_overlay_{metric}")
    plt.close(fig)


def metric_unit(metric: str) -> str:
    if metric in {"mean_nearest_neighbor_distance", "mean_pairwise_distance", "complete_mst_d50"}:
        return " (cells)"
    if metric == "chi":
        return " (habitat cells)"
    return ""


def check_plot_inputs(real: pd.DataFrame, distribution: pd.DataFrame,
                      response: pd.DataFrame, c: dict) -> None:
    """Reject incomplete samples or summaries inconsistent with raw observations."""
    expected = len(c["window_extents_m"]) * c["n_h_bins"] * c["windows_per_stratum"]
    if len(real) != expected or real["candidate_id"].duplicated().any():
        raise ValueError("Real-window sample is incomplete or has duplicate identifiers.")
    counts = real.groupby(["extent_m", "h_bin_index"]).size()
    if (len(counts) != len(c["window_extents_m"]) * c["n_h_bins"] or
            not counts.eq(c["windows_per_stratum"]).all() or
            set(real["extent_m"]) != set(c["window_extents_m"]) or
            set(real["h_bin_index"]) != set(range(c["n_h_bins"])) or
            not real["H"].between(0, 1).all()):
        raise ValueError("Real-window strata or habitat proportions are invalid.")
    keys = ["extent_m", "h_bin_index", "metric"]
    if (distribution.duplicated(keys).any() or
            len(distribution) != len(counts) * len(c["metrics"])):
        raise ValueError("Real-bin summaries are incomplete or duplicated.")
    for row in distribution.itertuples():
        values = real.loc[(real["extent_m"] == row.extent_m) &
                          (real["h_bin_index"] == row.h_bin_index), row.metric].to_numpy(float)
        valid = values[np.isfinite(values)]
        median = np.median(valid) if len(valid) else np.nan
        if (row.n_real_valid != len(valid) or
                not np.isclose(row.real_median, median, equal_nan=True)):
            raise ValueError("A real-window median or valid count disagrees with its source rows.")
    if (set(response["scenario_id"]) != set(SCENARIOS) or
            set(response["metric"]) != set(c["metrics"]) or
            response.duplicated(["scenario_id", "H_target", "metric"]).any() or
            not response["n_trajectories"].eq(c["n_simulation_seed_blocks"]).all()):
        raise ValueError("Saved simulation summaries do not match the focal design.")


def scatter_overlay(real: pd.DataFrame, distribution: pd.DataFrame,
                    response: pd.DataFrame, metric: str, extent: int,
                    ylim: tuple[float, float]) -> dict:
    """Show every finite observation and seven medians, with common per-metric limits."""
    fig, ax = plt.subplots(figsize=figure_size(128))
    fig.subplots_adjust(left=.15, right=.98, top=.96, bottom=.30)
    points = real.loc[(real["extent_m"] == extent) & np.isfinite(real[metric])]
    ax.scatter(points["H"], points[metric], s=5, color="#777777", alpha=.27,
               linewidths=0, zorder=1)
    for scenario in FOCAL_SCENARIOS:
        sim = response.loc[(response["metric"] == metric) &
                           (response["scenario_id"] == scenario["scenario_id"])].sort_values("H_target")
        ax.plot(sim["H_target"], sim["median"], color=scenario["color"],
                linestyle=scenario["linestyle"], linewidth=1.5, zorder=3)
    binned = distribution.loc[(distribution["metric"] == metric) &
                              (distribution["extent_m"] == extent)].sort_values("h_bin_index")
    ax.plot(binned["H_bin_midpoint"], binned["real_median"], color="#333333",
            linewidth=1.9, zorder=4)
    ax.text(.98, .97, f"{extent // 1000} km windows", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5)
    ax.set_xlim(1, 0)
    if metric in LOG_SINGLE_METRICS:
        ax.set_yscale("log")
        ax.minorticks_off()
    ax.set_ylim(*ylim)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.6)
    fig.supxlabel(r"Habitat proportion $H$")
    unit = metric_unit(metric)
    if metric in LOG_SINGLE_METRICS:
        unit = unit[:-1] + "; log scale)"
    fig.supylabel(LABELS[metric] + unit)
    handles = [Line2D([0], [0], color=s["color"], linestyle=s["linestyle"],
                      linewidth=1.5, label=s["label"]) for s in FOCAL_SCENARIOS]
    handles += [Line2D([0], [0], color="#333333", linewidth=1.9, label="Real median (0.05-wide bins)"),
                Line2D([0], [0], color="#777777", linestyle="none", marker="o",
                       markersize=3, alpha=.5, label="Real windows")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               columnspacing=1.2, handletextpad=.6, labelspacing=.4)
    stem = SINGLE_FIGURES / f"08_all_scenarios_{metric}_{extent // 1000:02d}km"
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"metric": metric, "extent_m": extent, "n_real_windows": int((real["extent_m"] == extent).sum()),
            "n_real_valid": len(points), "n_real_missing": int((real["extent_m"] == extent).sum() - len(points)),
            "n_scenarios": len(SCENARIOS), "y_limits": list(ylim),
            "value_scale": "logarithmic" if metric in LOG_SINGLE_METRICS else "linear",
            "files": [str(stem.with_suffix(suffix).relative_to(FINAL)) for suffix in (".pdf", ".svg", ".png")]}


def single_plot_captions(c: dict, plots: list[dict]) -> None:
    lines = ["# Single plots: all scenarios and observed windows", "",
             "Saved for later figure selection; not included in the thesis automatically.", ""]
    for plot in plots:
        metric, extent = plot["metric"], plot["extent_m"]
        stem = Path(plot["files"][0]).stem
        lines += [f"## {stem}", "",
                  f"Observed and simulated {LABELS[metric].lower()} for {extent // 1000} km real windows. "
                  f"Grey points show all {plot['n_real_valid']} valid observations out of "
                  f"{plot['n_real_windows']} selected windows at their measured habitat proportions. "
                  "The dark grey line joins medians of valid real windows in 0.05-wide $H$ bins, "
                  "plotted at the bin midpoints; each bin contains 100 selected windows before exclusions. "
                  f"Coloured lines join medians across up to {c['n_simulation_seed_blocks']} independent complete "
                  "simulated trajectories per scenario at each sampled $H$. Colours encode $\\rho$; "
                  "solid lines identify Only-Random or Edge-Random and dashed lines identify Access-based removal. "
                  "No variation or confidence intervals are displayed. Missing values remain omitted, "
                  "and defined zero values are retained. All landscapes use $200\\times200$ grids; "
                  + ("distances are grid-cell units, without a physical calibration of simulated landscapes. "
                     if metric_unit(metric) == " (cells)" else
                     "susceptibility uses habitat-cell counts in its patch-size weighting. " if metric == "chi" else
                     "edge density is boundary cell edges per cell area. " if metric == "edge_density_total" else
                     "largest cluster fraction is the share of remaining habitat in the largest patch. ")
                  + ("The value axis is logarithmic to display the full range of positive observations; values are not standardised. "
                     if metric in LOG_SINGLE_METRICS else "The value axis is linear. ")
                  + "The $H$ axis runs from 1 to 0, and value-axis limits are shared across the five window sizes. "
                  "These plots are descriptive comparisons of the balanced window sample.", ""]
    (SINGLE_FIGURES / "08_single_plot_captions.md").write_text("\n".join(lines), encoding="utf-8")
    index = ["# Real windows and all seven scenarios", "",
             "Thirty single plots saved for later selection; none is included in the thesis automatically.", "",
             "Grey points show finite real windows at their measured habitat proportions. The dark grey line "
             "shows real medians in 0.05-wide bins; coloured lines show the seven simulation medians. "
             "Susceptibility and mean nearest-neighbour distance use logarithmic value axes. "
             "All plots have matching editable SVG masters.", "",
             "| Metric | 2 km | 4 km | 8 km | 16 km | 32 km |",
             "| --- | --- | --- | --- | --- | --- |"]
    for metric in c["metrics"]:
        cells = []
        for extent in c["window_extents_m"]:
            stem = f"08_all_scenarios_{metric}_{extent // 1000:02d}km"
            cells.append(f"[PDF]({stem}.pdf) / [PNG]({stem}.png)")
        index.append("| " + LABELS[metric] + " | " + " | ".join(cells) + " |")
    index += ["", "[Captions](08_single_plot_captions.md) · "
              "[Sources and plotting settings](08_single_plot_manifest.json)", ""]
    (SINGLE_FIGURES / "README.md").write_text("\n".join(index), encoding="utf-8")


def captions(c: dict) -> None:
    lines = ["# Figure captions: Result A real-landscape coverage", ""]
    for metric in c["metrics"]:
        unit = ("Distance values are in cells of each $200\\times200$ grid. "
                if metric in {"mean_nearest_neighbor_distance", "mean_pairwise_distance", "complete_mst_d50"}
                else "Values are dimensionless. " if metric in {"edge_density_total", "largest_cluster_fraction_forest"}
                else "Susceptibility uses habitat-cell counts in its patch-size weighting. ")
        lines += [f"## 08_coverage_{metric}", "",
                  f"Coverage of observed {LABELS[metric].lower()} by each of the seven fixed scenarios. "
                  "Panels (a)–(e) show 2, 4, 8, 16, and 32 km real windows, respectively. "
                  "Each cell is the fraction of valid real windows inside the corresponding scenario's "
                  "5th–95th percentile interval across independent complete simulated trajectories "
                  "at the midpoint of that $H$-bin. The real sample contains 100 selected windows "
                  "per 0.05-wide bin and window size; mathematically undefined values are excluded. "
                  "The simulation contains 200 trajectories per scenario, with intervals withheld when "
                  "fewer than 30 have a defined metric. Light grey denotes an unavailable comparison, "
                  "and dark cells denote low coverage. " + unit +
                  "Coverage is descriptive for the balanced real sample; scenario intervals need not "
                  "represent the full real-value distribution.", ""]
    for metric in ("edge_density_total", "largest_cluster_fraction_forest"):
        lines += [f"## 08_real_simulation_overlay_{metric}", "",
                  f"Observed and simulated {LABELS[metric].lower()} across habitat proportion. "
                  "Panel (a) uses 2 km windows; panel (b) uses 32 km windows. "
                  "The dark grey line is the median of up to 100 valid real windows in each 0.05-wide "
                  "$H$-bin, and its grey ribbon shows their 5th–95th percentiles. Coloured lines are "
                  "medians across up to 200 valid independent simulated trajectories at each bin "
                  "midpoint, and matching ribbons show the 5th–95th percentiles across simulated "
                  "landscapes. Solid lines denote Only-Random or Edge-Random; the dashed line denotes "
                  "Access-based removal. Three scenarios are shown here to keep the distributions "
                  "readable; all seven appear in the coverage tables and heatmaps. " +
                  ("Edge density is the number of boundary cell edges per cell area on the "
                   "$200\\times200$ grids; real window extents differ. " if metric == "edge_density_total"
                   else "Values are dimensionless on the $200\\times200$ grids. ") +
                  "Undefined values are omitted.", ""]
    (FIGURES / "08_captions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    sources_path = FINAL / "config" / REL.parent / "figure_sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    coverage = pd.read_csv(TABLES / "08_coverage_by_bin.csv")
    distribution = read_metric_plot_table(TABLES / "08_real_bin_distributions.csv", FINAL, sources,
                                          ("real_median", "real_q05", "real_q95"))
    intervals = read_metric_plot_table(TABLES / "08_simulation_bin_intervals.csv", FINAL, sources,
                                       ("median", "q05", "q95"))
    real = read_metric_plot_table(REAL_DATA, FINAL, sources, (), wide=True)
    response = read_metric_plot_table(SIM_TABLE, FINAL, sources, ("median", "q05", "q95"))
    check_plot_inputs(real, distribution, response, c)
    real[["candidate_id", "extent_m", "h_bin_index", "H", *c["metrics"]]].to_csv(
        TABLES / "08_observed_figure_data.csv", index=False)
    distribution.to_csv(TABLES / "08_real_bin_figure_data.csv", index=False)
    intervals.to_csv(TABLES / "08_simulation_bin_figure_data.csv", index=False)
    response.to_csv(TABLES / "08_scenario_response_figure_data.csv", index=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_final_style()
    for metric in c["metrics"]:
        coverage_heatmap(coverage, metric, c)
    for metric in ("edge_density_total", "largest_cluster_fraction_forest"):
        distribution_overlay(distribution, intervals, metric)
    captions(c)
    plots = []
    for metric in c["metrics"]:
        values = np.concatenate((real[metric].to_numpy(float),
                                 response.loc[response["metric"] == metric, "median"].to_numpy(float)))
        ymax = 1.0 if metric == "largest_cluster_fraction_forest" else float(np.nanmax(values)) * 1.06
        ymin = float(np.nanmin(values)) / 1.06 if metric in LOG_SINGLE_METRICS else 0.0
        if metric in LOG_SINGLE_METRICS and ymin <= 0:
            raise ValueError(f"Log-scale plot would discard a defined non-positive value: {metric}")
        for extent in c["window_extents_m"]:
            plots.append(scatter_overlay(real, distribution, response, metric, extent, (ymin, ymax)))
    single_plot_captions(c, plots)
    inputs = [REAL_DATA, SIM_TABLE, CONFIG, sources_path, TABLES / "08_real_bin_distributions.csv",
              TABLES / "08_simulation_bin_intervals.csv", TABLES / "08_coverage_by_bin.csv"]
    manifest = {"inputs": {str(p.relative_to(FINAL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "scenario_styles": FOCAL_SCENARIOS, "h_axis_direction": "1 to 0",
                "value_scale": "logarithmic for susceptibility and mean nearest-neighbour distance; linear otherwise",
                "real_median": "finite observations per 0.05-wide H bin, plotted at bin midpoint",
                "simulation_median": "saved full-gradient trajectory medians; no interpolation to real windows",
                "edge_density_conversion": sources,
                "plot_data_tables": [str((TABLES / name).relative_to(FINAL)) for name in (
                    "08_observed_figure_data.csv", "08_real_bin_figure_data.csv",
                    "08_simulation_bin_figure_data.csv", "08_scenario_response_figure_data.csv")],
                "ribbons_in_single_plots": False, "included_in_thesis": False, "plots": plots}
    (SINGLE_FIGURES / "08_single_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported 8 PDF/SVG figure pairs and {len(plots)} single plots (PDF/SVG/300-dpi PNG) to {FIGURES}")


if __name__ == "__main__":
    main()
