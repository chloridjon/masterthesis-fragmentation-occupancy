"""Summarise and plot the complete Result A structural scenario sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
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

CONFIG = FINAL / "config" / REL / "07_structural_scenario_responses.json"
DATA = FINAL / "data/processed" / REL
RESULTS = FINAL / "results" / REL
METRIC_LABELS = {
    "edge_density_total": "Edge density ED",
    "largest_cluster_fraction_forest": "Largest cluster fraction LCF",
    "chi": r"Susceptibility $\chi$ (habitat cells)",
    "mean_nearest_neighbor_distance": r"Mean nearest-neighbour distance $\bar d_{\mathrm{NN}}$ (cells)",
    "mean_pairwise_distance": r"Mean inter-patch distance $\bar d_{\mathrm{pair}}$ (cells)",
    "complete_mst_d50": r"Complete-MST $D_{50}$ (cells)",
}
THRESHOLDS = (("H25", .25), ("H50", .5), ("H75", .75))


def response_summary(data: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    for (scenario, h), part in data.groupby(["scenario_id", "H_target"], sort=False):
        for metric in c["metrics"]:
            values = part[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            rows.append({"scenario_id": scenario, "H_target": h, "metric": metric,
                         "n_trajectories": len(part), "n_valid": len(finite),
                         "n_missing": len(part) - len(finite),
                         "median": np.median(finite) if len(finite) else np.nan,
                         "q05": np.quantile(finite, .05) if len(finite) else np.nan,
                         "q95": np.quantile(finite, .95) if len(finite) else np.nan})
    return pd.DataFrame(rows)


def downward_crossing(h: np.ndarray, y: np.ndarray, threshold: float) -> tuple[float, int]:
    """First downward crossing as habitat is lost from H=1.

    On an ascending H axis this is the highest-H upward crossing. A low-H
    rebound of the forest-normalised fraction cannot redefine the transition.
    """
    valid = np.isfinite(h) & np.isfinite(y)
    h, y = h[valid], y[valid]
    crossings = []
    for i in range(1, len(h)):
        if y[i - 1] < threshold <= y[i]:
            fraction = (threshold - y[i - 1]) / (y[i] - y[i - 1])
            crossings.append(float(h[i - 1] + fraction * (h[i] - h[i - 1])))
    return (max(crossings) if crossings else np.nan, len(crossings))


def curve_stats(h: np.ndarray, medians: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    values, counts = {}, {}
    for name, threshold in THRESHOLDS:
        values[name], counts[name] = downward_crossing(h, medians, threshold)
    values["H75_minus_H25"] = (values["H75"] - values["H25"]
                               if np.isfinite(values["H75"]) and np.isfinite(values["H25"]) else np.nan)
    return values, counts


def percolation_summary(data: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuning = c["percolation_bootstrap"]
    summary, history = [], []
    for scenario in c["scenarios"]:
        part = data[(data["scenario_id"] == scenario["scenario_id"]) & (data["H_target"] > 0)]
        matrix = part.pivot(index="seed", columns="H_target", values="largest_cluster_fraction_forest")
        if matrix.shape != (c["n_seed_blocks"], len(c["h_targets"]) - 1):
            raise AssertionError("Incomplete percolation curve matrix.")
        h = matrix.columns.to_numpy(float)
        y = matrix.to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(y, axis=0)
        points, crossing_counts = curve_stats(h, medians)
        keys = tuple(points)
        loo = {key: [] for key in keys}
        for omitted in range(y.shape[0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                values, _ = curve_stats(h, np.nanmedian(np.delete(y, omitted, axis=0), axis=0))
            for key in keys:
                loo[key].append(values[key])
        seed_text = f"result-a-percolation|{scenario['scenario_id']}|{c['base_seed']}"
        bootstrap_seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(bootstrap_seed)
        draws = {key: [] for key in keys}
        previous = {key: None for key in keys}
        stable = {key: 0 for key in keys}
        converged = {key: False for key in keys}
        accepted = {key: np.full(3, np.nan) for key in keys}
        total = 0
        while total < tuning["maximum_draws"]:
            batch = min(tuning["initial_draws"] if total == 0 else tuning["batch_draws"],
                        tuning["maximum_draws"] - total)
            for _ in range(batch):
                chosen = rng.integers(0, y.shape[0], size=y.shape[0])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sample_medians = np.nanmedian(y[chosen], axis=0)
                values, _ = curve_stats(h, sample_medians)
                for key in keys:
                    draws[key].append(values[key])
            total += batch
            for key in keys:
                finite = np.asarray(draws[key], float)
                finite = finite[np.isfinite(finite)]
                quantiles = np.quantile(finite, [.025, .5, .975]) if len(finite) else np.full(3, np.nan)
                change = (float(np.max(np.abs(quantiles - previous[key])))
                          if previous[key] is not None and np.isfinite(quantiles).all() else np.nan)
                eligible = (np.isfinite(points[key]) and len(finite) / total >= .95 and
                            np.isfinite(change) and change <= tuning["h_tolerance"])
                stable[key] = stable[key] + 1 if eligible else 0
                converged[key] = stable[key] >= tuning["consecutive_stable_batches"]
                if converged[key]:
                    accepted[key] = quantiles
                history.append({"scenario_id": scenario["scenario_id"], "statistic": key,
                                "bootstrap_seed": bootstrap_seed, "draws": total,
                                "finite_draws": len(finite), "maximum_quantile_change": change,
                                "consecutive_stable_batches": stable[key], "converged": converged[key],
                                "raw_q025": quantiles[0], "raw_q50": quantiles[1], "raw_q975": quantiles[2]})
                previous[key] = quantiles
            if all(converged[key] or not np.isfinite(points[key]) for key in keys):
                break
        for key in keys:
            omitted_values = np.asarray(loo[key], float)
            max_loo = (float(np.max(np.abs(omitted_values - points[key])))
                       if np.isfinite(points[key]) and np.isfinite(omitted_values).all() else np.nan)
            n_estimable_draws = int(np.isfinite(np.asarray(draws[key], float)).sum())
            summary.append({"scenario_id": scenario["scenario_id"], "statistic": key,
                            "estimate": points[key], "n_crossings": crossing_counts.get(key, np.nan),
                            "n_seed_blocks": y.shape[0], "bootstrap_seed": bootstrap_seed,
                            "bootstrap_draws": total, "bootstrap_converged": converged[key],
                            "n_bootstrap_estimable_draws": n_estimable_draws,
                            "bootstrap_estimable_fraction": n_estimable_draws / total,
                            "ci_low": accepted[key][0] if converged[key] else np.nan,
                            "ci_high": accepted[key][2] if converged[key] else np.nan,
                            "loo_max_change": max_loo,
                            "seed_influence_status": ("unavailable" if not np.isfinite(max_loo) else
                                                      "sensitive" if max_loo > c["percolation_seed_influence_tolerance"] else "stable")})
    return pd.DataFrame(summary), pd.DataFrame(history)


def event_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, h), part in data.groupby(["scenario_id", "H_target"], sort=False):
        rows.append({"scenario_id": scenario, "H_target": h, "n_trajectories": len(part),
                     "expected_random_removals": part["expected_random_removals"].iloc[0],
                     "mean_realized_random_removals": part["n_random_cumulative"].mean(),
                     "sd_realized_random_removals": part["n_random_cumulative"].std(),
                     "minimum_realized_random_removals": part["n_random_cumulative"].min(),
                     "maximum_realized_random_removals": part["n_random_cumulative"].max()})
    return pd.DataFrame(rows)


def seed_extension_audit(data: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Compare medians and distribution tails as matched seed blocks accrue."""
    rows = []
    for (scenario, h), part in data.groupby(["scenario_id", "H_target"], sort=False):
        for metric in c["metrics"]:
            values = part[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            full_q05, full_q95 = np.quantile(finite, [.05, .95]) if len(finite) else (np.nan, np.nan)
            span = float(full_q95 - full_q05)
            median = float(np.median(finite)) if len(finite) else np.nan
            for first_n in c["seed_extension_audit"]["checkpoints"]:
                subset = part.loc[part["replicate"] <= first_n, metric].to_numpy(float)
                valid = subset[np.isfinite(subset)]
                earlier = float(np.median(valid)) if len(valid) else np.nan
                earlier_q05, earlier_q95 = np.quantile(valid, [.05, .95]) if len(valid) else (np.nan, np.nan)
                change = abs(earlier - median) if np.isfinite(earlier) and np.isfinite(median) else np.nan
                tail_change = (max(abs(earlier_q05 - full_q05), abs(earlier_q95 - full_q95))
                               if np.isfinite(earlier_q05) and np.isfinite(full_q05) else np.nan)
                rows.append({"scenario_id": scenario, "H_target": h, "metric": metric,
                             "first_n_seed_blocks": first_n, "n_valid_first_n": len(valid),
                             "n_valid_full_sample": len(finite), "earlier_median": earlier,
                             "full_median": median, "full_q05_q95_span": span,
                             "absolute_median_change": change,
                             "change_divided_by_full_span": change / span if span > 0 else np.nan,
                             "earlier_q05": earlier_q05, "earlier_q95": earlier_q95,
                             "full_q05": full_q05, "full_q95": full_q95,
                             "maximum_endpoint_change": tail_change,
                             "endpoint_change_divided_by_full_span": tail_change / span if span > 0 else np.nan})
    return pd.DataFrame(rows)


def replication_adequacy(audit: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Apply predeclared median and tail stability checks to informative cells."""
    tolerance = c["seed_extension_audit"]["endpoint_tolerance_fraction_of_span"]
    median_tolerance = c["seed_extension_audit"]["median_tolerance_fraction_of_span"]
    maximum = c["seed_extension_audit"]["maximum_fraction_cells_exceeding_tolerance"]
    rows = []
    for checkpoint in c["seed_extension_audit"]["checkpoints"]:
        stage = audit[audit["first_n_seed_blocks"] == checkpoint]
        for metric in (*c["metrics"], "all_metrics"):
            part = stage if metric == "all_metrics" else stage[stage["metric"] == metric]
            informative = part.loc[part["full_q05_q95_span"] > 0,
                                   "endpoint_change_divided_by_full_span"].dropna()
            fraction = float((informative > tolerance).mean()) if len(informative) else np.nan
            medians = part.loc[part["full_q05_q95_span"] > 0,
                               "change_divided_by_full_span"].dropna()
            median_fraction = float((medians > median_tolerance).mean()) if len(medians) else np.nan
            rows.append({"first_n_seed_blocks": checkpoint, "full_n_seed_blocks": c["n_seed_blocks"],
                         "metric": metric, "n_informative_scenario_h_cells": len(informative),
                         "median_tolerance_fraction_of_span": median_tolerance,
                         "endpoint_tolerance_fraction_of_span": tolerance,
                         "maximum_fraction_cells_exceeding_tolerance": maximum,
                         "fraction_median_cells_exceeding_tolerance": median_fraction,
                         "fraction_cells_exceeding_tolerance": fraction,
                         "q95_normalized_median_change": (float(np.quantile(medians, .95))
                                                          if len(medians) else np.nan),
                         "q95_normalized_endpoint_change": (float(np.quantile(informative, .95))
                                                            if len(informative) else np.nan),
                         "meets_predeclared_criterion": (bool(fraction <= maximum and median_fraction <= maximum)
                                                         if np.isfinite(fraction) and np.isfinite(median_fraction)
                                                         else False)})
    return pd.DataFrame(rows)


def grid_resolution_audit(response: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Compare final transition interpolation with the 0.05-bin midpoint grid."""
    midpoints = np.arange(.025, 1.0, .05)
    coarse_h = np.r_[.005, .01, .02, midpoints, .03, .04, 1.0]
    rows = []
    for scenario in c["scenarios"]:
        part = response[(response["scenario_id"] == scenario["scenario_id"]) &
                        (response["metric"] == "largest_cluster_fraction_forest")].sort_values("H_target")
        h = part["H_target"].to_numpy(float)
        median = part["median"].to_numpy(float)
        selected = np.array([np.isclose(x, coarse_h, atol=1e-12).any() for x in h])
        dense, _ = curve_stats(h, median)
        coarse, _ = curve_stats(h[selected], median[selected])
        for statistic in dense:
            rows.append({"scenario_id": scenario["scenario_id"], "statistic": statistic,
                         "dense_grid_estimate": dense[statistic], "bin_midpoint_grid_estimate": coarse[statistic],
                         "dense_minus_bin_midpoint": (dense[statistic] - coarse[statistic]
                                                       if np.isfinite(dense[statistic]) and np.isfinite(coarse[statistic]) else np.nan)})
    return pd.DataFrame(rows)


def plot_response(response: pd.DataFrame, metric: str, figure_dir: Path, n_seeds: int) -> None:
    colors = {entry["scenario_id"]: entry["color"] for entry in FOCAL_SCENARIOS}
    reference = "only_random_rho0"
    fig, axes = plt.subplots(1, 2, figsize=figure_size(105), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.98, top=.96, bottom=.34, wspace=.12)
    scenarios = [
        ("edge_random", "(a) Edge-Random", "-"),
        ("access_weighted", "(b) Access-based", "--"),
    ]
    for ax, (method, panel, linestyle) in zip(axes, scenarios, strict=True):
        for scenario in (reference, *[s["scenario_id"] for s in FOCAL_SCENARIOS if s["scenario_id"] != reference and s["scenario_id"].startswith(method)]):
            part = response[(response["scenario_id"] == scenario) & (response["metric"] == metric)].sort_values("H_target")
            h = part["H_target"].to_numpy(float)
            median, low, high = (part[column].to_numpy(float) for column in ("median", "q05", "q95"))
            color = colors[scenario]
            ax.fill_between(h, low, high, color=color, alpha=.12, linewidth=0)
            ax.plot(h, median, color=color, linestyle="-" if scenario == reference else linestyle, linewidth=1.65)
        ax.text(0, 1.035, panel, transform=ax.transAxes, va="bottom", ha="left", fontsize=9.5)
        ax.set_xlim(1, 0)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=.6)
    fig.supxlabel(r"Habitat proportion $H$")
    fig.supylabel(METRIC_LABELS[metric])
    rho_entries = [("only_random_rho0", r"$\rho=0$"),
                   ("edge_random_rho09", r"$\rho=0.9$"),
                   ("edge_random_rho099", r"$\rho=0.99$"),
                   ("edge_random_rho0999", r"$\rho=0.999$")]
    handles = [Line2D([0], [0], color=colors[scenario], linewidth=1.8, label=label)
               for scenario, label in rho_entries]
    handles += [Line2D([0], [0], color="#404040", linewidth=1.8, label="Median"),
                Patch(facecolor="#777777", alpha=.18, edgecolor="none",
                      label="5th–95th percentiles across landscapes")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=3, frameon=False, columnspacing=1.2, handletextpad=.6, labelspacing=.4)
    export_figure(fig, figure_dir / f"07_response_{metric}")
    plt.close(fig)


def write_captions(c: dict, folder: Path) -> None:
    lines = ["# Figure captions: Result A structural responses", ""]
    messages = {
        "edge_density_total": "Median edge density decreased as directed removal increased; Access-based values were below Edge-Random values at matched positive $\\rho$ throughout the sampled interior $H$ range.",
        "largest_cluster_fraction_forest": "Directed removal shifted the main connectivity transition to lower $H$; Access-based medians exceeded Edge-Random medians at matched positive $\\rho$ throughout the sampled interior range. The Access-based $\\rho=0.999$ point median stayed above 0.5 at all sampled positive $H$, while some bootstrap curves crossed that level.",
        "chi": "The susceptibility peak shifted across $H$ and had wide replicate variation in several directed scenarios.",
        "mean_nearest_neighbor_distance": "The scenario ordering changed with $H$, especially below $H=0.05$.",
        "mean_pairwise_distance": "Median values stayed near 100 grid cells across much of the gradient despite strong differences in boundary and dominant-patch structure.",
        "complete_mst_d50": "Many connected directed landscapes had a defined value of zero, while low-$H$ values varied widely among replicate landscapes.",
    }
    for metric in c["metrics"]:
        label = METRIC_LABELS[metric].split(" (")[0]
        units = ("The distance is measured in grid-cell units on a dimensionless $200\\times200$ simulated landscape. "
                 if metric in {"mean_nearest_neighbor_distance", "mean_pairwise_distance", "complete_mst_d50"}
                 else "Susceptibility uses habitat-cell counts in its patch-size weighting. " if metric == "chi"
                 else "Edge density is the number of boundary cell edges per cell area on the $200\\times200$ simulated grid. " if metric == "edge_density_total"
                 else "The metric is dimensionless on the $200\\times200$ simulated grid. ")
        lines += [f"## 07_response_{metric}", "",
                  f"{label} across habitat proportion in the seven prespecified deforestation scenarios. "
                  "Panel (a) shows Edge-Random and panel (b) Access-based removal; the Only-Random reference "
                  f"appears in both panels. Coloured lines are medians across up to {c['n_seed_blocks']} valid independent complete trajectories "
                  "per scenario, and matching ribbons span the 5th–95th percentiles across landscapes. "
                  "Colours encode the fixed trajectory parameter $\\rho$; the line style identifies the directed "
                  "method. Each trajectory was sampled at all displayed $H$ values. " + units +
                  "Mathematically undefined values are omitted. "
                  "The 20 bin midpoints cover the full gradient; 0.01-spaced samples between $H=0.05$ and "
                  "$0.65$ resolve connectivity transitions, and five extra samples lie below $H=0.05$. "
                  + messages[metric], ""]
    (folder / "07_captions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true",
                        help="Re-export saved response summaries without rerunning analysis or bootstrap.")
    args = parser.parse_args()
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.figures_only:
        sources = json.loads((FINAL / "config" / REL.parent / "figure_sources.json").read_text())
        response = read_metric_plot_table(RESULTS / "tables/07_response_by_scenario_h.csv",
                                          FINAL, sources, ("median", "q05", "q95"))
        response.to_csv(RESULTS / "tables/07_response_figure_data.csv", index=False)
        apply_final_style()
        for metric in c["metrics"]:
            plot_response(response, metric, RESULTS / "figures", c["n_seed_blocks"])
        write_captions(c, RESULTS / "figures")
        print("Exported six structural-response PDF/SVG pairs from saved summaries.")
        return
    manifest = json.loads((DATA / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["producer_settings"] != {k: v for k, v in c.items() if k not in {
        "percolation_bootstrap", "percolation_seed_influence_tolerance", "seed_extension_audit"
    }}:
        raise RuntimeError("Processed data do not match the configured producer settings.")
    data = pd.read_csv(DATA / "landscape_metrics.csv")
    n = len(c["scenarios"]) * c["n_seed_blocks"] * len(c["h_targets"])
    if len(data) != n or data.duplicated(["scenario_id", "seed", "H_target"]).any():
        raise RuntimeError("The Result A sample is incomplete or has duplicate keys.")
    for scenario in c["scenarios"]:
        part = data[data["scenario_id"] == scenario["scenario_id"]]
        if len(part) != c["n_seed_blocks"] * len(c["h_targets"]) or part["seed"].nunique() != c["n_seed_blocks"]:
            raise RuntimeError(f"Incomplete scenario: {scenario['scenario_id']}")
    if not np.allclose(data["H_real"], data["H_target"], atol=1e-12):
        raise RuntimeError("A snapshot was not sampled at its requested habitat proportion.")
    if not (data["n_random_cumulative"] + data["n_directed_cumulative"]).eq(data["event_count"]).all():
        raise RuntimeError("Event counters do not sum to the number of removals.")
    tables, figures, logs = (RESULTS / name for name in ("tables", "figures", "logs"))
    for folder in (tables, figures, logs):
        folder.mkdir(parents=True, exist_ok=True)
    response = response_summary(data, c)
    response.to_csv(tables / "07_response_by_scenario_h.csv", index=False)
    percolation, convergence = percolation_summary(data, c)
    percolation.to_csv(tables / "07_percolation_transitions.csv", index=False)
    convergence.to_csv(tables / "07_bootstrap_convergence.csv", index=False)
    event_audit(data).to_csv(tables / "07_event_audit.csv", index=False)
    extension = seed_extension_audit(data, c)
    extension.to_csv(tables / "07_seed_extension_audit.csv", index=False)
    replication_adequacy(extension, c).to_csv(tables / "07_replication_adequacy.csv", index=False)
    grid_resolution_audit(response, c).to_csv(tables / "07_grid_resolution_audit.csv", index=False)
    apply_final_style()
    for metric in c["metrics"]:
        plot_response(response, metric, figures, c["n_seed_blocks"])
    write_captions(c, figures)
    metadata = {"analysis_plan_section": "1.3.1", "n_seed_blocks": c["n_seed_blocks"],
                "n_scenarios": len(c["scenarios"]), "n_h_targets": len(c["h_targets"]),
                "n_snapshots": len(data), "producer_fingerprint": manifest["producer_fingerprint"],
                "reused_50_seed_source": manifest.get("reused_50_seed_source"),
                "n_converged_percolation_statistics": int(percolation["bootstrap_converged"].sum()),
                "n_estimable_percolation_statistics": int(percolation["estimate"].notna().sum())}
    (logs / "07_summary_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
