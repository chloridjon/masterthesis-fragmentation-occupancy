"""Summarise the fixed seven-scenario metric sample by habitat proportion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from itertools import combinations
from pathlib import Path
import sys

FINAL = next(parent for parent in Path(__file__).resolve().parents if parent.name == "Masterarbeit_final")
REL = Path("01_landscape/01_02_preliminary_analyses/01_02_02_metric_redundancy")
sys.path.insert(0, str(FINAL / "code/base"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, FOCAL_SCENARIOS, MISSING

producer_path = FINAL / "code/analyses" / REL / "03_metric_redundancy.py"
spec = importlib.util.spec_from_file_location("metric_redundancy_producer", producer_path)
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
SHORT_LABELS = ["Edge density", "Largest cluster", r"$\chi$", "Nearest neighbour",
                "Inter-patch mean", r"MST $D_{50}$"]
SCENARIO_DIAGNOSTIC_PAIRS = (
    ("edge_density_total", "largest_cluster_fraction_forest"),
    ("edge_density_total", "chi"),
    ("edge_density_total", "mean_nearest_neighbor_distance"),
    ("largest_cluster_fraction_forest", "chi"),
)
DECISION_NOTES = {
    SCENARIO_DIAGNOSTIC_PAIRS[0]: (
        "Habitat–matrix boundaries versus the fraction of habitat in the largest patch.",
        "The pooled association varies across H and reverses sign. Neither metric replaces the other within the individual scenarios."),
    SCENARIO_DIAGNOSTIC_PAIRS[1]: (
        "Habitat–matrix boundaries versus the size-weighted distribution after one largest patch is excluded.",
        "Strong association is restricted to some H levels. The metrics describe different aspects of fragmentation."),
    SCENARIO_DIAGNOSTIC_PAIRS[2]: (
        "Habitat–matrix boundaries versus local isolation over exact direct Voronoi-neighbour edges.",
        "The locally strong pooled association does not reach the screening threshold within an individual scenario; the definitions are distinct."),
    SCENARIO_DIAGNOSTIC_PAIRS[3]: (
        "Dominance of the largest patch versus the size-weighted distribution after exactly one largest patch is excluded.",
        "Some within-scenario relationships are strong, but direction and strength change across H and scenarios; the definitions remain complementary."),
}


def rank_correlation(a, b):
    if len(a) < 4 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    return float(spearmanr(a, b).statistic)


def analyse_pairs(data, config):
    rows, influences, batches = [], [], []
    metrics = config["metrics"]
    tuning = config["bootstrap"]
    for h in sorted(float(x) for x in config["h_targets"]):
        hdata = data[data["H_target"].eq(h)]
        for m1, m2 in combinations(metrics, 2):
            part = hdata.loc[np.isfinite(hdata[m1].to_numpy(float)) & np.isfinite(hdata[m2].to_numpy(float))]
            a, b = part[m1].to_numpy(float), part[m2].to_numpy(float)
            seed_values = part["seed"].to_numpy(int)
            seeds = np.sort(hdata["seed"].unique())
            indices = {s: np.flatnonzero(seed_values == s) for s in seeds}
            key = (m1, m2, h)
            base = {"metric_1": m1, "metric_2": m2, "H_target": h,
                    "n_landscapes": len(part), "n_seed_blocks_with_valid_pairs": int(len(np.unique(seed_values))),
                    "n_distinct_1": int(np.unique(a).size), "n_distinct_2": int(np.unique(b).size)}
            point = rank_correlation(a, b)
            omitted_values = []
            for omitted in seeds:
                keep = seed_values != omitted
                value = rank_correlation(a[keep], b[keep])
                omitted_values.append(value)
                influences.append({**{k: base[k] for k in ("metric_1", "metric_2", "H_target")},
                                   "omitted_seed": int(omitted), "spearman": value,
                                   "n_valid_landscapes": int(keep.sum())})
            loo = np.asarray(omitted_values, dtype=float)
            change = float(np.max(np.abs(loo - point))) if np.isfinite(point) and np.isfinite(loo).all() else np.nan
            seed_status = "unavailable" if not np.isfinite(point) else (
                "sensitive" if not np.isfinite(change) or change > config["seed_influence_tolerance"] else "stable")
            local_seed = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16) ^ int(config["base_seed"])
            draws, converged, ci = 0, False, [np.nan, np.nan]
            if np.isfinite(point):
                rng = np.random.default_rng(local_seed)
                values = []
                previous = None
                stable_count = 0
                while draws < tuning["maximum_draws"]:
                    target = min(tuning["initial_draws"] if not draws else draws + tuning["batch_draws"], tuning["maximum_draws"])
                    for _ in range(target - draws):
                        chosen = rng.choice(seeds, len(seeds), replace=True)
                        sampled = np.concatenate([indices[s] for s in chosen])
                        values.append(rank_correlation(a[sampled], b[sampled]))
                    draws = target
                    finite = np.asarray(values)[np.isfinite(values)]
                    quantiles = np.quantile(finite, [.025, .5, .975]) if len(finite) else np.full(3, np.nan)
                    delta = float(np.max(np.abs(quantiles - previous))) if previous is not None and np.isfinite(quantiles).all() else np.nan
                    valid_fraction = len(finite) / draws
                    stable = previous is not None and valid_fraction >= .95 and np.isfinite(delta) and delta <= tuning["correlation_tolerance"]
                    stable_count = stable_count + 1 if stable else 0
                    converged = stable_count >= tuning["consecutive_stable_batches"]
                    batches.append({**{k: base[k] for k in ("metric_1", "metric_2", "H_target")},
                                    "bootstrap_seed": local_seed, "draws": draws,
                                    "finite_draws": int(len(finite)), "maximum_quantile_change": delta,
                                    "consecutive_stable_batches": stable_count, "converged": converged,
                                    "ci_low_raw": quantiles[0], "bootstrap_median": quantiles[1], "ci_high_raw": quantiles[2]})
                    if converged:
                        break
                    previous = quantiles
                ci = [quantiles[0], quantiles[2]]
            rows.append({**base, "spearman": point, "absolute_spearman": abs(point) if np.isfinite(point) else np.nan,
                         "loo_max_absolute_change": change, "seed_influence_status": seed_status,
                         "bootstrap_draws": draws, "bootstrap_seed": local_seed,
                         "bootstrap_converged": converged,
                         "spearman_ci_low": ci[0] if converged else np.nan,
                         "spearman_ci_high": ci[1] if converged else np.nan,
                         "raw_spearman_ci_low": ci[0], "raw_spearman_ci_high": ci[1]})
            print(f"{h:g} {m1} / {m2}: r={point:.3g}, draws={draws}, converged={converged}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(influences), pd.DataFrame(batches)


def across_h(table, config):
    records = []
    for (m1, m2), part in table.groupby(["metric_1", "metric_2"], sort=False):
        defined = part[part["spearman"].notna()]
        values = defined["absolute_spearman"]
        median = float(values.median()) if len(values) else np.nan
        fraction = float(values.ge(.8).mean()) if len(values) else np.nan
        signs = np.sign(defined["spearman"].to_numpy(float))
        status = "unavailable" if len(values) < 4 else (
            "potentially_redundant" if median >= config["redundancy_flag_min_median_abs_spearman"]
            and fraction >= config["redundancy_flag_min_fraction_strong_h"] else "not_flagged")
        records.append({"metric_1": m1, "metric_2": m2, "n_defined_h": len(values),
                        "n_total_h": len(part), "median_absolute_spearman": median,
                        "q25_absolute_spearman": float(values.quantile(.25)) if len(values) else np.nan,
                        "q75_absolute_spearman": float(values.quantile(.75)) if len(values) else np.nan,
                        "min_absolute_spearman": float(values.min()) if len(values) else np.nan,
                        "max_absolute_spearman": float(values.max()) if len(values) else np.nan,
                        "fraction_defined_h_absolute_ge_0_8": fraction,
                        "n_h_absolute_ge_0_8": int(values.ge(.8).sum()),
                        "diagnostic_plot_flag": bool(values.ge(.8).sum() >= config["diagnostic_plot_min_strong_h_levels"]),
                        "n_positive_h": int(np.sum(signs > 0)), "n_negative_h": int(np.sum(signs < 0)),
                        "n_seed_sensitive_h": int(part["seed_influence_status"].eq("sensitive").sum()),
                        "n_bootstrap_converged_h": int(part["bootstrap_converged"].sum()),
                        "redundancy_flag": status})
    return pd.DataFrame(records)


def heatmap(summary, config, figure_dir):
    apply_final_style()
    metrics = config["metrics"]
    matrix = np.full((6, 6), np.nan)
    for _, row in summary.iterrows():
        i, j = metrics.index(row["metric_1"]), metrics.index(row["metric_2"])
        matrix[i, j] = matrix[j, i] = row["median_absolute_spearman"]
    fig, ax = plt.subplots(figsize=figure_size(133))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(MISSING)
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, origin="upper", aspect="equal")
    for i in range(6):
        for j in range(6):
            value = matrix[i, j]
            if np.isfinite(value):
                rgb = np.asarray(cmap(value)[:3])
                linear_rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                luminance = np.dot(linear_rgb, [.2126, .7152, .0722])
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=8.5, color="#FFFFFF" if luminance < .18 else "#202020")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=8.5, color="#404040")
    ax.set_xticks(range(6), producer.METRIC_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(6), producer.METRIC_LABELS)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.035, label=r"Median $|r_s|$ across $H$")
    fig.subplots_adjust(left=.30, right=.91, top=.97, bottom=.30)
    export_figure(fig, figure_dir / "03_median_absolute_spearman_heatmap")
    plt.close(fig)


def stability_figures(table, summary, config, figure_dir):
    flagged = summary[summary["diagnostic_plot_flag"]]
    if flagged.empty:
        return 0
    apply_final_style()
    for page, start in enumerate(range(0, len(flagged), 4), start=1):
        subset = flagged.iloc[start:start + 4]
        fig, axes = plt.subplots(2, 2, figsize=figure_size(123), sharex=True, sharey=True)
        for panel, (_, pair) in enumerate(subset.iterrows()):
            ax = axes.flat[panel]
            rows = table[table["metric_1"].eq(pair["metric_1"]) & table["metric_2"].eq(pair["metric_2"])].sort_values("H_target")
            x = rows["H_target"].to_numpy(float)
            ax.plot(x, rows["spearman"], color="#377A9F", label="Spearman correlation")
            ax.fill_between(x, rows["spearman_ci_low"].to_numpy(float), rows["spearman_ci_high"].to_numpy(float),
                            color="#377A9F", alpha=.18)
            ax.axhline(0, color="#BDBDBD", linewidth=.6)
            ax.set_xlim(1, 0)
            ax.set_ylim(-1, 1)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.grid(axis="y", color="#E5E5E5", linewidth=.5)
            i, j = config["metrics"].index(pair["metric_1"]), config["metrics"].index(pair["metric_2"])
            ax.text(0, 1.03, f"({chr(97+panel)}) {SHORT_LABELS[i]} / {SHORT_LABELS[j]}",
                    transform=ax.transAxes, fontsize=8.5, va="bottom", clip_on=False)
        for ax in axes.flat[len(subset):]:
            ax.set_visible(False)
        fig.supxlabel(r"Forest cover $H$")
        fig.supylabel("Spearman correlation")
        fig.legend([Line2D([], [], color="#377A9F"), Patch(facecolor="#377A9F", alpha=.18)],
                   ["Spearman correlation", "95% bootstrap CI for Spearman correlation"],
                   loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(left=.13, right=.98, top=.92, bottom=.19, wspace=.23, hspace=.30)
        export_figure(fig, figure_dir / f"03_flagged_pair_stability_{page:02d}")
        plt.close(fig)
    return (len(flagged) + 3) // 4


def scenario_pair_correlations(frame, summary, config):
    """Revisit the four pairs flagged in the original ten-block diagnostic."""
    available_pairs = set(zip(summary["metric_1"], summary["metric_2"]))
    if not set(SCENARIO_DIAGNOSTIC_PAIRS).issubset(available_pairs):
        raise ValueError("A prespecified diagnostic metric pair is missing from the summary.")
    scenarios = producer.scenarios(config)
    scenario_ids = [item["scenario_id"] for item in scenarios]
    if scenario_ids != [item["scenario_id"] for item in FOCAL_SCENARIOS]:
        raise ValueError("The fixed scenario order differs from the shared plot mapping.")
    expected = len(scenarios) * config["n_seed_blocks"] * len(config["h_targets"])
    if len(frame) != expected or frame.duplicated(["scenario_id", "seed", "H_target"]).any():
        raise ValueError("The saved landscape sample is incomplete or contains duplicate keys.")
    rows = []
    for m1, m2 in SCENARIO_DIAGNOSTIC_PAIRS:
        for scenario in scenarios:
            for h in config["h_targets"]:
                subset = frame[frame["scenario_id"].eq(scenario["scenario_id"]) & frame["H_target"].eq(h)]
                if len(subset) != config["n_seed_blocks"]:
                    raise ValueError("A scenario has missing seed blocks at an H level.")
                valid = subset[np.isfinite(subset[m1].to_numpy(float)) & np.isfinite(subset[m2].to_numpy(float))]
                a, b = valid[m1].to_numpy(float), valid[m2].to_numpy(float)
                value = rank_correlation(a, b)
                rows.append({"metric_1": m1, "metric_2": m2, "scenario_id": scenario["scenario_id"],
                             "method": scenario["method"], "rho": scenario["rho"], "H_target": h,
                             "n_valid_landscapes": len(valid), "n_distinct_1": np.unique(a).size,
                             "n_distinct_2": np.unique(b).size, "spearman": value,
                             "status": ("defined" if np.isfinite(value) else
                                        "fewer_than_four_valid_landscapes" if len(valid) < 4 else
                                        "constant_metric")})
    return pd.DataFrame(rows)


def scenario_stability_figure(table, summary, config, figure_dir):
    if len(SCENARIO_DIAGNOSTIC_PAIRS) != 4:
        raise ValueError("The scenario diagnostic layout expects the four prespecified pairs.")
    apply_final_style()
    fig, axes = plt.subplots(2, 2, figsize=figure_size(158), sharex=True, sharey=True)
    for panel, (m1, m2) in enumerate(SCENARIO_DIAGNOSTIC_PAIRS):
        ax = axes.flat[panel]
        for scenario in FOCAL_SCENARIOS:
            rows = table[table["metric_1"].eq(m1) &
                         table["metric_2"].eq(m2) &
                         table["scenario_id"].eq(scenario["scenario_id"])].sort_values("H_target")
            ax.plot(rows["H_target"].to_numpy(float), rows["spearman"].to_numpy(float),
                    color=scenario["color"], linestyle=scenario["linestyle"], linewidth=1.45)
        ax.axhline(0, color="#BDBDBD", linewidth=.6, zorder=0)
        ax.set_xlim(1, 0)
        ax.set_ylim(-1, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_yticks([-1, -.5, 0, .5, 1])
        ax.grid(axis="y", color="#E5E5E5", linewidth=.5)
        i, j = config["metrics"].index(m1), config["metrics"].index(m2)
        ax.text(0, 1.03, f"({chr(97+panel)}) {SHORT_LABELS[i]} / {SHORT_LABELS[j]}",
                transform=ax.transAxes, fontsize=8.5, va="bottom", clip_on=False)
    fig.supxlabel(r"Habitat proportion $H$")
    fig.supylabel("Spearman correlation")
    handles = [Line2D([], [], color=s["color"], linestyle=s["linestyle"],
                      linewidth=1.45, label=s["label"]) for s in FOCAL_SCENARIOS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=.13, right=.98, top=.94, bottom=.27, wspace=.23, hspace=.30)
    export_figure(fig, figure_dir / "03_flagged_pair_stability_by_scenario")
    plt.close(fig)


def scenario_diagnostics(frame, summary, config, resultdir):
    table = scenario_pair_correlations(frame, summary, config)
    table.to_csv(resultdir / "tables/03_flagged_pair_by_scenario.csv", index=False)
    scenario_stability_figure(table, summary, config, resultdir / "figures")
    return table


def write_metric_decisions(summary, config, resultdir):
    flagged = summary[summary["diagnostic_plot_flag"]]
    flagged_pairs = set(zip(flagged["metric_1"], flagged["metric_2"]))
    if flagged_pairs != set(DECISION_NOTES) or summary["redundancy_flag"].ne("not_flagged").any():
        raise RuntimeError("The new sample changes the metric screen; review the retention decision manually.")
    records = []
    for _, row in flagged.iterrows():
        pair = (row["metric_1"], row["metric_2"])
        concept, reason = DECISION_NOTES[pair]
        records.append({"metric_1": pair[0], "metric_2": pair[1],
                        "n_defined_h": int(row["n_defined_h"]),
                        "n_h_absolute_ge_0_8": int(row["n_h_absolute_ge_0_8"]),
                        "median_absolute_spearman": float(row["median_absolute_spearman"]),
                        "conceptual_comparison": concept, "decision": "retain_both", "reason": reason})
    pd.DataFrame(records).to_csv(resultdir / "tables/03_metric_retention_decisions.csv", index=False)
    (resultdir / "logs/03_final_metric_set.json").write_text(json.dumps({
        "analysis_plan_section": "1.2.2", "decision": "retain_all_six_structural_metrics",
        "metrics": config["metrics"], "n_focal_scenarios": 7,
        "n_independent_seed_blocks": config["n_seed_blocks"],
        "metapopulation_capacity_role": "secondary species-dependent diagnostic; excluded from structural redundancy selection",
        "distance_representation": "depth-1 Voronoi hybrid for all-pair distances and MST; exact direct Voronoi-neighbour edges only for nearest-neighbour distance",
        "limitation": "Complete-MST D50 is constant at H >= 0.7; its correlations there are undefined, not zero.",
        "decision_table": "../tables/03_metric_retention_decisions.csv",
    }, indent=2) + "\n", encoding="utf-8")


def run(config, workdir, resultdir):
    frame = pd.read_csv(workdir / "landscape_metrics.csv")
    table, influences, batches = analyse_pairs(frame, config)
    summary = across_h(table, config)
    table.to_csv(resultdir / "tables/03_pairwise_by_h.csv", index=False)
    influences.to_csv(resultdir / "tables/03_leave_one_seed_out.csv", index=False)
    batches.to_csv(resultdir / "tables/03_bootstrap_convergence.csv", index=False)
    summary.to_csv(resultdir / "tables/03_redundancy_by_pair.csv", index=False)
    frame.groupby(["scenario_id", "method", "rho", "H_target"], as_index=False).agg(
        n_landscapes=("seed", "count"), n_seed_blocks=("seed", "nunique"),
        n_under_two_patches=("n_patches", lambda x: int(x.lt(2).sum())),
        median_patch_count=("n_patches", "median"),
    ).to_csv(resultdir / "tables/03_sample_availability.csv", index=False)
    heatmap(summary, config, resultdir / "figures")
    n_stability = stability_figures(table, summary, config, resultdir / "figures")
    scenario_table = scenario_diagnostics(frame, summary, config, resultdir)
    write_metric_decisions(summary, config, resultdir)
    (resultdir / "logs/03_summary_metadata.json").write_text(json.dumps({
        "analysis_plan_section": "1.2.2", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_metric_pairs": len(summary), "n_h_pair_cells": len(table),
        "n_stability_figures": n_stability,
        "n_pairs_in_stability_figures": int(summary["diagnostic_plot_flag"].sum()), "bootstrap_configuration": config["bootstrap"],
        "n_bootstrap_converged": int(table["bootstrap_converged"].sum()),
        "n_seed_sensitive": int(table["seed_influence_status"].eq("sensitive").sum()),
        "n_scenario_diagnostic_cells": len(scenario_table),
        "n_defined_scenario_diagnostic_cells": int(scenario_table["spearman"].notna().sum()),
    }, indent=2), encoding="utf-8")
    print(summary[["metric_1", "metric_2", "median_absolute_spearman", "fraction_defined_h_absolute_ge_0_8", "redundancy_flag"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-only", action="store_true",
                        help="Recalculate scenario diagnostics from saved metrics without repeating pooled bootstraps.")
    parser.add_argument("--decisions-only", action="store_true",
                        help="Write the reviewed metric decision from saved summary statistics.")
    options = parser.parse_args()
    config = producer.load_config(FINAL / "config" / REL / "03_metric_redundancy.json")
    workdir, resultdir = FINAL / "data/processed" / REL / producer.EXTENDED, FINAL / "results" / REL / producer.EXTENDED
    if options.decisions_only:
        summary = pd.read_csv(resultdir / "tables/03_redundancy_by_pair.csv")
        write_metric_decisions(summary, config, resultdir)
        metadata_path = resultdir / "logs/03_summary_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "metric_decision_updated_at_utc": datetime.now(timezone.utc).isoformat()})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    elif options.scenario_only:
        frame = pd.read_csv(workdir / "landscape_metrics.csv")
        summary = pd.read_csv(resultdir / "tables/03_redundancy_by_pair.csv")
        scenario_table = scenario_diagnostics(frame, summary, config, resultdir)
        metadata_path = resultdir / "logs/03_summary_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "scenario_diagnostics_at_utc": datetime.now(timezone.utc).isoformat(),
                         "n_scenario_diagnostic_cells": len(scenario_table),
                         "n_defined_scenario_diagnostic_cells": int(scenario_table["spearman"].notna().sum())})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    else:
        run(config, workdir, resultdir)
