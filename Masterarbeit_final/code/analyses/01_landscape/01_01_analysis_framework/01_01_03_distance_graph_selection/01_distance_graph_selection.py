"""Analysis 1.1.3: paired distance and graph comparisons on new trajectories.

Run from the repository root with its local virtual environment::

    .venv/bin/python Masterarbeit_final/code/analyses/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/01_distance_graph_selection.py

The run is deterministic for the JSON configuration, code, and seed. Each
scenario/replicate is one full trajectory. Seeds are matched across scenarios
and form the independent resampling blocks. Its per-snapshot metrics
are written atomically under the matching chapter path in data/processed; the
finished metric table and figures are rebuilt from those trajectory files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FINAL = next(parent for parent in Path(__file__).resolve().parents if parent.name == "Masterarbeit_final")
ROOT = FINAL.parent
ANALYSIS_REL = Path("01_landscape") / "01_01_analysis_framework" / "01_01_03_distance_graph_selection"
sys.path.insert(0, str(FINAL / "code" / "base"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

from src.landscape.generation import run_sequential_deforestation
from src.landscape.landscape import Landscape
from src.landscape.landscape_stats import compute_complete_mst_metrics_from_distances
from src.metapop.hanski import metapopulation_capacity_from_arrays
from src.tools.analysis_stability import adaptive_pair_intervals, summary_provenance
from src.tools.final_plot_style import METHOD_COLORS, MISSING, NEUTRAL, apply_final_style, export_figure, figure_size


REPRESENTATIONS = ("centroid_complete", "edge_complete", "voronoi_hybrid", "voronoi_graph")
# Post-comparison choice for subsequent analyses. The exact-edge matrix remains
# the benchmark within this script, so changing this choice does not alter the
# paired calculations or the generated trajectory data.
SELECTED_REPRESENTATION = "voronoi_hybrid"
SELECTED_DISTANCE_MODE = "voronoi_dt"
SELECTED_VORONOI_DEPTH = 1
SELECTED_MEAN_NN_REPRESENTATION = "voronoi_graph"
MAX_EXCLUSION_FRACTION_FOR_SELECTION = 0.25
CAPACITY_RHO_FOCUS_H = 0.5


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=FINAL / "config" / ANALYSIS_REL / "01_distance_graph_selection.json",
    )
    parser.add_argument("--processed-dir", type=Path, default=FINAL / "data" / "processed" / ANALYSIS_REL)
    parser.add_argument("--results-dir", type=Path, default=FINAL / "results" / ANALYSIS_REL)
    parser.add_argument("--summary-config", type=Path, default=FINAL / "config" / ANALYSIS_REL / "01_summary_stability.json")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=0, help="Optional development limit; incomplete runs are not summarized.")
    parser.add_argument("--workers", type=int, default=1, help="Independent trajectory workers.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config["grid_size_cells"]) != 200:
        raise ValueError("The final simulated landscape grid is fixed at 200 x 200 cells.")
    if config.get("voronoi_hybrid_neighbor_weights") != "exact_edge_min":
        raise ValueError("The Voronoi hybrid must use exact edge-min weights for direct neighbours.")
    h = np.asarray(config["h_targets"], dtype=float)
    if np.any(np.diff(h) >= 0) or np.any((h <= 0) | (h >= 1)):
        raise ValueError("h_targets must decrease strictly inside (0, 1); endpoint metrics are undefined.")
    n_removed_float = int(config["grid_size_cells"]) ** 2 * (1.0 - float(h.min()))
    n_removed = int(round(n_removed_float))
    if n_removed <= 1 or abs(n_removed_float - n_removed) > 1e-8:
        raise ValueError("The last habitat target must imply an integer number of removals greater than one.")
    n_targets = int(config["n_expected_random_removal_targets"])
    minimum = float(config["minimum_expected_random_removals"])
    if n_targets < 3 or not 1.0 <= minimum < n_removed:
        raise ValueError("Expected random-removal targets must span from at least one to all removals.")
    random_targets = np.geomspace(minimum, float(n_removed), n_targets)
    # E[N_random] = 1 + (N_removed - 1)(1-rho), including the forced first removal.
    rho = np.sort(1.0 - (random_targets[:-1] - 1.0) / (n_removed - 1.0))
    if not (0 < rho[0] < rho[-1] <= 1) or np.any(np.diff(rho) <= 0):
        raise ValueError("Mapped directed rho values must increase strictly inside (0, 1].")
    if int(config["n_replicates"]) < 2:
        raise ValueError("At least two independent trajectories per scenario are required.")
    d = np.asarray(config["half_distances_cells"], dtype=float)
    if d.size != 3 or np.any(d <= 0):
        raise ValueError("Specify exactly three positive half distances.")
    config["positive_rho_values"] = rho.tolist()
    config["rho_reference_h"] = float(h.min())
    config["n_removed_at_rho_reference_h"] = n_removed
    config["expected_random_removal_targets_at_rho_reference_h"] = random_targets.tolist()
    config["alpha_values"] = (np.log(2.0) / d).tolist()
    return config


def scenario_grid(config: dict) -> list[dict]:
    n_removed = int(config["n_removed_at_rho_reference_h"])
    scenarios = [{"id": "only_random_rho0", "rho": 0.0, "mode": "none", "expected_random_removals_at_reference_h": float(n_removed)}]
    for index, rho in enumerate(config["positive_rho_values"], start=1):
        for mode in ("edge_random", "access_weighted"):
            scenarios.append({
                "id": f"{mode}_rho{index:02d}", "rho": float(rho), "mode": mode,
                "expected_random_removals_at_reference_h": 1.0 + (n_removed - 1.0) * (1.0 - float(rho)),
            })
    return scenarios


def metric_values(dist: np.ndarray, areas: np.ndarray, alphas: list[float], *, graph: bool, kernel_epsilon: float) -> dict:
    n = len(areas)
    if n < 2:
        return {}
    work = np.asarray(dist, dtype=float).copy()
    np.fill_diagonal(work, np.inf)
    finite = np.isfinite(work)
    pair = work[np.triu(finite, 1)]
    result = {
        "mean_nn": float(np.mean(np.min(work, axis=1))) if np.all(finite.any(axis=1)) else np.nan,
        "mean_pair": float(pair.mean()) if len(pair) else np.nan,
        "d50": float(compute_complete_mst_metrics_from_distances(dist, areas)["complete_mst_d50"]),
        "n_pairs": int(len(pair)),
    }
    # A graph-only capacity would set every non-neighbour dispersal term to
    # zero, changing the biological kernel rather than approximating distance.
    if graph:
        return result
    for idx, alpha in enumerate(alphas, start=1):
        capacity, _ = metapopulation_capacity_from_arrays(
            areas, dist, alpha=float(alpha), b=1.0, x=0.5,
            self_colonization=0.0, method="symmetric", kernel_epsilon=kernel_epsilon,
            tol=1e-7, max_iter=3000,
            rng=np.random.default_rng(11003 + idx),
        )
        result[f"capacity_a{idx}"] = float(capacity.lambda_max) if capacity.converged else np.nan
        result[f"capacity_converged_a{idx}"] = bool(capacity.converged)
    return result


def analyse_snapshot(grid: np.ndarray, config: dict) -> list[dict]:
    # Keep the optional Numba dependency out of presentation-only rerenders.
    from src.landscape.complete_edge_distances import complete_edge_distances_open
    configs = (
        ("centroid_complete", "centroid"),
        ("voronoi_hybrid", "voronoi_dt"),
    )
    matrices: dict[str, np.ndarray] = {}
    elapsed: dict[str, float] = {}
    areas = np.empty(0)
    graph_mask = None
    for name, mode in configs:
        start = time.perf_counter()
        landscape = Landscape(grid)
        landscape.ensure_clusters(
            edge_depth=0.0, connectivity="4", boundary_mode=config["analysis_boundary"],
            habitat_threshold=0.5, distance_mode=mode, voronoi_depth=1,
        )
        elapsed[name] = time.perf_counter() - start
        matrices[name] = np.asarray(landscape.cluster_dist_matrix, dtype=float)
        if name == "centroid_complete":
            areas = np.asarray(landscape.cluster_areas, dtype=float)
            start_edge = time.perf_counter()
            matrices["edge_complete"] = complete_edge_distances_open(
                np.asarray(landscape.core_cluster_labels), len(areas)
            )
            elapsed["edge_complete"] = time.perf_counter() - start_edge
        if name == "voronoi_hybrid":
            graph_mask = landscape.voronoi_neighbor_graph.toarray().astype(bool)
    assert graph_mask is not None
    assert all(matrix.shape == (areas.size, areas.size) for matrix in matrices.values())
    # Compare Voronoi topology alone with complete edge distances: keep only
    # direct-neighbour edges and give them exact boundary weights.
    graph_dist = np.where(graph_mask, matrices["edge_complete"], np.inf)
    np.fill_diagonal(graph_dist, 0.0)
    matrices["voronoi_graph"] = graph_dist
    elapsed["voronoi_graph"] = 0.0
    records = []
    for name, dist in matrices.items():
        start = time.perf_counter()
        values = metric_values(
            dist, areas, config["alpha_values"],
            graph=name == "voronoi_graph", kernel_epsilon=float(config["kernel_epsilon"]),
        )
        records.append({
            "representation": name,
            "n_patches": int(len(areas)),
            "mean_patch_area": float(areas.mean()),
            "largest_patch_area": float(areas.max()),
            "graph_build_seconds": elapsed[name],
            "metric_seconds": time.perf_counter() - start,
            **values,
        })
    return records


def trajectory_rows(scenario: dict, replicate: int, config: dict) -> pd.DataFrame:
    seed = int(config["base_seed"]) + replicate - 1
    h_targets = [float(x) for x in config["h_targets"]]
    trajectory = run_sequential_deforestation(
        np.ones((int(config["grid_size_cells"]), int(config["grid_size_cells"])), dtype=np.int8),
        np.random.default_rng(seed),
        rho=float(scenario["rho"]), directed_mode=str(scenario["mode"]),
        target_habitat_fraction=min(h_targets),
        generation_boundary=config["generation_boundary"],
        k=int(config["access_k"]), lambda_decay=float(config["access_lambda"]),
        eta=float(config["access_eta"]), save_habitat_fractions=h_targets,
        include_initial_snapshot=False, seed=seed,
    )
    rows = []
    for snapshot in trajectory.snapshots:
        base = {
            "trajectory_id": f"{scenario['id']}__rep{replicate:03d}",
            "scenario_id": scenario["id"], "method": scenario["mode"] if scenario["mode"] != "none" else "only_random",
            "rho": scenario["rho"], "replicate": replicate, "seed": seed,
            "H_target": snapshot.requested_habitat_fractions[0],
            "H_real": snapshot.habitat_fraction,
            "event_count": snapshot.event_count,
            "grid_sha256": hashlib.sha256(snapshot.grid.tobytes()).hexdigest(),
            "generator_version": trajectory.generator_version,
        }
        try:
            for metric in analyse_snapshot(snapshot.grid, config):
                rows.append({**base, **metric, "metric_error": ""})
        except ValueError as exc:
            if "No core habitat clusters" not in str(exc):
                raise
            for name in REPRESENTATIONS:
                rows.append({**base, "representation": name, "n_patches": 0, "metric_error": str(exc)})
    return pd.DataFrame(rows)


def save_trajectory(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def paired_table(data: pd.DataFrame, metric: str, benchmark: str, candidate: str) -> pd.DataFrame:
    subset = data[data["representation"].isin((benchmark, candidate))]
    keys = ["trajectory_id", "scenario_id", "method", "rho", "replicate", "seed", "H_target", "H_real", "n_patches", "mean_patch_area", "largest_patch_area"]
    paired = subset.pivot(index=keys, columns="representation", values=metric).reset_index()
    paired = paired.dropna(subset=[benchmark, candidate]).copy()
    paired["reference"] = paired[benchmark].astype(float)
    paired["candidate"] = paired[candidate].astype(float)
    paired["difference"] = paired["candidate"] - paired["reference"]
    paired["relative_difference"] = np.where(
        np.abs(paired["reference"]) > 1e-12,
        paired["difference"] / np.abs(paired["reference"]), np.nan,
    )
    paired["metric"] = metric
    paired["reference_representation"] = benchmark
    paired["candidate_representation"] = candidate
    return paired


def _spearman(x: pd.Series, y: pd.Series) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 4 or np.unique(a[good]).size < 2 or np.unique(b[good]).size < 2:
        return np.nan
    return float(spearmanr(a[good], b[good]).statistic)


def summarize_pairs(pairs: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in pairs.groupby(group_keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        valid_rel = part["relative_difference"].dropna().abs()
        positive_reference = part["reference"].gt(1e-12)
        ratios = part.loc[positive_reference, "candidate"] / part.loc[positive_reference, "reference"]
        rows.append({
            **dict(zip(group_keys, key)), "n_landscapes": int(len(part)),
            "n_trajectories": int(part["trajectory_id"].nunique()),
            "n_nonzero_reference": int(len(valid_rel)),
            "spearman": _spearman(part["reference"], part["candidate"]),
            "median_difference": float(part["difference"].median()),
            "median_candidate_reference_ratio": float(ratios.median()) if len(ratios) else np.nan,
            "p05_difference": float(part["difference"].quantile(0.05)),
            "p95_difference": float(part["difference"].quantile(0.95)),
            "median_abs_relative_difference": float(valid_rel.median()) if len(valid_rel) else np.nan,
            "p05_abs_relative_difference": float(valid_rel.quantile(0.05)) if len(valid_rel) else np.nan,
            "p95_abs_relative_difference": float(valid_rel.quantile(0.95)) if len(valid_rel) else np.nan,
            "size_gap_spearman": _spearman(part["mean_patch_area"], part["difference"]),
            "largest_patch_gap_spearman": _spearman(part["largest_patch_area"], part["difference"]),
        })
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, figure_dir: Path, *, structural_only: bool = False) -> None:
    apply_final_style()
    panel_labels = {
        "mean_nn": "MNN", "mean_pair": "MIP",
        "d50": "D50", "capacity_a1": "Capacity (4.03 cells)",
        "capacity_a2": "Capacity (6.48 cells)", "capacity_a3": "Capacity (10.43 cells)",
    }
    definitions = [
        ("edge_complete", "centroid_complete"),
        ("edge_complete", "voronoi_hybrid"),
        ("edge_complete", "voronoi_graph"),
    ]
    for benchmark, candidate in definitions:
        if candidate == "voronoi_graph":
            metric_names = ("mean_nn", "d50")
        else:
            metric_names = (("mean_nn", "mean_pair", "d50") if structural_only else
                            ("mean_nn", "mean_pair", "d50", "capacity_a1", "capacity_a2", "capacity_a3"))
        subset = summary[
            summary["reference_representation"].eq(benchmark)
            & summary["candidate_representation"].eq(candidate)
            & summary["metric"].isin(metric_names)
        ]
        if subset.empty:
            continue
        rows, columns = (1, len(metric_names)) if len(metric_names) <= 3 else (2, 3)
        fig, axes = plt.subplots(rows, columns, figsize=figure_size(72 if rows == 1 else 128), sharex=True)
        for i, metric in enumerate(metric_names):
            axis = axes.flat[i]
            part = subset[subset["metric"].eq(metric)].sort_values("H_target")
            if part.empty:
                continue
            x = part["H_target"].to_numpy(float)
            axis.plot(x, part["spearman"], color="#377A9F", marker="o", label="Spearman correlation")
            sensitive = part["loo_max_spearman_change"].gt(part["loo_correlation_tolerance"])
            axis.scatter(x[sensitive], part.loc[sensitive, "spearman"], facecolors="white", edgecolors="#377A9F", s=22, zorder=5)
            unresolved = ~part["bootstrap_converged"]
            axis.scatter(x[unresolved], part.loc[unresolved, "spearman"], marker="x", color="#377A9F", s=24, zorder=6)
            axis.fill_between(x, part["spearman_ci_low"].to_numpy(float), part["spearman_ci_high"].to_numpy(float),
                              color="#377A9F", alpha=0.18, linewidth=0)
            axis.set_ylim(-1, 1)
            axis.axhline(0, color="#C4C4C4", linewidth=0.6)
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
            axis.set_xlim(1, 0)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(0, 1.025, f"({chr(97+i)}) {panel_labels[metric]}",
                      transform=axis.transAxes, va="bottom", fontsize=8.5, clip_on=False)
        for axis in axes.flat[len(metric_names):]:
            axis.set_visible(False)
        fig.supylabel("Spearman correlation", x=0.015)
        fig.supxlabel(r"Forest cover $H$", y=0.095)
        first_axis = next(axis for axis in axes.flat if axis.get_visible() and axis.collections)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [first_axis.lines[0], Patch(facecolor="#377A9F", alpha=0.18)]
        labels = ["Spearman correlation", "95% bootstrap CI for Spearman correlation"]
        if subset["loo_max_spearman_change"].gt(subset["loo_correlation_tolerance"]).any():
            handles.append(Line2D([], [], marker="o", markerfacecolor="white", color="#377A9F", linestyle="none"))
            labels.append("Sensitive to run omission")
        if (~subset["bootstrap_converged"]).any():
            handles.append(Line2D([], [], marker="x", color="#377A9F", linestyle="none"))
            labels.append("CI did not converge")
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
        fig.subplots_adjust(left=0.15, right=0.985, bottom=0.27 if rows == 1 else (0.22 if candidate == "centroid_complete" else 0.18), top=0.93, wspace=0.32, hspace=0.29)
        stem = figure_dir / ("01_" + candidate + "_rank_preservation")
        export_figure(fig, stem)
        if structural_only:
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(rows, columns, figsize=figure_size(72 if rows == 1 else 128), sharex=True)
        compressed_scale = candidate == "centroid_complete"
        for i, metric in enumerate(metric_names):
            axis = axes.flat[i]
            part = subset[subset["metric"].eq(metric)].sort_values("H_target")
            if part.empty:
                continue
            x = part["H_target"].to_numpy(float)
            axis.plot(x, 100 * part["median_abs_relative_difference"], color=NEUTRAL, marker="o")
            axis.fill_between(x, 100 * part["p05_abs_relative_difference"].to_numpy(float),
                              100 * part["p95_abs_relative_difference"].to_numpy(float),
                              color=NEUTRAL, alpha=0.13, linewidth=0)
            axis.set_xlim(1, 0)
            if compressed_scale:
                axis.set_yscale("function", functions=(
                    lambda percentage: np.log1p(percentage / 100),
                    lambda transformed: 100 * np.expm1(transformed),
                ))
            upper = float(100 * part["p95_abs_relative_difference"].max())
            if np.isfinite(upper) and upper > 0:
                axis.set_ylim(0, upper * 1.05)
                if compressed_scale:
                    if upper <= 120:
                        candidates = (0, 20, 40, 60, 80, 100)
                    elif upper <= 500:
                        candidates = (0, 50, 100, 200, 500)
                    elif upper <= 2000:
                        candidates = (0, 100, 200, 500, 1000, 2000)
                    elif upper <= 10000:
                        candidates = (0, 100, 500, 2000, 5000, 10000)
                    else:
                        candidates = (0, 100, 1000, 5000, 20000)
                    ticks = [value for value in candidates if value <= upper * 1.05]
                    axis.set_yticks(ticks)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(0, 1.025, f"({chr(97+i)}) {panel_labels[metric]}",
                      transform=axis.transAxes, va="bottom", fontsize=8.5, clip_on=False)
        for axis in axes.flat[len(metric_names):]:
            axis.set_visible(False)
        fig.supylabel(
            "Difference magnitude (%)" +
            ("\nlog(1 + fraction) scale" if compressed_scale else ""),
            x=0.015,
        )
        fig.supxlabel(r"Forest cover $H$", y=0.095)
        first_axis = next(axis for axis in axes.flat if axis.get_visible() and axis.collections)
        fig.legend(
            [first_axis.lines[0], first_axis.collections[0]],
            ["Median", "5th–95th percentiles across landscapes"],
            loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
        )
        fig.subplots_adjust(left=0.15, right=0.985, bottom=0.27 if rows == 1 else 0.18, top=0.93, wspace=0.32, hspace=0.29)
        export_figure(fig, figure_dir / ("01_" + candidate + "_relative_deviation"))
        plt.close(fig)


def capacity_shortfall_by_rho(capacity_pairs: pd.DataFrame) -> pd.DataFrame:
    """Paired hybrid shortfall; quantiles describe landscapes, not median uncertainty."""
    valid = capacity_pairs[np.isfinite(capacity_pairs["relative_difference"])].copy()
    valid["shortfall_percent"] = -100.0 * valid["relative_difference"]
    return valid.groupby(
        ["metric", "H_target", "scenario_id", "method", "rho"], as_index=False
    ).agg(
        n_landscapes=("shortfall_percent", "count"),
        median_shortfall_percent=("shortfall_percent", "median"),
        p05_shortfall_percent=("shortfall_percent", lambda values: float(values.quantile(0.05))),
        p95_shortfall_percent=("shortfall_percent", lambda values: float(values.quantile(0.95))),
    )


def make_capacity_rho_figures(
    summary: pd.DataFrame, figure_dir: Path, half_distances: list[float],
    eligible_h: list[float], n_removed_at_reference_h: int,
) -> None:
    """Show rho dependence at H=0.5 and over the full decision H domain."""
    apply_final_style()
    metrics = ("capacity_a1", "capacity_a2", "capacity_a3")
    methods = (("edge_random", "Edge-Random"), ("access_weighted", "Access-based"))
    comparison_colors = {"edge_random": "#377A9F", "access_weighted": "#6BA6C4"}
    rho_levels = sorted(float(rho) for rho in summary.loc[summary["rho"].gt(0), "rho"].unique())
    all_rho = [0.0, *rho_levels]

    def log_random_count(rho: np.ndarray | float) -> np.ndarray:
        return -np.log10(1.0 + (n_removed_at_reference_h - 1.0) * (1.0 - np.asarray(rho)))

    def rho_label(rho: float) -> str:
        return f"{rho:.4g}"

    tick_indices = np.unique(np.linspace(0, len(rho_levels), 6, dtype=int))
    tick_rho = [all_rho[index] for index in tick_indices]
    tick_positions = log_random_count(np.asarray(tick_rho))
    focused = summary[summary["H_target"].eq(CAPACITY_RHO_FOCUS_H)]
    y_max = max(10.0, 10.0 * np.ceil(float(focused["p95_shortfall_percent"].max()) / 10.0))
    fig, axes = plt.subplots(1, 3, figsize=figure_size(83), sharey=True)
    for column, (axis, metric) in enumerate(zip(axes, metrics)):
        part = focused[focused["metric"].eq(metric)]
        reference = part[part["method"].eq("only_random")]
        if len(reference) == 1:
            reference_median = float(reference["median_shortfall_percent"].iloc[0])
            axis.axhline(
                reference_median,
                color=METHOD_COLORS["only_random"], linestyle="solid",
                linewidth=1.4, label=r"Only-Random ($\rho=0$)",
            )
            errorbar_key = axis.errorbar(
                [float(log_random_count(0.0))], [reference_median],
                yerr=[
                    [reference_median - float(reference["p05_shortfall_percent"].iloc[0])],
                    [float(reference["p95_shortfall_percent"].iloc[0]) - reference_median],
                ],
                color=METHOD_COLORS["only_random"], marker="D", markersize=4,
                linestyle="none", capsize=2, linewidth=0.8,
            )
        for method, label in methods:
            series = part[part["method"].eq(method)].sort_values("rho")
            if series.empty:
                continue
            rho = series["rho"].to_numpy(float)
            positions = log_random_count(rho)
            median = series["median_shortfall_percent"].to_numpy(float)
            axis.fill_between(
                positions,
                series["p05_shortfall_percent"].to_numpy(float),
                series["p95_shortfall_percent"].to_numpy(float),
                color=comparison_colors[method], alpha=0.12, linewidth=0,
            )
            axis.plot(positions, median, color=comparison_colors[method],
                      linestyle="--" if method == "access_weighted" else "solid", label=label)
        axis.set_xlim(float(tick_positions[0]) - 0.12, 0.12)
        axis.set_ylim(0, y_max)
        axis.set_xticks(tick_positions, [rho_label(value) for value in tick_rho], rotation=45, ha="right")
        axis.minorticks_off()
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        axis.text(
            0.03, 0.96, f"({chr(97 + column)})  $\\ell_{{1/2}}={half_distances[column]:.2f}$ cells",
            transform=axis.transAxes, va="top", fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.75},
        )
    fig.supylabel("Capacity shortfall (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    handles.append((Patch(facecolor=NEUTRAL, alpha=0.18), errorbar_key))
    labels.append("5th–95th percentiles across landscapes")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.supxlabel(r"Deforestation parameter $\rho$ (log expected-random-removal scale)", y=0.21)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.40, top=0.95, wspace=0.09)
    export_figure(fig, figure_dir / "01_capacity_shortfall_by_rho_h05")
    plt.close(fig)

    habitat_levels = sorted(float(h) for h in eligible_h)
    plotted = summary[summary["H_target"].isin(habitat_levels)]
    color_max = max(5.0, 5.0 * np.ceil(float(plotted["median_shortfall_percent"].max()) / 5.0))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(MISSING)
    fig, axes = plt.subplots(2, 3, figsize=figure_size(128), sharex=True, sharey=True)
    for row, (method, label) in enumerate(methods):
        for column, metric in enumerate(metrics):
            axis = axes[row, column]
            values = np.full((len(habitat_levels), len(rho_levels) + 1), np.nan)
            for y, h in enumerate(habitat_levels):
                reference = summary[
                    summary["metric"].eq(metric) & summary["H_target"].eq(h)
                    & summary["method"].eq("only_random")
                ]
                if len(reference) == 1:
                    values[y, 0] = float(reference["median_shortfall_percent"].iloc[0])
                for x, rho in enumerate(rho_levels, start=1):
                    cell = summary[
                        summary["metric"].eq(metric) & summary["H_target"].eq(h)
                        & summary["method"].eq(method) & summary["rho"].eq(rho)
                    ]
                    if len(cell) == 1:
                        values[y, x] = float(cell["median_shortfall_percent"].iloc[0])
            image = axis.imshow(
                np.ma.masked_invalid(values), origin="lower", aspect="auto",
                interpolation="nearest", cmap=cmap, vmin=0, vmax=color_max,
            )
            axis.axvline(0.5, color="white", linewidth=1.3)
            axis.text(
                0.0, 1.01, f"({chr(97 + 3 * row + column)}) {label}\n"
                + f"$\\ell_{{1/2}}={half_distances[column]:.2f}$ cells",
                transform=axis.transAxes, va="bottom", fontsize=8.5,
            )
            axis.set_yticks(range(len(habitat_levels)), [f"{h:g}" for h in habitat_levels])
            if row == 1:
                axis.set_xticks(tick_indices, [rho_label(value) for value in tick_rho], rotation=45, ha="right")
    fig.supylabel(r"Habitat proportion $H$", x=0.015)
    fig.supxlabel(r"Scenario $\rho$ (log spacing in expected random removals)", y=0.015)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.19, top=0.91, wspace=0.08, hspace=0.30)
    color_axis = fig.add_axes((0.88, 0.19, 0.025, 0.72))
    fig.colorbar(image, cax=color_axis, label="Median capacity shortfall (%)")
    export_figure(fig, figure_dir / "01_capacity_shortfall_by_rho_h_heatmap")
    plt.close(fig)


def aggregate(config: dict, workdir: Path, resultdir: Path, scenarios: list[dict], summary_settings: dict) -> None:
    expected = [workdir / "trajectories" / f"{scenario['id']}__rep{replicate:03d}.csv"
                for scenario in scenarios for replicate in range(1, int(config["n_replicates"]) + 1)]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete run: {len(missing)} trajectories missing; first: {missing[0]}")
    data = pd.concat((pd.read_csv(path) for path in expected), ignore_index=True)
    data.to_csv(workdir / "landscape_metrics.csv", index=False)
    unique_landscapes = data.drop_duplicates(["trajectory_id", "H_target"])
    unique_landscapes.assign(excluded=lambda frame: frame["n_patches"].lt(2)).groupby(
        ["scenario_id", "method", "rho", "H_target"], as_index=False
    ).agg(
        n_generated=("trajectory_id", "count"),
        n_excluded=("excluded", "sum"),
        exclusion_fraction=("excluded", "mean"),
    ).to_csv(resultdir / "tables" / "01_exclusions_by_scenario_h.csv", index=False)
    comparisons = [
        ("edge_complete", "centroid_complete"),
        ("edge_complete", "voronoi_hybrid"),
        ("edge_complete", "voronoi_graph"),
    ]
    pair_frames = []
    for benchmark, candidate in comparisons:
        metrics = ["mean_nn", "d50"] if candidate == "voronoi_graph" else [
            "mean_nn", "mean_pair", "d50", "capacity_a1", "capacity_a2", "capacity_a3"
        ]
        for metric in metrics:
            pair_frames.append(paired_table(data, metric, benchmark, candidate))
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs.to_csv(workdir / "paired_comparisons.csv", index=False)
    capacity_pairs = pairs[
        pairs["candidate_representation"].eq("voronoi_hybrid")
        & pairs["metric"].str.startswith("capacity_a")
    ]
    capacity_rho_summary = capacity_shortfall_by_rho(capacity_pairs)
    capacity_rho_summary.to_csv(resultdir / "tables" / "01_capacity_shortfall_by_rho_h.csv", index=False)
    scenario_medians = capacity_pairs.groupby(
        ["metric", "H_target", "scenario_id", "method", "rho"], as_index=False
    ).agg(reference_median=("reference", "median"), candidate_median=("candidate", "median"))
    baselines = scenario_medians[scenario_medians["scenario_id"].eq("only_random_rho0")][
        ["metric", "H_target", "reference_median", "candidate_median"]
    ].rename(columns={
        "reference_median": "baseline_reference_median",
        "candidate_median": "baseline_candidate_median",
    })
    scenario_medians = scenario_medians.merge(baselines, on=["metric", "H_target"], validate="many_to_one")
    scenario_medians["reference_log_contrast"] = np.log(
        scenario_medians["reference_median"] / scenario_medians["baseline_reference_median"]
    )
    scenario_medians["candidate_log_contrast"] = np.log(
        scenario_medians["candidate_median"] / scenario_medians["baseline_candidate_median"]
    )
    scenario_medians.to_csv(resultdir / "tables" / "01_capacity_scenario_medians.csv", index=False)
    contrast_rows = []
    for (metric, h), group in scenario_medians[
        scenario_medians["scenario_id"].ne("only_random_rho0")
    ].groupby(["metric", "H_target"], sort=True):
        edge = group["reference_log_contrast"]
        hybrid = group["candidate_log_contrast"]
        contrast_rows.append({
            "metric": metric,
            "H_target": h,
            "n_positive_rho_scenarios": int(len(group)),
            "scenario_contrast_spearman": _spearman(edge, hybrid),
            "scenario_contrast_sign_agreement": float(np.mean(np.sign(edge) == np.sign(hybrid))),
            "median_absolute_log_contrast_gap": float(np.median(np.abs(hybrid - edge))),
        })
    pd.DataFrame(contrast_rows).to_csv(
        resultdir / "tables" / "01_capacity_scenario_contrast_preservation.csv", index=False
    )
    graph_pairs = pairs[pairs["candidate_representation"].eq("voronoi_graph")].copy()
    graph_exactness = graph_pairs.assign(exact=lambda frame: frame["difference"].abs().le(1e-10)).groupby(
        "metric", as_index=False
    ).agg(
        n_landscapes=("exact", "count"),
        n_exact=("exact", "sum"),
        maximum_absolute_difference=("difference", lambda values: float(values.abs().max())),
    )
    graph_exactness["graph_edge_weight_source"] = "exact_complete_edge_matrix"
    graph_exactness.to_csv(resultdir / "tables" / "01_graph_exactness.csv", index=False)
    hybrid_local_pairs = pairs[
        pairs["candidate_representation"].eq("voronoi_hybrid")
        & pairs["metric"].isin(("mean_nn", "d50"))
    ]
    hybrid_local_pairs.assign(
        exact=lambda frame: frame["difference"].abs().le(1e-10)
    ).groupby("metric", as_index=False).agg(
        n_landscapes=("exact", "count"),
        n_exact=("exact", "sum"),
        maximum_absolute_difference=("difference", lambda values: float(values.abs().max())),
        maximum_percentage_difference=("relative_difference", lambda values: float(100 * values.abs().max())),
    ).to_csv(resultdir / "tables" / "01_hybrid_local_metric_exactness.csv", index=False)
    summary = summarize_pairs(pairs, ["reference_representation", "candidate_representation", "metric", "H_target"])
    summary, batches, influence = adaptive_pair_intervals(summary, pairs, summary_settings, int(config["base_seed"]))
    batches.to_csv(resultdir / "tables/01_bootstrap_convergence.csv", index=False)
    influence.to_csv(resultdir / "tables/01_leave_one_seed_out.csv", index=False)
    (resultdir / "logs/01_summary_settings.json").write_text(json.dumps(summary_settings, indent=2), encoding="utf-8")
    summary.to_csv(resultdir / "tables" / "01_comparison_by_h.csv", index=False)
    exclusion_by_h = unique_landscapes.groupby("H_target")["n_patches"].apply(
        lambda values: float(values.lt(2).mean())
    )
    eligible_h = exclusion_by_h[exclusion_by_h.le(MAX_EXCLUSION_FRACTION_FOR_SELECTION)].index
    interpretable = summary[
        summary["H_target"].isin(eligible_h) & summary["spearman"].notna()
    ].copy()
    interpretable.groupby(
        ["reference_representation", "candidate_representation", "metric"], as_index=False
    ).agg(
        n_h_levels=("H_target", "count"),
        min_spearman=("spearman", "min"),
        median_spearman=("spearman", "median"),
        median_h_abs_relative_difference=("median_abs_relative_difference", "median"),
        max_h_abs_relative_difference=("median_abs_relative_difference", "max"),
        max_abs_mean_patch_size_gap_spearman=("size_gap_spearman", lambda values: float(values.abs().max())),
    ).to_csv(resultdir / "tables" / "01_decision_diagnostics.csv", index=False)
    method_summary = summarize_pairs(pairs, ["reference_representation", "candidate_representation", "metric", "method", "H_target"])
    method_summary.to_csv(resultdir / "tables" / "01_comparison_by_method_h.csv", index=False)
    rho_summary = summarize_pairs(pairs, ["reference_representation", "candidate_representation", "metric", "method", "rho", "H_target"])
    rho_summary.to_csv(resultdir / "tables" / "01_comparison_by_rho_h.csv", index=False)
    data.groupby(["method", "rho", "H_target", "representation"], as_index=False).agg(
        n_landscapes=("trajectory_id", "count"),
        median_graph_build_seconds=("graph_build_seconds", "median"),
        median_metric_seconds=("metric_seconds", "median"),
        median_patch_count=("n_patches", "median"),
    ).to_csv(resultdir / "tables" / "01_computation_cost.csv", index=False)
    # NN is a minimum over exact direct Voronoi edges. Other distance-based
    # uses take the full hybrid matrix. Occupancy was not evaluated here.
    selected_uses = [
        ("mean_nearest_neighbour", SELECTED_MEAN_NN_REPRESENTATION, "nearest_neighbor_from_exact_voronoi_edges", "exact_agreement_with_complete_edge_in_sample"),
        ("mean_interpatch", SELECTED_REPRESENTATION, "all_pair_mean_from_full_hybrid_matrix", "preserves_all_pair_estimand_with_accepted_approximation"),
        ("complete_mst_d50", SELECTED_REPRESENTATION, "complete_mst_from_full_hybrid_matrix", "exact_agreement_with_complete_edge_in_sample"),
        ("metapopulation_capacity", SELECTED_REPRESENTATION, "all_pair_dispersal_kernel_from_full_hybrid_matrix", "common_baseline_with_documented_capacity_magnitude_gap"),
        ("metapopulation_equilibrium_occupancy", SELECTED_REPRESENTATION, "all_pair_dispersal_kernel_from_full_hybrid_matrix", "common_baseline_not_directly_evaluated_in_analysis_1_1_3"),
    ]
    pd.DataFrame([
        {
            "metric_or_application": metric,
            "selected_representation": representation,
            "distance_mode": SELECTED_DISTANCE_MODE,
            "voronoi_depth": SELECTED_VORONOI_DEPTH,
            "pair_scope": pair_scope,
            "production_source": (
                "Landscape.ensure_voronoi_neighbor_graph"
                if metric == "mean_nearest_neighbour" else "Landscape.cluster_dist_matrix"
            ),
            "decision_basis": decision_basis,
        }
        for metric, representation, pair_scope, decision_basis in selected_uses
    ]).to_csv(resultdir / "tables" / "01_selected_representations.csv", index=False)
    make_figures(summary[summary["H_target"].isin(eligible_h)], resultdir / "figures")
    make_capacity_rho_figures(
        capacity_rho_summary, resultdir / "figures",
        [float(value) for value in config["half_distances_cells"]],
        [float(value) for value in eligible_h],
        int(config["n_removed_at_rho_reference_h"]),
    )
    (resultdir / "logs" / "01_data_summary.json").write_text(json.dumps({
        "n_trajectories": int(data["trajectory_id"].nunique()),
        "n_independent_seed_blocks": int(data["seed"].nunique()),
        "n_snapshots": int(data[["trajectory_id", "H_target"]].drop_duplicates().shape[0]),
        "n_landscapes_with_fewer_than_two_patches": int(data.loc[data["n_patches"].lt(2), ["trajectory_id", "H_target"]].drop_duplicates().shape[0]),
        "selected_all_pair_distance_representation": SELECTED_REPRESENTATION,
        "selected_mean_nn_representation": SELECTED_MEAN_NN_REPRESENTATION,
        "selected_mean_nn_rule": "mean of patchwise minimum exact direct Voronoi-neighbour edge distances",
        "selected_mean_nn_implementation": "Landscape.ensure_voronoi_neighbor_graph; nearest_neighbor_distances_from_voronoi_graph",
        "selected_distance_mode": SELECTED_DISTANCE_MODE,
        "selected_voronoi_depth": SELECTED_VORONOI_DEPTH,
        "maximum_exclusion_fraction_for_figures_and_decisions": MAX_EXCLUSION_FRACTION_FOR_SELECTION,
        "eligible_h_for_figures_and_decisions": [float(value) for value in eligible_h],
        "exclusion_fraction_by_h": {str(value): float(fraction) for value, fraction in exclusion_by_h.items()},
        "capacity_nonconvergence": {
            f"alpha_{i}": int((
                ~data[f"capacity_converged_a{i}"].fillna(False).astype(bool)
                & data["n_patches"].ge(2)
                & data["representation"].ne("voronoi_graph")
            ).sum()) for i in (1, 2, 3)
        },
    }, indent=2), encoding="utf-8")
    (resultdir / "logs" / "01_run_metadata.json").write_text(json.dumps({
        "analysis_plan_section": "1.1.3",
        "configuration_manifest_mtime_utc": datetime.fromtimestamp((workdir / "run_config.json").stat().st_mtime, timezone.utc).isoformat(),
        "summary_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        **summary_provenance(summary_settings),
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trajectory_source_fingerprint_status": "not_recorded_in_original_trajectory_files",
        "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "processed_directory": str(workdir),
        "results_directory": str(resultdir),
        "seed_block_count": int(data["seed"].nunique()),
        "exclusion_rule": "Fewer than two patches or a metric-specific undefined/nonconverged value",
    }, indent=2), encoding="utf-8")


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    workdir = args.processed_dir
    resultdir = args.results_dir
    (workdir / "trajectories").mkdir(parents=True, exist_ok=True)
    for folder in (resultdir / "tables", resultdir / "logs", resultdir / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
    manifest_path = workdir / "run_config.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        non_data_keys = {"bootstrap_draws"}
        if {k: v for k, v in previous.items() if k not in non_data_keys} != {k: v for k, v in config.items() if k not in non_data_keys}:
            raise RuntimeError("Existing trajectory data use another configuration; move the old directory first.")
        if previous != config:
            manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    scenarios = scenario_grid(config)
    pd.DataFrame(scenarios).to_csv(workdir / "scenarios.csv", index=False)
    if not args.summarize_only:
        jobs = []
        for scenario in scenarios:
            for replicate in range(1, int(config["n_replicates"]) + 1):
                path = workdir / "trajectories" / f"{scenario['id']}__rep{replicate:03d}.csv"
                if path.exists():
                    continue
                jobs.append((scenario, replicate, path))
        if args.max_trajectories:
            jobs = jobs[:args.max_trajectories]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(trajectory_rows, scenario, replicate, config): (scenario, replicate, path)
                for scenario, replicate, path in jobs
            }
            for count, future in enumerate(as_completed(futures), start=1):
                scenario, replicate, path = futures[future]
                save_trajectory(future.result(), path)
                print(f"[{count}/{len(jobs)}] {scenario['id']} rep={replicate}", flush=True)
        if args.max_trajectories:
            return
    aggregate(config, workdir, resultdir, scenarios, json.loads(args.summary_config.read_text(encoding="utf-8")))
    print(f"Completed {len(scenarios) * int(config['n_replicates'])} trajectories.", flush=True)


if __name__ == "__main__":
    main()
