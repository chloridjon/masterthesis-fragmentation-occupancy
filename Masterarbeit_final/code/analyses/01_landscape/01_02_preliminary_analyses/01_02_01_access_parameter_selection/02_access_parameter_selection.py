"""Final analysis 1.2.1: reproducible Access-parameter sensitivity.

Run from the repository root with its existing virtual environment. Each
parameter combination and seed produces one complete sequential trajectory.
The k=1 lambda settings are exactly equivalent and generated only once.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FINAL = next(parent for parent in Path(__file__).resolve().parents if parent.name == "Masterarbeit_final")
ROOT = FINAL.parent
REL = Path("01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection")
sys.path.insert(0, str(FINAL / "code/base"))

import numpy as np
import pandas as pd

from src.tools.analysis_stability import factor_seed_influence, summary_provenance
from src.landscape.generation import run_sequential_deforestation
from src.landscape.landscape import Landscape
from src.landscape.landscape_stats import compute_landscape_metrics_from_landscape


METRICS = (
    "edge_density_total",
    "largest_cluster_fraction_forest",
    "chi",
    "mean_nearest_neighbor_distance",
    "mean_pairwise_distance",
    "complete_mst_d50",
)
METRIC_LABELS = {
    "edge_density_total": "Edge density",
    "largest_cluster_fraction_forest": "Largest cluster fraction",
    "chi": r"Susceptibility $\chi$",
    "mean_nearest_neighbor_distance": "Nearest-neighbour distance",
    "mean_pairwise_distance": "Mean inter-patch distance",
    "complete_mst_d50": r"Complete-MST $D_{50}$",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FINAL / "config" / REL / "03_focal_rho_090_099_0999.json")
    parser.add_argument("--processed-dir", type=Path, default=FINAL / "data/processed" / REL / "focal_rho_090_099_0999")
    parser.add_argument("--results-dir", type=Path, default=FINAL / "results" / REL / "focal_rho_090_099_0999")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary-config", type=Path, default=FINAL / "config" / REL / "04_summary_stability.json")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=0, help="Development limit; does not summarize incomplete samples.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config["grid_size_cells"]) != 200:
        raise ValueError("The final grid must have 200 x 200 cells.")
    if config["generation_boundary"] != "periodic" or config["analysis_boundary"] != "open":
        raise ValueError("Generation must be periodic and analysis open.")
    if config["cluster_connectivity"] != "4" or config["distance_mode"] != "voronoi_dt" or int(config["voronoi_depth"]) != 1:
        raise ValueError("Metric definitions must use four-neighbour patches and the depth-1 Voronoi hybrid.")
    if [float(v) for v in config["eta_values"]] != [0.5, 1.0, 2.0, 4.0]:
        raise ValueError("Unexpected eta grid.")
    if [int(v) for v in config["k_values"]] != [1, 3, 5]:
        raise ValueError("Unexpected k grid.")
    if [float(v) for v in config["lambda_values"]] != [0.1, 0.5, 1.0]:
        raise ValueError("Unexpected lambda grid.")
    if float(config["canonical_lambda_when_k_is_one"]) != 0.5:
        raise ValueError("The canonical k=1 lambda must be a grid member.")
    if len(config["rho_values"]) != 3 or not np.all(np.diff(np.asarray(config["rho_values"], dtype=float)) > 0):
        raise ValueError("Expected three increasing directed rho values.")
    if not 0.0 < float(config["rho_values"][0]) < float(config["rho_values"][1]) < float(config["rho_values"][2]) <= 1.0:
        raise ValueError("The three directed rho values must increase within (0, 1].")
    if [float(v) for v in config["h_targets"]] != [0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.01]:
        raise ValueError("H targets must match analysis 1.1.3.")
    if tuple(config["selected_structural_metrics"]) != METRICS:
        raise ValueError("Metric selection differs from the six structural metrics.")
    if int(config["n_replicates"]) != 10:
        raise ValueError("Ten independent seed blocks are required.")
    return config


def unique_settings(config: dict) -> list[dict]:
    settings = []
    lam_values = [float(v) for v in config["lambda_values"]]
    canonical = float(config["canonical_lambda_when_k_is_one"])
    for rho_idx, rho in enumerate(config["rho_values"], start=1):
        for eta_idx, eta in enumerate(config["eta_values"], start=1):
            for k in config["k_values"]:
                for lam_idx, lam in enumerate(lam_values, start=1):
                    if k == 1 and lam != canonical:
                        continue
                    settings.append({
                        "rho_index": rho_idx, "rho": float(rho),
                        "eta_index": eta_idx, "eta": float(eta),
                        "k": int(k), "lambda_index": lam_idx, "lambda": float(lam),
                        "setting_id": f"rho{rho_idx:02d}_eta{eta_idx:02d}_k{k}_lambda{lam_idx:02d}",
                    })
    if len(settings) != 84:
        raise AssertionError("Expected 28 distinct parameter rules at each of three rho values.")
    return settings


def producer_fingerprint(config: dict) -> str:
    """Hash all code that produces trajectory rows, excluding summary-only code."""
    sources = [
        FINAL / "code/base/src/landscape/generation.py",
        FINAL / "code/base/src/landscape/deforestation.py",
        FINAL / "code/base/src/landscape/landscape.py",
        FINAL / "code/base/src/landscape/landscape_stats.py",
        FINAL / "code/base/src/landscape/boundary.py",
        FINAL / "code/base/src/landscape/complete_edge_distances.py",
    ]
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode())
    for source in sources:
        digest.update(str(source.relative_to(FINAL)).encode())
        digest.update(source.read_bytes())
    for function in (unique_settings, snapshot_metrics, trajectory_rows):
        digest.update(inspect.getsource(function).encode())
    digest.update(repr(METRICS).encode())
    return digest.hexdigest()


def snapshot_metrics(grid: np.ndarray, config: dict) -> dict[str, float | int]:
    landscape = Landscape(grid)
    landscape.ensure_clusters(
        edge_depth=0.0,
        connectivity=config["cluster_connectivity"],
        boundary_mode=config["analysis_boundary"],
        distance_mode=config["distance_mode"],
        voronoi_depth=config["voronoi_depth"],
        rebuild=True,
    )
    metrics = compute_landscape_metrics_from_landscape(
        landscape,
        metric_keys=[
            "n_clusters", "edge_density_total", "largest_cluster_fraction_forest",
            "chi", "mean_nearest_neighbor_distance", "complete_mst_d50",
        ],
    )
    n = int(landscape.n_clusters)
    if n != int(metrics["n_clusters"]):
        raise AssertionError("Prepared and reported patch counts differ.")
    if n >= 2:
        distance = np.asarray(landscape.cluster_dist_matrix, dtype=float)
        pairs = distance[np.triu_indices(n, k=1)]
        mean_pair = float(pairs.mean())
    else:
        mean_pair = np.nan
    metrics["mean_pairwise_distance"] = mean_pair
    metrics["n_patches"] = n
    return metrics


def trajectory_rows(setting: dict, replicate: int, config: dict, fingerprint: str) -> pd.DataFrame:
    seed = int(config["base_seed"]) + replicate - 1
    run = run_sequential_deforestation(
        np.ones((int(config["grid_size_cells"]), int(config["grid_size_cells"])), dtype=np.int8),
        np.random.default_rng(seed),
        rho=float(setting["rho"]),
        directed_mode="access_weighted",
        target_habitat_fraction=min(float(v) for v in config["h_targets"]),
        generation_boundary=config["generation_boundary"],
        k=int(setting["k"]),
        lambda_decay=float(setting["lambda"]),
        eta=float(setting["eta"]),
        save_habitat_fractions=[float(v) for v in config["h_targets"]],
        include_initial_snapshot=False,
        seed=seed,
    )
    if run.n_random_total + run.n_directed != run.n_events or run.n_random_forced_initial != 1:
        raise AssertionError("The event trace does not implement the planned removal mixture.")
    random_so_far = np.concatenate((
        [0], np.cumsum(np.asarray(run.event_modes) == "random", dtype=np.int64)
    ))
    rows = []
    for snapshot in run.snapshots:
        start = time.perf_counter()
        metrics = snapshot_metrics(snapshot.grid, config)
        event_count = int(snapshot.event_count)
        h_real = float(snapshot.habitat_fraction)
        if not np.isclose(h_real, 1.0 - event_count / float(run.total_cells), atol=1e-12):
            raise AssertionError("Snapshot habitat fraction does not match event count.")
        rows.append({
            **setting,
            "replicate": replicate,
            "seed": seed,
            "trajectory_id": f"{setting['setting_id']}__rep{replicate:03d}",
            "H_target": float(snapshot.requested_habitat_fractions[0]),
            "H_real": h_real,
            "event_count": event_count,
            "n_random_cumulative": int(random_so_far[event_count]),
            "n_directed_cumulative": int(event_count - random_so_far[event_count]),
            "grid_sha256": hashlib.sha256(snapshot.grid.tobytes()).hexdigest(),
            "producer_fingerprint": fingerprint,
            "generator_version": run.generator_version,
            "metric_seconds": time.perf_counter() - start,
            **metrics,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != len(config["h_targets"]):
        raise AssertionError("A trajectory did not produce every requested H snapshot.")
    return frame


def trajectory_path(workdir: Path, setting: dict, replicate: int) -> Path:
    return workdir / "trajectories" / f"{setting['setting_id']}__rep{replicate:03d}.csv"


def validate_cached_trajectory(path: Path, setting: dict, replicate: int, config: dict, fingerprint: str) -> None:
    frame = pd.read_csv(path)
    if len(frame) != len(config["h_targets"]):
        raise RuntimeError(f"Incomplete cached trajectory: {path}")
    if frame["producer_fingerprint"].nunique() != 1 or frame["producer_fingerprint"].iloc[0] != fingerprint:
        raise RuntimeError(f"Cached trajectory uses a different code fingerprint: {path}")
    if frame["trajectory_id"].nunique() != 1 or frame["trajectory_id"].iloc[0] != f"{setting['setting_id']}__rep{replicate:03d}":
        raise RuntimeError(f"Cached trajectory has the wrong identity: {path}")
    if not np.array_equal(np.sort(frame["H_target"].to_numpy(dtype=float)), np.sort(config["h_targets"])):
        raise RuntimeError(f"Cached trajectory has the wrong H values: {path}")


def save_trajectory(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def expand_equivalent_lambda(canonical: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Materialize the full factorial while identifying k=1 duplicate rules."""
    parts = [canonical.assign(lambda_reused=False, generated_lambda=canonical["lambda"])]
    template = canonical[canonical["k"].eq(1)]
    for lam_idx, lam in enumerate(config["lambda_values"], start=1):
        if float(lam) == float(config["canonical_lambda_when_k_is_one"]):
            continue
        copied = template.copy()
        copied["lambda"] = float(lam)
        copied["lambda_index"] = lam_idx
        copied["setting_id"] = copied.apply(
            lambda row: f"rho{int(row['rho_index']):02d}_eta{int(row['eta_index']):02d}_k1_lambda{lam_idx:02d}",
            axis=1,
        )
        copied["trajectory_id"] = copied.apply(
            lambda row: f"{row['setting_id']}__rep{int(row['replicate']):03d}", axis=1
        )
        copied["lambda_reused"] = True
        copied["generated_lambda"] = float(config["canonical_lambda_when_k_is_one"])
        parts.append(copied)
    return pd.concat(parts, ignore_index=True)


def metric_summary(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["rho_index", "rho", "eta", "k", "lambda", "H_target"]
    records = []
    for key, part in data.groupby(keys, sort=True):
        common = dict(zip(keys, key))
        n_patches = part["n_patches"].to_numpy(dtype=int)
        for metric in METRICS:
            values = part[metric].to_numpy(dtype=float)
            valid = values[np.isfinite(values)]
            records.append({
                **common, "metric": metric,
                "n_seed_blocks": int(len(part)), "n_valid": int(valid.size),
                "median": float(np.median(valid)) if valid.size else np.nan,
                "mean": float(valid.mean()) if valid.size else np.nan,
                "p05": float(np.quantile(valid, 0.05)) if valid.size else np.nan,
                "p95": float(np.quantile(valid, 0.95)) if valid.size else np.nan,
                "n_fewer_than_two_patches": int((n_patches < 2).sum()),
            })
    return pd.DataFrame(records)


def factorial_components(values: np.ndarray) -> dict[str, float]:
    """Balanced, descriptive ANOVA decomposition of the 4 x 3 x 3 cell medians."""
    if values.shape != (4, 3, 3) or not np.all(np.isfinite(values)):
        raise ValueError("A complete finite 4 x 3 x 3 grid is required.")
    grand = float(values.mean())
    eta = values.mean(axis=(1, 2)) - grand
    k = values.mean(axis=(0, 2)) - grand
    lam = values.mean(axis=(0, 1)) - grand
    eta_k = values.mean(axis=2) - grand - eta[:, None] - k[None, :]
    eta_lam = values.mean(axis=1) - grand - eta[:, None] - lam[None, :]
    k_lam = values.mean(axis=0) - grand - k[:, None] - lam[None, :]
    three = (
        values - grand - eta[:, None, None] - k[None, :, None] - lam[None, None, :]
        - eta_k[:, :, None] - eta_lam[:, None, :] - k_lam[None, :, :]
    )
    components = {
        "eta": eta[:, None, None],
        "k": k[None, :, None],
        "lambda": lam[None, None, :],
        "eta_k": eta_k[:, :, None],
        "eta_lambda": eta_lam[:, None, :],
        "k_lambda": k_lam[None, :, :],
        "eta_k_lambda": three,
    }
    total = float(np.square(values - grand).sum())
    sums = {
        name: float(np.square(np.broadcast_to(component, values.shape)).sum())
        for name, component in components.items()
    }
    if not np.isclose(sum(sums.values()), total, atol=1e-8, rtol=1e-8):
        raise AssertionError("Factorial components do not reconstruct the total variation.")
    result = {f"share_{name}": (amount / total if total > 1e-14 else np.nan) for name, amount in sums.items()}
    result.update({
        "grand_mean_of_cell_medians": grand,
        "total_sum_squares": total,
        "range_eta": float(np.ptp(values.mean(axis=(1, 2)))),
        "range_k": float(np.ptp(values.mean(axis=(0, 2)))),
        "range_lambda": float(np.ptp(values.mean(axis=(0, 1)))),
        "delta_eta_high_minus_low": float(eta[-1] - eta[0]),
        "delta_k_high_minus_low": float(k[-1] - k[0]),
        "delta_lambda_high_minus_low": float(lam[-1] - lam[0]),
    })
    return result


def factor_summary(summary: pd.DataFrame, config: dict) -> pd.DataFrame:
    records = []
    for (rho_idx, rho, h, metric), part in summary.groupby(
        ["rho_index", "rho", "H_target", "metric"], sort=True
    ):
        values = np.full((4, 3, 3), np.nan, dtype=float)
        n_valid = np.zeros_like(values, dtype=int)
        for _, row in part.iterrows():
            i = list(config["eta_values"]).index(row["eta"])
            j = list(config["k_values"]).index(row["k"])
            l = list(config["lambda_values"]).index(row["lambda"])
            values[i, j, l] = row["median"]
            n_valid[i, j, l] = row["n_valid"]
        record = {
            "rho_index": int(rho_idx), "rho": float(rho),
            "H_target": float(h), "metric": metric,
            "n_finite_cells": int(np.isfinite(values).sum()),
            "min_seed_replicates_per_cell": int(n_valid.min()),
            "sufficient_replication": bool(n_valid.min() >= 8),
        }
        if np.all(np.isfinite(values)):
            record.update(factorial_components(values))
        records.append(record)
    return pd.DataFrame(records)


def make_factor_figures(factors: pd.DataFrame, config: dict, figure_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.tools.final_plot_style import apply_final_style, export_figure, figure_size

    apply_final_style()
    # Keep labels above 8 pt when these appendix figures use 95% text width.
    plt.rcParams.update({
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
    })
    factor_style = {
        "eta": ("#0072B2", "solid", r"$\eta$"),
        "k": ("#E69F00", "solid", r"$k$"),
        "lambda": ("#009E73", "solid", r"$\lambda$"),
    }
    for rho_idx, rho in enumerate(config["rho_values"], start=1):
        fig, axes = plt.subplots(2, 3, figsize=figure_size(122.0), sharex=True, sharey=True)
        subset = factors[factors["rho_index"].eq(rho_idx)]
        for panel, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
            frame = subset[subset["metric"].eq(metric)].sort_values("H_target").copy()
            frame.loc[~frame["sufficient_replication"], [
                "share_eta", "share_k", "share_lambda",
            ]] = np.nan
            for factor, (color, linestyle, label) in factor_style.items():
                ax.plot(
                    frame["H_target"], frame[f"share_{factor}"],
                    color=color, linestyle=linestyle, marker="o", markersize=2.8,
                    label=label,
                )
            sensitive = frame["seed_influence_status"].eq("sensitive")
            for factor, (color, _, _) in factor_style.items():
                ax.scatter(frame.loc[sensitive, "H_target"], frame.loc[sensitive, f"share_{factor}"], facecolors="white", edgecolors=color, s=20, zorder=5)
            if frame["share_eta"].notna().sum() == 0:
                reason = (
                    "No between-setting\nvariation"
                    if subset[subset["metric"].eq(metric)]["total_sum_squares"].fillna(0).eq(0).all()
                    else "Insufficient\nreplication"
                )
                ax.text(
                    0.5, 0.5, reason,
                    transform=ax.transAxes, ha="center", va="center",
                    color="#404040", fontsize=8.5,
                )
            ax.text(
                0.0, 1.025, f"({chr(97 + panel)}) {METRIC_LABELS[metric]}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0,
                clip_on=False,
            )
            ax.set_xlim(1, 0)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            # Keep endpoint labels inside each panel to avoid adjacent collisions.
            for value, tick_label in zip(ax.get_xticks(), ax.get_xticklabels()):
                if value == 1.0:
                    tick_label.set_horizontalalignment("left")
                elif value == 0.0:
                    tick_label.set_horizontalalignment("right")
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        fig.supxlabel("Habitat proportion $H$", y=0.065)
        fig.supylabel("Main-effect share of variation", x=0.015)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D
        if subset["seed_influence_status"].eq("sensitive").any():
            handles.append(Line2D([], [], marker="o", markerfacecolor="white", color="#404040", linestyle="none"))
            labels.append("Sensitive to seed omission")
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.subplots_adjust(left=0.11, right=0.985, top=0.91, bottom=0.17, wspace=0.15, hspace=0.3)
        export_figure(fig, figure_dir / f"02_factor_main_effect_shares_rho{rho_idx:02d}")
        plt.close(fig)


def aggregate(config: dict, workdir: Path, resultdir: Path, settings: list[dict], fingerprint: str, summary_settings: dict) -> None:
    frames = []
    for setting in settings:
        for replicate in range(1, int(config["n_replicates"]) + 1):
            path = trajectory_path(workdir, setting, replicate)
            if not path.exists():
                raise RuntimeError(f"Missing trajectory: {path}")
            validate_cached_trajectory(path, setting, replicate, config, fingerprint)
            frames.append(pd.read_csv(path))
    canonical = pd.concat(frames, ignore_index=True)
    if canonical["trajectory_id"].nunique() != 840 or len(canonical) != 840 * 8:
        raise AssertionError("The canonical trajectory sample is incomplete.")
    full = expand_equivalent_lambda(canonical, config)
    if full["trajectory_id"].nunique() != 1080 or len(full) != 1080 * 8:
        raise AssertionError("The expanded factorial sample is incomplete.")
    if full.duplicated(["trajectory_id", "H_target"]).any():
        raise AssertionError("Duplicate trajectory/H key.")
    if not np.allclose(full["H_target"].to_numpy(), full["H_real"].to_numpy(), atol=1e-12):
        raise AssertionError("Habitat snapshots do not match their targets.")
    if not np.array_equal(
        (full["n_random_cumulative"] + full["n_directed_cumulative"]).to_numpy(dtype=int),
        full["event_count"].to_numpy(dtype=int),
    ):
        raise AssertionError("Removal mechanism counts do not sum to event counts.")
    if not np.all(full.loc[full["rho"].eq(1), "n_random_cumulative"].eq(1)):
        raise AssertionError("At rho=1 only the forced first removal may be random.")

    canonical.to_csv(workdir / "canonical_landscape_metrics.csv", index=False)
    full.to_csv(workdir / "landscape_metrics.csv", index=False)
    summary = metric_summary(full)
    factors, seed_influence = factor_seed_influence(full, config, metric_summary, factor_summary, summary_settings)
    seed_influence.to_csv(resultdir / "tables/02_leave_one_seed_out.csv", index=False)
    (resultdir / "logs/02_summary_settings.json").write_text(json.dumps(summary_settings, indent=2), encoding="utf-8")
    if len(summary) != 3 * 4 * 3 * 3 * 8 * len(METRICS):
        raise AssertionError("Metric summary does not cover the full parameter grid.")
    if len(factors) != 3 * 8 * len(METRICS):
        raise AssertionError("Factor summary does not cover every rho, H, and metric.")
    summary.to_csv(resultdir / "tables/02_metric_summary.csv", index=False)
    factors.to_csv(resultdir / "tables/02_factor_decomposition.csv", index=False)
    full.groupby(["rho_index", "rho", "eta", "k", "lambda", "H_target"], as_index=False).agg(
        n_seed_blocks=("seed", "nunique"),
        n_fewer_than_two_patches=("n_patches", lambda values: int((values < 2).sum())),
        median_patch_count=("n_patches", "median"),
        median_largest_cluster_fraction=("largest_cluster_fraction_forest", "median"),
        median_edge_density=("edge_density_total", "median"),
    ).to_csv(resultdir / "tables/02_structure_flags.csv", index=False)
    make_factor_figures(factors, config, resultdir / "figures")

    random_counts = [
        {
            "rho": float(rho),
            "expected_random_removals_at_H_0_01": float(1 + (39600 - 1) * (1 - float(rho))),
        }
        for rho in config["rho_values"]
    ]
    (resultdir / "logs/02_data_summary.json").write_text(json.dumps({
        "analysis_plan_section": "1.2.1",
        **summary_provenance(summary_settings),
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unique_generated_trajectories": 840,
        "factorial_parameter_labels": 36,
        "expanded_trajectory_labels": 1080,
        "independent_seed_blocks": int(full["seed"].nunique()),
        "snapshots_per_trajectory": 8,
        "structural_metrics": list(METRICS),
        "rho_grid": random_counts,
        "k1_lambda_equivalence": "Three lambda labels share one generated trajectory per eta, rho, and seed.",
        "producer_fingerprint": fingerprint,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    if args.workers < 1 or args.max_trajectories < 0:
        raise ValueError("workers must be positive and max-trajectories nonnegative.")
    fingerprint = producer_fingerprint(config)
    workdir = args.processed_dir
    resultdir = args.results_dir
    (workdir / "trajectories").mkdir(parents=True, exist_ok=True)
    for folder in ("figures", "tables", "logs"):
        (resultdir / folder).mkdir(parents=True, exist_ok=True)
    manifest = workdir / "run_manifest.json"
    expected = {"config": config, "producer_fingerprint": fingerprint}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Existing trajectory directory uses another configuration or producer code.")
    else:
        if any((workdir / "trajectories").glob("*.csv")):
            raise RuntimeError("Trajectory files exist without a producer manifest.")
        manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    settings = unique_settings(config)
    if not args.summarize_only:
        jobs = []
        for setting in settings:
            for replicate in range(1, int(config["n_replicates"]) + 1):
                path = trajectory_path(workdir, setting, replicate)
                if path.exists():
                    validate_cached_trajectory(path, setting, replicate, config, fingerprint)
                else:
                    jobs.append((setting, replicate, path))
        if args.max_trajectories:
            jobs = jobs[:args.max_trajectories]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(trajectory_rows, setting, replicate, config, fingerprint): path
                for setting, replicate, path in jobs
            }
            for count, future in enumerate(as_completed(futures), start=1):
                save_trajectory(future.result(), futures[future])
                if count == 1 or count % 10 == 0 or count == len(jobs):
                    print(f"Finished {count}/{len(jobs)} new trajectories", flush=True)
        if args.max_trajectories:
            return
    aggregate(config, workdir, resultdir, settings, fingerprint, json.loads(args.summary_config.read_text(encoding="utf-8")))
    print("Completed analysis 1.2.1: 840 unique trajectories.", flush=True)


if __name__ == "__main__":
    main()
