"""Exploratory geometry check for Chapter 4 TODO 4-06.

Regenerate 30 prespecified matched seed blocks at three forest proportions;
require archived raster hashes and cluster counts to match. Compare the
archived hybrid all-pair mean with the centroid all-pair mean and the spatial
spread of forest cells. This diagnoses the metric; it is not a new scenario
selection or a species-level connectivity analysis. Raw inputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.generation import run_sequential_deforestation  # noqa: E402
from src.landscape.landscape import Landscape  # noqa: E402
from src.tools.final_plot_style import (  # noqa: E402
    LANDSCAPE_COLORS, apply_final_style, export_figure, figure_size,
)

SCENARIOS = ("only_random_rho0", "edge_random_rho0999", "access_weighted_rho0999")
LABELS = (r"$\mathit{only\ random}$", r"$\mathit{edge\ random}$", r"$\mathit{access\ based}$")
H_LEVELS = (.775, .3, .025)
REPLICATES = tuple(int(x) for x in np.rint(np.linspace(1, 200, 30)))
OUT = FINAL / "results" / REL / "interpatch_geometry_diagnostic"
STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def rms_radius(points: np.ndarray) -> float:
    return float(np.sqrt(np.square(points - points.mean(axis=0)).sum(axis=1).mean()))


def centroid_pair_mean(points: np.ndarray) -> float:
    """All ordered off-diagonal pairs, in blocks to avoid a dense n*n array."""
    n = len(points)
    total = sum(float(cdist(points[start:start + 128], points).sum())
                for start in range(0, n, 128))
    return total / (n * (n - 1))


def inspect(grid: np.ndarray, archived: pd.Series) -> dict:
    assert hashlib.sha256(grid.tobytes()).hexdigest() == archived.grid_sha256
    assert int((grid == 1).sum()) == round(float(archived.H_target) * 40_000)
    labels, n = ndimage.label(grid == 1, structure=STRUCTURE)
    assert n == int(archived.n_clusters) and n >= 2
    areas = np.bincount(labels.ravel())[1:]
    coords = np.argwhere(grid == 1)
    centroids = np.asarray(ndimage.center_of_mass(grid, labels, range(1, n + 1)))
    centroid_mean = centroid_pair_mean(centroids)
    hybrid_mean = float(archived.mean_pairwise_distance)
    checked = False
    neighbour_fraction = np.nan
    if n <= 1200:
        landscape = Landscape(grid)
        landscape.ensure_clusters(edge_depth=0, connectivity="4", boundary_mode="open",
                                  distance_mode="voronoi_dt", voronoi_depth=1, rebuild=True)
        recomputed = float(landscape.cluster_dist_matrix.sum()) / (n * (n - 1))
        assert np.isclose(recomputed, hybrid_mean, rtol=0, atol=1e-9)
        neighbour_fraction = landscape.voronoi_neighbor_graph.nnz / (n * (n - 1))
        checked = True
    return {
        "scenario_id": archived.scenario_id, "replicate": int(archived.replicate),
        "seed": int(archived.seed), "H_target": float(archived.H_target),
        "grid_sha256": archived.grid_sha256, "n_clusters": n,
        "hybrid_pair_mean_cells": hybrid_mean,
        "centroid_pair_mean_cells": centroid_mean,
        "centroid_minus_hybrid_cells": centroid_mean - hybrid_mean,
        "centroid_rms_radius_cells": rms_radius(centroids),
        "forest_cell_rms_radius_cells": rms_radius(coords),
        "direct_neighbour_fraction_of_pairs": neighbour_fraction,
        "canonical_mean_independently_recomputed": checked,
        "largest_cluster_fraction_forest": float(areas.max() / areas.sum()),
        "mean_nn_cells": float(archived.mean_nearest_neighbor_distance),
        "complete_mst_d50_cells": float(archived.complete_mst_d50),
    }


def render(records: pd.DataFrame) -> int:
    # Select one matched seed closest to the nine group medians in the
    # preselected sample, scaling deviations by the corresponding sample SD.
    d = records.copy()
    d["deviation"] = d.groupby(["scenario_id", "H_target"])["hybrid_pair_mean_cells"].transform(
        lambda v: (v - v.median()).abs() / v.std())
    scores = d.groupby("replicate").deviation.mean().sort_values()
    replicate = int(scores.index[0])
    cmap = ListedColormap([LANDSCAPE_COLORS["non_habitat"], LANDSCAPE_COLORS["habitat"]])
    apply_final_style()
    fig, axes = plt.subplots(3, 3, figsize=figure_size(171))
    fig.subplots_adjust(left=.045, right=.99, top=.965, bottom=.075, wspace=.12, hspace=.29)
    for col, sid in enumerate(SCENARIOS):
        cache = np.load(OUT / "grids" / f"{sid}_rep{replicate:03d}.npz")
        for row, h in enumerate(H_LEVELS):
            ax = axes[row, col]
            grid = cache[f"h{h:g}"]
            record = d[(d.scenario_id == sid) & (d.replicate == replicate) & (d.H_target == h)].iloc[0]
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0, 1.035, f"({chr(97 + 3 * row + col)}) " + LABELS[col],
                    transform=ax.transAxes, fontsize=9, va="bottom")
            ax.text(.5, -.035,
                    f"{int(record.n_clusters):,} clusters; mean = {record.hybrid_pair_mean_cells:.1f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)
    for row, h in enumerate(H_LEVELS):
        axes[row, 0].text(-.065, .5, f"$H={h:g}$", transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center", fontsize=9)
    fig.legend(handles=[Patch(facecolor=LANDSCAPE_COLORS["habitat"], label="Forest"),
                        Patch(facecolor=LANDSCAPE_COLORS["non_habitat"], label="Non-forest")],
               loc="lower center", bbox_to_anchor=(.5, .01), ncol=2, frameon=False)
    export_figure(fig, OUT / "12_matched_landscapes")
    fig.savefig(OUT / "12_matched_landscapes.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return replicate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30, help="Initial prespecified seed blocks, 1–30")
    parser.add_argument("--render-only", action="store_true", help="Render the saved diagnostic table and verified raster cache")
    args = parser.parse_args()
    if args.render_only:
        d = pd.read_csv(OUT / "12_landscape_geometry.csv")
        log = json.loads((OUT / "12_audit.json").read_text())
        assert render(d) == log["figure_replicate"]
        return
    assert 1 <= args.limit <= 30
    config_path = FINAL / "config" / REL / "07_structural_scenario_responses.json"
    config = json.loads(config_path.read_text())
    assert config["grid_size_cells"] == 200 and config["generation_boundary"] == "periodic"
    assert config["analysis_boundary"] == "open" and config["cluster_connectivity"] == "4"
    assert config["distance_mode"] == "voronoi_dt" and config["voronoi_depth"] == 1
    source_path = FINAL / "data/processed" / REL / "landscape_metrics.csv"
    archived = pd.read_csv(source_path)
    assert not archived.duplicated(["scenario_id", "replicate", "H_target"]).any()
    scenario_config = {s["scenario_id"]: s for s in config["scenarios"]}
    (OUT / "grids").mkdir(parents=True, exist_ok=True)
    records = []
    start = time.monotonic()
    for rep in REPLICATES[:args.limit]:
        for sid in SCENARIOS:
            cache_path = OUT / "grids" / f"{sid}_rep{rep:03d}.npz"
            seed = config["base_seed"] + rep - 1
            if cache_path.exists():
                with np.load(cache_path) as saved:
                    grids = {h: saved[f"h{h:g}"] for h in H_LEVELS}
            else:
                scenario = scenario_config[sid]
                access = config["access_reference"]
                run = run_sequential_deforestation(
                    np.ones((200, 200), dtype=np.int8), np.random.default_rng(seed),
                    rho=float(scenario["rho"]), directed_mode=scenario["mode"],
                    target_habitat_fraction=min(H_LEVELS), generation_boundary="periodic",
                    eta=access["eta"], k=access["k"], lambda_decay=access["lambda"],
                    save_habitat_fractions=H_LEVELS, seed=seed)
                grids = {round(s.habitat_fraction, 6): s.grid for s in run.snapshots}
                assert set(grids) == set(H_LEVELS)
                # Never cache an unverified raster.
                for h, grid in grids.items():
                    row = archived[(archived.scenario_id == sid) & (archived.replicate == rep)
                                   & (archived.H_target == h)].iloc[0]
                    assert hashlib.sha256(grid.tobytes()).hexdigest() == row.grid_sha256
                np.savez_compressed(cache_path, **{f"h{h:g}": grid for h, grid in grids.items()})
            for h in H_LEVELS:
                rows = archived[(archived.scenario_id == sid) & (archived.replicate == rep)
                                & (archived.H_target == h)]
                assert len(rows) == 1 and int(rows.iloc[0].seed) == seed
                records.append(inspect(grids[h], rows.iloc[0]))
        print(f"Checked matched block {rep}: {len(records)} snapshots; {time.monotonic()-start:.1f}s", flush=True)
    d = pd.DataFrame(records)
    assert len(d) == args.limit * 9
    assert not d.duplicated(["scenario_id", "replicate", "H_target"]).any()
    d.to_csv(OUT / "12_landscape_geometry.csv", index=False)
    numeric = [c for c in d.select_dtypes(include="number").columns if c not in ("replicate", "seed", "H_target")]
    summary = d.groupby(["H_target", "scenario_id"])[numeric].median()
    summary.to_csv(OUT / "12_group_medians.csv")
    full = archived[archived.H_target.isin(H_LEVELS) & archived.scenario_id.isin(SCENARIOS)]
    assert len(full) == 200 * 9
    full.groupby(["H_target", "scenario_id"])[[
        "n_clusters", "mean_pairwise_distance", "mean_nearest_neighbor_distance",
        "largest_cluster_fraction_forest", "complete_mst_d50",
    ]].median().to_csv(OUT / "12_full_200_group_medians.csv")
    selected = render(d) if args.limit > 1 else None
    log = {
        "purpose": __doc__, "replicates": REPLICATES[:args.limit],
        "seed_selection": "30 evenly spaced replicate IDs, rounded from linspace(1, 200, 30); shared with diagnostic 10",
        "H_levels": H_LEVELS, "scenarios": SCENARIOS, "snapshots": len(d),
        "all_grid_hashes_cluster_counts_and_habitat_counts_match": True,
        "canonical_mean_checks": int(d.canonical_mean_independently_recomputed.sum()),
        "canonical_mean_check_rule": "Recompute with frozen canonical implementation when n_clusters <= 1200; use archived hash-verified value otherwise",
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_csv_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_replicate": selected,
        "figure_selection": "Smallest mean standardised absolute deviation from nine sample medians of hybrid pair distance",
        "limitations": ["Exploratory subset of 30/200 matched blocks; no confidence intervals or new scenario inference",
                        "Centroid spread and forest-cell spread differ; neither establishes bird movement",
                        "All distances are dimensionless cell units; generation periodic, analysis open",
                        "Mean pairs equally weight clusters, not habitat area; hybrid correction is a metric decomposition, not an ecological effect"],
    }
    (OUT / "12_audit.json").write_text(json.dumps(log, indent=2) + "\n")
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
