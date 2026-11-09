"""Inspect non-largest cluster sizes underlying Result A susceptibility.

This is an exploratory, reproducible diagnostic for 30 prespecified matched
trajectories. Each regenerated snapshot is checked against its archived grid
hash, patch count, and susceptibility before it contributes to the plot.
The figure is included in Appendix B1 as a supplementary diagnostic.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import ndimage

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.generation import run_sequential_deforestation  # noqa: E402
from src.landscape.landscape_stats import susceptibility_chi  # noqa: E402
from src.tools.final_plot_style import FOCAL_SCENARIOS, apply_final_style, export_figure, figure_size  # noqa: E402

SCENARIOS = ("only_random_rho0", "edge_random_rho0999", "access_weighted_rho0999")
H_LEVELS = (0.6, 0.3, 0.125, 0.025)
REPLICATES = tuple(int(x) for x in np.rint(np.linspace(1, 200, 30)))
XGRID = np.unique(np.rint(np.geomspace(1, 40_000, 240)).astype(int))
OUT = FINAL / "results" / REL
LANDSCAPE_TABLE = OUT / "tables/10_chi_nonlargest_landscapes.csv"
CURVE_TABLE = OUT / "tables/10_chi_nonlargest_cdfs.csv"
GROUP_TABLE = OUT / "tables/10_chi_nonlargest_group_summary.csv"
FIGURE = OUT / "figures/10_chi_nonlargest_cluster_size_cdfs"
LOG = OUT / "logs/10_chi_nonlargest_cluster_size_diagnostic.json"
FOUR_NEIGHBOURS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def render(curves_df: pd.DataFrame) -> None:
    mean_curves = curves_df.groupby(
        ["scenario_id", "H_target", "size_threshold_cells"], sort=False,
    )[["fraction_patches_le_threshold", "fraction_nonlargest_area_le_threshold"]].mean().reset_index()
    style = {s["scenario_id"]: s for s in FOCAL_SCENARIOS}
    apply_final_style()
    fig, axes = plt.subplots(4, 2, figsize=figure_size(183), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.985, top=.945, bottom=.115, wspace=.12, hspace=.36)
    for row, h in enumerate(H_LEVELS):
        for col, field in enumerate(("fraction_patches_le_threshold", "fraction_nonlargest_area_le_threshold")):
            ax = axes[row, col]
            for sid in SCENARIOS:
                q = mean_curves[(mean_curves.scenario_id == sid) & (mean_curves.H_target == h)]
                ax.step(q.size_threshold_cells, q[field], where="post", color=style[sid]["color"],
                        linestyle=style[sid]["linestyle"], linewidth=1.65)
            ax.set_xscale("log")
            ax.set_xlim(1, 40_000)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([1, 10, 100, 1000, 10_000], ["1", "10", "100", "1,000", "10,000"])
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.grid(axis="y", color="#D9D9D9", linewidth=.45)
            ax.text(0, 1.03, rf"({chr(97 + row * 2 + col)}) $H={h:g}$; " +
                    ("cluster count" if col == 0 else "habitat area"),
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, clip_on=False)
    axes[3, 0].set_xlabel("Cluster size (habitat cells; log scale)")
    axes[3, 1].set_xlabel("Cluster size (habitat cells; log scale)")
    axes[1, 0].set_ylabel("Mean cumulative fraction of clusters")
    axes[1, 1].set_ylabel("Mean cumulative fraction of remaining habitat area")
    fig.legend(handles=[Line2D([0], [0], color=style[s]["color"],
                               linestyle=style[s]["linestyle"], linewidth=1.7,
                               label=style[s]["label"]) for s in SCENARIOS],
               loc="lower center", bbox_to_anchor=(.5, .02), ncol=3, frameon=False)
    export_figure(fig, FIGURE)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--render-only":
        render(pd.read_csv(CURVE_TABLE))
        return
    assert len(sys.argv) == 1, "Usage: script.py [--render-only]"
    config_path = FINAL / "config" / REL / "07_structural_scenario_responses.json"
    archived_path = FINAL / "data/processed" / REL / "landscape_metrics.csv"
    config = json.loads(config_path.read_text())
    assert (config["grid_size_cells"], config["n_seed_blocks"]) == (200, 200)
    assert (config["generation_boundary"], config["analysis_boundary"], config["cluster_connectivity"]) == ("periodic", "open", "4")
    assert len(REPLICATES) == len(set(REPLICATES)) == 30
    assert set(H_LEVELS).issubset(config["h_targets"])
    scenario_configs = {item["scenario_id"]: item for item in config["scenarios"]}
    archived = pd.read_csv(
        archived_path,
        usecols=["scenario_id", "replicate", "H_target", "grid_sha256", "n_patches", "chi"],
    )
    reference = archived.set_index(["scenario_id", "replicate", "H_target"])
    assert not reference.index.has_duplicates
    landscape_rows: list[dict] = []
    curve_rows: list[dict] = []

    for ix, rep in enumerate(REPLICATES, start=1):
        seed = int(config["base_seed"]) + rep - 1
        for sid in SCENARIOS:
            scenario = scenario_configs[sid]
            run = run_sequential_deforestation(
                np.ones((200, 200), dtype=np.int8), np.random.default_rng(seed),
                rho=float(scenario["rho"]), directed_mode=scenario["mode"],
                target_habitat_fraction=min(H_LEVELS),
                generation_boundary=config["generation_boundary"],
                eta=config["access_reference"]["eta"],
                k=config["access_reference"]["k"],
                lambda_decay=config["access_reference"]["lambda"],
                save_habitat_fractions=H_LEVELS, seed=seed,
            )
            snapshots = {snap.habitat_fraction: snap for snap in run.snapshots}
            assert set(snapshots) == set(H_LEVELS)
            for h in H_LEVELS:
                grid = snapshots[h].grid
                ref = reference.loc[(sid, rep, h)]
                assert hashlib.sha256(grid.tobytes()).hexdigest() == ref.grid_sha256
                assert int(grid.sum()) == round(40_000 * h)
                labels, n_patches = ndimage.label(grid == 1, structure=FOUR_NEIGHBOURS)
                assert n_patches == int(ref.n_patches) and n_patches >= 2
                sizes = np.bincount(labels.ravel())[1:]
                largest_idx = int(np.argmax(sizes))
                largest = int(sizes[largest_idx])
                remaining = np.delete(sizes, largest_idx)
                assert remaining.size == n_patches - 1
                chi = susceptibility_chi(sizes)
                assert np.isclose(chi, float(ref.chi), rtol=1e-10, atol=1e-9)
                second = int(remaining.max())
                nonlargest_area = int(remaining.sum())
                squared_sum = int(np.dot(remaining.astype(np.int64), remaining.astype(np.int64)))
                sorted_sizes = np.sort(remaining)
                count_cdf = np.searchsorted(sorted_sizes, XGRID, side="right") / remaining.size
                area_cdf = np.concatenate(([0], np.cumsum(sorted_sizes)))[
                    np.searchsorted(sorted_sizes, XGRID, side="right")
                ] / nonlargest_area
                landscape_rows.append({
                    "scenario_id": sid, "replicate": rep, "seed": seed, "H_target": h,
                    "n_patches": n_patches, "n_nonlargest_patches": len(remaining),
                    "largest_patch_cells": largest, "nonlargest_area_cells": nonlargest_area,
                    "second_largest_cells": second, "chi_cells": chi,
                    "second_share_chi_numerator": second**2 / squared_sum,
                    "share_nonlargest_area_clusters_ge100": remaining[remaining >= 100].sum() / nonlargest_area,
                    "share_nonlargest_area_clusters_ge1000": remaining[remaining >= 1000].sum() / nonlargest_area,
                })
                curve_rows.extend({
                    "scenario_id": sid, "replicate": rep, "H_target": h,
                    "size_threshold_cells": int(x), "fraction_patches_le_threshold": float(c),
                    "fraction_nonlargest_area_le_threshold": float(a),
                } for x, c, a in zip(XGRID, count_cdf, area_cdf))
        print(f"Completed {ix}/{len(REPLICATES)} matched replicates", flush=True)

    landscape_df = pd.DataFrame(landscape_rows)
    curves_df = pd.DataFrame(curve_rows)
    assert len(landscape_df) == len(REPLICATES) * len(SCENARIOS) * len(H_LEVELS)
    assert len(curves_df) == len(landscape_df) * len(XGRID)
    group = landscape_df.groupby(["scenario_id", "H_target"], sort=False)
    summary = group.agg(
        n_trajectories=("replicate", "nunique"),
        median_chi_cells=("chi_cells", "median"),
        median_second_largest_cells=("second_largest_cells", "median"),
        median_nonlargest_area_cells=("nonlargest_area_cells", "median"),
        median_second_share_chi_numerator=("second_share_chi_numerator", "median"),
        median_share_area_ge100=("share_nonlargest_area_clusters_ge100", "median"),
        median_share_area_ge1000=("share_nonlargest_area_clusters_ge1000", "median"),
        median_n_nonlargest_patches=("n_nonlargest_patches", "median"),
    ).reset_index()
    assert (summary.n_trajectories == 30).all()
    for path, table in ((LANDSCAPE_TABLE, landscape_df), (CURVE_TABLE, curves_df), (GROUP_TABLE, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    render(curves_df)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps({
        "purpose": "Supplementary non-largest cluster-size diagnostic for susceptibility; Appendix B1",
        "H_levels": H_LEVELS, "scenarios": SCENARIOS, "replicates": REPLICATES,
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "archived_metrics_sha256": hashlib.sha256(archived_path.read_bytes()).hexdigest(),
        "all_grid_hashes_patch_counts_and_chi_match_archived": True,
        "exclusion": "Exactly one largest cluster per landscape; ties retain other tied clusters",
        "curve_aggregation": "Arithmetic mean of within-landscape CDFs; 30 landscapes equally weighted",
        "cluster_connectivity": "four orthogonal neighbours; open analysis boundary",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved {FIGURE}")


if __name__ == "__main__":
    main()
