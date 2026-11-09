"""Compare patch-level nearest-neighbour distributions at H=.125 and .025.

Exploratory Result A diagnostic for 30 prespecified, evenly spaced complete
trajectories from the 200 archived replicates. Scenario and H comparisons
reuse the same replicate numbers. Every regenerated grid and patch-weighted
mean is validated against the archived Result A table.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.generation import run_sequential_deforestation  # noqa: E402
from src.landscape.landscape import Landscape  # noqa: E402
from src.landscape.landscape_stats import nearest_neighbor_distances_from_voronoi_graph  # noqa: E402
from src.tools.final_plot_style import (  # noqa: E402
    FOCAL_SCENARIOS, apply_final_style, export_figure, figure_size,
)

SCENARIOS = ("only_random_rho0", "edge_random_rho0999", "access_weighted_rho0999")
H_LEVELS = (0.125, 0.025)
REPLICATES = tuple(int(x) for x in np.rint(np.linspace(1, 200, 30)))
PLOT_SCENARIOS = ("only_random_rho0", "access_weighted_rho0999")
PLOT_STEMS = {
    "only_random_rho0": FINAL / "results" / REL / "figures/08_nn_only_random_h_comparison",
    "access_weighted_rho0999": FINAL / "results" / REL / "figures/08_nn_access_based_h_comparison",
}
EARLY_COLORS = {"only_random_rho0": "#D18F82", "access_weighted_rho0999": "#93A2CE"}
PATCH_TABLE = FINAL / "results" / REL / "tables/08_nn_multiseed_patch_distances.csv"
LANDSCAPE_TABLE = FINAL / "results" / REL / "tables/08_nn_multiseed_landscape_summaries.csv"
SUMMARY_TABLE = FINAL / "results" / REL / "tables/08_nn_multiseed_group_summaries.csv"
LOG = FINAL / "results" / REL / "logs/08_nn_multiseed_distribution.json"
PAIRED_TABLE = FINAL / "results" / REL / "tables/08_nn_multiseed_paired_mean_changes_200.csv"
PAIRED_SUMMARY = FINAL / "results" / REL / "tables/08_nn_multiseed_paired_summary.csv"


def geometric_fixture() -> None:
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = grid[1, 1] = 1
    land = Landscape(grid)
    land.ensure_clusters(edge_depth=0.0, connectivity="4", boundary_mode="open",
                         distance_mode="voronoi_dt", voronoi_depth=1, rebuild=True)
    assert land.n_clusters == 2
    distances = nearest_neighbor_distances_from_voronoi_graph(land.ensure_voronoi_neighbor_graph())
    np.testing.assert_allclose(distances, np.full(2, np.sqrt(2)), atol=1e-12)


def render(patch_df: pd.DataFrame) -> None:
    """One figure per focal scenario, comparing the two H levels directly."""
    max_distance = float(patch_df.nn_distance_cells.max())
    max_axis = 10 ** np.ceil(np.log10(max_distance))
    xgrid = np.geomspace(1, max_axis, 360)
    styles = {s["scenario_id"]: s for s in FOCAL_SCENARIOS}
    apply_final_style()
    for sid in PLOT_SCENARIOS:
        style = styles[sid]
        fig, ax = plt.subplots(figsize=figure_size(96))
        fig.subplots_adjust(left=.12, right=.985, top=.96, bottom=.19)
        median_handles = []
        for h in H_LEVELS:
            group = patch_df[(patch_df.scenario_id == sid) & (patch_df.H_target == h)]
            curves = []
            for _, sample in group.groupby("replicate", sort=True):
                ordered = np.sort(sample.nn_distance_cells.to_numpy())
                curves.append(np.searchsorted(ordered, xgrid, side="right") / len(ordered))
            curves = np.asarray(curves)
            assert curves.shape == (30, len(xgrid))
            q05, median, q95 = np.quantile(curves, [.05, .5, .95], axis=0)
            color = EARLY_COLORS[sid] if h == 0.125 else style["color"]
            ax.fill_between(xgrid, q05, q95, color=color, alpha=.18, linewidth=0)
            ax.plot(xgrid, median, color=color, linestyle=style["linestyle"], linewidth=2.0)
            median_handles.append(Line2D([0], [0], color=color, linestyle=style["linestyle"],
                                         lw=2.0, label=rf"$H={h:.3f}$ median"))
        ax.set_xscale("log")
        ax.set_xlim(1, max_axis)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200],
                      ["1", "2", "5", "10", "20", "50", "100", "200"])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_xlabel("Patch nearest-neighbour distance (cells; log scale)")
        ax.set_ylabel("Cumulative fraction of patches")
        ax.grid(axis="y", color="#D9D9D9", linewidth=.45)
        ax.legend(handles=median_handles + [Patch(facecolor="#777777", alpha=.24,
                  label="5th–95th percentiles across 30 landscapes")],
                  loc="lower right", frameon=False)
        stem = PLOT_STEMS[sid]
        export_figure(fig, stem)
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def write_paired_summary(archived: pd.DataFrame, landscape_df: pd.DataFrame) -> None:
    records = []
    summaries = []
    for sid in SCENARIOS:
        ref = archived[(archived.scenario_id == sid) & archived.H_target.isin(H_LEVELS)]
        wide = ref.pivot(index="replicate", columns="H_target", values="mean_nearest_neighbor_distance")
        assert len(wide) == 200 and wide.notna().all().all()
        differences = wide[0.025] - wide[0.125]
        records.extend({
            "scenario_id": sid, "replicate": int(rep),
            "mean_nn_H0125_cells": float(row[0.125]),
            "mean_nn_H0025_cells": float(row[0.025]),
            "change_H0025_minus_H0125_cells": float(differences.loc[rep]),
        } for rep, row in wide.iterrows())
        sample = landscape_df[landscape_df.scenario_id == sid]
        fractions = sample.pivot(index="replicate", columns="H_target", values="fraction_patch_nn_le2")
        assert len(fractions) == 30 and fractions.notna().all().all()
        fraction_change = fractions[0.025] - fractions[0.125]
        summaries.append({
            "scenario_id": sid, "n_paired_archived_mean": 200,
            "median_mean_nn_change_cells_200": float(differences.median()),
            "q05_mean_nn_change_cells_200": float(differences.quantile(.05)),
            "q95_mean_nn_change_cells_200": float(differences.quantile(.95)),
            "n_mean_nn_increase_200": int((differences > 0).sum()),
            "n_paired_patch_fraction": 30,
            "median_change_fraction_nn_le2_30": float(fraction_change.median()),
            "n_fraction_nn_le2_increase_30": int((fraction_change > 0).sum()),
        })
    pd.DataFrame.from_records(records).to_csv(PAIRED_TABLE, index=False)
    pd.DataFrame.from_records(summaries).to_csv(PAIRED_SUMMARY, index=False)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--render-only":
        render(pd.read_csv(PATCH_TABLE))
        return
    if len(sys.argv) == 2 and sys.argv[1] == "--paired-only":
        archived = pd.read_csv(FINAL / "data/processed" / REL / "landscape_metrics.csv")
        write_paired_summary(archived, pd.read_csv(LANDSCAPE_TABLE))
        return
    assert len(sys.argv) == 1, "Usage: script.py [--render-only|--paired-only]"
    assert len(REPLICATES) == len(set(REPLICATES)) == 30
    assert REPLICATES[0] == 1 and REPLICATES[-1] == 200
    geometric_fixture()
    config_path = FINAL / "config" / REL / "07_structural_scenario_responses.json"
    config = json.loads(config_path.read_text())
    assert (config["grid_size_cells"], config["n_seed_blocks"]) == (200, 200)
    assert (config["generation_boundary"], config["analysis_boundary"], config["cluster_connectivity"], config["distance_mode"], config["voronoi_depth"]) == ("periodic", "open", "4", "voronoi_dt", 1)
    assert all(h in config["h_targets"] for h in H_LEVELS)
    scenarios = {item["scenario_id"]: item for item in config["scenarios"]}
    archived_path = FINAL / "data/processed" / REL / "landscape_metrics.csv"
    archived = pd.read_csv(archived_path)
    selected = archived[archived.scenario_id.isin(SCENARIOS) & archived.H_target.isin(H_LEVELS) & archived.replicate.isin(REPLICATES)]
    assert len(selected) == len(SCENARIOS) * len(H_LEVELS) * len(REPLICATES)
    assert not selected.duplicated(["scenario_id", "replicate", "H_target"]).any()
    reference = selected.set_index(["scenario_id", "replicate", "H_target"])
    patch_rows = []
    landscape_rows = []
    for ix, replicate in enumerate(REPLICATES, start=1):
        seed = int(config["base_seed"]) + replicate - 1
        for sid in SCENARIOS:
            scenario = scenarios[sid]
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
            assert len(run.snapshots) == 2
            snapshots = {snap.habitat_fraction: snap for snap in run.snapshots}
            assert set(snapshots) == set(H_LEVELS)
            for h in H_LEVELS:
                grid = snapshots[h].grid
                ref = reference.loc[(sid, replicate, h)]
                grid_hash = hashlib.sha256(grid.tobytes()).hexdigest()
                assert grid_hash == ref.grid_sha256
                assert int(grid.sum()) == round(h * 40_000)
                land = Landscape(grid)
                land.ensure_clusters(edge_depth=0.0, connectivity="4", boundary_mode="open",
                                     distance_mode="voronoi_dt", voronoi_depth=1, rebuild=True)
                assert land.n_clusters == int(ref.n_patches)
                values = nearest_neighbor_distances_from_voronoi_graph(land.ensure_voronoi_neighbor_graph())
                assert len(values) == land.n_clusters >= 2
                assert np.isfinite(values).all() and np.all(values >= 1)
                assert np.isclose(values.mean(), float(ref.mean_nearest_neighbor_distance), atol=1e-10)
                landscape_rows.append({
                    "scenario_id": sid, "replicate": replicate, "seed": seed, "H_target": h,
                    "grid_sha256": grid_hash, "n_patches": len(values),
                    "mean_nn_cells": float(values.mean()), "median_patch_nn_cells": float(np.median(values)),
                    "fraction_patch_nn_le2": float((values <= 2).mean()),
                    "q95_patch_nn_cells": float(np.quantile(values, .95)),
                    "max_patch_nn_cells": float(values.max()),
                })
                patch_rows.extend({
                    "scenario_id": sid, "replicate": replicate, "H_target": h,
                    "patch_id": patch_id, "nn_distance_cells": float(distance),
                } for patch_id, distance in enumerate(values, start=1))
        print(f"Completed {ix}/{len(REPLICATES)} replicates; replicate {replicate}", flush=True)

    patch_df = pd.DataFrame.from_records(patch_rows)
    landscape_df = pd.DataFrame.from_records(landscape_rows)
    assert len(landscape_df) == 180
    assert len(patch_df) == landscape_df.n_patches.sum()
    for path, df in ((PATCH_TABLE, patch_df), (LANDSCAPE_TABLE, landscape_df)):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    # Summarise landscapes as independent replicates, not pooled patches.
    group_rows = []
    for (sid, h), group in landscape_df.groupby(["scenario_id", "H_target"], sort=False):
        ref_group = archived[(archived.scenario_id == sid) & (archived.H_target == h)]
        assert ref_group.replicate.nunique() == 200
        group_rows.append({
            "scenario_id": sid, "H_target": h, "n_replicates_patch_distribution": len(group),
            "n_replicates_archived_mean": len(ref_group),
            "median_patch_count_30": float(group.n_patches.median()),
            "median_fraction_patch_nn_le2_30": float(group.fraction_patch_nn_le2.median()),
            "q05_fraction_patch_nn_le2_30": float(group.fraction_patch_nn_le2.quantile(.05)),
            "q95_fraction_patch_nn_le2_30": float(group.fraction_patch_nn_le2.quantile(.95)),
            "median_mean_nn_200": float(ref_group.mean_nearest_neighbor_distance.median()),
            "q05_mean_nn_200": float(ref_group.mean_nearest_neighbor_distance.quantile(.05)),
            "q95_mean_nn_200": float(ref_group.mean_nearest_neighbor_distance.quantile(.95)),
        })
    summary_df = pd.DataFrame.from_records(group_rows)
    summary_df.to_csv(SUMMARY_TABLE, index=False)
    write_paired_summary(archived, landscape_df)

    render(patch_df)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps({
        "purpose": "Exploratory multiseed patch-level NN distribution at two H levels",
        "selection": "Thirty unique replicate numbers rounded from an evenly spaced sequence from 1 through 200; chosen before inspecting patch distances",
        "replicates": REPLICATES, "scenarios": SCENARIOS, "H_levels": H_LEVELS,
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "archived_metrics_sha256": hashlib.sha256(archived_path.read_bytes()).hexdigest(),
        "all_regenerated_grid_hashes_and_nn_means_match_archived_rows": True,
        "figure_curves": "For each x, empirical fraction of patches with NN <= x within each landscape; median and 5th-95th percentiles across 30 independent trajectories",
        "figure_scenarios": PLOT_SCENARIOS,
        "figure_stems": [str(stem) for stem in PLOT_STEMS.values()],
        "n_landscapes": len(landscape_df), "n_patch_distances": len(patch_df),
    }, indent=2) + "\n")
    print(summary_df.to_string(index=False), flush=True)
    print(f"Saved {[str(stem) for stem in PLOT_STEMS.values()]}", flush=True)


if __name__ == "__main__":
    main()
