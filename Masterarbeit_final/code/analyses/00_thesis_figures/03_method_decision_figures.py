"""Compact Chapter 3 figures from verified, saved results; no simulation rerun.

Run from the repository root via codex_run plot-thesis-method-decisions.
SVG/PDF masters, 300 dpi previews, plotted tables, fixtures and source hashes
are saved together. --inspect checks sources without exporting figures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ufz-method-figures-mpl")
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
ROOT = FINAL.parent
sys.path.insert(0, str(FINAL / "code/base"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

from src.landscape.landscape import Landscape
from src.tools.final_plot_style import (
    LANDSCAPE_COLORS, MISSING, NEUTRAL, apply_final_style, export_figure,
    figure_size, roman_font_files,
)

REL = Path("00_thesis_figures/03_method_decisions")
DATA = FINAL / "data/processed" / REL
RESULTS = FINAL / "results" / REL
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CONFIG = FINAL / "config/00_thesis_figures/03_method_decisions.json"
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INPUTS: set[Path] = set()
METRICS = ["edge_density_total", "largest_cluster_fraction_forest", "chi",
           "mean_nearest_neighbor_distance", "mean_pairwise_distance", "complete_mst_d50"]


def read_csv(path: Path) -> pd.DataFrame:
    INPUTS.add(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    INPUTS.add(path)
    return json.loads(path.read_text())


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(fig: plt.Figure, name: str) -> None:
    export_figure(fig, FIGURES / name)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.04, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5)


def workflow() -> None:
    fig, ax = plt.subplots(figsize=figure_size(49))
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.02)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    boxes = [(.00, "Seven fixed scenarios", "Compare at the same $H$"),
             (.35, "Landscape structure", "Six structural metrics"),
             (.70, "Equilibrium occupancy", "Vary $r$ and $\\ell_{1/2}$")]
    rows = []
    for i, (left, label, detail) in enumerate(boxes):
        ax.add_patch(Rectangle((left, .48), .30, .39, facecolor="#F2F5F6", edgecolor="#9DA8AE", lw=.6))
        ax.text(left+.15, .735, label, ha="center", va="center", fontsize=9.5)
        ax.text(left+.15, .595, detail, ha="center", va="center", fontsize=8.5)
        rows.append({"step": i+1, "label": label, "detail": detail})
    for start, end in [(.30, .35), (.65, .70)]:
        ax.annotate("", xy=(end, .675), xytext=(start, .675), arrowprops={"arrowstyle": "->", "color": NEUTRAL, "lw": 1})
    ax.text(.50, .22, "Real forest windows\nContext for structural coverage", ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(.50, .48), xytext=(.50, .34), arrowprops={"arrowstyle": "->", "color": NEUTRAL, "lw": 1})
    ax.text(.85, .22, "Forest-bird responses\nCompare response shapes", ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(.85, .48), xytext=(.85, .34), arrowprops={"arrowstyle": "->", "color": NEUTRAL, "lw": 1})
    pd.DataFrame(rows).to_csv(TABLES / "study_design_steps.csv", index=False)
    export(fig, "study_design_overview")


def distance_schematic() -> dict:
    """One landscape, two directly labelled distances; illustrative, not sampled."""
    grid = np.zeros((200, 200), dtype=np.uint8)
    yy, xx = np.indices(grid.shape)
    radius = 13.
    # Cell-centre rasterisation of the same rounded outlines drawn below.
    outlines = []
    for cx in [49.5, 132.5]:
        patch = FancyBboxPatch((cx-40, 79.5), 80, 41,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=LANDSCAPE_COLORS["habitat"], edgecolor="none")
        grid[patch.get_path().contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(grid.shape)] = 1
        outlines.append(patch)
    landscape = Landscape(grid)
    landscape.ensure_clusters(connectivity=4, boundary_mode="open", distance_mode="voronoi_dt", voronoi_depth=1)
    labels = landscape.core_cluster_labels
    centres = landscape.cluster_centroids
    hybrid = landscape.cluster_dist_matrix
    graph = landscape.voronoi_neighbor_graph
    centroid = cdist(centres, centres)
    assert len(centres) == 2 and graph[0, 1] > 0
    points = [np.argwhere(labels == i) for i in range(1, 3)]
    # Landscape centroids use (x,y), while np.argwhere returns (row,column).
    assert np.allclose(centres, np.stack([p.mean(axis=0)[::-1] for p in points]))
    boundaries = [np.argwhere((labels == i) & ~binary_erosion(labels == i))
                  for i in range(1, 3)]
    exact = cdist(*boundaries).min()
    edge_points = np.array([[89., 100.], [93., 100.]])
    assert np.isclose(np.linalg.norm(edge_points[1]-edge_points[0]), exact)
    assert all(labels[int(y), int(x)] == i for i, (x, y) in enumerate(edge_points, 1))
    assert np.isclose(hybrid[0, 1], exact) and np.isclose(exact, 4)
    assert np.isclose(centroid[0, 1], 83)
    pd.DataFrame([dict(patch_i=1, patch_j=2, centroid_distance=centroid[0, 1],
                       exact_edge_distance=exact, hybrid_distance=hybrid[0, 1],
                       direct_neighbor=True)]).to_csv(TABLES / "distance_geometry_pairs.csv", index=False)
    np.savez_compressed(DATA / "distance_geometry.npz", grid=grid, labels=labels,
                        centroids=centres, hybrid=hybrid, centroid=centroid,
                        highlighted_edge_points_xy=edge_points)
    fig, ax = plt.subplots(figsize=figure_size(90))
    fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
    cmap = ListedColormap([LANDSCAPE_COLORS["non_habitat"], LANDSCAPE_COLORS["habitat"]])
    ax.imshow(np.zeros_like(grid), cmap=cmap, vmin=0, vmax=1, interpolation="nearest", origin="upper")
    for patch in outlines:
        ax.add_patch(patch)
    ax.set(xlim=(-.5, 183.5), ylim=(153.5, 50.5), aspect="equal")
    ax.set_axis_off()
    middle = centres[:, 0].mean()
    # Projection lines place the two measurement bars outside the habitat.
    for x, y in centres:
        ax.plot([x, x], [70, y], color=ORANGE, lw=.8)
        ax.scatter(x, y, s=36, facecolor="white", edgecolor=ORANGE, linewidth=1.5, zorder=5)
    for x, y in edge_points:
        ax.plot([x, x], [y, 130], color=BLUE, lw=.8)
        ax.scatter(x, y, s=16, facecolor="white", edgecolor=BLUE, linewidth=1, zorder=5)
    for xs, y, color in [(centres[:, 0], 70, ORANGE), (edge_points[:, 0], 130, BLUE)]:
        ax.add_patch(FancyArrowPatch((xs[0], y), (xs[1], y), arrowstyle="|-|",
                                    mutation_scale=8, shrinkA=0, shrinkB=0, lw=1.6, color=color))
    ax.text(middle, 54, "Center-to-center distance", ha="center", va="center", fontsize=9.5)
    ax.text(middle, 62, "83 cells between patch centers", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=NEUTRAL)
    ax.text(middle, 140, "Edge-to-edge distance", ha="center", va="center", fontsize=9.5)
    ax.text(middle, 148, "4 cells between patch edges", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=NEUTRAL)
    for i, (x, _) in enumerate(centres):
        ax.text(x, 111, f"Patch {chr(65+i)}", ha="center", va="center", color="white", fontsize=9.5)
    export(fig, "distance_representation_decision")
    return dict(n_illustrative_patches=2, fixture_shape=[200, 200],
                displayed_extent_xy=[-.5, 183.5, 50.5, 153.5],
                centroid_distance_cells=float(centroid[0, 1]), exact_edge_distance_cells=float(exact),
                hybrid_distance_cells=float(hybrid[0, 1]), simulated_scale="dimensionless",
                patch_geometry="rounded rectangles; smooth outlines rasterised at cell centres",
                neighbor_edge_definition="minimum Euclidean distance between boundary-cell centres")


def distance_agreement(inspect: bool = False) -> dict:
    """Compact descriptive overview of the selected structural representations."""
    directory = FINAL / "results/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/tables"
    data = read_csv(directory / "01_comparison_by_h.csv")
    exactness = read_csv(directory / "01_graph_exactness.csv")
    local = read_csv(directory / "01_hybrid_local_metric_exactness.csv")
    selections = [("mean_nn", "voronoi_graph", "Mean NN distance"),
                  ("mean_pair", "voronoi_hybrid", "Mean inter-patch\ndistance"),
                  ("d50", "voronoi_hybrid", r"Complete-MST $D_{50}$")]
    parts, rows = [], []
    for metric, representation, label in selections:
        part = data[data.metric.eq(metric) & data.candidate_representation.eq(representation)
                    & data.reference_representation.eq("edge_complete")].copy()
        assert len(part) == 8 and not part.H_target.duplicated().any()
        part["absolute_percentage_difference_median"] = 100 * part.median_abs_relative_difference
        parts.append(part)
        rows.append(dict(metric=metric, selected_representation=representation,
                         n_snapshots=int(part.n_landscapes.sum()), n_seed_blocks=int(part.n_seed_blocks.min()),
                         n_defined_correlations=int(part.spearman.notna().sum()),
                         n_defined_error_medians=int(part.median_abs_relative_difference.notna().sum()),
                         min_within_H_spearman=float(part.spearman.min()),
                         max_within_H_spearman=float(part.spearman.max()),
                         min_H_median_absolute_percentage_difference=float(part.absolute_percentage_difference_median.min()),
                         max_H_median_absolute_percentage_difference=float(part.absolute_percentage_difference_median.max())))
    summary = pd.DataFrame(rows)
    assert summary.n_snapshots.eq(1970).all() and summary.n_seed_blocks.eq(10).all()
    assert exactness.set_index("metric").loc["mean_nn", "n_exact"] == 1970
    assert local.set_index("metric").loc["d50", "n_exact"] == 1970
    assert summary.loc[summary.metric.eq("mean_pair"), "max_H_median_absolute_percentage_difference"].iloc[0] < 1
    audit = {"comparison": "selected structural representation versus complete exact edge reference",
             "n_snapshots": 1970, "n_independent_seed_blocks": 10,
             "n_sampled_H_levels": 8, "range_encoding": "minimum to maximum across defined H-specific summaries; not a confidence interval",
             "mean_nn_graph_weights": "exact_complete_edge_matrix; evaluates direct Voronoi topology",
             "metrics": summary.to_dict(orient="records")}
    if inspect:
        return audit
    pd.concat(parts).to_csv(TABLES / "distance_agreement_by_h.csv", index=False)
    summary.to_csv(TABLES / "distance_agreement_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=figure_size(72), sharey=True)
    fig.subplots_adjust(left=.255, right=.985, top=.84, bottom=.27, wspace=.30)
    for j, (ax, value, limits, xlabel, descriptor) in enumerate([
        (axes[0], "spearman", (-1, 1), "Within-$H$ Spearman correlation", "(a) Rank agreement"),
        (axes[1], "absolute_percentage_difference_median", (0, 1.15), "Median absolute difference (%)", "(b) Value agreement")]):
        for i, part in enumerate(parts):
            values = part[value].dropna().to_numpy(float)
            lo, hi = values.min(), values.max()
            ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round")
            ax.scatter(values, np.full(len(values), i), s=20, facecolor="white", edgecolor=BLUE, linewidth=1, zorder=3, clip_on=False)
            text_value = (f"{lo:.3f}" if np.isclose(lo, hi) else f"{lo:.3f}-{hi:.3f}") if j == 0 else ("0" if hi == 0 else f"{lo:.3f}-{hi:.3f}")
            ax.text(hi if j == 0 else (lo+hi)/2, i-.20, text_value, ha="right" if j == 0 else "center", va="bottom", fontsize=8.5)
        ax.set(xlim=limits, ylim=(2.45, -.55), yticks=[0, 1, 2])
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.axvline(1 if j == 0 else 0, color="#B9B9B9", lw=.6, zorder=0)
        panel(ax, descriptor)
    axes[0].set_yticklabels([label for _, _, label in selections])
    axes[0].set_xticks([-1, 0, 1])
    axes[1].set_xticks([0, .5, 1])
    fig.legend(handles=[Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=BLUE,
                              color="none", label="Within-H summary"),
                        Line2D([], [], color=BLUE, lw=2, label="Min-max across H (not a CI)")],
               loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False)
    export(fig, "distance_metric_agreement")
    return audit


def access_summary(c: dict, inspect: bool = False) -> dict:
    data = read_csv(FINAL / "results" / c["access_results"] / "tables/02_factor_decomposition.csv")
    rows = []
    for metric in METRICS[:2]:
        for rho in [.9, .99, .999]:
            part = data[data.metric.eq(metric) & data.rho.eq(rho)]
            part = part[part.sufficient_replication & part.total_sum_squares.gt(0)].copy()
            assert len(part) > 0 and not part.H_target.duplicated().any()
            cols = ["share_eta", "share_k", "share_lambda", "share_eta_k", "share_eta_lambda", "share_k_lambda", "share_eta_k_lambda"]
            assert np.allclose(part[cols].sum(axis=1), 1) and part[cols].notna().all().all()
            rows.append(dict(metric=metric, rho=rho, n_H=len(part), min_seeds=int(part.min_seed_replicates_per_cell.min()),
                             n_sensitive_H=int(part.seed_influence_status.eq("sensitive").sum()),
                             eta=part.share_eta.mean(), k=part.share_k.mean(), lambda_=part.share_lambda.mean(),
                             interactions=part[cols[3:]].sum(axis=1).mean()))
    summary = pd.DataFrame(rows)
    audit = {"n_bars": len(summary), "n_H_per_bar": summary.n_H.tolist(),
             "n_sensitive_H_per_bar": summary.n_sensitive_H.tolist(), "summary": "equally weighted arithmetic mean of shares across defined H levels"}
    if inspect:
        return audit
    summary.to_csv(TABLES / "access_effect_share_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=figure_size(68), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.25, top=.87, wspace=.17)
    styles = [("eta", BLUE, r"$\eta$"), ("k", ORANGE, "$k$"), ("lambda_", GREEN, r"$\lambda$"),
              ("interactions", "#B9B9B9", "Interactions")]
    for ax, metric, name in zip(axes, METRICS[:2], ["(a) Edge density ED", "(b) Largest cluster fraction LCF"]):
        part = summary[summary.metric.eq(metric)].reset_index(drop=True)
        left = np.zeros(len(part))
        for column, color, label in styles:
            ax.barh(np.arange(3), part[column], left=left, height=.54, color=color, label=label)
            left += part[column]
        for i, row in part.iterrows():
            if row.n_sensitive_H:
                ax.scatter(1.018, i, marker="*", color=NEUTRAL, s=24, clip_on=False)
        ax.set(yticks=[0, 1, 2], yticklabels=["0.9", "0.99", "0.999"], xlim=(0, 1), ylim=(2.6, -.6))
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel(ax, name)
    axes[0].set_ylabel(r"Removal parameter $\rho$")
    fig.supxlabel("Mean share of parameter-setting variation")
    handles = [Patch(facecolor=color, label=label) for _, color, label in styles]
    if summary.n_sensitive_H.gt(0).any():
        handles.append(Line2D([], [], marker="*", color=NEUTRAL, linestyle="none", label="Includes seed-sensitive levels"))
    fig.legend(handles=handles, loc="lower center", ncol=4 if len(handles)==4 else 3, frameon=False)
    export(fig, "access_parameter_effect_shares")
    return audit


def redundancy(c: dict, inspect: bool = False) -> dict:
    data = read_csv(FINAL / "results" / c["redundancy_results"] / "tables/03_redundancy_by_pair.csv")
    matrix = np.full((6, 6), np.nan)
    assert len(data) == 15 and not data[["metric_1", "metric_2"]].duplicated().any()
    for row in data.itertuples():
        i, j = METRICS.index(row.metric_1), METRICS.index(row.metric_2)
        matrix[max(i,j), min(i,j)] = row.median_absolute_spearman
    audit = {"max_median_absolute_correlation": float(data.median_absolute_spearman.max()),
             "flagging_level": .8, "all_pair_medians_below_flagging_level": bool(data.median_absolute_spearman.lt(.8).all()),
             "some_individual_H_correlations_exceed_flagging_level": bool(data.max_absolute_spearman.ge(.8).any())}
    if inspect:
        return audit
    data.to_csv(TABLES / "metric_redundancy_pairs.csv", index=False)
    fig, ax = plt.subplots(figsize=figure_size(92, width_mm=116))
    fig.subplots_adjust(left=.14, right=.78, bottom=.18, top=.96)
    ax.set_facecolor("white")
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for i in range(6):
        ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, facecolor=MISSING, edgecolor="white", lw=1))
    for i in range(6):
        for j in range(i):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if value < .55 else NEUTRAL)
    labels = ["ED", "LCF", r"$\chi$", "NN", "IPD", r"$D_{50}$"]
    ax.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=labels, yticklabels=labels)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    cax = fig.add_axes([.82, .24, .035, .65])
    bar = fig.colorbar(im, cax=cax, ticks=[0, .2, .4, .6, .8, 1])
    bar.set_label(r"Median $|r_s|$ across $H$")
    bar.ax.axhline(.8, color=NEUTRAL, lw=1.0)
    fig.text(.5, .035, "Largest pair median: 0.728; flagging level: 0.8", ha="center", fontsize=8.5)
    export(fig, "metric_redundancy_summary")
    return audit


def real_inputs(c: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    data = read_csv(FINAL / c["real_metrics"])
    viewer = read_csv(ROOT / "data/real_landscapes/selected_landscapes.csv")
    assert len(data) == 10000 and not data.candidate_id.duplicated().any()
    assert data.groupby(["extent_m", "h_bin_index"]).size().eq(100).all()
    assert set(data.candidate_id) == set(viewer.candidate_id)
    examples = []
    grids = []
    for extent in [2000, 4000, 8000, 16000, 32000]:
        part = data[data.extent_m.eq(extent)].copy()
        part["distance_to_target_H"] = abs(part.H - c["example_target_H"])
        row = part.sort_values(["distance_to_target_H", "candidate_id"]).iloc[0]
        if c["example_ids"]:
            row = part[part.candidate_id.eq(c["example_ids"][len(examples)])].iloc[0]
        meta = viewer[viewer.candidate_id.eq(row.candidate_id)].iloc[0]
        archive = ROOT / "data/real_landscapes" / meta.landscape_path
        INPUTS.add(archive.resolve())
        with np.load(archive, allow_pickle=False) as saved:
            grid = (np.asarray(saved["grid"], dtype=np.uint8) == 1).astype(np.int8)
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        if grid.shape != (200, 200) or digest != row.grid_sha256 or not np.isclose(grid.mean(), row.H):
            raise ValueError(f"Viewer raster does not match verified metrics: {row.candidate_id}")
        grids.append(grid)
        examples.append(row.to_dict() | {"panel_id": chr(97+len(examples)), "viewer_archive_key": "grid"})
    return data, pd.DataFrame(examples), grids


def world_geometry(c: dict):
    import geopandas as gpd
    path = FINAL / c["world_geometry"]
    if not path.exists():
        import pyogrio
        source = Path(pyogrio.__file__).parent / "tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
        if not source.exists():
            raise FileNotFoundError("Provide Natural Earth land polygons as world_land.geojson")
        land = gpd.read_file(source).to_crs(4326)
        land[["geometry"]].to_file(path, driver="GeoJSON")
        save_json(DATA / "world_land_source.json", {"source": "Natural Earth low-resolution land polygons bundled in pyogrio test fixtures",
                  "source_path": str(source), "source_shp_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "crs": "EPSG:4326", "usage": "geographical context only"})
    INPUTS.add(path)
    return gpd.read_file(path)


def real_sampling(c: dict, inspect: bool = False) -> dict:
    data, examples, grids = real_inputs(c)
    audit = {"n_windows": len(data), "n_unique_centres": len(data[["lon", "lat"]].drop_duplicates()),
             "n_windows_per_extent": data.groupby("extent_m").size().to_dict(),
             "examples": examples[["candidate_id", "extent_m", "H", "lat", "lon", "grid_sha256"]].to_dict("records"),
             "all_example_grids_verified_against_source_hashes": True}
    if inspect:
        return audit
    examples.to_csv(TABLES / "real_landscape_examples.csv", index=False)
    data[["candidate_id", "extent_m", "h_bin_index", "H", "lat", "lon"]].to_csv(TABLES / "real_sample_locations.csv", index=False)
    np.savez_compressed(DATA / "real_landscape_examples.npz", grids=np.stack(grids), candidate_ids=examples.candidate_id.to_numpy(dtype=str))
    land = world_geometry(c)
    fig = plt.figure(figsize=figure_size(115))
    ax = fig.add_axes([.015, .355, .97, .62])
    land.plot(ax=ax, facecolor="#ECECEC", edgecolor="#B9B9B9", linewidth=.3)
    ax.scatter(data.lon, data.lat, s=1.2, color="#377A9F", alpha=.5, linewidths=0, rasterized=True)
    offsets = [(-20, -13), (13, 13), (-16, 13), (16, -12), (0, 17)]
    for (_, row), offset in zip(examples.iterrows(), offsets):
        ax.scatter(row.lon, row.lat, s=21, facecolor="white", edgecolor=NEUTRAL, lw=.9, zorder=5)
        ax.annotate(row.panel_id, (row.lon, row.lat), xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=8.5, zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
                    arrowprops={"arrowstyle": "-", "color": NEUTRAL, "lw": .6})
    ax.set(xlim=(-180, 180), ylim=(-58, 85))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(handles=[Line2D([], [], marker="o", color="#377A9F", markersize=3, linestyle="none", label="10,000 sampled windows"),
                       Line2D([], [], marker="o", color=NEUTRAL, markerfacecolor="white", markersize=4, linestyle="none", label="Examples (a)–(e)")],
              loc="lower center", bbox_to_anchor=(.5, -.10), ncol=2, frameon=False)
    cmap = ListedColormap([LANDSCAPE_COLORS["non_habitat"], LANDSCAPE_COLORS["habitat"]])
    for i, ((_, row), grid) in enumerate(zip(examples.iterrows(), grids)):
        left = .015 + i*.197
        raster_ax = fig.add_axes([left, .075, .182, .254])
        extent_km = row.extent_m / 1000
        raster_ax.imshow(grid, interpolation="nearest", origin="upper", cmap=cmap, vmin=0, vmax=1,
                         extent=(0, extent_km, 0, extent_km), aspect="equal")
        raster_ax.set_axis_off()
        panel(raster_ax, f"({row.panel_id}) {extent_km:g} km")
        raster_ax.text(.5, -.06, f"$H={row.H:.3f}$", transform=raster_ax.transAxes, ha="center", va="top", fontsize=8.5)
    fig.legend(handles=[Patch(facecolor=LANDSCAPE_COLORS["habitat"], label="Forest"),
                        Patch(facecolor=LANDSCAPE_COLORS["non_habitat"], label="Non-forest")],
               loc="lower center", bbox_to_anchor=(.5, -.013), ncol=2, frameon=False)
    export(fig, "real_landscape_sampling_map")
    return audit


def occupancy_domain(c: dict, inspect: bool = False, full_pilot: bool = False) -> dict:
    directory = FINAL / "results" / c["species_results"]
    data = read_csv(directory / "tables/09_parameter_scenario_h_diagnostics.csv")
    decision = read_json(directory / "logs/09_main_parameter_decision.json")
    valid = data[data.n_valid.ge(3)]
    summary = valid.groupby(["log_r", "half_distance"], as_index=False).agg(
        mean_of_cell_medians=("median", "mean"), n_scenario_h_cells=("median", "size"),
        n_valid_landscape_responses=("n_valid", "sum"))
    assert len(summary) == 949 and summary.n_scenario_h_cells.eq(55).all()
    assert summary.mean_of_cell_medians.between(0, 1).all()
    # Verify eligibility is identical at every parameter combination.
    assert valid.groupby(["log_r", "half_distance"])["n_nonconverged"].sum().eq(0).all()
    audit = {"n_parameter_combinations": len(summary), "n_scenario_H_cells": 55,
             "n_independent_seed_blocks": 5, "valid_pilot_responses": int(data.n_valid.sum()),
             "selected_log10_r": [decision["log10_r_min"], decision["log10_r_max"]],
             "selected_half_distance": [decision["half_distance_min"], decision["half_distance_max"]],
             "statistic": "arithmetic mean of scenario-H median mean equilibrium patch occupancy; equal scenario-H weights",
             "interpolation": "none; rectangular tiles centred on sampled log parameter values"}
    if inspect:
        return audit
    summary["log10_r"] = summary.log_r / np.log(10)
    summary.to_csv(TABLES / "species_parameter_occupancy_surface.csv", index=False)
    matrix = summary.pivot(index="log10_r", columns="half_distance", values="mean_of_cell_medians").sort_index()
    xs, ys = np.log2(matrix.columns.to_numpy()), matrix.index.to_numpy()
    def cell_edges(values):
        mids = (values[:-1]+values[1:])/2
        return np.r_[values[0]-(mids[0]-values[0]), mids, values[-1]+(values[-1]-mids[-1])]
    fig, ax = plt.subplots(figsize=figure_size(98))
    fig.subplots_adjust(left=.13, right=.76, bottom=.18, top=.97)
    im = ax.pcolormesh(cell_edges(xs), cell_edges(ys), matrix.to_numpy(), cmap="cividis", vmin=0, vmax=1, shading="flat", rasterized=True)
    lo, hi = decision["log10_r_min"], decision["log10_r_max"]
    xlo, xhi = np.log2(decision["half_distance_min"]), np.log2(decision["half_distance_max"])
    for color, lw in [("white", 3.3), (NEUTRAL, 1.3)]:
        ax.add_patch(Rectangle((xlo, lo), xhi-xlo, hi-lo, fill=False, edgecolor=color, linewidth=lw, zorder=5))
    ax.set_xlim(-.5, 12.5)
    ax.set_ylim(-12.125 if full_pilot else -4.125, 6.125)
    ax.set_xticks([0, 3, 6, 9, 12], ["1", "8", "64", "512", "4096"])
    y_ticks = [-12, -9, -6, -3, 0, 3, 6] if full_pilot else [-4, -2, 0, 2, 4, 6]
    ax.set_yticks(y_ticks, ["$1$" if value == 0 else rf"$10^{{{value}}}$" for value in y_ticks])
    ax.set_xlabel(r"Dispersal half-distance $\ell_{1/2}$ (cells; log scale)")
    ax.set_ylabel(r"Extinction-to-colonization ratio $r$ (log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(4.5, -6.5 if full_pilot else -3.5, "Near-one responses", ha="center", fontsize=9.5, color=NEUTRAL,
            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
    ax.text(11.05, 5.0, "Near zero", ha="center", fontsize=8.5, color="white")
    ax.text(4.5, .4, "Selected main domain", ha="center", fontsize=8.5, color="white",
            bbox={"facecolor": NEUTRAL, "alpha": .88, "edgecolor": "none", "pad": 2})
    cax = fig.add_axes([.805, .18, .03, .79])
    bar = fig.colorbar(im, cax=cax, ticks=[0, .25, .5, .75, 1])
    bar.set_label("Mean of scenario–$H$ occupancy medians")
    export(fig, "species_parameter_full_pilot_heatmap" if full_pilot else "species_parameter_domain_heatmap")
    return audit


def font_audit() -> dict:
    audit = {}
    for pdf in sorted(FIGURES.glob("*.pdf")):
        result = subprocess.run(["pdffonts", str(pdf)], check=True, text=True, capture_output=True).stdout
        if "LMRoman8-Regular" not in result:
            raise RuntimeError(f"Roman 8 is not embedded in {pdf}")
        audit[pdf.name] = result.splitlines()[2:]
    return audit


def rerender_appendix_diagnostics(c: dict) -> None:
    """Presentation-only rerender from frozen tables; no bootstrap or simulation."""
    def module(relative, name):
        path = FINAL / "code/analyses" / relative
        INPUTS.add(path)
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    distance_rel = Path("01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection")
    distance = module(distance_rel / "01_distance_graph_selection.py", "method_distance_plotter")
    distance_figures = FINAL / "results" / distance_rel / "figures"
    summary = read_csv(FINAL / "results" / distance_rel / "tables/01_comparison_by_h.csv")
    assert len(summary) == 112
    distance.make_figures(summary, distance_figures)
    capacity = read_csv(FINAL / "results" / distance_rel / "tables/01_capacity_shortfall_by_rho_h.csv")
    distance_config = FINAL / "config" / distance_rel / "01_distance_graph_selection.json"
    INPUTS.add(distance_config)
    geometry_settings = distance.load_config(distance_config)
    distance.make_capacity_rho_figures(capacity, distance_figures,
        geometry_settings["half_distances_cells"], geometry_settings["h_targets"],
        geometry_settings["n_removed_at_rho_reference_h"])
    redundancy_rel = Path("01_landscape/01_02_preliminary_analyses/01_02_02_metric_redundancy")
    metric_plotter = module(redundancy_rel / "04_metric_redundancy_summary.py", "method_redundancy_plotter")
    directory = FINAL / "results" / c["redundancy_results"]
    settings = read_json(FINAL / "config" / redundancy_rel / "03_metric_redundancy.json")
    pairs = read_csv(directory / "tables/03_redundancy_by_pair.csv")
    by_h = read_csv(directory / "tables/03_pairwise_by_h.csv")
    by_scenario = read_csv(directory / "tables/03_flagged_pair_by_scenario.csv")
    metric_plotter.stability_figures(by_h, pairs, settings, directory / "figures")
    metric_plotter.scenario_stability_figure(by_scenario, pairs, settings, directory / "figures")
    paths = list(distance_figures.glob("*rank_preservation.pdf")) + list(distance_figures.glob("*relative_deviation.pdf"))
    paths += list(distance_figures.glob("*shortfall*.pdf"))
    paths += list((directory / "figures").glob("*stability*.pdf"))
    fonts = {}
    for path in sorted(paths):
        listing = subprocess.run(["pdffonts", str(path)], check=True, text=True, capture_output=True).stdout
        assert "LMRoman8-Regular" in listing
        fonts[str(path.relative_to(FINAL))] = listing.splitlines()[2:]
    save_json(RESULTS / "logs/appendix_figure_export_audit.json", {
        "scope": "plot-only axis direction, panel descriptions and single-curve blue; no numerical summaries changed",
        "n_updated_pdf_svg_pairs": len(paths), "embedded_fonts": fonts,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(INPUTS)}})
    print(f"Rerendered {len(paths)} appendix diagnostics from saved summaries")



def access_individual_effects(c: dict) -> None:
    """Illustrative one-at-a-time contrasts from frozen factorial summaries."""
    from src.tools.final_figure_data import read_metric_plot_table
    spec = c["access_individual_effects"]
    source = FINAL / "results" / c["access_results"] / "tables/02_metric_summary.csv"
    INPUTS.add(source)
    data = read_metric_plot_table(source, FINAL, spec, ["median", "mean", "p05", "p95"])
    fixed = spec["reference"]
    base = data[np.isclose(data.rho, spec["rho"]) & np.isclose(data.H_target, spec["H"])].copy()
    parts = []
    for parameter in ("eta", "k", "lambda"):
        mask = np.ones(len(base), dtype=bool)
        for other, value in fixed.items():
            if other != parameter:
                mask &= np.isclose(base[other], value)
        part = base.loc[mask].copy()
        part["varied_parameter"] = parameter
        parts.append(part)
    plotted = pd.concat(parts, ignore_index=True)
    assert len(plotted) == 60 and not plotted[["median", "p05", "p95"]].isna().any().any()
    assert (plotted.p05 <= plotted["median"]).all() and (plotted["median"] <= plotted.p95).all()
    plotted.to_csv(TABLES / "access_individual_parameter_effects.csv", index=False)
    fig, axes = plt.subplots(6, 3, figsize=figure_size(190), sharey="row")
    fig.subplots_adjust(left=.12, right=.985, bottom=.125, top=.965, hspace=.74, wspace=.27)
    names = ["ED", "LCF", r"$\chi$", "MNN", "MIP", "D50"]
    units = ["ED", "LCF", r"$\chi$ (cells)", "MNN (cells)", "MIP (cells)", "D50 (cells)"]
    parameter_names = [r"Weighting strength $\eta$", "Number of rings $k$", r"Distance decay $\lambda$"]
    for row, metric in enumerate(METRICS):
        for col, (parameter, color) in enumerate(zip(("eta", "k", "lambda"), (BLUE, ORANGE, GREEN))):
            ax = axes[row, col]
            part = plotted[plotted.metric.eq(metric) & plotted.varied_parameter.eq(parameter)].sort_values(parameter)
            x = part[parameter].to_numpy()
            y = part["median"].to_numpy()
            ax.errorbar(x, y, yerr=np.array([y-part.p05.to_numpy(), part.p95.to_numpy()-y]),
                        fmt="o-", color=color, lw=1.1, markersize=3.1, capsize=2, elinewidth=.8)
            for xx, yy, n in zip(x, part.p95, part.n_valid):
                if n < 10:
                    ax.annotate(f"n={n}", (xx, yy), xytext=(-3, 3), textcoords="offset points",
                                ha="right", fontsize=8.5)
            ax.set_xticks(x, [f"{v:g}" for v in x])
            ax.margins(x=.13, y=.22)
            ax.set_ylim(bottom=0)
            if metric == "largest_cluster_fraction_forest":
                ax.set_ylim(0, 1.05)
            ax.tick_params(axis="y", labelleft=(col == 0))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            panel(ax, f"({chr(97+row*3+col)}) {names[row]}")
            if col == 0:
                ax.set_ylabel(units[row])
            if row == 5:
                ax.set_xlabel(parameter_names[col], labelpad=7)
    fig.align_ylabels(axes[:, 0])
    median = Line2D([], [], marker="o", color=NEUTRAL, lw=1.1, markersize=3.1)
    interval = axes[0, 0].errorbar([], [], yerr=[], fmt="none", color=NEUTRAL, capsize=2)
    fig.legend([median, interval], ["Median", "5th–95th percentiles across landscapes"],
               loc="lower center", bbox_to_anchor=(.53, .018), ncol=2, frameon=False,
               handlelength=1.4, columnspacing=1.1)
    export(fig, "access_individual_parameter_effects")
    listing = subprocess.run(["pdffonts", str(FIGURES / "access_individual_parameter_effects.pdf")],
                             check=True, text=True, capture_output=True).stdout
    assert "LMRoman8-Regular" in listing
    save_json(RESULTS / "logs/access_individual_parameter_effects_audit.json", {
        "scope": "plot only; no simulation or new inference", "selection": spec,
        "plotted_rows": len(plotted), "min_valid_runs": int(plotted.n_valid.min()),
        "max_valid_runs": int(plotted.n_valid.max()),
        "interval": "5th–95th percentiles across finite landscapes, not confidence intervals",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "embedded_fonts": listing.splitlines()[2:]})
    print("Exported access_individual_parameter_effects (18 panels, saved summaries only)")



def access_combined_shares(c: dict) -> None:
    """One common grid for all six metrics and the three focal rho values."""
    source = FINAL / "results" / c["access_results"] / "tables/02_factor_decomposition.csv"
    data = read_csv(source)
    assert len(data) == 144 and not data.duplicated(["rho", "H_target", "metric"]).any()
    columns = ["share_eta", "share_k", "share_lambda"]
    data.loc[~data.sufficient_replication, columns] = np.nan
    data.to_csv(TABLES / "access_combined_main_effect_shares.csv", index=False)
    fig, axes = plt.subplots(6, 3, figsize=figure_size(185), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.985, bottom=.125, top=.935, wspace=.18, hspace=.52)
    names = ["ED", "LCF", r"$\chi$", "MNN", "MIP", "D50"]
    handles = [Line2D([], [], color=color, marker="o", markersize=3, lw=1.1, label=label)
               for color,label in [(BLUE,r"$\eta$"),(ORANGE,"$k$"),(GREEN,r"$\lambda$")]]
    handles.append(Line2D([],[],color=NEUTRAL,marker="o",markerfacecolor="white",linestyle="none",
                          markersize=4,label="Sensitive to run omission"))
    for row,metric in enumerate(METRICS):
        for col,rho in enumerate((.9,.99,.999)):
            ax=axes[row,col]
            part=data[data.metric.eq(metric)&np.isclose(data.rho,rho)].sort_values("H_target")
            assert len(part)==8
            for key,color in zip(columns,(BLUE,ORANGE,GREEN)):
                ax.plot(part.H_target,part[key],"o-",color=color,lw=1.1,markersize=2.6)
                flag=part.seed_influence_status.eq("sensitive")&part[key].notna()
                ax.scatter(part.loc[flag,"H_target"],part.loc[flag,key],facecolors="white",edgecolors=color,s=14,zorder=4)
            if part[columns].notna().sum().sum()==0:
                ax.text(.5,.5,"No defined shares",transform=ax.transAxes,ha="center",va="center",fontsize=8.5,color=NEUTRAL)
            panel(ax,f"({chr(97+3*row+col)}) {names[row]}")
            ax.set(xlim=(1,0),ylim=(-.025,1.025),xticks=[1,.5,0],yticks=[0,.5,1])
            ax.grid(axis="y",color="#E5E5E5",lw=.4)
            ax.spines[["top","right"]].set_visible(False)
            if row==0:
                ax.text(.5,1.40,rf"$\rho={rho:g}$",transform=ax.transAxes,ha="center",va="bottom",fontsize=9.5)
    fig.supxlabel("Forest cover $H$")
    fig.supylabel("Main-effect share of variation")
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.012),ncol=4,frameon=False,
               handlelength=1.3,columnspacing=1.1)
    export(fig,"access_combined_main_effect_shares")
    listing=subprocess.run(["pdffonts",str(FIGURES/"access_combined_main_effect_shares.pdf")],check=True,text=True,capture_output=True).stdout
    assert "LMRoman8-Regular" in listing
    save_json(RESULTS/"logs/access_combined_main_effect_shares_audit.json",{
        "scope":"layout-only consolidation of three existing six-metric share figures",
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"n_rows":len(data),
        "omitted_insufficient_rows":int((~data.sufficient_replication).sum()),
        "embedded_fonts":listing.splitlines()[2:]})
    print("Exported combined Access main-effect shares (18 panels)")



def distance_center_comparison() -> None:
    """First-stage distance comparison, using saved within-H summaries only."""
    source = FINAL / "results/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/tables/01_comparison_by_h.csv"
    data = read_csv(source)
    names = [("mean_nn", "MNN"), ("mean_pair", "MIP"), ("d50", "D50")]
    parts=[]
    for metric,_ in names:
        part=data[data.metric.eq(metric)&data.candidate_representation.eq("centroid_complete")
                  &data.reference_representation.eq("edge_complete")].copy()
        assert len(part)==8 and part.n_landscapes.sum()==1970 and part.n_seed_blocks.eq(10).all()
        part["absolute_percentage_difference_median"]=100*part.median_abs_relative_difference
        parts.append(part)
    pd.concat(parts).to_csv(TABLES/"center_to_center_comparison_by_h.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=figure_size(72),sharey=True)
    fig.subplots_adjust(left=.12,right=.985,top=.84,bottom=.28,wspace=.30)
    for j,(ax,key,limits,xlabel,descriptor) in enumerate([
        (axes[0],"spearman",(-1,1),"Within-$H$ Spearman correlation","(a) Rank agreement"),
        (axes[1],"absolute_percentage_difference_median",(0,1050),"Median absolute difference (%)","(b) Value differences")]):
        for i,part in enumerate(parts):
            values=part[key].dropna().to_numpy(float)
            lo,hi=values.min(),values.max()
            ax.plot([lo,hi],[i,i],color=BLUE,lw=2,solid_capstyle="round")
            ax.scatter(values,np.full(len(values),i),s=16,facecolor="white",edgecolor=BLUE,linewidth=1,zorder=3,clip_on=False)
            text=(f"{lo:.2f}–{hi:.2f}" if j==0 else f"{lo:.1f}–{hi:.1f}")
            x=(lo+hi)/2 if j==0 else max(hi,250)
            ax.text(x,i-.22,text,ha="center",va="bottom",fontsize=8.5)
        ax.set(xlim=limits,ylim=(2.5,-.65),yticks=range(3))
        ax.set_xlabel(xlabel)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.axvline(1 if j==0 else 0,color="#B9B9B9",lw=.6,zorder=0)
        panel(ax,descriptor)
    axes[0].set_yticklabels([label for _,label in names])
    axes[0].set_xticks([-1,0,1]);axes[1].set_xticks([0,500,1000])
    fig.legend(handles=[Line2D([],[],marker="o",markerfacecolor="white",markeredgecolor=BLUE,color="none",label="Within-H summary"),
                        Line2D([],[],color=BLUE,lw=2,label="Min–max across H (not a CI)")],
               loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
    export(fig,"center_to_center_comparison")
    module_path = FINAL / "code/analyses/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/01_distance_graph_selection.py"
    module_spec = importlib.util.spec_from_file_location("distance_structural_plotter", module_path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    appendix_dir = FIGURES / "distance_structural_only"
    appendix_dir.mkdir(exist_ok=True)
    module.make_figures(data, appendix_dir, structural_only=True)
    fonts={}
    for name in ("center_to_center_comparison","distance_representation_decision"):
        listing=subprocess.run(["pdffonts",str(FIGURES/f"{name}.pdf")],check=True,text=True,capture_output=True).stdout
        assert "LMRoman8-Regular" in listing
        fonts[name]=listing.splitlines()[2:]
    save_json(RESULTS/"logs/center_to_center_comparison_audit.json",{
        "scope":"first-stage center-to-center versus exact edge-to-edge comparison; saved results only",
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"n_H_per_metric":8,
        "range":"min–max across defined within-H summaries, not a confidence interval",
        "max_MNN_median_absolute_percentage_difference":float(parts[0].absolute_percentage_difference_median.max()),
        "embedded_fonts":fonts})
    print("Exported first-stage center-to-center comparison and relabelled schematic")



def redundancy_main_figures(c: dict) -> None:
    """Promote two existing appendix figures; only terminology changes."""
    def load_module(relative, name):
        path=FINAL/"code/analyses"/relative
        INPUTS.add(path)
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
        return module
    dots=load_module(Path("00_thesis_figures/06_supervisor_revision_figures.py"),"redundancy_dot_plotter")
    dots.correlation_dots()
    producer=load_module(Path("01_landscape/01_02_preliminary_analyses/01_02_02_metric_redundancy/04_metric_redundancy_summary.py"),"redundancy_curve_plotter")
    producer.SHORT_LABELS=["ED","LCF",r"$\chi$","MNN","MIP","D50"]
    def export_preview(fig, path):
        export_figure(fig,path)
        fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight",facecolor="white")
    producer.export_figure=export_preview
    directory=FINAL/"results"/c["redundancy_results"]
    table=read_csv(directory/"tables/03_pairwise_by_h.csv")
    summary=read_csv(directory/"tables/03_redundancy_by_pair.csv")
    settings=read_json(FINAL/"config/01_landscape/01_02_preliminary_analyses/01_02_02_metric_redundancy/03_metric_redundancy.json")
    assert len(summary)==15 and summary.diagnostic_plot_flag.sum()==4
    assert producer.stability_figures(table,summary,settings,directory/"figures")==1
    fonts={}
    for path in [dots.OUT/"figures/metric_redundancy_dots.pdf",directory/"figures/03_flagged_pair_stability_01.pdf"]:
        listing=subprocess.run(["pdffonts",str(path)],check=True,text=True,capture_output=True).stdout
        assert "LMRoman8-Regular" in listing
        fonts[str(path.relative_to(FINAL))]=listing.splitlines()[2:]
    save_json(RESULTS/"logs/redundancy_main_promotion_audit.json",{
        "scope":"two existing appendix plots promoted; labels only; no numerical analysis",
        "source_sha256":{str(path.relative_to(FINAL)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(INPUTS)},
        "embedded_fonts":fonts})
    print("Updated two redundancy figures for the main chapter; numerical tables unchanged")



def majority_aggregation_schematic() -> None:
    """Constructed binary example of the >=50% rule; no real-data rerun."""
    yy,xx=np.indices((16,16))
    source=((((xx-5)/5)**2+((yy-5)/4)**2<=1)|(((xx-11)/3.6)**2+((yy-11)/3.6)**2<=1)).astype(np.uint8)
    source[12:14,1:3]=1
    selected=(2,1)
    example=np.array([[1,1,1,0],[1,1,0,0],[1,0,0,0],[1,0,0,1]],dtype=np.uint8)
    source[8:12,4:8]=example
    counts=source.reshape(4,4,4,4).sum(axis=(1,3))
    coarse=(counts>=8).astype(np.uint8)
    assert example.sum()==8 and counts[selected]==8 and coarse[selected]==1
    assert source[8:12,4:8].sum()==counts[selected]
    np.savez_compressed(DATA/"majority_aggregation_example.npz",source=source,coarse=coarse,
                        forest_pixel_counts=counts,selected_coarse_cell=np.array(selected))
    pd.DataFrame([{"row":i,"column":j,"forest_source_pixels":int(counts[i,j]),
                   "total_source_pixels":16,"output_forest":int(coarse[i,j])}
                  for i in range(4) for j in range(4)]).to_csv(TABLES/"majority_aggregation_cells.csv",index=False)
    cmap=ListedColormap([LANDSCAPE_COLORS["non_habitat"],LANDSCAPE_COLORS["habitat"]])
    fig,axes=plt.subplots(1,3,figsize=figure_size(81))
    fig.subplots_adjust(left=.015,right=.985,top=.87,bottom=.27,wspace=.25)
    for ax,grid,desc in zip(axes,[source,example,coarse],
                            ["(a) Original forest map","(b) One output cell","(c) Aggregated landscape"]):
        n=grid.shape[0]
        ax.imshow(grid,cmap=cmap,vmin=0,vmax=1,origin="upper",interpolation="nearest",aspect="equal")
        ax.set_xticks(np.arange(-.5,n,1),minor=True);ax.set_yticks(np.arange(-.5,n,1),minor=True)
        ax.grid(which="minor",color="white",lw=.35 if n==16 else .7)
        ax.set_xticks([]);ax.set_yticks([]);ax.tick_params(which="both",length=0)
        ax.spines[:].set_visible(False)
        panel(ax,desc)
    for bound in [3.5,7.5,11.5]:
        axes[0].axhline(bound,color="#8A8A8A",lw=.6);axes[0].axvline(bound,color="#8A8A8A",lw=.6)
    for ax,xy,size in [(axes[0],(3.5,7.5),4),(axes[1],(-.5,-.5),4),(axes[2],(.5,1.5),1)]:
        ax.add_patch(Rectangle(xy,size,size,fill=False,edgecolor=ORANGE,lw=2,clip_on=False))
    axes[0].text(.5,-.08,"Finer forest pixels",transform=axes[0].transAxes,ha="center",va="top",fontsize=8.5)
    axes[1].text(.5,-.08,"8 of 16 pixels are forest\nAt least half: classified as forest",transform=axes[1].transAxes,ha="center",va="top",fontsize=8.5)
    axes[2].text(.5,-.08,"One class per output cell",transform=axes[2].transAxes,ha="center",va="top",fontsize=8.5)
    fig.legend(handles=[Patch(facecolor=LANDSCAPE_COLORS["habitat"],label="Forest"),
                        Patch(facecolor=LANDSCAPE_COLORS["non_habitat"],label="Non-forest")],
               loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
    export(fig,"majority_aggregation_schematic")
    listing=subprocess.run(["pdffonts",str(FIGURES/"majority_aggregation_schematic.pdf")],check=True,text=True,capture_output=True).stdout
    assert "LMRoman8-Regular" in listing
    save_json(RESULTS/"logs/majority_aggregation_schematic_audit.json",{
        "scope":"constructed illustration; not a sampled landscape or analysis rerun",
        "source_shape":[16,16],"output_shape":[4,4],"classification_rule":"forest when fraction >= 0.5",
        "tie_example_forest_pixels":8,"tie_example_total_pixels":16,"tie_classified_forest":True,
        "actual_analysis_output_shape":[200,200],"embedded_fonts":listing.splitlines()[2:],
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print("Exported majority-aggregation schematic with verified 50% tie example")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--figure", choices=["distance", "access-effects", "access-shares", "center-comparison", "redundancy-main", "majority-aggregation"], help="Revise the distance schematic and quantitative overview, and refresh the review PDF")
    args = parser.parse_args()
    if args.figure and (args.inspect or args.diagnostics_only):
        parser.error("--figure cannot be combined with --inspect or --diagnostics-only")
    c = read_json(CONFIG)
    for directory in [DATA, FIGURES, TABLES, RESULTS / "logs"]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.figure == "majority-aggregation":
        apply_final_style()
        majority_aggregation_schematic()
        return
    if args.figure == "redundancy-main":
        apply_final_style()
        redundancy_main_figures(c)
        return
    if args.figure == "center-comparison":
        apply_final_style()
        distance_center_comparison()
        return
    if args.figure == "access-shares":
        apply_final_style()
        access_combined_shares(c)
        return
    if args.figure == "access-effects":
        apply_final_style()
        access_individual_effects(c)
        return
    if args.diagnostics_only:
        rerender_appendix_diagnostics(c)
        return
    preserved = {}
    if args.figure:
        preflight = read_json(RESULTS / "logs/figure_source_and_export_audit.json")
        preserved = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in FIGURES.iterdir()
                     if path.suffix in {".pdf", ".svg", ".png"}
                     and path.stem not in {"distance_representation_decision", "distance_metric_agreement"}}
    else:
        preflight = {"access": access_summary(c, True), "redundancy": redundancy(c, True),
                     "real_sampling": real_sampling(c, True), "occupancy_domain": occupancy_domain(c, True),
                     "distance_agreement": distance_agreement(True)}
    if args.inspect:
        print(json.dumps(preflight, indent=2))
        return
    apply_final_style()
    preflight["distance"] = distance_schematic()
    preflight["distance_agreement"] = distance_agreement()
    if not args.figure:
        workflow()
        access_summary(c)
        redundancy(c)
        real_sampling(c)
        occupancy_domain(c)
        occupancy_domain(c, full_pilot=True)
    names = ["study_design_overview", "distance_representation_decision", "distance_metric_agreement", "access_parameter_effect_shares",
             "metric_redundancy_summary", "real_landscape_sampling_map", "species_parameter_domain_heatmap"]
    review = FINAL / c["review_pdf"]
    if review.exists():
        review.unlink()
    subprocess.run(["pdfunite", *[str(FIGURES / f"{name}.pdf") for name in names], str(review)], check=True, capture_output=True)
    preflight["embedded_fonts"] = font_audit()
    sources = preflight.get("source_sha256", {})
    sources.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(INPUTS) if path.name != "figure_source_and_export_audit.json"})
    preflight["source_sha256"] = sources
    preflight["font_files"] = [str(path) for path in roman_font_files().values()]
    preflight["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info = subprocess.run(["pdfinfo", str(review)], check=True, text=True, capture_output=True).stdout
    preflight["review_pdf_pages"] = int(next(line.split(":")[1] for line in info.splitlines() if line.startswith("Pages:")))
    assert preflight["review_pdf_pages"] == 7
    if preserved:
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in preserved.items())
        preflight["selective_revision"] = {"figure": args.figure, "unchanged_assets_sha256":
            {str(path.relative_to(RESULTS)): digest for path, digest in sorted(preserved.items())}}
    save_json(RESULTS / "logs/figure_source_and_export_audit.json", preflight)
    print(f"Exported {'two distance figures only' if args.figure else 'seven main-text figures'} and {review.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
