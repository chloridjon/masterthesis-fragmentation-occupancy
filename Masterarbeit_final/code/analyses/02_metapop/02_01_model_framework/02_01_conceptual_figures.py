"""Reproducible theory illustrations; constructed examples, not scenario results.

Run from the repository root with the local project runner. No random sampling
is used. Figure 1 uses schematic polygon geometry and illustrative occupancy
probabilities. Figure 2 uses two 40 x 40 rasters containing exactly the same
five patch masks, the depth-1 Voronoi hybrid distances, b=1, x=0.5,
half-distance=8 cells and no self-colonisation. r is plotted without capacity
normalisation. Zero at/above capacity and the r=0 limit are assigned analytically.
All positive equilibria are solved with the shared model and checked for
convergence, fixed-point residuals, and monotonicity.
"""
from __future__ import annotations

import csv
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.spatial.distance import cdist

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.landscape import Landscape
from src.metapop.hanski import (
    metapopulation_capacity_and_mean_si_from_kernel,
    prepare_dispersal_kernel, run_continuous_equilibrium_from_kernel,
)
from src.tools.final_plot_style import (
    LANDSCAPE_COLORS, apply_final_style, export_figure, figure_size,
)

FIGURES = FINAL / "latex/figures/02_theoretical_framework"
DATA = FINAL / "data/processed/02_metapop/02_01_model_framework"
BLUE, ORANGE = "#0072B2", "#E69F00"
GREEN = LANDSCAPE_COLORS["habitat"]
GREY = LANDSCAPE_COLORS["non_habitat"]
HALF_DISTANCE = 8.0
B, X = 1.0, 0.5


def save(fig, name):
    stem = FIGURES / name
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def polygon_points(cx, cy, radius):
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    radii = radius * np.array([1, .9, 1.05, .85, 1.1, .95, .85, 1, .85, 1.1, .9, 1.05])
    return np.column_stack((cx + radii * np.cos(angles), cy + .8 * radii * np.sin(angles)))


def polygon_area(vertices):
    if len(vertices) < 3:
        return 0.0
    return abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1))
               - np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1))) / 2


def clip_below(vertices, height):
    """Clip a patch to a horizontal fill boundary, preserving its exact shape."""
    clipped = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        inside_start, inside_end = start[1] <= height, end[1] <= height
        if inside_start != inside_end:
            intersection = start + (end-start) * ((height-start[1]) / (end[1]-start[1]))
            clipped.append(intersection)
        if inside_end:
            clipped.append(end)
    return np.asarray(clipped).reshape(-1, 2)


def occupancy_fill(vertices, probability):
    """Fill exactly p of the polygon area (not merely p of its bounding height)."""
    lower, upper = vertices[:, 1].min(), vertices[:, 1].max()
    target = polygon_area(vertices) * probability
    for _ in range(60):
        height = (lower + upper) / 2
        if polygon_area(clip_below(vertices, height)) < target:
            lower = height
        else:
            upper = height
    filled = clip_below(vertices, (lower + upper) / 2)
    np.testing.assert_allclose(polygon_area(filled), target, rtol=1e-12, atol=1e-12)
    return filled


def connectivity_figure():
    fig = plt.figure(figsize=figure_size(125))
    ax = fig.add_axes((.04, .31, .92, .68))
    ax.set(xlim=(0, 12), ylim=(0, 9), aspect="equal")
    ax.set_facecolor("#F6F6F6")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    focal = np.array([6.2, 4.3])
    sources = [(3.5, 6.6, 1.4, .9), (1.5, 1.6, .7, .9),
               (10.2, 7.0, 1.25, .2), (10.5, 1.7, .9, .6)]
    records = []
    vertices = [polygon_points(cx, cy, radius) for cx, cy, radius, p in sources]
    areas = [polygon_area(v) for v in vertices]
    # Schematic distances are centroid distances, used only for arrow widths.
    distances = [np.linalg.norm(np.array([cx, cy]) - focal) for cx, cy, _, _ in sources]
    weights = [np.exp(-.35 * d) * area * source[3]
               for d, area, source in zip(distances, areas, sources)]
    total = sum(weights)
    for index, (source, shape, area, distance, contribution) in enumerate(
            zip(sources, vertices, areas, distances, weights), start=1):
        cx, cy, radius, p = source
        direction = focal - np.array([cx, cy]); direction /= np.linalg.norm(direction)
        start = np.array([cx, cy]) + direction * radius
        end = focal - direction * 1.12
        width = 5.0 * contribution / max(weights)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=8 + width, linewidth=width,
                                    color=BLUE, zorder=2))
        filled = occupancy_fill(shape, p)
        ax.add_patch(Polygon(shape, facecolor="white", edgecolor="none", zorder=3))
        ax.add_patch(Polygon(filled, facecolor=GREEN, edgecolor="none", zorder=4))
        ax.add_patch(Polygon(shape, facecolor="none", edgecolor=GREEN, linewidth=.8, zorder=5))
        label_y = cy - .18 * radius if p >= .5 else cy
        ax.text(cx, label_y, f"{p:.0%}", color="white" if p >= .5 else GREEN,
                ha="center", va="center", fontsize=9.5, zorder=6)
        ax.text(cx, shape[:, 1].max() + .22, str(index), color=GREEN,
                ha="center", va="bottom", fontsize=9.5, zorder=6)
        records.append(dict(source=index, x=cx, y=cy, radius=radius, area=area, occupancy=p,
                            green_area=polygon_area(filled), schematic_distance=distance,
                            contribution=contribution, arrow_width_pt=width,
                            summation_segment_fraction=contribution / total,
                            total_connectivity=total))
    ax.add_patch(Polygon(polygon_points(*focal, 1.13), facecolor="white",
                         edgecolor=BLUE, linewidth=1.8, zorder=3))
    ax.text(focal[0], focal[1] + .20, "$i$", color=BLUE, ha="center", va="center", fontsize=12, zorder=4)
    ax.text(focal[0], focal[1] - .33, "Recipient", color=BLUE, ha="center", va="center", fontsize=9.5, zorder=4)
    fig.text(.5, .267, r"Green fraction $=p_j$     Arrow width $\propto K_{ij}A_j^b p_j$",
             ha="center", fontsize=9.5)
    fig.text(.5, .21, "Sum of source contributions $=S_i$", ha="center")
    bar = fig.add_axes((.10, .135, .80, .045))
    bar.set(xlim=(0, total), ylim=(0, 1)); bar.set_axis_off()
    offset = 0.0
    for index, contribution in enumerate(weights, start=1):
        bar.add_patch(Rectangle((offset, 0), contribution, 1, facecolor=BLUE,
                                edgecolor="white", linewidth=.8))
        bar.text(offset + contribution/2, .5, str(index), color="white",
                 ha="center", va="center", fontsize=8.5)
        offset += contribution
    np.testing.assert_allclose(offset, total, rtol=1e-14)
    np.testing.assert_allclose([row["arrow_width_pt"] / sum(r["arrow_width_pt"] for r in records)
                               for row in records], [row["summation_segment_fraction"] for row in records])
    save(fig, "spatial_connectivity_occupancy")
    with (DATA / "source_patch_contributions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    metadata = dict(illustration_only=True, focal_centroid=focal.tolist(),
                    focal_occupancy_displayed=False, source_area_exponent=B,
                    schematic_kernel_alpha=.35, distance_mode="schematic_centroid",
                    occupancy_display="Exact green polygon area fraction equals source-patch occupancy probability.",
                    summation_display="Bar segment lengths are proportional to the same contributions as arrow widths.",
                    total_connectivity=total, stochastic_sampling=False)
    (DATA / "source_patch_contributions_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def landscape_pair():
    templates = []
    for radius in (3, 3.5, 2, 2.5, 2):
        y, x = np.mgrid[-4:5, -4:5]
        templates.append((x * x + y * y <= radius * radius).astype(np.int8))
    positions = {
        "clustered": [(14, 14), (23, 14), (18, 23), (26, 23), (10, 24)],
        "dispersed": [(6, 6), (33, 6), (20, 20), (6, 33), (33, 33)],
    }
    examples = {}
    for name, centers in positions.items():
        grid = np.zeros((40, 40), dtype=np.int8)
        for (cx, cy), mask in zip(centers, templates):
            grid[cy-4:cy+5, cx-4:cx+5] += mask
        assert grid.max() == 1
        land = Landscape(grid, edge_depth=0)
        land.ensure_clusters(edge_depth=0, connectivity=4, boundary_mode="open",
                             distance_mode="voronoi_dt", voronoi_depth=1)
        assert land.n_clusters == 5
        areas = np.asarray(land.cluster_areas, dtype=float)
        distance = np.asarray(land.cluster_dist_matrix, dtype=float)
        kernel = prepare_dispersal_kernel(distance, alpha=np.log(2) / HALF_DISTANCE)
        capacity, _, matrix = metapopulation_capacity_and_mean_si_from_kernel(
            areas, kernel, b=B, x=X, tol=1e-12, return_matrix=True,
            rng=np.random.default_rng(0))
        exact_capacity = float(np.max(np.abs(np.linalg.eigvals(matrix))))
        assert capacity.converged
        np.testing.assert_allclose(capacity.lambda_max, exact_capacity, rtol=1e-10)
        assert np.all(np.diag(kernel) == 0)
        # Independent check of each exact neighbour edge against all habitat cells.
        labels = land.core_cluster_labels
        graph = land.voronoi_neighbor_graph.tocoo()
        for i, j, d in zip(graph.row, graph.col, graph.data):
            np.testing.assert_allclose(d, cdist(np.argwhere(labels == i+1),
                                               np.argwhere(labels == j+1)).min())
        examples[name] = dict(grid=grid, areas=areas, distance=distance,
                              kernel=kernel, matrix=matrix, capacity=exact_capacity)
    np.testing.assert_array_equal(np.sort(examples["clustered"]["areas"]),
                                  np.sort(examples["dispersed"]["areas"]))
    assert examples["clustered"]["capacity"] > examples["dispersed"]["capacity"]
    return examples


def capacity_figure():
    examples = landscape_pair()
    upper = 1.18 * examples["clustered"]["capacity"]
    ratios = np.unique(np.r_[np.linspace(0, upper, 180),
                             [item["capacity"] * f for item in examples.values()
                              for f in (.98, .99, .995, 1.0, 1.01)]])
    print(f"Illustrative r grid: 0 to {upper:.6f}, {len(ratios)} points; half-distance={HALF_DISTANCE}, b={B}, x={X}")
    rows = []
    fig = plt.figure(figsize=figure_size(125))
    colors = {"clustered": BLUE, "dispersed": ORANGE}
    for name, left, label in (("dispersed", .17, "(a) Dispersed patches"),
                              ("clustered", .62, "(b) Clustered patches")):
        ax = fig.add_axes((left, .73, .21, .269))
        ax.imshow(examples[name]["grid"], cmap=ListedColormap([GREY, GREEN]),
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_axis_off()
        ax.text(.5, 1.04, label, ha="center", transform=ax.transAxes, color=colors[name])
    fig.text(.5, .693, "Same five patches, same habitat amount; different spatial arrangement", ha="center", fontsize=8.5)
    ax = fig.add_axes((.14, .18, .83, .445))
    ax.text(0, 1.055, "(c) Equilibrium occupancy and persistence thresholds", transform=ax.transAxes)
    for name, item in examples.items():
        means = []
        for ratio in ratios:
            if ratio == 0:
                p = np.ones(5); iterations = 0; delta = 0.; method = "zero_extinction_limit"
            elif ratio >= item["capacity"]:
                p = np.zeros(5); iterations = 0; delta = 0.; method = "analytic_threshold"
            else:
                result = run_continuous_equilibrium_from_kernel(
                    item["areas"], item["kernel"], r=ratio, b=B, x=X,
                    max_iter=500_000, tol=1e-12)
                assert result.converged, (name, ratio)
                p = result.probabilities; iterations = result.n_iter
                delta = result.max_delta; method = "fixed_point"
                assert np.all(p > 0)
            pressure = item["kernel"] @ (item["areas"] ** B * p)
            residual = np.max(np.abs((1-p)*pressure - ratio*item["areas"]**(-X)*p))
            assert residual < 1e-9
            mean = float(p.mean()); means.append(mean)
            rows.append(dict(landscape=name, r=ratio, capacity=item["capacity"],
                             mean_occupancy=mean, iterations=iterations, max_delta=delta,
                             equilibrium_residual=residual, method=method,
                             **{f"p_{i+1}": value for i, value in enumerate(p)}))
        assert np.all(np.diff(means) <= 1e-9)
        ax.plot(ratios, means, color=colors[name], label=f"{name.capitalize()} patches", linewidth=1.7)
        cap = item["capacity"]
        ax.plot(cap, 0, "o", color=colors[name], markersize=4, clip_on=False)
        ax.annotate(fr"$\lambda_M={cap:.1f}$", xy=(cap, .015), xytext=(cap, .23),
                    ha="center", color=colors[name], fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", color=colors[name], linewidth=.8))
        print(f"{name}: area={item['areas'].sum():.0f}, capacity={cap:.8f}; all equilibria verified")
        np.savez_compressed(DATA / f"capacity_{name}.npz", **item)
    ax.set(xlim=(0, upper), ylim=(0, 1.02), yticks=[0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"Extinction-to-colonisation ratio $r=e_0/c$")
    ax.set_ylabel(r"Mean equilibrium patch occupancy $\bar p^{*}$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.54, .02), ncols=2, frameon=False)
    save(fig, "persistence_threshold_capacity")
    with (DATA / "capacity_equilibria.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    metadata = dict(illustration_only=True, independent_replicates=None,
                    grid_size=[40, 40], n_patches=5, total_habitat_cells=int(examples["clustered"]["areas"].sum()),
                    b=B, x=X, half_distance=HALF_DISTANCE, self_colonization=0,
                    distance_mode="voronoi_dt", voronoi_depth=1, boundary_mode="open",
                    ratio_min=0, ratio_max=upper, ratio_points=len(ratios),
                    equilibrium_tolerance=1e-12, residual_limit=1e-9,
                    capacities={name: item["capacity"] for name, item in examples.items()},
                    source_figure_distance_note="Schematic centroid distances for arrow-width illustration only.")
    (DATA / "conceptual_figures_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "connectivity", "capacity"), default="all")
    args = parser.parse_args()
    apply_final_style()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.figure in ("all", "connectivity"):
        connectivity_figure()
    if args.figure in ("all", "capacity"):
        capacity_figure()


if __name__ == "__main__":
    main()
