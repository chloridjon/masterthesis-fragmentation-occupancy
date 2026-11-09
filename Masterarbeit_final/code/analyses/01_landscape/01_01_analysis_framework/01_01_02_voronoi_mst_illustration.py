"""Show a small patch network and its minimum spanning tree.

The default is a constructed twelve-patch landscape for legible tree
construction. The earlier saved Access-based example remains reproducible
with --example saved-access. Both use the final implementation's exact
edge-to-edge neighbour weights and record a comparison with the complete
hybrid graph. No deforestation trajectory or analysis result is changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.landscape import Landscape
from src.landscape.landscape_stats import (
    cluster_centroids, _complete_graph_mst_edges_from_distances,
)
from src.tools.final_plot_style import (
    CLUSTER_PALETTE_RGB, VORONOI_NETWORK_COLORS,
    apply_final_style, export_figure, figure_size,
)

REL = Path("01_landscape/01_01_analysis_framework/01_01_02_landscape_metrics")
SOURCE_STEM = "access_voronoi_rho099_h030"
STEM = "access_voronoi_mst_rho099_h030"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("small", "saved-access"), default="small")
    args = parser.parse_args()
    source = None
    if args.example == "small":
        stem_name = "small_patch_voronoi_mst"
        # Deliberately uneven spacing: close patch groups separated by broad
        # gaps. This is illustrative geometry, not a selected simulation.
        patch_specs = [(6, 8, 2), (10, 16, 2), (17, 6, 3), (20, 18, 2),
                       (55, 10, 2), (65, 15, 3), (57, 25, 2),
                       (15, 61, 3), (24, 73, 2), (30, 61, 2), (40, 71, 2),
                       (73, 48, 3)]
        rows, cols = np.indices((80, 80))
        grid = np.zeros((80, 80), dtype=np.uint8)
        for row, col, radius in patch_specs:
            grid[(rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2] = 1
        provenance = {"constructed_example": True, "patch_specs_row_col_radius": patch_specs}
    else:
        stem_name = STEM
        source = FINAL / "data/processed" / REL / f"{SOURCE_STEM}.npz"
        source_manifest = FINAL / "results" / REL / "logs" / f"{SOURCE_STEM}_manifest.json"
        provenance = json.loads(source_manifest.read_text())
        with np.load(source, allow_pickle=False) as saved:
            grid = saved["grid"]
            labels = saved["labels"]
            pairs = saved["neighbor_pairs"] - 1
            colors = saved["cluster_colors"][1:] / 255
        if hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest() != provenance["grid_sha256"]:
            raise RuntimeError("Source landscape differs from Figure 2.4.")
    digest = hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest()

    landscape = Landscape(grid, edge_depth=0)
    landscape.ensure_clusters(edge_depth=0, connectivity=4, boundary_mode="open",
                              distance_mode="voronoi_dt", voronoi_depth=1)
    if args.example == "saved-access" and not np.array_equal(labels, landscape.core_cluster_labels):
        raise RuntimeError("Patch identities differ from the source illustration.")
    n = int(landscape.n_clusters)
    graph = landscape.voronoi_neighbor_graph
    upper = sparse.triu(graph, k=1).tocoo()
    actual_pairs = np.column_stack((upper.row, upper.col))
    actual_pairs = actual_pairs[np.lexsort((actual_pairs[:, 1], actual_pairs[:, 0]))]
    if args.example == "small":
        labels = landscape.core_cluster_labels
        pairs = actual_pairs
        colors = np.asarray(CLUSTER_PALETTE_RGB[:n]) / 255
        if n != 12 or len(pairs) != 24:
            raise RuntimeError("Constructed geometry no longer matches the caption.")
    if not np.array_equal(pairs, actual_pairs) or connected_components(graph)[0] != 1:
        raise RuntimeError("Voronoi topology differs from the source or is disconnected.")
    weights = np.asarray(graph[pairs[:, 0], pairs[:, 1]]).ravel()
    mst = minimum_spanning_tree(graph).tocoo()
    mst_pairs = np.sort(np.column_stack((mst.row, mst.col)), axis=1)
    retained_pairs = {tuple(pair) for pair in mst_pairs}
    retained = np.asarray([tuple(pair) in retained_pairs for pair in pairs])
    if (mst.nnz != n - 1 or int(retained.sum()) != n - 1
            or connected_components(mst, directed=False)[0] != 1):
        raise RuntimeError("The retained edges do not form a spanning tree.")
    # Independently verify minimal total weight using the existing Prim routine.
    restricted = np.full((n, n), np.inf)
    restricted[pairs[:, 0], pairs[:, 1]] = weights
    restricted[pairs[:, 1], pairs[:, 0]] = weights
    _, _, reference_weights = _complete_graph_mst_edges_from_distances(restricted)
    if not np.isclose(weights[retained].sum(), reference_weights.sum()):
        raise RuntimeError("Spanning tree is not minimum weight.")
    cr, cc, cw = _complete_graph_mst_edges_from_distances(landscape.cluster_dist_matrix)
    also_complete_mst = bool(np.isclose(weights[retained].sum(), cw.sum(), atol=1e-10, rtol=1e-12))
    complete_pairs = {tuple(sorted(pair)) for pair in zip(cr, cc)}
    non_voronoi_complete_edges = len(complete_pairs - {tuple(pair) for pair in pairs})

    centers = cluster_centroids(labels, n, boundary_mode="open")
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    # Retain the viewer's area mapping, with larger schematic nodes so
    # two-digit patch identifiers remain readable at the printed size.
    side = min(grid.shape)
    min_radius = max(1.15, side * .0035)
    max_radius = max(min_radius + .8, min(10.5, side * .028))
    if args.example == "small":
        min_radius, max_radius = 1.45, 3.0
    radii = min_radius + np.sqrt(areas / areas.max()) * (max_radius - min_radius)
    segments = centers[pairs]
    # Literal component merging on the complete hybrid MST defines D50.
    parent = np.arange(n)
    mass = areas.astype(float).copy()
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    d50 = 0.0
    if mass.max() < mass.sum() / 2:
        for idx in np.argsort(cw, kind="stable"):
            a, b = find(int(cr[idx])), find(int(cc[idx]))
            if a != b:
                parent[b] = a
                mass[a] += mass[b]
            if mass[a] >= areas.sum() / 2:
                d50 = float(cw[idx])
                break
    distances = landscape.cluster_dist_matrix
    mean_pair = float(distances[np.triu_indices(n, 1)].mean())
    mean_nn = float(np.mean([min(w for (a, b), w in zip(pairs, weights)
                                if i in (a, b)) for i in range(n)]))
    if args.example == "small" and not also_complete_mst:
        raise RuntimeError("The displayed tree must also represent the complete hybrid MST.")


    apply_final_style()
    height_mm = 100 if args.example == "small" else 97
    fig = plt.figure(figsize=figure_size(height_mm=height_mm))
    panel_width = .44
    panel_height = 160 * panel_width / height_mm
    padding = float(radii.max() + 1)
    axes = [fig.add_axes([left, .29, panel_width, panel_height]) for left in (.01, .55)]
    for ax in axes:
        ax.set_xlim(-.5 - padding, grid.shape[1] - .5 + padding)
        ax.set_ylim(grid.shape[0] - .5 + padding, -.5 - padding)
        ax.set_aspect("equal")
        ax.set_axis_off()
    axes[0].add_collection(LineCollection(segments, colors=VORONOI_NETWORK_COLORS["edge"],
                                          linewidths=.9, alpha=.65, zorder=1))
    axes[1].add_collection(LineCollection(segments[~retained],
                                          colors=VORONOI_NETWORK_COLORS["non_tree"],
                                          linewidths=.75, zorder=1))
    axes[1].add_collection(LineCollection(segments[retained],
                                          colors=VORONOI_NETWORK_COLORS["mst"],
                                          linewidths=1.5, zorder=2))
    highlighted = retained & (weights <= d50 + 1e-12)
    axes[1].add_collection(LineCollection(segments[highlighted], colors="#D55E00",
                                          linewidths=2.1, zorder=2.5))
    # Convert the map's cell radii to scatter point areas at the printed size.
    diameter_points = 2 * radii * (160 * panel_width / (grid.shape[1] + 2 * padding)) * 72 / 25.4
    order = np.argsort(areas, kind="stable")
    for ax in axes:
        ax.scatter(centers[order, 0], centers[order, 1], s=diameter_points[order] ** 2,
                   c=colors[order], edgecolors="white", linewidths=.3, zorder=3)
        if args.example == "small":
            for index, (x, y) in enumerate(centers, start=1):
                # Choose label contrast from each node's actual colour.
                rgb = colors[index - 1]
                linear_rgb = np.where(rgb <= .04045, rgb / 12.92,
                                      ((rgb + .055) / 1.055) ** 2.4)
                luminance = float(linear_rgb @ np.array([.2126, .7152, .0722]))
                label_color = "#202020" if luminance > .179 else "white"
                ax.text(x, y, str(index), ha="center", va="center",
                        color=label_color, fontsize=8.5, zorder=4)
                ax.annotate(f"A={areas[index-1]}", (x, y), xytext=(0, -9),
                            textcoords="offset points", ha="center", va="top", fontsize=8.5,
                            bbox=dict(facecolor="white", edgecolor="none", pad=.1), zorder=5)
    for ax, label in zip(axes, ("(a) Voronoi network", "(b) Minimum spanning tree")):
        ax.text(0, 1.045, label, transform=ax.transAxes, ha="left", va="bottom")
    mid_y = .29 + panel_height / 2
    fig.add_artist(FancyArrowPatch((.475, mid_y), (.525, mid_y),
                                  transform=fig.transFigure, arrowstyle="-|>",
                                  mutation_scale=9, linewidth=1.1, color="#404040"))
    fig.text(.5, .215,
             fr"Mean NN distance $\bar d_{{\mathrm{{NN}}}}={mean_nn:.2f}$; "
             fr"mean inter-patch distance $\bar d_{{\mathrm{{pair}}}}={mean_pair:.2f}$ cells",
             ha="center", fontsize=8.5)
    fig.text(.5, .155, fr"Complete-MST $D_{{50}}={d50:.2f}$ cells; orange links join at least half the forest area",
             ha="center", fontsize=8.5)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=np.asarray(CLUSTER_PALETTE_RGB[0]) / 255,
               markersize=5, label="Habitat patch"),
        Line2D([], [], color=VORONOI_NETWORK_COLORS["edge"], alpha=.65, linewidth=.55,
               label="Voronoi edge"),
        Line2D([], [], color=VORONOI_NETWORK_COLORS["mst"], linewidth=.8, label="MST edge"),
        Line2D([], [], color=VORONOI_NETWORK_COLORS["non_tree"], linewidth=.7, label="Non-tree edge"),
    ], loc="lower center", bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
    stem = FINAL / "results" / REL / "figures" / stem_name
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    data_dir = FINAL / "data/processed" / REL
    np.savez_compressed(data_dir / f"{stem_name}.npz", grid=grid, labels=labels,
                        centroids_xy=centers, patch_areas=areas, colors_rgb=colors,
                        node_radii_cells=radii, neighbor_pairs=pairs + 1,
                        edge_weights=weights, mst_edge_mask=retained)
    tables = FINAL / "results" / REL / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / f"{stem_name}_edges.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("patch_i", "patch_j", "edge_distance_cells", "in_voronoi_mst"))
        writer.writerows((int(a + 1), int(b + 1), float(w), int(keep))
                         for (a, b), w, keep in zip(pairs, weights, retained))
    metadata = {
        "source_npz": str(source.relative_to(FINAL)) if source is not None else None,
        "source_npz_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source is not None else None,
        "example": args.example, "construction": provenance if args.example == "small" else None,
        "grid_sha256": digest, "H": float(grid.mean()), "rho": provenance.get("rho"),
        "seed": provenance.get("seed"), "boundary_mode": "open", "connectivity": 4,
        "distance_mode": "voronoi_dt", "voronoi_depth": 1, "edge_depth": 0,
        "graph": "direct Voronoi-neighbour graph", "edge_weights": "exact edge-to-edge cell-centre distances",
        "mst_algorithm": "scipy.sparse.csgraph.minimum_spanning_tree",
        "mean_nearest_neighbor_distance": mean_nn, "mean_pairwise_distance": mean_pair,
        "complete_mst_d50": d50, "patch_areas": areas.tolist(), "n_patches": n, "n_voronoi_edges": len(pairs), "n_mst_edges": int(retained.sum()),
        "voronoi_mst_total_weight": float(weights[retained].sum()),
        "complete_hybrid_mst_total_weight": float(cw.sum()),
        "complete_hybrid_mst_non_voronoi_edges": non_voronoi_complete_edges,
        "voronoi_tree_is_also_complete_hybrid_mst": also_complete_mst,
        "style_reference": "applications/defragmentation_live.py::_render_voronoi_network",
        "script": str(Path(__file__).relative_to(FINAL)),
    }
    log = FINAL / "results" / REL / "logs" / f"{stem_name}_manifest.json"
    log.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Verified {n} patches, {len(pairs)} Voronoi edges, {int(retained.sum())} MST edges.")
    print(f"Complete hybrid MST has {non_voronoi_complete_edges} non-Voronoi edges.")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
