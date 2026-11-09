"""Explain the hybrid distance matrix on a deterministic invented landscape.

Run with Python from any directory. No simulation or raw data are used.
The production Landscape class determines neighbours and hybrid weights;
an independent exhaustive calculation verifies every displayed segment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hybrid-schematic-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/hybrid-schematic-numba")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1793923200")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(ROOT / "code/base"))
from src.landscape.landscape import Landscape
from src.landscape.landscape_stats import cluster_centroids
from src.tools.final_plot_style import (
    CLUSTER_PALETTE_RGB, LANDSCAPE_COLORS, NEUTRAL,
    apply_final_style, export_figure, figure_size, roman_font_files,
)

STEM = "voronoi_hybrid_schematic"
OUT = ROOT / "results/00_thesis_figures/21_voronoi_hybrid_schematic"
BLUE, ORANGE = [matplotlib.colors.to_hex(np.asarray(c) / 255)
                for c in CLUSTER_PALETTE_RGB[:2]]
# Each patch is the union of overlapping ellipses: (x, y, radius_x, radius_y).
# The arrangement is fixed by design, rather than selected from simulations.
PATCHES = (
    ((17, 19, 8, 5), (20, 23, 5, 5), (12, 17, 4, 4)),
    ((60, 14, 9, 5), (65, 17, 6, 5), (55, 18, 5, 5)),
    ((108, 29, 7, 8), (104, 24, 5, 4), (112, 33, 5, 4)),
    ((25, 67, 8, 6), (20, 71, 5, 4), (29, 63, 5, 5)),
    ((71, 79, 10, 5), (75, 74, 5, 5), (65, 77, 6, 5)),
    ((111, 66, 6, 7), (107, 63, 5, 4), (114, 70, 4, 4)),
)


def cell_polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Trace exact exposed cell sides; return vector polygon vertices."""
    edges = {}
    for y, x in np.argwhere(mask):
        # Doubled integer coordinates avoid floating-point joins.
        corners = ((2*x-1, 2*y-1), (2*x+1, 2*y-1),
                   (2*x+1, 2*y+1), (2*x-1, 2*y+1))
        for k in range(4):
            a, b = corners[k], corners[(k+1) % 4]
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = None
    following = {a: b for a, b in edges}
    if len(following) != len(edges):
        raise RuntimeError("Ambiguous contour in the constructed patch.")
    polygons = []
    while following:
        start = min(following)
        loop, point = [], start
        while True:
            loop.append(point)
            point = following.pop(point)
            if point == start:
                break
        polygons.append(np.asarray(loop) / 2)
    return polygons


def main() -> None:
    yy, xx = np.indices((92, 132))
    grid = np.zeros(xx.shape, dtype=np.uint8)
    for ellipses in PATCHES:
        patch = np.zeros_like(grid, dtype=bool)
        for x, y, rx, ry in ellipses:
            patch |= ((xx-x)/rx)**2 + ((yy-y)/ry)**2 <= 1
        if np.any(patch & (grid > 0)):
            raise RuntimeError("Invented patches overlap.")
        grid[patch] = 1
    landscape = Landscape(grid, edge_depth=0)
    landscape.ensure_clusters(edge_depth=0, connectivity=4, boundary_mode="open",
                              distance_mode="voronoi_dt", voronoi_depth=1)
    labels = landscape.core_cluster_labels
    n = int(landscape.n_clusters)
    if n != len(PATCHES):
        raise RuntimeError("Expected six separate patches.")
    # Keep A--F tied to the specified spatial arrangement, independent of
    # the connected-component scan order used by the production code.
    ids = np.asarray([labels[int(p[0][1]), int(p[0][0])] for p in PATCHES])
    centers = cluster_centroids(labels, n, boundary_mode="open")[ids-1]
    graph = landscape.voronoi_neighbor_graph.toarray()[np.ix_(ids-1, ids-1)]
    hybrid = landscape.cluster_dist_matrix[np.ix_(ids-1, ids-1)]
    points = [np.argwhere(labels == label)[:, ::-1] for label in ids]
    _, nearest = ndimage.distance_transform_edt(grid == 0, return_indices=True)
    regions = labels[tuple(nearest)]
    region_pairs = set()
    for a, b in ((regions[:, :-1], regions[:, 1:]),
                 (regions[:-1], regions[1:])):
        for i, j in zip(a[a != b], b[a != b]):
            region_pairs.add(tuple(sorted((int(i), int(j)))))

    records, edge_segments, center_segments = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            neighbour = graph[i, j] > 0
            if neighbour != (tuple(sorted((int(ids[i]), int(ids[j])))) in region_pairs):
                raise RuntimeError("Display regions and production topology disagree.")
            all_distances = cdist(points[i], points[j])
            a, b = np.unravel_index(np.argmin(all_distances), all_distances.shape)
            exact = float(all_distances[a, b])
            if neighbour:
                segment = np.asarray([points[i][a], points[j][b]], dtype=float)
                edge_segments.append(segment)
                kind = "edge-to-edge"
            else:
                segment = centers[[i, j]]
                center_segments.append(segment)
                kind = "center-to-center"
            displayed = float(np.linalg.norm(segment[1]-segment[0]))
            if not np.isclose(displayed, hybrid[i, j], rtol=0, atol=1e-10):
                raise RuntimeError("Displayed line length differs from the hybrid weight.")
            if neighbour and not np.isclose(displayed, exact, rtol=0, atol=1e-10):
                raise RuntimeError("Neighbour line is not the exact minimum distance.")
            records.append(dict(patch_i=chr(65+i), patch_j=chr(65+j),
                                voronoi_neighbours=bool(neighbour), distance_type=kind,
                                x_i=segment[0, 0], y_i=segment[0, 1],
                                x_j=segment[1, 0], y_j=segment[1, 1],
                                hybrid_distance=displayed, exact_edge_distance=exact))
    if len(records) != n*(n-1)//2 or not center_segments:
        raise RuntimeError("The figure must show every pair and both distance types.")

    apply_final_style()
    matplotlib.rcParams["svg.hashsalt"] = STEM
    fig = plt.figure(figsize=figure_size(height_mm=125))
    ax = fig.add_axes([2/160, 14.5/125, 156/160, (156*92/132)/125])
    ax.set_aspect("equal")
    ax.set_xlim(-.5, grid.shape[1]-.5)
    ax.set_ylim(grid.shape[0]-.5, -.5)
    ax.set_facecolor(LANDSCAPE_COLORS["non_habitat"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Low-contrast Voronoi boundaries support the neighbour definition
    # without adding a competing region-colour encoding.
    y, x = np.nonzero(regions[:, :-1] != regions[:, 1:])
    vertical = np.stack((np.column_stack((x+.5, y-.5)),
                         np.column_stack((x+.5, y+.5))), axis=1)
    y, x = np.nonzero(regions[:-1] != regions[1:])
    horizontal = np.stack((np.column_stack((x-.5, y+.5)),
                           np.column_stack((x+.5, y+.5))), axis=1)
    boundaries = np.concatenate((vertical, horizontal))
    ax.add_collection(LineCollection(boundaries, colors="#B5B5B5", linewidths=.45, zorder=1))
    for label in ids:
        for vertices in cell_polygons(labels == label):
            ax.add_patch(Polygon(vertices, closed=True,
                                 facecolor=LANDSCAPE_COLORS["habitat"],
                                 edgecolor="none", zorder=2))

    ax.add_collection(LineCollection(center_segments, colors=ORANGE,
                                     linewidths=1.25, zorder=3))
    ax.add_collection(LineCollection(edge_segments, colors=BLUE,
                                     linewidths=1.45, zorder=4))
    endpoints = np.asarray(edge_segments).reshape(-1, 2)
    ax.scatter(endpoints[:, 0], endpoints[:, 1], s=10, c=BLUE,
               edgecolors="white", linewidths=.35, zorder=5)
    ax.scatter(centers[:, 0], centers[:, 1], s=19, c=ORANGE,
               edgecolors="white", linewidths=.6, zorder=6)
    offsets = [(-11, -7), (-1, -10), (10, -6), (-12, 1), (-1, 11), (10, 5)]
    for i, ((x, y), (dx, dy)) in enumerate(zip(centers, offsets)):
        ax.text(x+dx, y+dy, chr(65+i), ha="center", va="center", color=NEUTRAL)

    handles = [
        Line2D([], [], color=BLUE, linewidth=1.45, marker="o", markersize=3,
               label="Voronoi neighbours: edge-to-edge"),
        Patch(facecolor=LANDSCAPE_COLORS["habitat"], edgecolor="none", label="Habitat patch"),
        Line2D([], [], color=ORANGE, linewidth=1.25, marker="o", markersize=3.8,
               label="All other pairs: center-to-center"),
        Line2D([], [], color="#B5B5B5", linewidth=.65, label="Voronoi boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 12.5/125),
               frameon=False, ncol=2, handlelength=2.3, labelspacing=.45)
    figure_dir = OUT / "figures"
    stem = figure_dir / STEM
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    table_dir = OUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "connections.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(table_dir / "landscape.npz", grid=grid, labels=labels,
                        patch_ids=ids, centers_xy=centers, regions=regions,
                        hybrid_distance_matrix=hybrid,
                        edge_to_edge_segments=np.asarray(edge_segments),
                        center_to_center_segments=np.asarray(center_segments))
    audit = dict(constructed_example=True, random_sampling=False,
                 patch_ellipses_xy_rx_ry=PATCHES, grid_shape=grid.shape,
                 grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),
                 n_patches=n, n_all_pairs=len(records),
                 n_edge_to_edge=len(edge_segments), n_center_to_center=len(center_segments),
                 connectivity=4, boundary_mode="open", edge_depth=0, voronoi_depth=1,
                 distance_units="dimensionless raster-cell units",
                 edge_convention="minimum Euclidean distance between boundary-cell centers",
                 verification="All 15 pair types and segment lengths checked against production; "
                              "neighbour distances independently checked exhaustively over all patch cells.",
                 roman_font_files={str(k): str(v) for k, v in roman_font_files().items()},
                 script=str(Path(__file__).relative_to(ROOT)))
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("n_patches", "n_all_pairs", "n_edge_to_edge",
                                           "n_center_to_center")}, indent=2))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
