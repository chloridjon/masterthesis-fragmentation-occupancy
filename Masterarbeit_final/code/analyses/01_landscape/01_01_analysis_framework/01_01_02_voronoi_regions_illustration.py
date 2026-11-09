"""Illustrate raster Voronoi regions on the saved Access-based Figure 2.3 panel.

Reuse the saved landscape, not a regenerated or visually selected realization.
Region assignment and categorical colours match the Live Viewer (open edges,
four-neighbour patches, no erosion). Boundary segments remain vector graphics.
"""
from __future__ import annotations

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
from matplotlib.patches import Patch, FancyArrowPatch
import numpy as np

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.boundary import distance_transform_edt_with_boundary
from src.landscape.landscape_stats import label_forest_clusters
from src.tools.final_plot_style import (
    CLUSTER_PALETTE_RGB, VORONOI_BACKGROUND_RGB, VORONOI_BOUNDARY_RGB,
    VORONOI_BOUNDARY_ALPHA, VORONOI_REGION_COLOR_WEIGHT,
    apply_final_style, export_figure, figure_size,
)

SOURCE_REL = Path("01_landscape/01_01_analysis_framework/01_01_01_landscape_examples")
OUT_REL = Path("01_landscape/01_01_analysis_framework/01_01_02_landscape_metrics")
SOURCE_STEM = "edge_access_landscapes_rho099_h030"
STEM = "access_voronoi_rho099_h030"
SCENARIO = "access_weighted_rho099"


def region_pairs(regions: np.ndarray) -> np.ndarray:
    pairs = []
    for a, b in ((regions[:, :-1], regions[:, 1:]), (regions[:-1], regions[1:])):
        mask = a != b
        pairs.append(np.sort(np.column_stack((a[mask], b[mask])), axis=1))
    return np.unique(np.vstack(pairs), axis=0)


def cluster_colors(n_clusters: int, pairs: np.ndarray) -> np.ndarray:
    """Use the Live Viewer's degree-ordered greedy categorical colouring."""
    neighbors = [set() for _ in range(n_clusters + 1)]
    for a, b in pairs:
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))
    indices = np.full(n_clusters + 1, -1, dtype=int)
    palette = np.asarray(CLUSTER_PALETTE_RGB, dtype=np.uint8)
    for label in sorted(range(1, n_clusters + 1), key=lambda i: (-len(neighbors[i]), i)):
        unavailable = {indices[i] for i in neighbors[label] if indices[i] >= 0}
        start = (label - 1) % len(palette)
        indices[label] = next(
            (start + offset) % len(palette) for offset in range(len(palette))
            if (start + offset) % len(palette) not in unavailable
        )
    if np.any(indices[pairs[:, 0]] == indices[pairs[:, 1]]):
        raise RuntimeError("Adjacent Voronoi regions have the same colour.")
    table = np.zeros((n_clusters + 1, 3), dtype=np.uint8)
    table[1:] = palette[indices[1:]]
    return table


def boundary_segments(regions: np.ndarray) -> np.ndarray:
    """Draw exact cell interfaces, including the raster's stepped boundaries."""
    row, col = np.nonzero(regions[:, :-1] != regions[:, 1:])
    vertical = np.stack((np.column_stack((col + .5, row - .5)),
                         np.column_stack((col + .5, row + .5))), axis=1)
    row, col = np.nonzero(regions[:-1] != regions[1:])
    horizontal = np.stack((np.column_stack((col - .5, row + .5)),
                           np.column_stack((col + .5, row + .5))), axis=1)
    return np.concatenate((vertical, horizontal))


def main() -> None:
    source = FINAL / "data/processed" / SOURCE_REL / f"{SOURCE_STEM}.npz"
    with np.load(source, allow_pickle=False) as saved:
        matches = np.flatnonzero(saved["scenario_ids"] == SCENARIO)
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one saved Access-based panel.")
        grid = np.asarray(saved["grids"][matches[0]], dtype=np.uint8)
        h, seed = float(saved["H"]), int(saved["seed"])
    provenance = FINAL / "results" / SOURCE_REL / "tables" / f"{SOURCE_STEM}.csv"
    with provenance.open(newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["scenario_id"] == SCENARIO]
    digest = hashlib.sha256(grid.tobytes()).hexdigest()
    if (len(records) != 1 or digest != records[0]["grid_sha256"]
            or grid.shape != (200, 200) or int(grid.sum()) != 12000
            or h != .3 or seed != int(records[0]["seed"])):
        raise RuntimeError("Saved landscape differs from the source figure provenance.")

    labels, n_clusters = label_forest_clusters(grid, connectivity=4, boundary_mode="open")
    _, indices = distance_transform_edt_with_boundary(
        labels == 0, return_indices=True, boundary_mode="open",
    )
    regions = labels[tuple(indices)]
    if np.any(regions <= 0) or not np.array_equal(regions[grid == 1], labels[grid == 1]):
        raise RuntimeError("Invalid nearest-patch assignment.")
    pairs = region_pairs(regions)
    colors = cluster_colors(n_clusters, pairs)
    background = np.asarray(VORONOI_BACKGROUND_RGB)
    rgb = np.rint(VORONOI_REGION_COLOR_WEIGHT * colors[regions]
                  + (1 - VORONOI_REGION_COLOR_WEIGHT) * background).astype(np.uint8)
    rgb[grid == 1] = colors[labels[grid == 1]]

    apply_final_style()
    fig = plt.figure(figsize=figure_size(height_mm=86))
    # Same categorical colour for a patch and its entire region; no patch
    # silhouette remains in the second panel.
    patch_rgb = np.full((*grid.shape, 3), 238, dtype=np.uint8)
    patch_rgb[grid == 1] = colors[labels[grid == 1]]
    region_rgb = colors[regions]
    boundary_color = np.asarray(VORONOI_BOUNDARY_RGB) / 255
    segments = boundary_segments(regions)
    for left, raster, title in ((.01, patch_rgb, "(a) Forest patches"),
                                (.55, region_rgb, "(b) Voronoi regions")):
        ax = fig.add_axes([left, .15, .44, 160 * .44 / 86])
        ax.imshow(raster, interpolation="nearest", origin="upper", aspect="equal")
        if left > .5:
            ax.add_collection(LineCollection(segments, colors=[boundary_color],
                                             linewidths=.30, alpha=.85))
        ax.set_xlim(-.5, grid.shape[1] - .5)
        ax.set_ylim(grid.shape[0] - .5, -.5)
        ax.set_axis_off()
        ax.text(0, 1.025, title, transform=ax.transAxes, va="bottom")
    fig.add_artist(FancyArrowPatch((.475, .56), (.525, .56),
                    transform=fig.transFigure, arrowstyle="-|>",
                    mutation_scale=10, color="#404040", linewidth=1))
    fig.text(.5, .035, "Same colours identify patches and their nearest-patch regions",
             ha="center", fontsize=8.5)
    stem = FINAL / "results" / OUT_REL / "figures" / STEM
    export_figure(fig, stem)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    data = FINAL / "data/processed" / OUT_REL
    data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data / f"{STEM}.npz", grid=grid, labels=labels, regions=regions,
                        neighbor_pairs=pairs, cluster_colors=colors)
    manifest = {"source_panel": "b", "scenario_id": SCENARIO, "rho": .99,
                "H": h, "seed": seed, "grid_sha256": digest,
                "source_npz": str(source.relative_to(FINAL)),
                "source_npz_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "n_clusters": n_clusters, "n_neighbor_pairs": len(pairs),
                "connectivity": 4, "boundary_mode": "open", "edge_depth": 0,
                "region_assignment": "scipy.ndimage.distance_transform_edt(return_indices=True)",
                "colour_reference": "applications/defragmentation_live.py",
                "selection_rule": "Reuse the Access-based panel of the existing source figure.",
                "script": str(Path(__file__).relative_to(FINAL))}
    log = FINAL / "results" / OUT_REL / "logs"
    log.mkdir(parents=True, exist_ok=True)
    (log / f"{STEM}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified source grid {digest}; {n_clusters} patches, {len(pairs)} neighbour pairs.")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
