"""Constructed geometry illustration for mean nearest-neighbour distance.

Both 20 x 20 grids have ten isolated single-cell patches and H=0.025.
They are conceptual examples, not outputs of a deforestation scenario.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.landscape import Landscape  # noqa: E402
from src.landscape.landscape_stats import nearest_neighbor_distances_from_voronoi_graph  # noqa: E402
from src.tools.final_plot_style import LANDSCAPE_COLORS, apply_final_style, export_figure, figure_size  # noqa: E402

STEM = FINAL / "results/01_landscape/01_03_result_a/01_03_01_structural_scenario_responses/figures/07_nn_explanatory_schematic"
POSITIONS = (
    [(r, c) for r in (5, 15) for c in (2, 6, 10, 14, 18)],
    [(3, 3), (3, 5), (5, 3), (5, 5), (7, 4),
     (12, 12), (12, 14), (14, 12), (14, 14), (16, 13)],
)


def main() -> None:
    apply_final_style()
    cmap = ListedColormap([LANDSCAPE_COLORS["non_habitat"], LANDSCAPE_COLORS["habitat"]])
    fig, axes = plt.subplots(1, 2, figsize=figure_size(83))
    fig.subplots_adjust(left=.045, right=.975, top=.82, bottom=.17, wspace=.19)
    names = ("Dispersed single-cell patches", "Patches in two distant groups")
    for i, (ax, positions) in enumerate(zip(axes, POSITIONS)):
        grid = np.zeros((20, 20), dtype=np.int8)
        for y, x in positions:
            grid[y, x] = 1
        landscape = Landscape(grid)
        landscape.ensure_clusters(edge_depth=0.0, connectivity="4", boundary_mode="open",
                                  distance_mode="voronoi_dt", voronoi_depth=1, rebuild=True)
        assert landscape.n_clusters == 10
        graph = landscape.ensure_voronoi_neighbor_graph().tocsr()
        values = nearest_neighbor_distances_from_voronoi_graph(graph)
        expected = 4.0 if i == 0 else 2.047213595499958
        assert np.isclose(values.mean(), expected, atol=1e-10)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", origin="upper")
        drawn = set()
        for patch_id in range(10):
            start, stop = graph.indptr[patch_id], graph.indptr[patch_id + 1]
            js, ds = graph.indices[start:stop], graph.data[start:stop]
            valid = js != patch_id
            js, ds = js[valid], ds[valid]
            target = int(js[np.argmin(ds)])
            pair = tuple(sorted((patch_id, target)))
            if pair in drawn:
                continue
            drawn.add(pair)
            ay, axx = positions[patch_id]
            by, bxx = positions[target]
            ax.plot([axx, bxx], [ay, by], color="#377A9F", linewidth=1.4, zorder=2)
            ax.scatter([axx, bxx], [ay, by], c="#377A9F", s=7, zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({chr(97+i)}) {names[i]}", loc="left", fontsize=9.5, pad=8)
        ax.text(.5, -.07, rf"$H=0.025$; 10 patches; mean NN = {values.mean():.2f} cells",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.5, .04, "Blue links join each patch to a nearest patch.", ha="center", fontsize=8.5)
    export_figure(fig, STEM)
    fig.savefig(STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {STEM}")


if __name__ == "__main__":
    main()
