"""Show actual low-H Result A raster crops with verified NN links.

Exploratory visual diagnostic. Uses matched replicate 141, H=0.025, and
40x40 crops chosen for legibility. All grids match archived Result A hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.generation import run_sequential_deforestation  # noqa: E402
from src.landscape.landscape import Landscape  # noqa: E402
from src.landscape.landscape_stats import nearest_neighbor_distances_from_voronoi_graph  # noqa: E402
from src.tools.final_plot_style import LANDSCAPE_COLORS, apply_final_style, export_figure, figure_size  # noqa: E402

EXAMPLES = (
    ("only_random_rho0", "Only-Random", (70, 10)),
    ("edge_random_rho0999", r"Edge-Random, $\rho=0.999$", (120, 90)),
    ("access_weighted_rho0999", r"Access-based, $\rho=0.999$", (0, 50)),
)
REPLICATE = 141
H = 0.025
CROP_SIZE = 40
STEM = FINAL / "results" / REL / "figures/07_low_h_nn_zoom_examples"
TABLE = FINAL / "results" / REL / "tables/07_low_h_nn_zoom_examples.csv"


def links_inside_crop(labels: np.ndarray, graph, crop: tuple[int, int], max_links: int = 6):
    y0, x0 = crop
    y1, x1 = y0 + CROP_SIZE, x0 + CROP_SIZE
    full_nearest = nearest_neighbor_distances_from_voronoi_graph(graph)
    visible = set(np.unique(labels[y0:y1, x0:x1])) - {0}
    patches = {}
    candidates = {}
    csr = graph.tocsr()
    for label_id in sorted(visible):
        i = int(label_id) - 1
        start, stop = csr.indptr[i], csr.indptr[i + 1]
        neighbours = csr.indices[start:stop]
        values = csr.data[start:stop]
        good = neighbours != i
        neighbours, values = neighbours[good], values[good]
        if not len(values):
            continue
        j = int(neighbours[np.argmin(values)])
        partner_id = j + 1
        if partner_id not in visible:
            continue
        key = tuple(sorted((int(label_id), int(partner_id))))
        if key in candidates:
            continue
        for member in key:
            if member not in patches:
                patches[member] = np.argwhere(labels == member)
        a = patches[key[0]]
        b = patches[key[1]]
        d, k = cKDTree(b).query(a, k=1)
        ix = int(np.argmin(d))
        ay, ax = map(int, a[ix])
        by, bx = map(int, b[int(k[ix])])
        assert np.isclose(float(d[ix]), float(values.min()), atol=1e-9)
        if y0 <= ay < y1 and x0 <= ax < x1 and y0 <= by < y1 and x0 <= bx < x1:
            candidates[key] = ((ay, ax), (by, bx), float(d[ix]))
    links = list(candidates.values())
    if len(links) <= max_links:
        return links, len(candidates), full_nearest

    # Select links spread across the crop, starting nearest its centre.
    centres = np.asarray([
        ((a[0] + b[0]) / 2 - y0, (a[1] + b[1]) / 2 - x0)
        for a, b, _ in links
    ], dtype=float)
    first = int(np.argmin(((centres - CROP_SIZE / 2) ** 2).sum(axis=1)))
    selected = [first]
    while len(selected) < max_links:
        d2 = ((centres[:, None, :] - centres[selected][None, :, :]) ** 2).sum(axis=2).min(axis=1)
        d2[selected] = -1
        selected.append(int(np.argmax(d2)))
    return [links[i] for i in selected], len(candidates), full_nearest


def main() -> None:
    config_path = FINAL / "config" / REL / "07_structural_scenario_responses.json"
    config = json.loads(config_path.read_text())
    assert config["generation_boundary"] == "periodic"
    assert (config["analysis_boundary"], config["cluster_connectivity"], config["distance_mode"], config["voronoi_depth"]) == ("open", "4", "voronoi_dt", 1)
    seed = int(config["base_seed"]) + REPLICATE - 1
    scenarios = {s["scenario_id"]: s for s in config["scenarios"]}
    stored = pd.read_csv(FINAL / "data/processed" / REL / "landscape_metrics.csv")
    apply_final_style()
    fig, axes = plt.subplots(2, 3, figsize=figure_size(125))
    fig.subplots_adjust(left=0.035, right=0.99, top=0.94, bottom=0.115, wspace=0.10, hspace=0.20)
    cmap = ListedColormap([LANDSCAPE_COLORS["non_habitat"], LANDSCAPE_COLORS["habitat"]])
    output = []
    for col, (scenario_id, scenario_label, crop) in enumerate(EXAMPLES):
        scenario = scenarios[scenario_id]
        run = run_sequential_deforestation(
            np.ones((200, 200), dtype=np.int8), np.random.default_rng(seed),
            rho=float(scenario["rho"]), directed_mode=scenario["mode"],
            target_habitat_fraction=H, generation_boundary=config["generation_boundary"],
            eta=config["access_reference"]["eta"], k=config["access_reference"]["k"],
            lambda_decay=config["access_reference"]["lambda"],
            save_habitat_fractions=[H], seed=seed,
        )
        assert len(run.snapshots) == 1
        grid = run.snapshots[0].grid
        archived = stored[
            (stored.scenario_id == scenario_id)
            & (stored.replicate == REPLICATE)
            & (stored.H_target == H)
        ]
        assert len(archived) == 1
        archived = archived.iloc[0]
        grid_hash = hashlib.sha256(grid.tobytes()).hexdigest()
        assert grid_hash == archived.grid_sha256
        landscape = Landscape(grid)
        landscape.ensure_clusters(
            edge_depth=0.0, connectivity="4", boundary_mode="open",
            distance_mode="voronoi_dt", voronoi_depth=1, rebuild=True,
        )
        assert landscape.n_clusters == int(archived.n_patches)
        labels = np.asarray(landscape.core_cluster_labels)
        links, n_eligible, all_nn = links_inside_crop(
            labels, landscape.ensure_voronoi_neighbor_graph(), crop
        )
        assert abs(float(all_nn.mean()) - float(archived.mean_nearest_neighbor_distance)) < 1e-10
        y0, x0 = crop
        ax = axes[0, col]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", origin="upper", aspect="equal")
        ax.add_patch(Rectangle(
            (x0 - .5, y0 - .5), CROP_SIZE, CROP_SIZE,
            fill=False, edgecolor="#C47723", linewidth=1.5,
        ))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.01, 1.025, f"({chr(97 + col)}) {scenario_label}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        detail = axes[1, col]
        detail.imshow(
            grid[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE],
            cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
            origin="upper", aspect="equal",
        )
        for a, b, _ in links:
            ay, axx = a
            by, bxx = b
            detail.plot(
                [axx - x0, bxx - x0], [ay - y0, by - y0],
                color="#377A9F", linewidth=1.4, zorder=3,
            )
            detail.scatter(
                [axx - x0, bxx - x0], [ay - y0, by - y0],
                c="#377A9F", s=8, zorder=4,
            )
        detail.set_xticks([])
        detail.set_yticks([])
        detail.text(
            0.01, 1.025, f"({chr(100 + col)}) Enlarged area",
            transform=detail.transAxes, ha="left", va="bottom", fontsize=9.5,
        )
        detail.text(
            0.5, -0.04, f"Whole-landscape NN = {archived.mean_nearest_neighbor_distance:.2f} cells",
            transform=detail.transAxes, ha="center", va="top", fontsize=8.5,
        )
        for spine in detail.spines.values():
            spine.set_visible(False)
        output.append({
            "scenario_id": scenario_id, "replicate": REPLICATE, "seed": seed,
            "H_target": H, "grid_sha256": grid_hash,
            "crop_top_cell": y0, "crop_left_cell": x0, "crop_side_cells": CROP_SIZE,
            "n_patches_in_full_grid": int(archived.n_patches),
            "mean_nn_in_full_grid": float(all_nn.mean()),
            "n_links_eligible_in_crop": n_eligible,
            "n_links_drawn_in_crop": len(links),
            "forest_cells_in_crop": int(grid[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE].sum()),
        })
    fig.legend(
        handles=[
            Patch(facecolor=LANDSCAPE_COLORS["habitat"], label="Forest"),
            Patch(facecolor=LANDSCAPE_COLORS["non_habitat"], edgecolor="#BDBDBD", label="Non-forest"),
            Patch(facecolor="none", edgecolor="#C47723", label="Enlarged area"),
            Line2D([0], [0], color="#377A9F", lw=1.4, label="Selected exact NN links"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=4, frameon=False,
    )
    export_figure(fig, STEM)
    fig.savefig(STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(TABLE, index=False)
    print(f"Saved {STEM}")
    print(pd.DataFrame(output)[["scenario_id", "n_links_eligible_in_crop", "n_links_drawn_in_crop", "forest_cells_in_crop"]].to_string(index=False))


if __name__ == "__main__":
    main()
