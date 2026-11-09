"""Descriptive grouping of the 15 reconstructed empirical bird responses.

Inputs are the checked Table-1 reconstructions on the common 77-level fit grid
and their species metadata. No simulation or raw empirical data are changed.
The three-group cut is an illustration, not an inferred biological taxonomy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ufz-bird-response-groups-mpl")
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from src.tools.final_plot_style import apply_final_style, export_figure, figure_size


ROOT = FINAL / "data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_reconstruction"
CURVES = ROOT / "11_bird_curves_main_H_grid.csv"
COEFFICIENTS = ROOT / "11_bird_curve_coefficients.csv"
OUT = FINAL / "results/00_thesis_figures/05_bird_response_groups"
FIGURE = OUT / "figures/05_empirical_bird_response_groups"
EXAMPLES = {"mf2021_07", "mf2021_08", "mf2021_09"}
GROUP_LABELS = {
    1: "Low forest-cover threshold",
    2: "Intermediate forest-cover threshold",
    3: "High forest-cover threshold",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def silhouette(distances: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for i, label in enumerate(labels):
        own = np.flatnonzero(labels == label)
        if len(own) == 1:
            values.append(0.0)
            continue
        within = distances[i, own[own != i]].mean()
        between = min(
            distances[i, labels == other].mean()
            for other in set(labels) if other != label
        )
        values.append((between - within) / max(within, between))
    return float(np.mean(values))


def main() -> None:
    meta = {row["species_id"]: row for row in read_csv(COEFFICIENTS)}
    rows = [row for row in read_csv(CURVES) if float(row["H"]) <= 0.925]
    species = sorted(meta)
    assert len(species) == 15 and len(rows) == 15 * 77
    by_species = {species_id: [] for species_id in species}
    for row in rows:
        assert row["species_id"] in by_species
        by_species[row["species_id"]].append(row)
    habitat = np.array([float(row["H"]) for row in by_species[species[0]]])
    assert len(habitat) == 77 and np.all(np.diff(habitat) > 0)
    assert np.isclose(habitat[0], 0.07) and np.isclose(habitat[-1], 0.925)
    curves = np.empty((15, 77), dtype=float)
    for i, species_id in enumerate(species):
        species_rows = by_species[species_id]
        assert len(species_rows) == 77
        assert np.array_equal(habitat, [float(row["H"]) for row in species_rows])
        curves[i] = [float(row["psi_reconstructed"]) for row in species_rows]
    assert np.isfinite(curves).all() and np.all((0 <= curves) & (curves <= 1))

    # Euclidean distance divided by sqrt(77) is whole-curve RMSE. No
    # within-species scaling is applied: absolute occupancy is part of shape.
    condensed = pdist(curves, metric="euclidean") / np.sqrt(len(habitat))
    tree = linkage(condensed, method="average")
    distances = squareform(condensed)
    silhouette_by_k = {}
    for k in (2, 3, 4):
        labels = fcluster(tree, k, criterion="maxclust")
        assert len(set(labels)) == k
        silhouette_by_k[str(k)] = silhouette(distances, labels)
    raw_labels = fcluster(tree, 3, criterion="maxclust")
    raw_groups = sorted(
        set(raw_labels),
        key=lambda group: np.median([
            float(meta[species[i]]["reconstructed_H_psi060"])
            for i in np.flatnonzero(raw_labels == group)
        ]),
    )
    ordered = {raw: number for number, raw in enumerate(raw_groups, start=1)}
    group_of = {species[i]: ordered[raw_labels[i]] for i in range(15)}
    assert Counter(group_of.values()) == {1: 2, 2: 7, 3: 6}
    assert {group_of[item] for item in EXAMPLES} == {1, 2, 3}

    OUT.mkdir(parents=True, exist_ok=True)
    assignment = OUT / "tables/05_bird_response_group_assignments.csv"
    assignment.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "group_label", "species_id", "species", "trophic_guild",
              "reconstructed_H_psi060", "main_text_example"]
    with assignment.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for species_id in sorted(species, key=lambda item: (
            group_of[item], float(meta[item]["reconstructed_H_psi060"])
        )):
            info = meta[species_id]
            writer.writerow({
                "group": group_of[species_id],
                "group_label": GROUP_LABELS[group_of[species_id]],
                "species_id": species_id,
                "species": info["species_as_published"],
                "trophic_guild": info["trophic_guild"],
                "reconstructed_H_psi060": info["reconstructed_H_psi060"],
                "main_text_example": species_id in EXAMPLES,
            })

    apply_final_style()
    fig, axes = plt.subplots(3, 1, figsize=figure_size(168), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.57, top=0.93, bottom=0.14, hspace=0.38)
    for group, ax in enumerate(axes, start=1):
        member_ids = sorted(
            (item for item in species if group_of[item] == group),
            key=lambda item: float(meta[item]["reconstructed_H_psi060"]),
        )
        for species_id in member_ids:
            if species_id in EXAMPLES:
                continue
            ax.plot(habitat, curves[species.index(species_id)], color="#A6A6A6",
                    linewidth=1.25, alpha=0.9)
        for species_id in member_ids:
            if species_id in EXAMPLES:
                ax.plot(habitat, curves[species.index(species_id)], color="#377A9F",
                        linewidth=2.0)
        ax.set_xlim(1, 0)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 0.8, 0.6, 0.4, 0.2, 0])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.4)
        ax.tick_params(axis="x", labelbottom=True)
        ax.text(0, 1.08, f"({chr(96 + group)}) {GROUP_LABELS[group]} (n={len(member_ids)})",
                transform=ax.transAxes, fontsize=9.5)
        x = 0.61
        y0 = ax.get_position().y1 - 0.009
        step = min(0.027, (ax.get_position().height - 0.018) / len(member_ids))
        for position, species_id in enumerate(member_ids):
            selected = species_id in EXAMPLES
            fig.text(x, y0 - position * step, meta[species_id]["species_as_published"],
                     va="top", fontsize=8.5, fontstyle="italic",
                     color="#377A9F" if selected else "#404040",
                     fontweight="bold" if selected else "normal")
    fig.supxlabel(r"Forest-cover proportion $H$")
    fig.supylabel("Reconstructed forest-site occupancy")
    fig.legend(handles=[
        Line2D([], [], color="#377A9F", linewidth=2.0, label="Main-text example"),
        Line2D([], [], color="#A6A6A6", linewidth=1.25, label="Other species in group"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    export_figure(fig, FIGURE)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    audit = {
        "analysis": "Descriptive average-linkage grouping of reconstructed empirical occupancy curves",
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (CURVES, COEFFICIENTS)},
        "comparison_levels": len(habitat),
        "habitat_domain": [float(habitat[0]), float(habitat[-1])],
        "distance": "Euclidean whole-curve RMSE on unscaled occupancy probabilities",
        "linkage": "average",
        "chosen_group_count": 3,
        "silhouette_by_k": silhouette_by_k,
        "group_sizes": dict(sorted(Counter(group_of.values()).items())),
        "guild_counts": {str(group): dict(sorted(Counter(
            meta[item]["trophic_guild"] for item in species if group_of[item] == group
        ).items())) for group in (1, 2, 3)},
        "interpretation": "Exploratory response-shape partition, not biological classes",
    }
    (OUT / "05_grouping_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
