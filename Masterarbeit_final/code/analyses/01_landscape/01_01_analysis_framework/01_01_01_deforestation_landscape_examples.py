"""Illustrate fixed deforestation scenarios at H=0.3 for thesis 2.1.2.

Reconstruct the first matched Result A seed block and require exact agreement
with its saved grid hashes. This script computes no landscape statistics.
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
sys.path.insert(0, str(FINAL / "code/base"))
from src.landscape.generation import run_sequential_deforestation
from src.tools.final_plot_style import (
    LANDSCAPE_COLORS, apply_final_style, export_figure, figure_size,
)

REL = Path("01_landscape/01_01_analysis_framework/01_01_01_landscape_examples")


def plot_landscapes(grids, labels, figure_dir: Path, stem: str) -> None:
    """Embed native grid pixels in vector exports, without DPI resampling."""
    apply_final_style()
    n_rows = len(grids) // 2
    height_mm = 90 if n_rows == 1 else 174
    fig, axes = plt.subplots(n_rows, 2, figsize=figure_size(height_mm, 160))
    cmap = ListedColormap([LANDSCAPE_COLORS["non_habitat"], LANDSCAPE_COLORS["habitat"]])
    for i, (ax, grid, label) in enumerate(zip(axes.flat, grids, labels)):
        # PDF/SVG retain the original 200 x 200 samples. Nearest-neighbour
        # resampling at figure DPI made cells periodically unequal in width.
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="none",
                  resample=False, aspect="equal")
        ax.set_axis_off()
        ax.text(0, 1.025, f"({chr(97+i)}) {label}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.92 if n_rows == 1 else 0.955,
                        bottom=0.14 if n_rows == 1 else 0.075,
                        wspace=0.12, hspace=0.19)
    fig.legend(handles=[Patch(facecolor=LANDSCAPE_COLORS[k], label=label)
                        for k, label in [("habitat", "Habitat"), ("non_habitat", "Non-habitat")]],
               loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    export_figure(fig, figure_dir / stem)
    fig.savefig(figure_dir / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_saved(args, config) -> None:
    """Rerender verified saved landscapes without changing trajectories."""
    stem = config["output_stem"]
    manifest_path = args.results_dir / "logs" / f"{stem}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with np.load(args.data_dir / f"{stem}.npz", allow_pickle=False) as saved:
        grids = saved["grids"]
        scenario_ids = saved["scenario_ids"].tolist()
        if scenario_ids != config["scenario_ids"]:
            raise RuntimeError("Saved scenario order differs from the figure configuration.")
        if float(saved["H"]) != config["habitat_proportion"]:
            raise RuntimeError("Saved habitat proportion differs from the figure configuration.")
        seed = int(saved["seed"])
    records = manifest["records"]
    if len(records) != len(grids):
        raise RuntimeError("Saved panels and provenance have different lengths.")
    labels = []
    for grid, scenario_id, record in zip(grids, scenario_ids, records):
        digest = hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest()
        if (digest != record["grid_sha256"] or scenario_id != record["scenario_id"]
                or grid.shape != (200, 200) or int(grid.sum()) != 12000
                or seed != record["seed"]):
            raise RuntimeError("Saved panel differs from its verified provenance.")
        name = "Only-Random" if scenario_id.startswith("only_random") else (
            "Edge-Random" if scenario_id.startswith("edge_random") else "Access-based")
        labels.append(name if name == "Only-Random" else fr"{name} ($\rho={record['rho']:g}$)")
    figure_dir = args.results_dir / "figures"
    plot_landscapes(grids, labels, figure_dir, stem)
    manifest.update({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "render_only": True,
        "vector_export_raster_shape": [200, 200],
        "raster_interpolation": "none (native pixels in PDF/SVG)",
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in figure_dir.glob(stem + ".*")},
    })
    manifest.pop("figure_review", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rerendered {len(grids)} unchanged, hash-verified panels: {stem}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FINAL / "config" / REL / "deforestation_landscapes_h030.json")
    parser.add_argument("--results-dir", type=Path, default=FINAL / "results" / REL)
    parser.add_argument("--data-dir", type=Path, default=FINAL / "data/processed" / REL)
    parser.add_argument("--render-only", action="store_true",
                        help="Rerender saved, hash-verified grids without generating new trajectories.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.render_only:
        render_saved(args, config)
        return
    source_config_path = FINAL / config["source_config"]
    source = json.loads(source_config_path.read_text())
    scenarios = {s["scenario_id"]: s for s in source["scenarios"]}
    h = float(config["habitat_proportion"])
    replicate = int(config["replicate"])
    seed = int(source["base_seed"]) + replicate - 1
    size = int(source["grid_size_cells"])
    assert h == 0.3 and size == 200 and replicate == 1
    assert len(config["scenario_ids"]) in (2, 4)
    grids, records, labels = [], [], []

    for scenario_id in config["scenario_ids"]:
        scenario = scenarios[scenario_id]
        source_csv = FINAL / config["source_trajectories"] / f"{scenario_id}__rep{replicate:03d}.csv"
        with source_csv.open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if float(r["H_target"]) == h]
        assert len(rows) == 1 and int(rows[0]["seed"]) == seed
        run = run_sequential_deforestation(
            np.ones((size, size), dtype=np.int8), np.random.default_rng(seed),
            rho=float(scenario["rho"]), directed_mode=scenario["mode"],
            target_habitat_fraction=h,
            generation_boundary=source["generation_boundary"],
            **{"eta": source["access_reference"]["eta"],
               "k": source["access_reference"]["k"],
               "lambda_decay": source["access_reference"]["lambda"]},
            save_habitat_fractions=[h], seed=seed,
        )
        grid = np.asarray(run.grid, dtype=np.uint8)
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        assert digest == rows[0]["grid_sha256"], f"Reconstruction differs from Result A: {scenario_id}"
        assert grid.shape == (200, 200) and set(np.unique(grid)) == {0, 1}
        assert int(grid.sum()) == 12000 and run.final_habitat_fraction == h
        assert run.n_events == 28000 and run.n_random_forced_initial == 1
        assert run.n_random_total + run.n_directed == run.n_events
        grids.append(grid)
        name = {"only_random": "Only-Random", "edge_random": "Edge-Random",
                "access_weighted": "Access-based"}[scenario["method"]]
        labels.append(name if scenario["method"] == "only_random" else fr"{name} ($\rho={scenario['rho']:g}$)")
        records.append({
            "panel": chr(97 + len(records)), "scenario_id": scenario_id,
            "rho": scenario["rho"], "seed": seed, "replicate": replicate,
            "H": h, "habitat_cells": int(grid.sum()), "grid_sha256": digest,
            "source_trajectory": str(source_csv.relative_to(FINAL)),
            "source_trajectory_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
            "generator_version": run.generator_version,
            "random_removals": run.n_random_total, "directed_removals": run.n_directed,
        })
        print(f"Verified {scenario_id}: H={h:g}, seed={seed}, saved grid hash matches.", flush=True)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    stem = config["output_stem"]
    np.savez_compressed(args.data_dir / f"{stem}.npz", grids=np.stack(grids),
                        scenario_ids=np.asarray(config["scenario_ids"]), H=h, seed=seed)
    table_dir = args.results_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    figure_dir = args.results_dir / "figures"
    plot_landscapes(grids, labels, figure_dir, stem)

    log_dir = args.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "source_config": str(source_config_path.relative_to(FINAL)),
                "source_config_sha256": hashlib.sha256(source_config_path.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "plot_style_sha256": hashlib.sha256((FINAL / "code/base/src/tools/final_plot_style.py").read_bytes()).hexdigest(),
                "grid_hashes_match_result_a": True, "records": records,
                "colours": LANDSCAPE_COLORS,
                "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in figure_dir.glob(stem + ".*")}}
    (log_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
