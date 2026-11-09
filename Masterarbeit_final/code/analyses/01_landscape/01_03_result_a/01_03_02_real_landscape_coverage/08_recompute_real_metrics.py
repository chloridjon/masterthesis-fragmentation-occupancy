"""Recalculate final structural metrics from the selected JRC binary windows.

The Alder metric CSV predates the frozen thesis definitions. Its selected
window manifest and original NPZ rasters are read-only inputs; no raw file is
modified. Run from the repository root with the existing .venv interpreter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_02_real_landscape_coverage")
CONFIG = FINAL / "config" / REL / "08_real_landscape_coverage.json"
DEST = FINAL / "data/processed" / REL
RESULTS = FINAL / "results" / REL
sys.path.insert(0, str(FINAL / "code/base"))

ACCESS_PATH = FINAL / "code/analyses/01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection/02_access_parameter_selection.py"
spec = importlib.util.spec_from_file_location("access_metrics_for_real_result_a", ACCESS_PATH)
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, help="Directory holding selected_landscapes.csv and selected_landscapes/.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_config() -> dict:
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (c["analysis_plan_section"] != "1.3.2" or c["grid_size_cells"] != 200 or
        c["window_extents_m"] != [2000, 4000, 8000, 16000, 32000] or
        c["n_h_bins"] != 20 or c["h_bin_width"] != .05 or
        c["windows_per_stratum"] != 100 or
        (c["analysis_boundary"], c["cluster_connectivity"], c["distance_mode"], c["voronoi_depth"]) !=
        ("open", "4", "voronoi_dt", 1) or tuple(c["metrics"]) != access.METRICS):
        raise ValueError("Configuration differs from the frozen real-window design or metric definitions.")
    return c


def selected_rows(root: Path, c: dict) -> tuple[pd.DataFrame, str]:
    path = root / c["selection_manifest"]
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    selected = pd.read_csv(path)
    expected = len(c["window_extents_m"]) * c["n_h_bins"] * c["windows_per_stratum"]
    if len(selected) != expected or selected["candidate_id"].duplicated().any():
        raise ValueError("Real-window selection has missing or duplicate candidates.")
    counts = selected.groupby(["extent_m", "h_bin_index"]).size()
    combinations = pd.MultiIndex.from_product([c["window_extents_m"], range(c["n_h_bins"])])
    if not counts.reindex(combinations).eq(c["windows_per_stratum"]).all():
        raise ValueError("Real-window sample is not balanced across all 100 strata.")
    if not selected["export_status"].eq("written").all() or not selected["passes_land_filter"].all():
        raise ValueError("A selected window was not exported or failed the land filter.")
    if selected["land_fraction"].min() < .9:
        raise ValueError("The selected sample contains a window with under 90% land.")
    h = selected["forest_cover"].to_numpy(float)
    if not (np.isfinite(h).all() and (h > 0).all() and (h <= 1).all()):
        raise ValueError("Selected habitat proportions lie outside (0,1].")
    bins = np.minimum(np.rint(h * 40000).astype(int) // 2000, 19)
    if not np.array_equal(bins, selected["h_bin_index"].to_numpy(int)):
        raise ValueError("The selected H-bin labels differ from exact 200x200 cell counts.")
    return selected, source_hash


def one_window(job: tuple[dict, str, dict]) -> dict:
    row, root_string, c = job
    identifier = row["candidate_id"]
    extent = int(row["extent_m"])
    h_bin = int(row["h_bin_index"])
    if not re.fullmatch(r"center_\d+__\d{2}km", identifier):
        raise ValueError(f"Unexpected window identifier: {identifier}")
    relative = Path("selected_landscapes") / f"{extent // 1000:02}km" / f"hbin_{h_bin:02d}" / f"{identifier}.npz"
    with np.load(Path(root_string) / relative, allow_pickle=False) as archive:
        grid = archive["grid"]
        source = json.loads(str(archive["metadata_json"]))
    if (grid.shape != (200, 200) or not np.isin(grid, [0, 1]).all() or
        source["candidate_id"] != identifier or int(source["extent_m"]) != extent or
        int(source["h_bin_index"]) != h_bin or source["majority_threshold"] != .5):
        raise ValueError(f"Invalid binary window or metadata: {relative}")
    h = float(grid.mean())
    if abs(h - float(row["forest_cover"])) > 1e-12:
        raise ValueError(f"Manifest and binary-window H differ: {relative}")
    metrics = access.snapshot_metrics(grid, c)
    return {"candidate_id": identifier, "extent_m": extent, "h_bin_index": h_bin,
            "H": h, "land_fraction": float(row["land_fraction"]),
            "lon": float(row["lon"]), "lat": float(row["lat"]),
            "source_relative_path": relative.as_posix(),
            "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(), **metrics}


def source_fingerprint(c: dict, selection_sha256: str) -> str:
    digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode())
    digest.update(selection_sha256.encode())
    for path in (Path(__file__), ACCESS_PATH, *sorted((FINAL / "code/base/src/landscape").glob("*.py"))):
        digest.update(str(path.relative_to(FINAL)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def legacy_discrepancies(new: pd.DataFrame, root: Path, c: dict) -> pd.DataFrame:
    old_path = root / "metrics/real_landscape_metrics.csv"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    mapped = {"edge_density_total": "edge_density_total", "largest_cluster_fraction_forest": "LCF",
              "chi": "Chi", "mean_nearest_neighbor_distance": "dist_nn_mean",
              "mean_pairwise_distance": "dist_pair_mean", "complete_mst_d50": "complete_mst_d50"}
    joined = new.merge(old[["candidate_id", *mapped.values()]], on="candidate_id", validate="one_to_one", suffixes=("_new", "_old"))
    if len(joined) != len(new):
        raise ValueError("The pre-existing metric table lacks selected windows.")
    rows = []
    for metric, old_column in mapped.items():
        a = joined[f"{metric}_new"].to_numpy(float) if metric == old_column else joined[metric].to_numpy(float)
        b = joined[f"{old_column}_old"].to_numpy(float) if metric == old_column else joined[old_column].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        difference = np.abs(a[both] - b[both])
        rows.append({"metric": metric, "n_compared": int(both.sum()),
                     "n_missingness_disagrees": int((np.isfinite(a) != np.isfinite(b)).sum()),
                     "n_values_differ_over_1e_8": int((difference > 1e-8).sum()),
                     "maximum_absolute_difference": float(difference.max()) if len(difference) else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    args = arguments()
    c = load_config()
    root = (args.real_root or Path(c["default_real_root"])).resolve()
    selected, selection_sha256 = selected_rows(root, c)
    jobs = ((row, str(root), c) for row in selected.to_dict("records"))
    DEST.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for index, result in enumerate(pool.map(one_window, jobs, chunksize=20), 1):
            rows.append(result)
            if index % 1000 == 0:
                print(f"Recomputed {index}/{len(selected)} real windows", flush=True)
    metrics = pd.DataFrame(rows)
    if (len(metrics) != len(selected) or metrics["candidate_id"].duplicated().any() or
        not metrics["H"].between(0, 1).all()):
        raise AssertionError("Incomplete recalculated real-window metrics.")
    destination = DEST / "08_real_landscape_metrics.csv"
    temporary = destination.with_suffix(".tmp")
    metrics.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    audit = legacy_discrepancies(metrics, root, c)
    if not audit.empty:
        (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
        audit.to_csv(RESULTS / "tables/08_previous_metric_definition_audit.csv", index=False)
    manifest = {"real_root": str(root), "selection_manifest_sha256": selection_sha256,
                "metric_producer_fingerprint": source_fingerprint(c, selection_sha256),
                "n_windows": len(metrics), "window_extents_m": c["window_extents_m"],
                "n_h_bins": c["n_h_bins"], "windows_per_stratum": c["windows_per_stratum"],
                "source_metric_table_used_for_analysis": False,
                "distance_units": "cells of each 200x200 window"}
    (DEST / "08_real_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
