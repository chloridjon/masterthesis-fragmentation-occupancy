"""Analysis 1.2.2: structural metric redundancy within fixed habitat levels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys

FINAL = next(parent for parent in Path(__file__).resolve().parents if parent.name == "Masterarbeit_final")
ROOT = FINAL.parent
REL = Path("01_landscape/01_02_preliminary_analyses/01_02_02_metric_redundancy")
EXTENDED = Path("n30")
LEGACY_SEED_BLOCKS = 10
ACCESS_REL = Path("01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection")
sys.path.insert(0, str(FINAL / "code/base"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.landscape.generation import run_sequential_deforestation
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size, MISSING

ACCESS_MODULE_PATH = FINAL / "code/analyses" / ACCESS_REL / "02_access_parameter_selection.py"
spec = importlib.util.spec_from_file_location("access_source_for_redundancy", ACCESS_MODULE_PATH)
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)
METRIC_LABELS = ["Edge density", "Largest cluster fraction", r"Susceptibility $\chi$",
                 "Nearest-neighbour distance", "Mean inter-patch distance", r"Complete-MST $D_{50}$"]


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FINAL / "config" / REL / "03_metric_redundancy.json")
    parser.add_argument("--processed-dir", type=Path, default=FINAL / "data/processed" / REL / EXTENDED)
    parser.add_argument("--results-dir", type=Path, default=FINAL / "results" / REL / EXTENDED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["analysis_plan_section"] != "1.2.2" or config["grid_size_cells"] != 200:
        raise ValueError("Analysis identity or grid size differs from the plan.")
    if config["rho_values"] != [0.9, 0.99, 0.999] or config["n_seed_blocks"] != 30:
        raise ValueError("Expected three fixed directed rho values and 30 seed blocks.")
    if config["h_targets"] != [0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.01]:
        raise ValueError("Habitat snapshots differ from the prior analysis.")
    if tuple(config["metrics"]) != access.METRICS:
        raise ValueError("The six metric definitions differ from analysis 1.2.1.")
    if config["generation_boundary"] != "periodic" or config["analysis_boundary"] != "open":
        raise ValueError("Boundary conventions differ from the selected design.")
    if (config["cluster_connectivity"], config["distance_mode"], config["voronoi_depth"]) != ("4", "voronoi_dt", 1):
        raise ValueError("Patch and distance definitions differ from the selected design.")
    if config["access_reference"] != {"eta": 1.0, "k": 3, "lambda": 0.5}:
        raise ValueError("Access reference differs from analysis 1.2.1.")
    b = config["bootstrap"]
    for key in ("initial_draws", "batch_draws", "maximum_draws", "consecutive_stable_batches"):
        if not isinstance(b[key], int) or b[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if b["initial_draws"] > b["maximum_draws"] or b["correlation_tolerance"] < 0:
        raise ValueError("Invalid bootstrap budget or tolerance.")
    return config


def scenarios(config):
    rows = [{"scenario_id": "only_random_rho0", "method": "only_random", "mode": "none", "rho": 0.0}]
    for rho in config["rho_values"]:
        for mode, method in (("edge_random", "edge_random"), ("access_weighted", "access_weighted")):
            rows.append({"scenario_id": f"{method}_rho{str(rho).replace('.', '')}", "method": method,
                         "mode": mode, "rho": float(rho)})
    if len(rows) != 7 or len({r["scenario_id"] for r in rows}) != 7:
        raise AssertionError("Expected seven unique focal scenarios.")
    return rows


def producer_fingerprint(config):
    digest = hashlib.sha256()
    data_keys = {k: v for k, v in config.items() if k not in {"bootstrap", "seed_influence_tolerance", "redundancy_flag_min_median_abs_spearman", "redundancy_flag_min_fraction_strong_h", "diagnostic_plot_min_strong_h_levels"}}
    digest.update(json.dumps(data_keys, sort_keys=True).encode())
    for source in (FINAL / "code/base/src/landscape/generation.py", FINAL / "code/base/src/landscape/deforestation.py",
                   FINAL / "code/base/src/landscape/landscape.py", FINAL / "code/base/src/landscape/landscape_stats.py",
                   FINAL / "code/base/src/landscape/boundary.py", FINAL / "code/base/src/landscape/complete_edge_distances.py",
                   ACCESS_MODULE_PATH):
        digest.update(source.read_bytes())
    digest.update(inspect.getsource(new_trajectory).encode())
    return digest.hexdigest()


def new_trajectory(scenario, replicate, config, fingerprint):
    seed = int(config["base_seed"]) + replicate - 1
    run = run_sequential_deforestation(
        np.ones((200, 200), dtype=np.int8), np.random.default_rng(seed),
        rho=float(scenario["rho"]), directed_mode=scenario["mode"],
        target_habitat_fraction=min(config["h_targets"]), generation_boundary="periodic",
        k=3, lambda_decay=0.5, eta=1.0,
        save_habitat_fractions=config["h_targets"], include_initial_snapshot=False, seed=seed,
    )
    if run.n_random_total + run.n_directed != run.n_events or run.n_random_forced_initial != 1:
        raise AssertionError("Removal mode counts or the first event differ from the model.")
    random_counts = np.r_[0, np.cumsum(np.asarray(run.event_modes) == "random", dtype=np.int64)]
    rows = []
    for snapshot in run.snapshots:
        metric_values = access.snapshot_metrics(snapshot.grid, config)
        count = int(snapshot.event_count)
        rows.append({"scenario_id": scenario["scenario_id"], "method": scenario["method"], "rho": scenario["rho"],
                     "replicate": replicate, "seed": seed,
                     "trajectory_id": f"{scenario['scenario_id']}__rep{replicate:03d}",
                     "H_target": float(snapshot.requested_habitat_fractions[0]),
                     "H_real": float(snapshot.habitat_fraction), "event_count": count,
                     "n_random_cumulative": int(random_counts[count]),
                     "n_directed_cumulative": count - int(random_counts[count]),
                     "grid_sha256": hashlib.sha256(snapshot.grid.tobytes()).hexdigest(),
                     "generator_version": run.generator_version,
                     "producer_fingerprint": fingerprint, **metric_values})
    if len(rows) != 8:
        raise AssertionError("A generated trajectory has missing habitat snapshots.")
    return pd.DataFrame(rows)


def load_access(config):
    previous_config = access.load_config(FINAL / "config" / config["access_source_config"])
    reference = config["access_reference"]
    if previous_config["reference_candidate"] != reference or previous_config["base_seed"] != config["base_seed"]:
        raise RuntimeError("Access reference or seed block differs from the source sample.")
    if previous_config["rho_values"] != config["rho_values"] or previous_config["h_targets"] != config["h_targets"]:
        raise RuntimeError("Access rho or H grid differs from the new analysis.")
    source_path = FINAL / "data/processed" / config["access_source_data"]
    source_manifest = json.loads((source_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    if source_manifest != {"config": previous_config, "producer_fingerprint": access.producer_fingerprint(previous_config)}:
        raise RuntimeError("Access source producer fingerprint no longer matches the current code.")
    raw = pd.read_csv(source_path)
    filtered = raw[raw["eta"].eq(reference["eta"]) & raw["k"].eq(reference["k"]) & raw["lambda"].eq(reference["lambda"])].copy()
    if len(filtered) != 240 or filtered["trajectory_id"].nunique() != 30 or filtered["seed"].nunique() != 10:
        raise RuntimeError("Access reference source is incomplete or duplicated.")
    if not filtered["producer_fingerprint"].eq(source_manifest["producer_fingerprint"]).all():
        raise RuntimeError("Access source rows have another producer fingerprint.")
    filtered["scenario_id"] = filtered["rho"].map(lambda rho: f"access_weighted_rho{str(rho).replace('.', '')}")
    filtered["method"] = "access_weighted"
    return filtered, {"path": str(source_path.relative_to(ROOT)), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      "manifest_sha256": hashlib.sha256((source_path.parent / "run_manifest.json").read_bytes()).hexdigest(),
                      "producer_fingerprint": source_manifest["producer_fingerprint"]}


def main():
    options = args()
    config = load_config(options.config)
    workdir, resultdir = options.processed_dir, options.results_dir
    for folder in (workdir / "trajectories", resultdir / "figures", resultdir / "tables", resultdir / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    all_scenarios = scenarios(config)
    access_rows, access_provenance = load_access(config)
    fingerprint = producer_fingerprint(config)
    legacy_config = {**config, "n_seed_blocks": LEGACY_SEED_BLOCKS}
    legacy_fingerprint = producer_fingerprint(legacy_config)
    legacy_workdir = FINAL / "data/processed" / REL
    legacy_manifest_path = legacy_workdir / "run_manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    summary_only_keys = {"bootstrap", "seed_influence_tolerance", "redundancy_flag_min_median_abs_spearman",
                         "redundancy_flag_min_fraction_strong_h", "diagnostic_plot_min_strong_h_levels"}
    legacy_settings = {k: v for k, v in legacy_config.items() if k not in summary_only_keys}
    if legacy_manifest != {"producer_settings": legacy_settings,
                           "producer_fingerprint": legacy_fingerprint,
                           "access_source": access_provenance}:
        raise RuntimeError("The ten-block source does not match the verified producer and Access source.")
    legacy_provenance = {"path": str(legacy_workdir.relative_to(ROOT)),
                         "manifest_sha256": hashlib.sha256(legacy_manifest_path.read_bytes()).hexdigest(),
                         "producer_fingerprint": legacy_fingerprint,
                         "reused_non_access_trajectories": 4 * LEGACY_SEED_BLOCKS}
    manifest_path = workdir / "run_manifest.json"
    producer_settings = {k: v for k, v in config.items() if k not in summary_only_keys}
    manifest = {"producer_settings": producer_settings, "producer_fingerprint": fingerprint,
                "access_source": access_provenance, "legacy_source": legacy_provenance}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing focal trajectories or the Access source use another producer version.")
    else:
        if any((workdir / "trajectories").glob("*.csv")):
            raise RuntimeError("Trajectory files exist without a matching manifest.")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(all_scenarios).to_csv(workdir / "scenarios.csv", index=False)
    if not options.summarize_only:
        jobs = []
        for scenario in all_scenarios:
            for replicate in range(1, config["n_seed_blocks"] + 1):
                if scenario["method"] == "access_weighted" and replicate <= LEGACY_SEED_BLOCKS:
                    continue
                path = workdir / "trajectories" / f"{scenario['scenario_id']}__rep{replicate:03d}.csv"
                expected_fingerprint = legacy_fingerprint if replicate <= LEGACY_SEED_BLOCKS else fingerprint
                if not path.exists() and replicate <= LEGACY_SEED_BLOCKS:
                    source = legacy_workdir / "trajectories" / path.name
                    if not source.exists():
                        raise RuntimeError(f"Missing validated ten-block trajectory: {source}")
                    source_frame = pd.read_csv(source)
                    if len(source_frame) != 8 or not source_frame["producer_fingerprint"].eq(legacy_fingerprint).all():
                        raise RuntimeError(f"Legacy trajectory has another producer: {source}")
                    shutil.copyfile(source, path)
                if path.exists():
                    frame = pd.read_csv(path)
                    if len(frame) != 8 or not frame["producer_fingerprint"].eq(expected_fingerprint).all():
                        raise RuntimeError(f"Cached trajectory does not match current producer: {path}")
                else:
                    jobs.append((scenario, replicate, path))
        with ProcessPoolExecutor(max_workers=options.workers) as pool:
            futures = {pool.submit(new_trajectory, scenario, replicate, config, fingerprint): path
                       for scenario, replicate, path in jobs}
            for i, future in enumerate(as_completed(futures), start=1):
                path = futures[future]
                frame = future.result()
                tmp = path.with_suffix(".csv.tmp")
                frame.to_csv(tmp, index=False)
                tmp.replace(path)
                print(f"Generated {i}/{len(jobs)}: {path.name}", flush=True)
    generated = []
    for scenario in all_scenarios:
        for replicate in range(1, config["n_seed_blocks"] + 1):
            if scenario["method"] == "access_weighted" and replicate <= LEGACY_SEED_BLOCKS:
                continue
            path = workdir / "trajectories" / f"{scenario['scenario_id']}__rep{replicate:03d}.csv"
            if not path.exists():
                raise RuntimeError(f"Missing trajectory: {path}")
            generated.append(pd.read_csv(path))
    frame = pd.concat([*generated, access_rows], ignore_index=True)
    if len(frame) != 7 * config["n_seed_blocks"] * len(config["h_targets"]) or frame["trajectory_id"].nunique() != 7 * config["n_seed_blocks"]:
        raise AssertionError("The seven-scenario sample is incomplete.")
    if frame.duplicated(["trajectory_id", "H_target"]).any() or frame["seed"].nunique() != config["n_seed_blocks"]:
        raise AssertionError("Scenario/H keys or independent seed blocks are inconsistent.")
    if not np.allclose(frame["H_target"], frame["H_real"], atol=1e-12):
        raise AssertionError("Target and realised habitat fractions differ.")
    if not np.array_equal(frame["n_random_cumulative"].to_numpy(int) + frame["n_directed_cumulative"].to_numpy(int), frame["event_count"].to_numpy(int)):
        raise AssertionError("Removal event counts are inconsistent.")
    frame.to_csv(workdir / "landscape_metrics.csv", index=False)
    (resultdir / "logs/03_data_summary.json").write_text(json.dumps({
        "analysis_plan_section": "1.2.2", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_scenarios": 7, "n_seed_blocks": 30, "n_trajectories": 210, "n_snapshots": 1680,
        "new_generated_trajectories": 140, "reused_non_access_trajectories": 40,
        "reused_access_trajectories": 30, "legacy_source": legacy_provenance,
        "access_source": access_provenance, "new_trajectory_producer_fingerprint": fingerprint,
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration": config,
    }, indent=2), encoding="utf-8")
    print("Ready: 210 trajectories, 1680 snapshots.", flush=True)
    summary_path = Path(__file__).with_name("04_metric_redundancy_summary.py")
    summary_spec = importlib.util.spec_from_file_location("metric_redundancy_summary", summary_path)
    summary = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary)
    summary.run(config, workdir, resultdir)


if __name__ == "__main__":
    main()
