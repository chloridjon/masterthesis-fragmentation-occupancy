"""Generate the seven fixed Result A trajectories and structural metrics.

Run from the repository root with its existing .venv Python. Each scenario is
generated once per matched seed, from H=1 to H=0; all H snapshots belong to
that complete trajectory. The separate 07_structural_scenario_summary.py
creates tables and figures after every trajectory is present.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
sys.path.insert(0, str(FINAL / "code/base"))

import numpy as np
import pandas as pd
from src.landscape.generation import run_sequential_deforestation

ACCESS_PATH = FINAL / "code/analyses/01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection/02_access_parameter_selection.py"
spec = importlib.util.spec_from_file_location("access_metrics_for_result_a", ACCESS_PATH)
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FINAL / "config" / REL / "07_structural_scenario_responses.json")
    parser.add_argument("--processed-dir", type=Path, default=FINAL / "data/processed" / REL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-trajectories", type=int, default=0, help="Development limit; do not aggregate an incomplete sample.")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def expected_random_removals(h: float, rho: float) -> float:
    n = round(40000 * (1 - h))
    return 0.0 if n == 0 else 1.0 + (n - 1) * (1.0 - rho)


def load_config(path: Path) -> dict:
    c = json.loads(path.read_text(encoding="utf-8"))
    if c["analysis_plan_section"] != "1.3.1" or c["grid_size_cells"] != 200:
        raise ValueError("Wrong analysis section or grid size.")
    h = np.asarray(c["h_targets"], dtype=float)
    if h[0] != 0 or h[-1] != 1 or not np.all(np.diff(h) > 0):
        raise ValueError("H targets must span [0, 1] in strictly increasing order.")
    midpoints = np.arange(0.025, 1.0, 0.05)
    if not all(np.isclose(h, m, atol=1e-12).any() for m in midpoints):
        raise ValueError("All real-sampling H-bin midpoints must be represented.")
    if not all(np.isclose(h, m, atol=1e-12).any() for m in (0.005, 0.01, 0.02, 0.03, 0.04)):
        raise ValueError("The additional low-H levels are missing.")
    if not all(np.isclose(h, m / 100, atol=1e-12).any() for m in range(5, 66)):
        raise ValueError("The 0.01-spaced transition grid from H=0.05 to 0.65 is incomplete.")
    if len(set(round(x * 40000) for x in h)) != len(h) or not np.allclose(h * 40000, np.round(h * 40000)):
        raise ValueError("H targets must correspond to distinct exact cell counts.")
    if c["n_seed_blocks"] != 200 or len(c["scenarios"]) != 7:
        raise ValueError("Incomplete matched replicate or scenario design.")
    from src.tools.final_plot_style import FOCAL_SCENARIOS
    if [s["scenario_id"] for s in c["scenarios"]] != [s["scenario_id"] for s in FOCAL_SCENARIOS]:
        raise ValueError("Scenario IDs differ from the fixed plotting map.")
    if [(s["method"], s["rho"], s["mode"]) for s in c["scenarios"]] != [
        ("only_random", 0.0, "none"),
        ("edge_random", 0.9, "edge_random"), ("access_weighted", 0.9, "access_weighted"),
        ("edge_random", 0.99, "edge_random"), ("access_weighted", 0.99, "access_weighted"),
        ("edge_random", 0.999, "edge_random"), ("access_weighted", 0.999, "access_weighted"),
    ]:
        raise ValueError("The seven scenario mechanisms or rho values have changed.")
    if (c["access_reference"], c["generation_boundary"], c["analysis_boundary"],
        c["cluster_connectivity"], c["distance_mode"], c["voronoi_depth"]) != (
        {"eta": 1.0, "k": 3, "lambda": 0.5}, "periodic", "open", "4", "voronoi_dt", 1
    ):
        raise ValueError("The selected generation or metric conventions have changed.")
    if tuple(c["metrics"]) != access.METRICS:
        raise ValueError("The six retained structural metrics have changed.")
    if c["percolation_bootstrap"]["initial_draws"] > c["percolation_bootstrap"]["maximum_draws"]:
        raise ValueError("Invalid bootstrap budget.")
    audit = c["seed_extension_audit"]
    if (audit["checkpoints"] != [50, 100, 150]
        or not 0 < audit["median_tolerance_fraction_of_span"] < 1
        or not 0 < audit["endpoint_tolerance_fraction_of_span"] < 1
        or not 0 < audit["maximum_fraction_cells_exceeding_tolerance"] < 1):
        raise ValueError("Unexpected seed-extension checkpoints or tolerance.")
    return c


def producer_settings(c: dict) -> dict:
    return {k: v for k, v in c.items() if k not in {
        "percolation_bootstrap", "percolation_seed_influence_tolerance", "seed_extension_audit"
    }}


def producer_fingerprint(c: dict) -> str:
    digest = hashlib.sha256(json.dumps(producer_settings(c), sort_keys=True).encode())
    for relative in (
        "code/base/src/landscape/generation.py", "code/base/src/landscape/deforestation.py",
        "code/base/src/landscape/landscape.py", "code/base/src/landscape/landscape_stats.py",
        "code/base/src/landscape/boundary.py", "code/base/src/landscape/complete_edge_distances.py",
    ):
        source = FINAL / relative
        digest.update(relative.encode())
        digest.update(source.read_bytes())
    for func in (access.snapshot_metrics, expected_random_removals, trajectory_rows):
        digest.update(inspect.getsource(func).encode())
    return digest.hexdigest()


def trajectory_rows(scenario: dict, replicate: int, c: dict, fingerprint: str) -> pd.DataFrame:
    seed = int(c["base_seed"]) + replicate - 1
    params = c["access_reference"]
    run = run_sequential_deforestation(
        np.ones((200, 200), dtype=np.int8), np.random.default_rng(seed),
        rho=scenario["rho"], directed_mode=scenario["mode"],
        target_habitat_fraction=0.0, generation_boundary=c["generation_boundary"],
        eta=params["eta"], k=params["k"], lambda_decay=params["lambda"],
        save_habitat_fractions=c["h_targets"], seed=seed,
    )
    if run.n_events != 40000 or run.n_random_forced_initial != 1 or run.n_random_total + run.n_directed != run.n_events:
        raise AssertionError("Unexpected event count or unrecorded mechanism fallback.")
    random_counts = np.r_[0, np.cumsum(np.asarray(run.event_modes) == "random", dtype=np.int64)]
    rows = []
    for snap in run.snapshots:
        h = float(snap.requested_habitat_fractions[0])
        n = int(snap.event_count)
        if n != round(40000 * (1 - h)) or snap.habitat_fraction != h:
            raise AssertionError("Snapshot does not contain the requested exact cell count.")
        # No habitat patches exist at H=0. Their fractions and distances are undefined.
        metrics = ({"n_clusters": 0, "n_patches": 0, "edge_density_total": 0.0,
                    "largest_cluster_fraction_forest": np.nan, "chi": np.nan,
                    "mean_nearest_neighbor_distance": np.nan, "mean_pairwise_distance": np.nan,
                    "complete_mst_d50": np.nan}
                   if h == 0 else access.snapshot_metrics(snap.grid, c))
        rows.append({
            "scenario_id": scenario["scenario_id"], "method": scenario["method"],
            "rho": scenario["rho"], "replicate": replicate, "seed": seed,
            "trajectory_id": f"{scenario['scenario_id']}__rep{replicate:03d}",
            "H_target": h, "H_real": snap.habitat_fraction, "event_count": n,
            "expected_random_removals": expected_random_removals(h, scenario["rho"]),
            "n_random_cumulative": int(random_counts[n]),
            "n_directed_cumulative": n - int(random_counts[n]),
            "grid_sha256": hashlib.sha256(snap.grid.tobytes()).hexdigest(),
            "generator_version": run.generator_version,
            "producer_fingerprint": fingerprint, **metrics,
        })
    if len(rows) != len(c["h_targets"]):
        raise AssertionError("Incomplete trajectory snapshots.")
    return pd.DataFrame(rows)


def validate_cached(path: Path, scenario: dict, replicate: int, c: dict, fingerprint: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    identity = f"{scenario['scenario_id']}__rep{replicate:03d}"
    if len(frame) != len(c["h_targets"]) or frame.duplicated(["trajectory_id", "H_target"]).any():
        raise RuntimeError(f"Incomplete cached trajectory: {path}")
    if not frame["trajectory_id"].eq(identity).all() or not frame["producer_fingerprint"].eq(fingerprint).all():
        raise RuntimeError(f"Stale or misidentified cached trajectory: {path}")
    if not np.array_equal(np.sort(frame["H_target"].to_numpy(float)), np.asarray(c["h_targets"])):
        raise RuntimeError(f"Wrong H targets: {path}")
    if not frame["seed"].eq(c["base_seed"] + replicate - 1).all() or not frame["rho"].eq(scenario["rho"]).all():
        raise RuntimeError(f"Wrong seed or rho: {path}")
    return frame


def main() -> None:
    args = arguments()
    c = load_config(args.config)
    destination = args.processed_dir
    trajectories = destination / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    fingerprint = producer_fingerprint(c)
    manifest = {"producer_settings": producer_settings(c), "producer_fingerprint": fingerprint}
    old_c = {**c, "n_seed_blocks": 50}
    old_manifest = {"producer_settings": producer_settings(old_c),
                    "producer_fingerprint": producer_fingerprint(old_c)}
    extended_manifest = {**manifest, "reused_50_seed_source": old_manifest}
    manifest_path = destination / "run_manifest.json"
    reuse_50 = False
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing == old_manifest and c["n_seed_blocks"] > 50:
            # The replicate count changes the manifest fingerprint, but no
            # trajectory's mechanism, seed, snapshots, or metric function.
            # Validate every old row before recording its separate provenance.
            for scenario in c["scenarios"]:
                for replicate in range(1, 51):
                    path = trajectories / f"{scenario['scenario_id']}__rep{replicate:03d}.csv"
                    if not path.exists():
                        raise RuntimeError(f"Incomplete 50-seed source: {path}")
                    validate_cached(path, scenario, replicate, c, old_manifest["producer_fingerprint"])
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(extended_manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, manifest_path)
            existing = extended_manifest
        if existing == extended_manifest:
            reuse_50 = True
        elif existing != manifest:
            raise RuntimeError("Cached data were produced with another configuration or source version.")
    else:
        if list(trajectories.glob("*.csv")):
            raise RuntimeError("Unversioned trajectory cache exists without a manifest.")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(c["scenarios"]).to_csv(destination / "scenarios.csv", index=False)
    jobs = []
    for scenario in c["scenarios"]:
        for replicate in range(1, c["n_seed_blocks"] + 1):
            path = trajectories / f"{scenario['scenario_id']}__rep{replicate:03d}.csv"
            if path.exists():
                expected_fingerprint = old_manifest["producer_fingerprint"] if reuse_50 and replicate <= 50 else fingerprint
                validate_cached(path, scenario, replicate, c, expected_fingerprint)
            else:
                jobs.append((scenario, replicate, path))
    if args.validate_only and jobs:
        raise RuntimeError("Some trajectories are missing.")
    if args.max_trajectories:
        jobs = jobs[:args.max_trajectories]
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(trajectory_rows, scenario, replicate, c, fingerprint): path
                       for scenario, replicate, path in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                frame = future.result()
                temporary = path.with_suffix(".tmp")
                frame.to_csv(temporary, index=False)
                os.replace(temporary, path)
                if index % 25 == 0 or index == len(jobs):
                    print(f"Generated {index}/{len(jobs)} new trajectories", flush=True)
    paths = sorted(trajectories.glob("*.csv"))
    if len(paths) != len(c["scenarios"]) * c["n_seed_blocks"]:
        print(f"Incomplete sample: {len(paths)} trajectories; aggregation deferred.", flush=True)
        return
    frames = []
    for scenario in c["scenarios"]:
        for replicate in range(1, c["n_seed_blocks"] + 1):
            path = trajectories / f"{scenario['scenario_id']}__rep{replicate:03d}.csv"
            expected_fingerprint = old_manifest["producer_fingerprint"] if reuse_50 and replicate <= 50 else fingerprint
            frames.append(validate_cached(path, scenario, replicate, c, expected_fingerprint))
    data = pd.concat(frames, ignore_index=True)
    if len(data) != len(c["scenarios"]) * c["n_seed_blocks"] * len(c["h_targets"]):
        raise AssertionError("Wrong aggregate sample size.")
    if data.duplicated(["scenario_id", "seed", "H_target"]).any():
        raise AssertionError("Duplicate scenario-seed-H key.")
    data.to_csv(destination / "landscape_metrics.csv", index=False)
    print(f"Validated {len(data)} snapshots from {len(paths)} complete trajectories.", flush=True)


if __name__ == "__main__":
    main()
