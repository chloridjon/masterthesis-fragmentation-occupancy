"""Result A 1.3.2: descriptive JRC-window coverage by fixed scenario and H-bin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_02_real_landscape_coverage")
SIM_REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
CONFIG = FINAL / "config" / REL / "08_real_landscape_coverage.json"
REAL = FINAL / "data/processed" / REL
SIM = FINAL / "data/processed" / SIM_REL
TABLES = FINAL / "results" / REL / "tables"
LOGS = FINAL / "results" / REL / "logs"
SCENARIOS = (
    "only_random_rho0", "edge_random_rho09", "access_weighted_rho09",
    "edge_random_rho099", "access_weighted_rho099",
    "edge_random_rho0999", "access_weighted_rho0999",
)
ANY = "any_scenario"


def checked_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (c["analysis_plan_section"] != "1.3.2" or c["n_simulation_seed_blocks"] != 200 or
        c["n_h_bins"] != 20 or c["h_bin_width"] != .05 or
        c["window_extents_m"] != [2000, 4000, 8000, 16000, 32000] or
        c["simulation_interval_quantiles"] != [.05, .95] or
        c["minimum_valid_simulation_trajectories"] != 30 or
        c["coverage_stability_checkpoints"] != [50, 100, 150]):
        raise ValueError("Coverage configuration differs from the frozen design.")
    manifest = json.loads((REAL / "08_real_source_manifest.json").read_text(encoding="utf-8"))
    real = pd.read_csv(REAL / "08_real_landscape_metrics.csv")
    sim_manifest = json.loads((SIM / "run_manifest.json").read_text(encoding="utf-8"))
    sim = pd.read_csv(SIM / "landscape_metrics.csv")
    if (len(real) != 10000 or real["candidate_id"].duplicated().any() or
        len(sim) != 7 * 200 * 88 or sim.duplicated(["scenario_id", "replicate", "H_target"]).any() or
        manifest["n_windows"] != len(real) or
        sim_manifest["producer_settings"]["n_seed_blocks"] != 200 or
        set(sim["scenario_id"]) != set(SCENARIOS)):
        raise ValueError("The real or simulated input sample is incomplete.")
    if not real.groupby(["extent_m", "h_bin_index"]).size().eq(c["windows_per_stratum"]).all():
        raise ValueError("The real sample is not balanced by size and H-bin.")
    metrics = c["metrics"]
    if any(metric not in real or metric not in sim for metric in metrics):
        raise ValueError("The six retained structural metrics are unavailable.")
    return c, real, sim, manifest


def choose_bin_snapshots(sim: pd.DataFrame, c: dict, n_seeds: int) -> pd.DataFrame:
    if not 1 <= n_seeds <= c["n_simulation_seed_blocks"]:
        raise ValueError("Invalid seed checkpoint.")
    part = sim.loc[sim["replicate"] <= n_seeds].copy()
    n_cells = np.rint(part["H_target"].to_numpy(float) * 40000).astype(int)
    part["h_bin_index"] = np.minimum(n_cells // 2000, 19)
    part["midpoint_distance"] = np.abs(part["H_target"] - (.025 + .05 * part["h_bin_index"]))
    part = part.sort_values(["scenario_id", "replicate", "h_bin_index", "midpoint_distance", "H_target"])
    part = part.drop_duplicates(["scenario_id", "replicate", "h_bin_index"])
    if len(part) != len(SCENARIOS) * n_seeds * c["n_h_bins"]:
        raise AssertionError("A scenario trajectory is missing an H-bin snapshot.")
    if not np.allclose(part["midpoint_distance"], 0, atol=1e-12):
        raise AssertionError("A simulated H-bin lacks its prespecified midpoint snapshot.")
    return part


def simulation_intervals(selected: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    minimum = c["minimum_valid_simulation_trajectories"]
    for (scenario, h_bin), part in selected.groupby(["scenario_id", "h_bin_index"], sort=False):
        for metric in c["metrics"]:
            values = part[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            available = len(finite) >= minimum
            rows.append({"scenario_id": scenario, "h_bin_index": h_bin,
                         "H_bin_midpoint": .025 + .05 * h_bin, "metric": metric,
                         "n_simulation_trajectories": len(part), "n_simulation_valid": len(finite),
                         "interval_available": available,
                         "median": float(np.median(finite)) if available else np.nan,
                         "q05": float(np.quantile(finite, .05)) if available else np.nan,
                         "q95": float(np.quantile(finite, .95)) if available else np.nan})
    result = pd.DataFrame(rows)
    if len(result) != len(SCENARIOS) * c["n_h_bins"] * len(c["metrics"]):
        raise AssertionError("The simulation interval table is incomplete.")
    return result


def real_distributions(real: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    for (extent, h_bin), part in real.groupby(["extent_m", "h_bin_index"], sort=False):
        for metric in c["metrics"]:
            values = part[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            rows.append({"extent_m": extent, "h_bin_index": h_bin,
                         "H_bin_midpoint": .025 + .05 * h_bin, "metric": metric,
                         "n_real_windows": len(part), "n_real_valid": len(finite),
                         "real_median": float(np.median(finite)) if len(finite) else np.nan,
                         "real_q05": float(np.quantile(finite, .05)) if len(finite) else np.nan,
                         "real_q95": float(np.quantile(finite, .95)) if len(finite) else np.nan})
    result = pd.DataFrame(rows)
    if len(result) != len(c["window_extents_m"]) * c["n_h_bins"] * len(c["metrics"]):
        raise AssertionError("The real-window distribution table is incomplete.")
    return result


def bin_coverage(real: pd.DataFrame, intervals: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = {(row.scenario_id, row.h_bin_index, row.metric): row
              for row in intervals.itertuples(index=False)}
    rows, gap_rows = [], []
    for (extent, h_bin), part in real.groupby(["extent_m", "h_bin_index"], sort=False):
        for metric in c["metrics"]:
            values = part[metric].to_numpy(float)
            valid = values[np.isfinite(values)]
            available = []
            for scenario in SCENARIOS:
                interval = lookup[(scenario, h_bin, metric)]
                supported = len(valid) > 0 and interval.interval_available
                count = int(((valid >= interval.q05) & (valid <= interval.q95)).sum()) if supported else np.nan
                rows.append({"extent_m": extent, "h_bin_index": h_bin,
                             "H_bin_midpoint": .025 + .05 * h_bin, "metric": metric,
                             "scenario_id": scenario, "n_real_windows": len(part),
                             "n_real_valid": len(valid), "n_simulation_valid": interval.n_simulation_valid,
                             "simulation_q05": interval.q05, "simulation_q95": interval.q95,
                             "covered_count": count,
                             "coverage_fraction": count / len(valid) if supported else np.nan})
                if interval.interval_available:
                    available.append((interval.q05, interval.q95))
            comparable = len(valid) > 0 and len(available) == len(SCENARIOS)
            if comparable:
                inside = np.zeros(len(valid), dtype=bool)
                for low, high in available:
                    inside |= (valid >= low) & (valid <= high)
                covered = int(inside.sum())
                below = int((valid < min(low for low, _ in available)).sum())
                above = int((valid > max(high for _, high in available)).sum())
                between = int((~inside).sum()) - below - above
            else:
                covered = below = above = between = np.nan
            rows.append({"extent_m": extent, "h_bin_index": h_bin,
                         "H_bin_midpoint": .025 + .05 * h_bin, "metric": metric,
                         "scenario_id": ANY, "n_real_windows": len(part),
                         "n_real_valid": len(valid), "n_simulation_valid": np.nan,
                         "simulation_q05": np.nan, "simulation_q95": np.nan,
                         "covered_count": covered,
                         "coverage_fraction": covered / len(valid) if comparable else np.nan})
            gap_rows.append({"extent_m": extent, "h_bin_index": h_bin,
                             "H_bin_midpoint": .025 + .05 * h_bin, "metric": metric,
                             "n_real_valid": len(valid), "n_covered_by_any": covered,
                             "n_below_all_intervals": below, "n_above_all_intervals": above,
                             "n_between_intervals": between,
                             "n_outside_all": len(valid) - covered if comparable else np.nan})
    coverage, gaps = pd.DataFrame(rows), pd.DataFrame(gap_rows)
    if len(coverage) != len(c["window_extents_m"]) * c["n_h_bins"] * len(c["metrics"]) * (len(SCENARIOS) + 1):
        raise AssertionError("Coverage strata are incomplete.")
    return coverage, gaps


def overall_coverage(coverage: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    for extent in c["window_extents_m"]:
        for metric in c["metrics"]:
            part = coverage[(coverage["extent_m"] == extent) & (coverage["metric"] == metric)]
            seven = part[part["scenario_id"].isin(SCENARIOS)]
            count = seven.groupby("h_bin_index")["coverage_fraction"].apply(lambda x: x.notna().sum())
            common = count[count == len(SCENARIOS)].index
            for scenario in (*SCENARIOS, ANY):
                values = part[(part["scenario_id"] == scenario) & (part["h_bin_index"].isin(common))]
                if len(values) != len(common) or values["coverage_fraction"].isna().any():
                    raise AssertionError("An included H-bin lacks a comparable coverage fraction.")
                rows.append({"extent_m": extent, "metric": metric, "scenario_id": scenario,
                             "mean_equal_bin_coverage": float(values["coverage_fraction"].mean()) if len(values) else np.nan,
                             "n_included_bins": len(common), "n_excluded_bins": c["n_h_bins"] - len(common),
                             "included_bin_indices": ",".join(str(int(x)) for x in common)})
    return pd.DataFrame(rows)


def coverage_stability(real: pd.DataFrame, sim: pd.DataFrame,
                       full: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, details = [], []
    keys = ["extent_m", "h_bin_index", "metric", "scenario_id"]
    for first_n in c["coverage_stability_checkpoints"]:
        sample = choose_bin_snapshots(sim, c, first_n)
        earlier_intervals = simulation_intervals(sample, c)
        earlier, _ = bin_coverage(real, earlier_intervals, c)
        merged = full[keys + ["coverage_fraction"]].merge(
            earlier[keys + ["coverage_fraction"]], on=keys,
            suffixes=("_full", "_earlier"), validate="one_to_one")
        for (extent, metric), group in merged.groupby(["extent_m", "metric"]):
            seven = group[group["scenario_id"].isin(SCENARIOS)]
            fully_defined = seven.assign(
                comparable=seven["coverage_fraction_full"].notna() &
                           seven["coverage_fraction_earlier"].notna()
            ).groupby("h_bin_index")["comparable"].sum()
            common_bins = fully_defined[fully_defined == len(SCENARIOS)].index
            detail = group.copy()
            detail["first_n_seed_blocks"] = first_n
            detail["full_n_seed_blocks"] = c["n_simulation_seed_blocks"]
            detail["included_in_common_bin_comparison"] = detail["h_bin_index"].isin(common_bins)
            detail["absolute_fraction_change"] = (
                detail["coverage_fraction_full"] - detail["coverage_fraction_earlier"]
            ).abs()
            details.append(detail)
            for scenario in (*SCENARIOS, ANY):
                common = group[(group["scenario_id"] == scenario) &
                               (group["h_bin_index"].isin(common_bins))]
                if len(common) != len(common_bins) or common[["coverage_fraction_full",
                                                               "coverage_fraction_earlier"]].isna().any().any():
                    raise AssertionError("A shared H-bin lacks coverage for one scenario.")
                full_mean = float(common["coverage_fraction_full"].mean()) if len(common) else np.nan
                old_mean = float(common["coverage_fraction_earlier"].mean()) if len(common) else np.nan
                rows.append({"first_n_seed_blocks": first_n, "full_n_seed_blocks": c["n_simulation_seed_blocks"],
                             "extent_m": extent, "metric": metric, "scenario_id": scenario,
                             "n_shared_evaluable_bins": len(common),
                             "full_mean_on_shared_bins": full_mean, "earlier_mean_on_shared_bins": old_mean,
                             "absolute_mean_change": abs(old_mean - full_mean) if len(common) else np.nan,
                             "maximum_bin_fraction_change": (float(np.max(np.abs(
                                 common["coverage_fraction_full"] - common["coverage_fraction_earlier"])))
                                 if len(common) else np.nan),
                             "within_overall_tolerance": (abs(old_mean - full_mean) <= c["coverage_stability_tolerance"]
                                                          if len(common) else False)})
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def main() -> None:
    c, real, sim, manifest = checked_inputs()
    selected = choose_bin_snapshots(sim, c, c["n_simulation_seed_blocks"])
    intervals = simulation_intervals(selected, c)
    distributions = real_distributions(real, c)
    coverage, gaps = bin_coverage(real, intervals, c)
    overall = overall_coverage(coverage, c)
    stability, bin_stability = coverage_stability(real, sim, coverage, c)
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("08_simulation_bin_intervals.csv", intervals),
        ("08_real_bin_distributions.csv", distributions),
        ("08_coverage_by_bin.csv", coverage),
        ("08_overall_coverage.csv", overall),
        ("08_uncovered_real_values.csv", gaps),
        ("08_coverage_seed_stability.csv", stability),
        ("08_bin_coverage_seed_stability.csv", bin_stability),
    ):
        frame.to_csv(TABLES / name, index=False)
    check = stability[stability["first_n_seed_blocks"] == 150]
    metadata = {"analysis_plan_section": "1.3.2", "n_real_windows": len(real),
                "n_simulation_seed_blocks": c["n_simulation_seed_blocks"],
                "n_simulation_bin_snapshots": len(selected),
                "simulation_producer_fingerprint": json.loads((SIM / "run_manifest.json").read_text(
                    encoding="utf-8"))["producer_fingerprint"],
                "n_common_bins_min": int(overall["n_included_bins"].min()),
                "n_common_bins_max": int(overall["n_included_bins"].max()),
                "n_150_to_200_overall_changes_over_0_02": int((check["absolute_mean_change"] > .02).sum()),
                "n_150_to_200_coverage_summaries": len(check),
                "real_source_manifest": manifest}
    (LOGS / "08_summary_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "real_source_manifest"}, indent=2))


if __name__ == "__main__":
    main()
