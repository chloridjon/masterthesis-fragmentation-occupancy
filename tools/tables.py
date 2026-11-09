"""Generate the two numerical tables present in the final thesis."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "Masterarbeit_final"


def design():
    config = FINAL / "config"
    landscape = json.loads(next(config.glob("01_landscape/01_03_result_a/*/07_structural_scenario_responses.json")).read_text())
    reference = json.loads(next(config.glob("02_metapop/02_03_main_analyses/02_03_01_scenario_occupancy/reference_conditional_256/10_reference_update.json")).read_text())
    parameter = json.loads(next(config.glob("02_metapop/02_03_main_analyses/02_03_02_species_dependence/confirmed_domain_256/11_replication_8_16.json")).read_text())
    n_h = len(landscape["h_targets"])
    values = [("Forest structure and real-landscape coverage", len(landscape["scenarios"])*landscape["n_seed_blocks"]),
              ("Occupancy at the reference parameter combination", len(landscape["scenarios"])*reference["n_reference_seeds"]),
              ("Occupancy and metric associations across 256 parameter combinations", sum(parameter["scenario_seed_targets"].values()))]
    result = pd.DataFrame([dict(comparison=label, runs=n, landscape_states=n*n_h) for label,n in values])
    assert result.runs.tolist() == [1400, 140, 104]
    return result


def access(path):
    data = pd.read_csv(path, float_precision="round_trip")
    part = data[np.isclose(data.k, 3) & np.isclose(data["lambda"], .5) & data.eta.isin([1.,2.])]
    index = ["metric", "rho", "H_target"]
    med = part.pivot(index=index, columns="eta", values="median")
    n = part.pivot(index=index, columns="eta", values="n_valid")
    valid = (n[1.] >= 8) & (n[2.] >= 8) & np.isfinite(med[1.]) & np.isfinite(med[2.])
    delta = (med[2.] - med[1.])[valid]
    rows = []
    order = ["edge_density_total", "largest_cluster_fraction_forest", "chi", "mean_nearest_neighbor_distance", "mean_pairwise_distance", "complete_mst_d50"]
    for metric in order:
        v = delta.xs(metric, level="metric").to_numpy()
        equal = np.isclose(v, 0, rtol=0, atol=1e-12)
        rows.append(dict(metric=metric, evaluated=len(v), lower=int(((v<0)&~equal).sum()), equal=int(equal.sum()), higher=int(((v>0)&~equal).sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["design", "access"])
    p.add_argument("--input", type=Path)
    args = p.parse_args()
    if args.kind == "design":result = design()
    else:
        path = args.input or next(FINAL.glob("results/01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection/focal_rho_090_099_0999/tables/02_metric_summary.csv"), None)
        if path is None:raise SystemExit("Run the access producer first or provide --input")
        result = access(path)
    out = FINAL / "results/thesis_tables"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / ("3_01_design.csv" if args.kind=="design" else "A_01_access_eta.csv"), index=False)
    print(result.to_string(index=False))
