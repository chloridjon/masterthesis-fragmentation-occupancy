"""Independently reproduce the original stratified IDs from a local catalogue.

Reads empirical inputs without changing them. Prints a compact audit only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sample(rows, seed):
    groups = {}
    for row in rows:
        key = (int(float(row["extent_m"])), int(row["h_bin_index"]))
        groups.setdefault(key, []).append(row)
    rng = np.random.default_rng(seed)
    result = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["candidate_id"])
        if len(group) < 100:
            raise ValueError(f"Incomplete stratum {key}")
        index = np.sort(rng.choice(len(group), size=100, replace=False))
        result.extend(group[int(i)]["candidate_id"] for i in index)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    root = args.root
    catalog = read(root / "candidate_catalog.csv")
    low = [r for r in catalog if r.get("status") == "eligible" and r.get("h_bin_index", "").strip()]
    high = []
    for row in catalog:
        if row.get("passes_land_filter", "").lower() != "true":
            continue
        h = float(row["forest_cover"])
        if not math.isclose(float(row["valid_fraction"]), 1, abs_tol=1e-9) or not 0.6 < h <= 1:
            continue
        row = row.copy()
        # Match the documented bin tolerance, including values on 0.65/0.70
        # boundaries that would otherwise fall into the preceding binary bin.
        row["h_bin_index"] = str(min(int(math.floor(h / 0.05 + 1e-12)), 19))
        high.append(row)
    original = [r["candidate_id"] for r in read(root / "selected_landscapes.csv")]
    recreated = sample(low, 20260833) + sample(high, 20260902)
    missing = sorted(set(original) - set(recreated))
    added = sorted(set(recreated) - set(original))
    report = dict(status="passed" if original == recreated else "failed",
        n_catalog_rows=len(catalog), n_low_candidates=len(low), n_high_candidates=len(high),
        n_original=len(original), n_recreated=len(recreated), exact_id_order=original == recreated,
        missing_ids=missing, unexpected_ids=added,
        selection_seeds=dict(low=20260833, high=20260902), numpy_version=np.__version__,
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in ["candidate_catalog.csv", "selected_landscapes.csv"]},
        scope="Selection from original catalogue; does not rerun source raster aggregation")
    print(json.dumps(report, indent=2))
    raise SystemExit(original != recreated)


if __name__ == "__main__":
    main()
