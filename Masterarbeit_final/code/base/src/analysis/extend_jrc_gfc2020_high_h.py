from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import rasterio

from src.analysis.sample_jrc_gfc2020_landscapes import (
    eligible_catalog_rows_for_h_range,
    export_stratified_landscapes,
    inspect_raster,
    read_csv_rows,
    write_csv_atomic,
    write_json_atomic,
)


DEFAULT_ROOT = Path("data/real_landscapes")
DEFAULT_SOURCE = Path("data/external/JRC_GFC2020_V3_COG.tif")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend the selected JRC sample with H > 0.60 through H = 1.00.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--samples-per-bin", type=int, default=100)
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--h-min-exclusive", type=float, default=0.60)
    parser.add_argument("--h-max", type=float, default=1.0)
    parser.add_argument("--selection-seed", type=int, default=20260902)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def combine_manifests(root: Path, high_rows: list[dict[str, object]]) -> dict[str, object]:
    manifest = root / "selected_landscapes.csv"
    original = root / "selected_landscapes_h_le_060.csv"
    if not original.exists():
        low_rows = read_csv_rows(manifest)
        write_csv_atomic(original, low_rows)
    else:
        low_rows = read_csv_rows(original)

    low_ids = {str(row["candidate_id"]) for row in low_rows}
    high_ids = {str(row["candidate_id"]) for row in high_rows}
    overlap = sorted(low_ids.intersection(high_ids))
    if overlap:
        raise RuntimeError(f"Low- and high-H manifests overlap for {len(overlap)} candidate ids.")
    combined = [*low_rows, *high_rows]
    write_csv_atomic(manifest, combined)
    summary = {
        "low_h_samples": len(low_rows),
        "high_h_samples": len(high_rows),
        "total_samples": len(combined),
        "candidate_id_overlap": 0,
        "low_h_manifest": str(original),
        "high_h_manifest": str(root / "selected_landscapes_h_gt_060.csv"),
        "combined_manifest": str(manifest),
    }
    write_json_atomic(root / "selection_summary_full_h.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    catalog = read_csv_rows(args.root / "candidate_catalog.csv")
    eligible = eligible_catalog_rows_for_h_range(
        catalog,
        h_min_exclusive=float(args.h_min_exclusive),
        h_max=float(args.h_max),
        bin_size=float(args.bin_size),
    )
    source_metadata_path = args.root / "source_metadata.json"
    land_metadata_path = args.root / "land_mask_metadata.json"
    land_metadata = json.loads(land_metadata_path.read_text(encoding="utf-8"))
    with rasterio.open(args.source) as source:
        source_metadata = (
            json.loads(source_metadata_path.read_text(encoding="utf-8"))
            if source_metadata_path.exists()
            else inspect_raster(source)
        )
        high_rows = export_stratified_landscapes(
            source,
            catalog_rows=eligible,
            outdir=args.root,
            n_per_stratum=int(args.samples_per_bin),
            selection_seed=int(args.selection_seed),
            majority_threshold=0.50,
            source_metadata=source_metadata,
            land_metadata=land_metadata,
            resume=not bool(args.no_resume),
            progress_every=int(args.progress_every),
            manifest_filename="selected_landscapes_h_gt_060.csv",
            summary_filename="selection_summary_high_h.json",
        )
    result = combine_manifests(args.root, high_rows)
    result["eligible_high_h_candidates"] = len(eligible)
    print(result)
    return result


if __name__ == "__main__":
    main()
