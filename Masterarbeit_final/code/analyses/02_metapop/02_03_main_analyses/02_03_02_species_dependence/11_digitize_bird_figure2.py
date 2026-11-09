"""Digitize Figure 2 curves and displayed bands under a documented CI assumption.

No regression is fitted to the band, no original covariance is inferred and
the Table 1 reconstruction remains a separate central-response source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, label

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CONFIG = ROOT / "Masterarbeit_final/config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/bird_reconstruction/11_figure2_digitization.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def groups(values: np.ndarray, maximum_gap: int = 1) -> list[np.ndarray]:
    return [g for g in np.split(values, np.flatnonzero(np.diff(values) > maximum_gap) + 1) if len(g)]


def calibrate(values: list[float], pixels: list[float]) -> dict:
    slope, intercept = np.polyfit(values, pixels, 1)
    residual = np.asarray(pixels) - (intercept + slope * np.asarray(values))
    return {"slope_pixels_per_unit": float(slope), "intercept_pixels": float(intercept),
            "tick_values": values, "tick_pixels": pixels,
            "maximum_tick_fit_residual_pixels": float(abs(residual).max()),
            "units_per_pixel": float(1 / abs(slope))}


def curve_mask(rgb: np.ndarray) -> np.ndarray:
    """Select the green connected curve with the greatest horizontal span.

    Component selection uses only source colour and connectivity, not the
    reconstructed Table 1 curve. Green bird illustrations have shorter spans.
    Original pixels are retained; dilation is only used to identify connectivity.
    """
    v = rgb.astype(float)
    green = ((v[:, :, 1] - v[:, :, 0] > 25) & (v[:, :, 1] - v[:, :, 2] > 25)
             & (v[:, :, 1] > 70) & (v[:, :, 1] < 190))
    components, count = label(binary_dilation(green, iterations=1))
    choices = []
    for k in range(1, count + 1):
        _, x = np.nonzero(green & (components == k))
        if len(x): choices.append((int(x.max() - x.min()), len(x), k))
    if not choices:
        raise ValueError("No green curve component")
    span, _, k = max(choices)
    if span < rgb.shape[1] * 0.8:
        raise ValueError("Longest green component covers too little of the panel")
    return green & (components == k)


def band_at_column(v: np.ndarray, y: np.ndarray, green_y: np.ndarray, mask_cfg: dict) -> dict:
    center = float(np.median(green_y))
    neutral = ((np.ptp(v, axis=1) <= mask_cfg["grey_chroma_max"])
               & (v.mean(axis=1) >= mask_cfg["grey_intensity_min"])
               & (v.mean(axis=1) <= mask_cfg["grey_intensity_max"]))
    # Include coloured line antialiasing so steep/thick lines do not split bands.
    fringe = ((v[:, 1] - v[:, 0] > 4) & (v[:, 1] - v[:, 2] > 4) & (v[:, 1] > 70)
              & (y >= green_y.min() - 6) & (y <= green_y.max() + 6))
    runs = [g for g in groups(y[neutral | fringe], mask_cfg["merge_vertical_gaps_pixels"])
            if len(g) > 1 and g[0] <= center <= g[-1]]
    if not runs:
        return {"upper_y": None, "lower_y": None, "neutral_pixels": 0}
    run = max(runs, key=len)
    pixels = int(neutral[(y >= run[0]) & (y <= run[-1])].sum())
    return {"upper_y": float(run[0]), "lower_y": float(run[-1]), "neutral_pixels": pixels}


def in_rectangle(x: float, y: float, rectangle: list[float], guard: float) -> bool:
    left, top, right, bottom = rectangle
    return left - guard <= x <= right + guard and top - guard <= y <= bottom + guard


def analyse_panel(a: np.ndarray, index: int, species: dict, cfg: dict) -> tuple[list[dict], dict]:
    row, col = divmod(index, 6)
    left, right = cfg["column_frame_bounds"][col]
    top, bottom = cfg["row_frame_bounds"][row]
    tick_strip = a[bottom + 2:bottom + 9, left + 3:right - 2]
    tick_x = np.flatnonzero((tick_strip.max(axis=2) < 130).sum(axis=0) >= 3) + left + 3
    ticks = [float(g.mean()) for g in groups(tick_x)]
    if len(ticks) != 5:
        raise ValueError(f"Expected five forest-cover ticks, got {ticks}")
    xc = calibrate(cfg["x_tick_values_H"], ticks)
    yc = calibrate(cfg["y_tick_values"], cfg["y_tick_centres_by_row"][row])
    assert xc["slope_pixels_per_unit"] > 0 and yc["slope_pixels_per_unit"] < 0
    assert max(xc["maximum_tick_fit_residual_pixels"], yc["maximum_tick_fit_residual_pixels"]) < 1
    # Exclude the frame's grey antialias fringe, which is not a band boundary.
    rgb = a[top + 6:bottom - 5, left + 3:right - 2]
    green = curve_mask(rgb)
    y = np.arange(top + 6, bottom - 5)
    traces = []
    for column in range(rgb.shape[1]):
        gy = y[green[:, column]]
        if not len(gy): continue
        x = column + left + 3
        h = (x - xc["intercept_pixels"]) / xc["slope_pixels_per_unit"]
        center_y = float(np.median(gy))
        center = (center_y - yc["intercept_pixels"]) / yc["slope_pixels_per_unit"]
        v = rgb[:, column].astype(float)
        band = band_at_column(v, y, gy, cfg)
        variants = [band_at_column(v, y, gy, {**cfg, **alternative}) for alternative in cfg["sensitivity_masks"]]
        trace = {"species_id": species["species_id"], "species_as_published": species["species_as_published"],
                 "H_source_pixel": float(h), "x_source_pixel": x, "curve_y_source_pixel": center_y,
                 "within_observed_domain": float(species["observed_H_min"]) <= h <= float(species["observed_H_max"]),
                 "nominal_ci_coverage_assumed": cfg["nominal_coverage_assumed"],
                 "band_definition_verified": cfg["band_definition_verified"],
                 "ci_interpretation": "assumed_pointwise_95_percent",
                 "psi_figure2_curve": float(np.clip(center, 0, 1)),
                 "curve_vertical_half_width_pixels": float((gy.max() - gy.min() + 1) / 2),
                 "H_per_pixel": xc["units_per_pixel"], "psi_per_pixel": yc["units_per_pixel"],
                 "band_neutral_pixel_count": band["neutral_pixels"]}
        for side in ("lower", "upper"):
            value_y = band[f"{side}_y"]
            valid = value_y is not None and band["neutral_pixels"] >= cfg["minimum_neutral_band_pixels"]
            status = "source_trace" if valid else "insufficient_visible_band_pixels"
            found = [variant[f"{side}_y"] for variant in [band, *variants] if variant[f"{side}_y"] is not None]
            spread = max(found) - min(found) if len(found) == 3 else None
            if valid and (spread is None or spread > cfg["maximum_mask_sensitivity_pixels"]):
                valid, status = False, "colour_mask_sensitive"
            rectangle = cfg["potential_upper_band_occlusion_rectangles"].get(species["species_id"])
            if side == "upper" and value_y is not None and rectangle and in_rectangle(x, value_y, rectangle, cfg["occlusion_guard_pixels"]):
                valid, status = False, "potential_illustration_occlusion"
            raw_probability = (value_y - yc["intercept_pixels"]) / yc["slope_pixels_per_unit"] if value_y is not None else None
            pixel_tolerance = yc["units_per_pixel"] * 1.5
            if raw_probability is not None and not -pixel_tolerance <= raw_probability <= 1 + pixel_tolerance:
                valid, status = False, "outside_probability_axis"
            clipped = raw_probability is not None and not 0 <= raw_probability <= 1
            if valid and clipped:
                status = "axis_resolution_snap"
            trace[f"{side}_candidate_y_pixel"] = value_y if value_y is not None else ""
            trace[f"{side}_candidate_probability"] = raw_probability if raw_probability is not None else ""
            trace[f"{side}_mask_spread_pixels"] = spread if spread is not None else ""
            trace[f"psi_ci_{side}_digitized"] = float(np.clip(raw_probability, 0, 1)) if valid else ""
            trace[f"{side}_available"] = valid
            trace[f"{side}_status"] = status
            trace[f"{side}_axis_resolution_snap"] = valid and clipped
        if trace["lower_available"]: assert trace["psi_ci_lower_digitized"] <= trace["psi_figure2_curve"] + 1e-12
        if trace["upper_available"]: assert trace["psi_ci_upper_digitized"] >= trace["psi_figure2_curve"] - 1e-12
        traces.append(trace)
    assert len(traces) > 190
    # An isolated large step is not silently smoothed or replaced.
    for side in ("lower", "upper"):
        for i in range(1, len(traces) - 1):
            before, current, after = traces[i - 1:i + 2]
            if all(t[f"{side}_available"] for t in (before, current, after)):
                key = f"psi_ci_{side}_digitized"
                if min(abs(current[key] - before[key]), abs(current[key] - after[key])) > 0.05 and abs(before[key] - after[key]) < 0.02:
                    current[f"{side}_available"] = False
                    current[key] = ""
                    current[f"{side}_status"] = "isolated_image_artifact"
    details = {"species_id": species["species_id"], "panel_index": index + 1,
               "plot_bounds_pixels": [left, top, right, bottom], "x_calibration": xc, "y_calibration": yc,
               "native_curve_columns": len(traces),
               "lower_available_columns": sum(t["lower_available"] for t in traces),
               "upper_available_columns": sum(t["upper_available"] for t in traces)}
    return traces, details


def nearest_grid_row(t: list[dict], source_row: dict, cfg: dict) -> dict:
    h = float(source_row["H"])
    actual = min(t, key=lambda v: abs(v["H_source_pixel"] - h))
    in_support = t[0]["H_source_pixel"] <= h <= t[-1]["H_source_pixel"]
    close = abs(actual["H_source_pixel"] - h) <= actual["H_per_pixel"] * 0.51
    available = in_support and close
    result = {"species_id": source_row["species_id"], "species_as_published": source_row["species_as_published"],
              "H": h, "psi_table1_reconstructed": float(source_row["psi_reconstructed"]),
              "H_source_pixel": actual["H_source_pixel"] if available else "",
              "H_sampling_difference": actual["H_source_pixel"] - h if available else "",
              "psi_figure2_curve": actual["psi_figure2_curve"] if available else "",
              "figure2_curve_available": available}
    for side in ("lower", "upper"):
        valid = available and actual[f"{side}_available"]
        result[f"psi_ci_{side}_digitized"] = actual[f"psi_ci_{side}_digitized"] if valid else ""
        result[f"{side}_available"] = valid
        result[f"{side}_status"] = actual[f"{side}_status"] if available else "outside_native_pixel_support"
        result[f"{side}_axis_resolution_snap"] = valid and actual[f"{side}_axis_resolution_snap"]
    result.update({"nominal_ci_coverage_assumed": cfg["nominal_coverage_assumed"],
                   "band_definition_verified": cfg["band_definition_verified"],
                   "ci_interpretation": "assumed_pointwise_95_percent",
                   "H_per_pixel": actual["H_per_pixel"], "psi_per_pixel": actual["psi_per_pixel"]})
    return result


def overlays(image: Image.Image, traces: list[dict], directory: Path, cfg: dict) -> list[Path]:
    """Source-coordinate overlays retain the original axes for verification."""
    out = image.copy(); draw = ImageDraw.Draw(out)
    for t in traces:
        draw.point((t["x_source_pixel"], round(t["curve_y_source_pixel"])), fill=(0, 114, 178))
        for side in ("lower", "upper"):
            if t[f"{side}_available"]:
                draw.point((t["x_source_pixel"], round(t[f"{side}_candidate_y_pixel"])), fill=(230, 159, 0))
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in range(3):
        for triplet in range(2):
            if row == 2 and triplet == 1: continue
            c0, c1 = triplet * 3, triplet * 3 + 2
            x0, x1 = cfg["column_frame_bounds"][c0][0] - 9, cfg["column_frame_bounds"][c1][1] + 6
            top, bottom = cfg["row_frame_bounds"][row]
            path = directory / f"11_figure2_source_overlay_{row * 2 + triplet + 1}.png"
            out.crop((x0, max(0, top - 50), min(out.width, x1), bottom + 42)).save(path, dpi=(300, 300))
            paths.append(path)
    return paths


def run(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    image_path = ROOT / cfg["image"]
    assert sha(image_path) == cfg["image_sha256"], "Source image changed"
    image = Image.open(image_path).convert("RGB")
    assert list(image.size) == cfg["image_size"]
    a = np.asarray(image)
    data_dir, results_dir = ROOT / cfg["data_directory"], ROOT / cfg["results_directory"]
    coef_path = data_dir / "11_bird_curve_coefficients.csv"
    main_path = data_dir / "11_bird_curves_main_H_grid.csv"
    with coef_path.open(newline="") as f: species = list(csv.DictReader(f))
    assert [s["species_id"] for s in species] == cfg["panel_order_species_ids"]
    traces, panels, grouped = [], [], {}
    for i, s in enumerate(species):
        ts, panel = analyse_panel(a, i, s, cfg)
        grouped[s["species_id"]] = ts; traces.extend(ts); panels.append(panel)
    with main_path.open(newline="") as f: main_rows = list(csv.DictReader(f))
    merged = [nearest_grid_row(grouped[r["species_id"]], r, cfg) for r in main_rows]
    assert len(merged) == 1170 and len({(r["species_id"], r["H"]) for r in merged}) == len(merged)
    comparison = []
    for s in species:
        b0, b1 = float(s["beta0_reconstructed_logit"]), float(s["beta1_reconstructed_per_H"])
        ts = grouped[s["species_id"]]
        for h in (0.2, 0.4):
            nearest = min(ts, key=lambda t: abs(t["H_source_pixel"] - h))
            p = 1 / (1 + np.exp(-(b0 + b1 * h)))
            # Steep lines span multiple y pixels at a given source column.
            uncertainty = ((nearest["curve_vertical_half_width_pixels"] + 1) * nearest["psi_per_pixel"]
                           + b1 * p * (1 - p) * (abs(nearest["H_source_pixel"] - h) + nearest["H_per_pixel"]) + 0.005)
            error = abs(nearest["psi_figure2_curve"] - p)
            assert error <= uncertainty, (s["species_id"], h, error, uncertainty)
            comparison.append({"species_id": s["species_id"], "H": h, "psi_table1": float(p),
                               "psi_figure2_nearest_pixel": nearest["psi_figure2_curve"],
                               "absolute_difference": float(error), "pixel_and_rounding_comparison_tolerance": float(uncertainty)})
    paths = [data_dir / "11_bird_figure2_native_pixel_traces.csv",
             data_dir / "11_bird_curves_main_H_grid_with_figure2_bands.csv",
             results_dir / "tables/11_figure2_anchor_comparison.csv"]
    for p, rows in zip(paths, (traces, merged, comparison)): write_csv(p, rows)
    calibration_path = data_dir / "11_figure2_panel_calibration.json"
    write_json(calibration_path, {"source_image_sha256": sha(image_path), "panels": panels})
    paths.append(calibration_path)
    overlay_paths = overlays(image, traces, results_dir / "figures/figure2_digitization_qa", cfg)
    source_manifest = image_path.parent / "figure2_source_manifest.json"
    write_json(source_manifest, {"doi": "https://doi.org/10.1007/s10980-020-01121-6",
                                "figure_page": cfg["figure_page"], "image_url": cfg["image_url"],
                                "source_image_sha256": sha(image_path), "retrieved_on": "2026-10-13",
                                "image_size_pixels": list(image.size), "format": "Publisher PNG, not the browser-transcoded WebP",
                                "band_interpretation": cfg["interpretation"], "band_definition_verified": False})
    audit = {"status": "numerically_checked_visual_review_pending", "visual_review_confirmed": False,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": cfg,
             "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (config_path, image_path, coef_path, main_path, Path(__file__).resolve())},
             "native_trace_rows": len(traces), "species": len(species), "main_grid_rows": len(merged),
             "figure2_curve_available_rows": sum(r["figure2_curve_available"] for r in merged),
             "lower_available_rows": sum(r["lower_available"] for r in merged),
             "upper_available_rows": sum(r["upper_available"] for r in merged),
             "complete_band_rows": sum(r["lower_available"] and r["upper_available"] for r in merged),
             "missing_by_species": {s["species_id"]: {side: sum(not r[f"{side}_available"] for r in merged if r["species_id"] == s["species_id"]) for side in ("lower", "upper")} for s in species},
             "missing_reasons": {side: {status: sum(r[f"{side}_status"] == status for r in merged) for status in sorted({r[f"{side}_status"] for r in merged if not r[f"{side}_available"]})} for side in ("lower", "upper")},
             "H_per_pixel_range": [min(t["H_per_pixel"] for t in traces), max(t["H_per_pixel"] for t in traces)],
             "psi_per_pixel_range": [min(t["psi_per_pixel"] for t in traces), max(t["psi_per_pixel"] for t in traces)],
             "maximum_anchor_difference": max(r["absolute_difference"] for r in comparison),
             "checks": ["Frozen image hash and dimensions", "15 source-panel identities", "Five x ticks and six y calibration ticks", "Green component horizontal coverage", "Colour-mask sensitivity", "Potential illustration occlusion flags", "Probability bounds and curve/band ordering", "Unique main-grid keys", "30 Table 1 anchor comparisons accounting for line width and pixel scale", "No native-support extrapolation"],
             "output_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths + overlay_paths}}
    audit_path = results_dir / "logs/11_figure2_digitization_audit.json"
    write_json(audit_path, audit)
    write_json(data_dir / "figure2_digitization_manifest.json", {
        "schema_version": cfg["schema_version"], "source_manifest": str(source_manifest.relative_to(ROOT)),
        "source_manifest_sha256": sha(source_manifest), "audit_path": str(audit_path.relative_to(ROOT)),
        "band_definition_verified": False, "nominal_coverage_assumed": 0.95,
        "interpretation": cfg["interpretation"], "output_sha256": audit["output_sha256"],
        "grid_sampling_method": cfg["grid_sampling_method"],
        "missing_values": "Blank numeric cells accompanied by availability flags and reasons; never zero substitutes",
        "ecological_unit": "Empirical forest-site occupancy probability conditional on forest cover",
        "statistical_limit": "Digitized displayed limits under an assumed pointwise CI interpretation; no original coefficient covariance and no joint curve uncertainty recovered",
    })
    return audit


def confirm(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text()); base = ROOT / cfg["results_directory"]
    path = base / "logs/11_figure2_digitization_audit.json"
    audit = json.loads(path.read_text())
    for rel, digest in {**audit["input_sha256"], **audit["output_sha256"]}.items():
        assert sha(ROOT / rel) == digest, f"Changed artifact: {rel}"
    audit.update({"visual_review_confirmed": True, "status": "numerically_and_visually_checked",
                  "visual_review_at_utc": datetime.now(timezone.utc).isoformat(),
                  "visual_review_scope": "All 15 panels in five native-resolution source overlays; occluded boundaries remain missing"})
    write_json(path, audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-visual-review", action="store_true", help="Only after inspecting all five source overlay images")
    args = parser.parse_args()
    result = confirm(args.config.resolve()) if args.confirm_visual_review else run(args.config.resolve())
    print(json.dumps({k: result[k] for k in ("status", "species", "native_trace_rows", "main_grid_rows", "lower_available_rows", "upper_available_rows", "complete_band_rows", "missing_reasons", "maximum_anchor_difference")}, indent=2))
