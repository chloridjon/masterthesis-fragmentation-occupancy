from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from rasterio.warp import (
    Resampling,
    reproject,
    transform as transform_coordinates,
    transform_bounds,
)

from src.landscape.gshhg_land_mask import GshhgLandMask


DEFAULT_EXTENTS_KM = (2, 4, 8, 16, 32)
DEFAULT_CELLS = 200
DEFAULT_MACROGRID_M = 64_000.0
DEFAULT_BIN_SIZE = 0.05
DEFAULT_H_MAX = 0.60
DEFAULT_MIN_LAND_FRACTION = 0.90
DEFAULT_MAJORITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class HBin:
    index: int
    lower: float
    upper: float
    includes_upper: bool

    @property
    def label(self) -> str:
        closing = "]" if self.includes_upper else ")"
        return f"[{self.lower:.2f},{self.upper:.2f}{closing}"


@dataclass(frozen=True)
class CandidateResult:
    lon: float
    lat: float
    extent_m: int
    cells: int
    target_cell_m: float
    forest_cover: float
    forest_fraction_mean: float
    land_fraction: float | None
    land_valid_fraction: float | None
    center_is_land: bool | None
    passes_land_filter: bool | None
    passes_h_filter: bool
    h_bin_index: int | None
    h_bin_label: str | None
    valid_fraction: float
    elapsed_seconds: float


def parse_number_list(raw: str, *, cast: type = float) -> list:
    values = [cast(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def local_laea_crs(lon: float, lat: float) -> CRS:
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError("lon must be in [-180, 180].")
    if not (-90.0 < float(lat) < 90.0):
        raise ValueError("lat must be in (-90, 90).")
    return CRS.from_proj4(
        f"+proj=laea +lat_0={float(lat):.12g} +lon_0={float(lon):.12g} "
        "+datum=WGS84 +units=m +no_defs"
    )


def local_target_grid(
    lon: float,
    lat: float,
    extent_m: float,
    cells: int = DEFAULT_CELLS,
) -> tuple[CRS, Affine]:
    if float(extent_m) <= 0.0:
        raise ValueError("extent_m must be > 0.")
    if int(cells) <= 0:
        raise ValueError("cells must be > 0.")
    pixel_m = float(extent_m) / float(cells)
    half = float(extent_m) / 2.0
    return local_laea_crs(lon, lat), from_origin(-half, half, pixel_m, pixel_m)


def _target_longitude_bounds(
    *,
    center_lon: float,
    dst_crs: CRS,
    dst_transform: Affine,
    cells: int,
    densify_points: int = 21,
) -> tuple[float, float, float, float]:
    """Return a densified WGS84 footprint with longitudes unwrapped around the center."""
    left = float(dst_transform.c)
    top = float(dst_transform.f)
    right = left + float(dst_transform.a) * int(cells)
    bottom = top + float(dst_transform.e) * int(cells)
    horizontal = np.linspace(left, right, max(2, int(densify_points)))
    vertical = np.linspace(bottom, top, max(2, int(densify_points)))
    xs = np.concatenate(
        [horizontal, horizontal, np.full(vertical.shape, left), np.full(vertical.shape, right)]
    )
    ys = np.concatenate(
        [np.full(horizontal.shape, bottom), np.full(horizontal.shape, top), vertical, vertical]
    )
    longitudes, latitudes = transform_coordinates(dst_crs, "EPSG:4326", xs, ys)
    unwrapped = np.asarray(
        [
            float(center_lon) + ((float(value) - float(center_lon) + 180.0) % 360.0 - 180.0)
            for value in longitudes
        ],
        dtype=float,
    )
    lat = np.asarray(latitudes, dtype=float)
    return float(unwrapped.min()), float(lat.min()), float(unwrapped.max()), float(lat.max())


def _read_antimeridian_mosaic(
    source: DatasetReader,
    *,
    band: int,
    unwrapped_bounds: tuple[float, float, float, float],
    padding_pixels: int = 3,
) -> tuple[np.ndarray, Affine]:
    """Read a small continuous source mosaic across the left/right edges of a global raster."""
    if source.crs != CRS.from_epsg(4326):
        raise ValueError("Antimeridian mosaics currently require an EPSG:4326 source.")
    transform = source.transform
    if transform.b != 0.0 or transform.d != 0.0 or transform.a <= 0.0 or transform.e >= 0.0:
        raise ValueError("Antimeridian mosaics require a north-up geographic raster.")
    raster_span = float(transform.a) * int(source.width)
    if not math.isclose(raster_span, 360.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Antimeridian mosaics require a 360-degree global raster.")

    xmin, ymin, xmax, ymax = (float(value) for value in unwrapped_bounds)
    pad = max(0, int(padding_pixels))
    col_start = math.floor((xmin - float(transform.c)) / float(transform.a)) - pad
    col_stop = math.ceil((xmax - float(transform.c)) / float(transform.a)) + pad
    row_start = math.floor((float(transform.f) - ymax) / abs(float(transform.e))) - pad
    row_stop = math.ceil((float(transform.f) - ymin) / abs(float(transform.e))) + pad
    row_start = max(0, row_start)
    row_stop = min(int(source.height), row_stop)
    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError("Antimeridian source window is empty.")
    if col_stop - col_start >= int(source.width):
        raise ValueError("Antimeridian source window unexpectedly spans the full raster.")

    chunks: list[np.ndarray] = []
    current = int(col_start)
    while current < int(col_stop):
        canonical_col = current % int(source.width)
        count = min(int(col_stop) - current, int(source.width) - canonical_col)
        chunks.append(
            source.read(
                int(band),
                window=Window(canonical_col, row_start, count, row_stop - row_start),
            )
        )
        current += count
    mosaic = np.concatenate(chunks, axis=1)
    mosaic_transform = transform * Affine.translation(col_start, row_start)
    return mosaic, mosaic_transform


def extract_binary_fraction(
    source: DatasetReader,
    *,
    lon: float,
    lat: float,
    extent_m: float,
    cells: int = DEFAULT_CELLS,
    band: int = 1,
    disable_overviews: bool = True,
) -> np.ndarray:
    """Average a binary 0/1 raster into a local equal-area target grid."""
    if source.crs is None:
        raise ValueError("Source raster has no CRS.")
    if int(band) < 1 or int(band) > int(source.count):
        raise ValueError(f"band must be between 1 and {source.count}.")

    dst_crs, dst_transform = local_target_grid(lon, lat, extent_m, cells)
    destination = np.full((int(cells), int(cells)), np.nan, dtype=np.float32)
    warp_options: dict[str, object] = {}
    if disable_overviews:
        # GDAL Warp otherwise may choose an overview whose resampling method is
        # unknown. Native source pixels are required for an auditable majority.
        warp_options["OVR_LEVEL"] = "NONE"

    reproject_source: object = rasterio.band(source, int(band))
    reproject_transform = source.transform
    if source.crs == CRS.from_epsg(4326):
        footprint_bounds = _target_longitude_bounds(
            center_lon=float(lon),
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            cells=int(cells),
        )
        crosses_antimeridian = (
            footprint_bounds[0] < float(source.bounds.left)
            or footprint_bounds[2] > float(source.bounds.right)
        )
        if crosses_antimeridian:
            reproject_source, reproject_transform = _read_antimeridian_mosaic(
                source,
                band=int(band),
                unwrapped_bounds=footprint_bounds,
            )

    reproject(
        source=reproject_source,
        destination=destination,
        src_transform=reproject_transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        init_dest_nodata=True,
        num_threads=1,
        **warp_options,
    )
    finite = np.isfinite(destination)
    if np.any(finite):
        lo = float(np.min(destination[finite]))
        hi = float(np.max(destination[finite]))
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            raise ValueError(
                "Expected a binary 0/1 source raster; averaged values fall outside [0, 1] "
                f"(min={lo}, max={hi})."
            )
        destination[finite] = np.clip(destination[finite], 0.0, 1.0)
    return destination


def majority_grid(
    fractions: np.ndarray,
    threshold: float = DEFAULT_MAJORITY_THRESHOLD,
) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.ndim != 2:
        raise ValueError("fractions must be a 2D array.")
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0, 1].")
    return (np.isfinite(values) & (values >= float(threshold))).astype(np.int8)


def forest_cover(grid: np.ndarray) -> float:
    values = np.asarray(grid)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("grid must be a non-empty 2D array.")
    return float(np.count_nonzero(values == 1)) / float(values.size)


def assign_h_bin(
    h: float,
    *,
    bin_size: float = DEFAULT_BIN_SIZE,
    h_min: float = 0.0,
    h_max: float = DEFAULT_H_MAX,
) -> HBin | None:
    value = float(h)
    step = float(bin_size)
    lower_limit = float(h_min)
    upper_limit = float(h_max)
    if step <= 0.0:
        raise ValueError("bin_size must be > 0.")
    if upper_limit <= lower_limit:
        raise ValueError("h_max must be greater than h_min.")
    tolerance = 1e-12
    if value < lower_limit - tolerance or value > upper_limit + tolerance:
        return None

    n_bins = int(math.ceil((upper_limit - lower_limit) / step - tolerance))
    if math.isclose(value, upper_limit, abs_tol=tolerance):
        index = n_bins - 1
    else:
        index = int(math.floor((value - lower_limit) / step + tolerance))
    index = max(0, min(index, n_bins - 1))
    lower = lower_limit + index * step
    upper = min(upper_limit, lower + step)
    return HBin(
        index=index,
        lower=float(lower),
        upper=float(upper),
        includes_upper=index == n_bins - 1,
    )


def passes_land_filter(
    land_fraction: float,
    minimum: float = DEFAULT_MIN_LAND_FRACTION,
) -> bool:
    value = float(land_fraction)
    limit = float(minimum)
    if not (0.0 <= value <= 1.0):
        raise ValueError("land_fraction must be in [0, 1].")
    if not (0.0 <= limit <= 1.0):
        raise ValueError("minimum must be in [0, 1].")
    return value >= limit


def passes_candidate_land_rules(
    *,
    center_is_land: bool,
    land_fraction: float,
    land_valid_fraction: float,
    minimum: float = DEFAULT_MIN_LAND_FRACTION,
) -> bool:
    if not (0.0 <= float(land_valid_fraction) <= 1.0):
        raise ValueError("land_valid_fraction must be in [0, 1].")
    return (
        bool(center_is_land)
        and math.isclose(float(land_valid_fraction), 1.0, abs_tol=1e-9)
        and passes_land_filter(land_fraction, minimum)
    )


def sample_binary_at_lonlat(source: DatasetReader, *, lon: float, lat: float) -> bool:
    """Return the binary source value at a WGS84 point."""
    if source.crs is None:
        raise ValueError("Source raster has no CRS.")
    xs, ys = transform_coordinates("EPSG:4326", source.crs, [float(lon)], [float(lat)])
    value = next(source.sample([(xs[0], ys[0])], indexes=1, masked=True))[0]
    if np.ma.is_masked(value):
        return False
    numeric = float(value)
    if numeric not in (0.0, 1.0):
        raise ValueError(f"Expected a binary 0/1 point value, got {numeric}.")
    return numeric == 1.0


def macrogrid_centers(
    bounds: tuple[float, float, float, float],
    *,
    cell_size_m: float = DEFAULT_MACROGRID_M,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Generate reproducibly offset projected-grid centers covering bounds."""
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    size = float(cell_size_m)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("bounds must satisfy xmin < xmax and ymin < ymax.")
    if size <= 0.0:
        raise ValueError("cell_size_m must be > 0.")
    rng = np.random.default_rng(int(seed))
    offset_x, offset_y = (float(value) for value in rng.uniform(0.0, size, size=2))

    first_x = math.floor((xmin - offset_x) / size) * size + offset_x + size / 2.0
    first_y = math.floor((ymin - offset_y) / size) * size + offset_y + size / 2.0
    xs = np.arange(first_x, xmax + size, size, dtype=float)
    ys = np.arange(first_y, ymax + size, size, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    inside = (
        (centers[:, 0] >= xmin)
        & (centers[:, 0] <= xmax)
        & (centers[:, 1] >= ymin)
        & (centers[:, 1] <= ymax)
    )
    return centers[inside], (offset_x, offset_y)


def select_macrogrid_land_centers(
    *,
    source_crs: CRS,
    source_bounds: tuple[float, float, float, float],
    land_mask: GshhgLandMask,
    macrogrid_crs: str | CRS = "EPSG:6933",
    macrogrid_size_m: float = DEFAULT_MACROGRID_M,
    seed: int = 0,
    max_centers: int = 0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    target_crs = CRS.from_user_input(macrogrid_crs)
    projected_bounds = transform_bounds(
        source_crs,
        target_crs,
        *source_bounds,
        densify_pts=21,
    )
    projected, offset = macrogrid_centers(
        projected_bounds,
        cell_size_m=float(macrogrid_size_m),
        seed=int(seed),
    )
    longitudes, latitudes = transform_coordinates(
        target_crs,
        "EPSG:4326",
        projected[:, 0].tolist(),
        projected[:, 1].tolist(),
    )
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    xmin, ymin, xmax, ymax = (float(value) for value in source_bounds)
    in_source = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
    source_indices = np.flatnonzero(in_source)
    land_flags = land_mask.centers_are_land(lon[source_indices], lat[source_indices])
    land_indices = source_indices[np.flatnonzero(land_flags)]

    if int(max_centers) > 0 and len(land_indices) > int(max_centers):
        rng = np.random.default_rng(int(seed) + 1)
        chosen = np.sort(rng.choice(land_indices, size=int(max_centers), replace=False))
    else:
        chosen = land_indices

    rows = [
        {
            "center_id": f"center_{int(index):06d}",
            "macrogrid_index": int(index),
            "macrogrid_x": float(projected[index, 0]),
            "macrogrid_y": float(projected[index, 1]),
            "lon": float(lon[index]),
            "lat": float(lat[index]),
        }
        for index in chosen
    ]
    metadata = {
        "macrogrid_crs": str(target_crs),
        "macrogrid_size_m": float(macrogrid_size_m),
        "macrogrid_seed": int(seed),
        "pilot_selection_seed": int(seed) + 1,
        "offset_x_m": float(offset[0]),
        "offset_y_m": float(offset[1]),
        "projected_bounds": [float(value) for value in projected_bounds],
        "n_projected_centers": int(len(projected)),
        "n_centers_in_source_bounds": int(len(source_indices)),
        "n_land_centers": int(len(land_indices)),
        "n_selected_centers": int(len(rows)),
        "max_centers": int(max_centers),
    }
    return rows, metadata


def sample_fixed_per_stratum(
    rows: Sequence[Mapping[str, object]],
    *,
    n_per_stratum: int,
    seed: int,
) -> list[dict[str, object]]:
    """Sample without replacement by (extent_m, h_bin_index)."""
    if int(n_per_stratum) <= 0:
        raise ValueError("n_per_stratum must be > 0.")
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (int(row["extent_m"]), int(row["h_bin_index"]))
        grouped.setdefault(key, []).append(row)

    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: str(row.get("candidate_id", "")))
        count = min(int(n_per_stratum), len(group))
        indices = np.sort(rng.choice(len(group), size=count, replace=False))
        probability = float(count) / float(len(group))
        for index in indices:
            out = dict(group[int(index)])
            out["stratum_candidate_count"] = int(len(group))
            out["stratum_sample_count"] = int(count)
            out["sampling_probability"] = probability
            out["sampling_weight"] = 1.0 / probability
            selected.append(out)
    return selected


def eligible_catalog_rows_for_h_range(
    rows: Sequence[Mapping[str, object]],
    *,
    h_min_exclusive: float,
    h_max: float,
    bin_size: float,
) -> list[dict[str, object]]:
    """Reclassify an existing full catalog for a non-overlapping upper-H extension."""
    selected: list[dict[str, object]] = []
    for row in rows:
        try:
            land_pass = str(row.get("passes_land_filter", "")).strip().lower() == "true"
            valid_fraction = float(row.get("valid_fraction", "nan"))
            h = float(row.get("forest_cover", "nan"))
        except (TypeError, ValueError):
            continue
        if not land_pass or not math.isclose(valid_fraction, 1.0, abs_tol=1e-9):
            continue
        if not (float(h_min_exclusive) < h <= float(h_max)):
            continue
        h_bin = assign_h_bin(h, bin_size=float(bin_size), h_min=0.0, h_max=float(h_max))
        if h_bin is None:
            continue
        out = dict(row)
        out.update(
            {
                "passes_h_filter": True,
                "h_bin_index": int(h_bin.index),
                "h_bin_label": h_bin.label,
                "status": "eligible",
            }
        )
        selected.append(out)
    return selected


def inspect_raster(source: DatasetReader) -> dict[str, object]:
    return {
        "path": str(source.name),
        "driver": str(source.driver),
        "shape": [int(source.height), int(source.width)],
        "count": int(source.count),
        "dtypes": list(source.dtypes),
        "crs": None if source.crs is None else str(source.crs),
        "bounds": [float(value) for value in source.bounds],
        "resolution": [float(value) for value in source.res],
        "nodata": source.nodata,
        "mask_flags": [[flag.name for flag in flags] for flags in source.mask_flag_enums],
        "block_shapes": [[int(y), int(x)] for y, x in source.block_shapes],
        "compression": None if source.compression is None else str(source.compression),
        "overviews": [int(value) for value in source.overviews(1)],
        "tags": source.tags(),
    }


def evaluate_candidate(
    forest_source: DatasetReader,
    *,
    lon: float,
    lat: float,
    extent_m: int,
    cells: int,
    majority_threshold: float,
    h_max: float,
    bin_size: float,
    land_source: DatasetReader | None = None,
    gshhg_mask: GshhgLandMask | None = None,
    min_land_fraction: float = DEFAULT_MIN_LAND_FRACTION,
) -> tuple[CandidateResult, np.ndarray, np.ndarray, np.ndarray | None]:
    if land_source is not None and gshhg_mask is not None:
        raise ValueError("Use either land_source or gshhg_mask, not both.")
    started = time.perf_counter()
    forest_fractions = extract_binary_fraction(
        forest_source,
        lon=lon,
        lat=lat,
        extent_m=extent_m,
        cells=cells,
    )
    grid = majority_grid(forest_fractions, threshold=majority_threshold)
    h = forest_cover(grid)
    finite_forest = np.isfinite(forest_fractions)
    valid_fraction = float(np.mean(finite_forest))
    forest_fraction_mean = (
        float(np.mean(forest_fractions[finite_forest])) if np.any(finite_forest) else float("nan")
    )

    land_fractions: np.ndarray | None = None
    land_fraction: float | None = None
    land_valid_fraction: float | None = None
    center_is_land: bool | None = None
    land_pass: bool | None = None
    if land_source is not None:
        land_fractions = extract_binary_fraction(
            land_source,
            lon=lon,
            lat=lat,
            extent_m=extent_m,
            cells=cells,
        )
        finite_land = np.isfinite(land_fractions)
        land_valid_fraction = float(np.mean(finite_land))
        land_fraction = float(np.mean(land_fractions[finite_land])) if np.any(finite_land) else 0.0
        center_is_land = sample_binary_at_lonlat(land_source, lon=lon, lat=lat)
        land_pass = passes_candidate_land_rules(
            center_is_land=center_is_land,
            land_fraction=land_fraction,
            land_valid_fraction=land_valid_fraction,
            minimum=min_land_fraction,
        )
    elif gshhg_mask is not None:
        local_crs, _ = local_target_grid(lon, lat, extent_m, cells)
        evaluation = gshhg_mask.evaluate(
            lon=lon,
            lat=lat,
            extent_m=extent_m,
            local_crs=local_crs,
        )
        land_fraction = float(evaluation.land_fraction)
        land_valid_fraction = 1.0
        center_is_land = bool(evaluation.center_is_land)
        land_pass = passes_candidate_land_rules(
            center_is_land=center_is_land,
            land_fraction=land_fraction,
            land_valid_fraction=land_valid_fraction,
            minimum=min_land_fraction,
        )

    h_bin = assign_h_bin(h, bin_size=bin_size, h_max=h_max)
    result = CandidateResult(
        lon=float(lon),
        lat=float(lat),
        extent_m=int(extent_m),
        cells=int(cells),
        target_cell_m=float(extent_m) / float(cells),
        forest_cover=float(h),
        forest_fraction_mean=forest_fraction_mean,
        land_fraction=land_fraction,
        land_valid_fraction=land_valid_fraction,
        center_is_land=center_is_land,
        passes_land_filter=land_pass,
        passes_h_filter=h_bin is not None,
        h_bin_index=None if h_bin is None else int(h_bin.index),
        h_bin_label=None if h_bin is None else h_bin.label,
        valid_fraction=valid_fraction,
        elapsed_seconds=float(time.perf_counter() - started),
    )
    return result, grid, forest_fractions, land_fractions


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_csv_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    write_csv(temporary, rows)
    temporary.replace(path)


def append_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(dict(row) for row in rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def completed_center_ids(
    rows: Iterable[Mapping[str, object]],
    extents_m: Sequence[int],
) -> set[str]:
    expected = {int(value) for value in extents_m}
    found: dict[str, set[int]] = {}
    for row in rows:
        try:
            found.setdefault(str(row["center_id"]), set()).add(int(float(row["extent_m"])))
        except (KeyError, TypeError, ValueError):
            continue
    return {center_id for center_id, extents in found.items() if expected.issubset(extents)}


def classify_catalog_status(
    *,
    land_pass: bool,
    valid_fraction: float | None = None,
    forest_cover_value: float | None = None,
    h_bin: HBin | None = None,
    exclude_zero_h: bool = True,
) -> str:
    if not land_pass:
        return "rejected_land"
    if valid_fraction is None or not math.isclose(float(valid_fraction), 1.0, abs_tol=1e-9):
        return "rejected_incomplete_forest_data"
    if exclude_zero_h and (forest_cover_value is None or float(forest_cover_value) <= 0.0):
        return "rejected_h_zero"
    if h_bin is None:
        return "rejected_h"
    return "eligible"


def save_candidate(
    path: Path,
    *,
    result: CandidateResult,
    grid: np.ndarray,
    forest_fractions: np.ndarray,
    land_fractions: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "grid": np.asarray(grid, dtype=np.int8),
        "forest_fraction": np.asarray(forest_fractions, dtype=np.float32),
        "metadata_json": np.array(json.dumps(asdict(result), sort_keys=True)),
    }
    if land_fractions is not None:
        payload["land_fraction"] = np.asarray(land_fractions, dtype=np.float32)
    np.savez_compressed(path, **payload)


def save_selected_landscape(
    path: Path,
    *,
    grid: np.ndarray,
    forest_fractions: np.ndarray,
    metadata: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grid=np.asarray(grid, dtype=np.int8),
        forest_fraction=np.asarray(forest_fractions, dtype=np.float32),
        edge_depth=np.array(0, dtype=np.int32),
        metadata_json=np.array(json.dumps(dict(metadata), sort_keys=True)),
    )


def run_macrogrid_catalog(
    forest_source: DatasetReader,
    *,
    land_mask: GshhgLandMask,
    outdir: Path,
    extents_m: Sequence[int],
    cells: int,
    majority_threshold: float,
    h_max: float,
    bin_size: float,
    min_land_fraction: float,
    macrogrid_crs: str,
    macrogrid_size_m: float,
    seed: int,
    max_centers: int,
    progress_every: int,
    resume: bool,
    exclude_zero_h: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    centers_path = outdir / (
        "macrogrid_land_centers.csv" if int(max_centers) == 0 else "selected_pilot_centers.csv"
    )
    centers_metadata_path = outdir / "macrogrid_definition.json"
    requested_definition = {
        "source_crs": str(forest_source.crs),
        "source_bounds": [float(value) for value in forest_source.bounds],
        "macrogrid_crs": str(CRS.from_user_input(macrogrid_crs)),
        "macrogrid_size_m": float(macrogrid_size_m),
        "macrogrid_seed": int(seed),
        "max_centers": int(max_centers),
    }
    cached_definition: dict[str, object] = {}
    if resume and centers_path.exists() and centers_metadata_path.exists():
        cached_definition = json.loads(centers_metadata_path.read_text(encoding="utf-8"))

    if cached_definition.get("requested_definition") == requested_definition:
        centers = [
            {
                "center_id": row["center_id"],
                "macrogrid_index": int(float(row["macrogrid_index"])),
                "macrogrid_x": float(row["macrogrid_x"]),
                "macrogrid_y": float(row["macrogrid_y"]),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
            }
            for row in read_csv_rows(centers_path)
        ]
        grid_metadata = dict(cached_definition["grid_metadata"])
        print(f"[RESUME] loaded_cached_centers={len(centers)}")
    else:
        centers, grid_metadata = select_macrogrid_land_centers(
            source_crs=forest_source.crs,
            source_bounds=tuple(float(value) for value in forest_source.bounds),
            land_mask=land_mask,
            macrogrid_crs=macrogrid_crs,
            macrogrid_size_m=macrogrid_size_m,
            seed=seed,
            max_centers=max_centers,
        )
        write_csv_atomic(centers_path, centers)
        write_json_atomic(
            centers_metadata_path,
            {
                "requested_definition": requested_definition,
                "grid_metadata": grid_metadata,
            },
        )

    catalog_path = outdir / "candidate_catalog.csv"
    catalog_definition_path = outdir / "catalog_definition.json"
    catalog_definition = {
        "extents_m": [int(value) for value in extents_m],
        "cells": int(cells),
        "majority_threshold": float(majority_threshold),
        "h_max": float(h_max),
        "bin_size": float(bin_size),
        "min_land_fraction": float(min_land_fraction),
        "exclude_zero_h": bool(exclude_zero_h),
        "source_path": str(forest_source.name),
        "land_source": str(land_mask.source),
        "land_resolution": str(land_mask.resolution),
    }
    if catalog_path.exists() and resume:
        if not catalog_definition_path.exists():
            raise RuntimeError(
                f"Cannot safely resume catalog without {catalog_definition_path}."
            )
        existing_definition = json.loads(catalog_definition_path.read_text(encoding="utf-8"))
        if existing_definition != catalog_definition:
            raise RuntimeError(
                "Catalog settings differ from the existing resumable run. Use a new outdir."
            )
    elif not catalog_path.exists():
        write_json_atomic(catalog_definition_path, catalog_definition)
    if catalog_path.exists() and not resume:
        raise FileExistsError(f"Catalog already exists; use --resume or a new outdir: {catalog_path}")
    existing_rows = read_csv_rows(catalog_path) if resume else []
    rows_by_id: dict[str, dict[str, object]] = {
        str(row["candidate_id"]): dict(row) for row in existing_rows if row.get("candidate_id")
    }
    completed = completed_center_ids(rows_by_id.values(), extents_m)
    if completed:
        print(f"[RESUME] completed_centers={len(completed)} catalog_rows={len(rows_by_id)}")
    status_counts: Counter[str] = Counter(str(row.get("status", "")) for row in rows_by_id.values())
    processed_new = 0

    for center_number, center in enumerate(centers, start=1):
        if str(center["center_id"]) in completed:
            continue
        lon = float(center["lon"])
        lat = float(center["lat"])
        center_rows: list[dict[str, object]] = []
        for extent_m in extents_m:
            candidate_started = time.perf_counter()
            local_crs, _ = local_target_grid(lon, lat, extent_m, cells)
            land = land_mask.evaluate(
                lon=lon,
                lat=lat,
                extent_m=float(extent_m),
                local_crs=local_crs,
            )
            land_pass = passes_candidate_land_rules(
                center_is_land=land.center_is_land,
                land_fraction=land.land_fraction,
                land_valid_fraction=1.0,
                minimum=min_land_fraction,
            )
            row: dict[str, object] = {
                **center,
                "candidate_id": f"{center['center_id']}__{int(extent_m) // 1000:02d}km",
                "extent_m": int(extent_m),
                "cells": int(cells),
                "target_cell_m": float(extent_m) / float(cells),
                "center_is_land": bool(land.center_is_land),
                "land_fraction": float(land.land_fraction),
                "passes_land_filter": bool(land_pass),
                "forest_cover": "",
                "forest_fraction_mean": "",
                "valid_fraction": "",
                "passes_h_filter": "",
                "h_bin_index": "",
                "h_bin_label": "",
                "status": "rejected_land",
                "elapsed_seconds": "",
            }

            if land_pass:
                forest_fractions = extract_binary_fraction(
                    forest_source,
                    lon=lon,
                    lat=lat,
                    extent_m=extent_m,
                    cells=cells,
                )
                finite = np.isfinite(forest_fractions)
                valid_fraction = float(np.mean(finite))
                grid = majority_grid(forest_fractions, threshold=majority_threshold)
                h = forest_cover(grid)
                mean_fraction = (
                    float(np.mean(forest_fractions[finite])) if np.any(finite) else float("nan")
                )
                h_bin = assign_h_bin(h, bin_size=bin_size, h_max=h_max)
                status = classify_catalog_status(
                    land_pass=True,
                    valid_fraction=valid_fraction,
                    forest_cover_value=h,
                    h_bin=h_bin,
                    exclude_zero_h=exclude_zero_h,
                )
                row.update(
                    {
                        "forest_cover": float(h),
                        "forest_fraction_mean": mean_fraction,
                        "valid_fraction": valid_fraction,
                        "passes_h_filter": h_bin is not None,
                        "h_bin_index": "" if h_bin is None else int(h_bin.index),
                        "h_bin_label": "" if h_bin is None else h_bin.label,
                        "status": status,
                    }
                )

            row["elapsed_seconds"] = float(time.perf_counter() - candidate_started)
            center_rows.append(row)

        append_csv(catalog_path, center_rows)
        for row in center_rows:
            candidate_id = str(row["candidate_id"])
            old = rows_by_id.get(candidate_id)
            if old is not None:
                status_counts[str(old.get("status", ""))] -= 1
            rows_by_id[candidate_id] = row
            status_counts[str(row["status"])] += 1
        completed.add(str(center["center_id"]))
        processed_new += 1

        if int(progress_every) > 0 and (
            processed_new % int(progress_every) == 0 or len(completed) == len(centers)
        ):
            status_payload = {
                "completed_centers": int(len(completed)),
                "total_centers": int(len(centers)),
                "catalog_rows": int(len(rows_by_id)),
                "status_counts": {key: value for key, value in status_counts.items() if value},
                "elapsed_seconds_this_run": float(time.perf_counter() - started),
            }
            write_json_atomic(outdir / "run_status.json", status_payload)
            print(
                f"[MACROGRID] centers={len(completed)}/{len(centers)} "
                f"rows={len(rows_by_id)} statuses={status_payload['status_counts']} "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )

    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (int(float(row["macrogrid_index"])), int(float(row["extent_m"]))),
    )
    write_csv_atomic(catalog_path, rows)
    eligible_counts: Counter[str] = Counter()
    for row in rows:
        if row["status"] == "eligible":
            eligible_counts[f"{int(row['extent_m']) // 1000}km:{row['h_bin_label']}"] += 1
    summary = {
        **grid_metadata,
        "extents_m": [int(value) for value in extents_m],
        "candidate_rows": int(len(rows)),
        "status_counts": dict(status_counts),
        "exclude_zero_h": bool(exclude_zero_h),
        "eligible_by_extent_h_bin": dict(sorted(eligible_counts.items())),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(outdir / "macrogrid_summary.json", summary)
    return rows, summary


def export_stratified_landscapes(
    forest_source: DatasetReader,
    *,
    catalog_rows: Sequence[Mapping[str, object]],
    outdir: Path,
    n_per_stratum: int,
    selection_seed: int,
    majority_threshold: float,
    source_metadata: Mapping[str, object],
    land_metadata: Mapping[str, object],
    resume: bool,
    progress_every: int,
    landscape_subdir: str = "selected_landscapes",
    manifest_filename: str = "selected_landscapes.csv",
    summary_filename: str = "selection_summary.json",
) -> list[dict[str, object]]:
    eligible = [
        row
        for row in catalog_rows
        if str(row.get("status", "")) == "eligible"
        and str(row.get("h_bin_index", "")).strip() != ""
    ]
    selected = sample_fixed_per_stratum(
        eligible,
        n_per_stratum=int(n_per_stratum),
        seed=int(selection_seed),
    )
    manifest: list[dict[str, object]] = []
    for number, row in enumerate(selected, start=1):
        extent_m = int(float(row["extent_m"]))
        h_bin_index = int(float(row["h_bin_index"]))
        candidate_id = str(row["candidate_id"])
        relative_path = (
            Path(landscape_subdir)
            / f"{extent_m // 1000:02d}km"
            / f"hbin_{h_bin_index:02d}"
            / f"{candidate_id}.npz"
        )
        output_path = outdir / relative_path
        export_status = "existing"
        if not (resume and output_path.exists()):
            fractions = extract_binary_fraction(
                forest_source,
                lon=float(row["lon"]),
                lat=float(row["lat"]),
                extent_m=extent_m,
                cells=int(float(row["cells"])),
            )
            grid = majority_grid(fractions, threshold=majority_threshold)
            exported_h = forest_cover(grid)
            catalog_h = float(row["forest_cover"])
            if not math.isclose(exported_h, catalog_h, abs_tol=1e-12):
                raise RuntimeError(
                    f"Re-extracted H differs for {candidate_id}: catalog={catalog_h}, export={exported_h}."
                )
            local_crs, _ = local_target_grid(
                float(row["lon"]),
                float(row["lat"]),
                extent_m,
                int(float(row["cells"])),
            )
            object_metadata = {
                **dict(row),
                "format": "JRC_GFC2020_real_landscape",
                "format_version": 1,
                "local_crs": str(local_crs),
                "majority_threshold": float(majority_threshold),
                "selection_seed": int(selection_seed),
                "source_raster": dict(source_metadata),
                "land_mask": dict(land_metadata),
            }
            save_selected_landscape(
                output_path,
                grid=grid,
                forest_fractions=fractions,
                metadata=object_metadata,
            )
            export_status = "written"
        manifest.append(
            {
                **dict(row),
                "landscape_path": str(output_path),
                "export_status": export_status,
            }
        )
        if int(progress_every) > 0 and (
            number % int(progress_every) == 0 or number == len(selected)
        ):
            print(f"[EXPORT] landscapes={number}/{len(selected)}")

    write_csv_atomic(outdir / manifest_filename, manifest)
    selection_summary = {
        "n_catalog_eligible": int(len(eligible)),
        "n_selected": int(len(selected)),
        "n_per_stratum_requested": int(n_per_stratum),
        "selection_seed": int(selection_seed),
        "n_existing": sum(row["export_status"] == "existing" for row in manifest),
        "n_written": sum(row["export_status"] == "written" for row in manifest),
    }
    write_json_atomic(outdir / summary_filename, selection_summary)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect JRC GFC2020 and run a small local-projection extraction pilot."
    )
    parser.add_argument("--source", type=Path, required=True)
    land_group = parser.add_mutually_exclusive_group()
    land_group.add_argument("--land-mask", type=Path, default=None)
    land_group.add_argument(
        "--gshhg-root",
        type=Path,
        default=None,
        help="Extracted gshhg-shp root containing GSHHS_shp.",
    )
    parser.add_argument("--gshhg-resolution", choices=("f", "h", "i", "l", "c"), default="f")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/jrc_gfc2020_pilot"))
    parser.add_argument("--mode", choices=("point", "macrogrid"), default="point")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--pilot-lon", type=float, default=13.4050)
    parser.add_argument("--pilot-lat", type=float, default=52.5200)
    parser.add_argument("--extents-km", default=",".join(str(value) for value in DEFAULT_EXTENTS_KM))
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument("--majority-threshold", type=float, default=DEFAULT_MAJORITY_THRESHOLD)
    parser.add_argument("--h-max", type=float, default=DEFAULT_H_MAX)
    parser.add_argument("--bin-size", type=float, default=DEFAULT_BIN_SIZE)
    parser.add_argument("--min-land-fraction", type=float, default=DEFAULT_MIN_LAND_FRACTION)
    parser.add_argument("--macrogrid-crs", default="EPSG:6933")
    parser.add_argument("--macrogrid-km", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--max-centers",
        type=int,
        default=100,
        help="Macrogrid pilot limit; use 0 for every land center.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--exclude-zero-h",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=0,
        help="After a complete macrogrid catalog, export this many landscapes per extent and H bin.",
    )
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--export-progress-every", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.source) as forest_source:
        metadata = inspect_raster(forest_source)
        (args.outdir / "source_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        if args.metadata_only:
            return

        land_context = rasterio.open(args.land_mask) if args.land_mask is not None else None
        gshhg_mask = None
        if args.gshhg_root is not None:
            gshhg_mask = GshhgLandMask.load(
                args.gshhg_root,
                resolution=str(args.gshhg_resolution),
            )
            land_metadata = gshhg_mask.metadata()
            (args.outdir / "land_mask_metadata.json").write_text(
                json.dumps(land_metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(
                f"[LAND MASK] resolution={gshhg_mask.resolution} "
                f"polygons={sum(gshhg_mask.polygon_counts.values())} "
                f"load_seconds={gshhg_mask.load_seconds:.3f}"
            )
        try:
            extents_m = [
                int(round(float(value) * 1000.0))
                for value in parse_number_list(args.extents_km, cast=float)
            ]
            if args.mode == "macrogrid":
                if gshhg_mask is None:
                    raise ValueError("--mode macrogrid requires --gshhg-root.")
                _, summary = run_macrogrid_catalog(
                    forest_source,
                    land_mask=gshhg_mask,
                    outdir=args.outdir,
                    extents_m=extents_m,
                    cells=int(args.cells),
                    majority_threshold=float(args.majority_threshold),
                    h_max=float(args.h_max),
                    bin_size=float(args.bin_size),
                    min_land_fraction=float(args.min_land_fraction),
                    macrogrid_crs=str(args.macrogrid_crs),
                    macrogrid_size_m=float(args.macrogrid_km) * 1000.0,
                    seed=int(args.seed),
                    max_centers=int(args.max_centers),
                    progress_every=int(args.progress_every),
                    resume=bool(args.resume),
                    exclude_zero_h=bool(args.exclude_zero_h),
                )
                print(json.dumps(summary, indent=2, sort_keys=True))
                if int(args.samples_per_bin) > 0:
                    catalog_rows = read_csv_rows(args.outdir / "candidate_catalog.csv")
                    selection_seed = (
                        int(args.selection_seed)
                        if args.selection_seed is not None
                        else int(args.seed) + 2
                    )
                    export_stratified_landscapes(
                        forest_source,
                        catalog_rows=catalog_rows,
                        outdir=args.outdir,
                        n_per_stratum=int(args.samples_per_bin),
                        selection_seed=selection_seed,
                        majority_threshold=float(args.majority_threshold),
                        source_metadata=metadata,
                        land_metadata=gshhg_mask.metadata(),
                        resume=bool(args.resume),
                        progress_every=int(args.export_progress_every),
                    )
                return

            rows: list[dict[str, object]] = []
            for extent_m in extents_m:
                result, grid, forest_fractions, land_fractions = evaluate_candidate(
                    forest_source,
                    lon=float(args.pilot_lon),
                    lat=float(args.pilot_lat),
                    extent_m=extent_m,
                    cells=int(args.cells),
                    majority_threshold=float(args.majority_threshold),
                    h_max=float(args.h_max),
                    bin_size=float(args.bin_size),
                    land_source=land_context,
                    gshhg_mask=gshhg_mask,
                    min_land_fraction=float(args.min_land_fraction),
                )
                rows.append(asdict(result))
                filename = f"pilot_lon{args.pilot_lon:+.5f}_lat{args.pilot_lat:+.5f}_{extent_m // 1000:02d}km.npz"
                save_candidate(
                    args.outdir / "landscapes" / filename,
                    result=result,
                    grid=grid,
                    forest_fractions=forest_fractions,
                    land_fractions=land_fractions,
                )
                print(
                    f"[PILOT] extent_km={extent_m / 1000:g} H={result.forest_cover:.5f} "
                    f"land={result.land_fraction} elapsed={result.elapsed_seconds:.3f}s"
                )
            write_csv(args.outdir / "pilot_summary.csv", rows)
        finally:
            if land_context is not None:
                land_context.close()


if __name__ == "__main__":
    main()
