from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import shapefile
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coordinates
from shapely import clip_by_rect, points
from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree


LEVEL_SIGNS = {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0}


@dataclass(frozen=True)
class LandEvaluation:
    center_is_land: bool
    land_fraction: float
    queried_polygon_count: int


class GshhgLandMask:
    """Indexed GSHHG L1-L4 polygons for exact local land fractions.

    L1 land/ocean polygons add land, L2 lakes subtract it, L3 islands in
    lakes add it back, and L4 ponds subtract it again. Antarctica levels are
    intentionally omitted because JRC GFC2020 stops at 60 degrees south.
    """

    def __init__(
        self,
        geometries_by_level: Mapping[int, Iterable[BaseGeometry]],
        *,
        source: str = "",
        resolution: str = "f",
        load_seconds: float = 0.0,
    ) -> None:
        self.source = str(source)
        self.resolution = str(resolution)
        self.load_seconds = float(load_seconds)
        self.geometries_by_level: dict[int, list[BaseGeometry]] = {}
        self.trees_by_level: dict[int, STRtree] = {}
        for level in sorted(LEVEL_SIGNS):
            geometries = [geometry for geometry in geometries_by_level.get(level, ()) if not geometry.is_empty]
            self.geometries_by_level[level] = geometries
            self.trees_by_level[level] = STRtree(geometries)

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        resolution: str = "f",
        latitude_min: float = -60.5,
        latitude_max: float = 80.5,
    ) -> "GshhgLandMask":
        started = time.perf_counter()
        resolution = str(resolution).strip().lower()
        if resolution not in {"f", "h", "i", "l", "c"}:
            raise ValueError("resolution must be one of f, h, i, l, c.")
        root_path = Path(root)
        shoreline_dir = root_path / "GSHHS_shp" / resolution
        if not shoreline_dir.is_dir():
            raise FileNotFoundError(
                f"Expected GSHHG shoreline directory at {shoreline_dir}."
            )

        by_level: dict[int, list[BaseGeometry]] = {}
        for level in sorted(LEVEL_SIGNS):
            path = shoreline_dir / f"GSHHS_{resolution}_L{level}.shp"
            if not path.is_file():
                raise FileNotFoundError(path)
            geometries: list[BaseGeometry] = []
            with shapefile.Reader(str(path)) as reader:
                for item in reader.iterShapes():
                    if item.bbox[3] < float(latitude_min) or item.bbox[1] > float(latitude_max):
                        continue
                    geometry = shape(item.__geo_interface__)
                    if not geometry.is_empty:
                        geometries.append(geometry)
            by_level[level] = geometries

        return cls(
            by_level,
            source=str(root_path),
            resolution=resolution,
            load_seconds=time.perf_counter() - started,
        )

    @property
    def polygon_counts(self) -> dict[int, int]:
        return {level: len(geometries) for level, geometries in self.geometries_by_level.items()}

    def metadata(self) -> dict[str, object]:
        return {
            "type": "GSHHG_vector_land_mask",
            "source": self.source,
            "resolution": self.resolution,
            "levels": dict(LEVEL_SIGNS),
            "polygon_counts": self.polygon_counts,
            "load_seconds": self.load_seconds,
            "antarctica_levels_included": False,
        }

    def _indices_for_boxes(self, level: int, query_boxes: list[BaseGeometry]) -> list[int]:
        tree = self.trees_by_level[level]
        indices: set[int] = set()
        for query_box in query_boxes:
            indices.update(int(index) for index in tree.query(query_box))
        return sorted(indices)

    def _inside_level(self, level: int, point: Point) -> bool:
        return bool(self.trees_by_level[level].query(point, predicate="covered_by").size)

    def center_is_land(self, lon: float, lat: float) -> bool:
        point = Point(float(lon), float(lat))
        score = sum(
            LEVEL_SIGNS[level]
            for level in sorted(LEVEL_SIGNS)
            if self._inside_level(level, point)
        )
        return score > 0.0

    def centers_are_land(
        self,
        longitudes: Iterable[float],
        latitudes: Iterable[float],
    ) -> np.ndarray:
        lon = np.asarray(list(longitudes), dtype=float)
        lat = np.asarray(list(latitudes), dtype=float)
        if lon.shape != lat.shape:
            raise ValueError("longitudes and latitudes must have the same shape.")
        point_array = points(lon, lat)
        score = np.zeros(lon.shape, dtype=np.int8)
        for level, sign in LEVEL_SIGNS.items():
            matches = self.trees_by_level[level].query(point_array, predicate="covered_by")
            if matches.size:
                matched_points = np.unique(matches[0])
                score[matched_points] += int(sign)
        return score > 0

    @staticmethod
    def _query_boxes(
        *,
        lon: float,
        lat: float,
        extent_m: float,
        local_crs: CRS,
    ) -> list[BaseGeometry]:
        half = float(extent_m) / 2.0
        xs = [-half, half, half, -half]
        ys = [-half, -half, half, half]
        lons, lats = transform_coordinates(local_crs, "EPSG:4326", xs, ys)
        unwrapped_lons = [
            float(lon) + ((float(value) - float(lon) + 180.0) % 360.0 - 180.0)
            for value in lons
        ]
        xmin, xmax = min(unwrapped_lons), max(unwrapped_lons)
        ymin, ymax = min(lats), max(lats)
        if xmin < -180.0:
            return [box(-180.0, ymin, xmax, ymax), box(xmin + 360.0, ymin, 180.0, ymax)]
        if xmax > 180.0:
            return [box(xmin, ymin, 180.0, ymax), box(-180.0, ymin, xmax - 360.0, ymax)]
        return [box(xmin, ymin, xmax, ymax)]

    @staticmethod
    def _project_geometry(geometry: BaseGeometry, dst_crs: CRS) -> BaseGeometry:
        def project(xs, ys, zs=None):
            out_x, out_y = transform_coordinates("EPSG:4326", dst_crs, xs, ys)
            return out_x, out_y

        return transform_geometry(project, geometry)

    def evaluate(
        self,
        *,
        lon: float,
        lat: float,
        extent_m: float,
        local_crs: CRS,
    ) -> LandEvaluation:
        if float(extent_m) <= 0.0:
            raise ValueError("extent_m must be > 0.")
        half = float(extent_m) / 2.0
        footprint = box(-half, -half, half, half)
        query_boxes = self._query_boxes(
            lon=float(lon),
            lat=float(lat),
            extent_m=float(extent_m),
            local_crs=local_crs,
        )

        signed_area = 0.0
        queried = 0
        for level, sign in LEVEL_SIGNS.items():
            geometries = self.geometries_by_level[level]
            indices = self._indices_for_boxes(level, query_boxes)
            queried += len(indices)
            for index in indices:
                geometry = geometries[index]
                for query_box in query_boxes:
                    if not geometry.intersects(query_box):
                        continue
                    clipped = clip_by_rect(geometry, *query_box.bounds)
                    if clipped.is_empty:
                        continue
                    projected = self._project_geometry(clipped, local_crs)
                    if projected.intersects(footprint):
                        signed_area += float(sign) * float(projected.intersection(footprint).area)

        fraction = signed_area / float(extent_m * extent_m)
        fraction = min(1.0, max(0.0, fraction))
        return LandEvaluation(
            center_is_land=self.center_is_land(lon, lat),
            land_fraction=float(fraction),
            queried_polygon_count=int(queried),
        )
