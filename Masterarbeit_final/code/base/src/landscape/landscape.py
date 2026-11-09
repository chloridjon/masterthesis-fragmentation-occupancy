# src/landscape/landscape.py
# start directly with .\.venv\Scripts\python.exe -m src.landscape.landscape
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import json
import numpy as np
import matplotlib
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from numba import njit as _numba_njit
    from numba import prange as _numba_prange
except Exception:  # pragma: no cover - optional speed dependency
    _numba_njit = None
    _numba_prange = range

from src.landscape.boundary import (
    binary_erosion_with_boundary,
    distance_transform_edt_with_boundary,
    label_connected_components,
    normalize_boundary_mode,
)
from src.landscape.deforestation import (
    deforest_mask,
    reforest_mask,
    apply_deforestation,
    apply_reforestation,
)

FOREST = 1
NON_FOREST = 0


if _numba_njit is not None:
    @_numba_njit(cache=True, parallel=True)
    def _minimum_image_pairwise_numba(
        points_a: np.ndarray,
        points_b: np.ndarray,
        height: float,
        width: float,
    ) -> np.ndarray:
        n_a = points_a.shape[0]
        n_b = points_b.shape[0]
        out = np.empty((n_a, n_b), dtype=np.float64)
        for i in _numba_prange(n_a):
            ax = points_a[i, 0]
            ay = points_a[i, 1]
            for j in range(n_b):
                dx = ax - points_b[j, 0]
                if dx < 0.0:
                    dx = -dx
                wrap_dx = width - dx
                if wrap_dx < dx:
                    dx = wrap_dx

                dy = ay - points_b[j, 1]
                if dy < 0.0:
                    dy = -dy
                wrap_dy = height - dy
                if wrap_dy < dy:
                    dy = wrap_dy

                out[i, j] = (dx * dx + dy * dy) ** 0.5
        return out

    @_numba_njit(cache=True)
    def _selected_boundary_minima_numba(
        points: np.ndarray,
        offsets: np.ndarray,
        pairs: np.ndarray,
        height: int,
        width: int,
        periodic: bool,
    ) -> np.ndarray:
        """Exact boundary minima for selected patch pairs, without Python pair loops."""
        distances = np.empty(pairs.shape[0], dtype=np.float64)
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]
            best_squared = np.inf
            for a in range(offsets[i], offsets[i + 1]):
                ax, ay = points[a, 0], points[a, 1]
                for b in range(offsets[j], offsets[j + 1]):
                    dx = abs(ax - points[b, 0])
                    dy = abs(ay - points[b, 1])
                    if periodic:
                        dx = min(dx, width - dx)
                        dy = min(dy, height - dy)
                    squared = dx * dx + dy * dy
                    if squared < best_squared:
                        best_squared = squared
            distances[k] = np.sqrt(best_squared)
        return distances
else:
    _minimum_image_pairwise_numba = None
    _selected_boundary_minima_numba = None


@dataclass(frozen=True)
class ClusterInfo:
    """Parameters used to build core habitat and clusters (connected components)."""
    edge_depth: float
    connectivity: str  # "4" or "8"
    boundary_mode: str  # "periodic" | "open"
    habitat_threshold: float
    distance_mode: str  # "centroid" | "edge_min" | "edge_softmin" | "voronoi_dt"
    softmin_tau: float
    voronoi_depth: Optional[int] = 1  # None means all patch pairs


def _ndimage_connectivity(connectivity: str) -> int:
    """
    Map connectivity "4"/"8" to ndimage connectivity (2D):
      - "4" -> 1 (von Neumann)
      - "8" -> 2 (Moore)
    """
    if connectivity == "4":
        return 1
    if connectivity == "8":
        return 2
    raise ValueError(f"connectivity must be '4' or '8', got {connectivity!r}")


def _normalize_connectivity(connectivity: str | int) -> int:
    """
    Normalize connectivity to scipy.ndimage convention:
      1 -> 4-neighborhood, 2 -> 8-neighborhood (for 2D)
    Accepts: "4"/"8", 4/8, legacy 1/2.
    """
    if connectivity in (1, 2):
        return int(connectivity)
    if connectivity in ("4", 4):
        return 1
    if connectivity in ("8", 8):
        return 2
    raise ValueError(f"connectivity must be '4'/'8' or 4/8 or legacy 1/2, got {connectivity!r}")


def normalize_voronoi_depth(value: int | str | None) -> Optional[int]:
    """Normalize Voronoi hop depth; None/'all'/'inf' means every patch pair."""
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"all", "inf", "infinity", "none"}:
            return None
        value = int(token)
    depth = int(value)
    if depth < 1:
        raise ValueError("voronoi_depth must be >= 1, or 'all'.")
    return depth


def _softmin(distances: np.ndarray, tau: float) -> float:
    """
    Soft-min over a vector of distances (stable).
    Returns approximately min(distances) when tau -> 0.
    """
    d = np.asarray(distances, dtype=float)
    d0 = float(d.min())
    if tau <= 0.0:
        return d0
    # d0 - tau * log(mean(exp(-(d-d0)/tau)))
    return d0 - float(tau) * float(np.log(np.mean(np.exp(-(d - d0) / float(tau)))))


def _circular_mean(coords: np.ndarray, period: int) -> float:
    vals = np.asarray(coords, dtype=float)
    if vals.size == 0:
        return 0.0
    angles = (2.0 * np.pi * vals) / float(period)
    z = np.exp(1j * angles).mean()
    if abs(z) < 1e-12:
        return float(vals.mean())
    angle = float(np.angle(z))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return float(period * angle / (2.0 * np.pi))


def _minimum_image_pairwise(points_a: np.ndarray, points_b: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    width = float(shape[1])
    height = float(shape[0])
    if _minimum_image_pairwise_numba is not None:
        try:
            return _minimum_image_pairwise_numba(
                np.asarray(points_a, dtype=np.float64),
                np.asarray(points_b, dtype=np.float64),
                height,
                width,
            )
        except Exception:
            pass
    dx = np.abs(points_a[:, None, 0] - points_b[None, :, 0])
    dy = np.abs(points_a[:, None, 1] - points_b[None, :, 1])
    dx = np.minimum(dx, width - dx)
    dy = np.minimum(dy, height - dy)
    return np.sqrt(dx * dx + dy * dy)


def _cluster_centroids_from_labels(
    labels: np.ndarray,
    n_labels: int,
    boundary_mode: str,
) -> np.ndarray:
    if n_labels <= 0:
        return np.zeros((0, 2), dtype=float)

    mask = labels > 0
    ys, xs = np.nonzero(mask)
    labs = labels[mask].astype(np.int64, copy=False)
    counts = np.bincount(labs, minlength=n_labels + 1).astype(float)
    sum_x = np.bincount(labs, weights=xs.astype(float), minlength=n_labels + 1)
    sum_y = np.bincount(labs, weights=ys.astype(float), minlength=n_labels + 1)

    centers = np.zeros((n_labels, 2), dtype=float)
    valid = counts[1:] > 0
    if not np.any(valid):
        return centers

    if normalize_boundary_mode(boundary_mode) == "periodic":
        h, w = labels.shape
        angle_x = (2.0 * np.pi * xs.astype(float)) / float(w)
        angle_y = (2.0 * np.pi * ys.astype(float)) / float(h)
        cos_x = np.bincount(labs, weights=np.cos(angle_x), minlength=n_labels + 1)
        sin_x = np.bincount(labs, weights=np.sin(angle_x), minlength=n_labels + 1)
        cos_y = np.bincount(labs, weights=np.cos(angle_y), minlength=n_labels + 1)
        sin_y = np.bincount(labs, weights=np.sin(angle_y), minlength=n_labels + 1)

        radius_x = np.hypot(cos_x[1:], sin_x[1:]) / np.maximum(counts[1:], 1.0)
        radius_y = np.hypot(cos_y[1:], sin_y[1:]) / np.maximum(counts[1:], 1.0)
        angle_mean_x = np.mod(np.arctan2(sin_x[1:], cos_x[1:]), 2.0 * np.pi)
        angle_mean_y = np.mod(np.arctan2(sin_y[1:], cos_y[1:]), 2.0 * np.pi)
        centers[:, 0] = float(w) * angle_mean_x / (2.0 * np.pi)
        centers[:, 1] = float(h) * angle_mean_y / (2.0 * np.pi)

        fallback_x = radius_x < 1e-12
        fallback_y = radius_y < 1e-12
        centers[fallback_x, 0] = sum_x[1:][fallback_x] / counts[1:][fallback_x]
        centers[fallback_y, 1] = sum_y[1:][fallback_y] / counts[1:][fallback_y]
        return centers

    centers[valid, 0] = sum_x[1:][valid] / counts[1:][valid]
    centers[valid, 1] = sum_y[1:][valid] / counts[1:][valid]
    return centers


def _boundary_points(mask: np.ndarray, structure: np.ndarray, boundary_mode: str) -> np.ndarray:
    """
    Return boundary points (x, y) of a boolean region mask.
    Uses binary erosion to find interior; boundary = mask & ~eroded.
    """
    eroded = binary_erosion_with_boundary(mask, structure=structure, boundary_mode=boundary_mode)
    boundary = mask & (~eroded)
    coords_yx = np.argwhere(boundary)  # (k, 2) as (y, x)
    return coords_yx[:, ::-1].astype(float)  # -> (x, y)


def _pairwise_edge_distances(
    labels: np.ndarray,
    n_labels: int,
    structure: np.ndarray,
    mode: str,
    softmin_tau: float,
    boundary_mode: str,
) -> np.ndarray:
    """
    Pairwise distances between labeled regions based on boundary points.

    mode:
      - "edge_min": minimum boundary-to-boundary distance
      - "edge_softmin": symmetric soft-min using nearest-neighbor distances in both directions
    """
    points: list[np.ndarray] = []
    trees: list[Optional[cKDTree]] = []
    shape = labels.shape
    periodic = normalize_boundary_mode(boundary_mode) == "periodic"
    for lab in range(1, n_labels + 1):
        pts = _boundary_points(labels == lab, structure=structure, boundary_mode=boundary_mode)
        points.append(pts)
        if pts.shape[0] == 0:
            trees.append(None)
            continue
        if periodic:
            h, w = shape
            tiled = np.concatenate(
                [
                    pts + np.array([dx, dy], dtype=float)
                    for dy in (-h, 0, h)
                    for dx in (-w, 0, w)
                ],
                axis=0,
            )
            trees.append(cKDTree(tiled))
        else:
            trees.append(cKDTree(pts))

    dist = np.zeros((n_labels, n_labels), dtype=float)

    for i in range(n_labels):
        ti = trees[i]
        pi = points[i]
        for j in range(i + 1, n_labels):
            tj = trees[j]
            pj = points[j]

            if ti is None or tj is None or pi.size == 0 or pj.size == 0:
                d_ij = np.nan
            else:
                # Directed NN query: distances from each point in j to nearest point in i
                nn_j_to_i = ti.query(pj, k=1)[0]

                if mode == "edge_min":
                    d_ij = float(nn_j_to_i.min())

                elif mode == "edge_softmin":
                    # Symmetric soft-min: average in both directions
                    nn_i_to_j = tj.query(pi, k=1)[0]
                    d1 = _softmin(nn_j_to_i, softmin_tau)
                    d2 = _softmin(nn_i_to_j, softmin_tau)
                    d_ij = 0.5 * (float(d1) + float(d2))

                else:
                    raise ValueError(f"Unknown edge distance mode: {mode!r}")

            dist[i, j] = d_ij
            dist[j, i] = d_ij

    np.fill_diagonal(dist, 0.0)
    return dist


def _pairwise_voronoi_contact_distances(
    labels: np.ndarray,
    core_mask: np.ndarray,
    n_labels: int,
    include_diagonals: bool,
    boundary_mode: str,
) -> np.ndarray:
    """
    Estimate contact distances between clusters using a global Euclidean Distance Transform (EDT)
    + Voronoi-like nearest-cluster assignment.

    Returns a matrix with:
      - finite distances for cluster pairs that are Voronoi neighbors (share a boundary)
      - +inf for non-neighbor pairs
      - 0 on the diagonal

    Contact estimates identify Voronoi neighbours; selected pairs are then
    reweighted with exact boundary-to-boundary distances.

    Method
    ------
    1) Compute EDT on ~core_mask (matrix cells are True, core cells are False).
       EDT gives distance to the nearest core cell (distance=0 on core cells).
       With return_indices=True we also get the coordinates of that nearest core cell.
    2) Map each cell to the label of its nearest core cell -> nearest_labels (Voronoi assignment).
    3) Scan adjacency (4-neigh; optionally diagonals). Where adjacent cells have different nearest labels,
       we create a candidate pair distance:
           d(i,j) <= dist[p] + dist[q] + step
       and keep the minimum candidate per pair.

    This scales ~O(n_cells) and avoids explicit pairwise cluster loops.
    """
    if n_labels <= 1:
        out = np.full((n_labels, n_labels), np.inf, dtype=float)
        if n_labels == 1:
            out[0, 0] = 0.0
        return out

    dist, inds = distance_transform_edt_with_boundary(
        ~core_mask,
        boundary_mode=boundary_mode,
        return_indices=True,
    )
    nearest_labels = labels[tuple(inds)]  # (H,W), values in 0..n_labels

    out = np.full((n_labels, n_labels), np.inf, dtype=float)
    np.fill_diagonal(out, 0.0)

    def _update_from_neighbors(
        a_lab: np.ndarray,
        b_lab: np.ndarray,
        a_dist: np.ndarray,
        b_dist: np.ndarray,
        step: float,
    ) -> None:
        mask = (a_lab != b_lab) & (a_lab > 0) & (b_lab > 0)
        if not np.any(mask):
            return

        ii = (a_lab[mask] - 1).astype(np.int64)
        jj = (b_lab[mask] - 1).astype(np.int64)
        vals = (a_dist[mask] + b_dist[mask] + float(step)).astype(float)

        # symmetric minimum update
        np.minimum.at(out, (ii, jj), vals)
        np.minimum.at(out, (jj, ii), vals)

    mode = normalize_boundary_mode(boundary_mode)
    if mode == "periodic":
        _update_from_neighbors(
            nearest_labels, np.roll(nearest_labels, -1, axis=1),
            dist, np.roll(dist, -1, axis=1),
            step=1.0,
        )
        _update_from_neighbors(
            nearest_labels, np.roll(nearest_labels, -1, axis=0),
            dist, np.roll(dist, -1, axis=0),
            step=1.0,
        )
        if include_diagonals:
            s2 = float(np.sqrt(2.0))
            _update_from_neighbors(
                nearest_labels, np.roll(np.roll(nearest_labels, -1, axis=0), -1, axis=1),
                dist, np.roll(np.roll(dist, -1, axis=0), -1, axis=1),
                step=s2,
            )
            _update_from_neighbors(
                nearest_labels, np.roll(np.roll(nearest_labels, -1, axis=0), 1, axis=1),
                dist, np.roll(np.roll(dist, -1, axis=0), 1, axis=1),
                step=s2,
            )
        return out

    _update_from_neighbors(
        nearest_labels[:, :-1], nearest_labels[:, 1:],
        dist[:, :-1], dist[:, 1:],
        step=1.0,
    )
    _update_from_neighbors(
        nearest_labels[:-1, :], nearest_labels[1:, :],
        dist[:-1, :], dist[1:, :],
        step=1.0,
    )

    if include_diagonals:
        s2 = float(np.sqrt(2.0))
        _update_from_neighbors(
            nearest_labels[:-1, :-1], nearest_labels[1:, 1:],
            dist[:-1, :-1], dist[1:, 1:],
            step=s2,
        )
        _update_from_neighbors(
            nearest_labels[:-1, 1:], nearest_labels[1:, :-1],
            dist[:-1, 1:], dist[1:, :-1],
            step=s2,
        )

    return out


def _voronoi_graph_from_contact_matrix(contact: np.ndarray) -> sparse.csr_matrix:
    """Convert finite upper-triangle Voronoi contacts to a symmetric sparse graph."""
    matrix = np.asarray(contact, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Voronoi contact matrix must be square.")
    n = int(matrix.shape[0])
    upper = np.triu(np.isfinite(matrix), k=1)
    rows, cols = np.nonzero(upper)
    values = matrix[rows, cols]
    graph = sparse.coo_matrix(
        (
            np.concatenate([values, values]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n, n),
        dtype=float,
    )
    return graph.tocsr()


def _pairs_within_graph_depth(
    graph: sparse.spmatrix,
    depth: Optional[int],
) -> np.ndarray:
    """Unique node pairs whose unweighted graph distance is <= depth."""
    g = sparse.csr_matrix(graph)
    n = int(g.shape[0])
    if depth is None:
        rows, cols = np.triu_indices(n, k=1)
        return np.column_stack([rows, cols]).astype(np.int64, copy=False)
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    if int(depth) == 1:
        rows, cols = sparse.triu(g, k=1).nonzero()
        return np.column_stack((rows, cols)).astype(np.int64, copy=False)

    indptr = g.indptr
    indices = g.indices
    pairs: list[tuple[int, int]] = []
    for source in range(n):
        visited = {source}
        frontier = {source}
        for _ in range(int(depth)):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(int(x) for x in indices[indptr[node]:indptr[node + 1]])
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        pairs.extend((source, target) for target in visited if target > source)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _selected_pairwise_edge_min_distances(
    labels: np.ndarray,
    n_labels: int,
    structure: np.ndarray,
    boundary_mode: str,
    pairs: np.ndarray,
) -> np.ndarray:
    """Exact minimum boundary-to-boundary distances for selected cluster pairs."""
    selected = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if selected.size == 0:
        return np.zeros(0, dtype=float)

    # Find every labelled patch boundary in two raster passes, instead of
    # eroding the full raster once for each selected patch.
    periodic = normalize_boundary_mode(boundary_mode) == "periodic"
    filter_options = {"mode": "wrap"} if periodic else {"mode": "constant", "cval": 0}
    lowest = ndimage.minimum_filter(labels, footprint=structure, **filter_options)
    highest = ndimage.maximum_filter(labels, footprint=structure, **filter_options)
    boundary = (labels > 0) & ((lowest != labels) | (highest != labels))
    yy, xx = np.nonzero(boundary)
    boundary_labels = labels[yy, xx]
    order = np.argsort(boundary_labels, kind="stable")
    coordinates = np.column_stack((xx[order], yy[order]))
    counts = np.bincount(boundary_labels, minlength=n_labels + 1)[1:]
    offsets = np.empty(n_labels + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)

    if _selected_boundary_minima_numba is not None:
        return _selected_boundary_minima_numba(
            coordinates, offsets, selected, labels.shape[0], labels.shape[1], periodic,
        )

    needed = np.unique(selected)
    points: dict[int, np.ndarray] = {}
    trees: dict[int, cKDTree] = {}
    h, w = labels.shape
    for idx in needed:
        start, stop = offsets[int(idx)], offsets[int(idx) + 1]
        pts = coordinates[start:stop]
        points[int(idx)] = pts
        if periodic:
            tiled = np.concatenate(
                [
                    pts + np.array([dx, dy], dtype=float)
                    for dy in (-h, 0, h)
                    for dx in (-w, 0, w)
                ],
                axis=0,
            )
            trees[int(idx)] = cKDTree(tiled)
        else:
            trees[int(idx)] = cKDTree(pts)

    distances = np.empty(selected.shape[0], dtype=float)
    for k, (i, j) in enumerate(selected):
        distances[k] = float(trees[int(i)].query(points[int(j)], k=1)[0].min())
    return distances


class Landscape:
    """
    Holds the habitat grid and static landscape-related operations.
    No simulation loops, no time logic.

    Convention:
      - grid is 2D
      - 1 = forest, 0 = non-forest
    """

    def __init__(self, grid: np.ndarray, edge_depth: int = 0):
        if not isinstance(grid, np.ndarray):
            raise TypeError("grid must be a numpy array.")
        if grid.ndim != 2:
            raise ValueError("Landscape grid must be 2D.")
        if edge_depth < 0:
            raise ValueError("edge_depth must be >= 0.")

        self.grid = (grid > 0).astype(np.int8, copy=False)
        self.edge_depth = int(edge_depth)
        self._invalidate_cluster_cache()

    def _invalidate_cluster_cache(self) -> None:
        """Drop cached cluster/core attributes after any in-place grid mutation."""
        for attr_name in (
            "cluster_info",
            "core_mask",
            "core_cluster_labels",
            "n_clusters",
            "cluster_areas",
            "cluster_centroids",
            "cluster_protected_fraction",
            "cluster_dist_matrix",
            "voronoi_neighbor_graph",
            "voronoi_exact_pair_count",
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    # ----------------------------
    # Convenience properties
    # ----------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    @property
    def n_forest(self) -> int:
        return int((self.grid == FOREST).sum())

    @property
    def n_nonforest(self) -> int:
        return int((self.grid == NON_FOREST).sum())

    def copy(self) -> "Landscape":
        return Landscape(self.grid.copy(), edge_depth=self.edge_depth)

    # ----------------------------
    # Apply change (deforest / reforest)
    # ----------------------------
    def apply_change(
        self,
        kind: str,  # "deforest" | "reforest"
        mode: str,
        r: float,
        rng: np.random.Generator,
        selection_mode: str = "exact",
        inplace: bool = True,
        protected_mask: Optional[np.ndarray] = None,  # optional hook (not stored)
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Build & apply a change mask. Returns the applied mask.

        Notes
        -----
        - protected_mask is optional and NOT stored on the Landscape.
        - If you want protection to be used in cluster metrics, set self.protected_mask.
        """
        if protected_mask is not None and protected_mask.shape != self.grid.shape:
            raise ValueError("protected_mask must have same shape as grid.")

        if kind == "deforest":
            mask = deforest_mask(
                grid=self.grid,
                mode=mode,
                r=r,
                rng=rng,
                selection_mode=selection_mode,
                protected_mask=protected_mask,
                **kwargs,
            )
            if inplace:
                self.grid[np.asarray(mask, dtype=bool)] = NON_FOREST
                self._invalidate_cluster_cache()
            return mask

        if kind == "reforest":
            mask = reforest_mask(
                grid=self.grid,
                mode=mode,
                r=r,
                rng=rng,
                selection_mode=selection_mode,
                protected_mask=protected_mask,
                **kwargs,
            )
            if inplace:
                self.grid[np.asarray(mask, dtype=bool)] = FOREST
                self._invalidate_cluster_cache()
            return mask

        raise ValueError("kind must be 'deforest' or 'reforest'.")

    # ----------------------------
    # Small helpers for logging/debug
    # ----------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "edge_depth": self.edge_depth,
            "n_forest": self.n_forest,
            "n_nonforest": self.n_nonforest,
        }

    # ----------------------------
    # Core habitat + clusters (connected components)
    # ----------------------------
    def ensure_clusters(
        self,
        edge_depth: float = 0.0,
        connectivity: str | int = "4",
        boundary_mode: str = "open",
        habitat_threshold: float = 0.5,
        distance_mode: str = "voronoi_dt",   # canonical; centroid remains an explicit sensitivity mode
        softmin_tau: float = 1.0,          # only used for "edge_softmin"
        voronoi_depth: int | str | None = 1,
        rebuild: bool = False,
    ) -> None:
        """
        Compute core habitat and cluster-level attributes and store them on the Landscape.

        Stored attributes
        -----------------
        - self.core_mask: bool raster
        - self.core_cluster_labels: int raster (0=no core, 1..n cluster id)
        - self.n_clusters: int
        - self.cluster_areas: float (n,)
        - self.cluster_centroids: float (n, 2)
        - self.cluster_protected_fraction: float (n,) or None
        - self.cluster_dist_matrix: float (n, n)
        - self.voronoi_neighbor_graph: sparse symmetric edge-to-edge graph when distance_mode="voronoi_dt"

        distance_mode
        -------------
        - "centroid": euclidean distance between cluster centroids
        - "edge_min": minimum boundary-to-boundary distance (KDTree)
        - "edge_softmin": symmetric soft-min over boundary NN distances (KDTree), controlled by softmin_tau
        - "voronoi_dt": EDT/Voronoi contacts identify neighbours; exact
          edge-min distances weight direct neighbours, with centroid fallback
          for other pairs. Larger voronoi_depth values also replace pairs
          within that many direct Voronoi hops by exact edge-min distances.
          voronoi_depth=None/'all' computes exact edge-min distances for all pairs.
        """
        edge_depth = float(edge_depth)
        habitat_threshold = float(habitat_threshold)
        softmin_tau = float(softmin_tau)
        voronoi_depth = normalize_voronoi_depth(voronoi_depth)
        boundary_mode = normalize_boundary_mode(boundary_mode)

        conn_nd = _normalize_connectivity(connectivity)  # 1 (4-neigh) or 2 (8-neigh)
        conn_str = "4" if conn_nd == 1 else "8"

        if distance_mode not in ("centroid", "edge_min", "edge_softmin", "voronoi_dt"):
            raise ValueError("distance_mode must be 'centroid', 'edge_min', 'edge_softmin', or 'voronoi_dt'.")

        if distance_mode == "edge_softmin" and softmin_tau < 0.0:
            raise ValueError("softmin_tau must be >= 0.")

        info: Optional[ClusterInfo] = getattr(self, "cluster_info", None)
        if (not rebuild) and (info is not None):
            if (
                info.edge_depth == edge_depth
                and info.connectivity == conn_str
                and info.boundary_mode == boundary_mode
                and info.habitat_threshold == habitat_threshold
                and info.distance_mode == distance_mode
                and float(info.softmin_tau) == float(softmin_tau)
                and getattr(info, "voronoi_depth", 1) == voronoi_depth
            ):
                return

        habitat = self.grid > habitat_threshold

        dist_to_nonhabitat = distance_transform_edt_with_boundary(habitat, boundary_mode=boundary_mode)
        if edge_depth <= 0.0:
            core_mask = habitat
        else:
            core_mask = habitat & (dist_to_nonhabitat > edge_depth)

        struct = ndimage.generate_binary_structure(core_mask.ndim, conn_nd)
        labels, n_clusters = label_connected_components(
            core_mask,
            structure=struct,
            connectivity=conn_str,
            boundary_mode=boundary_mode,
        )

        if n_clusters == 0:
            raise ValueError("No core habitat clusters found. Reduce edge_depth or check habitat_threshold.")

        flat_labels = labels.ravel()
        cluster_areas = np.bincount(flat_labels)[1:].astype(float)

        cluster_centroids = _cluster_centroids_from_labels(labels, int(n_clusters), boundary_mode=boundary_mode)

        cluster_protected_fraction = None
        protected_mask = getattr(self, "protected_mask", None)
        if protected_mask is not None:
            prot = np.asarray(protected_mask, dtype=bool)
            if prot.shape != labels.shape:
                raise ValueError("protected_mask shape does not match Landscape.grid.")
            prot_counts = np.bincount(flat_labels, weights=prot.ravel().astype(float))[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                cluster_protected_fraction = np.where(cluster_areas > 0, prot_counts / cluster_areas, 0.0)

        # ----------------------------
        # Distance matrix
        # ----------------------------
        if distance_mode == "centroid":
            if boundary_mode == "periodic":
                cluster_dist_matrix = _minimum_image_pairwise(cluster_centroids, cluster_centroids, self.grid.shape)
            else:
                diffs = cluster_centroids[:, None, :] - cluster_centroids[None, :, :]
                cluster_dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=-1))
            np.fill_diagonal(cluster_dist_matrix, 0.0)

        elif distance_mode in ("edge_min", "edge_softmin"):
            cluster_dist_matrix = _pairwise_edge_distances(
                labels=labels,
                n_labels=int(n_clusters),
                structure=struct,
                mode=distance_mode,
                softmin_tau=softmin_tau,
                boundary_mode=boundary_mode,
            )

        elif distance_mode == "voronoi_dt":
            # Full centroid matrix as baseline (needed for all pairs)
            if boundary_mode == "periodic":
                centroid_dist = _minimum_image_pairwise(cluster_centroids, cluster_centroids, self.grid.shape)
            else:
                centroid_dist = cdist(cluster_centroids, cluster_centroids)
            np.fill_diagonal(centroid_dist, 0.0)

            # Contact distances for Voronoi neighbors (others remain +inf)
            contact = _pairwise_voronoi_contact_distances(
                labels=labels,
                core_mask=core_mask,
                n_labels=int(n_clusters),
                include_diagonals=(conn_nd == 2),
                boundary_mode=boundary_mode,
            )

            cluster_dist_matrix = centroid_dist
            mask = np.isfinite(contact)
            cluster_dist_matrix[mask] = contact[mask]
            np.fill_diagonal(cluster_dist_matrix, 0.0)
            voronoi_neighbor_graph = _voronoi_graph_from_contact_matrix(contact)
            exact_pairs = _pairs_within_graph_depth(voronoi_neighbor_graph, voronoi_depth)
            if exact_pairs.size:
                exact_values = _selected_pairwise_edge_min_distances(
                    labels=labels,
                    n_labels=int(n_clusters),
                    structure=struct,
                    boundary_mode=boundary_mode,
                    pairs=exact_pairs,
                )
                ii = exact_pairs[:, 0]
                jj = exact_pairs[:, 1]
                cluster_dist_matrix[ii, jj] = exact_values
                cluster_dist_matrix[jj, ii] = exact_values
            # Graph topology comes from the EDT contacts, but its stored edge
            # weights must agree with the exact values in the hybrid matrix.
            graph_coo = voronoi_neighbor_graph.tocoo()
            graph_coo.data = cluster_dist_matrix[graph_coo.row, graph_coo.col].copy()
            voronoi_neighbor_graph = graph_coo.tocsr()
            voronoi_exact_pair_count = int(exact_pairs.shape[0])

        else:
            raise ValueError(f"Unknown distance_mode: {distance_mode!r}")

        # Store cache
        self.cluster_info = ClusterInfo(
            edge_depth=edge_depth,
            connectivity=conn_str,
            boundary_mode=boundary_mode,
            habitat_threshold=habitat_threshold,
            distance_mode=distance_mode,
            softmin_tau=softmin_tau,
            voronoi_depth=voronoi_depth,
        )
        self.core_mask = core_mask.astype(bool)
        self.core_cluster_labels = labels.astype(np.int32)
        self.n_clusters = int(n_clusters)
        self.cluster_areas = cluster_areas
        self.cluster_centroids = cluster_centroids
        self.cluster_protected_fraction = cluster_protected_fraction
        self.cluster_dist_matrix = cluster_dist_matrix
        if distance_mode == "voronoi_dt":
            self.voronoi_neighbor_graph = voronoi_neighbor_graph
            self.voronoi_exact_pair_count = voronoi_exact_pair_count

    def ensure_voronoi_neighbor_graph(self, rebuild: bool = False) -> sparse.csr_matrix:
        """
        Return the sparse Voronoi-neighbor graph with edge-to-edge distances.

        Nodes are cluster indices 0..n_clusters-1. An undirected edge exists
        only where the EDT/Voronoi regions share a boundary.
        """
        graph = getattr(self, "voronoi_neighbor_graph", None)
        if not rebuild and sparse.issparse(graph):
            return graph.tocsr()
        if not all(hasattr(self, name) for name in ("core_cluster_labels", "core_mask", "n_clusters", "cluster_info")):
            raise ValueError("Landscape must be prepared with ensure_clusters(...) before building the Voronoi graph.")
        conn_nd = _normalize_connectivity(getattr(self.cluster_info, "connectivity", "4"))
        contact = _pairwise_voronoi_contact_distances(
            labels=np.asarray(self.core_cluster_labels),
            core_mask=np.asarray(self.core_mask, dtype=bool),
            n_labels=int(self.n_clusters),
            include_diagonals=(conn_nd == 2),
            boundary_mode=str(getattr(self.cluster_info, "boundary_mode", "open")),
        )
        graph = _voronoi_graph_from_contact_matrix(contact)
        pairs = _pairs_within_graph_depth(graph, 1)
        if pairs.size:
            structure = ndimage.generate_binary_structure(np.asarray(self.core_mask).ndim, conn_nd)
            values = _selected_pairwise_edge_min_distances(
                labels=np.asarray(self.core_cluster_labels),
                n_labels=int(self.n_clusters),
                structure=structure,
                boundary_mode=str(getattr(self.cluster_info, "boundary_mode", "open")),
                pairs=pairs,
            )
            rows, cols = pairs[:, 0], pairs[:, 1]
            graph = sparse.coo_matrix(
                (np.concatenate((values, values)),
                 (np.concatenate((rows, cols)), np.concatenate((cols, rows)))),
                shape=graph.shape,
            ).tocsr()
        self.voronoi_neighbor_graph = graph
        return self.voronoi_neighbor_graph

    # Backwards-compatible alias while refactoring other modules
    def ensure_patches(
        self,
        edge_depth: float = 0.0,
        connectivity: str | int = "4",
        boundary_mode: str = "open",
        habitat_threshold: float = 0.5,
        rebuild: bool = False,
    ) -> None:
        """
        Legacy name kept temporarily.
        Accepts old connectivity values 1/2 and maps them to "4"/"8".
        """
        if connectivity == 1:
            connectivity = "4"
        elif connectivity == 2:
            connectivity = "8"
        self.ensure_clusters(
            edge_depth=edge_depth,
            connectivity=str(connectivity),
            boundary_mode=boundary_mode,
            habitat_threshold=habitat_threshold,
            rebuild=rebuild,
        )

    # ----------------------------
    # Plot
    # ----------------------------
    @staticmethod
    def _rgba_from_mask(
        mask: np.ndarray,
        rgb: tuple[float, float, float] = (0.4, 0.7, 1.0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Build an RGBA image for an overlay from a boolean mask."""
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[..., 0] = rgb[0]
        rgba[..., 1] = rgb[1]
        rgba[..., 2] = rgb[2]
        rgba[..., 3] = mask.astype(float) * alpha
        return rgba

    def show(
        self,
        ax: Optional[plt.Axes] = None,
        im_landscape: Optional[plt.AxesImage] = None,
        im_overlay: Optional[plt.AxesImage] = None,
        overlay_mask: Optional[np.ndarray] = None,
        overlay_rgb: tuple[float, float, float] = (0.4, 0.7, 1.0),
        overlay_alpha: float = 0.4,
        title: Optional[str] = None,
        cmap: str = "Greens",
        vmin: int = 0,
        vmax: int = 1,
    ) -> tuple[plt.Axes, plt.AxesImage, Optional[plt.AxesImage]]:
        """
        Plot or update the landscape.

        Animation-safe:
          - images are created once
          - later calls only update image data
        """
        if ax is None:
            _, ax = plt.subplots()

        if im_landscape is None:
            im_landscape = ax.imshow(
                self.grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="upper",
            )
        else:
            im_landscape.set_data(self.grid)

        if overlay_mask is not None:
            if overlay_mask.shape != self.grid.shape:
                raise ValueError("overlay_mask must have same shape as grid.")
            rgba = self._rgba_from_mask(overlay_mask, rgb=overlay_rgb, alpha=overlay_alpha)

            if im_overlay is None:
                im_overlay = ax.imshow(rgba, interpolation="nearest", origin="upper")
            else:
                im_overlay.set_data(rgba)
        else:
            im_overlay = None

        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

        return ax, im_landscape, im_overlay

    # ----------------------------
    # Save / Load
    # ----------------------------
    def save(self, path: str | Path, include_cache: bool = True) -> None:
        """
        Save the Landscape to a compressed .npz file.

        Stores:
          - grid (int8), edge_depth (int32)
          - optional: protected_mask (uint8)
          - optional cache (if include_cache=True and present):
              cluster_info, core_mask, core_cluster_labels, n_clusters,
              cluster_areas, cluster_centroids, cluster_protected_fraction,
              cluster_dist_matrix and sparse Voronoi graph arrays
        """
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {
            "grid": self.grid.astype(np.int8, copy=False),
            "edge_depth": np.array(self.edge_depth, dtype=np.int32),
        }

        prot = getattr(self, "protected_mask", None)
        if prot is not None:
            prot = np.asarray(prot, dtype=bool)
            if prot.shape != self.grid.shape:
                raise ValueError("protected_mask shape does not match Landscape.grid.")
            payload["protected_mask"] = prot.astype(np.uint8)

        meta: dict[str, Any] = {
            "format": "LandscapeNPZ",
            "version": 5,
            "shape": list(self.grid.shape),
            "has_protected_mask": prot is not None,
            "has_cache": False,
        }

        info = getattr(self, "cluster_info", None)

        if include_cache:
            if info is not None:
                meta["cluster_info"] = {
                    "edge_depth": float(info.edge_depth),
                    "connectivity": str(info.connectivity),
                    "boundary_mode": str(info.boundary_mode),
                    "habitat_threshold": float(info.habitat_threshold),
                    "distance_mode": str(info.distance_mode),
                    "softmin_tau": float(info.softmin_tau),
                    "voronoi_depth": info.voronoi_depth,
                }

            def _store(attr_name: str) -> None:
                val = getattr(self, attr_name, None)
                if val is not None:
                    payload[attr_name] = np.asarray(val)

            _store("core_mask")
            _store("core_cluster_labels")
            _store("n_clusters")
            _store("cluster_areas")
            _store("cluster_centroids")
            _store("cluster_dist_matrix")
            _store("cluster_protected_fraction")
            _store("voronoi_exact_pair_count")
            graph = getattr(self, "voronoi_neighbor_graph", None)
            if sparse.issparse(graph):
                graph = graph.tocsr()
                payload["voronoi_graph_data"] = np.asarray(graph.data, dtype=float)
                payload["voronoi_graph_indices"] = np.asarray(graph.indices, dtype=np.int64)
                payload["voronoi_graph_indptr"] = np.asarray(graph.indptr, dtype=np.int64)
                payload["voronoi_graph_shape"] = np.asarray(graph.shape, dtype=np.int64)

            meta["has_cache"] = any(
                k in payload
                for k in (
                    "core_mask",
                    "core_cluster_labels",
                    "cluster_areas",
                    "cluster_centroids",
                    "cluster_dist_matrix",
                    "voronoi_graph_data",
                )
            )
        else:
            meta["has_cache"] = False

        payload["metadata_json"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Landscape":
        """Load a Landscape from a .npz created by Landscape.save()."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "grid" in data.files:
                grid = np.asarray(data["grid"], dtype=np.int8)
            else:
                grid = np.asarray(data["A"], dtype=np.int8)

            edge_depth = int(np.asarray(data["edge_depth"]))
            landscape = cls(grid=grid, edge_depth=edge_depth)

            meta: dict[str, Any] = {}
            if "metadata_json" in data.files:
                meta = json.loads(str(np.asarray(data["metadata_json"])))

            if "protected_mask" in data.files:
                landscape.protected_mask = np.asarray(data["protected_mask"], dtype=bool)

            info = meta.get("cluster_info") or meta.get("patch_info")
            if info is not None:
                conn = info.get("connectivity", "4")
                if conn == 1:
                    conn = "4"
                elif conn == 2:
                    conn = "8"
                distance_mode = info.get("distance_mode", "centroid")
                softmin_tau = float(info.get("softmin_tau", 1.0))
                voronoi_depth = normalize_voronoi_depth(info.get("voronoi_depth", 1))
                boundary_mode = info.get("boundary_mode", "open")

                landscape.cluster_info = ClusterInfo(
                    edge_depth=float(info["edge_depth"]),
                    connectivity=str(conn),
                    boundary_mode=str(boundary_mode),
                    habitat_threshold=float(info["habitat_threshold"]),
                    distance_mode=str(distance_mode),
                    softmin_tau=float(softmin_tau),
                    voronoi_depth=voronoi_depth,
                )

            def _restore(key: str) -> None:
                if key in data.files:
                    setattr(landscape, key, np.asarray(data[key]))

            _restore("core_mask")

            if "core_cluster_labels" in data.files:
                _restore("core_cluster_labels")
            elif "labels_core" in data.files:
                landscape.core_cluster_labels = np.asarray(data["labels_core"])

            if "n_clusters" in data.files:
                _restore("n_clusters")
            elif "n_patches" in data.files:
                landscape.n_clusters = int(np.asarray(data["n_patches"]))

            if "cluster_areas" in data.files:
                _restore("cluster_areas")
            elif "patch_areas" in data.files:
                landscape.cluster_areas = np.asarray(data["patch_areas"])

            if "cluster_centroids" in data.files:
                _restore("cluster_centroids")
            elif "patch_centroids" in data.files:
                landscape.cluster_centroids = np.asarray(data["patch_centroids"])

            if "cluster_dist_matrix" in data.files:
                _restore("cluster_dist_matrix")
            elif "patch_dist_matrix" in data.files:
                landscape.cluster_dist_matrix = np.asarray(data["patch_dist_matrix"])

            if "cluster_protected_fraction" in data.files:
                _restore("cluster_protected_fraction")
            elif "patch_protected_fraction" in data.files:
                landscape.cluster_protected_fraction = np.asarray(data["patch_protected_fraction"])

            graph_keys = {
                "voronoi_graph_data",
                "voronoi_graph_indices",
                "voronoi_graph_indptr",
                "voronoi_graph_shape",
            }
            if graph_keys.issubset(data.files):
                shape = tuple(int(x) for x in np.asarray(data["voronoi_graph_shape"]).tolist())
                landscape.voronoi_neighbor_graph = sparse.csr_matrix(
                    (
                        np.asarray(data["voronoi_graph_data"], dtype=float),
                        np.asarray(data["voronoi_graph_indices"], dtype=np.int64),
                        np.asarray(data["voronoi_graph_indptr"], dtype=np.int64),
                    ),
                    shape=shape,
                )
            if "voronoi_exact_pair_count" in data.files:
                landscape.voronoi_exact_pair_count = int(np.asarray(data["voronoi_exact_pair_count"]))

            return landscape


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    import numpy as _np
    import matplotlib.pyplot as _plt

    def _assert(cond: bool, msg: str) -> None:
        assert cond, msg

    def _assert_close(a: float, b: float, tol: float = 1e-9, msg: str = "") -> None:
        assert abs(float(a) - float(b)) <= tol, msg or f"Expected {b}, got {a}"

    def _count_forest(grid: _np.ndarray) -> int:
        return int((grid == 1).sum())

    # ------------------------------------------------------------
    # 1) Construction + basic properties
    # ------------------------------------------------------------
    raw = _np.array(
        [
            [2, 0, 0],
            [1, 1, -3],
            [0, 0, 1],
        ],
        dtype=float,
    )
    L = Landscape(raw, edge_depth=0)

    _assert(L.grid.dtype == _np.int8, "grid should be int8")
    _assert(L.grid.shape == (3, 3), "shape mismatch")
    _assert(_np.all((L.grid == 0) | (L.grid == 1)), "grid must be binary after init")
    _assert(L.n_forest == _count_forest(L.grid), "n_forest property mismatch")
    _assert(L.n_nonforest == int(L.grid.size - L.n_forest), "n_nonforest property mismatch")

    L2 = L.copy()
    _assert(_np.array_equal(L.grid, L2.grid), "copy should preserve grid")
    L2.grid[0, 0] = 0
    _assert(L.grid[0, 0] == 1, "copy must be deep (modifying copy must not affect original)")

    print("[OK] init + properties + copy")

    # ------------------------------------------------------------
    # 2) apply_change (deforest / reforest) deterministic behavior
    # ------------------------------------------------------------
    grid = _np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )
    L = Landscape(grid.copy())
    rng = _np.random.default_rng(123)

    n0 = L.n_forest
    r_def = 0.25  # exact: round(n_forest * r)
    mask_def = L.apply_change(
        kind="deforest",
        mode="random",
        r=r_def,
        rng=rng,
        selection_mode="exact",
        inplace=True,
    )

    expected_pick = int(_np.round(n0 * r_def))
    _assert(mask_def.dtype == bool, "deforest mask must be boolean")
    _assert(int(mask_def.sum()) == expected_pick, "deforest exact pick count mismatch for random mode")
    _assert(L.n_forest == n0 - expected_pick, "deforestation did not change forest count as expected")
    _assert(_np.all(L.grid[mask_def] == 0), "deforested cells must be non-forest")

    grid_before = L.grid.copy()
    rng2 = _np.random.default_rng(123)
    mask_noop = L.apply_change(
        kind="deforest",
        mode="random",
        r=0.10,
        rng=rng2,
        selection_mode="exact",
        inplace=False,
    )
    _assert(_np.array_equal(L.grid, grid_before), "inplace=False must not modify grid")
    _assert(mask_noop.shape == L.grid.shape, "mask shape mismatch")

    print("[OK] apply_change deforest (exact) + inplace behavior")

    rng3 = _np.random.default_rng(999)
    n_non0 = L.n_nonforest
    r_ref = 0.20
    mask_ref = L.apply_change(
        kind="reforest",
        mode="ref_edge_random",
        r=r_ref,
        rng=rng3,
        selection_mode="exact",
        inplace=True,
        connectivity="4",
    )
    _assert(mask_ref.dtype == bool, "reforest mask must be boolean")
    _assert(int(mask_ref.sum()) <= int(_np.round(n_non0 * r_ref)), "reforest picked more than target non-forest fraction")
    _assert(_np.all(L.grid[mask_ref] == 1), "reforested cells must be forest")

    print("[OK] apply_change reforest (exact)")

    # ------------------------------------------------------------
    # 3) ensure_clusters connectivity sanity + distance modes
    # ------------------------------------------------------------
    diag = _np.array([[1, 0],
                      [0, 1]], dtype=int)

    L4 = Landscape(diag)
    L4.ensure_clusters(edge_depth=0.0, connectivity="4", habitat_threshold=0.5, rebuild=True)
    _assert(L4.n_clusters == 2, "diag should have 2 clusters under 4-neighborhood")
    _assert_close(float(L4.cluster_areas.sum()), 2.0, msg="cluster_areas should sum to total core cells")
    _assert(L4.core_mask.dtype == bool, "core_mask must be boolean")
    _assert(L4.core_cluster_labels.dtype == _np.int32, "core_cluster_labels must be int32")
    _assert(L4.cluster_dist_matrix.shape == (2, 2), "dist matrix shape mismatch")
    _assert_close(float(_np.diag(L4.cluster_dist_matrix).sum()), 0.0, msg="dist matrix diagonal must be 0")

    L4.ensure_clusters(edge_depth=0.0, connectivity="4", habitat_threshold=0.5, distance_mode="edge_min", rebuild=True)
    _assert(_np.all(_np.isfinite(L4.cluster_dist_matrix)), "edge_min distances must be finite")
    _assert(_np.all(L4.cluster_dist_matrix >= 0.0), "distances must be non-negative")

    L4.ensure_clusters(edge_depth=0.0, connectivity="4", habitat_threshold=0.5, distance_mode="edge_softmin", softmin_tau=1.0, rebuild=True)
    _assert(_np.all(_np.isfinite(L4.cluster_dist_matrix)), "edge_softmin distances must be finite")
    _assert(_np.all(L4.cluster_dist_matrix >= 0.0), "softmin distances must be non-negative")

    L4.ensure_clusters(edge_depth=0.0, connectivity="4", habitat_threshold=0.5, distance_mode="voronoi_dt", rebuild=True)
    _assert(_np.all(_np.isfinite(L4.cluster_dist_matrix)), "voronoi_dt distances must be finite (centroid fallback)")
    _assert(_np.all(L4.cluster_dist_matrix >= 0.0), "voronoi_dt distances must be non-negative")
    _assert_close(float(_np.diag(L4.cluster_dist_matrix).sum()), 0.0, msg="voronoi_dt diagonal must be 0")
    _assert(sparse.isspmatrix_csr(L4.voronoi_neighbor_graph), "Voronoi graph must be CSR")
    _assert(L4.voronoi_neighbor_graph.shape == (2, 2), "Voronoi graph shape mismatch")
    _assert((L4.voronoi_neighbor_graph != L4.voronoi_neighbor_graph.T).nnz == 0, "Voronoi graph must be symmetric")

    depth_grid = _np.zeros((7, 15), dtype=int)
    depth_grid[1:3, 1:3] = 1
    depth_grid[4:6, 5:7] = 1
    depth_grid[1:3, 9:11] = 1
    depth_grid[4:6, 13:15] = 1
    L_depth1 = Landscape(depth_grid)
    L_depth1.ensure_clusters(
        connectivity="4",
        boundary_mode="open",
        distance_mode="voronoi_dt",
        voronoi_depth=1,
        rebuild=True,
    )
    L_depth2 = Landscape(depth_grid)
    L_depth2.ensure_clusters(
        connectivity="4",
        boundary_mode="open",
        distance_mode="voronoi_dt",
        voronoi_depth=2,
        rebuild=True,
    )
    L_depth_all = Landscape(depth_grid)
    L_depth_all.ensure_clusters(
        connectivity="4",
        boundary_mode="open",
        distance_mode="voronoi_dt",
        voronoi_depth="all",
        rebuild=True,
    )
    L_edge_exact = Landscape(depth_grid)
    L_edge_exact.ensure_clusters(
        connectivity="4",
        boundary_mode="open",
        distance_mode="edge_min",
        rebuild=True,
    )
    depth2_pairs = _pairs_within_graph_depth(L_depth1.voronoi_neighbor_graph, 2)
    _assert(depth2_pairs.size > 0, "depth-2 Voronoi expansion should contain pairs")
    for i, j in depth2_pairs:
        _assert_close(
            float(L_depth2.cluster_dist_matrix[i, j]),
            float(L_edge_exact.cluster_dist_matrix[i, j]),
            msg="depth-2 distance must match exact edge-min distance",
        )
    _assert(
        _np.allclose(L_depth_all.cluster_dist_matrix, L_edge_exact.cluster_dist_matrix),
        "voronoi_depth='all' must match the full edge_min distance matrix",
    )
    _assert(
        L_depth1.voronoi_exact_pair_count <= L_depth2.voronoi_exact_pair_count
        <= L_depth_all.voronoi_exact_pair_count,
        "Voronoi resolved pair count must increase monotonically with depth",
    )
    _assert(
        L_depth_all.voronoi_exact_pair_count == L_depth_all.n_clusters * (L_depth_all.n_clusters - 1) // 2,
        "voronoi_depth='all' must resolve every unique patch pair",
    )
    _assert(L_depth1.cluster_info.voronoi_depth == 1, "default Voronoi depth changed")
    print("[OK] Voronoi depth expansion")

    L8 = Landscape(diag)
    L8.ensure_clusters(edge_depth=0.0, connectivity="8", habitat_threshold=0.5, rebuild=True)
    _assert(L8.n_clusters == 1, "diag should have 1 cluster under 8-neighborhood")
    _assert_close(float(L8.cluster_areas.sum()), 2.0, msg="cluster_areas should sum to total core cells")

    print("[OK] ensure_clusters connectivity + distance modes")

    cache_grid = _np.array([[1, 1, 0],
                            [0, 0, 0],
                            [0, 1, 1]], dtype=int)
    Lcache = Landscape(cache_grid)
    Lcache.ensure_clusters(edge_depth=0.0, connectivity="4", boundary_mode="open", habitat_threshold=0.5, rebuild=True)
    _assert(Lcache.n_clusters == 2, "cache test should start with 2 clusters")
    Lcache.apply_change(
        kind="deforest",
        mode="random",
        r=0.25,
        rng=_np.random.default_rng(0),
        selection_mode="exact",
        inplace=True,
    )
    _assert(not hasattr(Lcache, "cluster_info"), "apply_change must invalidate cached cluster info")
    Lcache.ensure_clusters(edge_depth=0.0, connectivity="4", boundary_mode="open", habitat_threshold=0.5, rebuild=False)
    _assert(Lcache.n_clusters >= 1, "cache invalidation recompute must yield valid cluster count")
    print("[OK] apply_change invalidates cluster cache")

    # ------------------------------------------------------------
    # 4) protected fraction plausibility
    # ------------------------------------------------------------
    grid_pf = _np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    Lp = Landscape(grid_pf)
    Lp.protected_mask = _np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    Lp.ensure_clusters(edge_depth=0.0, connectivity="4", habitat_threshold=0.5, rebuild=True)
    pf = Lp.cluster_protected_fraction
    _assert(pf is not None, "cluster_protected_fraction should be computed when protected_mask exists")
    _assert(pf.shape == (Lp.n_clusters,), "protected fraction length mismatch")
    _assert(_np.all((pf >= 0.0) & (pf <= 1.0)), "protected fractions must be in [0, 1]")

    print("[OK] protected fraction per cluster")

    # ------------------------------------------------------------
    # 5) save/load roundtrip (with cache)
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "landscape_test.npz"
        Lp.save(p, include_cache=True)
        L_loaded = Landscape.load(p)

        _assert(_np.array_equal(L_loaded.grid, Lp.grid), "loaded grid mismatch")
        _assert(L_loaded.edge_depth == Lp.edge_depth, "loaded edge_depth mismatch")

        if hasattr(L_loaded, "core_mask"):
            _assert(L_loaded.core_mask.shape == L_loaded.grid.shape, "loaded core_mask shape mismatch")
        if hasattr(L_loaded, "core_cluster_labels"):
            _assert(L_loaded.core_cluster_labels.shape == L_loaded.grid.shape, "loaded labels shape mismatch")

        graph_path = Path(tmp) / "landscape_graph_test.npz"
        L4.save(graph_path, include_cache=True)
        graph_loaded = Landscape.load(graph_path)
        _assert(sparse.isspmatrix_csr(graph_loaded.voronoi_neighbor_graph), "loaded Voronoi graph must be CSR")
        _assert(
            (graph_loaded.voronoi_neighbor_graph != L4.voronoi_neighbor_graph).nnz == 0,
            "loaded Voronoi graph mismatch",
        )

    print("[OK] save/load roundtrip")

    # ------------------------------------------------------------
    # 6) show() smoke test (no GUI popups)
    # ------------------------------------------------------------
    fig, ax = _plt.subplots()
    ax, im_base, im_overlay = Lp.show(ax=ax, title="Landscape test")
    _assert(im_base is not None, "base image handle missing")
    ax, im_base2, _ = Lp.show(ax=ax, im_landscape=im_base, im_overlay=im_overlay, overlay_mask=Lp.protected_mask)
    _assert(im_base2 is im_base, "show should reuse base image handle when provided")
    _plt.close(fig)

    print("[OK] show() smoke test")
    print("\nAll Landscape tests passed.")
