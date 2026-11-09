# src/landscape/landscape_stats.py
# start directly with .\.venv\Scripts\python.exe -m src.landscape.landscape_stats
from __future__ import annotations
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
try:
    from numba import njit as _numba_njit
    from numba import prange as _numba_prange
except Exception:  # pragma: no cover - optional speed dependency
    _numba_njit = None
    _numba_prange = range

from src.landscape.boundary import (
    label_connected_components,
    normalize_boundary_mode,
)

FOREST = 1
NON_FOREST = 0

ALL_LANDSCAPE_METRIC_KEYS: tuple[str, ...] = (
    "forest_cover",
    "n_clusters",
    "mean_cluster_size",
    "mean_cluster_size_excluding_largest",
    "mean_cluster_size_excluding_percolating",
    "var_cluster_size",
    "largest_cluster_size",
    "largest_cluster_fraction_forest",
    "largest_cluster_fraction_total",
    "largest_cluster_fraction",
    "landscape_division_index",
    "cluster_size_distribution",
    "percolates_top_bottom",
    "percolates_left_right",
    "largest_cluster_shape",
    "mean_nearest_neighbor_distance",
    "mean_centroid_nearest_neighbor_distance",
    "nearest_neighbor_distance_distribution",
    "mean_full_occupancy_Si",
    "chi",
    "S2_size",
    "S2_fraction_forest",
    "S2_fraction_total",
    "aggregation_index",
    "percentage_of_like_adjacencies",
    "edge_density_total",
    "edge_density_forest",
    "cohesion_index",
    "edge_density",
    "complete_mst_n_edges",
    "complete_mst_total_length",
    "complete_mst_mean_edge_length",
    "complete_mst_max_edge_length",
    "complete_mst_d20",
    "complete_mst_d50",
    "complete_mst_d80",
)

JSON_LANDSCAPE_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "cluster_size_distribution",
        "nearest_neighbor_distance_distribution",
    }
)

_CLUSTER_METRIC_KEYS = {
    "n_clusters",
    "mean_cluster_size",
    "mean_cluster_size_excluding_largest",
    "mean_cluster_size_excluding_percolating",
    "var_cluster_size",
    "largest_cluster_size",
    "largest_cluster_fraction_forest",
    "largest_cluster_fraction_total",
    "largest_cluster_fraction",
    "landscape_division_index",
    "cluster_size_distribution",
    "percolates_top_bottom",
    "percolates_left_right",
    "largest_cluster_shape",
    "mean_nearest_neighbor_distance",
    "mean_centroid_nearest_neighbor_distance",
    "nearest_neighbor_distance_distribution",
    "mean_full_occupancy_Si",
    "chi",
    "S2_size",
    "S2_fraction_forest",
    "S2_fraction_total",
}

_SIZE_BASED_METRIC_KEYS = {
    "mean_cluster_size",
    "mean_cluster_size_excluding_largest",
    "mean_cluster_size_excluding_percolating",
    "var_cluster_size",
    "largest_cluster_size",
    "largest_cluster_fraction_forest",
    "largest_cluster_fraction_total",
    "largest_cluster_fraction",
    "landscape_division_index",
    "cluster_size_distribution",
    "chi",
    "S2_size",
    "S2_fraction_forest",
    "S2_fraction_total",
}

_PERCO_FLAG_KEYS = {"percolates_top_bottom", "percolates_left_right"}
_PERCO_EXCLUSION_KEYS = {"mean_cluster_size_excluding_percolating"}
_ADJACENCY_METRIC_KEYS = {
    "aggregation_index",
    "percentage_of_like_adjacencies",
    "edge_density_total",
    "edge_density_forest",
    "cohesion_index",
    "edge_density",
}
_COMPLETE_MST_METRIC_KEYS = {
    "complete_mst_n_edges",
    "complete_mst_total_length",
    "complete_mst_mean_edge_length",
    "complete_mst_max_edge_length",
    "complete_mst_d20",
    "complete_mst_d50",
    "complete_mst_d80",
}
_CLUSTER_METRIC_KEYS |= _COMPLETE_MST_METRIC_KEYS
SI_DISTANCE_MODES: tuple[str, ...] = ("voronoi_dt", "centroid", "edge_min", "edge_softmin")


if _numba_njit is not None:
    @_numba_njit(cache=True, parallel=True)
    def _minimum_image_distances_numba(
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


    @_numba_njit(cache=True, parallel=True)
    def _mean_full_occupancy_si_numba(
        sizes: np.ndarray,
        dist: np.ndarray,
        alpha: float,
        b: float,
    ) -> float:
        n = sizes.shape[0]
        if n <= 1:
            return 0.0

        sources = np.empty(n, dtype=np.float64)
        rows = np.empty(n, dtype=np.float64)
        for i in _numba_prange(n):
            sources[i] = sizes[i] ** b

        for i in _numba_prange(n):
            row_sum = 0.0
            for j in range(n):
                if i != j:
                    row_sum += np.exp(-alpha * dist[i, j]) * sources[j]
            rows[i] = row_sum

        total = 0.0
        for i in range(n):
            total += rows[i]
        return total / float(n)
else:
    _minimum_image_distances_numba = None
    _mean_full_occupancy_si_numba = None


# ============================================================
# Connectivity / labeling
# ============================================================

def _normalize_connectivity(connectivity: str | int) -> str:
    """Accept '4'/'8' or legacy 4/8 ints and normalize to '4'/'8'."""
    if connectivity in ("4", "8"):
        return str(connectivity)
    if connectivity in (4, 8):
        return str(int(connectivity))
    raise ValueError(f"connectivity must be '4' or '8' (or 4/8), got {connectivity!r}")


def connectivity_structure(connectivity: str | int = "4") -> np.ndarray:
    """
    Return the binary structure for connected-component labeling.

    Notes
    -----
    For scipy.ndimage.label, the structure typically includes the center cell (=1).
    """
    conn = _normalize_connectivity(connectivity)
    if conn == "4":
        return np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=int,
        )
    # conn == "8"
    return np.ones((3, 3), dtype=int)


def label_forest_clusters(
    grid: np.ndarray,
    connectivity: str | int = "4",
    boundary_mode: str = "open",
) -> tuple[np.ndarray, int]:
    """Label forest clusters (connected components) in a binary grid."""
    struct = connectivity_structure(connectivity)
    cluster_labels, n_clusters = label_connected_components(
        grid == FOREST,
        structure=struct,
        connectivity=_normalize_connectivity(connectivity),
        boundary_mode=boundary_mode,
    )
    return cluster_labels, int(n_clusters)


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


def _minimum_image_distances(
    points_a: np.ndarray,
    points_b: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    width = float(shape[1])
    height = float(shape[0])
    if _minimum_image_distances_numba is not None:
        try:
            return _minimum_image_distances_numba(
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


def _mask_perimeter(mask: np.ndarray, boundary_mode: str = "open") -> float:
    mode = normalize_boundary_mode(boundary_mode)
    m = np.asarray(mask, dtype=bool)
    if mode == "periodic":
        right = m & (~np.roll(m, -1, axis=1))
        down = m & (~np.roll(m, -1, axis=0))
        left = m & (~np.roll(m, 1, axis=1))
        up = m & (~np.roll(m, 1, axis=0))
        return float(right.sum() + down.sum() + left.sum() + up.sum())

    right = m[:, :-1] & (~m[:, 1:])
    left = m[:, 1:] & (~m[:, :-1])
    down = m[:-1, :] & (~m[1:, :])
    up = m[1:, :] & (~m[:-1, :])
    border = float(m[0, :].sum() + m[-1, :].sum() + m[:, 0].sum() + m[:, -1].sum())
    return float(right.sum() + left.sum() + down.sum() + up.sum()) + border


# ============================================================
# Basic metrics
# ============================================================

def forest_cover(grid: np.ndarray) -> float:
    """Fraction of cells that are forest (value = 1)."""
    total = int(grid.size)
    if total == 0:
        return 0.0
    return float((grid == FOREST).sum()) / float(total)


def cluster_sizes(cluster_labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Sizes (cell counts) for labels 1..n_clusters (background label 0 ignored)."""
    if n_clusters <= 0:
        return np.zeros(0, dtype=int)
    return np.bincount(cluster_labels.ravel())[1:n_clusters + 1]


def _sizes_excluding_one_largest(sizes: np.ndarray) -> np.ndarray:
    """
    Remove exactly one largest cluster entry.

    This keeps tied-largest clusters except for a single representative, which is
    the conventional interpretation of "excluding the largest cluster".
    """
    s = np.asarray(sizes)
    if s.size <= 1:
        return np.zeros(0, dtype=s.dtype)
    return np.delete(s, int(np.argmax(s)))


def _chi_and_second_largest_from_sizes(sizes: np.ndarray) -> tuple[float, int]:
    """Exclude exactly one largest patch, independently of percolation."""
    remaining = _sizes_excluding_one_largest(sizes)
    if remaining.size == 0:
        return np.nan, 0
    values = remaining.astype(float)
    return float(np.dot(values, values) / values.sum()), int(remaining.max())

def susceptibility_chi(sizes: np.ndarray, exclude_largest: bool = True) -> float:
    """
    Susceptibility chi (finite clusters).
    Common definition: chi = sum(s^2) / sum(s), optionally excluding the largest cluster.

    Notes
    -----
    - Excluding the largest cluster is the standard "remove giant component" trick.
    - Returns NaN if not computable (no clusters or denominator 0).
    """
    s = np.asarray(sizes, dtype=float)
    if s.size == 0:
        return np.nan

    if exclude_largest:
        if s.size <= 1:
            return np.nan
        # drop the largest cluster (one entry)
        s = np.delete(s, int(np.argmax(s)))

    denom = float(s.sum())
    if denom <= 0.0:
        return np.nan

    return float((s * s).sum() / denom)


def second_largest_cluster_size(sizes: np.ndarray) -> int:
    """
    Size (cell count) of the 2nd largest cluster.
    Returns 0 if fewer than 2 clusters exist.
    """
    s = np.asarray(sizes, dtype=int)
    if s.size < 2:
        return 0
    # fast without full sort
    return int(np.partition(s, -2)[-2])


def second_largest_cluster_fractions(
    sizes: np.ndarray,
    total_cells: int,
) -> dict[str, float]:
    """
    Convenience: S2 normalized by total raster cells and by forest cells.

    Returns
    -------
    dict with:
      - S2_size
      - S2_total_fraction
      - S2_forest_fraction
    """
    s = np.asarray(sizes, dtype=int)
    s2 = second_largest_cluster_size(s)

    forest_cells = int(s.sum())
    total_cells = int(total_cells)

    s2_total = (float(s2) / float(total_cells)) if total_cells > 0 else 0.0
    s2_forest = (float(s2) / float(forest_cells)) if forest_cells > 0 else 0.0

    return {
        "S2_size": float(s2),
        "S2_total_fraction": float(s2_total),
        "S2_forest_fraction": float(s2_forest),
    }


def landscape_division_index(sizes: np.ndarray) -> float:
    """
    Landscape division index based on cluster sizes.
    Returns 0 for degenerate cases (<=1 total forest cell).
    """
    total_forest = int(sizes.sum())
    if total_forest <= 1:
        return 0.0
    num = float((sizes * (sizes - 1)).sum())
    den = float(total_forest * (total_forest - 1))
    return 1.0 - (num / den)


# ============================================================
# Nearest-neighbor distances between cluster centroids
# ============================================================

def cluster_centroids(
    cluster_labels: np.ndarray,
    n_clusters: int,
    boundary_mode: str = "open",
) -> np.ndarray:
    """
    Centroids for labels 1..n_clusters as (x, y) coordinates.

    Implementation note:
    - We compute unweighted centroids with bincount aggregations over all labeled cells.
    """
    mode = normalize_boundary_mode(boundary_mode)
    if n_clusters <= 0:
        return np.zeros((0, 2), dtype=float)

    mask = cluster_labels > 0
    ys, xs = np.nonzero(mask)
    labs = cluster_labels[mask].astype(np.int64, copy=False)
    counts = np.bincount(labs, minlength=n_clusters + 1).astype(float)
    sum_x = np.bincount(labs, weights=xs.astype(float), minlength=n_clusters + 1)
    sum_y = np.bincount(labs, weights=ys.astype(float), minlength=n_clusters + 1)

    centers = np.zeros((n_clusters, 2), dtype=float)
    valid = counts[1:] > 0
    if not np.any(valid):
        return centers

    if mode == "periodic":
        h, w = cluster_labels.shape
        angle_x = (2.0 * np.pi * xs.astype(float)) / float(w)
        angle_y = (2.0 * np.pi * ys.astype(float)) / float(h)
        cos_x = np.bincount(labs, weights=np.cos(angle_x), minlength=n_clusters + 1)
        sin_x = np.bincount(labs, weights=np.sin(angle_x), minlength=n_clusters + 1)
        cos_y = np.bincount(labs, weights=np.cos(angle_y), minlength=n_clusters + 1)
        sin_y = np.bincount(labs, weights=np.sin(angle_y), minlength=n_clusters + 1)

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


def nearest_neighbor_distances(
    cluster_labels: np.ndarray,
    n_clusters: int,
    boundary_mode: str = "open",
) -> np.ndarray:
    """
    Nearest-neighbor distance for each cluster centroid.

    Returns
    -------
    nn_dist : np.ndarray
        Array of length n_clusters+1 with nn_dist[label] = nearest neighbor distance.
        Index 0 is background and kept as NaN.
    """
    nn_dist = np.full(n_clusters + 1, np.nan, dtype=float)
    if n_clusters <= 1:
        return nn_dist

    centers = cluster_centroids(cluster_labels, n_clusters, boundary_mode=boundary_mode)
    nn_values = nearest_neighbor_distances_from_centroids(
        centers,
        shape=cluster_labels.shape,
        boundary_mode=boundary_mode,
    )
    nn_dist[1:] = nn_values
    return nn_dist


def nearest_neighbor_distances_from_centroids(
    centers: np.ndarray,
    *,
    shape: tuple[int, int],
    boundary_mode: str = "open",
) -> np.ndarray:
    """Nearest-neighbor distances for centroid coordinates, returned as length n."""
    centers = np.asarray(centers, dtype=float)
    n_clusters = int(centers.shape[0])
    nn = np.full(n_clusters, np.nan, dtype=float)
    if n_clusters <= 1:
        return nn

    valid = np.isfinite(centers).all(axis=1)
    if int(valid.sum()) < 2:
        return nn

    pts = centers[valid]
    if normalize_boundary_mode(boundary_mode) == "periodic":
        h, w = shape
        pts_mod = pts.copy()
        pts_mod[:, 0] = np.mod(pts_mod[:, 0], float(w))
        pts_mod[:, 1] = np.mod(pts_mod[:, 1], float(h))
        tree = cKDTree(pts_mod, boxsize=np.array([float(w), float(h)], dtype=float))
        distances, _ = tree.query(pts_mod, k=2)
        vals = distances[:, 1]
    else:
        tree = cKDTree(pts)
        distances, _ = tree.query(pts, k=2)  # k=2: first is self (0), second is NN
        vals = distances[:, 1]

    nn[np.flatnonzero(valid)] = vals
    return nn


def nearest_neighbor_distances_from_matrix(distances: np.ndarray) -> np.ndarray:
    """Nearest finite off-diagonal distance for every patch in a square matrix."""
    matrix = np.asarray(distances, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    if matrix.shape[0] <= 1:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    work = matrix.copy()
    np.fill_diagonal(work, np.inf)
    nearest = np.min(work, axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    return nearest


def nearest_neighbor_distances_from_voronoi_graph(graph: sparse.spmatrix) -> np.ndarray:
    """Exact neighbour-edge minimum per patch; absent edges are not distances."""
    if not sparse.issparse(graph):
        raise TypeError("Voronoi neighbour graph must be sparse so absent edges stay absent.")
    edges = graph.tocsr()
    if edges.shape[0] != edges.shape[1]:
        raise ValueError("Voronoi neighbour graph must be square.")
    nearest = np.full(edges.shape[0], np.nan, dtype=float)
    for i in range(edges.shape[0]):
        start, stop = edges.indptr[i], edges.indptr[i + 1]
        columns = edges.indices[start:stop]
        weights = edges.data[start:stop]
        weights = weights[columns != i]
        if weights.size:
            if not np.all(np.isfinite(weights) & (weights > 0)):
                raise ValueError("Voronoi neighbour distances must be finite and positive.")
            nearest[i] = float(weights.min())
    if edges.shape[0] >= 2 and not np.all(np.isfinite(nearest)):
        raise ValueError("Every patch needs a direct Voronoi neighbour when at least two patches exist.")
    return nearest


def mean_centroid_nearest_neighbor_distance(
    cluster_labels: np.ndarray,
    n_clusters: int,
    boundary_mode: str = "open",
) -> float:
    """Explicit centroid-to-centroid sensitivity metric."""
    nn = nearest_neighbor_distances(cluster_labels, n_clusters, boundary_mode=boundary_mode)
    vals = nn[1:]  # drop background
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(vals.mean())


def mean_nearest_neighbor_distance_from_values(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(vals.mean())


def nearest_neighbor_distance_distribution(
    cluster_labels: np.ndarray,
    n_clusters: int,
    boundary_mode: str = "open",
    n_bins: int = 20,
) -> dict[str, list[float] | list[int]]:
    """
    Histogram-style frequency distribution of nearest-neighbor distances.

    Returns
    -------
    dict with:
      - "bin_edges": histogram bin edges
      - "counts": counts per bin
    """
    nn = nearest_neighbor_distances(cluster_labels, n_clusters, boundary_mode=boundary_mode)
    vals = nn[1:]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"bin_edges": [], "counts": []}

    n_bins = max(1, int(n_bins))
    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        delta = 0.5 if np.isclose(vmin, 0.0) else max(1e-9, abs(vmin) * 1e-6)
        edges = np.array([vmin - delta, vmax + delta], dtype=float)
        counts = np.array([int(vals.size)], dtype=int)
        return {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
        }

    counts, edges = np.histogram(vals, bins=n_bins, range=(vmin, vmax))
    return {
        "bin_edges": edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
    }


def nearest_neighbor_distance_distribution_from_values(
    values: np.ndarray,
    n_bins: int = 20,
) -> dict[str, list[float] | list[int]]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"bin_edges": [], "counts": []}

    n_bins = max(1, int(n_bins))
    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        delta = 0.5 if np.isclose(vmin, 0.0) else max(1e-9, abs(vmin) * 1e-6)
        edges = np.array([vmin - delta, vmax + delta], dtype=float)
        counts = np.array([int(vals.size)], dtype=int)
        return {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
        }

    counts, edges = np.histogram(vals, bins=n_bins, range=(vmin, vmax))
    return {
        "bin_edges": edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
    }


def mean_full_occupancy_Si(
    grid: np.ndarray,
    *,
    connectivity: str | int = "4",
    boundary_mode: str = "open",
    alpha: float = 0.15,
    b: float = 1.0,
    distance_mode: str = "voronoi_dt",
    softmin_tau: float = 1.0,
    voronoi_depth: int | str | None = 1,
) -> float:
    """
    Mean metapopulation connectivity S_i over all clusters for a fully occupied landscape.

    Uses
      S_i = sum_{j != i} exp(-alpha * d_ij) * A_j^b
    with p_j = 1 for all clusters.
    """
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("grid must be 2D for mean_full_occupancy_Si.")
    if distance_mode not in SI_DISTANCE_MODES:
        raise ValueError(
            f"distance_mode must be one of {SI_DISTANCE_MODES!r}, got {distance_mode!r}."
        )
    alpha = float(alpha)
    b = float(b)
    if alpha < 0.0:
        raise ValueError("alpha must be >= 0 for mean_full_occupancy_Si.")
    if not np.any(g > 0):
        return 0.0
    from src.landscape.landscape import Landscape

    L = Landscape((g > 0).astype(np.int8, copy=False))
    L.ensure_clusters(
        edge_depth=0.0,
        connectivity=connectivity,
        boundary_mode=boundary_mode,
        habitat_threshold=0.5,
        distance_mode=distance_mode,
        softmin_tau=softmin_tau,
        voronoi_depth=voronoi_depth,
        rebuild=True,
    )
    sizes = np.asarray(L.cluster_areas, dtype=float)
    if sizes.size <= 1:
        return 0.0
    dist = np.asarray(L.cluster_dist_matrix, dtype=float)
    K = np.exp(-alpha * dist)
    np.fill_diagonal(K, 0.0)
    sources = sizes ** b
    S = K @ sources
    if S.size == 0:
        return 0.0
    return float(np.mean(S))


def mean_full_occupancy_Si_from_clusters(
    cluster_labels: np.ndarray,
    sizes: np.ndarray,
    *,
    boundary_mode: str = "open",
    alpha: float = 0.15,
    b: float = 1.0,
) -> float:
    """
    Mean fully occupied S_i from already computed cluster labels and sizes.

    This is equivalent to mean_full_occupancy_Si(..., distance_mode="centroid"),
    but avoids rebuilding a Landscape and relabeling the raster.
    """
    s = np.asarray(sizes, dtype=float)
    n_clusters = int(s.size)
    if n_clusters <= 1:
        return 0.0
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be >= 0 for mean_full_occupancy_Si.")

    centers = cluster_centroids(cluster_labels, n_clusters, boundary_mode=boundary_mode)
    if normalize_boundary_mode(boundary_mode) == "periodic":
        dist = _minimum_image_distances(centers, centers, cluster_labels.shape)
    else:
        diffs = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diffs * diffs, axis=-1))

    K = np.exp(-alpha * dist)
    np.fill_diagonal(K, 0.0)
    S = K @ (s ** float(b))
    if S.size == 0:
        return 0.0
    return float(np.mean(S))


def mean_full_occupancy_Si_from_distances(
    sizes: np.ndarray,
    dist: np.ndarray,
    *,
    alpha: float = 0.15,
    b: float = 1.0,
) -> float:
    """Mean fully occupied S_i from precomputed cluster sizes and distances."""
    s = np.asarray(sizes, dtype=float)
    d = np.asarray(dist, dtype=float)
    if s.size <= 1:
        return 0.0
    if d.shape != (s.size, s.size):
        raise ValueError("cluster distance matrix has wrong shape for mean_full_occupancy_Si.")
    alpha = float(alpha)
    b = float(b)
    if alpha < 0.0:
        raise ValueError("alpha must be >= 0 for mean_full_occupancy_Si.")
    if _mean_full_occupancy_si_numba is not None:
        try:
            return float(
                _mean_full_occupancy_si_numba(
                    np.asarray(s, dtype=np.float64),
                    np.asarray(d, dtype=np.float64),
                    alpha,
                    b,
                )
            )
        except Exception:
            pass
    K = np.exp(-alpha * d)
    np.fill_diagonal(K, 0.0)
    S = K @ (s ** b)
    if S.size == 0:
        return 0.0
    return float(np.mean(S))


def nearest_neighbor_index(
    cluster_labels: np.ndarray,
    n_clusters: int,
    cellsize: float = 1.0,
    boundary_mode: str = "open",
) -> float:
    """
    Nearest Neighbor Index (NNI) for cluster centroids.

    NNI = observed mean NN distance / expected mean NN distance under CSR.
    """
    r_obs_cells = mean_centroid_nearest_neighbor_distance(
        cluster_labels, n_clusters, boundary_mode=boundary_mode
    )
    if not np.isfinite(r_obs_cells):
        return np.nan

    nn = nearest_neighbor_distances(cluster_labels, n_clusters, boundary_mode=boundary_mode)
    valid_nn = nn[1:]
    valid_nn = valid_nn[np.isfinite(valid_nn)]
    n_used = int(valid_nn.size)

    if n_used < 2:
        return np.nan

    n_rows, n_cols = cluster_labels.shape
    area = float(n_rows * n_cols) * float(cellsize ** 2)

    r_exp = 0.5 * np.sqrt(area / float(n_used))
    r_obs = float(r_obs_cells) * float(cellsize)
    return r_obs / float(r_exp)


# ============================================================
# Adjacency metrics (currently 4-neighborhood)
# ============================================================

def aggregation_index(adj_forest_forest: float, n_forest: int) -> float:
    """
    Aggregation Index (AI). Guarded for tiny n_forest to avoid div-by-zero noise.
    """
    if n_forest <= 1:
        return np.nan
    denom = (2 * n_forest - 2 * np.sqrt(n_forest))
    if denom <= 0:
        return np.nan
    return float(adj_forest_forest) / float(denom)


def cohesion_index(boundary: float, n_forest: int) -> float:
    if n_forest <= 0:
        return 0.0
    return 1.0 - float(boundary) / float(boundary + 2.0 * np.sqrt(n_forest))


def compute_adjacency_metrics(
    grid: np.ndarray,
    connectivity: str | int = "4",
    boundary_mode: str = "open",
) -> dict[str, float]:
    """
    Compute adjacency-based fragmentation metrics.

    Notes
    -----
    Only 4-neighborhood is implemented here.
    """
    conn = _normalize_connectivity(connectivity)
    if conn != "4":
        raise ValueError("Only 4-neighborhood ('4') is implemented at the moment.")

    g = (grid > 0).astype(bool)
    mode = normalize_boundary_mode(boundary_mode)

    if mode == "periodic":
        right = np.roll(g, -1, axis=1)
        down = np.roll(g, -1, axis=0)
        adj_forest_forest = float((g & right).sum() + (g & down).sum())
        adj_forest_nonforest = float((g != right).sum() + (g != down).sum())
    else:
        horizontal_ff = g[1:, :] & g[:-1, :]
        vertical_ff = g[:, 1:] & g[:, :-1]
        adj_forest_forest = float(horizontal_ff.sum() + vertical_ff.sum())

        horizontal_fn = (g[:, 1:] != g[:, :-1])
        vertical_fn = (g[1:, :] != g[:-1, :])
        adj_forest_nonforest = float(horizontal_fn.sum() + vertical_fn.sum())

    n_forest = int(g.sum())
    ai = aggregation_index(adj_forest_forest, n_forest)

    pladj = np.nan
    if (adj_forest_forest + adj_forest_nonforest) > 0:
        pladj = float(adj_forest_forest) / float(adj_forest_forest + adj_forest_nonforest)

    edge_density_total = (
        0.0
        if g.size == 0
        else float(adj_forest_nonforest) / float(g.size)
    )
    edge_density_forest = np.nan if n_forest == 0 else float(adj_forest_nonforest) / float(n_forest)

    ci = cohesion_index(adj_forest_nonforest, n_forest)

    return {
        "aggregation_index": float(ai) if np.isfinite(ai) else np.nan,
        "percentage_of_like_adjacencies": float(pladj) if np.isfinite(pladj) else np.nan,
        "edge_density_total": float(edge_density_total),
        "edge_density_forest": float(edge_density_forest) if np.isfinite(edge_density_forest) else np.nan,
        "cohesion_index": float(ci),
        # Backwards-compatible alias:
        "edge_density": float(edge_density_total),
    }


def _complete_graph_mst_edges_from_distances(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return MST edges for a complete graph represented by a dense distance matrix.

    This uses a simple O(n^2) Prim implementation to avoid materializing all
    n*(n-1)/2 complete-graph edges as a separate sparse matrix.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix.")

    n = int(d.shape[0])
    if n <= 1:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=float),
        )

    work = np.array(d, dtype=float, copy=True)
    work = np.minimum(work, work.T)
    np.fill_diagonal(work, np.inf)
    work[~np.isfinite(work)] = np.inf

    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    min_dist = work[0].copy()
    parent = np.zeros(n, dtype=np.int64)

    rows: list[int] = []
    cols: list[int] = []
    lengths: list[float] = []

    for _ in range(n - 1):
        candidate_dist = np.where(selected, np.inf, min_dist)
        node = int(np.argmin(candidate_dist))
        length = float(candidate_dist[node])
        if not np.isfinite(length):
            break

        rows.append(int(parent[node]))
        cols.append(node)
        lengths.append(length)
        selected[node] = True

        improve = (~selected) & (work[node] < min_dist)
        parent[improve] = node
        min_dist[improve] = work[node, improve]

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(lengths, dtype=float),
    )


def _critical_area_thresholds_from_mst(
    rows: np.ndarray,
    cols: np.ndarray,
    lengths: np.ndarray,
    areas: np.ndarray,
    thresholds: tuple[float, ...] = (0.2, 0.5),
) -> dict[float, float]:
    """Area-weighted critical dispersal thresholds from MST edge activation."""
    a = np.asarray(areas, dtype=float).reshape(-1)
    n = int(a.size)
    total_area = float(a.sum())
    result = {float(q): float("nan") for q in thresholds}
    if n == 0 or total_area <= 0.0:
        return result

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)
    component_area = a.astype(float, copy=True)
    max_area = float(component_area.max()) if n else 0.0

    for q in thresholds:
        if max_area / total_area >= float(q):
            result[float(q)] = 0.0

    def find(node: int) -> int:
        root = int(node)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(left: int, right: int) -> float:
        root_l = find(left)
        root_r = find(right)
        if root_l == root_r:
            return float(component_area[root_l])
        if rank[root_l] < rank[root_r]:
            root_l, root_r = root_r, root_l
        parent[root_r] = root_l
        component_area[root_l] += component_area[root_r]
        if rank[root_l] == rank[root_r]:
            rank[root_l] += 1
        return float(component_area[root_l])

    order = np.argsort(np.asarray(lengths, dtype=float), kind="mergesort")
    for edge_idx in order:
        distance = float(lengths[int(edge_idx)])
        merged_area = union(int(rows[int(edge_idx)]), int(cols[int(edge_idx)]))
        if merged_area > max_area:
            max_area = merged_area
        fraction = max_area / total_area
        for q in thresholds:
            qf = float(q)
            if np.isnan(result[qf]) and fraction >= qf:
                result[qf] = distance

    return result


def compute_complete_mst_metrics_from_distances(
    distances: np.ndarray,
    areas: np.ndarray,
) -> dict[str, float | int]:
    """
    MST metrics on the complete patch graph defined by the distance matrix.

    Edges are weighted by patch edge-to-edge/selected-mode distances. D20,
    D50, and D80 are the first MST edge distances at which the largest connected
    component contains at least 20%, 50%, or 80% of total habitat area.
    """
    a = np.asarray(areas, dtype=float).reshape(-1)
    d = np.asarray(distances, dtype=float)
    if d.shape != (a.size, a.size):
        raise ValueError("Distance matrix shape does not match cluster areas.")

    rows, cols, lengths = _complete_graph_mst_edges_from_distances(d)
    n_edges = int(lengths.size)
    thresholds = _critical_area_thresholds_from_mst(rows, cols, lengths, a, thresholds=(0.2, 0.5, 0.8))
    return {
        "complete_mst_n_edges": n_edges,
        "complete_mst_total_length": float(lengths.sum()) if n_edges else 0.0,
        "complete_mst_mean_edge_length": float(lengths.mean()) if n_edges else 0.0,
        "complete_mst_max_edge_length": float(lengths.max()) if n_edges else 0.0,
        "complete_mst_d20": float(thresholds[0.2]),
        "complete_mst_d50": float(thresholds[0.5]),
        "complete_mst_d80": float(thresholds[0.8]),
    }


# ============================================================
# Shape / percolation
# ============================================================

def percolates_top_bottom(cluster_labels: np.ndarray, n_clusters: int) -> bool:
    """True if any forest cluster touches both top and bottom border."""
    if n_clusters <= 0:
        return False
    top = set(np.unique(cluster_labels[0, :]))
    bottom = set(np.unique(cluster_labels[-1, :]))
    top.discard(0)
    bottom.discard(0)
    return len(top & bottom) > 0


def percolates_left_right(cluster_labels: np.ndarray, n_clusters: int) -> bool:
    """True if any forest cluster touches both left and right border."""
    if n_clusters <= 0:
        return False
    left = set(np.unique(cluster_labels[:, 0]))
    right = set(np.unique(cluster_labels[:, -1]))
    left.discard(0)
    right.discard(0)
    return len(left & right) > 0


def compute_shape_stats(
    cluster_labels: np.ndarray,
    n_clusters: int,
    boundary_mode: str = "periodic",
) -> dict[str, float]:
    """
    Simple shape metric for the largest cluster:
      shape = 0.25 * perimeter / sqrt(area)
    """
    if n_clusters <= 0:
        return {"largest_cluster_shape": 0.0}

    areas = np.bincount(cluster_labels.ravel(), minlength=n_clusters + 1)[1:]
    largest_idx = int(np.argmax(areas))
    largest_label = largest_idx + 1

    mask = (cluster_labels == largest_label)
    perimeter = _mask_perimeter(mask, boundary_mode=boundary_mode)

    area = float(areas[largest_idx])
    if area <= 0:
        return {"largest_cluster_shape": 0.0}

    shape = 0.25 * perimeter / np.sqrt(area)
    return {"largest_cluster_shape": float(shape)}

def percolating_cluster_labels_top_bottom(cluster_labels: np.ndarray) -> set[int]:
    """Return all cluster labels that touch both top and bottom border."""
    top = set(np.unique(cluster_labels[0, :]))
    bottom = set(np.unique(cluster_labels[-1, :]))
    top.discard(0)
    bottom.discard(0)
    return top & bottom


def percolating_cluster_labels_left_right(cluster_labels: np.ndarray) -> set[int]:
    """Return all cluster labels that touch both left and right border."""
    left = set(np.unique(cluster_labels[:, 0]))
    right = set(np.unique(cluster_labels[:, -1]))
    left.discard(0)
    right.discard(0)
    return left & right


def percolating_cluster_labels(cluster_labels: np.ndarray) -> set[int]:
    """Union of top-bottom and left-right percolating cluster labels."""
    return percolating_cluster_labels_top_bottom(cluster_labels) | percolating_cluster_labels_left_right(cluster_labels)


# ============================================================
# Bundles
# ============================================================

def compute_cluster_stats(
    cluster_labels: np.ndarray,
    n_clusters: int,
    boundary_mode: str = "periodic",
) -> dict[str, float]:
    """Basic statistics of cluster sizes (plus a few clarified / derived metrics)."""
    sizes = cluster_sizes(cluster_labels, n_clusters)

    if n_clusters <= 0 or sizes.size == 0:
        return {
            "n_clusters": 0,
            "mean_cluster_size": 0.0,
            "mean_cluster_size_excluding_largest": 0.0,
            "mean_cluster_size_excluding_percolating": 0.0,
            "var_cluster_size": 0.0,
            "largest_cluster_size": 0,
            "largest_cluster_fraction_forest": 0.0,
            "largest_cluster_fraction_total": 0.0,
            "largest_cluster_fraction": 0.0,  # alias (forest-based)
            "landscape_division_index": 0.0,
            "mean_centroid_nearest_neighbor_distance": np.nan,

            # NEW
            "chi": np.nan,
            "S2_size": 0.0,
            "S2_fraction_forest": 0.0,
            "S2_fraction_total": 0.0,  # set in compute_landscape_metrics
        }

    total_forest = float(sizes.sum())
    largest = int(sizes.max())
    largest_fraction_forest = float(largest / total_forest) if total_forest > 0 else 0.0

    # Exclude exactly one largest patch, retaining other tied-largest patches.
    sizes_excl_largest = _sizes_excluding_one_largest(sizes)
    mean_excl_largest = float(sizes_excl_largest.mean()) if sizes_excl_largest.size > 0 else 0.0

    # Percolating cluster labels (top-bottom OR left-right)
    perc_labels = set(percolating_cluster_labels(cluster_labels))

    # Mean excluding percolating clusters
    if len(perc_labels) == 0:
        mean_excl_perc = float(sizes.mean())
    else:
        # sizes[i] corresponds to label (i+1)
        keep_idx = [i for i in range(n_clusters) if (i + 1) not in perc_labels]
        sizes_excl_perc = sizes[np.array(keep_idx, dtype=int)] if len(keep_idx) > 0 else np.zeros(0, dtype=sizes.dtype)
        mean_excl_perc = float(sizes_excl_perc.mean()) if sizes_excl_perc.size > 0 else 0.0

    chi, s2 = _chi_and_second_largest_from_sizes(sizes)

    s2_fraction_forest = float(s2 / total_forest) if total_forest > 0 else 0.0

    # Total-landscape fractions need grid size, set later
    return {
        "n_clusters": float(n_clusters),
        "mean_cluster_size": float(sizes.mean()),
        "mean_cluster_size_excluding_largest": mean_excl_largest,
        "mean_cluster_size_excluding_percolating": mean_excl_perc,
        "var_cluster_size": float(sizes.var()),
        "largest_cluster_size": float(largest),
        "largest_cluster_fraction_forest": largest_fraction_forest,
        "largest_cluster_fraction_total": np.nan,  # set in compute_landscape_metrics
        "largest_cluster_fraction": largest_fraction_forest,  # alias (forest-based)
        "landscape_division_index": float(landscape_division_index(sizes)),
        "mean_centroid_nearest_neighbor_distance": float(
            mean_centroid_nearest_neighbor_distance(cluster_labels, n_clusters, boundary_mode=boundary_mode)
        ),

        # NEW
        "chi": float(chi),
        "S2_size": float(s2),
        "S2_fraction_forest": float(s2_fraction_forest),
        "S2_fraction_total": np.nan,  # set in compute_landscape_metrics
    }


def cluster_size_distribution(cluster_labels: np.ndarray, n_clusters: int) -> dict[int, int]:
    """Frequency distribution: {size -> count}."""
    sizes = cluster_sizes(cluster_labels, n_clusters)
    unique, counts = np.unique(sizes, return_counts=True)
    return {int(s): int(c) for s, c in zip(unique, counts)}


def largest_cluster_fraction(grid: np.ndarray, cluster_labels: np.ndarray, n_clusters: int) -> float:
    """Fraction of all cells that belong to the largest forest cluster."""
    total = int(grid.size)
    if total == 0 or n_clusters <= 0:
        return 0.0
    sizes = cluster_sizes(cluster_labels, n_clusters)
    return float(sizes.max()) / float(total)


def _normalize_metric_keys(
    metric_keys: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str] | None:
    if metric_keys is None:
        return None
    requested = {str(key) for key in metric_keys}
    invalid = sorted(requested.difference(ALL_LANDSCAPE_METRIC_KEYS))
    if invalid:
        raise ValueError(f"Unknown landscape metric keys requested: {invalid}")
    return requested


def compute_landscape_metrics_from_landscape(
    landscape: Any,
    *,
    connectivity: str | int | None = None,
    boundary_mode: str | None = None,
    metric_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    si_alpha: float = 0.15,
    si_b: float = 1.0,
    si_distance_mode: str | None = None,
    si_softmin_tau: float = 1.0,
    si_voronoi_depth: int | str | None = 1,
) -> dict[str, Any]:
    """
    Compute landscape metrics from a prepared Landscape cache.

    This reuses core_cluster_labels, cluster_areas and cluster_dist_matrix from
    Landscape.ensure_clusters(...), which is especially important for
    distance_mode="voronoi_dt" sweeps.
    """
    requested = _normalize_metric_keys(metric_keys)
    if requested is not None and len(requested) == 0:
        return {}

    grid = np.asarray(getattr(landscape, "grid", getattr(landscape, "A", None)))
    if grid.ndim != 2:
        raise ValueError("landscape must provide a 2D grid or A attribute.")

    info = getattr(landscape, "cluster_info", None)
    conn = _normalize_connectivity(
        connectivity if connectivity is not None else getattr(info, "connectivity", "4")
    )
    mode = normalize_boundary_mode(
    boundary_mode if boundary_mode is not None else getattr(info, "boundary_mode", "open")
    )
    cached_distance_mode = str(getattr(info, "distance_mode", "centroid"))
    si_mode = cached_distance_mode if si_distance_mode is None else str(si_distance_mode)
    if si_mode not in SI_DISTANCE_MODES:
        raise ValueError(f"si_distance_mode must be one of {SI_DISTANCE_MODES!r}, got {si_mode!r}.")

    total_cells = int(grid.size)
    metrics: dict[str, Any] = {}

    if requested is None or "forest_cover" in requested:
        metrics["forest_cover"] = float(forest_cover(grid))

    need_cluster_metrics = requested is None or bool(requested & _CLUSTER_METRIC_KEYS)
    need_adjacency_metrics = conn == "4" and (requested is None or bool(requested & _ADJACENCY_METRIC_KEYS))

    if need_cluster_metrics:
        if not hasattr(landscape, "core_cluster_labels"):
            raise ValueError("landscape must be prepared with ensure_clusters(...) before cached metric computation.")
        cluster_labels = np.asarray(landscape.core_cluster_labels)
        n_clusters = int(getattr(landscape, "n_clusters"))
        sizes = np.asarray(getattr(landscape, "cluster_areas"), dtype=float)
        if sizes.shape != (n_clusters,):
            raise ValueError("landscape.cluster_areas has wrong shape.")

        if requested is None or "n_clusters" in requested:
            metrics["n_clusters"] = int(n_clusters)

        if n_clusters <= 0 or sizes.size == 0:
            zero_float_metrics = {
                "mean_cluster_size",
                "mean_cluster_size_excluding_largest",
                "mean_cluster_size_excluding_percolating",
                "var_cluster_size",
                "largest_cluster_fraction_forest",
                "largest_cluster_fraction_total",
                "largest_cluster_fraction",
                "landscape_division_index",
                "largest_cluster_shape",
                "S2_fraction_forest",
                "S2_fraction_total",
                "complete_mst_total_length",
                "complete_mst_mean_edge_length",
                "complete_mst_max_edge_length",
                "complete_mst_d20",
                "complete_mst_d50",
                "complete_mst_d80",
            }
            zero_int_metrics = {
                "largest_cluster_size",
                "S2_size",
                "complete_mst_n_edges",
            }
            zero_bool_metrics = {"percolates_top_bottom", "percolates_left_right"}
            keys = ALL_LANDSCAPE_METRIC_KEYS if requested is None else requested
            for key in keys:
                if key in zero_float_metrics:
                    metrics[key] = 0.0
                elif key in zero_int_metrics:
                    metrics[key] = 0
                elif key in zero_bool_metrics:
                    metrics[key] = False
                elif key == "cluster_size_distribution":
                    metrics[key] = {}
                elif key == "nearest_neighbor_distance_distribution":
                    metrics[key] = {"bin_edges": [], "counts": []}
                elif key == "mean_full_occupancy_Si":
                    metrics[key] = 0.0
                elif key == "mean_nearest_neighbor_distance":
                    metrics[key] = np.nan
                elif key == "mean_centroid_nearest_neighbor_distance":
                    metrics[key] = np.nan
                elif key == "chi":
                    metrics[key] = np.nan
        else:
            total_forest = float(sizes.sum())
            largest_size = int(sizes.max())
            largest_fraction_forest = float(largest_size / total_forest) if total_forest > 0 else 0.0
            largest_fraction_total = (float(largest_size) / float(total_cells)) if total_cells > 0 else 0.0
            sizes_excl_largest = _sizes_excluding_one_largest(sizes)

            need_perco_flags = requested is None or bool(requested & _PERCO_FLAG_KEYS)
            need_perco_exclusion = requested is None or bool(requested & _PERCO_EXCLUSION_KEYS)
            perc_tb = False
            perc_lr = False
            perc_labels: set[int] = set()
            if need_perco_flags or need_perco_exclusion:
                perc_tb = bool(percolates_top_bottom(cluster_labels, n_clusters))
                perc_lr = bool(percolates_left_right(cluster_labels, n_clusters))
                if need_perco_exclusion and (perc_tb or perc_lr):
                    perc_labels = set(percolating_cluster_labels(cluster_labels))

            if requested is None or "mean_cluster_size" in requested:
                metrics["mean_cluster_size"] = float(sizes.mean())
            if requested is None or "mean_cluster_size_excluding_largest" in requested:
                metrics["mean_cluster_size_excluding_largest"] = (
                    float(sizes_excl_largest.mean()) if sizes_excl_largest.size > 0 else 0.0
                )
            if requested is None or "mean_cluster_size_excluding_percolating" in requested:
                if len(perc_labels) == 0:
                    metrics["mean_cluster_size_excluding_percolating"] = float(sizes.mean())
                else:
                    keep_idx = [i for i in range(n_clusters) if (i + 1) not in perc_labels]
                    sizes_excl_perc = (
                        sizes[np.array(keep_idx, dtype=int)] if len(keep_idx) > 0 else np.zeros(0, dtype=sizes.dtype)
                    )
                    metrics["mean_cluster_size_excluding_percolating"] = (
                        float(sizes_excl_perc.mean()) if sizes_excl_perc.size > 0 else 0.0
                    )
            if requested is None or "var_cluster_size" in requested:
                metrics["var_cluster_size"] = float(sizes.var())
            if requested is None or "largest_cluster_size" in requested:
                metrics["largest_cluster_size"] = int(largest_size)
            if requested is None or "largest_cluster_fraction_forest" in requested:
                metrics["largest_cluster_fraction_forest"] = float(largest_fraction_forest)
            if requested is None or "largest_cluster_fraction_total" in requested:
                metrics["largest_cluster_fraction_total"] = float(largest_fraction_total)
            if requested is None or "largest_cluster_fraction" in requested:
                metrics["largest_cluster_fraction"] = float(largest_fraction_forest)
            if requested is None or "landscape_division_index" in requested:
                metrics["landscape_division_index"] = float(landscape_division_index(sizes))
            if requested is None or "cluster_size_distribution" in requested:
                unique, counts = np.unique(sizes.astype(int), return_counts=True)
                metrics["cluster_size_distribution"] = {int(s): int(c) for s, c in zip(unique, counts)}
            if requested is None or "percolates_top_bottom" in requested:
                metrics["percolates_top_bottom"] = bool(perc_tb)
            if requested is None or "percolates_left_right" in requested:
                metrics["percolates_left_right"] = bool(perc_lr)
            if requested is None or "largest_cluster_shape" in requested:
                metrics["largest_cluster_shape"] = float(
                    compute_shape_stats(cluster_labels, n_clusters, boundary_mode=mode)["largest_cluster_shape"]
                )
            need_voronoi_nn_metrics = requested is None or bool(
                requested & {"mean_nearest_neighbor_distance", "nearest_neighbor_distance_distribution"}
            )
            voronoi_nn_values = None
            if need_voronoi_nn_metrics:
                if cached_distance_mode == "voronoi_dt":
                    voronoi_graph = landscape.ensure_voronoi_neighbor_graph()
                else:
                    from src.landscape.landscape import Landscape

                    voronoi_landscape = Landscape(np.asarray(grid, dtype=np.int8))
                    voronoi_landscape.ensure_clusters(
                        edge_depth=float(getattr(info, "edge_depth", 0.0)),
                        connectivity=conn,
                        boundary_mode=mode,
                        habitat_threshold=float(getattr(info, "habitat_threshold", 0.5)),
                        distance_mode="voronoi_dt",
                        softmin_tau=si_softmin_tau,
                        voronoi_depth=si_voronoi_depth,
                        rebuild=True,
                    )
                    voronoi_graph = voronoi_landscape.ensure_voronoi_neighbor_graph()
                voronoi_nn_values = nearest_neighbor_distances_from_voronoi_graph(voronoi_graph)

            need_centroid_nn = requested is None or "mean_centroid_nearest_neighbor_distance" in requested
            centroid_nn_values = None
            if need_centroid_nn:
                cached_centers = getattr(landscape, "cluster_centroids", None)
                if cached_centers is not None:
                    centroid_nn_values = nearest_neighbor_distances_from_centroids(
                        np.asarray(cached_centers, dtype=float),
                        shape=cluster_labels.shape,
                        boundary_mode=mode,
                    )
                else:
                    centroid_nn_values = nearest_neighbor_distances(
                        cluster_labels, n_clusters, boundary_mode=mode
                    )[1:]

            if requested is None or "mean_nearest_neighbor_distance" in requested:
                metrics["mean_nearest_neighbor_distance"] = float(
                    mean_nearest_neighbor_distance_from_values(voronoi_nn_values)
                )
            if requested is None or "mean_centroid_nearest_neighbor_distance" in requested:
                metrics["mean_centroid_nearest_neighbor_distance"] = float(
                    mean_nearest_neighbor_distance_from_values(centroid_nn_values)
                )
            if requested is None or "nearest_neighbor_distance_distribution" in requested:
                metrics["nearest_neighbor_distance_distribution"] = nearest_neighbor_distance_distribution_from_values(
                    voronoi_nn_values
                )
            if requested is None or "mean_full_occupancy_Si" in requested:
                if si_mode == cached_distance_mode:
                    metrics["mean_full_occupancy_Si"] = float(
                        mean_full_occupancy_Si_from_distances(
                            sizes,
                            np.asarray(getattr(landscape, "cluster_dist_matrix"), dtype=float),
                            alpha=si_alpha,
                            b=si_b,
                        )
                    )
                elif si_mode == "centroid":
                    metrics["mean_full_occupancy_Si"] = float(
                        mean_full_occupancy_Si_from_clusters(
                            cluster_labels,
                            sizes,
                            boundary_mode=mode,
                            alpha=si_alpha,
                            b=si_b,
                        )
                    )
                else:
                    raise ValueError(
                        "Prepared Landscape distance cache is "
                        f"{cached_distance_mode!r}, but mean_full_occupancy_Si requested {si_mode!r}."
                    )

            need_finite_cluster_stats = requested is None or bool(
                requested & {"chi", "S2_size", "S2_fraction_forest", "S2_fraction_total"}
            )
            if need_finite_cluster_stats:
                chi, s2 = _chi_and_second_largest_from_sizes(sizes)

                s2_fraction_forest = float(s2 / total_forest) if total_forest > 0 else 0.0
                s2_fraction_total = (float(s2) / float(total_cells)) if total_cells > 0 else 0.0

                if requested is None or "chi" in requested:
                    metrics["chi"] = float(chi)
                if requested is None or "S2_size" in requested:
                    metrics["S2_size"] = int(s2)
                if requested is None or "S2_fraction_forest" in requested:
                    metrics["S2_fraction_forest"] = float(s2_fraction_forest)
                if requested is None or "S2_fraction_total" in requested:
                    metrics["S2_fraction_total"] = float(s2_fraction_total)

            need_complete_mst = requested is None or bool(requested & _COMPLETE_MST_METRIC_KEYS)
            if need_complete_mst:
                mst_metrics = compute_complete_mst_metrics_from_distances(
                    np.asarray(getattr(landscape, "cluster_dist_matrix"), dtype=float),
                    sizes,
                )
                if requested is None:
                    metrics.update(mst_metrics)
                else:
                    for key in requested & _COMPLETE_MST_METRIC_KEYS:
                        metrics[key] = mst_metrics[key]

    if need_adjacency_metrics:
        adj = compute_adjacency_metrics(grid, connectivity="4", boundary_mode=mode)
        if requested is None:
            metrics.update(adj)
        else:
            for key in requested & _ADJACENCY_METRIC_KEYS:
                metrics[key] = adj[key]

    return metrics


def compute_landscape_metrics(
    grid: np.ndarray,
    connectivity: str | int = "4",
    boundary_mode: str = "open",
    metric_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    si_alpha: float = 0.15,
    si_b: float = 1.0,
    si_distance_mode: str = "voronoi_dt",
    si_softmin_tau: float = 1.0,
    si_voronoi_depth: int | str | None = 1,
) -> dict[str, Any]:
    """Compute a set of fragmentation metrics for a binary landscape."""
    requested = _normalize_metric_keys(metric_keys)
    if requested is not None and len(requested) == 0:
        return {}

    conn = _normalize_connectivity(connectivity)
    mode = normalize_boundary_mode(boundary_mode)
    total_cells = int(grid.size)
    metrics: dict[str, Any] = {}

    if requested is None or "forest_cover" in requested:
        metrics["forest_cover"] = float(forest_cover(grid))

    need_cluster_metrics = requested is None or bool(requested & _CLUSTER_METRIC_KEYS)
    need_adjacency_metrics = conn == "4" and (requested is None or bool(requested & _ADJACENCY_METRIC_KEYS))

    if not need_cluster_metrics and not need_adjacency_metrics:
        return metrics

    if need_cluster_metrics:
        cluster_labels, n_clusters = label_forest_clusters(
            grid,
            connectivity=conn,
            boundary_mode=mode,
        )
        sizes = cluster_sizes(cluster_labels, n_clusters)

        if requested is None or "n_clusters" in requested:
            metrics["n_clusters"] = int(n_clusters)

        if n_clusters <= 0 or sizes.size == 0:
            zero_float_metrics = {
                "mean_cluster_size",
                "mean_cluster_size_excluding_largest",
                "mean_cluster_size_excluding_percolating",
                "var_cluster_size",
                "largest_cluster_fraction_forest",
                "largest_cluster_fraction_total",
                "largest_cluster_fraction",
                "landscape_division_index",
                "largest_cluster_shape",
                "S2_fraction_forest",
                "S2_fraction_total",
                "complete_mst_total_length",
                "complete_mst_mean_edge_length",
                "complete_mst_max_edge_length",
                "complete_mst_d20",
                "complete_mst_d50",
                "complete_mst_d80",
            }
            zero_int_metrics = {
                "largest_cluster_size",
                "S2_size",
                "complete_mst_n_edges",
            }
            zero_bool_metrics = {"percolates_top_bottom", "percolates_left_right"}

            keys = ALL_LANDSCAPE_METRIC_KEYS if requested is None else requested
            for key in keys:
                if key in zero_float_metrics:
                    metrics[key] = 0.0
                elif key in zero_int_metrics:
                    metrics[key] = 0
                elif key in zero_bool_metrics:
                    metrics[key] = False
                elif key == "cluster_size_distribution":
                    metrics[key] = {}
                elif key == "nearest_neighbor_distance_distribution":
                    metrics[key] = {"bin_edges": [], "counts": []}
                elif key == "mean_full_occupancy_Si":
                    metrics[key] = 0.0
                elif key == "mean_nearest_neighbor_distance":
                    metrics[key] = np.nan
                elif key == "mean_centroid_nearest_neighbor_distance":
                    metrics[key] = np.nan
                elif key == "chi":
                    metrics[key] = np.nan
            if need_adjacency_metrics:
                adj = compute_adjacency_metrics(grid, connectivity="4", boundary_mode=mode)
                if requested is None:
                    metrics.update(adj)
                else:
                    for key in requested & _ADJACENCY_METRIC_KEYS:
                        metrics[key] = adj[key]
            return metrics

        total_forest = float(sizes.sum())
        largest_size = int(sizes.max())
        largest_fraction_forest = float(largest_size / total_forest) if total_forest > 0 else 0.0
        largest_fraction_total = (float(largest_size) / float(total_cells)) if total_cells > 0 else 0.0
        sizes_excl_largest = _sizes_excluding_one_largest(sizes)

        need_perco_flags = requested is None or bool(requested & _PERCO_FLAG_KEYS)
        need_perco_exclusion = requested is None or bool(requested & _PERCO_EXCLUSION_KEYS)
        perc_tb = False
        perc_lr = False
        perc_labels: set[int] = set()
        if need_perco_flags or need_perco_exclusion:
            perc_tb = bool(percolates_top_bottom(cluster_labels, n_clusters))
            perc_lr = bool(percolates_left_right(cluster_labels, n_clusters))
            if need_perco_exclusion and (perc_tb or perc_lr):
                perc_labels = set(percolating_cluster_labels(cluster_labels))

        if requested is None or "mean_cluster_size" in requested:
            metrics["mean_cluster_size"] = float(sizes.mean())
        if requested is None or "mean_cluster_size_excluding_largest" in requested:
            metrics["mean_cluster_size_excluding_largest"] = (
                float(sizes_excl_largest.mean()) if sizes_excl_largest.size > 0 else 0.0
            )
        if requested is None or "mean_cluster_size_excluding_percolating" in requested:
            if len(perc_labels) == 0:
                metrics["mean_cluster_size_excluding_percolating"] = float(sizes.mean())
            else:
                keep_idx = [i for i in range(n_clusters) if (i + 1) not in perc_labels]
                sizes_excl_perc = (
                    sizes[np.array(keep_idx, dtype=int)] if len(keep_idx) > 0 else np.zeros(0, dtype=sizes.dtype)
                )
                metrics["mean_cluster_size_excluding_percolating"] = (
                    float(sizes_excl_perc.mean()) if sizes_excl_perc.size > 0 else 0.0
                )
        if requested is None or "var_cluster_size" in requested:
            metrics["var_cluster_size"] = float(sizes.var())
        if requested is None or "largest_cluster_size" in requested:
            metrics["largest_cluster_size"] = int(largest_size)
        if requested is None or "largest_cluster_fraction_forest" in requested:
            metrics["largest_cluster_fraction_forest"] = float(largest_fraction_forest)
        if requested is None or "largest_cluster_fraction_total" in requested:
            metrics["largest_cluster_fraction_total"] = float(largest_fraction_total)
        if requested is None or "largest_cluster_fraction" in requested:
            metrics["largest_cluster_fraction"] = float(largest_fraction_forest)
        if requested is None or "landscape_division_index" in requested:
            metrics["landscape_division_index"] = float(landscape_division_index(sizes))
        if requested is None or "cluster_size_distribution" in requested:
            metrics["cluster_size_distribution"] = cluster_size_distribution(cluster_labels, n_clusters)
        if requested is None or "percolates_top_bottom" in requested:
            metrics["percolates_top_bottom"] = bool(perc_tb)
        if requested is None or "percolates_left_right" in requested:
            metrics["percolates_left_right"] = bool(perc_lr)
        if requested is None or "largest_cluster_shape" in requested:
            metrics["largest_cluster_shape"] = float(
                compute_shape_stats(cluster_labels, n_clusters, boundary_mode=mode)["largest_cluster_shape"]
            )
        need_voronoi_nn_metrics = requested is None or bool(
            requested & {"mean_nearest_neighbor_distance", "nearest_neighbor_distance_distribution"}
        )
        voronoi_landscape = None
        voronoi_nn_values = None
        if need_voronoi_nn_metrics:
            from src.landscape.landscape import Landscape

            voronoi_landscape = Landscape(np.asarray(grid, dtype=np.int8))
            voronoi_landscape.ensure_clusters(
                edge_depth=0.0,
                connectivity=conn,
                boundary_mode=mode,
                habitat_threshold=0.5,
                distance_mode="voronoi_dt",
                softmin_tau=si_softmin_tau,
                voronoi_depth=si_voronoi_depth,
                rebuild=True,
            )
            voronoi_nn_values = nearest_neighbor_distances_from_voronoi_graph(
                voronoi_landscape.ensure_voronoi_neighbor_graph()
            )

        centroid_nn_values = None
        if requested is None or "mean_centroid_nearest_neighbor_distance" in requested:
            centroid_nn_values = nearest_neighbor_distances(
                cluster_labels, n_clusters, boundary_mode=mode
            )[1:]

        if requested is None or "mean_nearest_neighbor_distance" in requested:
            metrics["mean_nearest_neighbor_distance"] = float(
                mean_nearest_neighbor_distance_from_values(voronoi_nn_values)
            )
        if requested is None or "mean_centroid_nearest_neighbor_distance" in requested:
            metrics["mean_centroid_nearest_neighbor_distance"] = float(
                mean_nearest_neighbor_distance_from_values(centroid_nn_values)
            )
        if requested is None or "nearest_neighbor_distance_distribution" in requested:
            metrics["nearest_neighbor_distance_distribution"] = nearest_neighbor_distance_distribution_from_values(
                voronoi_nn_values
            )
        if requested is None or "mean_full_occupancy_Si" in requested:
            if si_distance_mode == "centroid":
                metrics["mean_full_occupancy_Si"] = float(
                    mean_full_occupancy_Si_from_clusters(
                        cluster_labels,
                        sizes,
                        boundary_mode=mode,
                        alpha=si_alpha,
                        b=si_b,
                    )
                )
            else:
                metrics["mean_full_occupancy_Si"] = float(
                    mean_full_occupancy_Si(
                        grid,
                        connectivity=conn,
                        boundary_mode=mode,
                        alpha=si_alpha,
                        b=si_b,
                        distance_mode=si_distance_mode,
                        softmin_tau=si_softmin_tau,
                        voronoi_depth=si_voronoi_depth,
                    )
                )

        need_finite_cluster_stats = requested is None or bool(requested & {"chi", "S2_size", "S2_fraction_forest", "S2_fraction_total"})
        if need_finite_cluster_stats:
            chi, s2 = _chi_and_second_largest_from_sizes(sizes)

            s2_fraction_forest = float(s2 / total_forest) if total_forest > 0 else 0.0
            s2_fraction_total = (float(s2) / float(total_cells)) if total_cells > 0 else 0.0

            if requested is None or "chi" in requested:
                metrics["chi"] = float(chi)
            if requested is None or "S2_size" in requested:
                metrics["S2_size"] = int(s2)
            if requested is None or "S2_fraction_forest" in requested:
                metrics["S2_fraction_forest"] = float(s2_fraction_forest)
            if requested is None or "S2_fraction_total" in requested:
                metrics["S2_fraction_total"] = float(s2_fraction_total)

        need_complete_mst = requested is None or bool(requested & _COMPLETE_MST_METRIC_KEYS)
        if need_complete_mst:
            from src.landscape.landscape import Landscape

            mst_distance_mode = si_distance_mode if si_distance_mode in SI_DISTANCE_MODES else "voronoi_dt"
            if mst_distance_mode == "voronoi_dt" and voronoi_landscape is not None:
                mst_landscape = voronoi_landscape
            else:
                mst_landscape = Landscape(np.asarray(grid, dtype=np.int8))
                mst_landscape.ensure_clusters(
                    edge_depth=0.0,
                    connectivity=conn,
                    boundary_mode=mode,
                    habitat_threshold=0.5,
                    distance_mode=mst_distance_mode,
                    softmin_tau=si_softmin_tau,
                    voronoi_depth=si_voronoi_depth,
                    rebuild=True,
                )
            mst_metrics = compute_complete_mst_metrics_from_distances(
                np.asarray(mst_landscape.cluster_dist_matrix, dtype=float),
                np.asarray(mst_landscape.cluster_areas, dtype=float),
            )
            if requested is None:
                metrics.update(mst_metrics)
            else:
                for key in requested & _COMPLETE_MST_METRIC_KEYS:
                    metrics[key] = mst_metrics[key]

    if need_adjacency_metrics:
        adj = compute_adjacency_metrics(grid, connectivity="4", boundary_mode=mode)
        if requested is None:
            metrics.update(adj)
        else:
            for key in requested & _ADJACENCY_METRIC_KEYS:
                metrics[key] = adj[key]

    return metrics

# ============================================================
# Local test (only runs when called directly)
# ============================================================

if __name__ == "__main__":
    import numpy as _np

    def _assert_close(a: float, b: float, tol: float = 1e-9, msg: str = "") -> None:
        if _np.isnan(b):
            assert _np.isnan(a), msg or f"Expected NaN, got {a}"
            return
        assert abs(float(a) - float(b)) <= tol, msg or f"Expected {b}, got {a}"

    def _assert_in_range(x: float, lo: float, hi: float, msg: str = "") -> None:
        assert (x >= lo) and (x <= hi), msg or f"Expected in [{lo},{hi}], got {x}"

    def _assert_keys(metrics: dict, keys: tuple[str, ...], name: str) -> None:
        for k in keys:
            assert k in metrics, f"{name}: missing key '{k}'"

    def _run_case(
        name: str,
        grid: _np.ndarray,
        connectivity: str,
        expected: dict,
    ) -> None:
        metrics = compute_landscape_metrics(grid, connectivity=connectivity)

        # --- required keys (core) ---
        _assert_keys(
            metrics,
            (
                "forest_cover",
                "n_clusters",
                "mean_cluster_size",
                "mean_cluster_size_excluding_largest",
                "mean_cluster_size_excluding_percolating",
                "var_cluster_size",
                "largest_cluster_size",
                "largest_cluster_fraction_forest",
                "largest_cluster_fraction_total",
                "largest_cluster_fraction",  # alias
                "landscape_division_index",
                "cluster_size_distribution",
                "percolates_top_bottom",
                "percolates_left_right",
                "largest_cluster_shape",
                "mean_nearest_neighbor_distance",
            ),
            name,
        )

        # --- basic sanity ---
        _assert_in_range(float(metrics["forest_cover"]), 0.0, 1.0, f"{name}: forest_cover out of range")
        assert int(metrics["n_clusters"]) >= 0, f"{name}: n_clusters negative"

        _assert_in_range(float(metrics["largest_cluster_fraction_forest"]), 0.0, 1.0, f"{name}: lcf_forest out of range")
        _assert_in_range(float(metrics["largest_cluster_fraction_total"]), 0.0, 1.0, f"{name}: lcf_total out of range")

        # alias should match forest-based fraction
        _assert_close(
            float(metrics["largest_cluster_fraction"]),
            float(metrics["largest_cluster_fraction_forest"]),
            msg=f"{name}: largest_cluster_fraction alias mismatch",
        )

        # NNI distance convention: <2 clusters => NaN or 0.0 are acceptable
        mnn = float(metrics["mean_nearest_neighbor_distance"])
        if int(metrics["n_clusters"]) < 2:
            assert (_np.isnan(mnn) or abs(mnn) <= 1e-12), (
                f"{name}: expected NaN or 0.0 mean_nearest_neighbor_distance for <2 clusters, got {mnn}"
            )
        else:
            assert _np.isfinite(mnn) and (mnn >= 0.0), f"{name}: invalid mean_nearest_neighbor_distance {mnn}"

        # Mean sizes excluding rules should be non-negative
        assert float(metrics["mean_cluster_size_excluding_largest"]) >= 0.0, f"{name}: mean_excl_largest negative"
        assert float(metrics["mean_cluster_size_excluding_percolating"]) >= 0.0, f"{name}: mean_excl_perc negative"

        # Fractions should be consistent with sizes
        total_cells = int(grid.size)
        largest_size = int(metrics["largest_cluster_size"])
        lcf_total_expected = 0.0 if total_cells == 0 else largest_size / float(total_cells)
        _assert_close(float(metrics["largest_cluster_fraction_total"]), float(lcf_total_expected),
                      msg=f"{name}: largest_cluster_fraction_total inconsistent with grid size")

        # Adjacency metrics presence/absence
        if connectivity == "4":
            _assert_keys(
                metrics,
                ("aggregation_index", "percentage_of_like_adjacencies", "edge_density_total", "edge_density_forest", "edge_density", "cohesion_index"),
                name,
            )
            _assert_in_range(float(metrics["cohesion_index"]), 0.0, 1.0, f"{name}: cohesion_index out of range")
            assert float(metrics["edge_density_total"]) >= 0.0, f"{name}: edge_density_total negative"
            # alias must match total-based
            _assert_close(float(metrics["edge_density"]), float(metrics["edge_density_total"]),
                          msg=f"{name}: edge_density alias mismatch")
            # forest-based edge density is NaN if no forest
            if int((grid > 0).sum()) == 0:
                assert _np.isnan(float(metrics["edge_density_forest"])), f"{name}: expected NaN edge_density_forest with 0 forest"
            else:
                assert float(metrics["edge_density_forest"]) >= 0.0, f"{name}: edge_density_forest negative"
        else:
            # For connectivity "8", adjacency metrics are not auto-added in this implementation
            for k in ("aggregation_index", "percentage_of_like_adjacencies", "edge_density_total", "edge_density_forest", "edge_density", "cohesion_index"):
                assert k not in metrics, f"{name}: adjacency metric '{k}' should not be present for connectivity={connectivity}"

        # --- check expected values ---
        for key, exp in expected.items():
            assert key in metrics, f"{name}: expected key '{key}' missing from metrics"
            got = metrics[key]

            if isinstance(exp, dict):
                assert got == exp, f"{name}: {key} expected {exp}, got {got}"
            elif isinstance(exp, bool):
                assert bool(got) == exp, f"{name}: {key} expected {exp}, got {got}"
            elif isinstance(exp, int):
                assert int(got) == exp, f"{name}: {key} expected {exp}, got {got}"
            elif isinstance(exp, float):
                _assert_close(float(got), exp, msg=f"{name}: {key} mismatch")
            else:
                raise TypeError(f"{name}: unsupported expected type for {key}: {type(exp)}")

        print(f"[OK] {name}")

    # ------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------

    # 1) All non-forest
    grid0 = _np.zeros((3, 3), dtype=int)
    _run_case(
        name="all_zero_3x3_conn4",
        grid=grid0,
        connectivity="4",
        expected={
            "forest_cover": 0.0,
            "n_clusters": 0,
            "cluster_size_distribution": {},
            "percolates_top_bottom": False,
            "percolates_left_right": False,
            "largest_cluster_fraction_forest": 0.0,
            "largest_cluster_fraction_total": 0.0,
            "largest_cluster_shape": 0.0,
            "mean_cluster_size": 0.0,
            "mean_cluster_size_excluding_largest": 0.0,
            "mean_cluster_size_excluding_percolating": 0.0,
        },
    )

    # 2) All forest
    grid1 = _np.ones((3, 3), dtype=int)
    _run_case(
        name="all_one_3x3_conn4",
        grid=grid1,
        connectivity="4",
        expected={
            "forest_cover": 1.0,
            "n_clusters": 1,
            "cluster_size_distribution": {9: 1},
            "percolates_top_bottom": True,
            "percolates_left_right": True,
            "largest_cluster_fraction_forest": 1.0,
            "largest_cluster_fraction_total": 1.0,
            "mean_cluster_size": 9.0,
            # excluding largest leaves nothing -> 0.0 by convention
            "mean_cluster_size_excluding_largest": 0.0,
            # excluding percolating leaves nothing -> 0.0 by convention
            "mean_cluster_size_excluding_percolating": 0.0,
        },
    )

    # 3) Diagonal connectivity difference: 2x2 with two diagonal forest cells
    diag = _np.array([[1, 0],
                      [0, 1]], dtype=int)

    _run_case(
        name="diag_2x2_conn4",
        grid=diag,
        connectivity="4",
        expected={
            "forest_cover": 0.5,
            "n_clusters": 2,
            "cluster_size_distribution": {1: 2},
            "percolates_top_bottom": False,
            "percolates_left_right": False,
            "largest_cluster_fraction_forest": 0.5,  # largest=1, total_forest=2
            "largest_cluster_fraction_total": 0.25,  # largest=1, total_cells=4
            "mean_cluster_size": 1.0,
            # excluding largest removes only one tied-largest cluster -> 1.0
            "mean_cluster_size_excluding_largest": 1.0,
            # excluding percolating doesn't remove anything -> 1.0
            "mean_cluster_size_excluding_percolating": 1.0,
        },
    )

    _run_case(
        name="diag_2x2_conn8",
        grid=diag,
        connectivity="8",
        expected={
            "forest_cover": 0.5,
            "n_clusters": 1,
            "cluster_size_distribution": {2: 1},
            "percolates_top_bottom": True,
            "percolates_left_right": True,
            "largest_cluster_fraction_forest": 1.0,  # largest=2, total_forest=2
            "largest_cluster_fraction_total": 0.5,  # largest=2, total_cells=4
            "mean_cluster_size": 2.0,
            "mean_cluster_size_excluding_largest": 0.0,
            "mean_cluster_size_excluding_percolating": 0.0,
        },
    )

    # 4) Vertical corridor: percolates top-bottom but not left-right (conn4)
    corridor = _np.zeros((5, 5), dtype=int)
    corridor[:, 2] = 1
    _run_case(
        name="vertical_corridor_5x5_conn4",
        grid=corridor,
        connectivity="4",
        expected={
            "n_clusters": 1,
            "percolates_top_bottom": True,
            "percolates_left_right": False,
            "largest_cluster_fraction_forest": 1.0,
            "largest_cluster_fraction_total": 5 / 25,
            # excluding percolating removes the only cluster -> 0.0
            "mean_cluster_size_excluding_percolating": 0.0,
        },
    )

    # 5) Two separated blocks: NN distance should be finite (conn4)
    blocks = _np.zeros((6, 6), dtype=int)
    blocks[0:2, 0:2] = 1
    blocks[4:6, 4:6] = 1
    _run_case(
        name="two_blocks_6x6_conn4",
        grid=blocks,
        connectivity="4",
        expected={
            "n_clusters": 2,
            "cluster_size_distribution": {4: 2},
            "percolates_top_bottom": False,
            "percolates_left_right": False,
            "largest_cluster_fraction_forest": 0.5,   # largest=4, total_forest=8
            "largest_cluster_fraction_total": 4 / 36, # largest=4, total_cells=36
            "mean_cluster_size": 4.0,
            # excluding largest removes only one tied largest cluster -> 4.0
            "mean_cluster_size_excluding_largest": 4.0,
            # excluding percolating removes nothing -> 4.0
            "mean_cluster_size_excluding_percolating": 4.0,
        },
    )

    seam = _np.zeros((3, 3), dtype=int)
    seam[1, 0] = 1
    seam[1, -1] = 1
    metrics_periodic = compute_landscape_metrics(seam, connectivity="4", boundary_mode="periodic")
    _assert_close(float(metrics_periodic["n_clusters"]), 1.0, msg="periodic seam cells should form one cluster")
    metrics_open = compute_landscape_metrics(seam, connectivity="4", boundary_mode="open")
    _assert_close(float(metrics_open["n_clusters"]), 2.0, msg="open seam cells should stay separate")
    print("[OK] periodic seam connectivity")

    subset_metrics = compute_landscape_metrics(
        blocks,
        connectivity="4",
        boundary_mode="periodic",
        metric_keys=["forest_cover", "n_clusters"],
    )
    assert set(subset_metrics.keys()) == {"forest_cover", "n_clusters"}, "subset metric selection returned extra keys"
    _assert_close(float(subset_metrics["forest_cover"]), 8.0 / 36.0, msg="subset forest_cover mismatch")
    _assert_close(float(subset_metrics["n_clusters"]), 2.0, msg="subset n_clusters mismatch")
    empty_metrics = compute_landscape_metrics(blocks, connectivity="4", metric_keys=[])
    assert empty_metrics == {}, "empty metric selection should return an empty dict"
    nn_dist_metrics = compute_landscape_metrics(
        blocks,
        connectivity="4",
        boundary_mode="periodic",
        metric_keys=["nearest_neighbor_distance_distribution"],
    )
    assert set(nn_dist_metrics.keys()) == {"nearest_neighbor_distance_distribution"}, "NN distribution subset returned extra keys"
    nn_dist = nn_dist_metrics["nearest_neighbor_distance_distribution"]
    assert isinstance(nn_dist, dict), "NN distribution must be a dict"
    assert "bin_edges" in nn_dist and "counts" in nn_dist, "NN distribution keys missing"
    si_metrics = compute_landscape_metrics(
        blocks,
        connectivity="4",
        boundary_mode="periodic",
        metric_keys=["mean_full_occupancy_Si"],
        si_alpha=0.2,
        si_distance_mode="edge_min",
    )
    assert set(si_metrics.keys()) == {"mean_full_occupancy_Si"}, "S_i subset returned extra keys"
    assert float(si_metrics["mean_full_occupancy_Si"]) >= 0.0, "mean_full_occupancy_Si must be non-negative"

    from src.landscape.landscape import Landscape as _Landscape
    import tempfile as _tempfile
    from pathlib import Path as _Path

    L_cached = _Landscape(blocks)
    L_cached.ensure_clusters(
        edge_depth=0.0,
        connectivity="4",
        boundary_mode="periodic",
        habitat_threshold=0.5,
        distance_mode="voronoi_dt",
        rebuild=True,
    )
    direct_voronoi = compute_landscape_metrics(
        blocks,
        connectivity="4",
        boundary_mode="periodic",
        metric_keys=["n_clusters", "largest_cluster_size", "mean_full_occupancy_Si"],
        si_alpha=0.2,
        si_distance_mode="voronoi_dt",
    )
    cached_voronoi = compute_landscape_metrics_from_landscape(
        L_cached,
        metric_keys=[
            "n_clusters",
            "largest_cluster_size",
            "mean_full_occupancy_Si",
            "complete_mst_total_length",
            "complete_mst_d50",
        ],
        si_alpha=0.2,
        si_distance_mode="voronoi_dt",
    )
    assert set(direct_voronoi).issubset(cached_voronoi), "cached Landscape metric keys mismatch"
    _assert_close(
        float(cached_voronoi["mean_full_occupancy_Si"]),
        float(direct_voronoi["mean_full_occupancy_Si"]),
        msg="cached Landscape voronoi S_i mismatch",
    )
    _assert_close(
        float(cached_voronoi["largest_cluster_size"]),
        float(direct_voronoi["largest_cluster_size"]),
        msg="cached Landscape largest cluster mismatch",
    )
    with _tempfile.TemporaryDirectory() as _tmp:
        _path = _Path(_tmp) / "cached_landscape.npz"
        L_cached.save(_path, include_cache=True)
        L_loaded = _Landscape.load(_path)
        loaded_voronoi = compute_landscape_metrics_from_landscape(
            L_loaded,
            metric_keys=["mean_full_occupancy_Si", "complete_mst_total_length", "complete_mst_d50"],
            si_alpha=0.2,
            si_distance_mode="voronoi_dt",
        )
        _assert_close(
            float(loaded_voronoi["mean_full_occupancy_Si"]),
            float(cached_voronoi["mean_full_occupancy_Si"]),
            msg="loaded cached Landscape S_i mismatch",
        )
        _assert_close(
            float(loaded_voronoi["complete_mst_total_length"]),
            float(cached_voronoi["complete_mst_total_length"]),
            msg="loaded cached complete MST length mismatch",
        )
        _assert_close(
            float(loaded_voronoi["complete_mst_d50"]),
            float(cached_voronoi["complete_mst_d50"]),
            msg="loaded cached complete MST D50 mismatch",
        )
    print("[OK] cached Landscape metric computation")

    complete_dist = _np.array(
        [
            [0.0, 2.0, 10.0],
            [2.0, 0.0, 4.0],
            [10.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    complete_mst = compute_complete_mst_metrics_from_distances(
        complete_dist,
        _np.array([1.0, 2.0, 7.0]),
    )
    assert int(complete_mst["complete_mst_n_edges"]) == 2
    _assert_close(float(complete_mst["complete_mst_total_length"]), 6.0)
    _assert_close(float(complete_mst["complete_mst_mean_edge_length"]), 3.0)
    _assert_close(float(complete_mst["complete_mst_max_edge_length"]), 4.0)
    _assert_close(float(complete_mst["complete_mst_d20"]), 0.0)
    _assert_close(float(complete_mst["complete_mst_d50"]), 0.0)
    _assert_close(float(complete_mst["complete_mst_d80"]), 4.0)

    complete_mst_threshold = compute_complete_mst_metrics_from_distances(
        complete_dist,
        _np.array([1.0, 1.0, 1.0]),
    )
    _assert_close(float(complete_mst_threshold["complete_mst_d20"]), 0.0)
    _assert_close(float(complete_mst_threshold["complete_mst_d50"]), 2.0)
    _assert_close(float(complete_mst_threshold["complete_mst_d80"]), 4.0)
    print("[OK] complete-distance MST metrics")

    ties = _np.array([[1, 0, 1],
                      [0, 0, 0],
                      [1, 0, 1]], dtype=int)
    tie_metrics = compute_landscape_metrics(ties, connectivity="4", boundary_mode="open")
    _assert_close(
        float(tie_metrics["mean_cluster_size_excluding_largest"]),
        1.0,
        msg="excluding largest must remove only one tied-largest cluster",
    )
    _assert_close(float(tie_metrics["chi"]), 1.0, msg="chi with tied-largest singletons should be 1")
    _assert_close(float(tie_metrics["S2_size"]), 1.0, msg="S2 with tied-largest singletons should be 1")
    print("[OK] selective metric computation")

    print("\nAll plausibility checks passed.")
