"""Fast exact complete edge-to-edge distances for open-boundary patch labels."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _pair_minima(points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    n = len(offsets) - 1
    result = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i + 1, n):
            best = 1000000000
            for a in range(offsets[i], offsets[i + 1]):
                ax = points[a, 0]
                ay = points[a, 1]
                for b in range(offsets[j], offsets[j + 1]):
                    dx = ax - points[b, 0]
                    dy = ay - points[b, 1]
                    squared = dx * dx + dy * dy
                    if squared < best:
                        best = squared
            distance = np.sqrt(best)
            result[i, j] = distance
            result[j, i] = distance
    return result


def complete_edge_distances_open(labels: np.ndarray, n_patches: int) -> np.ndarray:
    """Return all exact minimum cell-centre boundary distances.

    ``labels`` has background 0 and patch labels 1..n_patches, generated with
    open boundaries and four-neighbour connectivity. Patch boundaries follow
    the existing landscape core's four-neighbour binary erosion definition.
    """
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 2 or n_patches < 0:
        raise ValueError("Expected a two-dimensional label raster and non-negative patch count.")
    if n_patches == 0:
        return np.zeros((0, 0), dtype=float)
    # The outer raster border is a patch boundary under open boundaries.
    interior = np.zeros_like(labels, dtype=bool)
    interior[1:-1, 1:-1] = (
        (labels[1:-1, 1:-1] > 0)
        & (labels[1:-1, 1:-1] == labels[:-2, 1:-1])
        & (labels[1:-1, 1:-1] == labels[2:, 1:-1])
        & (labels[1:-1, 1:-1] == labels[1:-1, :-2])
        & (labels[1:-1, 1:-1] == labels[1:-1, 2:])
    )
    boundary = (labels > 0) & ~interior
    yy, xx = np.nonzero(boundary)
    patch_labels = labels[yy, xx]
    order = np.argsort(patch_labels, kind="stable")
    points = np.column_stack((xx[order], yy[order])).astype(np.int32, copy=False)
    counts = np.bincount(patch_labels, minlength=n_patches + 1)[1:]
    if np.any(counts == 0):
        raise ValueError("Every patch must have at least one boundary cell.")
    offsets = np.empty(n_patches + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    return _pair_minima(points, offsets)
