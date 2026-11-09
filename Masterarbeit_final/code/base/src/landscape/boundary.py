from __future__ import annotations

import numpy as np
from scipy import ndimage


def normalize_boundary_mode(boundary_mode: str = "periodic") -> str:
    raw = str(boundary_mode).strip().lower()
    if raw in {"periodic", "wrap", "wrapped", "toroidal", "torus"}:
        return "periodic"
    if raw in {"open", "fixed", "constant", "nonperiodic", "non-periodic"}:
        return "open"
    raise ValueError(f"boundary_mode must be 'periodic' or 'open', got {boundary_mode!r}")


def possible_within_raster_orthogonal_adjacencies(
    shape: tuple[int, int],
    *,
    boundary_mode: str = "open",
) -> int:
    """Return the number of counted right/down four-neighbour adjacencies.

    Open rasters include only orthogonal pairs wholly inside the raster. For
    periodic rasters, each cell contributes one right and one down adjacency,
    matching the raster-adjacency counting convention used by landscape
    metrics and dynamic deforestation rates.
    """
    height, width = (int(shape[0]), int(shape[1]))
    if normalize_boundary_mode(boundary_mode) == "periodic":
        return int(2 * height * width)
    return int(height * max(width - 1, 0) + max(height - 1, 0) * width)


def binary_erosion_with_boundary(
    mask: np.ndarray,
    *,
    structure: np.ndarray,
    boundary_mode: str = "periodic",
) -> np.ndarray:
    mode = normalize_boundary_mode(boundary_mode)
    if mode == "open":
        return ndimage.binary_erosion(mask, structure=structure, border_value=0)

    pad_y = int(structure.shape[0] // 2)
    pad_x = int(structure.shape[1] // 2)
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="wrap")
    eroded = ndimage.binary_erosion(padded, structure=structure, border_value=0)
    return eroded[pad_y:pad_y + mask.shape[0], pad_x:pad_x + mask.shape[1]]


def distance_transform_edt_with_boundary(
    mask: np.ndarray,
    *,
    boundary_mode: str = "periodic",
    return_indices: bool = False,
):
    mode = normalize_boundary_mode(boundary_mode)
    if mode == "open":
        return ndimage.distance_transform_edt(mask, return_indices=return_indices)

    h, w = mask.shape
    tiled = np.tile(mask, (3, 3))
    out = ndimage.distance_transform_edt(tiled, return_indices=return_indices)

    if return_indices:
        dist, inds = out
        dist_center = dist[h:2 * h, w:2 * w]
        inds_center = inds[:, h:2 * h, w:2 * w].copy()
        inds_center[0] %= h
        inds_center[1] %= w
        return dist_center, inds_center

    return out[h:2 * h, w:2 * w]


def label_connected_components(
    mask: np.ndarray,
    *,
    structure: np.ndarray,
    connectivity: str,
    boundary_mode: str = "periodic",
) -> tuple[np.ndarray, int]:
    labels, n_labels = ndimage.label(mask, structure=structure)
    labels = labels.astype(np.int32, copy=False)

    mode = normalize_boundary_mode(boundary_mode)
    if mode == "open" or n_labels <= 1:
        return labels, int(n_labels)

    h, w = mask.shape
    parent = np.arange(n_labels + 1, dtype=np.int32)
    center_y = int(structure.shape[0] // 2)
    center_x = int(structure.shape[1] // 2)
    offsets = np.argwhere(structure > 0) - np.array([center_y, center_x], dtype=int)
    offsets = [tuple(int(v) for v in off) for off in offsets if not (off[0] == 0 and off[1] == 0)]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    if h > 1:
        for col in range(w):
            lab_a = int(labels[0, col])
            if lab_a == 0:
                continue
            for dy, dx in offsets:
                if dy != -1:
                    continue
                nbr_col = (col + dx) % w
                lab_b = int(labels[h - 1, nbr_col])
                if lab_b > 0:
                    union(lab_a, lab_b)

    if w > 1:
        for row in range(h):
            lab_a = int(labels[row, 0])
            if lab_a == 0:
                continue
            for dy, dx in offsets:
                if dx != -1:
                    continue
                nbr_row = (row + dy) % h
                lab_b = int(labels[nbr_row, w - 1])
                if lab_b > 0:
                    union(lab_a, lab_b)

    roots = np.arange(n_labels + 1, dtype=np.int32)
    for idx in range(1, n_labels + 1):
        roots[idx] = find(idx)

    unique_roots = np.unique(roots[1:])
    new_ids = np.zeros(n_labels + 1, dtype=np.int32)
    new_ids[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.int32)
    relabeled = new_ids[roots[labels]]
    return relabeled.astype(np.int32, copy=False), int(unique_roots.size)
