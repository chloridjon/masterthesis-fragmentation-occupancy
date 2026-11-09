# src/landscape/deforestation.py
# start directly with .\.venv\Scripts\python.exe -m src.landscape.deforestation
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from src.landscape.boundary import (
    normalize_boundary_mode,
)

try:  # Optional acceleration; the pure-Python implementation remains the fallback.
    from numba import njit
except Exception:  # pragma: no cover - depends on optional local environment
    njit = None

# Grid encoding
FOREST = 1
NON_FOREST = 0


# ============================================================
# Neighborhood helpers
# ============================================================

FOUR_NEIGHBOR_KERNEL = np.array(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]],
    dtype=int,
)
FOUR_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbor_count(grid: np.ndarray, kernel: np.ndarray, boundary_mode: str = "periodic") -> np.ndarray:
    """Count forest neighbors using the given kernel."""
    mode = normalize_boundary_mode(boundary_mode)
    if mode == "periodic":
        return convolve(grid, kernel, mode="wrap")
    return convolve(grid, kernel, mode="constant", cval=0)


def update_neighbor_count_after_change(
    neighbor_count: np.ndarray,
    changed_mask: np.ndarray,
    *,
    delta: int,
    boundary_mode: str = "periodic",
) -> None:
    """
    Incrementally update 4-neighbor forest counts after cells changed state.

    delta=-1 for deforestation, delta=+1 for reforestation. The count of a cell
    depends only on its four neighbors, so only neighbors of changed cells need
    updates. Multiple changed cells are handled correctly because each changed
    cell contributes one +/- update to every adjacent cell.
    """
    if neighbor_count.ndim != 2:
        raise ValueError("neighbor_count must be a 2D array.")
    changed = np.asarray(changed_mask, dtype=bool)
    if changed.shape != neighbor_count.shape:
        raise ValueError("changed_mask must have same shape as neighbor_count.")

    mode = normalize_boundary_mode(boundary_mode)
    h, w = neighbor_count.shape
    if h == 0 or w == 0 or not changed.any():
        return

    ys, xs = np.nonzero(changed)
    d = int(delta)
    for dy, dx in FOUR_NEIGHBOR_OFFSETS:
        ny = ys + dy
        nx = xs + dx
        if mode == "periodic":
            np.add.at(neighbor_count, (ny % h, nx % w), d)
        else:
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            if np.any(valid):
                np.add.at(neighbor_count, (ny[valid], nx[valid]), d)


def _without_neighbor_count(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "neighbor_count" not in kwargs:
        return kwargs
    out = dict(kwargs)
    out.pop("neighbor_count", None)
    return out


# ============================================================
# Candidate builders (NO randomness)
# ============================================================

def candidates_random(grid: np.ndarray, **_: Any) -> np.ndarray:
    """All forest cells are candidates."""
    return grid == FOREST


def candidates_edge(
    grid: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    boundary_mode: str = "periodic",
    neighbor_count: Optional[np.ndarray] = None,
    count_canvas_boundary_as_edge: bool = True,
    **_: Any,
) -> np.ndarray:
    """
    Forest edge cells are candidates.
    Uses 4-neighborhood (von Neumann).
    """
    kernel = FOUR_NEIGHBOR_KERNEL if kernel is None else kernel
    nbh = neighbor_count if neighbor_count is not None else _neighbor_count(grid, kernel, boundary_mode=boundary_mode)
    max_n = int(kernel.sum())

    # Edge = forest cell with fewer than max_n forest neighbors
    if not bool(count_canvas_boundary_as_edge) and normalize_boundary_mode(boundary_mode) == "open":
        valid_neighbor_count = convolve(np.ones_like(grid, dtype=int), kernel, mode="constant", cval=0)
        return (grid == FOREST) & (nbh < valid_neighbor_count)
    return (grid == FOREST) & (nbh < max_n)


def candidates_ref_edge(
    grid: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    boundary_mode: str = "periodic",
    neighbor_count: Optional[np.ndarray] = None,
    **_: Any,
) -> np.ndarray:
    """Reforestation candidates: non-forest cells adjacent to forest."""
    kernel = FOUR_NEIGHBOR_KERNEL if kernel is None else kernel
    nbh = neighbor_count if neighbor_count is not None else _neighbor_count(grid, kernel, boundary_mode=boundary_mode)
    max_n = int(kernel.sum())

    return (grid == NON_FOREST) & (nbh > 0) & (nbh < max_n)


# ============================================================
# Weight builders (optional; used by weighted selection)
# ============================================================

if njit is not None:
    @njit(cache=True)
    def _accessibility_weights_front_numba(
        grid: np.ndarray,
        candidates: np.ndarray,
        k: int,
        lambda_decay: float,
        eta: float,
        periodic: bool,
    ) -> np.ndarray:
        h = grid.shape[0]
        w = grid.shape[1]
        n_cells = h * w
        weights = np.zeros((h, w), dtype=np.float64)
        visited = np.zeros(n_cells, dtype=np.int32)
        frontier = np.empty(n_cells, dtype=np.int64)
        next_frontier = np.empty(n_cells, dtype=np.int64)
        stamp = 1

        layer_weights = np.empty(k, dtype=np.float64)
        for i in range(k):
            layer_weights[i] = np.exp(-lambda_decay * float(i))

        for cy in range(h):
            for cx in range(w):
                if not candidates[cy, cx]:
                    continue

                stamp += 1
                if stamp >= 2147483000:
                    visited[:] = 0
                    stamp = 1

                frontier_count = 0
                for direction in range(4):
                    if direction == 0:
                        ny = cy - 1
                        nx = cx
                    elif direction == 1:
                        ny = cy + 1
                        nx = cx
                    elif direction == 2:
                        ny = cy
                        nx = cx - 1
                    else:
                        ny = cy
                        nx = cx + 1

                    if periodic:
                        if ny < 0:
                            ny = h - 1
                        elif ny >= h:
                            ny = 0
                        if nx < 0:
                            nx = w - 1
                        elif nx >= w:
                            nx = 0
                    elif ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue

                    if grid[ny, nx] == NON_FOREST:
                        flat = ny * w + nx
                        if visited[flat] != stamp:
                            visited[flat] = stamp
                            frontier[frontier_count] = flat
                            frontier_count += 1

                if frontier_count == 0:
                    continue

                accessibility = layer_weights[0] * float(frontier_count)
                for depth in range(2, k + 1):
                    next_count = 0
                    for idx in range(frontier_count):
                        flat = frontier[idx]
                        y = flat // w
                        x = flat - y * w

                        for direction in range(4):
                            if direction == 0:
                                ny = y - 1
                                nx = x
                            elif direction == 1:
                                ny = y + 1
                                nx = x
                            elif direction == 2:
                                ny = y
                                nx = x - 1
                            else:
                                ny = y
                                nx = x + 1

                            if periodic:
                                if ny < 0:
                                    ny = h - 1
                                elif ny >= h:
                                    ny = 0
                                if nx < 0:
                                    nx = w - 1
                                elif nx >= w:
                                    nx = 0
                            elif ny < 0 or ny >= h or nx < 0 or nx >= w:
                                continue

                            if grid[ny, nx] == NON_FOREST:
                                n_flat = ny * w + nx
                                if visited[n_flat] != stamp:
                                    visited[n_flat] = stamp
                                    next_frontier[next_count] = n_flat
                                    next_count += 1

                    if next_count == 0:
                        break

                    accessibility += layer_weights[depth - 1] * float(next_count)
                    for idx in range(next_count):
                        frontier[idx] = next_frontier[idx]
                    frontier_count = next_count

                weights[cy, cx] = accessibility ** eta

        return weights

    @njit(cache=True)
    def _accessibility_weights_selected_numba(
        grid: np.ndarray,
        candidate_flats: np.ndarray,
        k: int,
        layer_weights: np.ndarray,
        periodic: bool,
        visited: np.ndarray,
        frontier: np.ndarray,
        next_frontier: np.ndarray,
        stamp: int,
    ) -> tuple[np.ndarray, int]:
        """Compute weights only for supplied candidates using reusable scratch."""
        h = grid.shape[0]
        w = grid.shape[1]
        result = np.zeros(candidate_flats.size, dtype=np.float64)

        for candidate_index in range(candidate_flats.size):
            stamp += 1
            if stamp >= 2147483000:
                visited[:] = 0
                stamp = 1

            candidate_flat = candidate_flats[candidate_index]
            cy = candidate_flat // w
            cx = candidate_flat - cy * w
            frontier_count = 0
            for direction in range(4):
                if direction == 0:
                    ny = cy - 1
                    nx = cx
                elif direction == 1:
                    ny = cy + 1
                    nx = cx
                elif direction == 2:
                    ny = cy
                    nx = cx - 1
                else:
                    ny = cy
                    nx = cx + 1

                if periodic:
                    if ny < 0:
                        ny = h - 1
                    elif ny >= h:
                        ny = 0
                    if nx < 0:
                        nx = w - 1
                    elif nx >= w:
                        nx = 0
                elif ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue

                if grid[ny, nx] == NON_FOREST:
                    open_flat = ny * w + nx
                    if visited[open_flat] != stamp:
                        visited[open_flat] = stamp
                        frontier[frontier_count] = open_flat
                        frontier_count += 1

            if frontier_count == 0:
                continue

            accessibility = layer_weights[0] * float(frontier_count)
            for depth in range(2, k + 1):
                next_count = 0
                for frontier_index in range(frontier_count):
                    open_flat = frontier[frontier_index]
                    y = open_flat // w
                    x = open_flat - y * w
                    for direction in range(4):
                        if direction == 0:
                            ny = y - 1
                            nx = x
                        elif direction == 1:
                            ny = y + 1
                            nx = x
                        elif direction == 2:
                            ny = y
                            nx = x - 1
                        else:
                            ny = y
                            nx = x + 1

                        if periodic:
                            if ny < 0:
                                ny = h - 1
                            elif ny >= h:
                                ny = 0
                            if nx < 0:
                                nx = w - 1
                            elif nx >= w:
                                nx = 0
                        elif ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue

                        if grid[ny, nx] == NON_FOREST:
                            neighbor_flat = ny * w + nx
                            if visited[neighbor_flat] != stamp:
                                visited[neighbor_flat] = stamp
                                next_frontier[next_count] = neighbor_flat
                                next_count += 1

                if next_count == 0:
                    break
                accessibility += layer_weights[depth - 1] * float(next_count)
                for frontier_index in range(next_count):
                    frontier[frontier_index] = next_frontier[frontier_index]
                frontier_count = next_count

            result[candidate_index] = accessibility

        return result, stamp
else:
    _accessibility_weights_front_numba = None
    _accessibility_weights_selected_numba = None

def weights_accessibility_front(
    grid: np.ndarray,
    candidates: np.ndarray,
    k: int = 1,
    lambda_decay: float = 0.0,
    eta: float = 1.0,
    kernel: Optional[np.ndarray] = None,
    boundary_mode: str = "periodic",
    neighbor_count: Optional[np.ndarray] = None,
    **_: Any,
) -> np.ndarray:
    """
    Accessibility-based weights for forest edge candidates.

    For each edge candidate x, explore connected non-forest cells up to depth k
    starting from the open neighbors of x and compute

      A_k(x) = sum_{r=1..k} exp(-lambda_decay * (r-1)) * o_r(x)
      q(x)   = A_k(x)^eta

    where o_r(x) counts open cells reachable in exactly r open-cell steps.

    Notes
    -----
    - Uses 4-neighborhood.
    - For k=1 this reduces to counting direct open neighbors, i.e. the old
      edge-weighted method with exponent eta.
    """
    if k < 1:
        raise ValueError("k must be >= 1 for accessibility-based edge deforestation.")
    if lambda_decay < 0.0:
        raise ValueError("lambda_decay must be >= 0.")
    if eta < 0.0:
        raise ValueError("eta must be >= 0.")

    kernel = FOUR_NEIGHBOR_KERNEL if kernel is None else kernel
    nbh = neighbor_count if neighbor_count is not None else _neighbor_count(grid, kernel, boundary_mode=boundary_mode)
    max_n = int(kernel.sum())
    open_neighbor_count = (max_n - nbh).astype(int, copy=False)

    weights = np.zeros_like(grid, dtype=float)
    if not candidates.any():
        return weights

    # Fast path: k=1 is direct open-neighbor weighting.
    if int(k) == 1:
        weights[candidates] = open_neighbor_count[candidates].astype(float) ** float(eta)
        return weights

    if _accessibility_weights_front_numba is not None:
        try:
            return _accessibility_weights_front_numba(
                np.asarray(grid, dtype=np.int8),
                np.asarray(candidates, dtype=np.bool_),
                int(k),
                float(lambda_decay),
                float(eta),
                normalize_boundary_mode(boundary_mode) == "periodic",
            )
        except Exception:
            # Keep optional acceleration non-disruptive; correctness comes from the fallback below.
            pass

    mode = normalize_boundary_mode(boundary_mode)
    open_mask = (grid == NON_FOREST)
    h, w = grid.shape
    n_cells = int(h * w)
    open_flat = open_mask.ravel()
    visit_stamp = np.zeros(n_cells, dtype=np.int32)
    stamp = np.int32(1)
    layer_weights = np.exp(-float(lambda_decay) * np.arange(int(k), dtype=float))
    max_stamp = np.iinfo(np.int32).max

    def _open_neighbor_flats(flat_idx: int) -> tuple[int, ...]:
        y, x = divmod(int(flat_idx), w)
        out: list[int] = []
        for dy, dx in FOUR_NEIGHBOR_OFFSETS:
            ny = y + dy
            nx = x + dx
            if mode == "periodic":
                ny %= h
                nx %= w
            elif ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue

            n_flat = ny * w + nx
            if open_flat[n_flat] and n_flat not in out:
                out.append(n_flat)
        return tuple(out)

    open_neighbor_cache: dict[int, tuple[int, ...]] = {}
    accessibility_cache: dict[tuple[int, ...], float] = {}

    def _cached_open_neighbor_flats(flat_idx: int) -> tuple[int, ...]:
        cached = open_neighbor_cache.get(flat_idx)
        if cached is None:
            cached = _open_neighbor_flats(flat_idx)
            open_neighbor_cache[flat_idx] = cached
        return cached

    def _accessibility_from_starts(starts: tuple[int, ...]) -> float:
        nonlocal stamp
        if not starts:
            return 0.0
        cached = accessibility_cache.get(starts)
        if cached is not None:
            return cached

        if stamp == max_stamp:
            visit_stamp.fill(0)
            stamp = np.int32(1)

        frontier: list[int] = []
        for flat_idx in starts:
            if visit_stamp[flat_idx] != stamp:
                visit_stamp[flat_idx] = stamp
                frontier.append(flat_idx)

        accessibility = layer_weights[0] * float(len(frontier))
        for depth in range(2, int(k) + 1):
            next_frontier: list[int] = []
            for flat_idx in frontier:
                for n_flat in _cached_open_neighbor_flats(flat_idx):
                    if visit_stamp[n_flat] != stamp:
                        visit_stamp[n_flat] = stamp
                        next_frontier.append(n_flat)

            if not next_frontier:
                break
            accessibility += layer_weights[depth - 1] * float(len(next_frontier))
            frontier = next_frontier

        stamp = np.int32(stamp + 1)
        accessibility_cache[starts] = float(accessibility)
        return float(accessibility)

    for cand_flat in np.flatnonzero(candidates):
        cy, cx = divmod(int(cand_flat), w)
        starts = _cached_open_neighbor_flats(int(cand_flat))

        if not starts:
            continue

        accessibility = _accessibility_from_starts(starts)
        weights[cy, cx] = accessibility ** float(eta)

    return weights


# ============================================================
# Selection helpers (global rate; exact / stochastic)
# ============================================================

def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    """Create a boolean mask from integer coordinates (N, 2)."""
    mask = np.zeros(shape, dtype=bool)
    if len(coords) > 0:
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _mask_from_flat_indices(shape: tuple[int, int], flat_indices: np.ndarray) -> np.ndarray:
    """Create a boolean mask from flat indices into a 2D array."""
    mask = np.zeros(shape, dtype=bool)
    if flat_indices.size > 0:
        mask.reshape(-1)[flat_indices] = True
    return mask


def _base_count_for_rate(grid: np.ndarray, current_base: np.ndarray, rate_base: str) -> int:
    """
    Return the population size used to translate r into a target cell count.

    rate_base="current" keeps the historical behavior: r is applied to the
    current changeable state, e.g. current forest for deforestation.
    rate_base="grid" applies r to the total grid size, yielding a fixed target
    count per step until capped by available candidates.
    """
    key = str(rate_base).lower()
    if key in {"current", "current_state", "habitat", "forest", "available"}:
        return int(current_base.sum())
    if key in {"grid", "total", "total_grid", "grid_size"}:
        return int(grid.size)
    raise ValueError("rate_base must be 'current' or 'grid'.")


def _exclude_protected_candidates(candidates: np.ndarray, protected_mask: Optional[np.ndarray]) -> np.ndarray:
    """Remove protected cells from a candidate mask."""
    if protected_mask is None:
        return candidates
    protected = np.asarray(protected_mask, dtype=bool)
    if protected.shape != candidates.shape:
        raise ValueError("protected_mask must have same shape as grid.")
    return candidates & ~protected


def make_gaussian_pressure_field(
    shape: tuple[int, int],
    sigma: float,
    rng: np.random.Generator,
    boundary_mode: str = "periodic",
) -> np.ndarray:
    """Create a smoothed random field for optional heterogeneous deforestation pressure."""
    if len(shape) != 2:
        raise ValueError("pressure field shape must be 2D.")
    raw = rng.standard_normal((int(shape[0]), int(shape[1])))
    mode = "wrap" if normalize_boundary_mode(boundary_mode) == "periodic" else "nearest"
    return gaussian_filter(raw, sigma=max(0.0, float(sigma)), mode=mode).astype(float, copy=False)


def pressure_weight_multiplier(
    grid: np.ndarray,
    pressure_field: Optional[np.ndarray],
    pressure_beta: float = 0.0,
) -> Optional[np.ndarray]:
    """Return exp(beta * z-scored pressure_field), or None when disabled."""
    if pressure_field is None or float(pressure_beta) <= 0.0:
        return None
    pressure = np.asarray(pressure_field, dtype=float)
    if pressure.shape != grid.shape:
        raise ValueError("pressure_field must have same shape as grid.")
    finite = np.isfinite(pressure)
    if not finite.any():
        return None
    mean = float(np.nanmean(pressure))
    std = float(np.nanstd(pressure))
    if not np.isfinite(std) or std <= 0.0:
        return None
    z = (pressure - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    exponent = np.clip(float(pressure_beta) * z, -50.0, 50.0)
    return np.exp(exponent).astype(float, copy=False)


def deforestation_edge_density(grid: np.ndarray, boundary_mode: str = "periodic") -> float:
    """
    Edge density used for optional dynamic deforestation rates.

    This matches landscape_stats.compute_adjacency_metrics(...):
    edge_density_total = forest/non-forest adjacencies / total raster cells.
    """
    g = np.asarray(grid) == FOREST

    mode = normalize_boundary_mode(boundary_mode)
    if mode == "periodic":
        right = np.roll(g, -1, axis=1)
        down = np.roll(g, -1, axis=0)
        adj_forest_nonforest = float((g != right).sum() + (g != down).sum())
    else:
        horizontal_fn = g[:, 1:] != g[:, :-1]
        vertical_fn = g[1:, :] != g[:-1, :]
        adj_forest_nonforest = float(horizontal_fn.sum() + vertical_fn.sum())

    total_cells = int(g.size)
    if total_cells == 0:
        return 0.0
    return float(adj_forest_nonforest) / float(total_cells)


def effective_total_deforestation_rate(
    grid: np.ndarray,
    r0: float,
    rate_base: str = "current",
    *,
    edge_density_r: float = 0.0,
    edge_density_gamma: float = 1.0,
    boundary_mode: str = "periodic",
) -> float:
    """
    Return the total deforestation rate for the current step.

    For rate_base="edge_density", r0 is augmented as
    r_t = r0 + edge_density_r * edge_density_total**edge_density_gamma.
    Other rate bases keep r0 unchanged.
    """
    key = str(rate_base).lower()
    if key not in {"edge_density", "edge_density_total", "ed"}:
        return max(0.0, float(r0))
    gamma = float(edge_density_gamma)
    if gamma < 0.0:
        raise ValueError("edge_density_gamma must be >= 0.")
    edge_density = deforestation_edge_density(grid, boundary_mode=boundary_mode)
    return max(0.0, float(r0) + float(edge_density_r) * (edge_density ** gamma))


def exact_selection_global_uniform(
    grid: np.ndarray,
    candidates: np.ndarray,
    r: float,
    rng: np.random.Generator,
    rate_base: str = "current",
) -> np.ndarray:
    """
    Exact selection with global rate r, uniform over candidates.

    By default, r is interpreted as fraction of ALL current forest cells that
    should be changed. With rate_base="grid", r is interpreted as fraction of
    the total grid size.
    Selection is restricted to allowed candidate cells.
    """
    forest = (grid == FOREST)
    n_base = _base_count_for_rate(grid, forest, rate_base)

    allowed_flat = np.flatnonzero(forest & candidates)
    n_allowed = int(allowed_flat.size)

    if n_base == 0 or n_allowed == 0 or r <= 0:
        return np.zeros_like(grid, dtype=bool)

    n_target = int(np.round(n_base * r))
    n_pick = max(0, min(n_target, n_allowed))
    if n_pick == 0:
        return np.zeros_like(grid, dtype=bool)

    chosen = allowed_flat[rng.choice(n_allowed, size=n_pick, replace=False)]
    return _mask_from_flat_indices(grid.shape, chosen)


def stochastic_selection_global_uniform(
    grid: np.ndarray,
    candidates: np.ndarray,
    r: float,
    rng: np.random.Generator,
    rate_base: str = "current",
) -> np.ndarray:
    """
    Stochastic selection with global rate r, uniform over candidates.

    Each allowed candidate is changed independently with probability p,
    chosen such that E[total] = r * n_base (capped if not enough candidates).
    """
    forest = (grid == FOREST)
    n_base = _base_count_for_rate(grid, forest, rate_base)

    allowed = forest & candidates
    allowed_flat = np.flatnonzero(allowed)
    n_allowed = int(allowed_flat.size)

    if n_base == 0 or n_allowed == 0 or r <= 0:
        return np.zeros_like(grid, dtype=bool)

    expected_total = float(r) * float(n_base)
    p = float(np.clip(expected_total / n_allowed, 0.0, 1.0))

    selected = rng.random(n_allowed) < p
    return _mask_from_flat_indices(grid.shape, allowed_flat[selected])


def exact_selection_global_weighted(
    grid: np.ndarray,
    candidates: np.ndarray,
    weights: np.ndarray,
    r: float,
    rng: np.random.Generator,
    rate_base: str = "current",
) -> np.ndarray:
    """Exact selection with global rate r using weighted sampling without replacement."""
    forest = (grid == FOREST)
    n_base = _base_count_for_rate(grid, forest, rate_base)

    allowed_flat = np.flatnonzero(forest & candidates)
    n_allowed = int(allowed_flat.size)

    if n_base == 0 or n_allowed == 0 or r <= 0:
        return np.zeros_like(grid, dtype=bool)

    n_target = int(np.round(n_base * r))
    n_pick = max(0, min(n_target, n_allowed))
    if n_pick == 0:
        return np.zeros_like(grid, dtype=bool)

    if weights.shape != grid.shape:
        raise ValueError("weights must have same shape as grid.")

    w = weights.reshape(-1)[allowed_flat].astype(float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    wsum = float(w.sum())

    # Fallback to uniform if weights are degenerate
    if not np.isfinite(wsum) or wsum <= 0:
        chosen = allowed_flat[rng.choice(n_allowed, size=n_pick, replace=False)]
        return _mask_from_flat_indices(grid.shape, chosen)

    positive = w > 0.0
    positive_flat = allowed_flat[positive]
    positive_weights = w[positive]
    n_positive = int(positive_flat.size)
    n_pick = min(n_pick, n_positive)
    if n_pick == 0:
        return np.zeros_like(grid, dtype=bool)

    p = positive_weights / float(positive_weights.sum())

    chosen = positive_flat[rng.choice(n_positive, size=n_pick, replace=False, p=p)]
    return _mask_from_flat_indices(grid.shape, chosen)


def stochastic_selection_global_weighted(
    grid: np.ndarray,
    candidates: np.ndarray,
    weights: np.ndarray,
    r: float,
    rng: np.random.Generator,
    rate_base: str = "current",
) -> np.ndarray:
    """
    Stochastic weighted selection:
      - normalize weights over allowed to sum to 1
      - p_i = expected_total * w_i_norm clipped to [0,1]
      - expected_total = r * n_base
    """
    forest = (grid == FOREST)
    n_base = _base_count_for_rate(grid, forest, rate_base)

    allowed_flat = np.flatnonzero(forest & candidates)
    n_allowed = int(allowed_flat.size)

    if n_base == 0 or n_allowed == 0 or r <= 0:
        return np.zeros_like(grid, dtype=bool)

    if weights.shape != grid.shape:
        raise ValueError("weights must have same shape as grid.")

    w = weights.reshape(-1)[allowed_flat].astype(float)
    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        w_norm = np.full(n_allowed, 1.0 / n_allowed, dtype=float)
    else:
        w_norm = w / wsum

    expected_total = float(r) * float(n_base)
    p_allowed = np.clip(expected_total * w_norm, 0.0, 1.0)
    selected = rng.random(n_allowed) < p_allowed
    return _mask_from_flat_indices(grid.shape, allowed_flat[selected])


# ============================================================
# Legacy / utility batch API: build masks + apply them
#
# These rate-based selectors are retained for backwards compatibility and
# experiments. They are NOT used to generate final thesis landscapes. The
# primary generator is run_sequential_deforestation in landscape/generation.py.
# ============================================================

def deforest_mask(
    grid: np.ndarray,
    mode: str,
    r: float,
    rng: np.random.Generator,
    selection_mode: str = "exact",
    rate_base: str = "current",
    **kwargs: Any,
) -> np.ndarray:
    """
    Legacy/utility API: build a batch mask using mode + global rate r.

    Not used for final landscape generation. Final trajectories use a forced
    initial random event followed by one-cell Bernoulli(rho) events.

    rate_base:
      - "current": historical behavior, r is based on current forest cells
      - "grid": r is based on total grid size, giving a fixed target count

    Modes:
      - "random"        uniform over all forest
      - "edge_random"   uniform over edge candidates
      - "access_weighted" accessibility-weighted over edge candidates
                          (k, lambda_decay, eta supported)
    """
    protected_mask = kwargs.get("protected_mask")
    pressure_mult = pressure_weight_multiplier(
        grid,
        kwargs.get("pressure_field"),
        float(kwargs.get("pressure_beta", 0.0)),
    )

    if mode == "random":
        candidates = _exclude_protected_candidates(candidates_random(grid, **kwargs), protected_mask)
        if pressure_mult is not None:
            if selection_mode == "exact":
                return exact_selection_global_weighted(grid, candidates, pressure_mult, r, rng, rate_base=rate_base)
            if selection_mode == "stochastic":
                return stochastic_selection_global_weighted(grid, candidates, pressure_mult, r, rng, rate_base=rate_base)
            raise ValueError(f"Unknown selection_mode: {selection_mode}")
        if selection_mode == "exact":
            return exact_selection_global_uniform(grid, candidates, r, rng, rate_base=rate_base)
        if selection_mode == "stochastic":
            return stochastic_selection_global_uniform(grid, candidates, r, rng, rate_base=rate_base)
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    if mode == "edge_random":
        edge_kwargs = dict(kwargs)
        edge_kwargs.setdefault("count_canvas_boundary_as_edge", False)
        candidates = _exclude_protected_candidates(candidates_edge(grid, **edge_kwargs), protected_mask)
        if pressure_mult is not None:
            if selection_mode == "exact":
                return exact_selection_global_weighted(grid, candidates, pressure_mult, r, rng, rate_base=rate_base)
            if selection_mode == "stochastic":
                return stochastic_selection_global_weighted(grid, candidates, pressure_mult, r, rng, rate_base=rate_base)
            raise ValueError(f"Unknown selection_mode: {selection_mode}")
        if selection_mode == "exact":
            return exact_selection_global_uniform(grid, candidates, r, rng, rate_base=rate_base)
        if selection_mode == "stochastic":
            return stochastic_selection_global_uniform(grid, candidates, r, rng, rate_base=rate_base)
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    if mode == "access_weighted":
        kernel = kwargs.get("kernel", FOUR_NEIGHBOR_KERNEL)
        boundary_mode = kwargs.get("boundary_mode", "periodic")
        nbh = kwargs.get("neighbor_count")
        if nbh is None:
            nbh = _neighbor_count(grid, kernel, boundary_mode=boundary_mode)
        clean_kwargs = _without_neighbor_count(kwargs)
        candidates = _exclude_protected_candidates(candidates_edge(grid, neighbor_count=nbh, **clean_kwargs), protected_mask)
        weights = weights_accessibility_front(grid, candidates=candidates, neighbor_count=nbh, **clean_kwargs)
        if pressure_mult is not None:
            weights = weights * pressure_mult
        if selection_mode == "exact":
            return exact_selection_global_weighted(grid, candidates, weights, r, rng, rate_base=rate_base)
        if selection_mode == "stochastic":
            return stochastic_selection_global_weighted(grid, candidates, weights, r, rng, rate_base=rate_base)
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    raise ValueError(f"Unknown deforestation mode: {mode}")


def mixed_deforestation_step(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    r_random: float,
    r_edge: float,
    edge_mode: str,
    selection_mode: str = "exact",
    rate_base: str = "current",
    step_composition: str = "sequential",
    random_kwargs: Optional[dict[str, Any]] = None,
    edge_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy/utility API: apply a mixed random + directed batch step.

    Not used for final landscape generation. ``r_random`` and ``r_edge`` are
    batch fractions, not the event probability ``rho``.

    step_composition:
      - "sequential": historical behavior. Random deforestation is applied
        first; the edge/correlated mask is then built on the updated grid.
      - "simultaneous_disjoint": both masks are built on the same input grid;
        random-selected cells are excluded from the edge/correlated mask before
        applying the union.
    """
    composition = str(step_composition).lower()
    if composition == "simultaneous":
        composition = "simultaneous_disjoint"
    if composition not in {"sequential", "simultaneous_disjoint"}:
        raise ValueError("step_composition must be 'sequential' or 'simultaneous_disjoint'.")

    random_kwargs = dict(random_kwargs or {})
    edge_kwargs = dict(edge_kwargs or {})
    base = np.asarray(grid)
    empty = np.zeros_like(base, dtype=bool)

    if composition == "sequential":
        out = base.copy()
        mask_random = empty
        mask_edge = empty

        if float(r_random) > 0 and int((out == FOREST).sum()) > 0:
            mask_random = deforest_mask(
                grid=out,
                mode="random",
                r=float(r_random),
                rng=rng,
                selection_mode=selection_mode,
                rate_base=rate_base,
                **random_kwargs,
            )
            out[mask_random] = NON_FOREST

        if float(r_edge) > 0 and int((out == FOREST).sum()) > 0:
            mask_edge = deforest_mask(
                grid=out,
                mode=edge_mode,
                r=float(r_edge),
                rng=rng,
                selection_mode=selection_mode,
                rate_base=rate_base,
                **edge_kwargs,
            )
            out[mask_edge] = NON_FOREST

        return out, mask_random, mask_edge

    mask_random = empty
    if float(r_random) > 0 and int((base == FOREST).sum()) > 0:
        mask_random = deforest_mask(
            grid=base,
            mode="random",
            r=float(r_random),
            rng=rng,
            selection_mode=selection_mode,
            rate_base=rate_base,
            **random_kwargs,
        )

    mask_edge = empty
    if float(r_edge) > 0 and int((base == FOREST).sum()) > 0:
        edge_kwargs_disjoint = dict(edge_kwargs)
        existing_protected = edge_kwargs_disjoint.get("protected_mask")
        if existing_protected is None:
            protected = mask_random
        else:
            protected = np.asarray(existing_protected, dtype=bool) | mask_random
        edge_kwargs_disjoint["protected_mask"] = protected
        mask_edge = deforest_mask(
            grid=base,
            mode=edge_mode,
            r=float(r_edge),
            rng=rng,
            selection_mode=selection_mode,
            rate_base=rate_base,
            **edge_kwargs_disjoint,
        )

    out = base.copy()
    out[mask_random | mask_edge] = NON_FOREST
    return out, mask_random, mask_edge


# --- NOTE: reforestation needs a different "base population" than deforestation. ---

def _exact_selection_global_uniform_on_base(
    base_mask: np.ndarray,
    candidates: np.ndarray,
    r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exact selection where r is a fraction of base_mask.sum().
    Selection happens only on (base_mask & candidates).
    """
    n_base = int(base_mask.sum())
    allowed_flat = np.flatnonzero(base_mask & candidates)
    n_allowed = int(allowed_flat.size)

    if n_base == 0 or n_allowed == 0 or r <= 0:
        return np.zeros_like(base_mask, dtype=bool)

    n_target = int(np.round(n_base * r))
    n_pick = max(0, min(n_target, n_allowed))
    if n_pick == 0:
        return np.zeros_like(base_mask, dtype=bool)

    chosen = allowed_flat[rng.choice(n_allowed, size=n_pick, replace=False)]
    return _mask_from_flat_indices(base_mask.shape, chosen)


def _stochastic_selection_global_uniform_on_base(
    base_mask: np.ndarray,
    candidates: np.ndarray,
    r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stochastic selection where E[total] = r * base_mask.sum(),
    applied uniformly over allowed = (base_mask & candidates).
    """
    n_base = int(base_mask.sum())
    allowed_flat = np.flatnonzero(base_mask & candidates)
    n_allowed = int(allowed_flat.size)

    if n_base == 0 or n_allowed == 0 or r <= 0:
        return np.zeros_like(base_mask, dtype=bool)

    expected_total = float(r) * float(n_base)
    p = float(np.clip(expected_total / n_allowed, 0.0, 1.0))
    selected = rng.random(n_allowed) < p
    return _mask_from_flat_indices(base_mask.shape, allowed_flat[selected])


def reforest_mask(
    grid: np.ndarray,
    mode: str,
    r: float,
    rng: np.random.Generator,
    selection_mode: str = "exact",
    **kwargs: Any,
) -> np.ndarray:
    """
    Build a reforestation mask.

    Interpretation of r:
      fraction of ALL non-forest cells to be reforested (subject to candidates).

    Modes:
      - "ref_edge_random"
    """
    if mode != "ref_edge_random":
        raise ValueError(f"Unknown reforestation mode: {mode}")

    kernel = kwargs.get("kernel", FOUR_NEIGHBOR_KERNEL)
    boundary_mode = kwargs.get("boundary_mode", "periodic")
    nbh = kwargs.get("neighbor_count")
    if nbh is None:
        nbh = _neighbor_count(grid, kernel, boundary_mode=boundary_mode)
    candidates = candidates_ref_edge(grid, neighbor_count=nbh, **_without_neighbor_count(kwargs))
    base = (grid == NON_FOREST)  # <- base population for reforestation

    if selection_mode == "exact":
        return _exact_selection_global_uniform_on_base(base, candidates, r, rng)
    if selection_mode == "stochastic":
        return _stochastic_selection_global_uniform_on_base(base, candidates, r, rng)

    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def apply_deforestation(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set selected forest cells to NON_FOREST."""
    if mask.dtype != bool:
        raise TypeError(f"mask must be boolean, got {mask.dtype}")
    new_grid = grid.copy()
    new_grid[mask] = NON_FOREST
    return new_grid


def apply_reforestation(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set selected non-forest cells to FOREST."""
    new_grid = grid.copy()
    new_grid[mask.astype(bool)] = FOREST
    return new_grid


@dataclass
class GridChangeState:
    """
    Stateful grid updater for simulation loops.

    It keeps a 4-neighbor forest-count array current across steps, avoiding a
    full convolution for edge/accessibility candidate construction after every
    small local change. It is intentionally limited to the default 4-neighbor
    kernel used by the deforestation model.
    """

    grid: np.ndarray
    boundary_mode: str = "periodic"
    neighbor_count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid = (np.asarray(self.grid) > 0).astype(np.int8, copy=True)
        self.boundary_mode = normalize_boundary_mode(self.boundary_mode)
        self.neighbor_count = _neighbor_count(
            self.grid,
            FOUR_NEIGHBOR_KERNEL,
            boundary_mode=self.boundary_mode,
        ).astype(np.int16, copy=False)

    def apply_change(
        self,
        *,
        kind: str,
        mode: str,
        r: float,
        rng: np.random.Generator,
        selection_mode: str = "exact",
        protected_mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if protected_mask is not None and protected_mask.shape != self.grid.shape:
            raise ValueError("protected_mask must have same shape as grid.")

        kwargs = dict(kwargs)
        kwargs.setdefault("boundary_mode", self.boundary_mode)
        use_cached_neighbors = kwargs.get("kernel", FOUR_NEIGHBOR_KERNEL) is FOUR_NEIGHBOR_KERNEL
        if use_cached_neighbors:
            kwargs["neighbor_count"] = self.neighbor_count

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
            mask = np.asarray(mask, dtype=bool)
            if mask.any():
                self.grid[mask] = NON_FOREST
                if use_cached_neighbors:
                    update_neighbor_count_after_change(
                        self.neighbor_count,
                        mask,
                        delta=-1,
                        boundary_mode=self.boundary_mode,
                    )
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
            mask = np.asarray(mask, dtype=bool)
            if mask.any():
                self.grid[mask] = FOREST
                if use_cached_neighbors:
                    update_neighbor_count_after_change(
                        self.neighbor_count,
                        mask,
                        delta=1,
                        boundary_mode=self.boundary_mode,
                    )
            return mask

        raise ValueError("kind must be 'deforest' or 'reforest'.")


class _DenseIndexSet:
    """Mutable flat-index set with O(1) sampling, insertion and removal."""

    def __init__(self, mask: np.ndarray):
        flat = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.intp, copy=False)
        self._items = np.empty(mask.size, dtype=np.intp)
        self._items[: flat.size] = flat
        self._positions = np.full(mask.size, -1, dtype=np.intp)
        self._positions[flat] = np.arange(flat.size, dtype=np.intp)
        self._size = int(flat.size)

    def __len__(self) -> int:
        return self._size

    @property
    def indices(self) -> np.ndarray:
        """Active indices; callers must treat the returned view as read-only."""
        return self._items[: self._size]

    @property
    def positions(self) -> np.ndarray:
        """Reverse index map (-1 for absent cells); treat as read-only."""
        return self._positions

    def add(self, flat: int) -> None:
        flat = int(flat)
        if self._positions[flat] >= 0:
            return
        self._items[self._size] = flat
        self._positions[flat] = self._size
        self._size += 1

    def discard(self, flat: int) -> None:
        flat = int(flat)
        position = int(self._positions[flat])
        if position < 0:
            return
        last_position = self._size - 1
        last_flat = int(self._items[last_position])
        if position != last_position:
            self._items[position] = last_flat
            self._positions[last_flat] = position
        self._positions[flat] = -1
        self._size = last_position

    def sample(self, rng: np.random.Generator) -> Optional[int]:
        if self._size == 0:
            return None
        return int(self._items[int(rng.integers(self._size))])


class _FenwickTree:
    """Dynamic non-negative weights over fixed flat cell indices."""

    def __init__(self, values: np.ndarray):
        flat = np.asarray(values, dtype=float).reshape(-1)
        if np.any(~np.isfinite(flat)) or np.any(flat < 0.0):
            raise ValueError("Fenwick weights must be finite and non-negative.")
        self._values = flat.copy()
        self._tree = np.zeros(flat.size + 1, dtype=float)
        prefix = np.empty(flat.size + 1, dtype=float)
        prefix[0] = 0.0
        np.cumsum(flat, out=prefix[1:])
        if flat.size:
            tree_indices = np.arange(1, flat.size + 1, dtype=np.intp)
            lower = tree_indices - (tree_indices & -tree_indices)
            self._tree[1:] = prefix[tree_indices] - prefix[lower]
        self._total = float(prefix[-1])

    @property
    def total(self) -> float:
        return float(self._total)

    @property
    def values(self) -> np.ndarray:
        return self._values

    def update(self, flat: int, value: float) -> None:
        flat = int(flat)
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Fenwick weights must be finite and non-negative.")
        delta = value - float(self._values[flat])
        if delta == 0.0:
            return
        self._values[flat] = value
        self._total += delta
        tree_index = flat + 1
        while tree_index < self._tree.size:
            self._tree[tree_index] += delta
            tree_index += tree_index & -tree_index

    def sample(self, rng: np.random.Generator) -> Optional[int]:
        """Draw a flat index proportionally to its positive weight."""
        if not np.isfinite(self._total) or self._total <= 0.0:
            return None
        target = float(rng.random()) * self._total
        tree_index = 0
        bit = 1 << (self._values.size.bit_length() - 1)
        while bit:
            next_index = tree_index + bit
            if next_index < self._tree.size and self._tree[next_index] <= target:
                target -= float(self._tree[next_index])
                tree_index = next_index
            bit >>= 1
        if tree_index >= self._values.size:
            return None
        return int(tree_index)


class DeforestationEventState(GridChangeState):
    """Incremental state for true, sequential single-cell deforestation.

    Forest and edge candidates are stored in swap-delete index sets. When
    accessibility weighting is enabled, weights live in a Fenwick tree, so a
    proportional draw and each changed weight cost O(log N).

    A forest -> non-forest transition at ``z`` can affect an existing
    candidate's finite-depth accessibility only if a newly available or
    shortened open path of length at most ``k`` passes through ``z``. Thus the
    candidate must border an open cell reachable from ``z`` through the
    updated non-forest graph in at most ``k - 1`` steps. The reverse search in
    :meth:`_accessibility_affected_candidates` enumerates exactly that set;
    candidates outside it cannot have a changed depth-1..k open frontier.

    Accessibility tracking is initialized immediately for
    ``mode='access_weighted'`` and lazily on the first later weighted draw.
    Once enabled it remains current even when random or edge-random events are
    interleaved, so mechanisms can be switched without rebuilding the state.
    """

    MODES = frozenset({"random", "edge_random", "access_weighted"})

    def __init__(
        self,
        grid: np.ndarray,
        mode: str = "random",
        boundary_mode: str = "periodic",
        k: int = 1,
        lambda_decay: float = 0.0,
        eta: float = 1.0,
        count_canvas_boundary_as_edge: bool = False,
    ) -> None:
        super().__init__(grid=grid, boundary_mode=boundary_mode)
        self.mode = self._validated_mode(mode)
        self.k = int(k)
        self.lambda_decay = float(lambda_decay)
        self.eta = float(eta)
        if self.k < 1:
            raise ValueError("k must be >= 1 for accessibility-based edge deforestation.")
        if self.lambda_decay < 0.0:
            raise ValueError("lambda_decay must be >= 0.")
        if self.eta < 0.0:
            raise ValueError("eta must be >= 0.")
        self.count_canvas_boundary_as_edge = bool(count_canvas_boundary_as_edge)

        self._height, self._width = self.grid.shape
        self._n_cells = int(self.grid.size)
        self._forest = _DenseIndexSet(self.grid == FOREST)

        if self.boundary_mode == "open" and not self.count_canvas_boundary_as_edge:
            self._edge_neighbor_limit = convolve(
                np.ones_like(self.grid, dtype=np.int8),
                FOUR_NEIGHBOR_KERNEL,
                mode="constant",
                cval=0,
            ).astype(np.int16, copy=False)
        else:
            self._edge_neighbor_limit = np.full(self.grid.shape, 4, dtype=np.int16)

        self.edge_candidate_mask = (self.grid == FOREST) & (
            self.neighbor_count < self._edge_neighbor_limit
        )
        self._edge = _DenseIndexSet(self.edge_candidate_mask)

        self.accessibility_weights = np.zeros(self.grid.shape, dtype=float)
        self.weighted_sampler = _FenwickTree(self.accessibility_weights)
        self._accessibility_enabled = False
        self._nonfinite_weight_count = 0
        self._layer_weights = np.exp(
            -self.lambda_decay * np.arange(self.k, dtype=float)
        )
        self._access_visit_stamp = np.zeros(self._n_cells, dtype=np.int32)
        self._access_frontier = np.empty(self._n_cells, dtype=np.int64)
        self._access_next_frontier = np.empty(self._n_cells, dtype=np.int64)
        self._access_stamp = 0
        if self.mode == "access_weighted":
            self._enable_accessibility()

    @classmethod
    def _validated_mode(cls, mode: str) -> str:
        key = str(mode).lower()
        if key not in cls.MODES:
            raise ValueError(f"Unknown deforestation mode: {mode}")
        return key

    @property
    def forest_indices(self) -> np.ndarray:
        return self._forest.indices

    @property
    def forest_positions(self) -> np.ndarray:
        return self._forest.positions

    @property
    def edge_candidate_indices(self) -> np.ndarray:
        return self._edge.indices

    @property
    def edge_indices(self) -> np.ndarray:
        return self._edge.indices

    @property
    def edge_positions(self) -> np.ndarray:
        return self._edge.positions

    @property
    def accessibility_enabled(self) -> bool:
        return self._accessibility_enabled

    def set_mode(self, mode: str) -> None:
        """Set the default mechanism without rebuilding grid/index state."""
        self.mode = self._validated_mode(mode)
        if self.mode == "access_weighted":
            self._enable_accessibility()

    def apply_change(self, **_: Any) -> np.ndarray:
        """Disallow the parent batch mutator, which would bypass event caches."""
        raise NotImplementedError(
            "DeforestationEventState is single-event only; use deforest_one()."
        )

    def _neighbor_flats(self, flat: int) -> tuple[int, ...]:
        y, x = divmod(int(flat), self._width)
        neighbors: list[int] = []
        for dy, dx in FOUR_NEIGHBOR_OFFSETS:
            ny = y + dy
            nx = x + dx
            if self.boundary_mode == "periodic":
                ny %= self._height
                nx %= self._width
            elif ny < 0 or ny >= self._height or nx < 0 or nx >= self._width:
                continue
            neighbor = ny * self._width + nx
            if neighbor not in neighbors:
                neighbors.append(neighbor)
        return tuple(neighbors)

    def _is_edge(self, flat: int) -> bool:
        y, x = divmod(int(flat), self._width)
        return bool(
            self.grid[y, x] == FOREST
            and self.neighbor_count[y, x] < self._edge_neighbor_limit[y, x]
        )

    def _set_edge_status(self, flat: int) -> None:
        y, x = divmod(int(flat), self._width)
        is_edge = self._is_edge(flat)
        self.edge_candidate_mask[y, x] = is_edge
        if is_edge:
            self._edge.add(flat)
        else:
            self._edge.discard(flat)

    def _enable_accessibility(self) -> None:
        if self._accessibility_enabled:
            return
        self.accessibility_weights = weights_accessibility_front(
            self.grid,
            candidates=self.edge_candidate_mask,
            k=self.k,
            lambda_decay=self.lambda_decay,
            eta=self.eta,
            boundary_mode=self.boundary_mode,
            neighbor_count=self.neighbor_count,
        )
        flat_weights = self.accessibility_weights.reshape(-1)
        finite_sampler_weights = np.where(
            np.isfinite(flat_weights) & (flat_weights > 0.0),
            flat_weights,
            0.0,
        )
        self.weighted_sampler = _FenwickTree(finite_sampler_weights)
        edge_flat = self.edge_candidate_mask.reshape(-1)
        self._nonfinite_weight_count = int(
            np.count_nonzero(edge_flat & ~np.isfinite(flat_weights))
        )
        self._accessibility_enabled = True

    def _accessibility_weight_one(self, flat: int) -> float:
        if not self._is_edge(flat):
            return 0.0
        y, x = divmod(int(flat), self._width)

        # This intentionally retains the existing open-boundary k=1
        # numerical behavior: missing canvas neighbors contribute through
        # 4 - neighbor_count, although edge candidacy can ignore the canvas.
        if self.k == 1:
            accessibility = float(4 - int(self.neighbor_count[y, x]))
            return accessibility ** self.eta

        grid_flat = self.grid.reshape(-1)
        visited: set[int] = set()
        frontier: list[int] = []
        for neighbor in self._neighbor_flats(flat):
            if grid_flat[neighbor] == NON_FOREST and neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
        if not frontier:
            return 0.0

        accessibility = self._layer_weights[0] * float(len(frontier))
        for depth in range(2, self.k + 1):
            next_frontier: list[int] = []
            for open_flat in frontier:
                for neighbor in self._neighbor_flats(open_flat):
                    if grid_flat[neighbor] == NON_FOREST and neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            if not next_frontier:
                break
            accessibility += self._layer_weights[depth - 1] * float(len(next_frontier))
            frontier = next_frontier
        return float(accessibility) ** self.eta

    def _set_accessibility_weight(self, flat: int, weight: float) -> None:
        flat_weights = self.accessibility_weights.reshape(-1)
        old_weight = float(flat_weights[flat])
        old_nonfinite = not np.isfinite(old_weight)
        new_nonfinite = not np.isfinite(weight)
        if old_nonfinite != new_nonfinite:
            self._nonfinite_weight_count += 1 if new_nonfinite else -1
        flat_weights[flat] = weight
        sampler_weight = float(weight) if np.isfinite(weight) and weight > 0.0 else 0.0
        self.weighted_sampler.update(flat, sampler_weight)

    def _accessibility_affected_candidates(self, changed_flat: int) -> set[int]:
        """Return precisely the current forest cells whose weight may change."""
        grid_flat = self.grid.reshape(-1)
        visited_open = {int(changed_flat)}
        frontier = [int(changed_flat)]
        affected: set[int] = set()

        # An open cell at graph distance d from z can contribute to a forest
        # candidate one step farther away. Hence d <= k - 1 is necessary and
        # sufficient for that candidate's depth-k frontier to involve z.
        for open_depth in range(self.k):
            next_frontier: list[int] = []
            for open_flat in frontier:
                for neighbor in self._neighbor_flats(open_flat):
                    if grid_flat[neighbor] == FOREST:
                        affected.add(neighbor)
                    elif open_depth + 1 < self.k and neighbor not in visited_open:
                        visited_open.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return affected

    def _recompute_accessibility_weights(self, candidates: set[int]) -> None:
        if not candidates:
            return
        candidate_flats = np.fromiter(sorted(candidates), dtype=np.int64)
        if self.k > 1 and _accessibility_weights_selected_numba is not None:
            accessibility, self._access_stamp = _accessibility_weights_selected_numba(
                self.grid,
                candidate_flats,
                self.k,
                self._layer_weights,
                self.boundary_mode == "periodic",
                self._access_visit_stamp,
                self._access_frontier,
                self._access_next_frontier,
                self._access_stamp,
            )
            weights = accessibility ** self.eta
            for candidate, weight in zip(candidate_flats, weights):
                self._set_accessibility_weight(int(candidate), float(weight))
            return

        for candidate in candidate_flats:
            self._set_accessibility_weight(
                int(candidate),
                self._accessibility_weight_one(int(candidate)),
            )

    def _apply_deforestation_event(self, changed_flat: int) -> None:
        changed_flat = int(changed_flat)
        grid_flat = self.grid.reshape(-1)
        if grid_flat[changed_flat] != FOREST:
            raise ValueError("changed cell must currently be forest.")

        y, x = divmod(changed_flat, self._width)
        grid_flat[changed_flat] = NON_FOREST
        self._forest.discard(changed_flat)

        # Keep the four directional contributions, including coincident
        # wrapped neighbors on degenerate periodic grids, consistent with the
        # convolution/update_neighbor_count_after_change implementation.
        for dy, dx in FOUR_NEIGHBOR_OFFSETS:
            ny = y + dy
            nx = x + dx
            if self.boundary_mode == "periodic":
                ny %= self._height
                nx %= self._width
            elif ny < 0 or ny >= self._height or nx < 0 or nx >= self._width:
                continue
            self.neighbor_count[ny, nx] -= 1

        self._set_edge_status(changed_flat)
        for neighbor in self._neighbor_flats(changed_flat):
            self._set_edge_status(neighbor)

        if self._accessibility_enabled:
            self._set_accessibility_weight(changed_flat, 0.0)
            self._recompute_accessibility_weights(
                self._accessibility_affected_candidates(changed_flat)
            )

    def deforest_one(
        self,
        rng: np.random.Generator,
        mode: Optional[str] = None,
    ) -> Optional[int]:
        """Remove one cell using the requested current distribution.

        Returns its flat index, or ``None`` when the selected mechanism has no
        valid current candidate. A per-call ``mode`` does not change the
        configured default, which makes mixed event controllers straightforward.
        """
        selected_mode = self.mode if mode is None else self._validated_mode(mode)
        if selected_mode == "random":
            changed_flat = self._forest.sample(rng)
        elif selected_mode == "edge_random":
            changed_flat = self._edge.sample(rng)
        else:
            self._enable_accessibility()
            if self._nonfinite_weight_count:
                # Match exact_selection_global_weighted: a non-finite total
                # falls back to a uniform draw over all allowed candidates.
                changed_flat = self._edge.sample(rng)
            else:
                changed_flat = self.weighted_sampler.sample(rng)
                if changed_flat is None:
                    changed_flat = self._edge.sample(rng)

        if changed_flat is None:
            return None
        self._apply_deforestation_event(changed_flat)
        return int(changed_flat)


def step_deforestation(
    grid: np.ndarray,
    mode: str,
    r: float,
    rng: np.random.Generator,
    selection_mode: str = "exact",
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy batch-rate convenience wrapper; not a final-data generator."""
    mask = deforest_mask(
        grid=grid,
        mode=mode,
        r=r,
        rng=rng,
        selection_mode=selection_mode,
        **kwargs,
    )
    return apply_deforestation(grid, mask), mask

if __name__ == "__main__":
    import numpy as _np

    def _assert(cond: bool, msg: str) -> None:
        assert cond, msg

    def _assert_close(a: float, b: float, tol: float = 1e-9, msg: str = "") -> None:
        assert abs(float(a) - float(b)) <= tol, msg or f"Expected {b}, got {a}"

    def _count_forest(grid: _np.ndarray) -> int:
        return int((grid == 1).sum())

    def _count_nonforest(grid: _np.ndarray) -> int:
        return int((grid == 0).sum())

    # ------------------------------------------------------------
    # 1) Candidate builders (edge vs random)
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

    cand_all = candidates_random(grid)
    _assert(cand_all.dtype == bool, "candidates_random must return bool mask")
    _assert(int(cand_all.sum()) == _count_forest(grid), "candidates_random should include all forest cells")

    cand_edge = candidates_edge(grid)
    _assert(cand_edge.dtype == bool, "candidates_edge must return bool mask")
    _assert(int(cand_edge.sum()) <= _count_forest(grid), "edge candidates cannot exceed forest cells")
    _assert(_np.all(grid[cand_edge] == 1), "edge candidates must be forest cells")

    # If the grid has any "interior" forest (rare in tiny grids), edge candidates should be smaller
    # We don't enforce strict inequality because the grid may have no interior.
    print("[OK] candidates_random / candidates_edge")

    # Reforestation candidates must be non-forest and adjacent to forest
    cand_ref = candidates_ref_edge(grid)
    _assert(_np.all(grid[cand_ref] == 0), "reforest candidates must be non-forest cells")
    _assert(int(cand_ref.sum()) >= 0, "reforest candidates mask invalid")
    print("[OK] candidates_ref_edge")

    # ------------------------------------------------------------
    # 2) Accessibility weights
    # ------------------------------------------------------------
    eta_test = 1.7
    w_access_k1 = weights_accessibility_front(
        grid,
        candidates=cand_edge,
        k=1,
        lambda_decay=0.7,
        eta=eta_test,
    )
    nbh_test = _neighbor_count(grid, FOUR_NEIGHBOR_KERNEL)
    expected_k1 = _np.zeros_like(grid, dtype=float)
    expected_k1[cand_edge] = (int(FOUR_NEIGHBOR_KERNEL.sum()) - nbh_test[cand_edge]).astype(float) ** eta_test
    _assert(w_access_k1.shape == grid.shape, "weights must match grid shape")
    _assert(_np.all(w_access_k1[~cand_edge] == 0.0), "weights outside candidates must be 0")
    _assert(_np.all(w_access_k1[cand_edge] >= 0.0), "weights must be non-negative")
    _assert_close(
        float(_np.abs(w_access_k1 - expected_k1).max()),
        0.0,
        msg="access_weighted with k=1 must match direct open-neighbor weighting",
    )
    print("[OK] weights_accessibility_front k=1 special case")

    full_grid = _np.ones((4, 4), dtype=int)
    cand_edge_periodic = candidates_edge(full_grid)
    _assert(int(cand_edge_periodic.sum()) == 0, "periodic full forest should have no edge candidates")
    print("[OK] periodic edge candidates")

    cand_edge_open_canvas_only = candidates_edge(
        full_grid,
        boundary_mode="open",
        count_canvas_boundary_as_edge=False,
    )
    _assert(
        int(cand_edge_open_canvas_only.sum()) == 0,
        "open full forest should have no edge candidates when the canvas boundary is ignored",
    )
    full_with_hole = full_grid.copy()
    full_with_hole[1, 1] = 0
    cand_hole = candidates_edge(
        full_with_hole,
        boundary_mode="open",
        count_canvas_boundary_as_edge=False,
    )
    _assert(int(cand_hole.sum()) == 4, "only forest cells adjacent to the internal hole should be edge candidates")
    mask_open_edge = deforest_mask(
        grid=full_grid,
        mode="edge_random",
        r=0.5,
        rng=_np.random.default_rng(11),
        selection_mode="exact",
        boundary_mode="open",
    )
    _assert(
        int(mask_open_edge.sum()) == 0,
        "edge_random/open must not treat the outside of the canvas as a deforestation edge",
    )
    print("[OK] open edge_random ignores canvas boundary")

    checker = _np.array([[1, 0], [0, 1]], dtype=int)
    _assert_close(
        deforestation_edge_density(checker, boundary_mode="periodic"),
        2.0,
        msg="periodic checkerboard edge density should match adjacency metric definition",
    )
    edge_rate = effective_total_deforestation_rate(
        checker,
        0.01,
        "edge_density",
        edge_density_r=0.02,
        edge_density_gamma=1.0,
        boundary_mode="periodic",
    )
    _assert_close(edge_rate, 0.05, msg="edge-density dynamic rate mismatch")
    print("[OK] edge-density dynamic rate")

    # ------------------------------------------------------------
    # 3) Exact selection: count should match round(n_base * r) capped by candidates
    # ------------------------------------------------------------
    rng = _np.random.default_rng(0)

    n_forest0 = _count_forest(grid)
    r = 0.30
    target = int(_np.round(n_forest0 * r))

    # random mode: candidates == all forest, so exact should pick exactly target
    mask = deforest_mask(grid=grid, mode="random", r=r, rng=rng, selection_mode="exact")
    _assert(mask.dtype == bool, "deforest_mask must return boolean mask")
    _assert(int(mask.sum()) == target, "exact uniform (random) should pick round(n_forest*r)")
    _assert(_np.all(grid[mask] == 1), "deforestation mask must select forest cells")
    print("[OK] deforest_mask random/exact")

    # With rate_base="grid", r maps to a fixed fraction of total grid cells.
    rng = _np.random.default_rng(0)
    grid_target = int(_np.round(grid.size * r))
    mask_grid_base = deforest_mask(
        grid=grid,
        mode="random",
        r=r,
        rng=rng,
        selection_mode="exact",
        rate_base="grid",
    )
    _assert(int(mask_grid_base.sum()) == grid_target, "rate_base='grid' should pick round(grid.size*r)")
    _assert(_np.all(grid[mask_grid_base] == 1), "grid-based deforestation must select forest cells")
    print("[OK] deforest_mask random/exact rate_base=grid")

    # Mixed simultaneous mode should build disjoint masks on the same input grid.
    rng = _np.random.default_rng(1)
    grid_mixed, mask_mixed_random, mask_mixed_edge = mixed_deforestation_step(
        grid,
        rng,
        r_random=0.20,
        r_edge=0.20,
        edge_mode="edge_random",
        selection_mode="exact",
        rate_base="current",
        step_composition="simultaneous_disjoint",
        random_kwargs={"boundary_mode": "periodic"},
        edge_kwargs={"boundary_mode": "periodic"},
    )
    _assert(not _np.any(mask_mixed_random & mask_mixed_edge), "simultaneous_disjoint masks must not overlap")
    _assert(
        _count_forest(grid_mixed) == n_forest0 - int((mask_mixed_random | mask_mixed_edge).sum()),
        "mixed_deforestation_step must apply the union of both masks",
    )
    print("[OK] mixed_deforestation_step simultaneous_disjoint")

    pressure = make_gaussian_pressure_field(grid.shape, sigma=1.5, rng=_np.random.default_rng(3))
    _assert(pressure.shape == grid.shape, "pressure field must match requested shape")
    mult = pressure_weight_multiplier(grid, pressure, pressure_beta=0.5)
    _assert(mult is not None and mult.shape == grid.shape, "pressure multiplier must match grid shape")
    mask_pressure = deforest_mask(
        grid=grid,
        mode="access_weighted",
        r=0.2,
        rng=_np.random.default_rng(4),
        selection_mode="exact",
        pressure_field=pressure,
        pressure_beta=0.5,
        k=1,
        eta=1.0,
    )
    _assert(mask_pressure.dtype == bool, "pressure-weighted mask must be boolean")
    _assert(_np.all(grid[mask_pressure] == 1), "pressure-weighted deforestation must select forest cells")
    print("[OK] heterogeneous pressure weighting")

    # edge_random: may have fewer candidates than target -> must be capped
    rng = _np.random.default_rng(0)
    mask_edge = deforest_mask(grid=grid, mode="edge_random", r=r, rng=rng, selection_mode="exact")
    _assert(int(mask_edge.sum()) <= target, "edge_random should not exceed target")
    _assert(_np.all(grid[mask_edge] == 1), "edge_random mask must select forest cells")
    _assert(_np.all(cand_edge[mask_edge]), "edge_random must be subset of edge candidates")
    print("[OK] deforest_mask edge_random/exact")

    # access_weighted k=1: also subset of edge candidates
    rng = _np.random.default_rng(0)
    mask_w = deforest_mask(
        grid=grid,
        mode="access_weighted",
        r=r,
        rng=rng,
        selection_mode="exact",
        k=1,
        eta=1.0,
    )
    _assert(int(mask_w.sum()) <= target, "access_weighted k=1 should not exceed target")
    _assert(_np.all(cand_edge[mask_w]), "access_weighted k=1 must be subset of edge candidates")
    print("[OK] deforest_mask access_weighted k=1/exact")

    rng = _np.random.default_rng(0)
    mask_access = deforest_mask(
        grid=grid,
        mode="access_weighted",
        r=r,
        rng=rng,
        selection_mode="exact",
        k=3,
        lambda_decay=0.5,
        eta=1.2,
    )
    _assert(int(mask_access.sum()) <= target, "access_weighted should not exceed target")
    _assert(_np.all(cand_edge[mask_access]), "access_weighted must be subset of edge candidates")
    print("[OK] deforest_mask access_weighted/exact")

    # ------------------------------------------------------------
    # 4) Stochastic selection: should be subset + expected count in a plausible range
    # ------------------------------------------------------------
    # We don't enforce exact counts due to randomness; we just check basic plausibility.
    rng = _np.random.default_rng(42)
    mask_stoch = deforest_mask(grid=grid, mode="random", r=0.5, rng=rng, selection_mode="stochastic")
    _assert(mask_stoch.dtype == bool, "stochastic mask must be boolean")
    _assert(_np.all(grid[mask_stoch] == 1), "stochastic deforestation mask must select forest cells")
    _assert(int(mask_stoch.sum()) <= n_forest0, "stochastic cannot select more than forest cells")
    print("[OK] deforest_mask random/stochastic")

    # ------------------------------------------------------------
    # 5) Apply functions preserve shape and enforce types
    # ------------------------------------------------------------
    grid2 = apply_deforestation(grid, mask)
    _assert(grid2.shape == grid.shape, "apply_deforestation shape mismatch")
    _assert(_count_forest(grid2) == n_forest0 - int(mask.sum()), "apply_deforestation forest count mismatch")

    # apply_deforestation should reject non-bool masks
    try:
        _ = apply_deforestation(grid, mask.astype(int))
        raise AssertionError("apply_deforestation should raise TypeError for non-bool masks")
    except TypeError:
        pass

    # Reforestation apply: set mask cells to forest
    rng = _np.random.default_rng(7)
    n_non0 = _count_nonforest(grid)
    mask_ref2 = reforest_mask(grid=grid, mode="ref_edge_random", r=0.25, rng=rng, selection_mode="exact")
    grid3 = apply_reforestation(grid, mask_ref2)
    _assert(grid3.shape == grid.shape, "apply_reforestation shape mismatch")
    _assert(_count_forest(grid3) >= n_forest0, "reforestation should not decrease forest")
    _assert(int(mask_ref2.sum()) <= int(_np.round(n_non0 * 0.25)), "reforest exact should be capped by candidates")
    print("[OK] apply_deforestation / apply_reforestation")

    print("\nAll deforestation.py tests passed.")
