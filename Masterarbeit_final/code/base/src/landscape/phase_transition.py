# src/landscape/phase_transition_mixed.py
# run: .\.venv\Scripts\python.exe -m src.landscape.phase_transition

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from src.landscape.boundary import label_connected_components
from src.landscape.landscape import Landscape, FOREST


Direction = Literal["top_bottom", "left_right"]
Connectivity = Literal["4", "8"]


# ============================================================
# Cluster labeling + percolation + basic metrics (self-contained)
# ============================================================

def forest_cover(grid: np.ndarray) -> float:
    """Forest cover p = (#forest cells) / (#all cells)."""
    total = int(grid.size)
    if total == 0:
        return 0.0
    return float((grid == FOREST).sum()) / float(total)


def _structure(connectivity: Connectivity) -> np.ndarray:
    """Connected-component structure for 2D rasters."""
    if connectivity == "4":
        return np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=int)
    if connectivity == "8":
        return np.ones((3, 3), dtype=int)
    raise ValueError(f"connectivity must be '4' or '8', got {connectivity!r}")


def label_forest_clusters(
    grid: np.ndarray,
    *,
    connectivity: Connectivity,
    boundary_mode: str = "periodic",
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Label forest clusters on the full forest grid.

    Returns
    -------
    labels : int array, same shape as grid (0 = non-forest, 1..n = cluster id)
    n_labels : number of clusters
    sizes : int array (n_labels,), sizes[k] = size of cluster (k+1)
    """
    s = _structure(connectivity)
    labels, n = label_connected_components(
        grid == FOREST,
        structure=s,
        connectivity=connectivity,
        boundary_mode=boundary_mode,
    )
    if n == 0:
        return labels.astype(np.int32), 0, np.zeros(0, dtype=int)

    sizes = np.bincount(labels.ravel())[1:n + 1].astype(int)
    return labels.astype(np.int32), int(n), sizes


def spanning_labels(labels: np.ndarray, *, direction: Direction) -> set[int]:
    """Set of cluster labels that span the chosen direction."""
    if direction == "top_bottom":
        a = np.unique(labels[0, :])
        b = np.unique(labels[-1, :])
    elif direction == "left_right":
        a = np.unique(labels[:, 0])
        b = np.unique(labels[:, -1])
    else:
        raise ValueError(f"direction must be 'top_bottom' or 'left_right', got {direction!r}")

    a = set(int(x) for x in a if x > 0)
    b = set(int(x) for x in b if x > 0)
    return a & b


def percolates(labels: np.ndarray, *, direction: Direction) -> bool:
    """Simple percolation: any cluster spans the chosen direction?"""
    return len(spanning_labels(labels, direction=direction)) > 0


def landscape_division_index_total(grid: np.ndarray, sizes: np.ndarray) -> float:
    """
    LDI normalized by TOTAL landscape area (all cells):
      DI_total = 1 - sum_i s_i (s_i - 1) / (N (N - 1))
    """
    N = int(grid.size)
    if N <= 1 or sizes.size == 0:
        return 0.0
    num = float((sizes * (sizes - 1)).sum())
    den = float(N * (N - 1))
    return float(1.0 - num / den)


def landscape_division_index_habitat(sizes: np.ndarray) -> float:
    """
    LDI normalized by HABITAT (forest cells only):
      DI_hab = 1 - sum_i s_i (s_i - 1) / (H (H - 1)), H = sum(sizes)
    """
    H = int(sizes.sum())
    if H <= 1 or sizes.size == 0:
        return 0.0
    num = float((sizes * (sizes - 1)).sum())
    den = float(H * (H - 1))
    return float(1.0 - num / den)


def second_largest_cluster_size(
    sizes: np.ndarray,
    *,
    exclude_labels: Optional[set[int]] = None,
) -> int:
    """Second-largest cluster size; optionally exclude certain cluster labels (1-based)."""
    if sizes.size == 0:
        return 0

    if exclude_labels:
        keep = [sizes[i - 1] for i in range(1, sizes.size + 1) if i not in exclude_labels]
        s = np.array(keep, dtype=int)
    else:
        s = sizes

    if s.size < 2:
        return 0
    s_sorted = np.sort(s)[::-1]
    return int(s_sorted[1])


def susceptibility_chi(
    sizes: np.ndarray,
    *,
    exclude_labels: Optional[set[int]] = None,
) -> float:
    """
    Percolation-style susceptibility:
      chi = sum(s^2) / sum(s)
    excluding certain labels (typically spanning cluster(s); else exclude largest).
    """
    if sizes.size == 0:
        return 0.0

    if exclude_labels:
        keep = [sizes[i - 1] for i in range(1, sizes.size + 1) if i not in exclude_labels]
        s = np.array(keep, dtype=float)
    else:
        s = sizes.astype(float)

    if s.size == 0:
        return 0.0
    den = float(s.sum())
    if den <= 0.0:
        return 0.0
    return float((s * s).sum() / den)


# ============================================================
# Mixed deforestation: one fixed set of ops per step
# ============================================================

@dataclass(frozen=True)
class DeforestOp:
    """
    One operation applied every step (in given order).

    mode: deforestation mode, e.g. "random", "edge_random", "access_weighted"
    r: global rate (fraction of current forest removed, but restricted to candidates)
    selection_mode: "exact" or "stochastic"
    kwargs: forwarded into your deforestation API (e.g. k/eta for access_weighted)
    """
    mode: str
    r: float
    selection_mode: str = "exact"
    kwargs: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class Trajectory:
    p: np.ndarray
    percolates: np.ndarray
    di_total: np.ndarray
    di_habitat: np.ndarray
    chi: np.ndarray
    s2: np.ndarray
    n_clusters: np.ndarray


def simulate_mixed_deforestation_trajectory(
    landscape: Landscape,
    *,
    ops: list[DeforestOp],
    rng: np.random.Generator,
    connectivity: Connectivity = "4",
    direction: Direction = "top_bottom",
    max_steps: int = 50_000,
    stop_cover: float = 0.0,
    post_break_steps: int = 25,
    inplace: bool = False,
) -> Trajectory:
    """
    Simulate a trajectory where in EACH step you apply a fixed sequence of deforestation ops.

    We record metrics once per step (before applying ops), then apply all ops sequentially.

    Stop conditions:
      - p <= stop_cover
      - forest empty
      - max_steps reached
      - once percolation breaks, continue post_break_steps more steps (to capture peaks/kinks)
      - if a full step removes 0 cells (all masks empty), stop
    """
    if len(ops) == 0:
        raise ValueError("ops must contain at least one DeforestOp.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if post_break_steps < 0:
        raise ValueError("post_break_steps must be >= 0.")

    L = landscape if inplace else landscape.copy()

    p_list: list[float] = []
    perc_list: list[bool] = []
    diT_list: list[float] = []
    diH_list: list[float] = []
    chi_list: list[float] = []
    s2_list: list[int] = []
    nc_list: list[int] = []

    broke = False
    after_break = 0

    for _ in range(max_steps):
        grid = L.grid
        p = forest_cover(grid)

        labels, n, sizes = label_forest_clusters(grid, connectivity=connectivity)
        span = spanning_labels(labels, direction=direction)
        is_perc = len(span) > 0

        # Exclude spanning clusters; else exclude the largest cluster
        exclude = span
        if not exclude and sizes.size > 0:
            exclude = {int(np.argmax(sizes) + 1)}

        p_list.append(float(p))
        perc_list.append(bool(is_perc))
        diT_list.append(float(landscape_division_index_total(grid, sizes)))
        diH_list.append(float(landscape_division_index_habitat(sizes)))
        chi_list.append(float(susceptibility_chi(sizes, exclude_labels=exclude)))
        s2_list.append(int(second_largest_cluster_size(sizes, exclude_labels=exclude)))
        nc_list.append(int(n))

        # stop criteria based on current state
        if p <= float(stop_cover) or int((grid == FOREST).sum()) == 0:
            break

        if broke:
            after_break += 1
            if after_break >= post_break_steps:
                break
        else:
            if not is_perc:
                broke = True

        # apply all ops sequentially
        removed_total = 0
        for op in ops:
            if op.r <= 0.0:
                continue
            kw = op.kwargs or {}
            mask = L.apply_change(
                kind="deforest",
                mode=op.mode,
                r=float(op.r),
                rng=rng,
                selection_mode=op.selection_mode,
                inplace=True,
                **kw,
            )
            removed_total += int(mask.sum())

        if removed_total == 0:
            break

    return Trajectory(
        p=np.asarray(p_list, dtype=float),
        percolates=np.asarray(perc_list, dtype=bool),
        di_total=np.asarray(diT_list, dtype=float),
        di_habitat=np.asarray(diH_list, dtype=float),
        chi=np.asarray(chi_list, dtype=float),
        s2=np.asarray(s2_list, dtype=float),
        n_clusters=np.asarray(nc_list, dtype=float),
    )


# ============================================================
# Transition detection (incl. multiple LDI kinks)
# ============================================================

def _peak_p(p: np.ndarray, y: np.ndarray) -> float:
    if p.size == 0:
        return float("nan")
    idx = int(np.nanargmax(y))
    return float(p[idx])


def _kinks_p(
    p: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 3,
    min_separation: int = 2,
) -> np.ndarray:
    """
    Return up to k kink locations as peaks in |dy/dp|.

    - We compute discrete slope dy/dp.
    - Then find local maxima in abs(slope).
    - Return p midpoints between the two samples for each selected kink.
    """
    if p.size < 2 or k <= 0:
        return np.full(0, np.nan, dtype=float)

    dp = np.diff(p)
    dy = np.diff(y)

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = dy / dp

    s = np.abs(slope)
    if not np.any(np.isfinite(s)):
        return np.full(0, np.nan, dtype=float)

    # local maxima indices in slope array (length n-1)
    candidates = []
    for i in range(1, s.size - 1):
        if np.isfinite(s[i]) and (s[i] >= s[i - 1]) and (s[i] >= s[i + 1]):
            candidates.append(i)

    if len(candidates) == 0:
        # fallback: take global max
        i = int(np.nanargmax(s))
        return np.array([0.5 * (p[i] + p[i + 1])], dtype=float)

    # sort by strength
    candidates = sorted(candidates, key=lambda i: float(s[i]), reverse=True)

    chosen = []
    for idx in candidates:
        if all(abs(idx - c) >= int(min_separation) for c in chosen):
            chosen.append(idx)
        if len(chosen) >= k:
            break

    out = [0.5 * (p[i] + p[i + 1]) for i in chosen]
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class TransitionPoints:
    p_last_perc: float
    p_first_nonperc: float
    p_perc_mid: float

    p_peak_chi: float
    p_peak_s2: float

    # multiple kinks:
    p_kinks_di_total: np.ndarray      # shape (<=k,)
    p_kinks_di_habitat: np.ndarray    # shape (<=k,)


def detect_transition_points(traj: Trajectory, *, ldi_k: int = 3) -> TransitionPoints:
    p = traj.p
    perc = traj.percolates

    p_last_perc = float("nan")
    p_first_nonperc = float("nan")
    p_mid = float("nan")

    if p.size >= 2 and np.any(perc) and np.any(~perc):
        for i in range(1, p.size):
            if perc[i - 1] and (not perc[i]):
                p_last_perc = float(p[i - 1])
                p_first_nonperc = float(p[i])
                p_mid = 0.5 * (p_last_perc + p_first_nonperc)
                break

    return TransitionPoints(
        p_last_perc=p_last_perc,
        p_first_nonperc=p_first_nonperc,
        p_perc_mid=p_mid,
        p_peak_chi=_peak_p(p, traj.chi),
        p_peak_s2=_peak_p(p, traj.s2),
        p_kinks_di_total=_kinks_p(p, traj.di_total, k=ldi_k),
        p_kinks_di_habitat=_kinks_p(p, traj.di_habitat, k=ldi_k),
    )


# ============================================================
# Replicates + summaries
# ============================================================

@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    median: float
    ci95_low: float
    ci95_high: float
    n_valid: int


def _summarize(values: np.ndarray) -> SummaryStats:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return SummaryStats(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size >= 2 else 0.0
    med = float(np.median(v))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return SummaryStats(mean, std, med, float(lo), float(hi), int(v.size))


def _summarize_kinks(kinks_per_run: list[np.ndarray], *, k: int) -> list[SummaryStats]:
    """
    Summarize kink positions by rank (strongest, 2nd strongest, ...).

    We align by rank: kink[0] = strongest. If a run has fewer, it's NaN.
    """
    arr = np.full((len(kinks_per_run), k), np.nan, dtype=float)
    for i, ks in enumerate(kinks_per_run):
        m = min(k, int(ks.size))
        if m > 0:
            arr[i, :m] = ks[:m]
    return [_summarize(arr[:, j]) for j in range(k)]


@dataclass(frozen=True)
class MixedPhaseTransitionSummary:
    ops: list[DeforestOp]
    n_runs: int

    # percolation
    p_perc_mid: SummaryStats
    p_last_perc: SummaryStats
    p_first_nonperc: SummaryStats

    # peaks
    p_peak_chi: SummaryStats
    p_peak_s2: SummaryStats

    # kinks (ranked)
    p_kinks_di_total: list[SummaryStats]
    p_kinks_di_habitat: list[SummaryStats]


def estimate_phase_transition_mixed(
    landscape: Landscape,
    *,
    ops: list[DeforestOp],
    n_runs: int,
    seed: int = 0,
    connectivity: Connectivity = "4",
    direction: Direction = "top_bottom",
    max_steps: int = 50_000,
    stop_cover: float = 0.0,
    post_break_steps: int = 25,
    ldi_k: int = 3,
    return_raw: bool = False,
) -> MixedPhaseTransitionSummary | tuple[MixedPhaseTransitionSummary, list[TransitionPoints]]:
    """
    Many replicates for ONE fixed mixed-op recipe (ops), summarize transition p-values.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")
    if ldi_k < 1:
        raise ValueError("ldi_k must be >= 1.")

    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(n_runs)

    points: list[TransitionPoints] = []

    for i in range(n_runs):
        rng = np.random.default_rng(children[i])
        traj = simulate_mixed_deforestation_trajectory(
            landscape,
            ops=ops,
            rng=rng,
            connectivity=connectivity,
            direction=direction,
            max_steps=max_steps,
            stop_cover=stop_cover,
            post_break_steps=post_break_steps,
            inplace=False,
        )
        points.append(detect_transition_points(traj, ldi_k=ldi_k))

    p_mid = np.array([p.p_perc_mid for p in points], dtype=float)
    p_last = np.array([p.p_last_perc for p in points], dtype=float)
    p_first = np.array([p.p_first_nonperc for p in points], dtype=float)

    p_chi = np.array([p.p_peak_chi for p in points], dtype=float)
    p_s2 = np.array([p.p_peak_s2 for p in points], dtype=float)

    kT = [p.p_kinks_di_total for p in points]
    kH = [p.p_kinks_di_habitat for p in points]

    summary = MixedPhaseTransitionSummary(
        ops=ops,
        n_runs=int(n_runs),
        p_perc_mid=_summarize(p_mid),
        p_last_perc=_summarize(p_last),
        p_first_nonperc=_summarize(p_first),
        p_peak_chi=_summarize(p_chi),
        p_peak_s2=_summarize(p_s2),
        p_kinks_di_total=_summarize_kinks(kT, k=ldi_k),
        p_kinks_di_habitat=_summarize_kinks(kH, k=ldi_k),
    )

    return (summary, points) if return_raw else summary


# ============================================================
# Local example / smoke test
# ============================================================

if __name__ == "__main__":
    def _fmt(s: SummaryStats) -> str:
        return f"mean={s.mean:.4f}, std={s.std:.4f}, med={s.median:.4f}, CI95=[{s.ci95_low:.4f},{s.ci95_high:.4f}], n={s.n_valid}"

    # Start from full forest (clean percolating regime)
    L0 = Landscape(np.ones((120, 120), dtype=int))

    ops = [
        DeforestOp(mode="random", r=0.02, selection_mode="exact"),
        DeforestOp(mode="access_weighted", r=0.02, selection_mode="exact", kwargs={"k": 1, "eta": 1.0}),
    ]

    summary = estimate_phase_transition_mixed(
        L0,
        ops=ops,
        n_runs=40,
        seed=123,
        connectivity="4",
        direction="top_bottom",
        post_break_steps=40,
        ldi_k=3,
    )

    print("\n=== MIXED RECIPE ===")
    for op in summary.ops:
        print(f"- {op.mode}: r={op.r} ({op.selection_mode}) kwargs={op.kwargs}")

    print("\nPercolation:")
    print("  p_mid:   ", _fmt(summary.p_perc_mid))
    print("  p_last:  ", _fmt(summary.p_last_perc))
    print("  p_first: ", _fmt(summary.p_first_nonperc))

    print("\nPeaks:")
    print("  peak chi:", _fmt(summary.p_peak_chi))
    print("  peak S2: ", _fmt(summary.p_peak_s2))

    print("\nLDI kinks (ranked, strongest first):")
    for i, s in enumerate(summary.p_kinks_di_total, start=1):
        print(f"  DI_total kink #{i}:   {_fmt(s)}")
    for i, s in enumerate(summary.p_kinks_di_habitat, start=1):
        print(f"  DI_hab   kink #{i}:   {_fmt(s)}")

    print("\nDone.")
