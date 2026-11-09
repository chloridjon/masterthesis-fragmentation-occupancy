# src/landscape/generation.py
# run: .\.venv\Scripts\python.exe -m src.landscape.generation

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from src.landscape.boundary import label_connected_components
from src.landscape.deforestation import DeforestationEventState
from src.landscape.landscape import Landscape, FOREST


Direction = Literal["top_bottom", "left_right"]
SEQUENTIAL_DEFORESTATION_GENERATOR_VERSION = "rho_sequential_single_cell_v2"


def forest_cover(grid: np.ndarray) -> float:
    """Fraction of forest cells (FOREST=1) in the full raster."""
    total = int(grid.size)
    if total == 0:
        return 0.0
    return float((grid == FOREST).sum()) / float(total)


FOUR_NEIGHBOR_STRUCTURE = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
    dtype=int,
)


def percolates(grid: np.ndarray, *, direction: Direction, boundary_mode: str = "periodic") -> bool:
    """
    Percolation check on the FULL forest grid (not core habitat).
    True if at least one forest cluster touches both opposite borders.
    """
    labels, n = label_connected_components(
        grid == FOREST,
        structure=FOUR_NEIGHBOR_STRUCTURE,
        connectivity="4",
        boundary_mode=boundary_mode,
    )
    if n == 0:
        return False

    if direction == "top_bottom":
        a = np.unique(labels[0, :])
        b = np.unique(labels[-1, :])
    elif direction == "left_right":
        a = np.unique(labels[:, 0])
        b = np.unique(labels[:, -1])
    else:
        raise ValueError(f"direction must be 'top_bottom' or 'left_right', got {direction!r}")

    a = set(a[a > 0])
    b = set(b[b > 0])
    return len(a & b) > 0


@dataclass
class GenerationHistory:
    """
    Minimal history for debugging / analysis.
    All arrays have length (n_steps+1) for covers/percolation and (n_steps) for removed.
    """
    forest_cover: list[float]
    percolates: list[bool]
    removed_cells: list[int]


@dataclass(frozen=True)
class SequentialDeforestationSnapshot:
    """One stored state from a single sequential deforestation trajectory."""

    requested_habitat_fractions: tuple[float, ...]
    target_forest_cells: int
    habitat_fraction: float
    forest_cells: int
    event_count: int
    reasons: tuple[str, ...]
    grid: np.ndarray


@dataclass(frozen=True)
class SequentialDeforestationResult:
    """Final grid, requested snapshots, event history, and mechanism counters."""

    grid: np.ndarray
    grid_shape: tuple[int, int]
    L: Optional[int]
    snapshots: tuple[SequentialDeforestationSnapshot, ...]
    changed_flat_indices: np.ndarray
    event_modes: tuple[str, ...]
    initial_forest_cells: int
    target_forest_cells: int
    final_forest_cells: int
    total_cells: int
    target_habitat_fraction: float
    final_habitat_fraction: float
    n_events: int
    n_random_total: int
    n_random_forced_initial: int
    n_random_stochastic: int
    n_directed: int
    generator_version: str
    forced_initial_random: bool
    generation_boundary: str
    rho: float
    realized_random_fraction: float
    realized_directed_fraction: float
    realized_random_fraction_post_initial: float
    realized_directed_fraction_post_initial: float
    directed_mode: str
    k: Optional[int]
    lambda_decay: Optional[float]
    eta: Optional[float]
    expected_events_per_stochastic_random: float
    seed: Optional[int]
    rng_bit_generator: str

    @property
    def L_random(self) -> float:
        """Expected post-initial events per stochastic random event."""
        return self.expected_events_per_stochastic_random


class SequentialDeforestationBlockedError(RuntimeError):
    """Raised when the selected mechanism has no candidate; no fallback occurs."""

    def __init__(self, diagnostics: dict[str, object]):
        self.diagnostics = dict(diagnostics)
        details = ", ".join(f"{key}={value!r}" for key, value in self.diagnostics.items())
        super().__init__(f"Sequential deforestation event is blocked ({details}).")


@dataclass(frozen=True)
class SequentialDeforestationEvent:
    """One applied single-cell event from the shared incremental controller."""

    event_number: int
    changed_flat_index: int
    mode: str
    forest_cells: int
    habitat_fraction: float


class SequentialDeforestationController:
    """Stateful controller shared by batch generation and interactive apps.

    Event one is random when ``forced_initial_random`` is enabled. Each later
    event draws ``Bernoulli(rho)`` exactly once to choose the directed mode;
    the selected mechanism then removes exactly one forest cell.
    """

    def __init__(
        self,
        grid: np.ndarray,
        rng: np.random.Generator,
        *,
        rho: float,
        directed_mode: str,
        generation_boundary: str = "periodic",
        k: int = 3,
        lambda_decay: float = 0.5,
        eta: float = 1.0,
        seed: Optional[int] = None,
        forced_initial_random: bool = True,
    ) -> None:
        rho_value = float(rho)
        if not 0.0 <= rho_value <= 1.0:
            raise ValueError("rho must be in [0, 1].")
        mode = str(directed_mode).lower()
        if mode not in {"none", "edge_random", "access_weighted"}:
            raise ValueError("directed_mode must be 'none', 'edge_random', or 'access_weighted'.")
        if mode == "none" and rho_value != 0.0:
            raise ValueError("directed_mode='none' is only valid when rho=0.")

        initial_grid = np.asarray(grid)
        if initial_grid.ndim != 2 or initial_grid.size == 0:
            raise ValueError("grid must be a non-empty 2D array.")
        self.rng = rng
        self.rho = rho_value
        self.directed_mode = mode
        self.seed = None if seed is None else int(seed)
        self.forced_initial_random = bool(forced_initial_random)
        self.k = int(k) if mode == "access_weighted" else None
        self.lambda_decay = float(lambda_decay) if mode == "access_weighted" else None
        self.eta = float(eta) if mode == "access_weighted" else None
        self.state = DeforestationEventState(
            initial_grid,
            mode="random",
            boundary_mode=str(generation_boundary),
            k=int(k),
            lambda_decay=float(lambda_decay),
            eta=float(eta),
        )
        self.initial_forest_cells = int(len(self.state.forest_indices))
        self.total_cells = int(self.state.grid.size)
        self.n_events = 0
        self.n_random_forced_initial = 0
        self.n_random_stochastic = 0
        self.n_directed = 0

    @property
    def grid(self) -> np.ndarray:
        return self.state.grid

    @property
    def generation_boundary(self) -> str:
        return str(self.state.boundary_mode)

    @property
    def forest_cells(self) -> int:
        return int(len(self.state.forest_indices))

    def metadata(self) -> dict[str, object]:
        """Canonical metadata for the controller's current trajectory state."""
        return {
            "generator_version": SEQUENTIAL_DEFORESTATION_GENERATOR_VERSION,
            "L": int(self.grid.shape[0]) if self.grid.shape[0] == self.grid.shape[1] else None,
            "rows": int(self.grid.shape[0]),
            "cols": int(self.grid.shape[1]),
            "rho": float(self.rho),
            "directed_mode": self.directed_mode,
            "k": self.k,
            "lambda": self.lambda_decay,
            "eta": self.eta,
            "generation_boundary": self.generation_boundary,
            "seed": self.seed,
            "forced_initial_random": self.forced_initial_random,
            "event_count": int(self.n_events),
            "n_random_forced_initial": int(self.n_random_forced_initial),
            "n_random_stochastic": int(self.n_random_stochastic),
            "n_directed": int(self.n_directed),
            "forest_cells": self.forest_cells,
            "H": float(self.forest_cells) / float(self.total_cells),
        }

    def step(self) -> SequentialDeforestationEvent:
        event_index = int(self.n_events)
        if event_index == 0 and self.forced_initial_random:
            selected_mode = "random"
        else:
            selected_mode = (
                self.directed_mode
                if float(self.rng.random()) < self.rho
                else "random"
            )

        changed_flat = self.state.deforest_one(self.rng, mode=selected_mode)
        if changed_flat is None:
            raise SequentialDeforestationBlockedError(
                {
                    "event_index_zero_based": event_index,
                    "event_number": event_index + 1,
                    "selected_mode": selected_mode,
                    "rho": self.rho,
                    "directed_mode": self.directed_mode,
                    "forest_cells": self.forest_cells,
                    "edge_candidates": int(len(self.state.edge_candidate_indices)),
                    "generation_boundary": self.generation_boundary,
                    "k": self.k,
                    "lambda_decay": self.lambda_decay,
                    "eta": self.eta,
                }
            )

        self.n_events += 1
        if event_index == 0 and self.forced_initial_random:
            self.n_random_forced_initial += 1
        elif selected_mode == "random":
            self.n_random_stochastic += 1
        else:
            self.n_directed += 1
        return SequentialDeforestationEvent(
            event_number=int(self.n_events),
            changed_flat_index=int(changed_flat),
            mode=selected_mode,
            forest_cells=self.forest_cells,
            habitat_fraction=float(self.forest_cells) / float(self.total_cells),
        )


def _forest_count_for_fraction(habitat_fraction: float, total_cells: int) -> int:
    """Map H to the nearest attainable integer count, avoiding float triggers."""
    value = float(habitat_fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("habitat fractions must be in [0, 1].")
    return int(np.floor(value * float(total_cells) + 0.5))


def run_sequential_deforestation(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    rho: float,
    directed_mode: str,
    target_habitat_fraction: float,
    generation_boundary: str = "periodic",
    k: int = 3,
    lambda_decay: float = 0.5,
    eta: float = 1.0,
    save_habitat_fractions: Optional[list[float] | tuple[float, ...]] = None,
    save_every_n_events: Optional[int] = None,
    include_initial_snapshot: bool = False,
    seed: Optional[int] = None,
) -> SequentialDeforestationResult:
    """Run one rho-controlled trajectory with true single-cell events.

    The first removal is forced to ``random``. Every later removal consumes
    exactly one ``rng.random()`` draw to choose ``directed_mode`` with
    probability ``rho`` and ``random`` otherwise. A selected mode that has no
    candidate raises :class:`SequentialDeforestationBlockedError`; mechanisms
    are never silently substituted.

    Requested habitat fractions are converted once to integer forest-cell
    counts. A snapshot is copied only when its exact count is reached, ensuring
    that all snapshots belong to this one stateful trajectory.
    """
    rho_value = float(rho)
    if not 0.0 <= rho_value <= 1.0:
        raise ValueError("rho must be in [0, 1].")
    mode = str(directed_mode).lower()
    if mode not in {"none", "edge_random", "access_weighted"}:
        raise ValueError("directed_mode must be 'none', 'edge_random', or 'access_weighted'.")
    if mode == "none" and rho_value != 0.0:
        raise ValueError("directed_mode='none' is only valid when rho=0.")
    if save_every_n_events is not None and int(save_every_n_events) < 1:
        raise ValueError("save_every_n_events must be >= 1 when provided.")

    initial_grid = np.asarray(grid)
    if initial_grid.ndim != 2 or initial_grid.size == 0:
        raise ValueError("grid must be a non-empty 2D array.")
    total_cells = int(initial_grid.size)
    initial_forest_cells = int(np.count_nonzero(initial_grid == FOREST))
    if initial_forest_cells != total_cells:
        raise ValueError("sequential rho trajectories must start from a fully forested grid.")

    target_fraction = float(target_habitat_fraction)
    target_forest_cells = _forest_count_for_fraction(target_fraction, total_cells)
    requested = tuple(float(value) for value in (save_habitat_fractions or ()))
    requested_by_count: dict[int, list[float]] = {}
    for requested_fraction in requested:
        requested_count = _forest_count_for_fraction(requested_fraction, total_cells)
        if requested_count < target_forest_cells or requested_count > initial_forest_cells:
            raise ValueError(
                "save_habitat_fractions must lie on the trajectory between the initial and target counts."
            )
        requested_by_count.setdefault(requested_count, []).append(requested_fraction)

    controller = SequentialDeforestationController(
        initial_grid,
        rng,
        rho=rho_value,
        directed_mode=mode,
        generation_boundary=generation_boundary,
        k=int(k),
        lambda_decay=float(lambda_decay),
        eta=float(eta),
        seed=seed,
    )
    state = controller.state
    snapshots: list[SequentialDeforestationSnapshot] = []
    changed_indices: list[int] = []
    event_modes: list[str] = []

    def maybe_store_snapshot(forest_cells: int, event_count: int) -> None:
        fractions = tuple(requested_by_count.pop(int(forest_cells), ()))
        reasons: list[str] = []
        if fractions:
            reasons.append("requested_habitat_fraction")
        if save_every_n_events is not None and event_count > 0:
            if event_count % int(save_every_n_events) == 0:
                reasons.append("event_stride")
        if event_count == 0 and include_initial_snapshot:
            reasons.append("initial")
        if not reasons:
            return
        snapshots.append(
            SequentialDeforestationSnapshot(
                requested_habitat_fractions=fractions,
                target_forest_cells=int(forest_cells),
                habitat_fraction=float(forest_cells) / float(total_cells),
                forest_cells=int(forest_cells),
                event_count=int(event_count),
                reasons=tuple(reasons),
                grid=state.grid.copy(),
            )
        )

    maybe_store_snapshot(initial_forest_cells, 0)
    removals_needed = initial_forest_cells - target_forest_cells
    for _ in range(removals_needed):
        event = controller.step()
        changed_indices.append(int(event.changed_flat_index))
        event_modes.append(event.mode)
        maybe_store_snapshot(event.forest_cells, event.event_number)

    n_events = len(changed_indices)
    n_forced = int(controller.n_random_forced_initial)
    n_random_stochastic = int(controller.n_random_stochastic)
    n_directed = int(controller.n_directed)
    n_random_total = n_forced + n_random_stochastic
    post_initial_events = max(0, n_events - n_forced)
    final_forest_cells = int(len(state.forest_indices))
    if requested_by_count:
        raise RuntimeError(f"Internal error: snapshot counts were not reached: {requested_by_count!r}")

    return SequentialDeforestationResult(
        grid=state.grid,
        grid_shape=(int(state.grid.shape[0]), int(state.grid.shape[1])),
        L=int(state.grid.shape[0]) if state.grid.shape[0] == state.grid.shape[1] else None,
        snapshots=tuple(snapshots),
        changed_flat_indices=np.asarray(changed_indices, dtype=np.intp),
        event_modes=tuple(event_modes),
        initial_forest_cells=initial_forest_cells,
        target_forest_cells=target_forest_cells,
        final_forest_cells=final_forest_cells,
        total_cells=total_cells,
        target_habitat_fraction=target_fraction,
        final_habitat_fraction=float(final_forest_cells) / float(total_cells),
        n_events=n_events,
        n_random_total=n_random_total,
        n_random_forced_initial=n_forced,
        n_random_stochastic=n_random_stochastic,
        n_directed=n_directed,
        generator_version=SEQUENTIAL_DEFORESTATION_GENERATOR_VERSION,
        forced_initial_random=True,
        generation_boundary=str(state.boundary_mode),
        rho=rho_value,
        realized_random_fraction=(float(n_random_total) / n_events if n_events else 0.0),
        realized_directed_fraction=(float(n_directed) / n_events if n_events else 0.0),
        realized_random_fraction_post_initial=(
            float(n_random_stochastic) / post_initial_events if post_initial_events else 0.0
        ),
        realized_directed_fraction_post_initial=(
            float(n_directed) / post_initial_events if post_initial_events else 0.0
        ),
        directed_mode=mode,
        k=int(k) if mode == "access_weighted" else None,
        lambda_decay=float(lambda_decay) if mode == "access_weighted" else None,
        eta=float(eta) if mode == "access_weighted" else None,
        expected_events_per_stochastic_random=(
            float(1.0 / (1.0 - rho_value)) if rho_value < 1.0 else float("inf")
        ),
        seed=None if seed is None else int(seed),
        rng_bit_generator=type(rng.bit_generator).__name__,
    )


def deforest_to_forest_cover(
    landscape: Landscape,
    *,
    target_cover: float,
    mode: str,
    r: float,
    rng: np.random.Generator,
    selection_mode: str = "exact",
    max_steps: int = 50_000,
    inplace: bool = False,
    return_history: bool = False,
    adjust_last_step: bool = True,
    **kwargs: Any,
) -> Landscape | tuple[Landscape, GenerationHistory]:
    """
    Legacy/utility batch-rate generator; not used for final thesis landscapes.

    Repeatedly apply deforestation until forest_cover(grid) <= target_cover.

    Notes
    -----
    - Uses Landscape.apply_change(kind="deforest", ...)
    - If adjust_last_step=True: tries to set the final step's r so the cover lands close to target.
      This is exact in terms of cell counts only for candidate sets that have enough allowed cells.
    - Stops early if no cells can be removed in a step (stuck).
    """
    if not (0.0 <= float(target_cover) <= 1.0):
        raise ValueError("target_cover must be in [0, 1].")
    if r <= 0.0:
        raise ValueError("r must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")

    L = landscape if inplace else landscape.copy()

    hist = GenerationHistory(forest_cover=[], percolates=[], removed_cells=[])
    hist.forest_cover.append(forest_cover(L.grid))
    hist.percolates.append(False)  # placeholder (not used here)

    total_cells = int(L.grid.size)
    target_forest_cells = int(np.round(float(target_cover) * float(total_cells)))

    for _step in range(max_steps):
        current_forest_cells = int((L.grid == FOREST).sum())
        if current_forest_cells <= target_forest_cells:
            break

        step_r = float(r)
        if adjust_last_step:
            # Try to remove exactly what is needed in terms of FOREST cell count.
            need_remove = current_forest_cells - target_forest_cells
            if need_remove > 0:
                step_r = min(step_r, float(need_remove) / float(max(1, current_forest_cells)))

        mask = L.apply_change(
            kind="deforest",
            mode=mode,
            r=step_r,
            rng=rng,
            selection_mode=selection_mode,
            inplace=True,
            **kwargs,
        )

        removed = int(mask.sum())
        hist.removed_cells.append(removed)
        hist.forest_cover.append(forest_cover(L.grid))
        hist.percolates.append(False)

        if removed == 0:
            # No more eligible candidates -> can't reach target with this mode/settings.
            break

    return (L, hist) if return_history else L


def deforest_until_percolation_break(
    landscape: Landscape,
    *,
    direction: Direction,
    mode: str,
    r: float,
    rng: np.random.Generator,
    selection_mode: str = "exact",
    max_steps: int = 50_000,
    inplace: bool = False,
    return_history: bool = False,
    return_last_percolating: bool = True,
    **kwargs: Any,
) -> Landscape | tuple[Landscape, GenerationHistory]:
    """
    Legacy/utility batch-rate generator; not used for final thesis landscapes.

    Repeatedly apply deforestation until the forest no longer percolates.

    Parameters
    ----------
    return_last_percolating:
        - True  -> return the last landscape that still percolates (useful for threshold bracketing)
        - False -> return the first landscape that DOES NOT percolate

    Notes
    -----
    - Percolation is checked on the full grid (forest clusters), not core habitat.
    - Stops early if no cells can be removed in a step (stuck).
    """
    if r <= 0.0:
        raise ValueError("r must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")

    L = landscape if inplace else landscape.copy()

    hist = GenerationHistory(forest_cover=[], percolates=[], removed_cells=[])
    hist.forest_cover.append(forest_cover(L.grid))
    hist.percolates.append(percolates(L.grid, direction=direction))

    last_perc_grid = L.grid.copy()
    last_perc_cover = hist.forest_cover[-1]

    for _step in range(max_steps):
        is_perc = percolates(L.grid, direction=direction)
        if not is_perc:
            break

        # remember last percolating state
        last_perc_grid = L.grid.copy()
        last_perc_cover = forest_cover(L.grid)

        mask = L.apply_change(
            kind="deforest",
            mode=mode,
            r=float(r),
            rng=rng,
            selection_mode=selection_mode,
            inplace=True,
            **kwargs,
        )

        removed = int(mask.sum())
        hist.removed_cells.append(removed)
        hist.forest_cover.append(forest_cover(L.grid))
        hist.percolates.append(percolates(L.grid, direction=direction))

        if removed == 0:
            break

    if return_last_percolating:
        L_out = landscape.copy() if not inplace else landscape
        L_out.grid = last_perc_grid
        if return_history:
            # history corresponds to the full run; that's usually what you want for analysis
            return L_out, hist
        return L_out

    return (L, hist) if return_history else L


if __name__ == "__main__":
    # Small smoke tests
    import numpy as _np

    def _assert(cond: bool, msg: str) -> None:
        assert cond, msg

    rng = _np.random.default_rng(0)

    # Start from full forest
    L0 = Landscape(_np.ones((30, 30), dtype=int))

    # 1) Deforest to target cover
    L1, h1 = deforest_to_forest_cover(
        L0,
        target_cover=0.6,
        mode="random",
        r=0.02,
        rng=rng,
        selection_mode="exact",
        return_history=True,
    )
    _assert(forest_cover(L1.grid) <= 0.6 + 1e-12, "target cover not reached (or overshoot not handled)")
    _assert(all(x >= y for x, y in zip(h1.forest_cover[:-1], h1.forest_cover[1:])), "cover should be non-increasing")
    print("[OK] deforest_to_forest_cover")

    # 2) Percolation break (bracket)
    rng2 = _np.random.default_rng(1)
    L2, h2 = deforest_until_percolation_break(
        Landscape(_np.ones((40, 40), dtype=int)),
        direction="top_bottom",
        mode="random",
        r=0.01,
        rng=rng2,
        selection_mode="exact",
        return_history=True,
        return_last_percolating=True,
    )
    _assert(percolates(L2.grid, direction="top_bottom"), "should return last percolating")
    print("[OK] deforest_until_percolation_break (return_last_percolating=True)")

    print("\nAll generation tests passed.")
