# src/models/hanski.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, Tuple

import numpy as np
from scipy import sparse


InitialOccupancy = Union[
    Literal["all", "none"],
    float,     # fraction in [0, 1]
    np.ndarray # bool vector (n_clusters,)
]
InitialExpectation = Union[
    Literal["all", "none"],
    float,     # uniform p_i in [0, 1]
    np.ndarray # float/bool vector (n_clusters,)
]


ColonizationMode = Literal["exp", "linear", "sigmoid"]


def _validated_self_colonization(value: float) -> float:
    """Return a valid Schnell-style self-colonization multiplier.

    ``0`` keeps the conventional zero diagonal; ``1`` reproduces the
    area-weighted diagonal used by Schnell et al. (2013).
    """
    multiplier = float(value)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("self_colonization must be finite and >= 0.")
    return multiplier


def rates_from_ratio_speed(r: float, s: float) -> tuple[float, float]:
    """Convert ``r = e0 / c`` and ``s = c`` to the model parameters.

    This is an exact algebraic reparameterization: ``c = s`` and
    ``e0 = r * s``. In the discrete model, ``s`` is only an approximate
    common speed parameter because colonization uses ``1 - exp(-c * S)``
    while baseline extinction is linear in ``e0``.
    """
    ratio = float(r)
    speed = float(s)
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("r must be finite and >= 0.")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("s must be finite and > 0 so r = e0 / c is defined.")
    extinction_scale = ratio * speed
    if not np.isfinite(extinction_scale):
        raise ValueError("r * s must be finite.")
    return speed, extinction_scale


def resolve_rate_parameters(
    *,
    c: float | None = None,
    e0: float | None = None,
    r: float | None = None,
    s: float | None = None,
) -> tuple[float, float]:
    """Resolve exactly one complete rate parameterization to ``(c, e0)``."""
    direct_supplied = c is not None or e0 is not None
    ratio_speed_supplied = r is not None or s is not None
    if direct_supplied and ratio_speed_supplied:
        raise ValueError("Use either r/s or c/e0, not both.")
    if direct_supplied:
        if c is None or e0 is None:
            raise ValueError("Direct parameterization requires both c and e0.")
        return float(c), float(e0)
    if ratio_speed_supplied:
        if r is None or s is None:
            raise ValueError("Ratio/speed parameterization requires both r and s.")
        return rates_from_ratio_speed(r, s)
    raise ValueError("Provide either r/s or c/e0.")


def ratio_speed_from_rates(c: float, e0: float) -> tuple[float, float]:
    """Convert the direct parameters to ``r = e0 / c`` and ``s = c``."""
    colonization_scale = float(c)
    extinction_scale = float(e0)
    if not np.isfinite(colonization_scale) or colonization_scale <= 0.0:
        raise ValueError("c must be finite and > 0 to define r = e0 / c.")
    if not np.isfinite(extinction_scale) or extinction_scale < 0.0:
        raise ValueError("e0 must be finite and >= 0.")
    return extinction_scale / colonization_scale, colonization_scale


# ============================================================
# Config / Result
# ============================================================

@dataclass(frozen=True)
class HanskiConfig:
    # Colonization
    c: float = 1.0
    alpha: float = 0.15
    b: float = 1.0
    # Schnell et al. (2013): 0 = no diagonal, 1 = full area-weighted diagonal
    self_colonization: float = 0.0

    # Colonization functional form
    colonization_mode: ColonizationMode = "exp"
    hill_h: float = 2.0  # used only for colonization_mode="sigmoid"

    # Extinction
    e0: float = 0.2
    x: float = 0.5

    # Optional rescue effect (reduces extinction with increasing connectivity)
    gamma: float = 0.0

    # Protected area effect on extinction (optional)
    prot_ext_mult: float = 1.0  # <1 reduces extinction in protected parts

    clip_probs: bool = True

    def __post_init__(self) -> None:
        _validated_self_colonization(self.self_colonization)

    @classmethod
    def from_ratio_speed(
        cls,
        *,
        r: float = 0.2,
        s: float = 1.0,
        **kwargs: Any,
    ) -> "HanskiConfig":
        """Build a config using the ratio/speed parameterization."""
        if "c" in kwargs or "e0" in kwargs:
            raise ValueError("Do not combine r/s with direct c/e0 arguments.")
        c, e0 = rates_from_ratio_speed(r, s)
        return cls(c=c, e0=e0, **kwargs)

    @property
    def r(self) -> float:
        """Extinction-to-colonization ratio ``e0 / c``."""
        if float(self.c) == 0.0:
            return float("nan") if float(self.e0) == 0.0 else float("inf")
        return float(self.e0) / float(self.c)

    @property
    def s(self) -> float:
        """Speed coordinate of the ratio/speed parameterization (``s = c``)."""
        return float(self.c)


@dataclass
class HanskiResult:
    config: HanskiConfig
    edge_depth: float
    connectivity: str
    boundary_mode: str
    habitat_threshold: float
    distance_mode: str
    softmin_tau: float
    voronoi_depth: int | None
    states: np.ndarray              # (T+1, n_clusters) bool
    occupancy_fraction: np.ndarray  # (T+1,) float


@dataclass
class HanskiExpectationResult:
    config: HanskiConfig
    edge_depth: float
    connectivity: str
    boundary_mode: str
    habitat_threshold: float
    distance_mode: str
    softmin_tau: float
    voronoi_depth: int | None
    probabilities: np.ndarray       # (T+1, n_clusters) float
    occupancy_fraction: np.ndarray  # (T+1,) float


@dataclass
class MetapopEquilibriumResult:
    probabilities: np.ndarray
    occupancy_fraction: float
    n_iter: int
    converged: bool
    max_delta: float


# ============================================================
# Small helpers
# ============================================================

def _normalize_connectivity(connectivity: str | int) -> str:
    """
    Normalize connectivity to "4" or "8".
    Accepts legacy:
      - 1 -> "4"
      - 2 -> "8"
      - 4/8 -> "4"/"8"
    """
    if connectivity in ("4", "8"):
        return str(connectivity)
    if connectivity in (4, 8):
        return str(int(connectivity))
    if connectivity == 1:
        return "4"
    if connectivity == 2:
        return "8"
    raise ValueError(f"connectivity must be '4' or '8' (or legacy 1/2), got {connectivity!r}")


def _get_attr(obj: Any, *names: str) -> Any:
    """Return the first existing attribute among names, else raise AttributeError."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"Object is missing required attributes: {names}")


# ============================================================
# Hanski / IFM Model
# ============================================================

class HanskiModel:
    """
    Hanski / IFM on cluster attributes stored in Landscape.

    Requirements on the landscape object (after ensure_*):
      - n_clusters (or legacy n_patches)
      - cluster_areas (or legacy patch_areas)
      - cluster_dist_matrix (or legacy patch_dist_matrix)
      - core_cluster_labels (or legacy labels_core) for state_to_raster
      - cluster_protected_fraction (or legacy patch_protected_fraction), optional
    """

    def __init__(
        self,
        landscape: Any,
        edge_depth: float = 0.0,
        connectivity: str | int = "4",
        boundary_mode: str = "periodic",
        habitat_threshold: float = 0.5,
        distance_mode: str = "voronoi_dt",
        softmin_tau: float = 1.0,
        voronoi_depth: int | str | None = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.L = landscape
        self.edge_depth = float(edge_depth)
        self.connectivity = _normalize_connectivity(connectivity)
        self.boundary_mode = str(boundary_mode)
        self.habitat_threshold = float(habitat_threshold)
        self.distance_mode = str(distance_mode)
        self.softmin_tau = float(softmin_tau)
        from src.landscape.landscape import normalize_voronoi_depth

        self.voronoi_depth = normalize_voronoi_depth(voronoi_depth)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ----------------------------
    # Landscape preparation / accessors
    # ----------------------------

    def _ensure(self, rebuild: bool = False) -> None:
        """
        Ensure the landscape provides cluster attributes. Supports both:
          - Landscape.ensure_clusters(...) (preferred)
          - legacy Landscape.ensure_patches(...)
        """
        if hasattr(self.L, "ensure_clusters"):
            self.L.ensure_clusters(
                edge_depth=self.edge_depth,
                connectivity=self.connectivity,
                boundary_mode=self.boundary_mode,
                habitat_threshold=self.habitat_threshold,
                distance_mode=self.distance_mode,
                softmin_tau=self.softmin_tau,
                voronoi_depth=self.voronoi_depth,
                rebuild=rebuild,
            )
        elif hasattr(self.L, "ensure_patches"):
            # legacy: many implementations expect 1/2 instead of 4/8
            legacy_conn = 1 if self.connectivity == "4" else 2
            self.L.ensure_patches(
                edge_depth=self.edge_depth,
                connectivity=legacy_conn,
                habitat_threshold=self.habitat_threshold,
                rebuild=rebuild,
            )
        else:
            raise AttributeError("Landscape object must implement ensure_clusters(...) or ensure_patches(...).")

        # Validate required attributes (cluster_* OR patch_* legacy)
        _ = self.n_clusters
        _ = self.cluster_areas
        _ = self.cluster_dist_matrix
        # Protected fraction may be None; only validate shape later if present

    @property
    def n_clusters(self) -> int:
        return int(_get_attr(self.L, "n_clusters", "n_patches"))

    @property
    def cluster_areas(self) -> np.ndarray:
        arr = np.asarray(_get_attr(self.L, "cluster_areas", "patch_areas"), dtype=float)
        if arr.shape != (self.n_clusters,):
            raise ValueError("cluster_areas has wrong shape.")
        return arr

    @property
    def cluster_dist_matrix(self) -> np.ndarray:
        mat = np.asarray(_get_attr(self.L, "cluster_dist_matrix", "patch_dist_matrix"), dtype=float)
        n = self.n_clusters
        if mat.shape != (n, n):
            raise ValueError("cluster_dist_matrix has wrong shape.")
        return mat

    @property
    def cluster_protected_fraction(self) -> Optional[np.ndarray]:
        if hasattr(self.L, "cluster_protected_fraction"):
            val = getattr(self.L, "cluster_protected_fraction")
        elif hasattr(self.L, "patch_protected_fraction"):
            val = getattr(self.L, "patch_protected_fraction")
        else:
            val = None

        if val is None:
            return None

        frac = np.asarray(val, dtype=float)
        if frac.shape != (self.n_clusters,):
            raise ValueError("cluster_protected_fraction has wrong shape.")
        return frac

    # ----------------------------
    # Initialization / kernel
    # ----------------------------

    def _init_state(self, initial: InitialOccupancy, n: int) -> np.ndarray:
        """Initialize occupancy state vector (bool, shape (n,))."""
        if isinstance(initial, str):
            if initial == "all":
                return np.ones(n, dtype=bool)
            if initial == "none":
                return np.zeros(n, dtype=bool)
            raise ValueError("initial must be 'all', 'none', a fraction float, or a bool array.")

        if isinstance(initial, (float, int)):
            frac = float(initial)
            if not (0.0 <= frac <= 1.0):
                raise ValueError("initial fraction must be in [0, 1].")
            return self.rng.random(n) < frac

        arr = np.asarray(initial)
        if arr.dtype != bool or arr.shape != (n,):
            raise ValueError("initial array must be boolean with shape (n_clusters,).")
        return arr.copy()

    def _init_probabilities(self, initial: InitialExpectation, n: int) -> np.ndarray:
        """Initialize occupancy probabilities p_i in [0, 1], shape (n,)."""
        if isinstance(initial, str):
            if initial == "all":
                return np.ones(n, dtype=float)
            if initial == "none":
                return np.zeros(n, dtype=float)
            raise ValueError("initial must be 'all', 'none', a fraction float, or a probability array.")

        if isinstance(initial, (float, int)):
            frac = float(initial)
            if not (0.0 <= frac <= 1.0):
                raise ValueError("initial fraction must be in [0, 1].")
            return np.full(n, frac, dtype=float)

        arr = np.asarray(initial, dtype=float)
        if arr.shape != (n,):
            raise ValueError("initial array must have shape (n_clusters,).")
        if np.any(~np.isfinite(arr)):
            raise ValueError("initial probability array must be finite.")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("initial probability array must lie in [0, 1].")
        return arr.copy()

    def _kernel_matrix(self, alpha: float) -> np.ndarray:
        """K_ij = exp(-alpha * d_ij), with K_ii = 0."""
        dist = self.cluster_dist_matrix
        K = np.exp(-float(alpha) * dist)
        np.fill_diagonal(K, 0.0)
        return K

    def _connectivity(self, occupancy: np.ndarray, cfg: HanskiConfig, K: np.ndarray) -> np.ndarray:
        """Expected connectivity, optionally including the Schnell diagonal."""
        areas = self.cluster_areas
        sources = (areas ** float(cfg.b)) * np.asarray(occupancy, dtype=float)
        self_term = _validated_self_colonization(cfg.self_colonization) * sources
        return K @ sources + self_term

    # ----------------------------
    # Colonization / extinction probabilities
    # ----------------------------

    def _colonization_prob(self, S: np.ndarray, cfg: HanskiConfig) -> np.ndarray:
        """
        Compute colonization probability C_i as a function of connectivity S_i.

        Modes
        -----
        - "exp":    C = 1 - exp(-c*S)         (classic IFM)
        - "linear": C = c*S                  (linearization for small S; clip may apply)
        - "sigmoid": Hill function:
                   z = (c*S)^h
                   C = z / (1 + z)          (S-shaped for h>1, and C(0)=0)
        """
        c = float(cfg.c)
        mode = cfg.colonization_mode

        if mode == "exp":
            return 1.0 - np.exp(-c * S)

        if mode == "linear":
            return c * S

        if mode == "sigmoid":
            h = float(cfg.hill_h)
            if h <= 0:
                raise ValueError("hill_h must be > 0 for colonization_mode='sigmoid'.")
            z = (c * S) ** h
            return z / (1.0 + z)

        raise ValueError(f"Unknown colonization_mode: {mode!r}")

    def _extinction_prob(self, S: np.ndarray, cfg: HanskiConfig, areas: np.ndarray) -> np.ndarray:
        """
        Compute extinction probability E_i with optional rescue and protection.
        """
        with np.errstate(divide="ignore"):
            E = float(cfg.e0) * (areas ** (-float(cfg.x)))

        # Rescue effect: extinction decreases with increasing connectivity
        if cfg.gamma > 0.0:
            E = E * np.exp(-float(cfg.gamma) * S)

        # Protection effect: extinction reduced on protected fraction
        prot_frac = self.cluster_protected_fraction
        if (prot_frac is not None) and (cfg.prot_ext_mult != 1.0):
            f = prot_frac
            mult = (1.0 - f) + f * float(cfg.prot_ext_mult)
            E = E * mult

        return E

    # ----------------------------
    # Dynamics step
    # ----------------------------

    def step(self, state: np.ndarray, cfg: HanskiConfig, K: np.ndarray) -> np.ndarray:
        """
        One stochastic update step.

        Schnell-style self-colonization is an incidence-function/mean-field
        term. A binary occupied/empty state cannot retain the remnant
        individuals it represents, so use ``run_expectation`` or the
        continuous equilibrium solver when that term is enabled.
        """
        if _validated_self_colonization(cfg.self_colonization) > 0.0:
            raise ValueError(
                "Schnell-style self_colonization is not defined for the binary stochastic update; "
                "use run_expectation or run_continuous_equilibrium_from_kernel."
            )
        S = self._connectivity(state.astype(float), cfg, K)
        areas = self.cluster_areas

        C = self._colonization_prob(S, cfg)
        E = self._extinction_prob(S, cfg, areas)

        if cfg.clip_probs:
            C = np.clip(C, 0.0, 1.0)
            E = np.clip(E, 0.0, 1.0)

        extinct = self.rng.random(self.n_clusters) < E
        colonize = self.rng.random(self.n_clusters) < C

        new_state = state.copy()
        new_state[state] = ~extinct[state]
        new_state[~state] = colonize[~state]
        return new_state

    def step_expectation(self, p: np.ndarray, cfg: HanskiConfig, K: np.ndarray) -> np.ndarray:
        """
        One deterministic expectation update step.

        Update:
          p_next = p * (1 - E) + (1 - p) * C
        """
        probs = np.asarray(p, dtype=float)
        if probs.shape != (self.n_clusters,):
            raise ValueError("p must have shape (n_clusters,).")
        S = self._connectivity(probs, cfg, K)
        areas = self.cluster_areas

        C = self._colonization_prob(S, cfg)
        E = self._extinction_prob(S, cfg, areas)

        if cfg.clip_probs:
            C = np.clip(C, 0.0, 1.0)
            E = np.clip(E, 0.0, 1.0)

        p_next = probs * (1.0 - E) + (1.0 - probs) * C
        if cfg.clip_probs:
            p_next = np.clip(p_next, 0.0, 1.0)
        return p_next

    # ----------------------------
    # Simulation
    # ----------------------------

    def run(
        self,
        steps: int,
        config: Optional[HanskiConfig] = None,
        initial: InitialOccupancy = 0.5,
        rebuild_clusters: bool = False,
    ) -> HanskiResult:
        """
        Run the model for a number of steps.

        Returns states with shape (steps+1, n_clusters).
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")

        self._ensure(rebuild=rebuild_clusters)
        cfg = config if config is not None else HanskiConfig()
        K = self._kernel_matrix(cfg.alpha)

        x = self._init_state(initial, self.n_clusters)
        states = np.zeros((steps + 1, self.n_clusters), dtype=bool)
        states[0] = x

        for t in range(steps):
            x = self.step(x, cfg, K)
            states[t + 1] = x

        occ = states.mean(axis=1).astype(float)
        return HanskiResult(
            config=cfg,
            edge_depth=self.edge_depth,
            connectivity=self.connectivity,
            boundary_mode=self.boundary_mode,
            habitat_threshold=self.habitat_threshold,
            distance_mode=self.distance_mode,
            softmin_tau=self.softmin_tau,
            voronoi_depth=self.voronoi_depth,
            states=states,
            occupancy_fraction=occ,
        )

    def run_expectation(
        self,
        steps: int,
        config: Optional[HanskiConfig] = None,
        initial: InitialExpectation = 0.5,
        rebuild_clusters: bool = False,
    ) -> HanskiExpectationResult:
        """
        Run deterministic expectation dynamics for a number of steps.

        Returns probabilities with shape (steps+1, n_clusters).
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")

        self._ensure(rebuild=rebuild_clusters)
        cfg = config if config is not None else HanskiConfig()
        K = self._kernel_matrix(cfg.alpha)

        p = self._init_probabilities(initial, self.n_clusters)
        probabilities = np.zeros((steps + 1, self.n_clusters), dtype=float)
        probabilities[0] = p

        for t in range(steps):
            p = self.step_expectation(p, cfg, K)
            probabilities[t + 1] = p

        occ = probabilities.mean(axis=1).astype(float)
        return HanskiExpectationResult(
            config=cfg,
            edge_depth=self.edge_depth,
            connectivity=self.connectivity,
            boundary_mode=self.boundary_mode,
            habitat_threshold=self.habitat_threshold,
            distance_mode=self.distance_mode,
            softmin_tau=self.softmin_tau,
            voronoi_depth=self.voronoi_depth,
            probabilities=probabilities,
            occupancy_fraction=occ,
        )

    # ----------------------------
    # Visualization helper
    # ----------------------------

    def state_to_raster(self, state: np.ndarray, fill_value: int = 0) -> np.ndarray:
        """
        Convert cluster occupancy state to a raster.

        Values:
          0 = no core
          1 = core cluster exists but unoccupied
          2 = core cluster occupied
        """
        if state.shape != (self.n_clusters,):
            raise ValueError("state must have shape (n_clusters,)")

        labels = _get_attr(self.L, "core_cluster_labels", "labels_core")
        labels = np.asarray(labels)

        out = np.full(labels.shape, fill_value, dtype=np.int8)
        out[labels > 0] = 1

        occ_ids = np.nonzero(state)[0] + 1  # labels are 1..n
        if occ_ids.size > 0:
            out[np.isin(labels, occ_ids)] = 2
        return out


# ============================================================
# Metapopulation capacity (largest eigenvalue)
# ============================================================

@dataclass(frozen=True)
class MetapopCapacityResult:
    lambda_max: float
    eigenvector: np.ndarray  # (n_clusters,) normalized (L2=1)
    n_clusters: int
    converged: bool
    n_iter: int
    tol: float


def kernel_distance_cutoff(alpha: float, kernel_epsilon: float) -> float:
    """
    Distance cutoff implied by exp(-alpha * d) >= kernel_epsilon.

    kernel_epsilon <= 0 disables cutoff-based sparsification. alpha <= 0 means
    there is no distance decay, so no finite cutoff is valid.
    """
    eps = float(kernel_epsilon)
    if eps < 0.0 or eps >= 1.0:
        raise ValueError("kernel_epsilon must be in [0, 1).")
    if eps <= 0.0 or float(alpha) <= 0.0:
        return float("inf")
    return float(-np.log(eps) / float(alpha))


def _kernel_from_distances(
    dist: np.ndarray,
    *,
    alpha: float,
    kernel_epsilon: float = 0.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray | sparse.csr_matrix:
    target_dtype = np.dtype(dtype)
    if target_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("kernel dtype must be float32 or float64.")
    d = np.asarray(dist, dtype=target_dtype)
    cutoff = kernel_distance_cutoff(alpha, kernel_epsilon)

    if not np.isfinite(cutoff):
        K = np.exp(-target_dtype.type(alpha) * d)
        np.fill_diagonal(K, 0.0)
        return K

    offdiag = ~np.eye(d.shape[0], dtype=bool)
    finite_offdiag = d[offdiag & np.isfinite(d)]
    if finite_offdiag.size == 0 or cutoff >= float(finite_offdiag.max()):
        K = np.exp(-float(alpha) * d)
        np.fill_diagonal(K, 0.0)
        return K

    keep = (d <= cutoff) & offdiag
    rows, cols = np.nonzero(keep)
    data = np.exp(-target_dtype.type(alpha) * d[rows, cols]).astype(target_dtype, copy=False)
    return sparse.csr_matrix((data, (rows, cols)), shape=d.shape)


def prepare_dispersal_kernel(
    dist: np.ndarray,
    *,
    alpha: float,
    kernel_epsilon: float = 0.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray | sparse.csr_matrix:
    """
    Build a reusable K_ij = exp(-alpha*d_ij) dispersal kernel.

    Reuse this object for every metric or model run that uses the same
    distance matrix, alpha and kernel_epsilon.
    """
    d = np.asarray(dist)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("cluster_dist_matrix must be square.")
    return _kernel_from_distances(
        d,
        alpha=float(alpha),
        kernel_epsilon=float(kernel_epsilon),
        dtype=dtype,
    )


def _as_1d_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _power_iteration_spectral_radius(
    M: np.ndarray,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray, bool, int]:
    """
    Power iteration for dominant eigenvalue/vector of a nonnegative matrix M.

    Returns (lambda_max, v, converged, n_iter).
    """
    n = int(M.shape[0])
    if n == 0:
        return (np.nan, np.zeros(0, dtype=float), False, 0)

    if rng is None:
        rng = np.random.default_rng(0)

    v = rng.random(n) + 1e-12
    v = v / np.linalg.norm(v)
    shift = float(np.max(_as_1d_array(M @ np.ones(n, dtype=float))))
    if shift == 0.0 or not np.isfinite(shift):
        return (0.0, np.zeros(n, dtype=float), True, 0)

    lam_old = 0.0
    for it in range(1, max_iter + 1):
        mv = _as_1d_array(M @ v)
        w = mv + shift * v
        norm_w = float(np.linalg.norm(w))
        if norm_w == 0.0 or not np.isfinite(norm_w):
            return (0.0, np.zeros(n, dtype=float), True, it)

        v = w / norm_w
        lam_shifted = norm_w

        if abs(lam_shifted - lam_old) <= tol * max(1.0, abs(lam_shifted)):
            lam = float(v @ _as_1d_array(M @ v))  # Rayleigh quotient of unshifted M
            return (lam, v, True, it)

        lam_old = lam_shifted

    lam = float(v @ _as_1d_array(M @ v))
    return (lam, v, False, max_iter)


def _power_iteration_capacity_from_kernel(
    areas: np.ndarray,
    K: np.ndarray | sparse.csr_matrix,
    *,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray, bool, int]:
    a = np.asarray(areas, dtype=float)
    n = int(a.size)
    if n == 0:
        return (np.nan, np.zeros(0, dtype=float), False, 0)

    if rng is None:
        rng = np.random.default_rng(0)

    weight = a ** ((float(x) + float(b)) * 0.5)
    eigenvector_scale = a ** ((float(x) - float(b)) * 0.5)
    self_multiplier = _validated_self_colonization(self_colonization)

    def matvec_unshifted(v_in: np.ndarray) -> np.ndarray:
        between_patch = weight * _as_1d_array(K @ (weight * v_in))
        return between_patch + self_multiplier * (weight ** 2) * v_in

    shift = float(np.max(matvec_unshifted(np.ones(n, dtype=float))))
    if shift == 0.0 or not np.isfinite(shift):
        return (0.0, np.zeros(n, dtype=float), True, 0)

    def matvec(v_in: np.ndarray) -> np.ndarray:
        return matvec_unshifted(v_in) + shift * v_in

    v = rng.random(n) + 1e-12
    v = v / np.linalg.norm(v)

    lam_old = 0.0
    for it in range(1, max_iter + 1):
        w = matvec(v)
        norm_w = float(np.linalg.norm(w))
        if norm_w == 0.0 or not np.isfinite(norm_w):
            return (0.0, np.zeros(n, dtype=float), True, it)

        v = w / norm_w
        lam_shifted = norm_w

        if abs(lam_shifted - lam_old) <= tol * max(1.0, abs(lam_shifted)):
            lam = float(v @ matvec_unshifted(v))
            vm = eigenvector_scale * v
            norm_vm = float(np.linalg.norm(vm))
            if norm_vm > 0.0 and np.isfinite(norm_vm):
                vm = vm / norm_vm
            return (lam, vm, True, it)

        lam_old = lam_shifted

    vm = eigenvector_scale * v
    norm_vm = float(np.linalg.norm(vm))
    if norm_vm > 0.0 and np.isfinite(norm_vm):
        vm = vm / norm_vm
    return (lam_old, vm, False, max_iter)


def mean_full_occupancy_si_from_kernel(
    areas: np.ndarray,
    kernel: np.ndarray | sparse.csr_matrix,
    *,
    b: float,
    self_colonization: float = 0.0,
) -> float:
    """Mean fully occupied S_i using an already prepared dispersal kernel."""
    a = np.asarray(areas, dtype=float)
    if kernel.shape != (a.size, a.size):
        raise ValueError("kernel shape does not match cluster_areas.")
    Ab = a ** float(b)
    self_multiplier = _validated_self_colonization(self_colonization)
    connectivity = _as_1d_array(kernel @ Ab) + self_multiplier * Ab
    return float(np.mean(connectivity)) if a.size else 0.0


def metapopulation_capacity_and_mean_si_from_kernel(
    areas: np.ndarray,
    kernel: np.ndarray | sparse.csr_matrix,
    *,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
    return_matrix: bool = False,
    method: Literal["matrix", "symmetric"] = "matrix",
) -> Tuple[MetapopCapacityResult, float, Optional[np.ndarray]]:
    """Capacity and mean S_i from one reusable, precomputed kernel."""
    a = np.asarray(areas, dtype=float)
    n = int(a.size)
    if kernel.shape != (n, n):
        raise ValueError("kernel shape does not match cluster_areas.")
    if n == 0:
        res = MetapopCapacityResult(
            lambda_max=float("nan"),
            eigenvector=np.zeros(0, dtype=float),
            n_clusters=0,
            converged=False,
            n_iter=0,
            tol=float(tol),
        )
        return res, 0.0, np.zeros((0, 0), dtype=float) if return_matrix else None

    Ab = a ** float(b)
    self_multiplier = _validated_self_colonization(self_colonization)
    mean_si = mean_full_occupancy_si_from_kernel(
        a,
        kernel,
        b=float(b),
        self_colonization=self_multiplier,
    )

    if method == "matrix":
        Ax = a ** float(x)
        if sparse.issparse(kernel):
            M = sparse.diags(Ax) @ kernel @ sparse.diags(Ab)
            M.setdiag(self_multiplier * (a ** (float(x) + float(b))))
            M.eliminate_zeros()
        else:
            M = (kernel * Ax[:, None]) * Ab[None, :]
            np.fill_diagonal(M, self_multiplier * (a ** (float(x) + float(b))))
        lam, v, conv, it = _power_iteration_spectral_radius(
            M,
            max_iter=int(max_iter),
            tol=float(tol),
            rng=rng,
        )
    elif method == "symmetric":
        lam, v, conv, it = _power_iteration_capacity_from_kernel(
            a,
            kernel,
            b=float(b),
            x=float(x),
            self_colonization=self_multiplier,
            max_iter=int(max_iter),
            tol=float(tol),
            rng=rng,
        )
        M = None
    else:
        raise ValueError("method must be 'matrix' or 'symmetric'.")

    res = MetapopCapacityResult(
        lambda_max=float(lam),
        eigenvector=v.astype(float),
        n_clusters=n,
        converged=bool(conv),
        n_iter=int(it),
        tol=float(tol),
    )
    if not return_matrix:
        return res, mean_si, None

    if M is None:
        Ax = a ** float(x)
        if sparse.issparse(kernel):
            M = sparse.diags(Ax) @ kernel @ sparse.diags(Ab)
            M.setdiag(self_multiplier * (a ** (float(x) + float(b))))
            M.eliminate_zeros()
        else:
            M = (kernel * Ax[:, None]) * Ab[None, :]
            np.fill_diagonal(M, self_multiplier * (a ** (float(x) + float(b))))
    return res, mean_si, M


def run_expectation_from_kernel(
    areas: np.ndarray,
    kernel: np.ndarray | sparse.csr_matrix,
    *,
    b: float,
    x: float,
    steps: int,
    c: float | None = None,
    e0: float | None = None,
    r: float | None = None,
    s: float | None = None,
    self_colonization: float = 0.0,
    initial: float | np.ndarray = 1.0,
    min_steps: int = 1,
    tol: float = 0.0,
    dtype: np.dtype | type = np.float64,
) -> tuple[float, int]:
    """Run deterministic occupancy dynamics with either ``r/s`` or ``c/e0``."""
    c_resolved, e0_resolved = resolve_rate_parameters(c=c, e0=e0, r=r, s=s)
    target_dtype = np.dtype(dtype)
    if target_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("expectation dtype must be float32 or float64.")
    a = np.asarray(areas, dtype=target_dtype)
    n = int(a.size)
    if kernel.shape != (n, n):
        raise ValueError("kernel shape does not match cluster_areas.")
    if n == 0:
        return float("nan"), 0

    if np.isscalar(initial):
        initial_value = float(initial)
        if not 0.0 <= initial_value <= 1.0:
            raise ValueError("initial occupancy must lie in [0, 1].")
        p = np.full(n, initial_value, dtype=target_dtype)
    else:
        p = np.asarray(initial, dtype=target_dtype).copy()
        if p.shape != (n,) or np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("initial occupancy array must have shape (n_clusters,) and lie in [0, 1].")

    sources_area = a ** target_dtype.type(b)
    self_multiplier = target_dtype.type(_validated_self_colonization(self_colonization))
    extinction = target_dtype.type(e0_resolved) * (a ** target_dtype.type(-float(x)))
    extinction = np.clip(extinction, 0.0, 1.0)
    effective_tol = max(float(tol), 8.0 * float(np.finfo(target_dtype).eps))
    final_step = 0
    for step in range(1, int(steps) + 1):
        connectivity = np.asarray(
            kernel @ (sources_area * p),
            dtype=target_dtype,
        ).reshape(-1)
        connectivity += self_multiplier * sources_area * p
        colonization = 1.0 - np.exp(-target_dtype.type(c_resolved) * connectivity)
        colonization = np.clip(colonization, 0.0, 1.0)
        next_p = p * (1.0 - extinction) + (1.0 - p) * colonization
        next_p = np.clip(next_p, 0.0, 1.0).astype(target_dtype, copy=False)
        delta = float(np.max(np.abs(next_p - p)))
        p = next_p
        final_step = step
        if step >= int(min_steps) and delta <= effective_tol:
            break
    return float(np.mean(p)), int(final_step)


def _run_equilibrium_fixed_point(
    areas: np.ndarray,
    kernel: np.ndarray | sparse.csr_matrix,
    *,
    r: float,
    b: float,
    x: float,
    self_colonization: float,
    discrete_speed: float | None,
    initial: float | np.ndarray,
    max_iter: int,
    tol: float,
    relaxation: float,
    dtype: np.dtype | type,
) -> MetapopEquilibriumResult:
    """Solve the maximal occupancy equilibrium from an occupied initial state."""
    ratio = float(r)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("r must be finite and > 0 for a unique extinction-colonization equilibrium.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1.")
    if float(tol) < 0.0:
        raise ValueError("tol must be >= 0.")
    weight = float(relaxation)
    if not 0.0 < weight <= 1.0:
        raise ValueError("relaxation must lie in (0, 1].")
    speed = None
    if discrete_speed is not None:
        speed = float(discrete_speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("discrete_speed must be finite and > 0.")

    target_dtype = np.dtype(dtype)
    if target_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("equilibrium dtype must be float32 or float64.")
    a = np.asarray(areas, dtype=target_dtype)
    n = int(a.size)
    if kernel.shape != (n, n):
        raise ValueError("kernel shape does not match cluster_areas.")
    if n == 0:
        return MetapopEquilibriumResult(
            probabilities=np.zeros(0, dtype=target_dtype),
            occupancy_fraction=float("nan"),
            n_iter=0,
            converged=True,
            max_delta=0.0,
        )
    if np.any(~np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("cluster_areas must be finite and > 0.")

    if np.isscalar(initial):
        initial_value = float(initial)
        if not 0.0 <= initial_value <= 1.0:
            raise ValueError("initial occupancy must lie in [0, 1].")
        p = np.full(n, initial_value, dtype=target_dtype)
    else:
        p = np.asarray(initial, dtype=target_dtype).copy()
        if p.shape != (n,) or np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("initial occupancy array must have shape (n_clusters,) and lie in [0, 1].")

    sources_area = a ** target_dtype.type(b)
    self_multiplier = target_dtype.type(_validated_self_colonization(self_colonization))
    extinction_rate = target_dtype.type(ratio) * (a ** target_dtype.type(-float(x)))
    effective_tol = max(float(tol), 8.0 * float(np.finfo(target_dtype).eps))
    converged = False
    max_delta = float("inf")
    final_iter = 0

    for iteration in range(1, int(max_iter) + 1):
        connectivity = np.asarray(
            kernel @ (sources_area * p),
            dtype=target_dtype,
        ).reshape(-1)
        connectivity += self_multiplier * sources_area * p
        if speed is None:
            colonization_term = np.maximum(connectivity, target_dtype.type(0.0))
            extinction_term = extinction_rate
        else:
            colonization_term = 1.0 - np.exp(-target_dtype.type(speed) * connectivity)
            colonization_term = np.clip(colonization_term, 0.0, 1.0)
            extinction_term = np.clip(target_dtype.type(speed) * extinction_rate, 0.0, 1.0)
        denominator = colonization_term + extinction_term
        target = np.divide(
            colonization_term,
            denominator,
            out=np.zeros_like(p),
            where=denominator > 0.0,
        )
        next_p = ((1.0 - weight) * p + weight * target).astype(target_dtype, copy=False)
        next_p = np.clip(next_p, 0.0, 1.0).astype(target_dtype, copy=False)
        max_delta = float(np.max(np.abs(next_p - p)))
        p = next_p
        final_iter = iteration
        if max_delta <= effective_tol:
            converged = True
            break

    return MetapopEquilibriumResult(
        probabilities=p.astype(float, copy=False),
        occupancy_fraction=float(np.mean(p)),
        n_iter=int(final_iter),
        converged=bool(converged),
        max_delta=float(max_delta),
    )


def run_continuous_equilibrium_from_kernel(
    areas: np.ndarray,
    kernel: np.ndarray | sparse.csr_matrix,
    *,
    r: float,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    initial: float | np.ndarray = 1.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    relaxation: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> MetapopEquilibriumResult:
    """Solve the continuous-time IFM equilibrium, which depends on r but not s.

    The optional Schnell term changes connectivity to
    ``S_i = sum_(j!=i) K_ij A_j^b p_j + self_colonization A_i^b p_i``.
    """
    return _run_equilibrium_fixed_point(
        areas,
        kernel,
        r=r,
        b=b,
        x=x,
        self_colonization=self_colonization,
        discrete_speed=None,
        initial=initial,
        max_iter=max_iter,
        tol=tol,
        relaxation=relaxation,
        dtype=dtype,
    )


def run_discrete_equilibrium_from_kernel(
    areas: np.ndarray,
    kernel: np.ndarray | sparse.csr_matrix,
    *,
    r: float,
    s: float,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    initial: float | np.ndarray = 1.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    relaxation: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> MetapopEquilibriumResult:
    """Solve the equilibrium implied by the previous discrete expectation model."""
    return _run_equilibrium_fixed_point(
        areas,
        kernel,
        r=r,
        b=b,
        x=x,
        self_colonization=self_colonization,
        discrete_speed=s,
        initial=initial,
        max_iter=max_iter,
        tol=tol,
        relaxation=relaxation,
        dtype=dtype,
    )


def _power_iteration_capacity_kernel(
    areas: np.ndarray,
    dist: np.ndarray,
    *,
    alpha: float,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
    kernel_epsilon: float = 0.0,
) -> Tuple[float, np.ndarray, bool, int, np.ndarray | sparse.csr_matrix]:
    """
    Power iteration for M = diag(A^x) @ K @ diag(A^b), without materializing M.

    The dense K matrix is still needed, but this avoids a second dense matrix
    allocation and lets callers reuse precomputed cluster areas/distances.

    M is similar to the symmetric matrix
      B = diag(A^((x+b)/2)) @ K @ diag(A^((x+b)/2))
    when K is symmetric. We iterate on B for stability, then transform the
    eigenvector back to the original M-space.
    """
    a = np.asarray(areas, dtype=float)
    d = np.asarray(dist, dtype=float)
    n = int(a.size)
    if n == 0:
        return (np.nan, np.zeros(0, dtype=float), False, 0, np.zeros((0, 0), dtype=float))
    if d.shape != (n, n):
        raise ValueError("cluster_dist_matrix has wrong shape.")

    if rng is None:
        rng = np.random.default_rng(0)

    K = _kernel_from_distances(d, alpha=float(alpha), kernel_epsilon=float(kernel_epsilon))
    lam, v, conv, it = _power_iteration_capacity_from_kernel(
        a,
        K,
        b=float(b),
        x=float(x),
        self_colonization=self_colonization,
        max_iter=int(max_iter),
        tol=float(tol),
        rng=rng,
    )
    return (lam, v, conv, it, K)


def metapopulation_capacity_from_arrays(
    areas: np.ndarray,
    dist: np.ndarray,
    *,
    alpha: float,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
    return_matrix: bool = False,
    method: Literal["matrix", "symmetric"] = "matrix",
    kernel_epsilon: float = 0.0,
) -> Tuple[MetapopCapacityResult, Optional[np.ndarray]]:
    """
    Metapopulation capacity from precomputed cluster areas and distances.

    Use this in sweeps after Landscape.ensure_clusters(...) has already run.
    """
    a = np.asarray(areas, dtype=float)
    d = np.asarray(dist, dtype=float)
    n = int(a.size)
    if a.shape != (n,):
        raise ValueError("cluster_areas has wrong shape.")
    if d.shape != (n, n):
        raise ValueError("cluster_dist_matrix has wrong shape.")

    K = _kernel_from_distances(d, alpha=float(alpha), kernel_epsilon=float(kernel_epsilon))
    self_multiplier = _validated_self_colonization(self_colonization)

    if method == "matrix":
        Ax = a ** float(x)
        Ab = a ** float(b)
        if sparse.issparse(K):
            M = sparse.diags(Ax) @ K @ sparse.diags(Ab)
            M.setdiag(self_multiplier * (a ** (float(x) + float(b))))
            M.eliminate_zeros()
        else:
            M = (K * Ax[:, None]) * Ab[None, :]
            np.fill_diagonal(M, self_multiplier * (a ** (float(x) + float(b))))
        lam, v, conv, it = _power_iteration_spectral_radius(M, max_iter=int(max_iter), tol=float(tol), rng=rng)
    elif method == "symmetric":
        lam, v, conv, it = _power_iteration_capacity_from_kernel(
            a,
            K,
            b=float(b),
            x=float(x),
            self_colonization=self_multiplier,
            max_iter=int(max_iter),
            tol=float(tol),
            rng=rng,
        )
        M = None
    else:
        raise ValueError("method must be 'matrix' or 'symmetric'.")

    res = MetapopCapacityResult(
        lambda_max=float(lam),
        eigenvector=v.astype(float),
        n_clusters=n,
        converged=bool(conv),
        n_iter=int(it),
        tol=float(tol),
    )

    if not return_matrix:
        return res, None

    if M is None:
        Ax = a ** float(x)
        Ab = a ** float(b)
        if sparse.issparse(K):
            M = sparse.diags(Ax) @ K @ sparse.diags(Ab)
            M.setdiag(self_multiplier * (a ** (float(x) + float(b))))
            M.eliminate_zeros()
        else:
            M = (K * Ax[:, None]) * Ab[None, :]
            np.fill_diagonal(M, self_multiplier * (a ** (float(x) + float(b))))
    return res, M


def metapopulation_capacity_and_mean_si_from_arrays(
    areas: np.ndarray,
    dist: np.ndarray,
    *,
    alpha: float,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
    return_matrix: bool = False,
    method: Literal["matrix", "symmetric"] = "matrix",
    kernel_epsilon: float = 0.0,
    kernel_dtype: np.dtype | type = np.float64,
) -> Tuple[MetapopCapacityResult, float, Optional[np.ndarray]]:
    """
    Compute metapopulation capacity and mean full-occupancy S_i together.

    Both metrics use K_ij = exp(-alpha * d_ij) and A_j^b, so this avoids
    recomputing the expensive dense kernel when the same alpha/b are needed.
    """
    a = np.asarray(areas, dtype=float)
    d = np.asarray(dist, dtype=float)
    n = int(a.size)
    if a.shape != (n,):
        raise ValueError("cluster_areas has wrong shape.")
    if d.shape != (n, n):
        raise ValueError("cluster_dist_matrix has wrong shape.")

    if n == 0:
        res = MetapopCapacityResult(
            lambda_max=float("nan"),
            eigenvector=np.zeros(0, dtype=float),
            n_clusters=0,
            converged=False,
            n_iter=0,
            tol=float(tol),
        )
        return res, 0.0, np.zeros((0, 0), dtype=float) if return_matrix else None

    K = prepare_dispersal_kernel(
        d,
        alpha=float(alpha),
        kernel_epsilon=float(kernel_epsilon),
        dtype=kernel_dtype,
    )
    return metapopulation_capacity_and_mean_si_from_kernel(
        a,
        K,
        b=float(b),
        x=float(x),
        self_colonization=self_colonization,
        max_iter=int(max_iter),
        tol=float(tol),
        rng=rng,
        return_matrix=return_matrix,
        method=method,
    )


def metapopulation_capacity(
    landscape: Any,
    *,
    alpha: float,
    b: float,
    x: float,
    self_colonization: float = 0.0,
    edge_depth: float = 0.0,
    connectivity: str | int = "4",
    boundary_mode: str = "periodic",
    habitat_threshold: float = 0.5,
    distance_mode: str = "voronoi_dt",
    softmin_tau: float = 1.0,
    voronoi_depth: int | str | None = 1,
    rebuild_clusters: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
    return_matrix: bool = False,
    method: Literal["matrix", "symmetric"] = "matrix",
) -> Tuple[MetapopCapacityResult, Optional[np.ndarray]]:
    """
    Metapopulation capacity (largest eigenvalue) for an IFM/Hanski-style landscape.

    Build:
      K_ij = exp(-alpha * d_ij), K_ii = 0
      M_ij = A_i^x K_ij A_j^b for i != j
      M_ii = self_colonization * A_i^(x+b)
    Then lambda_max = spectral radius of M.
    """
    conn = _normalize_connectivity(connectivity)

    if rng is None:
        rng = np.random.default_rng(0)

    # Ensure cluster attributes exist
    if hasattr(landscape, "ensure_clusters"):
        landscape.ensure_clusters(
            edge_depth=float(edge_depth),
            connectivity=conn,
            boundary_mode=str(boundary_mode),
            habitat_threshold=float(habitat_threshold),
            distance_mode=str(distance_mode),
            softmin_tau=float(softmin_tau),
            voronoi_depth=voronoi_depth,
            rebuild=bool(rebuild_clusters),
        )
    elif hasattr(landscape, "ensure_patches"):
        legacy_conn = 1 if conn == "4" else 2
        landscape.ensure_patches(
            edge_depth=float(edge_depth),
            connectivity=int(legacy_conn),
            habitat_threshold=float(habitat_threshold),
            rebuild=bool(rebuild_clusters),
        )
    else:
        raise AttributeError("Landscape must implement ensure_clusters(...) or ensure_patches(...).")

    n = int(_get_attr(landscape, "n_clusters", "n_patches"))
    areas = np.asarray(_get_attr(landscape, "cluster_areas", "patch_areas"), dtype=float)
    dist = np.asarray(_get_attr(landscape, "cluster_dist_matrix", "patch_dist_matrix"), dtype=float)

    if areas.shape != (n,):
        raise ValueError("cluster_areas has wrong shape.")
    if dist.shape != (n, n):
        raise ValueError("cluster_dist_matrix has wrong shape.")

    return metapopulation_capacity_from_arrays(
        areas,
        dist,
        alpha=float(alpha),
        b=float(b),
        x=float(x),
        self_colonization=self_colonization,
        max_iter=int(max_iter),
        tol=float(tol),
        rng=rng,
        return_matrix=return_matrix,
        method=method,
    )


# ============================================================
# Local tests (smoke + plausibility)
# ============================================================

if __name__ == "__main__":
    import numpy as _np

    def _assert(cond: bool, msg: str) -> None:
        assert cond, msg

    def _assert_in_range(x: float, lo: float, hi: float, msg: str) -> None:
        _assert((x >= lo) and (x <= hi), msg)

    def _assert_close(a: float, b: float, tol: float = 1e-8, msg: str = "") -> None:
        _assert(abs(float(a) - float(b)) <= tol, msg or f"Expected {b}, got {a}")

    # Use the real Landscape class if available (recommended)
    try:
        from src.landscape.landscape import Landscape as _Landscape
    except Exception as e:
        raise RuntimeError("Test requires src.landscape.landscape.Landscape to be importable.") from e

    # Small grid with two separated forest components (two clusters under 4-neighborhood)
    grid = _np.zeros((6, 6), dtype=int)
    grid[0:2, 0:2] = 1
    grid[4:6, 4:6] = 1

    L = _Landscape(grid)
    model = HanskiModel(L, edge_depth=0.0, connectivity="4", habitat_threshold=0.5, rng=_np.random.default_rng(123))
    model._ensure(rebuild=True)

    _assert(model.n_clusters == 2, "Expected 2 clusters under connectivity='4'")

    # 1) Colonization forms: basic invariants
    S_test = _np.array([0.0, 0.5, 2.0], dtype=float)

    cfg_exp = HanskiConfig(colonization_mode="exp", c=1.0)
    C_exp = model._colonization_prob(S_test, cfg_exp)
    _assert_close(C_exp[0], 0.0, msg="exp colonization must be 0 at S=0")
    _assert(_np.all((C_exp >= 0.0) & (C_exp <= 1.0)), "exp colonization must be in [0,1]")

    cfg_lin = HanskiConfig(colonization_mode="linear", c=1.0, clip_probs=True)
    C_lin = _np.clip(model._colonization_prob(S_test, cfg_lin), 0.0, 1.0)
    _assert_close(C_lin[0], 0.0, msg="linear colonization must be 0 at S=0")
    _assert(_np.all((C_lin >= 0.0) & (C_lin <= 1.0)), "linear colonization (clipped) must be in [0,1]")

    cfg_sig = HanskiConfig(colonization_mode="sigmoid", c=1.0, hill_h=2.0)
    C_sig = model._colonization_prob(S_test, cfg_sig)
    _assert_close(C_sig[0], 0.0, msg="sigmoid colonization must be 0 at S=0")
    _assert(_np.all((C_sig >= 0.0) & (C_sig <= 1.0)), "sigmoid colonization must be in [0,1]")

    print("[OK] colonization_mode invariants")

    # 2) Run smoke test
    cfg = HanskiConfig(
        c=1.0,
        alpha=0.2,
        b=1.0,
        e0=0.2,
        x=0.5,
        gamma=0.0,
        colonization_mode="exp",
        clip_probs=True,
    )
    res = model.run(steps=20, config=cfg, initial="none", rebuild_clusters=False)

    _assert(res.states.shape == (21, model.n_clusters), "states shape mismatch")
    _assert(res.states.dtype == bool, "states dtype must be bool")
    _assert(res.occupancy_fraction.shape == (21,), "occupancy_fraction shape mismatch")
    _assert_in_range(float(res.occupancy_fraction.min()), 0.0, 1.0, "occupancy_fraction out of [0,1]")
    _assert_in_range(float(res.occupancy_fraction.max()), 0.0, 1.0, "occupancy_fraction out of [0,1]")

    print("[OK] run() smoke + bounds")

    # 2b) Deterministic expectation dynamics smoke + formula check
    exp_res = model.run_expectation(steps=5, config=cfg, initial=0.25, rebuild_clusters=False)
    _assert(exp_res.probabilities.shape == (6, model.n_clusters), "probabilities shape mismatch")
    _assert(_np.issubdtype(exp_res.probabilities.dtype, _np.floating), "probabilities dtype must be float")
    _assert(exp_res.occupancy_fraction.shape == (6,), "expectation occupancy_fraction shape mismatch")
    _assert_in_range(float(exp_res.probabilities.min()), 0.0, 1.0, "probabilities out of [0,1]")
    _assert_in_range(float(exp_res.probabilities.max()), 0.0, 1.0, "probabilities out of [0,1]")

    K = model._kernel_matrix(cfg.alpha)
    p0 = _np.full(model.n_clusters, 0.25, dtype=float)
    p1 = model.step_expectation(p0, cfg, K)
    S0 = model._connectivity(p0, cfg, K)
    C0 = model._colonization_prob(S0, cfg)
    E0 = model._extinction_prob(S0, cfg, model.cluster_areas)
    if cfg.clip_probs:
        C0 = _np.clip(C0, 0.0, 1.0)
        E0 = _np.clip(E0, 0.0, 1.0)
    p1_manual = p0 * (1.0 - E0) + (1.0 - p0) * C0
    if cfg.clip_probs:
        p1_manual = _np.clip(p1_manual, 0.0, 1.0)
    _assert(_np.allclose(p1, p1_manual), "deterministic expectation step formula mismatch")
    _assert(_np.allclose(exp_res.probabilities[1], p1_manual), "stored p trajectory must match first deterministic step")
    print("[OK] run_expectation() smoke + formula")

    # 3) state_to_raster smoke test
    raster = model.state_to_raster(res.states[-1])
    _assert(raster.shape == grid.shape, "state_to_raster shape mismatch")
    _assert(_np.all(_np.isin(raster, [0, 1, 2])), "state_to_raster values must be in {0,1,2}")
    print("[OK] state_to_raster")

    # 4) Metapopulation capacity smoke test
    cap, _ = metapopulation_capacity(
        L,
        alpha=0.2,
        b=1.0,
        x=0.5,
        edge_depth=0.0,
        connectivity="4",
        habitat_threshold=0.5,
        rebuild_clusters=False,
        rng=_np.random.default_rng(0),
        return_matrix=False,
    )
    _assert(cap.n_clusters == model.n_clusters, "capacity n_clusters mismatch")
    _assert(cap.lambda_max >= 0.0 or _np.isnan(cap.lambda_max), "lambda_max must be non-negative (or NaN)")
    if cap.eigenvector.size > 0:
        _assert_close(float(_np.linalg.norm(cap.eigenvector)), 1.0, tol=1e-6, msg="eigenvector must be L2-normalized")
    print("[OK] metapopulation_capacity")

    uneven = _np.zeros((7, 7), dtype=int)
    uneven[0:2, 0:2] = 1
    uneven[5, 5] = 1
    L_uneven = _Landscape(uneven)
    L_uneven.ensure_clusters(edge_depth=0.0, connectivity="4", habitat_threshold=0.5, rebuild=True)
    cap_matrix, M = metapopulation_capacity(
        L_uneven,
        alpha=0.2,
        b=1.0,
        x=0.5,
        edge_depth=0.0,
        connectivity="4",
        habitat_threshold=0.5,
        rebuild_clusters=False,
        rng=_np.random.default_rng(0),
        return_matrix=True,
    )
    cap_sym, _ = metapopulation_capacity(
        L_uneven,
        alpha=0.2,
        b=1.0,
        x=0.5,
        edge_depth=0.0,
        connectivity="4",
        habitat_threshold=0.5,
        rebuild_clusters=False,
        rng=_np.random.default_rng(0),
        return_matrix=False,
        method="symmetric",
    )
    eigvals = _np.linalg.eigvals(M)
    spectral_radius = float(_np.max(_np.abs(eigvals))) if eigvals.size else float("nan")
    _assert_close(
        cap_matrix.lambda_max,
        spectral_radius,
        tol=1e-7,
        msg="matrix capacity iteration must match explicit spectral radius",
    )
    _assert_close(
        cap_sym.lambda_max,
        spectral_radius,
        tol=1e-7,
        msg="symmetric capacity iteration must match explicit spectral radius",
    )
    if cap_matrix.eigenvector.size > 0 and _np.isfinite(cap_matrix.lambda_max):
        residual = _np.linalg.norm(M @ cap_matrix.eigenvector - cap_matrix.lambda_max * cap_matrix.eigenvector)
        _assert(residual <= 1e-6 * max(1.0, abs(cap_matrix.lambda_max)), "capacity eigenvector residual too large")
    print("[OK] metapopulation_capacity symmetric equivalence")

    cap_joint, si_joint, _ = metapopulation_capacity_and_mean_si_from_arrays(
        L_uneven.cluster_areas,
        L_uneven.cluster_dist_matrix,
        alpha=0.2,
        b=1.0,
        x=0.5,
        rng=_np.random.default_rng(0),
        return_matrix=False,
    )
    K = _np.exp(-0.2 * _np.asarray(L_uneven.cluster_dist_matrix, dtype=float))
    _np.fill_diagonal(K, 0.0)
    si_separate = float(_np.mean(K @ (_np.asarray(L_uneven.cluster_areas, dtype=float) ** 1.0)))
    _assert_close(cap_joint.lambda_max, cap_matrix.lambda_max, tol=1e-7, msg="joint capacity mismatch")
    _assert_close(si_joint, si_separate, tol=1e-10, msg="joint mean S_i mismatch")
    print("[OK] joint capacity + mean_full_occupancy_Si")

    prepared_kernel = prepare_dispersal_kernel(
        L_uneven.cluster_dist_matrix,
        alpha=0.2,
        kernel_epsilon=0.0,
    )
    cap_reused, si_reused, _ = metapopulation_capacity_and_mean_si_from_kernel(
        L_uneven.cluster_areas,
        prepared_kernel,
        b=1.0,
        x=0.5,
        rng=_np.random.default_rng(0),
    )
    _assert_close(cap_reused.lambda_max, cap_joint.lambda_max, tol=1e-7, msg="reused-kernel capacity mismatch")
    _assert_close(si_reused, si_joint, tol=1e-10, msg="reused-kernel mean S_i mismatch")
    occupancy, final_step = run_expectation_from_kernel(
        L_uneven.cluster_areas,
        prepared_kernel,
        c=1.0,
        e0=0.2,
        b=1.0,
        x=0.5,
        steps=10,
        initial=1.0,
    )
    _assert_in_range(occupancy, 0.0, 1.0, "reused-kernel occupancy out of [0,1]")
    _assert(final_step == 10, "unexpected deterministic expectation step count")
    print("[OK] reusable kernel capacity + metrics + expectation")

    print("\nAll hanski.py tests passed.")
