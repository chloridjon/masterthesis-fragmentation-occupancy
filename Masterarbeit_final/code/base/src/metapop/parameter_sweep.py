"""Exact, simultaneous continuous equilibria for ratios sharing one kernel."""
from __future__ import annotations

import numpy as np
from src.metapop.hanski import MetapopEquilibriumResult


def continuous_ratio_sweep(areas, kernel, ratios, *, b=1.0, x=0.5,
                           max_iter=20000, tol=1e-10):
    """Iterate from full occupancy; freeze columns only after convergence.

    No kernel truncation, area normalisation, or extrapolation is performed.
    Every ratio has its own stopping diagnostic, as in the scalar solver.
    """
    a, r = np.asarray(areas, float), np.asarray(ratios, float)
    if a.ndim != 1 or r.ndim != 1 or not len(a) or not len(r):
        raise ValueError("Nonempty one-dimensional areas and ratios required.")
    if np.any(~np.isfinite(a)) or np.any(a <= 0) or np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("Areas and ratios must be finite and positive.")
    if kernel.shape != (len(a), len(a)) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Invalid kernel shape or stopping settings.")
    if not np.isfinite(b) or not np.isfinite(x):
        raise ValueError("Area exponents must be finite.")
    p = np.ones((len(a), len(r)))
    source = a ** b
    extinction = a[:, None] ** (-x) * r[None, :]
    active = np.arange(len(r))
    iterations = np.zeros(len(r), dtype=int)
    deltas = np.full(len(r), np.inf)
    converged = np.zeros(len(r), dtype=bool)
    for step in range(1, max_iter + 1):
        s = np.maximum(kernel @ (source[:, None] * p[:, active]), 0.0)
        target = s / (s + extinction[:, active])
        deltas[active] = np.max(np.abs(target - p[:, active]), axis=0)
        p[:, active] = target
        iterations[active] = step
        done = deltas[active] <= max(tol, 8 * np.finfo(float).eps)
        converged[active[done]] = True
        active = active[~done]
        if not len(active):
            break
    return [MetapopEquilibriumResult(p[:, j].copy(), float(p[:, j].mean()),
                                    int(iterations[j]), bool(converged[j]), float(deltas[j]))
            for j in range(len(r))]
