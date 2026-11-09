"""Scenario-balanced weighted Spearman associations on paired landscapes.

Weights define the empirical distribution. Both ranks are weighted mid-CDFs,
and their Pearson correlation uses the same weights. Integer frequency weights
are equivalent to literal replication, including ties.
"""
from __future__ import annotations
import numpy as np


def weighted_midranks(values, weights):
    """Vectorised weighted mid-distribution ranks on the last axis."""
    a, w = np.broadcast_arrays(np.asarray(values, float), np.asarray(weights, float))
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError('Finite nonnegative weights required')
    w = np.where(np.isfinite(a), w, 0.)
    order = np.argsort(a, axis=-1, kind='stable')
    v = np.take_along_axis(a, order, -1)
    sw = np.take_along_axis(w, order, -1)
    cumulative = np.cumsum(sw, axis=-1)
    n = a.shape[-1]
    index = np.arange(n)
    start = np.concatenate([np.ones(v.shape[:-1]+(1,), bool), v[..., 1:] != v[..., :-1]], -1)
    end = np.concatenate([v[..., :-1] != v[..., 1:], np.ones(v.shape[:-1]+(1,), bool)], -1)
    first = np.maximum.accumulate(np.where(start, index, 0), axis=-1)
    last = np.minimum.accumulate(np.where(end, index, n-1)[..., ::-1], axis=-1)[..., ::-1]
    before = np.concatenate([np.zeros(v.shape[:-1]+(1,)), cumulative[..., :-1]], -1)
    ranked = (np.take_along_axis(before, first, -1) + np.take_along_axis(cumulative, last, -1))/2
    result = np.empty_like(ranked)
    np.put_along_axis(result, order, ranked, -1)
    return result


def balanced_spearman(x, y, groups, frequencies=None, minimum_per_group=3,
                      minimum_response_range=1e-10):
    """Equal total weight for each required scenario after paired exclusions.

    x, y broadcast over [..., landscape]. All groups must have the configured
    number of valid observations (with bootstrap multiplicity); none is silently
    dropped. Zero-weight and missing observations cannot affect variation checks.
    """
    if frequencies is None:
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        f = np.ones(x.shape)
    else:
        x, y, f = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(frequencies, float))
    groups = np.asarray(groups)
    if groups.shape != (x.shape[-1],):
        raise ValueError('One group label required per landscape')
    if np.any(~np.isfinite(f)) or np.any(f < 0):
        raise ValueError('Finite nonnegative frequencies required')
    valid = np.isfinite(x) & np.isfinite(y) & (f > 0)
    f = np.where(valid, f, 0.)
    w = np.zeros_like(f)
    eligible = np.ones(x.shape[:-1], bool)
    for group in np.unique(groups):
        mask = groups == group
        total = f[..., mask].sum(axis=-1)
        eligible &= total >= minimum_per_group
        w[..., mask] = f[..., mask]/np.where(total > 0, total, 1)[..., None]
    w /= np.where(w.sum(axis=-1) > 0, w.sum(axis=-1), 1)[..., None]
    xr, yr = weighted_midranks(x, w), weighted_midranks(y, w)
    xr -= (xr*w).sum(axis=-1, keepdims=True)
    yr -= (yr*w).sum(axis=-1, keepdims=True)
    numerator = (w*xr*yr).sum(axis=-1)
    denominator = np.sqrt((w*xr*xr).sum(axis=-1)*(w*yr*yr).sum(axis=-1))
    def span(a):
        return np.max(np.where(valid, a, -np.inf), axis=-1)-np.min(np.where(valid, a, np.inf), axis=-1)
    eligible &= (span(y) > minimum_response_range) & (span(x) > 0) & (denominator > 1e-28)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.clip(numerator/denominator, -1, 1)
    return np.where(eligible, result, np.nan)
