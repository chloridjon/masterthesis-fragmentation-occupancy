"""Conservative certificates for saturation boundaries of the q=0 model.

Bounds are for landscape quantiles at fixed scenario/H, not interpolated
occupancy predictions. Every source interval must already be verified.
"""
from __future__ import annotations
import numpy as np


def parameter_order(half_distances, log_ratios, ell, log_r, distance_upper):
    """Entrywise bounds on K/r, including different kernel scales.

For d in [0,D], log(K_new/K_old)=ln(2)d(1/ell_old-1/ell_new).
Taking its minimum/maximum gives sufficient occupancy-order conditions.
The exact same parameter point is included; a small outward margin avoids
certifying comparisons based on rounding at a nonzero kernel-ratio bound.
"""
    e=np.asarray(half_distances,float);r=np.asarray(log_ratios,float)
    if np.any(e<=0) or ell<=0 or distance_upper<0:raise ValueError('Invalid kernel scales')
    penalty=np.log(2)*distance_upper*(1/ell-1/e)
    more=log_r-r<=-np.maximum(penalty,0)-1e-12
    less=log_r-r>=np.maximum(-penalty,0)+1e-12
    same=(e==ell)&(r==log_r)
    return more|same,less|same


def quantile_bounds(ell_values,log_ratios,lower,upper,ell,log_r,distance_upper):
    lower,upper=np.asarray(lower),np.asarray(upper)
    if lower.shape!=upper.shape or lower.ndim!=3 or lower.shape[1]!=2:
        raise ValueError('Expected source x two quantiles x eligible cells')
    if np.any(lower>upper+1e-12):raise ValueError('Inverted source intervals')
    more,less=parameter_order(ell_values,log_ratios,ell,log_r,distance_upper)
    lo=lower[more].max(axis=0) if more.any() else np.zeros(lower.shape[1:])
    hi=upper[less].min(axis=0) if less.any() else np.ones(upper.shape[1:])
    if np.any(lo>hi+1e-9):raise ValueError('Contradictory response bounds')
    # Tiny numerical discrepancies cannot strengthen an interval.
    return np.minimum(lo,hi),np.maximum(lo,hi)


def envelope_certificate(lower,upper,epsilon=.05):
    lower,upper=np.asarray(lower),np.asarray(upper)
    if lower.shape!=upper.shape or lower.shape[0]!=2 or np.any(lower>upper+1e-12):
        raise ValueError('Invalid quantile intervals')
    Llo,Lhi=float(lower[0].min()),float(upper[0].min())
    Ulo,Uhi=float(lower[1].max()),float(upper[1].max())
    category=3
    if Llo>=1-epsilon:category=0
    elif Uhi<=epsilon:category=2
    elif Lhi<1-epsilon and Ulo>epsilon:category=1
    return dict(category=category,L_lower=Llo,L_upper=Lhi,U_lower=Ulo,U_upper=Uhi)


def cells_requiring_evaluation(lower,upper,epsilon=.05):
    """Cells whose exact quantiles are needed to decide the same global rule."""
    c=envelope_certificate(lower,upper,epsilon)
    if c['category']!=3:return np.zeros(lower.shape[1],bool)
    needed=np.zeros(lower.shape[1],bool)
    if c['L_upper']>=1-epsilon:
        needed|=(lower[0]<1-epsilon)&(upper[0]>=1-epsilon)
    if c['U_lower']<=epsilon:
        needed|=(lower[1]<=epsilon)&(upper[1]>epsilon)
    return needed


def boundary_bracket(ell_values,log_ratios,categories,ell,boundary,window):
    """Coordinate-order brackets; absent sides remain explicit window bounds."""
    e,r,c=np.asarray(ell_values),np.asarray(log_ratios),np.asarray(categories)
    if boundary=='lower':a=(e<=ell)&(c==0);b=(e>=ell)&np.isin(c,[1,2])
    elif boundary=='upper':a=(e<=ell)&np.isin(c,[0,1]);b=(e>=ell)&(c==2)
    else:raise ValueError('Unknown boundary')
    low=float(r[a].max()) if a.any() else float(window[0])
    high=float(r[b].min()) if b.any() else float(window[1])
    if low>high+1e-9:raise ValueError('Contradictory boundary witnesses')
    return low,high,bool(a.any()),bool(b.any())
