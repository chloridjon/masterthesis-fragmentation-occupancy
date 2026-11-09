"""Transparent selection of a rectangle from resolved occupancy transitions."""
from __future__ import annotations
import numpy as np
import pandas as pd


def transition_crossing(log10_r, occupancy, target):
    """Linear crossing in log10(r), with its actual sampled bracket retained."""
    x=np.asarray(log10_r,float); y=np.asarray(occupancy,float)
    if len(x)<2 or np.any(np.diff(x)<=0):
        raise ValueError('Ordered distinct ratio levels required.')
    if not np.isfinite(y).all():
        return dict(value=np.nan,low=np.nan,high=np.nan,status='missing')
    if np.any(np.diff(y)>1e-7):
        raise ValueError('Occupancy must be nonincreasing along r.')
    if y[0]<target:
        return dict(value=-np.inf,low=-np.inf,high=x[0],status='below_grid')
    if y[-1]>target:
        return dict(value=np.inf,low=x[-1],high=np.inf,status='above_grid')
    i=np.flatnonzero((y[:-1]>=target)&(y[1:]<=target))[0]
    value=x[i] if y[i]==y[i+1] else x[i]+(target-y[i])/(y[i+1]-y[i])*(x[i+1]-x[i])
    return dict(value=float(value),low=float(x[i]),high=float(x[i+1]),status='resolved')


def select_transition_rectangle(transitions, *, lower_half_distance=1.,
        central_fraction=.9, plateau_tolerance=.05, consecutive_plateau_steps=2,
        margin_log10_r=.25):
    """Equal scenario-H-cell weighting; censored tails cannot silently vanish."""
    levels=np.sort(transitions.half_distance.unique())
    if not (0<central_fraction<1 and plateau_tolerance>0 and margin_log10_r>=0):
        raise ValueError('Invalid selection settings.')
    if lower_half_distance not in levels or len(levels)<consecutive_plateau_steps+1:
        raise ValueError('Insufficient sampled half-distance range.')
    keys=['scenario_id','H_target']
    matrix=transitions.pivot(index=keys,columns='half_distance',values='p50')
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError('All eligible median transitions must be resolved.')
    steps=[]
    for left,right in zip(levels[:-1],levels[1:]):
        if not np.isclose(right/left,2.):
            raise ValueError('Plateau diagnostics require doubling steps.')
        shifts=matrix[right]-matrix[left]
        if (shifts < -1e-6).any(): raise ValueError('Transitions decrease with dispersal.')
        steps.append(dict(half_distance_low=float(left),half_distance_high=float(right),
            median_shift_log10_r=float(np.median(shifts)),
            q95_shift_log10_r=float(np.quantile(shifts,.95)),
            maximum_shift_log10_r=float(shifts.max()),n_cells=len(shifts)))
    plateau=None
    for i in range(len(steps)-consecutive_plateau_steps+1):
        if all(s['q95_shift_log10_r']<=plateau_tolerance for s in steps[i:]):
            plateau=steps[i]['half_distance_low']; break
    if plateau is None: raise ValueError('Plateau not confirmed within pilot range.')
    retained=transitions[transitions.half_distance.between(lower_half_distance,plateau)]
    tail=(1-central_fraction)/2
    # Use the outside edges of the sampled crossing brackets, then add a margin.
    # Thus interpolation alone cannot narrow the main rectangle.
    endpoints=[]
    for ell,part in retained.groupby('half_distance'):
        low=float(np.quantile(part.p95_low.to_numpy(),tail,method='nearest'))
        high=float(np.quantile(part.p05_high.to_numpy(),1-tail,method='nearest'))
        if not np.isfinite([low,high]).all():
            raise ValueError('Central transition envelope is censored; extend ratio grid.')
        endpoints.append(dict(half_distance=float(ell),log10_r_low=low,log10_r_high=high))
    return dict(half_distance_min=float(lower_half_distance),half_distance_max=float(plateau),
        log10_r_min=min(e['log10_r_low'] for e in endpoints)-margin_log10_r,
        log10_r_max=max(e['log10_r_high'] for e in endpoints)+margin_log10_r,
        n_eligible_cells=len(matrix),plateau_steps=steps,transition_envelopes=endpoints)
