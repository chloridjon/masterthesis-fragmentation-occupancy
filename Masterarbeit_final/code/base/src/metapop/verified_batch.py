"""Raw continuous equilibria sharing an untruncated kernel, with residual checks."""
from __future__ import annotations
import numpy as np
from src.metapop.parameter_sweep import continuous_ratio_sweep


def raw_ratio_group(areas,distances,half_distance,ratios,*,b=1.,x=.5,q=0.,max_iter=20000,tol=1e-12,empty_check_probability=1e-4):
    """Include self-colonization once as kernel diagonal q; default is unchanged."""
    a=np.asarray(areas,float);d=np.asarray(distances,float);r=np.asarray(ratios,float)
    if half_distance<=0 or d.shape!=(len(a),len(a)) or np.any(d<0) or not np.isfinite(d).all():
        raise ValueError('Invalid geometry/kernel')
    if len(a)<2 or np.any(a<=0) or not np.isfinite(a).all() or np.any(r<=0) or not np.isfinite(r).all():
        raise ValueError('Positive raw areas/ratios in the multi-patch domain required')
    if not np.isfinite(q) or q<0:raise ValueError('Finite nonnegative self-colonization required')
    k=np.exp(-np.log(2)/half_distance*d);np.fill_diagonal(k,q)
    strength=(k@a**b)*a**x
    empty=r>strength.max()*(1+1e-13)
    p=np.zeros((len(a),len(r)));converged=empty.copy();steps=np.zeros(len(r),int);delta=np.zeros(len(r))
    active=np.flatnonzero(~empty)
    if len(active):
        answers=continuous_ratio_sweep(a,k,r[active],b=b,x=x,max_iter=max_iter,tol=tol)
        for index,result in zip(active,answers):
            p[:,index]=result.probabilities;converged[index]=result.converged
            steps[index]=result.n_iter;delta[index]=result.max_delta
    source=k@(a[:,None]**b*p);rates=a[:,None]**(-x)*r
    denom=source+rates;target=source/denom
    residual=np.max(np.abs(target-p),axis=0)
    balance=(1-p)*source-rates*p
    reports=[]
    for i in range(len(r)):
        needs=(not converged[i]) or residual[i]>tol or (p[:,i].max()<empty_check_probability and not empty[i])
        reports.append(dict(converged=bool(converged[i]),n_iter=int(steps[i]),max_delta=float(delta[i]),
            fixed_point_residual=float(residual[i]),relative_rate_balance=float(np.max(np.abs(balance[:,i])/denom[:,i])),
            solver='row_sum_verified_empty_equilibrium' if empty[i] else 'raw_continuous_shared_kernel_batch',
            needs_scalar_verification=bool(needs)))
    return p,reports,k
