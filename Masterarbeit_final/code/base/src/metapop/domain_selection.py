"""Limit-based species domains and numerically scaled maximal equilibria.

Row normalization is an algebraic solver scaling, not a normalized dispersal
model: the original row strengths are retained in logarithmic form.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp,expit
from scipy.sparse.linalg import LinearOperator,eigsh
from src.metapop.hanski import MetapopEquilibriumResult


def half_distance_bounds(d_min_positive,d_max,epsilon=.01):
    if not (0<d_min_positive<=d_max and 0<epsilon<.5):
        raise ValueError('Positive distance extrema and kernel tolerance required.')
    return (np.log(2)/np.log(1/epsilon)*d_min_positive,
            np.log(2)/(-np.log1p(-epsilon))*d_max)


def normalized_colonization(areas,distances,half_distance,*,q=0.,b=1.,x=.5):
    a=np.asarray(areas,float); d=np.asarray(distances,float)
    if d.shape!=(len(a),len(a)) or len(a)<2 or np.any(a<=0) or np.any(d<0) or not np.isfinite(d).all() or half_distance<=0 or q<0:
        raise ValueError('Invalid geometry or model parameters.')
    log_weights=-np.log(2)/half_distance*d
    log_weights += b*np.log(a)[None,:]
    np.fill_diagonal(log_weights, np.log(q)+b*np.log(a) if q else -np.inf)
    log_sources=logsumexp(log_weights,axis=1)
    if not np.isfinite(log_sources).all(): raise ValueError('No finite colonization source in a row.')
    log_weights -= log_sources[:,None]
    np.exp(log_weights,out=log_weights)
    return log_weights,log_sources+x*np.log(a)


def scaled_ratio_sweep(weights,log_row_strengths,log_ratios,*,max_iter=20000,tol=1e-10):
    """Solve the original model using F_i=(B p)_i/((B p)_i+r/Q_i).

    log(Q_i) retains absolute unnormalised colonization strength, including
    areas and self-colonization. Ratios need not fit a float64 value.
    """
    w=np.asarray(weights,float); strengths=np.asarray(log_row_strengths,float)
    ratios=np.atleast_1d(np.asarray(log_ratios,float))
    if w.shape!=(len(strengths),len(strengths)) or np.any(w<0) or not np.isfinite(strengths).all() or not np.isfinite(ratios).all():
        raise ValueError('Invalid scaled model.')
    if max_iter<1 or tol<=0: raise ValueError('Invalid stopping settings.')
    np.testing.assert_allclose(w.sum(axis=1),1,atol=1e-12,rtol=0)
    p=np.ones((len(strengths),len(ratios)))
    count=np.zeros(len(ratios),int); delta=np.full(len(ratios),np.inf); conv=np.zeros(len(ratios),bool)
    # Collatz row-sum bound proves absence of a positive equilibrium here.
    empty=ratios>strengths.max()+1e-12
    p[:,empty]=0.; conv[empty]=True; delta[empty]=0.
    active=np.where(~empty)[0]
    for step in range(1,max_iter+1):
        if not len(active): break
        sources=w@p[:,active]
        with np.errstate(divide='ignore'):
            target=expit(np.log(sources)-ratios[active][None,:]+strengths[:,None])
        delta[active]=np.max(np.abs(target-p[:,active]),axis=0)
        p[:,active]=target; count[active]=step
        done=delta[active]<=max(tol,8*np.finfo(float).eps)
        conv[active[done]]=True; active=active[~done]
    return [MetapopEquilibriumResult(p[:,j].copy(),float(p[:,j].mean()),int(count[j]),bool(conv[j]),float(delta[j])) for j in range(len(ratios))]


def scaled_residual(weights,strengths,log_ratio,p):
    with np.errstate(divide='ignore'):
        target=expit(np.log(weights@p)-log_ratio+strengths)
    return float(np.max(np.abs(target-p)))


@dataclass
class CapacityBracket:
    value:float
    lower:float
    upper:float
    relative_residual:float


def capacity_bracket(areas,distances,half_distance,*,q=0.,b=1.,x=.5,tol=1e-10,max_iter=2000):
    a=np.asarray(areas,float); d=np.asarray(distances,float)
    k=np.exp(-np.log(2)/half_distance*d); np.fill_diagonal(k,q)
    weight=a**((b+x)/2)
    def mv(v): return weight*(k@(weight*v))
    if len(a)<=3:
        m=weight[:,None]*k*weight[None,:]
        values,vectors=np.linalg.eigh(m); value=float(values[-1]); v=vectors[:,-1]
    else:
        op=LinearOperator(k.shape,matvec=mv,dtype=float)
        values,vectors=eigsh(op,k=1,which='LA',tol=tol,maxiter=max_iter,v0=np.ones(len(a))/np.sqrt(len(a)))
        value=float(values[0]); v=vectors[:,0]
    if v.sum()<0: v=-v
    if np.any(v<=0): raise ArithmeticError('Capacity eigenvector is not strictly positive.')
    product=mv(v); ratios=product/v
    residual=float(np.max(np.abs(product-value*v))/max(value,np.finfo(float).tiny))
    return CapacityBracket(value,float(ratios.min()),float(ratios.max()),residual)


def lower_occupancy_bracket(weights,strengths,*,target=.99,log_tolerance=1e-5,max_iter=20000,retry_max_iter=100000,tol=1e-10):
    """Return feasible and infeasible logarithmic endpoints around mean target."""
    if not 0<target<1: raise ValueError('Occupancy target must lie strictly between 0 and 1.')
    feasible=float(np.log1p(-target)+strengths.min())
    infeasible=float(strengths.max()+np.log(2))
    calls=0; max_iterations=0
    def solve(lr):
        nonlocal calls,max_iterations
        result=scaled_ratio_sweep(weights,strengths,[lr],max_iter=max_iter,tol=tol)[0]
        if not result.converged:
            result=scaled_ratio_sweep(weights,strengths,[lr],max_iter=retry_max_iter,tol=tol)[0]
        if not result.converged: raise ArithmeticError('Lower-bound equilibrium did not converge.')
        calls+=1; max_iterations=max(max_iterations,result.n_iter)
        return result
    low=solve(feasible); high=solve(infeasible)
    # Move slightly inside the guaranteed box if rounding puts the mean below target.
    if low.occupancy_fraction<target:
        feasible-=log_tolerance; low=solve(feasible)
    if low.occupancy_fraction<target or high.occupancy_fraction>=target:
        raise ArithmeticError('Invalid lower-bound bracket.')
    while infeasible-feasible>log_tolerance:
        middle=(feasible+infeasible)/2; result=solve(middle)
        if result.occupancy_fraction>=target: feasible=middle; low=result
        else: infeasible=middle; high=result
    return dict(log_r_feasible=feasible,log_r_infeasible=infeasible,
                occupancy_feasible=low.occupancy_fraction,occupancy_infeasible=high.occupancy_fraction,
                bracket_log_width=infeasible-feasible,n_solver_calls=calls,max_iterations=max_iterations,
                residual_feasible=scaled_residual(weights,strengths,feasible,low.probabilities)),low.probabilities,high.probabilities


def scientific_ratio(log_r):
    exponent=int(np.floor(log_r/np.log(10)))
    mantissa=np.exp(log_r-exponent*np.log(10))
    return f'{mantissa:.12g}e{exponent:+d}'
