"""Maximal continuous equilibria with the local self term solved analytically."""
from __future__ import annotations
import numpy as np
from src.metapop.hanski import MetapopEquilibriumResult


def self_colonization_sweep(areas,kernel,ratios,*,q=1.,b=1.,x=.5,tol=1e-12,max_iter=100000):
    """Iterate external connectivity from above, solving each local quadratic.

    The kernel must have a zero diagonal. With external source E, local source
    u=q*A**b and extinction v=r*A**(-x), the positive root solves
    u*p**2+(E+v-u)*p-E=0. Starting from one selects the maximal equilibrium,
    including max(1-v/u,0) for disconnected patches. No distances are truncated.
    """
    a=np.asarray(areas,float);k=np.asarray(kernel,float);r=np.atleast_1d(np.asarray(ratios,float))
    if a.ndim!=1 or not len(a) or k.shape!=(len(a),len(a)) or np.any(np.diag(k)!=0):raise ValueError('Zero-diagonal square kernel required')
    if not np.isfinite(k).all() or np.any(k<0) or not np.isfinite(a).all() or np.any(a<=0) or not np.isfinite(r).all() or np.any(r<=0):raise ValueError('Finite positive areas and ratios, nonnegative kernel required')
    if not np.isfinite(q) or q<=0 or tol<=0 or max_iter<1:raise ValueError('Positive q, tolerance and iteration budget required')
    p=np.ones((len(a),len(r)));u=q*a[:,None]**b;v=a[:,None]**(-x)*r
    count=np.zeros(len(r),int);delta=np.full(len(r),np.inf);converged=np.zeros(len(r),bool)
    active=np.arange(len(r))
    for step in range(1,max_iter+1):
        external=k@(a[:,None]**b*p[:,active]);linear=external+v[:,active]-u
        discriminant=np.hypot(linear,2*np.sqrt(u*external))
        # Stable quadratic roots on either side of the local critical point.
        root=np.divide(2*external,linear+discriminant,out=np.zeros_like(external),where=(linear+discriminant)>0)
        root=np.where(linear<0,(discriminant-linear)/(2*u),root)
        root=np.clip(root,0,1)
        delta[active]=np.max(np.abs(root-p[:,active]),axis=0);p[:,active]=root;count[active]=step
        done=delta[active]<=tol;converged[active[done]]=True;active=active[~done]
        if not len(active):break
    return [MetapopEquilibriumResult(p[:,j].copy(),float(p[:,j].mean()),int(count[j]),bool(converged[j]),float(delta[j])) for j in range(len(r))]
