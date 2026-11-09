"""Exact repeated-seed medians and fixed-design Spearman coefficients."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


class PreparedSeedMedian:
    """Pre-sort seeds once; integer frequencies reproduce expanded medians."""
    def __init__(self, values):
        a=np.asarray(values,dtype=float)
        if a.ndim<2:raise ValueError('Seed and response axes required')
        self.order=np.argsort(a,axis=0,kind='stable')
        self.sorted=np.take_along_axis(a,self.order,axis=0)
        self.finite=np.isfinite(self.sorted)
        self.n_seeds=len(a)

    def evaluate(self, frequencies, minimum_valid=3):
        f=np.asarray(frequencies)
        if f.ndim!=2 or f.shape[1]!=self.n_seeds or np.any(f<0) or not np.equal(f,np.floor(f)).all():
            raise ValueError('Nonnegative integer seed frequencies required')
        if f.sum(axis=1).max(initial=0)>65535:raise ValueError('Frequency total too large')
        ordered=f.astype(np.uint16)[:,self.order]
        ordered=np.where(self.finite[None],ordered,0)
        cumulative=np.cumsum(ordered,axis=1,dtype=np.uint16)
        total=cumulative[:,-1]
        # Cast before subtracting: zero valid seeds must not underflow unsigned.
        lower=(total.astype(np.int32)-1)//2+1;upper=total.astype(np.int32)//2+1
        il=np.argmax(cumulative>=lower[:,None],axis=1)
        iu=np.argmax(cumulative>=upper[:,None],axis=1)
        a=np.take_along_axis(self.sorted[None],il[:,None],axis=1)[:,0]
        b=np.take_along_axis(self.sorted[None],iu[:,None],axis=1)[:,0]
        result=(a+b)/2
        result[total<minimum_valid]=np.nan
        return result,total


def fixed_design_spearman(responses, predictors, minimum_range=1e-10):
    """Average-tie Spearman on a fixed complete parameter sample.

    responses[..., profile], predictors[parameter, profile]. A missing profile
    invalidates the entire coefficient; the parameter population never changes
    between draws. Numerically constant response vectors are undefined.
    """
    a,x=np.asarray(responses,float),np.asarray(predictors,float)
    if x.ndim!=2 or a.shape[-1]!=x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Finite fixed predictors must match the profile axis')
    yr=rankdata(a,axis=-1,method='average',nan_policy='propagate')
    xr=rankdata(x,axis=-1,method='average')
    yr=yr-yr.mean(axis=-1,keepdims=True);xr=xr-xr.mean(axis=-1,keepdims=True)
    numerator=yr@xr.T
    denom=np.sqrt(np.sum(yr*yr,axis=-1)[...,None]*np.sum(xr*xr,axis=-1))
    with np.errstate(invalid='ignore',divide='ignore'):rho=numerator/denom
    usable=np.isfinite(a).all(axis=-1)&(np.ptp(a,axis=-1)>minimum_range)
    rho=np.where(usable[...,None]&(denom>0),np.clip(rho,-1,1),np.nan)
    return rho


def two_cohort_indices(maximum_draws, seed, cohort_size=8):
    """Independent cohort streams, reproducible regardless of processing batch."""
    a,b=np.random.SeedSequence(seed).spawn(2)
    shared=np.random.default_rng(a).integers(0,cohort_size,size=(maximum_draws,cohort_size),dtype=np.uint8)
    extra=np.random.default_rng(b).integers(0,cohort_size,size=(maximum_draws,cohort_size),dtype=np.uint8)
    return shared,extra


def frequencies(indices,n=8):
    indices=np.asarray(indices)
    if indices.ndim!=2 or np.any(indices<0) or np.any(indices>=n):raise ValueError('Invalid seed indices')
    return np.eye(n,dtype=np.uint16)[indices].sum(axis=1,dtype=np.uint16)
