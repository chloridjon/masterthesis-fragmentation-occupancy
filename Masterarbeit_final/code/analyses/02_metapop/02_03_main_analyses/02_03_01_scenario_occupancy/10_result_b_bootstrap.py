"""Adaptive pointwise contrasts from jointly resampled complete seed blocks."""
from __future__ import annotations
import warnings
import numpy as np


def medians(values,axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(values,axis=axis)


def paired_contrast_bootstrap(values,settings,*,minimum_seeds=3,minimum_valid_fraction=.95):
    """values has axes seed, scenario (baseline first), habitat proportion.

    Each bootstrap index matrix is shared by all scenarios AND habitat levels.
    Missing responses remain missing within their original complete seed block.
    Every point freezes only after its own consecutive stability checks pass.
    """
    a=np.asarray(values,float)
    if a.ndim!=3 or a.shape[0]<minimum_seeds or a.shape[1]<2:
        raise ValueError('Matched seed/scenario/H array required.')
    count=np.isfinite(a).sum(axis=0)
    eligible=(count[1:]>=minimum_seeds)&(count[0][None,:]>=minimum_seeds)
    shape=eligible.shape;done=np.zeros(shape,bool);steady=np.zeros(shape,int)
    low=np.full(shape,np.nan);high=low.copy();centre=low.copy()
    draws_used=np.zeros(shape,int);valid_fraction=np.full(shape,np.nan)
    previous=np.full((3,*shape),np.nan);history=[];blocks=[];total=0
    rng=np.random.default_rng(settings['seed'])
    while total<settings['maximum_draws'] and np.any(eligible&~done):
        n=settings['initial_draws'] if total==0 else settings['batch_draws']
        n=min(n,settings['maximum_draws']-total)
        indices=rng.integers(0,len(a),size=(n,len(a)))
        block=np.empty((n,*shape))
        for h in range(a.shape[2]):
            m=medians(a[indices,:,h],axis=1)
            block[:,:,h]=m[:,1:]-m[:,0,None]
        blocks.append(block);total+=n;samples=np.concatenate(blocks,axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            current=np.nanquantile(samples,[.025,.5,.975],axis=0)
        fraction=np.isfinite(samples).mean(axis=0)
        available=eligible & (fraction>=minimum_valid_fraction)
        changes=np.abs(current-previous)
        stable=available & np.isfinite(changes).all(axis=0)
        stable &= (changes[1]<=settings['central_tolerance'])
        stable &= (np.maximum(changes[0],changes[2])<=settings['endpoint_tolerance'])
        active=eligible&~done;steady[active]=np.where(stable[active],steady[active]+1,0)
        new=active & (steady>=settings['consecutive_stable_batches'])
        low[new]=current[0][new];centre[new]=current[1][new];high[new]=current[2][new]
        draws_used[new]=total;valid_fraction[new]=fraction[new];done[new]=True
        for s,h in zip(*np.where(active)):
            history.append(dict(scenario_index=int(s+1),h_index=int(h),draws=total,
                valid_bootstrap_fraction=float(fraction[s,h]),
                low=float(current[0,s,h]),centre=float(current[1,s,h]),high=float(current[2,s,h]),
                low_change=float(changes[0,s,h]),centre_change=float(changes[1,s,h]),high_change=float(changes[2,s,h]),
                consecutive_stable_batches=int(steady[s,h]),converged=bool(done[s,h])))
        previous=current
        print(f'Bootstrap: {total} matched-block draws; {done.sum()}/{eligible.sum()} eligible contrasts stable.',flush=True)
    unresolved=eligible&~done
    draws_used[unresolved]=total
    if total:valid_fraction[unresolved]=fraction[unresolved]
    status=np.full(shape,'undefined_scenario_or_baseline',dtype=object)
    status[eligible]='budget_exhausted';status[done]='converged'
    status[unresolved & (valid_fraction<minimum_valid_fraction)]='insufficient_bootstrap_availability'
    return dict(low=low,high=high,bootstrap_centre=centre,draws_used=draws_used,
                valid_fraction=valid_fraction,converged=done,status=status,
                eligible=eligible,history=history,total_draws=total)
