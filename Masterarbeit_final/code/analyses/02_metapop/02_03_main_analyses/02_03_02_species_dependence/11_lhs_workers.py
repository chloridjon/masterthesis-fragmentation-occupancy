"""Replay matched landscapes and compute resumable, untruncated LHS responses."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-main-mpl')
FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
from src.landscape.generation import run_sequential_deforestation
from src.landscape.landscape import Landscape
from src.metapop.hanski import run_continuous_equilibrium_from_kernel
from src.metapop.domain_selection import normalized_colonization, scaled_ratio_sweep, capacity_bracket


def atomic_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def solve(a, d, item, c):
    k = np.exp(-float(item.alpha)*d)
    np.fill_diagonal(k,0.)
    r = float(np.exp(item.log_r))
    rates = r*a**(-c['x'])
    strength = (k @ a**c['b'])*a**c['x']
    method = 'raw_continuous_fixed_point'
    iterations = 0
    capacity = None
    if r > strength.max():
        p = np.zeros(len(a)); delta = 0.; converged = True
        method = 'row_sum_verified_empty_equilibrium'
    else:
        result = run_continuous_equilibrium_from_kernel(a,k,r=r,b=c['b'],x=c['x'],
                    initial=1.,tol=c['solver']['tol'],max_iter=c['solver']['max_iter'])
        iterations = result.n_iter
        p = result.probabilities; delta = result.max_delta; converged = result.converged
        if not converged or p.max() < c['solver']['empty_verification_probability']:
            try:
                capacity = capacity_bracket(a,d,float(item.half_distance),q=0.,b=c['b'],x=c['x'],
                                           tol=1e-11,max_iter=10000)
            except ArithmeticError:
                capacity = None
            if capacity is not None and r > capacity.upper:
                p = np.zeros(len(a)); delta = 0.; converged = True
                method = 'capacity_verified_empty_equilibrium'
        if not converged:
            result = run_continuous_equilibrium_from_kernel(a,k,r=r,b=c['b'],x=c['x'],
                initial=p,tol=c['solver']['tol'],max_iter=c['solver']['continuation_max_iter'])
            iterations += result.n_iter
            p=result.probabilities; delta=result.max_delta; converged=result.converged
            method='raw_continuous_continued_fixed_point'
    source = k@(a**c['b']*p)
    denominator = source+rates
    target = source/denominator
    residual = float(np.max(np.abs(target-p)))
    # A stopping-step delta does not guarantee the following-step residual
    # for every asymmetric landscape. Continue the same maximal iterate until
    # both satisfy the unchanged tolerance; retain all additional iterations.
    if converged and residual > c['solver']['tol']:
        converged = False
        for extra in range(1,c['solver']['continuation_max_iter']+1):
            delta=residual; p=target
            source=k@(a**c['b']*p); denominator=source+rates
            target=source/denominator
            residual=float(np.max(np.abs(target-p)))
            if delta<=c['solver']['tol'] and residual<=c['solver']['tol']:
                converged=True; break
        iterations+=extra
        method += '_residual_refined'
    balance = (1-p)*source-rates*p
    relative_balance = float(np.max(np.abs(balance)/denominator))
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    if converged:
        assert residual <= c['solver']['tol']*1.1
        assert relative_balance <= c['solver']['tol']*1.1
    # Independent algebraic scaling for four LHS points on manageable fixtures.
    comparison = np.nan
    if converged and len(a)<=500 and item.parameter_id in c['scalar_check_ids']:
        w,s=normalized_colonization(a,d,float(item.half_distance),q=0.,b=c['b'],x=c['x'])
        other=scaled_ratio_sweep(w,s,[float(item.log_r)],tol=c['solver']['tol'],
                                 max_iter=c['solver']['continuation_max_iter'])[0]
        if other.converged:
            comparison = float(np.max(np.abs(other.probabilities-p)))
            assert comparison <= c['solver']['independent_probability_tolerance']
    return p,dict(occupancy=float(p.mean()) if converged else np.nan,
        last_iterate_mean=float(p.mean()),converged=converged,n_iter=iterations,
        max_delta=delta,fixed_point_residual=residual,relative_rate_balance=relative_balance,
        absolute_rate_balance=float(np.max(np.abs(balance))),solver=method,
        independent_probability_difference=comparison,
        capacity_upper=capacity.upper if capacity is not None else np.nan,
        status='valid' if converged else 'nonconverged')


def cached_snapshot(csv, c, fp, grid_hash):
    if not csv.exists(): return None
    marker=csv.with_suffix('.json')
    npz=csv.with_suffix('.npz')
    if not marker.exists() or not npz.exists(): return None
    meta=json.loads(marker.read_text())
    if meta['producer_fingerprint']!=fp or meta['grid_sha256']!=grid_hash:
        raise RuntimeError(f'Stale LHS snapshot: {csv}')
    for path in (csv,npz):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['hashes'][path.suffix]:
            raise RuntimeError(f'Corrupt LHS snapshot: {path}')
    frame=pd.read_csv(csv)
    assert len(frame)==128 and frame.parameter_id.nunique()==128
    assert frame.producer_fingerprint.eq(fp).all() and frame.grid_sha256.eq(grid_hash).all()
    with np.load(npz) as z:
        p=z['probabilities']; offsets=z['offsets']
        assert len(offsets)==129 and offsets[0]==0 and offsets[-1]==len(p)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        for i,row in frame.iterrows():
            values=p[offsets[i]:offsets[i+1]]
            assert len(values)==(row.n_patches if row.n_patches>=2 else 0)
            if row.status=='valid': assert abs(values.mean()-row.occupancy)<1e-14
    return frame


def trajectory(scenario, rep, c, lc, benchmark, lhs, directory, fp):
    started=time.monotonic(); directory=Path(directory)
    identity=f"{scenario['scenario_id']}__rep{rep:03d}"
    progress=directory/'progress'/f'{identity}.json'
    seed=lc['base_seed']+rep-1; pars=lc['access_reference']
    run=run_sequential_deforestation(np.ones((200,200),dtype=np.int8),np.random.default_rng(seed),
        rho=scenario['rho'],directed_mode=scenario['mode'],target_habitat_fraction=0.,
        generation_boundary=lc['generation_boundary'],eta=pars['eta'],k=pars['k'],
        lambda_decay=pars['lambda'],save_habitat_fractions=c['h_targets'],seed=seed)
    assert run.n_events==40000 and run.n_random_forced_initial==1
    assert run.n_random_total+run.n_directed==40000
    random_counts=np.r_[0,np.cumsum(np.asarray(run.event_modes)=='random')]
    checks=[]; all_rows=[]
    for number,snap in enumerate(run.snapshots,1):
        h=float(snap.requested_habitat_fractions[0]); old=benchmark[benchmark.H_target.eq(h)].iloc[0]
        grid_hash=hashlib.sha256(snap.grid.tobytes()).hexdigest()
        assert grid_hash==old.grid_sha256 and snap.habitat_fraction==h
        event=int(snap.event_count)
        assert event==round(40000*(1-h)) and int(random_counts[event])==old.n_random_cumulative
        csv=directory/'snapshots'/f'{identity}__h{h:.3f}.csv'
        prior=cached_snapshot(csv,c,fp,grid_hash)
        if prior is not None:
            all_rows.append(prior)
            atomic_json(progress,dict(trajectory=identity,completed_snapshots=number,
                                     total_snapshots=len(run.snapshots),last_H=h,elapsed_seconds=time.monotonic()-started))
            continue
        if h>0:
            land=Landscape(snap.grid)
            land.ensure_clusters(edge_depth=0,connectivity='4',boundary_mode='open',
                                 distance_mode='voronoi_dt',voronoi_depth=1)
            a=np.asarray(land.cluster_areas,float); d=np.asarray(land.cluster_dist_matrix,float)
        else:
            a=np.empty(0);d=np.empty((0,0))
        n=len(a)
        assert n==old.n_clusters and a.sum()==round(40000*h)
        assert d.shape==(n,n) and np.isfinite(d).all() and (d>=0).all()
        assert np.array_equal(d,d.T)
        # Independently compare the overlapping original pilot geometries.
        pilot=FINAL/'data/processed/02_metapop/02_02_species_parameter_selection/limit_based/geometry'/f'{identity}__h{h:.3f}.npz'
        if pilot.exists():
            with np.load(pilot) as z:
                assert str(z['grid_sha256'])==grid_hash
                np.testing.assert_array_equal(a,z['areas'])
                np.testing.assert_allclose(d,z['distances'],atol=1e-12,rtol=0)
            checks.append(h)
        records=[]; vectors=[];offsets=[0]
        for item in lhs.itertuples(index=False):
            if n>=2:
                p,report=solve(a,d,item,c)
            else:
                p=np.empty(0)
                report=dict(occupancy=np.nan,last_iterate_mean=np.nan,converged=False,n_iter=0,
                            max_delta=np.nan,fixed_point_residual=np.nan,relative_rate_balance=np.nan,
                            absolute_rate_balance=np.nan,solver='not_applicable',
                            independent_probability_difference=np.nan,capacity_upper=np.nan,
                            status='excluded_fewer_than_two_patches')
            records.append(dict(scenario_id=scenario['scenario_id'],replicate=rep,seed=seed,
                trajectory_id=identity,H_target=h,n_patches=n,grid_sha256=grid_hash,
                event_count=event,n_random_cumulative=int(random_counts[event]),
                n_directed_cumulative=event-int(random_counts[event]),q=0.,
                parameter_id=item.parameter_id,r=float(np.exp(item.log_r)),log_r=float(item.log_r),
                log10_r=float(item.log10_r),half_distance=float(item.half_distance),alpha=float(item.alpha),
                producer_fingerprint=fp,**report))
            vectors.append(p);offsets.append(offsets[-1]+len(p))
        frame=pd.DataFrame(records)
        tmp=csv.with_suffix('.csv.tmp');frame.to_csv(tmp,index=False);tmp.replace(csv)
        tmp=csv.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,probabilities=np.concatenate(vectors),offsets=np.array(offsets))
        tmp.replace(csv.with_suffix('.npz'))
        atomic_json(csv.with_suffix('.json'),dict(producer_fingerprint=fp,grid_sha256=grid_hash,
            hashes={p.suffix:hashlib.sha256(p.read_bytes()).hexdigest() for p in [csv,csv.with_suffix('.npz')]}))
        all_rows.append(frame)
        atomic_json(progress,dict(trajectory=identity,completed_snapshots=number,total_snapshots=len(run.snapshots),
            last_H=h,elapsed_seconds=time.monotonic()-started,pilot_geometry_checks=checks))
        if number%10==0:
            print(f'{identity}: {number}/{len(run.snapshots)} snapshots, H={h:g}',flush=True)
    full=pd.concat(all_rows,ignore_index=True)
    assert len(full)==len(c['h_targets'])*128
    tmp=directory/'trajectories'/f'{identity}.csv.tmp'
    full.to_csv(tmp,index=False);tmp.replace(tmp.with_suffix(''))
    return identity,len(full),int(full.status.eq('nonconverged').sum()),time.monotonic()-started
