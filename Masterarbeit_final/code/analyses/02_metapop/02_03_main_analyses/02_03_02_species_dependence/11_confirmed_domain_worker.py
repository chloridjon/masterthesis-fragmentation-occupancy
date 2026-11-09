"""Resumable equilibrium responses for the fixed 256-point conditional design."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_02_species_parameter_selection'))
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from src.metapop.verified_batch import raw_ratio_group
R=importlib.import_module('09_response_boundary_refinement')
W=importlib.import_module('11_lhs_workers')
scaled=R.original_scaled
def independent_reference(*args,**kwargs):
    kwargs['tol']=min(kwargs['tol'],1e-13)
    return scaled(*args,**kwargs)
W.scaled_ratio_sweep=independent_reference


def group(a,d,items,c):
    ell=float(items.half_distance.iloc[0]);assert items.half_distance.eq(ell).all()
    ratios=np.exp(items.log_r.to_numpy())
    p,reports,k=raw_ratio_group(a,d,ell,ratios,b=c['b'],x=c['x'],max_iter=c['solver']['max_iter'],
        tol=c['solver']['tol'],empty_check_probability=c['solver']['empty_verification_probability'])
    options={**c,'scalar_check_ids':[]}
    check_all=len(a)<=c['independent_checks']['all_parameters_max_patches']
    control=None
    if check_all or (len(a)<=500 and items.parameter_id.isin(c['independent_checks']['parameter_ids']).any()):
        w,s=W.normalized_colonization(a,d,ell,q=0.,b=c['b'],x=c['x'])
        control=scaled(w,s,items.log_r.to_numpy(),tol=c['solver']['independent_reference_tol'],max_iter=c['solver']['continuation_max_iter'])
    answers=[]
    for j,item in enumerate(items.itertuples(index=False)):
        used_tol=c['solver']['tol'];report=reports[j]
        if report.pop('needs_scalar_verification'):
            # W returns a scalar vector separately; preserve its exact values.
            scalar,report=W.solve(a,d,SimpleNamespace(parameter_id=item.parameter_id,alpha=item.alpha,
                log_r=item.log_r,half_distance=ell),options)
            p[:,j]=scalar
        assert report['converged'] and report['fixed_point_residual']<=1.1*used_tol
        difference=np.nan;checked=False
        if control is not None and (check_all or item.parameter_id in c['independent_checks']['parameter_ids']):
            other=control[j]
            assert other.converged,'Designated independent reference did not converge'
            if other.converged:
                checked=True;difference=float(np.max(np.abs(p[:,j]-other.probabilities)))
                if difference>c['solver']['independent_probability_tolerance']:
                    used_tol=c['solver']['precision_retry_tol']
                    tight={**options,'solver':{**c['solver'],'tol':used_tol}}
                    scalar,report=W.solve(a,d,SimpleNamespace(parameter_id=item.parameter_id,alpha=item.alpha,
                        log_r=item.log_r,half_distance=ell),tight)
                    p[:,j]=scalar;difference=float(np.max(np.abs(scalar-other.probabilities)))
                assert difference<=c['solver']['independent_probability_tolerance']
        report.update(occupancy=float(p[:,j].mean()),status='valid',primary_tolerance_used=used_tol,
            independent_probability_difference=difference,independent_check_converged=checked,
            independent_check_requested=bool(control is not None and (check_all or item.parameter_id in c['independent_checks']['parameter_ids'])))
        answers.append(report)
    return p,answers


def snapshot_cache(path,fp,grid_hash,parameter_ids):
    marker=path.with_suffix('.json')
    if not marker.exists():return None
    meta=json.loads(marker.read_text());assert meta['producer_fingerprint']==fp and meta['grid_sha256']==grid_hash
    for suffix,digest in meta['hashes'].items():assert R.sha(path.with_suffix(suffix))==digest
    table=pd.read_csv(path,float_precision='round_trip')
    assert table.parameter_id.tolist()==parameter_ids and len(table)==256
    with np.load(path.with_suffix('.npz')) as z:
        p,offsets=z['probabilities'],z['offsets'];assert len(offsets)==257 and offsets[0]==0 and offsets[-1]==len(p)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        for i,row in enumerate(table.itertuples(index=False)):
            v=p[offsets[i]:offsets[i+1]]
            assert len(v)==(row.n_patches if row.n_patches>=2 else 0)
            if row.status=='valid':assert abs(v.mean()-row.occupancy)<1e-14
    return table


def trajectory(scenario,rep,c,base,lc,benchmark,design,data_name,geometry_name,fp,gfp,pilot_H=None):
    started=time.monotonic();data=Path(data_name);geo=Path(geometry_name)
    identity=f"{scenario['scenario_id']}__rep{rep:03d}"
    file=data/'trajectories'/f'{identity}.csv';marker=file.with_suffix('.json')
    if marker.exists() and pilot_H is None:
        meta=json.loads(marker.read_text());assert meta['producer_fingerprint']==fp and R.sha(file)==meta['sha256']
        return identity,meta['elapsed_seconds'],True
    gs,hs=R.grids(scenario,rep,base,lc,benchmark,geo,gfp)
    records=[];total_chunks=0
    for grid,h in zip(gs,hs):
        h=float(h)
        if pilot_H is not None and h!=pilot_H:continue
        old=benchmark[benchmark.H_target.eq(h)].iloc[0]
        path=data/'snapshots'/f'{identity}__h{h:.3f}.csv'
        cached=snapshot_cache(path,fp,old.grid_sha256,design.parameter_id.tolist())
        if cached is not None:records.append(cached);continue
        n=int(old.n_clusters)
        if n>=2:
            a,d=R.geometry(grid,old,identity,geo,gfp)
            probabilities=np.empty((n,256));reports=[]
            for group_id,items in design.groupby('dispersion_group',sort=False):
                values,report=group(a,d,items,c)
                indices=items.index.to_numpy();probabilities[:,indices]=values
                reports.extend(report)
        else:
            probabilities=np.empty((0,256));reports=[dict(occupancy=np.nan,converged=False,n_iter=0,
                fixed_point_residual=np.nan,relative_rate_balance=np.nan,max_delta=np.nan,
                solver='not_applicable',status='excluded_fewer_than_two_patches',primary_tolerance_used=np.nan,
                independent_probability_difference=np.nan,independent_check_converged=False,independent_check_requested=False)]*256
        rows=[]
        for item,report in zip(design.itertuples(index=False),reports):
            rows.append(dict(parameter_id=item.parameter_id,dispersion_group=item.dispersion_group,
                log_r=item.log_r,log10_r=item.log10_r,r=item.r,half_distance=item.half_distance,alpha=item.alpha,
                scenario_id=scenario['scenario_id'],replicate=rep,seed=lc['base_seed']+rep-1,H_target=h,
                n_patches=n,grid_sha256=old.grid_sha256,q=0.,producer_fingerprint=fp,**report))
        table=pd.DataFrame(rows);tmp=path.with_suffix('.csv.tmp');table.to_csv(tmp,index=False);tmp.replace(path)
        tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,probabilities=probabilities.T.ravel(),offsets=np.arange(257,dtype=np.int64)*(n if n>=2 else 0))
        tmp.replace(path.with_suffix('.npz'))
        R.save_json(path.with_suffix('.json'),dict(producer_fingerprint=fp,grid_sha256=old.grid_sha256,
            hashes={s:R.sha(path.with_suffix(s)) for s in ['.csv','.npz']}))
        records.append(table);total_chunks+=1
        R.save_json(data/'progress'/f'{identity}.json',dict(last_H=h,completed_snapshots=len(records),total_snapshots=88,
            elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
        if len(records)%10==0:print(f'{identity}: {len(records)}/88, H={h:g}',flush=True)
    if pilot_H is not None:return identity,time.monotonic()-started,False
    full=pd.concat(records,ignore_index=True)
    assert len(full)==88*256 and full.status.isin(['valid','excluded_fewer_than_two_patches']).all()
    tmp=file.with_suffix('.csv.tmp');full.to_csv(tmp,index=False);tmp.replace(file)
    seconds=time.monotonic()-started
    R.save_json(marker,dict(producer_fingerprint=fp,sha256=R.sha(file),elapsed_seconds=seconds,
        n_rows=len(full),n_new_snapshots=total_chunks,complete=True))
    return identity,seconds,False
