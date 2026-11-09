"""Numerical workers for verified limit-based species parameter domains."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import sys,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
from src.metapop.domain_selection import normalized_colonization,scaled_ratio_sweep,scaled_residual,capacity_bracket,lower_occupancy_bracket,scientific_ratio


def source_fingerprint(c,geometry_fingerprint):
    digest=hashlib.sha256(json.dumps(c,sort_keys=True).encode()); digest.update(geometry_fingerprint.encode())
    for p in [Path(__file__),FINAL/'code/base/src/metapop/domain_selection.py',FINAL/'code/base/src/metapop/hanski.py']:
        digest.update(str(p.relative_to(FINAL)).encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def geometry(row,directory,fp):
    with np.load(Path(directory)/row.geometry_file) as z:
        if str(z['geometry_fingerprint'])!=fp or str(z['grid_sha256'])!=row.grid_sha256:
            raise RuntimeError('Stale geometry.')
        a=z['areas']; d=z['distances']
    if len(a)!=row.n_patches: raise AssertionError('Geometry patch count mismatch.')
    return a,d


def boundaries(frame,c,ell_bounds,geometry_directory,destination,fp,gfp):
    started=time.monotonic(); destination=Path(destination)
    rows=[]; vectors={}
    for _,row in frame[frame.n_patches.ge(2)].iterrows():
        a,d=geometry(row,geometry_directory,gfp)
        for q in c['model_formulations']:
            cap=capacity_bracket(a,d,ell_bounds[1],q=q,b=c['b'],x=c['x'],**c['capacity'])
            w,s=normalized_colonization(a,d,ell_bounds[0],q=q,b=c['b'],x=c['x'])
            report,low,high=lower_occupancy_bracket(w,s,target=c['lower_occupancy_target'],
                    log_tolerance=c['boundary_log_relative_tolerance'],**c['solver'])
            rows.append({**row.to_dict(),'q':q,'capacity':cap.value,'capacity_lower':cap.lower,
                'capacity_upper':cap.upper,'capacity_relative_residual':cap.relative_residual,
                'min_log_row_strength':float(s.min()),'max_log_row_strength':float(s.max()),
                'producer_fingerprint':fp,**report})
            key=f"h{row.H_target:.3f}_q{int(q)}"
            vectors[key+'_feasible']=low; vectors[key+'_infeasible']=high
    name=f"{frame.iloc[0].scenario_id}__rep{int(frame.iloc[0].replicate):03d}"
    pd.DataFrame(rows).to_csv(destination/f'{name}.csv',index=False)
    np.savez_compressed(destination/f'{name}.npz',**vectors)
    return len(rows),time.monotonic()-started


def responses(frame,c,domain,geometry_directory,destination,fp,gfp):
    started=time.monotonic(); destination=Path(destination); q=domain['q']
    rows=[]; vectors=[]; offsets=[0]; lower_validated=[]
    grid=domain['grid']; profiles=domain['profiles']
    groups=[]
    for ell_id,ell in enumerate(domain['half_distance_values']):
        items=[('pilot',item) for item in grid if item['ell_id']==ell_id]
        items += [('profiles',item) for item in profiles if item['ell_id']==ell_id]
        groups.append((ell,items))
    for _,land in frame.iterrows():
        a,d=geometry(land,geometry_directory,gfp); n=len(a)
        for ell,items in groups:
            logs=[item['log_r'] for _,item in items]
            if n>=2:
                w,s=normalized_colonization(a,d,ell,q=q,b=c['b'],x=c['x'])
                results=scaled_ratio_sweep(w,s,logs,max_iter=c['solver']['max_iter'],tol=c['solver']['tol'])
            else: results=[None]*len(items)
            for (phase,item),result in zip(items,results):
                retried=False
                if result is not None and not result.converged:
                    retried=True
                    result=scaled_ratio_sweep(w,s,[item['log_r']],max_iter=c['solver']['retry_max_iter'],tol=c['solver']['tol'])[0]
                p=result.probabilities if result else np.empty(0)
                lr=item['log_r']
                ratio=np.exp(lr) if np.log(np.finfo(float).tiny)<lr<np.log(np.finfo(float).max) else np.nan
                rows.append({**land.to_dict(),'q':q,'phase':phase,'parameter_id':item['parameter_id'],
                    'log_r':lr,'log10_r':lr/np.log(10),'r':scientific_ratio(lr),'r_float':ratio,
                    'half_distance':ell,'alpha':np.log(2)/ell,
                    'occupancy':result.occupancy_fraction if result and result.converged else np.nan,
                    'last_iterate_mean':result.occupancy_fraction if result else np.nan,
                    'converged':bool(result.converged) if result else False,
                    'n_iter':result.n_iter if result else 0,'max_delta':result.max_delta if result else np.nan,
                    'fixed_point_residual':scaled_residual(w,s,lr,p) if result else np.nan,
                    'retried':retried,'solver':'algebraic_row_scaling',
                    'status':'excluded_fewer_than_two_patches' if n<2 else ('valid' if result.converged else 'nonconverged'),
                    'producer_fingerprint':fp})
                vectors.append(p); offsets.append(offsets[-1]+len(p))
    name=f"q{int(q)}__{frame.iloc[0].scenario_id}__rep{int(frame.iloc[0].replicate):03d}"
    table=pd.DataFrame(rows)
    table.to_csv(destination/f'{name}.csv',index=False)
    np.savez_compressed(destination/f'{name}.npz',probabilities=np.concatenate(vectors),offsets=np.asarray(offsets))
    return len(rows),int(table.status.eq('nonconverged').sum()),time.monotonic()-started
