"""Supplement the unchanged pilot producer at exactly critical local self rates.

Existing complete snapshots are hash-checked and retained. Missing snapshots
use the same q=1 model, parameters, geometry and residual tolerance, with an
analytic local-quadratic solver. This supplemental source is recorded separately
from the original producer fingerprint; no original source hash is rewritten.
"""
from pathlib import Path
import importlib.util
import json
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('self_pilot',HERE/'13_self_colonization_pilot.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
from src.metapop.self_equilibrium import self_colonization_sweep
from src.metapop.domain_selection import normalized_colonization,scaled_ratio_sweep


def recovered_group(a,d,items,c):
    ell=float(items.half_distance.iloc[0]);k=np.exp(-np.log(2)/ell*d);np.fill_diagonal(k,0)
    answers=self_colonization_sweep(a,k,items.r.to_numpy(),tol=1e-13,max_iter=c['continuation_max_iter'])
    p=np.stack([r.probabilities for r in answers],axis=1)
    source=k@(a[:,None]*p)+a[:,None]*p
    rates=a[:,None]**(-.5)*items.r.to_numpy();residual=np.max(np.abs(source/(source+rates)-p),axis=0)
    reports=[]
    # Independent raw iteration is retained where it actually converges;
    # exact local critical cases are checked analytically and by original residual.
    independent=None
    if len(a)<=200:
        w,s=normalized_colonization(a,d,ell,q=1)
        independent=scaled_ratio_sweep(w,s,np.log(items.r),tol=1e-13,max_iter=20000)
    for j,result in enumerate(answers):
        assert result.converged and residual[j]<=1.1*c['tol']
        difference=None
        if independent is not None and independent[j].converged:
            difference=float(np.max(np.abs(result.probabilities-independent[j].probabilities)))
            assert difference<2e-7
        reports.append(dict(converged=True,n_iter=result.n_iter,max_delta=result.max_delta,
            fixed_point_residual=float(residual[j]),solver='analytic_local_quadratic_maximal_iteration',
            independent_difference=difference,independent_raw_converged=None if independent is None else independent[j].converged))
    return p,reports


def main():
    c,design,ext,land,out,data,fp=P.inputs(P.DEFAULT)
    sources=[Path(__file__),P.FINAL/'code/base/src/metapop/self_equilibrium.py']
    receipt=dict(original_fingerprint=fp,supplemental_sources={str(p.relative_to(P.FINAL)):P.sha(p) for p in sources},
        purpose='Recover missing snapshots at exact local critical self-colonization rates; same equation and tolerance.',trajectories=[])
    before={p.name:P.sha(p) for p in (data/'snapshots').iterdir()}
    P.solve_group=recovered_group
    for scenario in land.scenario_id.unique():
        for rep in c['replicates']:
            identity=f'{scenario}__rep{rep:03d}'
            if (data/f'{identity}.json').exists():continue
            P.task(scenario,rep,c,design,ext,land,str(data),fp)
            receipt['trajectories'].append(identity)
    assert all(P.sha(data/'snapshots'/name)==digest for name,digest in before.items())
    receipt['retained_snapshot_files']=len(before)
    receipt['new_snapshot_hashes']={p.name:P.sha(p) for p in (data/'snapshots').iterdir() if p.name not in before}
    P.save(out/'logs/13_critical_recovery.json',receipt)
    P.summaries(c,design,land,out,data)

if __name__=='__main__':main()
