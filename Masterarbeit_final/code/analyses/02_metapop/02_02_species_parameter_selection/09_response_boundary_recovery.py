"""Resume unchanged boundary design with diagnosed, declared precision repairs."""
from __future__ import annotations
import argparse
import fcntl
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
FOLDER=Path(__file__).parent
sys.path.insert(0,str(FOLDER))
R=importlib.import_module('09_response_boundary_refinement')
np,pd=R.np,R.pd
original_worker=R.worker


def guarded_worker(qid,ell,lr,scenario,rep,hs,base,lc,benchmark,data_name,fp):
    data=Path(data_name);identity=f"{scenario['scenario_id']}__rep{rep:03d}"
    file=data/'responses'/qid/f'{identity}.csv';marker=file.with_suffix('.json')
    helper_sha=R.sha(Path(__file__))
    if marker.exists():
        meta=json.loads(marker.read_text())
        if 'precision_source_sha256' in meta:
            assert meta['precision_source_sha256']==helper_sha
            assert (meta['declared_query'],meta['declared_trajectory'])==(qid,identity)
    try:
        return original_worker(qid,ell,lr,scenario,rep,hs,base,lc,benchmark,data_name,fp)
    except AssertionError:
        # Never overwrite an already completed and checked trajectory.
        if marker.exists():raise
        gs,levels=R.grids(scenario,rep,base,lc,benchmark,data,fp)
        item=SimpleNamespace(parameter_id=qid,half_distance=ell,alpha=np.log(2)/ell,log_r=lr)
        control={**base,'scalar_check_ids':[]};diagnosis=None
        for h in hs:
            land=benchmark[benchmark.H_target.eq(h)].iloc[0]
            if land.n_clusters<2:continue
            grid=gs[np.flatnonzero(levels==h)[0]];a,d=R.geometry(grid,land,identity,data,fp)
            if len(a)>500:continue
            p,report=R.W.solve(a,d,item,control)
            w,s=R.W.normalized_colonization(a,d,ell,q=0,b=1,x=.5)
            other=R.original_scaled(w,s,[lr],tol=1e-12,max_iter=1000000)[0]
            if not other.converged:continue
            diff=float(np.max(np.abs(p-other.probabilities)))
            if diff<=base['solver']['independent_probability_tolerance']:continue
            tight={**control,'solver':{**base['solver'],'tol':1e-12}}
            pt,rt=R.W.solve(a,d,item,tight)
            ref=R.original_scaled(w,s,[lr],tol=1e-13,max_iter=1000000)[0]
            if not ref.converged:raise RuntimeError('Precision diagnosis lacks a converged reference')
            new_diff=float(np.max(np.abs(pt-ref.probabilities)))
            if new_diff>base['solver']['independent_probability_tolerance']:
                raise RuntimeError('Tighter primary precision did not resolve the control discrepancy')
            diagnosis=dict(query_id=qid,trajectory=identity,H=float(h),n_patches=len(a),
                old_probability_difference=diff,tighter_probability_difference=new_diff,
                old_occupancy=float(p.mean()),tighter_occupancy=float(pt.mean()),
                primary_vector_change=float(np.max(np.abs(pt-p))),old_residual=report['fixed_point_residual'],
                tighter_residual=rt['fixed_point_residual'],primary_tol_used=1e-12,
                tighter_control_tol=1e-13,unchanged_probability_tolerance=2e-7,
                precision_source_sha256=helper_sha,baseline_producer_fingerprint=fp)
            break
        if diagnosis is None:raise RuntimeError('No independent-primary-precision cause found; recovery refused')
        result_folder=R.FINAL/'results/02_metapop/02_02_species_parameter_selection/response_boundary_refinement/logs/precision_repairs'
        result_folder.mkdir(exist_ok=True)
        R.save_json(result_folder/f'{qid}__{identity}__diagnosis.json',diagnosis)
        options={**base,'solver':{**base['solver'],'tol':1e-12}}
        table,seconds=original_worker(qid,ell,lr,scenario,rep,hs,options,lc,benchmark,data_name,fp)
        table['primary_tolerance_used']=np.where(table.status.eq('valid'),1e-12,np.nan)
        tmp=file.with_suffix('.csv.tmp');table.to_csv(tmp,index=False);tmp.replace(file)
        meta=json.loads(marker.read_text())
        meta.update(precision_source_sha256=helper_sha,declared_query=qid,declared_trajectory=identity,
            primary_tolerance_used=1e-12,precision_diagnosis=diagnosis)
        meta['hashes']['.csv']=R.sha(file);R.save_json(marker,meta)
        R.save_json(result_folder/f'{qid}__{identity}__receipt.json',dict(**diagnosis,
            rows=len(table),valid_rows=int(table.status.eq('valid').sum()),
            files_sha256={str(file.with_suffix(s).relative_to(R.FINAL)):R.sha(file.with_suffix(s)) for s in ['.csv','.npz','.json']}))
        print(f'Precision-repaired {qid} {identity}: {diff:.3g} -> {new_diff:.3g}; model/seed/design unchanged.',flush=True)
        return table,seconds


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    ctx=R.setup(R.DEFAULT);data,results=ctx[-2:]
    R.save_json(results/'logs/09_recovery_controller.json',dict(status='running',
        precision_source_sha256=R.sha(Path(__file__)),original_selection_source_sha256=R.sha(Path(R.__file__)),
        policy='Only uncompleted trajectories with demonstrated independent-primary precision discrepancy; tighten primary tolerance to 1e-12 and preserve all valid cached data.',
        probability_tolerance_unchanged=2e-7,original_producer_fingerprint=ctx[-3]))
    R.worker=guarded_worker
    with (data/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        R.compute(ctx,min(args.workers,ctx[0]['maximum_workers']))


if __name__=='__main__':main()
