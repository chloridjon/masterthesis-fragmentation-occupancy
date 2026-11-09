"""Compute fixed conditional-design profiles on complete matched seed prefixes."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
V=importlib.import_module('11_confirmed_domain_worker');R=V.R
DEFAULT=FINAL/'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/confirmed_domain_256/11_confirmed_domain_occupancy.json'


def inputs(path=DEFAULT):
    c=json.loads(path.read_text());base=json.loads((FINAL/c['baseline_configuration']).read_text())
    lc=json.loads((FINAL/base['landscape_config']).read_text())
    design=pd.read_csv(FINAL/c['parameter_design'],float_precision='round_trip')
    verification=json.loads((FINAL/c['design_preflight']).read_text())
    assert verification['status']=='design_verified'
    assert R.sha(FINAL/c['parameter_design'])==verification['output_sha256'][c['parameter_design']]
    assert len(design)==256 and design.parameter_id.nunique()==256
    assert design.groupby('dispersion_group').size().eq(4).all() and design.half_distance.nunique()==64
    assert (c['q'],c['b'],c['x'])==(0,1,.5)
    np.testing.assert_allclose(design.r,np.exp(design.log_r),rtol=1e-14)
    np.testing.assert_allclose(design.alpha,np.log(2)/design.half_distance,rtol=1e-14)
    benchmark=pd.read_csv(FINAL/base['landscape_data'])
    benchmark=benchmark[benchmark.replicate.isin(base['replicates'])&benchmark.H_target.isin(base['h_targets'])]
    assert len(benchmark)==12320 and len(base['h_targets'])==88
    geo=FINAL/c['geometry_cache'];gm=json.loads((geo/'manifest.json').read_text());gfp=gm['producer_fingerprint']
    for name,digest in gm['source_sha256'].items():assert R.sha(FINAL/name)==digest,name
    sources=[path,Path(__file__),Path(V.__file__),Path(R.__file__),Path(V.W.__file__),
        FINAL/c['parameter_design'],FINAL/c['design_preflight'],FINAL/c['baseline_configuration'],
        FINAL/base['landscape_config'],FINAL/base['landscape_data'],geo/'manifest.json']
    sources+=[FINAL/'code/base/src/metapop'/f'{n}.py' for n in ['hanski','domain_selection','parameter_sweep','verified_batch']]
    hashes={str(p.relative_to(FINAL)):R.sha(p) for p in sources}
    fp=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    data,results=FINAL/c['data_directory'],FINAL/c['results_directory']
    for p in [data/'snapshots',data/'trajectories',data/'progress',data/'blocks',results/'logs',results/'tables',results/'figures']:p.mkdir(parents=True,exist_ok=True)
    manifest=dict(configuration=c,source_sha256=hashes,producer_fingerprint=fp,geometry_provider_fingerprint=gfp,
        numpy_version=np.__version__,baseline_replicates=base['replicates'],H_targets=base['h_targets'])
    p=data/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Changed main producer; cache reuse refused'
    else:R.save_json(p,manifest)
    R.save_json(results/'logs/11_preflight.json',dict(status='passed',n_parameters=256,n_dispersion_groups=64,
        n_H=88,n_scenarios=7,initial_seed_blocks=5,maximum_seed_blocks=20,
        response_rows_per_seed_block=256*7*88,producer_fingerprint=fp,source_sha256=hashes))
    return c,base,lc,design,benchmark,geo,gfp,data,results,fp


def deadline_worker(deadline,*args):
    if deadline:
        import time
        seconds=deadline-time.time()
        if seconds<=0:raise TimeoutError('Optional-extension deadline reached before starting trajectory')
        def alarm(signum,frame):raise TimeoutError('Optional-extension compute deadline reached')
        signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(seconds)))
    try:return V.trajectory(*args)
    finally:
        if deadline:signal.alarm(0)


def matched_prefix(c,lc,data,fp):
    done=[]
    for rep in range(1,c['n_maximum_seed_blocks']+1):
        found=[]
        for scenario in lc['scenarios']:
            p=data/'trajectories'/f"{scenario['scenario_id']}__rep{rep:03d}.json"
            if not p.exists():break
            meta=json.loads(p.read_text());assert meta['producer_fingerprint']==fp and meta['complete']
            csv=p.with_suffix('.csv');assert R.sha(csv)==meta['sha256'];found.append(dict(scenario_id=scenario['scenario_id'],sha256=meta['sha256']))
        if len(found)!=7:break
        R.save_json(data/'blocks'/f'rep{rep:03d}.json',dict(replicate=rep,producer_fingerprint=fp,trajectories=found,complete=True))
        done.append(rep)
    return len(done)


def run(ctx,n_seeds,workers,deadline=None,pilot=False):
    c,base,lc,design,benchmark,geo,gfp,data,results,fp=ctx
    assert 1<=n_seeds<=20
    if pilot:
        scenario=next(s for s in lc['scenarios'] if s['scenario_id']=='only_random_rho0')
        frame=benchmark[benchmark.scenario_id.eq(scenario['scenario_id'])&benchmark.replicate.eq(1)]
        identity,seconds,_=V.trajectory(scenario,1,c,base,lc,frame,design,str(data),str(geo),fp,gfp,pilot_H=.325)
        R.save_json(results/'logs/11_real_geometry_pilot.json',dict(status='computed_and_checked',trajectory=identity,
            H=.325,n_parameters=256,elapsed_seconds=seconds,producer_fingerprint=fp))
        return
    jobs=[]
    # Prioritise the expensive baseline trajectories while other scenario
    # jobs fill remaining workers; adoption still requires full matched blocks.
    for scenario in sorted(lc['scenarios'],key=lambda s:s['scenario_id']!='only_random_rho0'):
        for rep in range(1,n_seeds+1):
            frame=benchmark[benchmark.scenario_id.eq(scenario['scenario_id'])&benchmark.replicate.eq(rep)]
            jobs.append((scenario,rep,c,base,lc,frame,design,str(data),str(geo),fp,gfp))
    R.save_json(results/'logs/11_run_status.json',dict(status='running',requested_seed_prefix=n_seeds,
        requested_trajectories=len(jobs),producer_fingerprint=fp,started_at_utc=datetime.now(timezone.utc).isoformat(),
        optional_deadline_epoch=deadline))
    errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(deadline_worker,deadline,*job):job for job in jobs}
        for count,f in enumerate(as_completed(futures),1):
            try:
                identity,seconds,cached=f.result()
                print(f'Completed {count}/{len(jobs)}: {identity}, {seconds:.1f}s, cached={cached}',flush=True)
            except Exception as e:
                errors.append(dict(type=type(e).__name__,message=str(e)));print(f'Checked failure: {type(e).__name__}: {e}',flush=True)
            prefix=matched_prefix(c,lc,data,fp)
            R.save_json(results/'logs/11_run_status.json',dict(status='running',completed_futures=count,
                requested_trajectories=len(jobs),requested_seed_prefix=n_seeds,complete_matched_prefix=prefix,
                producer_fingerprint=fp,errors=errors,optional_deadline_epoch=deadline))
    prefix=matched_prefix(c,lc,data,fp)
    status='complete_requested_prefix' if prefix>=n_seeds else ('optional_deadline_prefix_preserved' if deadline else 'incomplete_checked_failure')
    R.save_json(results/'logs/11_run_status.json',dict(status=status,requested_seed_prefix=n_seeds,
        complete_matched_prefix=prefix,producer_fingerprint=fp,errors=errors))
    if prefix<n_seeds and not deadline:raise RuntimeError('Required initial prefix incomplete; valid checkpoints retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['preflight','pilot','compute'],default='compute')
    parser.add_argument('--n-seeds',type=int,default=5);parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--optional-deadline-utc');args=parser.parse_args();ctx=inputs();data=ctx[-3]
    with (data/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='preflight':print('Confirmed-domain compute preflight passed.',flush=True)
        else:
            deadline=datetime.fromisoformat(args.optional_deadline_utc).timestamp() if args.optional_deadline_utc else None
            run(ctx,args.n_seeds,args.workers,deadline,pilot=args.stage=='pilot')
