"""Compute the unchanged 128-point main LHS on matched Result A landscapes."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-main-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
FOLDER=Path(__file__).parent
sys.path.insert(0,str(FOLDER));sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
WORKER=importlib.import_module('11_lhs_workers')
REL=Path('02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds')
CONFIG=FINAL/'config'/REL/'11_lhs_occupancy.json'
DATA=FINAL/'data/processed'/REL
RESULTS=FINAL/'results'/REL


def inputs():
    c=json.loads(CONFIG.read_text())
    lc=json.loads((FINAL/c['landscape_config']).read_text())
    lhs=pd.read_csv(FINAL/c['lhs_design'])
    assert len(lhs)==128 and lhs.parameter_id.nunique()==128 and lhs.q.eq(0).all()
    assert c['h_targets']==lc['h_targets'] and c['replicates']==list(range(1,21))
    assert (c['b'],c['x'],c['q'],c['initial_occupancy'])==(1.,.5,0.,1.)
    assert (lc['generation_boundary'],lc['analysis_boundary'],lc['distance_mode'],lc['voronoi_depth'])==('periodic','open','voronoi_dt',1)
    for col in ['unit_log_r','unit_log_half_distance']:
        assert np.array_equal(np.sort(np.floor(lhs[col]*128).astype(int)),np.arange(128))
    # The display r column was exported with 12 significant digits; log_r is
    # the original full-precision numerical design coordinate.
    assert np.allclose(lhs.r,np.exp(lhs.log_r),atol=0,rtol=5e-12)
    assert np.allclose(lhs.alpha,np.log(2)/lhs.half_distance,atol=0,rtol=1e-13)
    assert lhs.half_distance.between(1,512).all()
    assert lhs.log10_r.between(-2.5,5.75).all()
    assert np.allclose(lhs.log10_r,-2.5+lhs.unit_log_r*8.25)
    assert np.allclose(np.log(lhs.half_distance),lhs.unit_log_half_distance*np.log(512))
    benchmark=pd.read_csv(FINAL/c['landscape_data'])
    benchmark=benchmark[benchmark.replicate.isin(c['replicates']) & benchmark.H_target.isin(c['h_targets'])]
    assert len(benchmark)==7*20*len(c['h_targets'])
    assert not benchmark.duplicated(['scenario_id','replicate','H_target']).any()
    assert benchmark.H_real.eq(benchmark.H_target).all()
    assert (benchmark.n_random_cumulative+benchmark.n_directed_cumulative).eq(benchmark.event_count).all()
    assert set(benchmark.scenario_id)=={s['scenario_id'] for s in lc['scenarios']}
    paths=[CONFIG,FINAL/c['landscape_config'],FINAL/c['landscape_data'],FINAL/c['lhs_design'],
           Path(__file__),FOLDER/'11_lhs_workers.py']
    paths += [FINAL/'code/base/src/landscape'/f'{n}.py' for n in
              ('generation','deforestation','landscape','boundary','complete_edge_distances')]
    paths += [FINAL/'code/base/src/metapop'/f'{n}.py' for n in ('hanski','domain_selection')]
    hashes={str(p.relative_to(FINAL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fp=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return c,lc,lhs,benchmark,hashes,fp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','compute','summary','all'],default='all')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--max-trajectories',type=int,default=0,help='Development limit; no complete-run claims.')
    args=parser.parse_args()
    for d in (DATA/'snapshots',DATA/'trajectories',DATA/'progress',RESULTS/'tables',RESULTS/'logs'):
        d.mkdir(parents=True,exist_ok=True)
    c,lc,lhs,benchmark,hashes,fp=inputs()
    manifest=dict(configuration=c,source_hashes=hashes,producer_fingerprint=fp)
    path=DATA/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Inputs changed; refusing to reuse this main-run directory.')
    WORKER.atomic_json(path,manifest)
    counts=dict(n_parameters=128,n_scenarios=7,n_seed_blocks=20,n_h_levels=len(c['h_targets']),
        n_landscape_snapshots=len(benchmark),n_expected_response_rows=len(benchmark)*128,
        n_excluded_landscapes=int(benchmark.n_clusters.lt(2).sum()),
        n_expected_valid_response_rows=int(benchmark.n_clusters.ge(2).sum())*128,
        lhs_parameter_bounds={k:{'min':float(lhs[k].min()),'max':float(lhs[k].max())} for k in ['r','half_distance','alpha']},
        lhs_strata_verified=True,old_geometry_cache_reused=False,
        source_hashes=hashes,producer_fingerprint=fp)
    WORKER.atomic_json(RESULTS/'logs/11_preflight.json',counts)
    print(f'Preflight: {len(benchmark)} matched landscapes, {len(c["h_targets"])} H levels, {len(benchmark)*128} expected responses.',flush=True)
    if args.stage=='preflight': return
    if args.stage in ['compute','all']:
        jobs=[]
        for rep in c['replicates']:
            for scenario in lc['scenarios']:
                frame=benchmark[benchmark.scenario_id.eq(scenario['scenario_id']) & benchmark.replicate.eq(rep)]
                jobs.append((scenario,rep,c,lc,frame,lhs,str(DATA),fp))
        if args.max_trajectories: jobs=jobs[:args.max_trajectories]
        WORKER.atomic_json(RESULTS/'logs/11_run_status.json',dict(status='running',requested_trajectories=len(jobs),
            expected_trajectories=140,producer_fingerprint=fp))
        failed=0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(WORKER.trajectory,*job):job for job in jobs}
            for i,future in enumerate(as_completed(futures),1):
                identity,n,nonconverged,seconds=future.result();failed+=nonconverged
                print(f'Completed {i}/{len(jobs)}: {identity}, {n} responses, {nonconverged} unconverged, {seconds:.1f}s.',flush=True)
                WORKER.atomic_json(RESULTS/'logs/11_run_status.json',dict(status='running',completed_this_invocation=i,
                    requested_trajectories=len(jobs),expected_trajectories=140,producer_fingerprint=fp,
                    nonconverged_this_invocation=failed))
        if args.max_trajectories:
            print('Development-limited run; summary withheld.',flush=True)
            return
    if args.stage in ['summary','all']:
        summary=importlib.import_module('11_lhs_summary')
        summary.finish(c,DATA,RESULTS,fp)


if __name__=='__main__': main()
