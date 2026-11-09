"""Select an illustrative reference from the checked design and verify 20 seeds."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,hashlib,importlib,json,os,sys,time
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(FINAL/'code/base'))
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_02_species_parameter_selection'))
sys.path.insert(0,str(HERE.parent/'02_03_01_scenario_occupancy'))
import numpy as np
import pandas as pd
M=importlib.import_module('11_confirmed_domain_compute');V=M.V;R=V.R
S=importlib.import_module('11_confirmed_domain_summary')
WIDTH=importlib.import_module('09_reference_response_diagnostics').intermediate_width
BOOT=importlib.import_module('10_result_b_bootstrap')
CONFIG=FINAL/'config/02_metapop/02_03_main_analyses/02_03_01_scenario_occupancy/reference_conditional_256/10_reference_update.json'


def describe(values):
    return S.statistics(values[:,:,:,None])[1][:,:,:,0]


def criteria(values,hs,design):
    n=np.isfinite(values[:,:,:,0]).sum(axis=0);keep=(n>=3).all(axis=0)
    q=S.statistics(values)[1];h=hs[keep];median=q[1][:,keep]
    rows=[]
    for i,item in design.iterrows():
        a=median[:,:,i]
        u=(item.log10_r+2.5)/8.25;w=np.log(item.half_distance)/np.log(512)
        rows.append(dict(parameter_id=item.parameter_id,r=item.r,log_r=item.log_r,half_distance=item.half_distance,
            minimum_scenario_occupancy_span=float(np.ptp(a,axis=1).min()),
            minimum_intermediate_H_width=float(min(WIDTH(h,p) for p in a)),
            mean_pairwise_scenario_separation=float(np.mean([np.trapezoid(abs(a[j]-a[k]),h)/(h[-1]-h[0]) for j in range(7) for k in range(j+1,7)])),
            distance_to_log_midpoint=float(np.hypot(u-.5,w-.5)),interior=bool(0<u<1 and 0<w<1),
            all_scenarios_reach_10_and_90_percent=bool((a.min(axis=1)<=.1).all() and (a.max(axis=1)>=.9).all())))
    out=pd.DataFrame(rows);cols=['minimum_scenario_occupancy_span','minimum_intermediate_H_width','mean_pairwise_scenario_separation']
    scores=out[cols].to_numpy();out['response_pareto']=np.array([not ((scores>=x).all(axis=1)&(scores>x).any(axis=1)).any() for x in scores])
    return out,h


def setup():
    c=json.loads(CONFIG.read_text());ctx=M.inputs();cc,base,lc,design,bench,geo,gfp,source,_,sourcefp=ctx
    source_out=FINAL/c['source_snapshot'];audit=json.loads((source_out/'11_numerical_audit.json').read_text())
    assert audit['visual_review_confirmed'] and audit['n_seed_blocks']==5
    for p,h in {**audit['input_sha256'],**audit['output_sha256']}.items():assert R.sha(FINAL/p)==h,p
    with np.load(source_out/'tables/11_response_arrays.npz') as z:values=z['values'].copy()
    hs=np.array(base['h_targets']);candidate,h=criteria(values,hs,design)
    data=FINAL/c['data_directory'];out=FINAL/c['results_directory']
    for p in [data/'snapshots',data/'trajectories',out/'tables',out/'logs',out/'figures']:p.mkdir(parents=True,exist_ok=True)
    selected=candidate[candidate.parameter_id.eq(c['selected_parameter_id'])].iloc[0]
    assert selected.interior and selected.response_pareto and selected.all_scenarios_reach_10_and_90_percent
    candidate.to_csv(out/'tables/10_reference_candidate_tradeoffs.csv',index=False)
    checks=[]
    for omit in range(5):
        t,_=criteria(np.delete(values,omit,axis=0),hs,design)
        checks.append(dict(omitted_seed=omit+1,**t[t.parameter_id.eq(c['selected_parameter_id'])].iloc[0].to_dict()))
    pd.DataFrame(checks).to_csv(out/'tables/10_selection_seed_omission.csv',index=False)
    item=design[design.parameter_id.eq(c['selected_parameter_id'])].copy()
    worker={**cc,'independent_checks':{**cc['independent_checks'],'parameter_ids':[c['selected_parameter_id']]}}
    sources={str(p.relative_to(FINAL)):R.sha(p) for p in [CONFIG,Path(__file__),Path(V.__file__),Path(R.__file__),Path(V.W.__file__),source/'manifest.json',source_out/'11_numerical_audit.json']}
    sources.update(json.loads((source/'manifest.json').read_text())['source_sha256'])
    fp=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    manifest=dict(configuration=c,selected=selected.to_dict(),source_sha256=sources,producer_fingerprint=fp,source_design_fingerprint=sourcefp)
    path=data/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:R.save_json(path,manifest)
    return c,worker,base,lc,item,bench,geo,gfp,data,out,fp,source,sourcefp


def trajectory(scenario,rep,ctx):
    c,worker,base,lc,item,bench,geo,gfp,data,out,fp,source,sourcefp=ctx
    sid=scenario['scenario_id'];identity=f'{sid}__rep{rep:03d}';full=data/'trajectories'/f'{identity}.csv';start=time.monotonic()
    if full.with_suffix('.json').exists():
        receipt=json.loads(full.with_suffix('.json').read_text());assert receipt['producer_fingerprint']==fp and R.sha(full)==receipt['sha256'];return identity
    frame=bench[bench.scenario_id.eq(sid)&bench.replicate.eq(rep)]
    grids,hs=R.grids(scenario,rep,base,lc,frame,geo,gfp);records=[];idx=int(item.index[0]);item=item.reset_index(drop=True)
    for grid,h in zip(grids,hs):
        h=float(h);land=frame[frame.H_target.eq(h)].iloc[0];path=data/'snapshots'/f'{identity}__h{h:.3f}.csv';meta=path.with_suffix('.json')
        if meta.exists():
            r=json.loads(meta.read_text());assert r['producer_fingerprint']==fp
            for suffix,digest in r['hashes'].items():assert R.sha(path.with_suffix(suffix))==digest
            records.append(pd.read_csv(path,float_precision='round_trip'));continue
        grid_hash=hashlib.sha256(grid.tobytes()).hexdigest();assert grid_hash==land.grid_sha256;n=int(land.n_clusters)
        if rep<=5:
            old=source/'snapshots'/path.name;r=json.loads(old.with_suffix('.json').read_text());assert r['producer_fingerprint']==sourcefp
            for suffix,digest in r['hashes'].items():assert R.sha(old.with_suffix(suffix))==digest
            t=pd.read_csv(old,float_precision='round_trip').iloc[[idx]].copy()
            assert t.parameter_id.iloc[0]==c['selected_parameter_id']
            with np.load(old.with_suffix('.npz')) as z:p=z['probabilities'][z['offsets'][idx]:z['offsets'][idx+1]].copy()
            t['source_producer_fingerprint']=sourcefp;t['producer_fingerprint']=fp;t['reuse_source']='exact_checked_256_profile_snapshot'
        else:
            if n>=2:
                a,d=R.geometry(grid,land,identity,geo,gfp);p,reports=V.group(a,d,item,worker);p=p[:,0];report=reports[0]
            else:p=np.empty(0);report=dict(status='excluded_fewer_than_two_patches',occupancy=np.nan,converged=False,fixed_point_residual=np.nan,primary_tolerance_used=np.nan)
            row=item.iloc[0].to_dict();row.update(scenario_id=sid,replicate=rep,seed=lc['base_seed']+rep-1,H_target=h,n_patches=n,grid_sha256=grid_hash,producer_fingerprint=fp,reuse_source='computed_selected_profile',**report);t=pd.DataFrame([row])
        assert len(p)==(n if n>=2 else 0)
        if n>=2:
            a,d=R.geometry(grid,land,identity,geo,gfp);k=np.exp(-float(item.alpha.iloc[0])*d);np.fill_diagonal(k,0.)
            incoming=k@(a*p);target=incoming/(incoming+float(item.r.iloc[0])/np.sqrt(a));res=float(abs(target-p).max())
            assert res<=1.1*float(t.primary_tolerance_used.iloc[0]) and abs(p.mean()-float(t.occupancy.iloc[0]))<1e-14
        t.to_csv(path,index=False)
        np.savez_compressed(path.with_suffix('.npz'),probabilities=p)
        R.save_json(meta,dict(producer_fingerprint=fp,hashes={s:R.sha(path.with_suffix(s)) for s in ['.csv','.npz']}))
        records.append(t)
    pd.concat(records,ignore_index=True).to_csv(full,index=False)
    R.save_json(full.with_suffix('.json'),dict(producer_fingerprint=fp,sha256=R.sha(full),elapsed_seconds=time.monotonic()-start,complete=True))
    return identity


def compute(ctx):
    c,worker,base,lc,item,bench,geo,gfp,data,out,fp,source,sourcefp=ctx
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        jobs=[pool.submit(trajectory,s,rep,ctx) for rep in range(1,c['n_reference_seeds']+1) for s in lc['scenarios']]
        for i,f in enumerate(as_completed(jobs),1):
            identity=f.result();R.save_json(out/'logs/10_compute_status.json',dict(status='computing',completed_trajectories=i,total_trajectories=len(jobs),last=identity));print(f'Complete {i}/{len(jobs)}: {identity}',flush=True)
    R.save_json(out/'logs/10_compute_status.json',dict(status='compute_complete_summary_pending',completed_trajectories=len(jobs),total_trajectories=len(jobs)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['preflight','compute'],default='compute');a=p.parse_args();ctx=setup()
    print(json.dumps(dict(selected=ctx[4].iloc[0].to_dict(),n_reference_seeds=ctx[0]['n_reference_seeds']),indent=2),flush=True)
    if a.stage=='compute':compute(ctx)
