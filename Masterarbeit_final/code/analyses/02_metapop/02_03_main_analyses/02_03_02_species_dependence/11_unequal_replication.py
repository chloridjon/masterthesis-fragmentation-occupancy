"""Extend the fixed design to 8/16 seeds using unchanged, snapshot-level producers."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,fcntl,hashlib,importlib,json,os,subprocess,sys,time
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final');ROOT=FINAL.parent
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
M=importlib.import_module('11_confirmed_domain_compute');V=M.V;R=V.R
CONFIG=FINAL/'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/confirmed_domain_256/11_replication_8_16.json'
ALIAS='final-confirmed-domain-unequal'
_CTX=None
_SAVE_JSON=R.save_json

def now():return datetime.now(timezone.utc).isoformat()
def targets(scenarios):return {s['scenario_id']:(8 if s['scenario_id']=='only_random_rho0' else 16) for s in scenarios}
def expected_jobs(scenarios,hs):return [(sid,rep,float(h)) for sid,n in targets(scenarios).items() for rep in range(1,n+1) for h in hs]
def planned_mask(scenarios):return np.array([[rep<=targets(scenarios)[s['scenario_id']] for s in scenarios] for rep in range(1,17)],bool)
def timing_bounds(data,scenarios):
    costs=defaultdict(list);full=defaultdict(list)
    for s in scenarios:
        sid=s['scenario_id']
        for rep in range(1,6):
            meta=data/'trajectories'/f'{sid}__rep{rep:03d}.json';a=json.loads(meta.read_text());full[sid].append(a['elapsed_seconds'])
            start=meta.stat().st_mtime-a['elapsed_seconds'];previous=start
            for p in sorted((data/'snapshots').glob(f'{sid}__rep{rep:03d}__h*.json'),key=lambda p:float(p.stem.rsplit('__h',1)[1]),reverse=True):
                stamp=p.stat().st_mtime;h=float(p.stem.rsplit('__h',1)[1])
                # A pre-existing pilot snapshot is not a measurement of this invocation.
                if stamp<start or stamp<previous:continue
                elapsed=stamp-previous;previous=stamp
                if elapsed>0:costs[(sid,h)].append(elapsed)
    return {k:float(np.quantile(v,.75)) for k,v in costs.items()},{k:dict(median_seconds=float(np.median(v)),maximum_seconds=float(max(v))) for k,v in full.items()}

def setup():
    plan=json.loads(CONFIG.read_text());ctx=M.inputs();c,base,lc,design,bench,geo,gfp,data,oldout,fp=ctx
    assert plan['scenario_seed_targets']==targets(lc['scenarios']) and plan['n_parameters']==256
    assert plan['scientific_producer_fingerprint']==fp
    out=FINAL/plan['results_directory'];out.mkdir(parents=True,exist_ok=True);(out/'logs/task_receipts').mkdir(parents=True,exist_ok=True)
    audit_path=FINAL/plan['retained_five_seed_audit'];audit=json.loads(audit_path.read_text());assert audit['visual_review_confirmed'] and audit['n_seed_blocks']==5
    for name,h in {**audit['input_sha256'],**audit['output_sha256']}.items():assert R.sha(FINAL/name)==h,name
    costs,bounds=timing_bounds(data,lc['scenarios']);jobs=expected_jobs(lc['scenarios'],base['h_targets'])
    complete=set();cached=set();checks=[]
    for sid,rep in {(s,r) for s,r,h in jobs}:
        p=data/'trajectories'/f'{sid}__rep{rep:03d}.json'
        if p.exists():
            a=json.loads(p.read_text());assert a['producer_fingerprint']==fp and a['complete'] and R.sha(p.with_suffix('.csv'))==a['sha256'];complete.add((sid,rep))
    for sid,rep,h in jobs:
        p=data/'snapshots'/f'{sid}__rep{rep:03d}__h{h:.3f}.json'
        if not p.exists():continue
        a=json.loads(p.read_text());assert a['producer_fingerprint']==fp
        for suffix,digest in a['hashes'].items():assert R.sha(p.with_suffix(suffix))==digest,p
        cached.add((sid,rep,h))
    pending=[k for k in jobs if k not in cached]
    pending.sort(key=lambda k:(-costs.get((k[0],k[2]),bounds[k[0]]['maximum_seconds']/88),k[0],k[1],k[2]))
    schedule=pd.DataFrame([dict(scenario_id=sid,replicate=rep,H=h,estimated_seconds=costs.get((sid,h),bounds[sid]['maximum_seconds']/88)) for sid,rep,h in pending])
    schedule.to_csv(out/'logs/11_pending_snapshot_schedule.csv',index=False)
    sources={str(p.relative_to(FINAL)):R.sha(p) for p in [CONFIG,Path(__file__),data/'manifest.json',audit_path]}
    orchestration_fp=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    manifest=dict(plan=plan,orchestration_source_sha256=sources,orchestration_fingerprint=orchestration_fp,scientific_producer_fingerprint=fp)
    p=out/'11_extension_manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Changed active extension controller/configuration'
    else:R.save_json(p,manifest)
    preflight=dict(status='passed',at_utc=now(),scenario_seed_targets=targets(lc['scenarios']),n_parameters=256,total_trajectories=104,total_snapshots=len(jobs),total_response_rows=len(jobs)*256,complete_trajectories=len(complete),cached_snapshots=len(cached),pending_snapshots=len(pending),historical_trajectory_bounds=bounds,estimated_pending_service_seconds=sum(x['estimated_seconds'] for x in schedule.to_dict('records')),scheduling='75th-percentile historical snapshot duration, longest first; unchanged producer pilot_H checkpoints',source_sha256=sources,scientific_sources_unchanged=True)
    R.save_json(out/'logs/11_preflight.json',preflight)
    return plan,ctx,out,pending,cached,complete,preflight

def guarded_progress_json(path,value):
    path=Path(path)
    if _CTX is not None and path.parent==_CTX[7]/'progress':
        locks=path.parent/'.write_locks';locks.mkdir(exist_ok=True)
        with (locks/(path.name+'.lock')).open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);return _SAVE_JSON(path,value)
    return _SAVE_JSON(path,value)

def initialise(ctx):
    global _CTX;_CTX=ctx
    # Only concurrent administrative progress writes require serialization.
    # Scientific snapshot/trajectory files and numerical functions are unchanged.
    R.save_json=guarded_progress_json

def snapshot(key):
    c,base,lc,design,bench,geo,gfp,data,out,fp=_CTX;sid,rep,h=key
    scenario=next(s for s in lc['scenarios'] if s['scenario_id']==sid);frame=bench[bench.scenario_id.eq(sid)&bench.replicate.eq(rep)]
    started=time.monotonic();answer=V.trajectory(scenario,rep,c,base,lc,frame,design,str(data),str(geo),fp,gfp,pilot_H=h)
    return dict(scenario_id=sid,replicate=rep,H=h,elapsed_seconds=time.monotonic()-started,pid=os.getpid(),snapshot_receipt_sha256=R.sha(data/'snapshots'/f'{sid}__rep{rep:03d}__h{h:.3f}.json'))

def finalise(key):
    c,base,lc,design,bench,geo,gfp,data,out,fp=_CTX;sid,rep=key
    scenario=next(s for s in lc['scenarios'] if s['scenario_id']==sid);frame=bench[bench.scenario_id.eq(sid)&bench.replicate.eq(rep)]
    # All88 checkpoints already exist. This unchanged producer verifies/assembles them.
    return V.trajectory(scenario,rep,c,base,lc,frame,design,str(data),str(geo),fp,gfp)

def run(state):
    plan,ctx,out,pending,cached,complete,preflight=state;c,base,lc,design,bench,geo,gfp,data,oldout,fp=ctx
    remaining=Counter((sid,rep) for sid,rep,h in pending);final_queued=set();errors=[];tasks={};todo=iter(pending);exhausted=False;done=len(cached)
    def status(stage):
        R.save_json(out/'logs/11_run_status.json',dict(status=stage,updated_at_utc=now(),pid=os.getpid(),scientific_producer_fingerprint=fp,scenario_seed_targets=targets(lc['scenarios']),complete_trajectories=len(complete),total_trajectories=104,completed_snapshots=done,total_snapshots=9152,pending_tasks=len(tasks),errors=errors))
    status('computing')
    with ProcessPoolExecutor(max_workers=plan['workers'],initializer=initialise,initargs=(ctx,)) as pool:
        while not exhausted or tasks or len(complete)<104:
            while not exhausted and len(tasks)<2*plan['workers']:
                try:key=next(todo)
                except StopIteration:exhausted=True;break
                tasks[pool.submit(snapshot,key)]=('snapshot',key)
            for sid,n in targets(lc['scenarios']).items():
                for rep in range(1,n+1):
                    key=(sid,rep)
                    if key not in complete and key not in final_queued and remaining[key]==0:
                        tasks[pool.submit(finalise,key)]=('finalise',key);final_queued.add(key)
            if not tasks:break
            finished,_=wait(tasks,timeout=30,return_when=FIRST_COMPLETED)
            for f in finished:
                kind,key=tasks.pop(f)
                try:
                    answer=f.result()
                    if kind=='snapshot':
                        sid,rep,h=key;cached.add(key);done+=1;remaining[(sid,rep)]-=1
                        R.save_json(out/'logs/task_receipts'/f'{sid}__rep{rep:03d}__h{h:.3f}.json',answer)
                        if done%50==0:print(f'Checkpoints {done}/9152',flush=True)
                    else:
                        complete.add(key);print(f'Complete trajectories {len(complete)}/104: {key}',flush=True)
                except Exception as exc:
                    errors.append(dict(kind=kind,key=list(key),type=type(exc).__name__,message=str(exc)))
                    print(f'Checked task failure {kind} {key}: {exc}',flush=True)
            status('computing')
    prefix=M.matched_prefix(c,lc,data,fp)
    assert prefix>=8 or errors,'Expected the shared eight-seed prefix'
    if errors or len(complete)!=104:
        status('incomplete_checked_failure');raise RuntimeError('Incomplete8/16 request; valid checkpoints retained; inspect exact errors')
    assert done==9152
    # Preserve original cached-resume timing fields; record snapshot service times separately.
    timings=[]
    for p in (out/'logs/task_receipts').glob('*.json'):timings.append(json.loads(p.read_text()))
    pd.DataFrame(timings).to_csv(out/'logs/11_new_snapshot_timings.csv',index=False)
    R.save_json(out/'logs/11_completion.json',dict(status='all_requested_8_16_computed_independent_audit_pending',completed_at_utc=now(),n_trajectories=104,n_snapshots=9152,n_rows=9152*256,common_matched_prefix=8,extra_directed_seed_blocks=list(range(9,17)),scenario_seed_targets=targets(lc['scenarios']),scientific_producer_fingerprint=fp))
    status('compute_complete_independent_audit_pending')
    print('All requested8/16 trajectories computed; independent summary follows.',flush=True)
    command=[sys.executable,'-m','src.tools.codex_run','final-confirmed-domain-unequal-summary','--workers',str(plan['workers'])]
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode:raise RuntimeError(f'Independent summary failed: {result.returncode}')
    status('numerical_summary_complete_export_and_visual_review_pending')

def live(pid):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return (Path(f'/proc/{pid}/cwd').resolve()==ROOT and ALIAS.encode() in args and b'compute' in args)
    except (FileNotFoundError,PermissionError,ProcessLookupError):return False

def launch():
    plan=json.loads(CONFIG.read_text());out=FINAL/plan['results_directory'];(out/'logs').mkdir(parents=True,exist_ok=True)
    with (out/'launch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=out/'logs/11_launch.json'
        if p.exists() and live(json.loads(p.read_text())['pid']):print(p.read_text());return
        with (out/'logs/11_compute.log').open('a',buffering=1) as log:
            process=subprocess.Popen([sys.executable,'-m','src.tools.codex_run',ALIAS,'--stage','compute'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        R.save_json(p,dict(pid=process.pid,launched_at_utc=now(),command=[ALIAS,'--stage','compute'],shutdown=False));print(p.read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['preflight','compute','launch'],default='preflight');a=p.parse_args()
    if a.stage=='launch':launch()
    elif a.stage=='preflight':print(json.dumps(setup()[-1],indent=2))
    else:
        initial=json.loads(M.DEFAULT.read_text());data=FINAL/initial['data_directory']
        with (data/'runner.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(setup())
