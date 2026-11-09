"""Extend and refine saturation boundaries without selecting fitted bird regions."""
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
import sys
import time
from types import SimpleNamespace
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-response-boundary-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
FOLDER=Path(__file__).parent
sys.path.insert(0,str(FINAL/'code/base'))
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence'))
sys.path.insert(0,str(FOLDER))
import numpy as np
import pandas as pd
from src.metapop.response_domain import quantile_bounds,envelope_certificate,cells_requiring_evaluation,boundary_bracket
from src.tools.final_plot_style import apply_final_style,figure_size,export_figure,MISSING
W=importlib.import_module('11_lhs_workers')
P=importlib.import_module('09_lhs_response_domain')
original_scaled=W.scaled_ratio_sweep
def tighter_scaled(*args,**kwargs):
    kwargs['tol']=min(kwargs['tol'],1e-12)
    return original_scaled(*args,**kwargs)
W.scaled_ratio_sweep=tighter_scaled
DEFAULT=FINAL/'config/02_metapop/02_02_species_parameter_selection/response_boundary_refinement/09_response_boundary_refinement.json'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    W.atomic_json(path,value)


def setup(path):
    c=json.loads(path.read_text());base=json.loads((FINAL/c['baseline_configuration']).read_text())
    lc=json.loads((FINAL/base['landscape_config']).read_text())
    assert base['replicates']==list(range(1,21)) and len(base['h_targets'])==88
    assert (base['q'],base['b'],base['x'])==(0,1,.5)
    audit_path=FINAL/c['baseline_domain_audit'];audit=json.loads(audit_path.read_text())
    assert audit['status']=='numerically_and_visually_checked_candidate'
    # Verify the exact sources of the previous quantile-envelope diagnostic.
    for name in [c['baseline_summary'],base['lhs_design'],c['baseline_configuration']]:
        assert sha(FINAL/name)==audit['input_sha256'][name],name
    old=pd.read_csv(FINAL/base['lhs_design'],float_precision='round_trip')
    s=pd.read_csv(FINAL/c['baseline_summary'],float_precision='round_trip')
    counts=s.groupby(['scenario_id','H_target']).n_valid.first()
    keys=counts[counts>=c['minimum_valid_seeds']].index
    assert len(keys)==601
    cells=keys.to_frame(index=False)
    q=np.stack([s.pivot(index=['scenario_id','H_target'],columns='parameter_id',values=v).reindex(index=keys,columns=old.parameter_id).to_numpy().T for v in ['q05','q95']],axis=1)
    assert q.shape==(128,2,601) and np.isfinite(q).all()
    benchmark=pd.read_csv(FINAL/base['landscape_data'])
    benchmark=benchmark[benchmark.replicate.isin(base['replicates'])&benchmark.H_target.isin(base['h_targets'])]
    assert len(benchmark)==12320 and not benchmark.duplicated(['scenario_id','replicate','H_target']).any()
    assert (benchmark.n_random_cumulative+benchmark.n_directed_cumulative).eq(benchmark.event_count).all()
    assert base['h_targets']==lc['h_targets']
    sources=[path,Path(__file__),Path(W.__file__),FINAL/'code/base/src/metapop/response_domain.py',
        FINAL/c['baseline_configuration'],FINAL/c['baseline_summary'],FINAL/base['lhs_design'],audit_path,
        FINAL/base['landscape_config'],FINAL/base['landscape_data']]
    sources+=list((FINAL/'code/base/src/landscape').glob('*.py'))
    sources+=[FINAL/'code/base/src/metapop'/f'{n}.py' for n in ['hanski','domain_selection']]
    hashes={str(p.relative_to(FINAL)):sha(p) for p in sources}
    fp=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    data,results=FINAL/c['data_directory'],FINAL/c['results_directory']
    for p in [data/'landscapes',data/'geometry',data/'queries',data/'responses',results/'logs',results/'tables',results/'figures']:p.mkdir(parents=True,exist_ok=True)
    manifest={'configuration':c,'producer_fingerprint':fp,'source_sha256':hashes}
    m=data/'manifest.json'
    if m.exists():assert json.loads(m.read_text())==manifest,'Changed refinement sources; cache reuse refused'
    else:save_json(m,manifest)
    save_json(results/'logs/09_preflight.json',dict(configuration=c,producer_fingerprint=fp,
        n_original_profiles=128,n_seed_blocks=20,n_scenarios=7,n_H_levels=88,n_eligible_cells=601,
        half_distances=c['half_distances'],ratio_window=c['initial_log10_r_window'],
        mathematical_zero_bound=c['r_zero_certificate_bound'],source_sha256=hashes))
    return c,base,lc,old,cells,q,benchmark,fp,data,results


def grids(scenario,rep,base,lc,benchmark,data,fp):
    identity=f"{scenario['scenario_id']}__rep{rep:03d}"
    file=data/'landscapes'/f'{identity}.npz';receipt=file.with_suffix('.json')
    if receipt.exists():
        meta=json.loads(receipt.read_text());assert meta['producer_fingerprint']==fp and sha(file)==meta['sha256']
        with np.load(file) as z:return z['grids'].copy(),z['H'].copy()
    seed=lc['base_seed']+rep-1;pars=lc['access_reference']
    run=W.run_sequential_deforestation(np.ones((200,200),dtype=np.int8),np.random.default_rng(seed),
        rho=scenario['rho'],directed_mode=scenario['mode'],target_habitat_fraction=0,
        generation_boundary=lc['generation_boundary'],eta=pars['eta'],k=pars['k'],lambda_decay=pars['lambda'],
        save_habitat_fractions=base['h_targets'],seed=seed)
    assert run.n_events==40000 and run.n_random_forced_initial==1
    assert run.n_random_total+run.n_directed==40000
    counts=np.r_[0,np.cumsum(np.asarray(run.event_modes)=='random')]
    arr=[];hs=[];identities=[]
    for snap in run.snapshots:
        h=float(snap.requested_habitat_fractions[0]);old=benchmark[benchmark.H_target.eq(h)].iloc[0]
        digest=hashlib.sha256(snap.grid.tobytes()).hexdigest()
        assert digest==old.grid_sha256 and snap.habitat_fraction==h
        event=int(snap.event_count)
        assert event==old.event_count and counts[event]==old.n_random_cumulative
        arr.append(snap.grid);hs.append(h);identities.append(digest)
    tmp=file.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,grids=np.asarray(arr,dtype=np.int8),H=np.asarray(hs))
    tmp.replace(file)
    save_json(receipt,dict(producer_fingerprint=fp,sha256=sha(file),grid_sha256=identities,
        full_event_trace_verified=True,seed=seed,n_snapshots=len(hs)))
    return np.asarray(arr),np.asarray(hs)


def geometry(grid,old,identity,data,fp):
    file=data/'geometry'/f'{identity}__h{old.H_target:.3f}.npz'
    if file.exists():
        meta=json.loads(file.with_suffix('.json').read_text())
        assert meta['producer_fingerprint']==fp and sha(file)==meta['sha256']
        with np.load(file) as z:
            assert str(z['grid_sha256'])==old.grid_sha256
            a,d=z['areas'].copy(),z['distances'].copy()
    else:
        land=W.Landscape(grid)
        land.ensure_clusters(edge_depth=0,connectivity='4',boundary_mode='open',distance_mode='voronoi_dt',voronoi_depth=1)
        a=np.asarray(land.cluster_areas,float);d=np.asarray(land.cluster_dist_matrix,float)
        tmp=file.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,areas=a,distances=d,grid_sha256=np.array(old.grid_sha256))
        tmp.replace(file)
        save_json(file.with_suffix('.json'),dict(producer_fingerprint=fp,sha256=sha(file)))
    assert len(a)==old.n_clusters and a.sum()==round(40000*old.H_target)
    assert np.isfinite(d).all() and (d>=0).all() and np.array_equal(d,d.T)
    assert d.max()<=282.8427124746191
    # Compare already available pilot geometry where the snapshot overlaps.
    prior=FINAL/'data/processed/02_metapop/02_02_species_parameter_selection/limit_based/geometry'/file.name
    if prior.exists():
        with np.load(prior) as z:
            assert str(z['grid_sha256'])==old.grid_sha256
            np.testing.assert_array_equal(a,z['areas']);np.testing.assert_allclose(d,z['distances'],atol=1e-12,rtol=0)
    return a,d


def worker(qid,ell,lr,scenario,rep,hs,base,lc,benchmark,data_name,fp):
    started=time.monotonic();data=Path(data_name);identity=f"{scenario['scenario_id']}__rep{rep:03d}"
    folder=data/'responses'/qid;folder.mkdir(exist_ok=True)
    file=folder/f'{identity}.csv';receipt=file.with_suffix('.json')
    if receipt.exists():
        meta=json.loads(receipt.read_text());assert meta['producer_fingerprint']==fp
        for suffix,digest in meta['hashes'].items():assert sha(file.with_suffix(suffix))==digest
        return pd.read_csv(file,float_precision='round_trip'),meta['elapsed_seconds']
    gs,levels=grids(scenario,rep,base,lc,benchmark,data,fp)
    item=SimpleNamespace(parameter_id=qid,half_distance=ell,alpha=np.log(2)/ell,log_r=lr)
    options={**base,'scalar_check_ids':[qid]}
    records=[];vectors=[];offsets=[0]
    for h in hs:
        old=benchmark[benchmark.H_target.eq(h)].iloc[0];grid=gs[np.flatnonzero(levels==h)[0]]
        assert hashlib.sha256(grid.tobytes()).hexdigest()==old.grid_sha256
        if old.n_clusters>=2:
            a,d=geometry(grid,old,identity,data,fp)
            p,report=W.solve(a,d,item,options)
            assert report['status']=='valid','Do not certify a nonconverged response'
        else:
            p=np.empty(0);report=dict(occupancy=np.nan,converged=False,fixed_point_residual=np.nan,
                n_iter=0,independent_probability_difference=np.nan,status='excluded_fewer_than_two_patches')
        records.append(dict(query_id=qid,half_distance=ell,log_r=lr,scenario_id=scenario['scenario_id'],
            replicate=rep,seed=lc['base_seed']+rep-1,H_target=h,n_patches=old.n_clusters,
            grid_sha256=old.grid_sha256,producer_fingerprint=fp,**report))
        vectors.append(p);offsets.append(offsets[-1]+len(p))
    table=pd.DataFrame(records);tmp=file.with_suffix('.csv.tmp');table.to_csv(tmp,index=False);tmp.replace(file)
    tmp=file.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,probabilities=np.concatenate(vectors),offsets=np.asarray(offsets))
    tmp.replace(file.with_suffix('.npz'))
    save_json(receipt,dict(producer_fingerprint=fp,elapsed_seconds=time.monotonic()-started,
        hashes={s:sha(file.with_suffix(s)) for s in ['.csv','.npz']}))
    return table,time.monotonic()-started


def evaluate(qid,ell,lr,label,state,ctx,pool):
    c,base,lc,old,cells,q,benchmark,fp,data,results=ctx
    path=data/'queries'/f'{qid}.json';array=path.with_suffix('.npz')
    if path.exists():
        record=json.loads(path.read_text());assert record['producer_fingerprint']==fp
        assert record['half_distance']==ell and record['log_r']==lr and sha(array)==record['interval_sha256']
        with np.load(array) as z:lo,hi=z['lower'].copy(),z['upper'].copy()
    else:
        started=time.monotonic()
        lo,hi=quantile_bounds(state['ell'],state['lr'],state['lo'],state['hi'],ell,lr,c['distance_upper_bound'])
        method='entrywise_kernel_ratio_bounds'
        if lr>np.log(c['r_zero_certificate_bound']):
            lo=np.zeros_like(lo);hi=lo.copy();method='universal_row_sum_empty_certificate'
        # An invariant lower box for the maximal equilibrium, valid for every
        # off-diagonal distance <=D and every raw patch area >=1.
        safe_lr=-np.log(2)*c['distance_upper_bound']/ell+np.log(.025)
        if lr<=safe_lr:
            lo=np.maximum(lo,.975);method='universal_invariant_lower_box'
        needed=cells_requiring_evaluation(lo,hi,c['saturation_threshold'])
        raw=[];hashes={};valid=excluded=0;residual=0.
        if needed.any():
            selected=cells[needed]
            futures={}
            for sid,part in selected.groupby('scenario_id'):
                scenario=next(s for s in lc['scenarios'] if s['scenario_id']==sid)
                for rep in base['replicates']:
                    frame=benchmark[benchmark.scenario_id.eq(sid)&benchmark.replicate.eq(rep)]
                    f=pool.submit(worker,qid,ell,lr,scenario,rep,part.H_target.tolist(),base,lc,frame,str(data),fp)
                    futures[f]=(sid,rep)
            for f in as_completed(futures):
                table,seconds=f.result();raw.append(table)
                sid,rep=futures[f];file=data/'responses'/qid/f'{sid}__rep{rep:03d}.csv'
                for suffix in ['.csv','.npz','.json']:hashes[str(file.with_suffix(suffix).relative_to(FINAL))]=sha(file.with_suffix(suffix))
            raw=pd.concat(raw,ignore_index=True)
            assert not raw.duplicated(['scenario_id','replicate','H_target']).any()
            stats=raw.groupby(['scenario_id','H_target']).occupancy.quantile([.05,.95],interpolation='linear').unstack()
            got=cells[needed].merge(stats.reset_index(),on=['scenario_id','H_target'],validate='one_to_one')
            exact=got[[.05,.95]].to_numpy().T
            assert exact.shape==(2,int(needed.sum())) and np.isfinite(exact).all()
            # Independent scalar quantiles and source interval consistency.
            for k,row in enumerate(cells[needed].itertuples(index=False)):
                vals=raw[raw.scenario_id.eq(row.scenario_id)&raw.H_target.eq(row.H_target)].occupancy.dropna().to_numpy()
                assert len(vals)>=c['minimum_valid_seeds']
                np.testing.assert_allclose(exact[:,k],np.quantile(vals,[.05,.95],method='linear'),atol=1e-14,rtol=0)
            assert (exact>=lo[:,needed]-1e-8).all() and (exact<=hi[:,needed]+1e-8).all()
            lo[:,needed]=exact;hi[:,needed]=exact
            valid=int(raw.status.eq('valid').sum());excluded=int(len(raw)-valid)
            residual=float(raw.fixed_point_residual.max())
            assert residual<=base['solver']['tol']*1.1
            method+=' + direct_matched_seed_quantiles'
        cert=envelope_certificate(lo,hi,c['saturation_threshold']);assert cert['category']!=3
        with array.open('wb') as f:np.savez_compressed(f,lower=lo,upper=hi,direct_cells=needed)
        record=dict(query_id=qid,half_distance=float(ell),log_r=float(lr),log10_r=float(lr/np.log(10)),
            purpose=label,producer_fingerprint=fp,**cert,method=method,n_direct_cells=int(needed.sum()),
            n_valid_vectors=valid,n_exclusions=excluded,max_residual=residual,raw_sha256=hashes,
            interval_sha256=sha(array),elapsed_seconds=time.monotonic()-started)
        save_json(path,record)
    state['ell'].append(ell);state['lr'].append(lr);state['categories'].append(record['category'])
    state['lo']=np.concatenate([state['lo'],lo[None]],axis=0);state['hi']=np.concatenate([state['hi'],hi[None]],axis=0)
    state['records'].append(record)
    print(f"{qid}: ell={ell:g}, log10(r)={lr/np.log(10):.5g}, class={record['category']}, cells={record['n_direct_cells']}, {record['elapsed_seconds']:.1f}s",flush=True)
    save_json(results/'logs/09_progress.json',dict(status='running',completed_queries=len(state['records']),
        direct_cells=sum(z['n_direct_cells'] for z in state['records']),last_query=record))
    return record


def boundary_table(state,c,window):
    rows=[]
    for ell in c['half_distances']:
        for name in ['lower','upper']:
            a,b,sa,sb=boundary_bracket(state['ell'],state['lr'],state['categories'],ell,name,window)
            width=(b-a)/np.log(10)
            rows.append(dict(half_distance=ell,boundary=name,log10_r_low=a/np.log(10),log10_r_high=b/np.log(10),
                bracket_width_decades=width,lower_side_supported=sa,upper_side_supported=sb,
                resolution_reached=bool(sa and sb and width<=c['boundary_tolerance_decades']+1e-10)))
    return pd.DataFrame(rows)


def compute(ctx,workers):
    c,base,lc,old,cells,q,benchmark,fp,data,results=ctx
    state=dict(ell=old.half_distance.tolist(),lr=old.log_r.tolist(),lo=q.copy(),hi=q.copy(),
        categories=[envelope_certificate(v,v)['category'] for v in q],records=[])
    window=np.asarray(c['initial_log10_r_window'])*np.log(10)
    initial=boundary_table(state,c,window);initial.to_csv(results/'tables/09_before_boundary_brackets.csv',index=False)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def query(ell,lr,label):
            for record in state['records']:
                if record['half_distance']==ell and record['log_r']==lr:return record
            return evaluate(f"ref{len(state['records'])+1:03d}",float(ell),float(lr),label,state,ctx,pool)
        # The lowest ell is the hardest ceiling endpoint by monotonicity.
        low=query(min(c['half_distances']),window[0],'lower-window closure')
        for extension in range(c['maximum_lower_window_extensions']):
            if low['category']==0:break
            window[0]-=c['lower_window_extension_step_decades']*np.log(10)
            low=query(min(c['half_distances']),window[0],'lower-window extension')
        assert low['category']==0,'Lower rate window still censored; no completion claim'
        query(max(c['half_distances']),window[0],'lower-window far corner')
        query(min(c['half_distances']),window[1],'upper-window near corner')
        query(max(c['half_distances']),window[1],'upper-window closure')
        for round_number in range(1,c['maximum_bisection_rounds']+1):
            for ell in c['half_distances']:
                for name in ['lower','upper']:
                    a,b,sa,sb=boundary_bracket(state['ell'],state['lr'],state['categories'],ell,name,window)
                    if (b-a)/np.log(10)<=c['boundary_tolerance_decades']+1e-10:continue
                    query(ell,(a+b)/2,f'{name} boundary round {round_number}')
            boundary_table(state,c,window).to_csv(results/'tables/09_latest_boundary_brackets.csv',index=False)
    records=pd.DataFrame(state['records']);records.to_csv(results/'tables/09_refinement_queries.csv',index=False)
    after=boundary_table(state,c,window);after.to_csv(results/'tables/09_refined_boundary_brackets.csv',index=False)
    # Save all certificates with their actual intervals, not made-up occupancy.
    np.savez_compressed(data/'09_combined_quantile_bounds.npz',half_distance=np.asarray(state['ell']),
        log_r=np.asarray(state['lr']),lower=state['lo'],upper=state['hi'],category=np.asarray(state['categories']),
        scenario_id=cells.scenario_id.to_numpy(dtype=str),H=cells.H_target.to_numpy())
    save_json(results/'logs/09_compute_receipt.json',dict(status='classified_boundary_queries_complete',
        producer_fingerprint=fp,final_log10_r_window=(window/np.log(10)).tolist(),
        n_queries=len(records),n_direct_cells=int(records.n_direct_cells.sum()),
        n_valid_vectors=int(records.n_valid_vectors.sum()),n_exclusions=int(records.n_exclusions.sum()),
        max_residual=float(records.max_residual.max()),n_brackets=len(after),
        n_resolved_brackets=int(after.resolution_reached.sum()),all_rate_endpoints_closed=True,
        outputs_sha256={str(p.relative_to(FINAL)):sha(p) for p in [data/'09_combined_quantile_bounds.npz',results/'tables/09_refinement_queries.csv',results/'tables/09_refined_boundary_brackets.csv']}))
    print('Compute complete; independent audit and visual review still required.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['preflight','compute'],default='compute')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();ctx=setup(args.config.resolve());data=ctx[-2]
    with (data/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='preflight':print('Expanded boundary preflight passed.',flush=True)
        else:compute(ctx,min(args.workers,ctx[0]['maximum_workers']))


if __name__=='__main__':main()
