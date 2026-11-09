"""Staged 3->5->8-seed comparison and q=1 transition-domain refinement.

All original pilot files remain immutable. Complete seed blocks are the units
of extension and omission. Domain selection uses a separate regular grid and
the existing transition-envelope/plateau protocol, never scenario contrasts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-self-extension-mpl')
HERE=Path(__file__).resolve().parent
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
P=module('pilot_extension_core','13_self_colonization_pilot.py')
R=module('pilot_extension_recovery','13_self_colonization_recovery.py')
V=module('pilot_extension_review','13_self_colonization_review.py')
import numpy as np
import pandas as pd
from src.metapop.transition_domain import transition_crossing,select_transition_rectangle
from src.tools.weighted_rank_associations import balanced_spearman
REL='02_metapop/02_04_robustness/self_colonization_extension_20261015'
DEFAULT=P.FINAL/'config'/REL/'13_extension.json'

def setup(path):
    c=json.loads(path.read_text());data=P.FINAL/'data/processed'/REL;out=P.FINAL/'results'/REL
    for p in [data/'snapshots',data/'domain',out/'logs',out/'tables',out/'figures']:p.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(P.FINAL/c['parameter_design'],float_precision='round_trip')
    d=d[d.dispersion_group.isin(np.rint(np.linspace(0,63,16)).astype(int))|d.parameter_id.eq(c['reference'])].reset_index(drop=True)
    ext=pd.DataFrame([dict(parameter_id=f'extension_e{i:02d}_r{j:02d}',r=r,half_distance=ell,log_r=np.log(r),alpha=np.log(2)/ell)
        for i,ell in enumerate(c['extension_half_distances']) for j,r in enumerate(c['extension_ratios'])])
    domain=pd.DataFrame([dict(parameter_id=f'q1grid_e{i:02d}_r{j:02d}',r=10.**lr,log10_r=lr,log_r=lr*np.log(10),half_distance=ell,alpha=np.log(2)/ell)
        for i,ell in enumerate(c['domain_half_distances']) for j,lr in enumerate(c['domain_log10_ratios'])])
    land=pd.read_csv(P.FINAL/c['landscape_data'],float_precision='round_trip')
    land=land[land.replicate.isin(c['replicates'])&land.H_target.isin(c['h_targets'])]
    assert len(d)==65 and len(domain)==481 and len(land)==8*7*13
    assert not land.duplicated(['scenario_id','replicate','H_target']).any()
    assert (land.n_random_cumulative+land.n_directed_cumulative).eq(land.event_count).all()
    sources=[path,Path(__file__),Path(P.__file__),Path(R.__file__),Path(V.__file__),P.FINAL/c['parameter_design'],P.FINAL/c['landscape_data']]
    sources += [P.FINAL/'code/base/src/metapop'/f'{n}.py' for n in ['hanski','self_equilibrium','domain_selection','transition_domain']]
    sources += [P.FINAL/'code/base/src/tools/weighted_rank_associations.py']
    hashes={str(p.relative_to(P.FINAL)):P.sha(p) for p in sources}
    fp=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    manifest=dict(configuration=c,source_sha256=hashes,fingerprint=fp)
    file=data/'manifest.json'
    if file.exists():assert json.loads(file.read_text())==manifest,'Changed extension sources; cache reuse refused'
    else:P.save(file,manifest)
    old=P.FINAL/c['pilot_data'];old_out=P.FINAL/c['pilot_results']
    check=json.loads((old_out/'logs/13_final_audit.json').read_text());assert check['status']=='numerically_and_visually_checked_pilot'
    assert P.sha(old_out/'logs/13_independent_audit.json')==check['independent_audit_sha256']
    recovered=json.loads((old_out/'logs/13_critical_recovery.json').read_text())
    for name,digest in recovered['supplemental_sources'].items():assert P.sha(P.FINAL/name)==digest
    prior={}
    for csv in sorted((old/'snapshots').glob('*.csv')):
        meta=json.loads(csv.with_suffix('.json').read_text())
        assert P.sha(csv)==meta['csv_sha256'] and P.sha(csv.with_suffix('.npz'))==meta['vectors_sha256']
        for suffix in ['.csv','.npz','.json']:
            source=csv.with_suffix(suffix);target=data/'snapshots'/source.name
            if not target.exists():target.symlink_to(source)
            assert target.resolve()==source.resolve();prior[source.name]=P.sha(source)
    assert len(prior)==273*3
    P.save(out/'logs/13_retained_pilot.json',dict(pilot_final_audit_sha256=P.sha(old_out/'logs/13_final_audit.json'),files=prior))
    d.to_csv(data/'13_matched_parameters.csv',index=False);domain.to_csv(data/'13_domain_parameters.csv',index=False)
    P.save(out/'logs/13_preflight.json',dict(status='passed',fingerprint=fp,
        stages=c['seed_stages'],n_matched_profiles=len(d),n_domain_profiles=len(domain),domain_H=c['extension_h_targets'],
        domain_log10_r=[float(domain.log10_r.min()),float(domain.log10_r.max())],
        domain_r=[float(domain.r.min()),float(domain.r.max())],domain_half_distances=c['domain_half_distances'],
        endpoint_certificates='q=1, integer raw areas>=1: at r=.01 occupancy>=.99 by the local self limit; r=1e7 exceeds universal row-sum bound 8e6.',
        selection=c['domain_selection'],stage_extension_rules=c['stage_extension_rules']))
    return c,d,ext,domain,land,data,out,fp

def matched_task(scenario,rep,c,d,ext,land,data,fp):
    P.solve_group=R.recovered_group
    return P.task(scenario,rep,c,d,ext,land,data,fp)

def geometry(row,c):
    identity=f'{row.scenario_id}__rep{int(row.replicate):03d}__h{row.H_target:.3f}'
    path=P.FINAL/c['geometry_cache']/'geometry'/f'{identity}.npz'
    marker=json.loads(path.with_suffix('.json').read_text());assert P.sha(path)==marker['sha256']
    with np.load(path) as z:
        a,d=z['areas'],z['distances'];assert str(z['grid_sha256'])==row.grid_sha256
    assert len(a)==row.n_clusters and a.sum()==round(40000*row.H_target)
    assert d.shape==(len(a),len(a)) and np.isfinite(d).all() and (d>=0).all() and np.array_equal(d,d.T)
    return a,d

def domain_task(row,c,domain,data_name,fp):
    data=Path(data_name);identity=f'{row.scenario_id}__rep{int(row.replicate):03d}__h{row.H_target:.3f}'
    path=data/'domain'/f'{identity}.csv';marker=path.with_suffix('.json')
    if marker.exists():
        meta=json.loads(marker.read_text());assert meta['fingerprint']==fp
        assert P.sha(path)==meta['csv_sha256'] and P.sha(path.with_suffix('.npz'))==meta['vectors_sha256']
        return identity,True
    a,d=geometry(row,c);answers={};reused=0
    old=data/'snapshots'/path.name
    prior=pd.read_csv(old,float_precision='round_trip')
    pm=json.loads(old.with_suffix('.json').read_text());assert P.sha(old.with_suffix('.npz'))==pm['vectors_sha256']
    with np.load(old.with_suffix('.npz')) as z:
        vectors=z['probabilities'];offsets=z['offsets']
    # Reuse only exactly matched coarse-grid model parameters, with original
    # vector diagnostics and per-source hashes retained. Remaining points are new.
    for ell,group in domain.groupby('half_distance',sort=False):
        missing=[]
        for index,item in group.iterrows():
            match=prior.index[(prior.half_distance==ell)&np.isclose(prior.r,item.r,rtol=1e-14,atol=0)]
            if len(match):
                i=int(match[0]);v=vectors[offsets[i]:offsets[i+1]].copy();report=prior.loc[i].to_dict()
                answers[index]=(v,dict(occupancy=float(v.mean()),fixed_point_residual=float(report['fixed_point_residual']),
                    solver='retained_verified_pilot_vector',n_iter=int(report['n_iter']),independent_difference=report.get('independent_difference',np.nan)));reused+=1
            else:missing.append(index)
        if missing:
            values,reports=R.recovered_group(a,d,domain.loc[missing],c)
            for j,index in enumerate(missing):answers[index]=(values[:,j],reports[j])
    p=np.stack([answers[i][0] for i in domain.index],axis=1);records=[]
    for i,item in domain.iterrows():
        report=answers[i][1]
        records.append(dict(parameter_id=item.parameter_id,r=item.r,log10_r=item.log10_r,half_distance=item.half_distance,
            scenario_id=row.scenario_id,replicate=row.replicate,H_target=row.H_target,seed=row.seed,
            n_patches=len(a),grid_sha256=row.grid_sha256,**{k:v for k,v in report.items() if k!='occupancy'},occupancy=float(p[:,i].mean())))
    # Recompute every original-equation residual, including reused coarse vectors.
    for ell,group in domain.groupby('half_distance',sort=False):
        indices=group.index.to_numpy();v=p[:,indices];k=np.exp(-np.log(2)/ell*d);np.fill_diagonal(k,1)
        source=k@(a[:,None]*v);target=source/(source+a[:,None]**(-.5)*group.r.to_numpy())
        assert np.max(np.abs(target-v))<=1.1*c['tol']
    table=pd.DataFrame(records);table.to_csv(path,index=False)
    np.savez_compressed(path.with_suffix('.npz'),probabilities=p)
    P.save(marker,dict(fingerprint=fp,csv_sha256=P.sha(path),vectors_sha256=P.sha(path.with_suffix('.npz')),
        coarse_source_sha256=P.sha(old),coarse_vector_sha256=P.sha(old.with_suffix('.npz')),n_reused=reused,grid_sha256=row.grid_sha256))
    return identity,False

def view(data,n):
    result=data/'views'/f'seeds{n:03d}';(result/'snapshots').mkdir(parents=True,exist_ok=True)
    for p in (data/'snapshots').iterdir():
        rep=int(p.name.split('__rep')[1][:3])
        if rep<=n:
            target=result/'snapshots'/p.name
            if not target.exists():target.symlink_to(p)
    assert len(list((result/'snapshots').glob('*.csv')))==7*13*n
    return result

def matched_summary(c,d,land,data,out,n):
    stage=out/f'seeds{n:03d}'
    for p in [stage/'tables',stage/'logs',stage/'figures']:p.mkdir(parents=True,exist_ok=True)
    cc={**c,'replicates':list(range(1,n+1))};ll=land[land.replicate<=n]
    P.summaries(cc,d,ll,stage,view(data,n))
    file=stage/'logs/13_numeric_audit.json';audit=json.loads(file.read_text())
    audit['limitations']=f'{n} shared seeds; 65 fixed matched profiles; descriptive sensitivity, no bootstrap CIs or bird refits.'
    P.save(file,audit);return stage

def transitions(table,c,n,omit=None):
    part=table[table.replicate<=n]
    if omit is not None:part=part[part.replicate!=omit]
    summary=part.groupby(['scenario_id','H_target','half_distance','log10_r']).occupancy.agg(n='count',median='median').reset_index()
    summary.loc[summary.n<c['minimum_valid_seeds'],'median']=np.nan
    rows=[]
    for (scenario,h,ell),g in summary.groupby(['scenario_id','H_target','half_distance']):
        g=g.sort_values('log10_r')
        if g['median'].isna().all():continue
        row=dict(scenario_id=scenario,H_target=h,half_distance=ell)
        for target,name in [(.95,'p95'),(.5,'p50'),(.05,'p05')]:
            cross=transition_crossing(g.log10_r,g['median'],target)
            assert cross['status']=='resolved',row
            row.update({name:cross['value'],name+'_low':cross['low'],name+'_high':cross['high']})
        rows.append(row)
    return pd.DataFrame(rows)

def domain_summary(c,data,out,n):
    files=[p for p in (data/'domain').glob('*.csv') if int(p.name.split('__rep')[1][:3])<=n]
    assert len(files)==7*5*n
    table=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in files],ignore_index=True)
    assert len(table)==481*7*5*n and not table.duplicated(['scenario_id','replicate','H_target','parameter_id']).any()
    stage=out/f'seeds{n:03d}';tt=transitions(table,c,n)
    tt.to_csv(stage/'tables/13_q1_transitions.csv',index=False)
    rules=c['domain_selection'];decision=select_transition_rectangle(tt,**rules)
    P.save(stage/'logs/13_q1_domain.json',dict(status='verified_sample_conditional_working_domain',q=1,n_seeds=n,**decision))
    checks=[]
    variants=[('main',rules)]+[(f'plateau_{v}',{**rules,'plateau_tolerance':v}) for v in [.025,.1]]+[(f'margin_{v}',{**rules,'margin_log10_r':v}) for v in [.125,.5]]+[('lower_distance_2',{**rules,'lower_half_distance':2.})]
    for name,settings in variants:
        try:
            result=select_transition_rectangle(tt,**settings);checks.append(dict(check=name,omitted_seed=None,**{k:result[k] for k in ['half_distance_min','half_distance_max','log10_r_min','log10_r_max']},status='resolved'))
        except ValueError as error:checks.append(dict(check=name,omitted_seed=None,status=str(error)))
    for seed in range(1,n+1):
        result=select_transition_rectangle(transitions(table,c,n,omit=seed),**rules)
        checks.append(dict(check='complete_seed_omission',omitted_seed=seed,**{k:result[k] for k in ['half_distance_min','half_distance_max','log10_r_min','log10_r_max']},status='resolved'))
    pd.DataFrame(checks).to_csv(stage/'tables/13_domain_sensitivity.csv',index=False)
    return decision

def compare_stages(old,new,c):
    keys=['q','parameter_id','scenario_id','H_target']
    a=pd.read_csv(old/'tables/13_B_C_medians_contrasts.csv').set_index(keys)
    b=pd.read_csv(new/'tables/13_B_C_medians_contrasts.csv').set_index(keys)
    result={}
    for col in ['median','contrast']:
        difference=(b[col]-a[col]).abs().dropna();result[col+'_maximum_change']=float(difference.max())
        result[col+'_fraction_over_005']=float((difference>.05).mean())
    keys=['q','parameter_id','H_target','metric']
    a=pd.read_csv(old/'tables/13_D_metric_associations.csv').set_index(keys).coefficient
    b=pd.read_csv(new/'tables/13_D_metric_associations.csv').set_index(keys).coefficient
    result['D_maximum_change']=float((b-a).abs().max())
    result['extend_to_eight']=result['median_maximum_change']>c['stage_extension_rules']['occupancy'] or result['contrast_maximum_change']>c['stage_extension_rules']['contrast'] or result['D_maximum_change']>c['stage_extension_rules']['association']
    P.save(new/'logs/13_stage_comparison.json',result);return result

def compute(c,d,ext,domain,land,data,out,fp,workers):
    for n in c['seed_stages']:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[]
            for scenario in land.scenario_id.unique():
                for rep in range(4,n+1):
                    if (data/f'{scenario}__rep{rep:03d}.json').exists():continue
                    futures.append(pool.submit(matched_task,scenario,rep,c,d,ext,land[land.replicate<=n],str(data),fp))
            for f in as_completed(futures):f.result()
        stage=matched_summary(c,d,land,data,out,n)
        print(f'Matched stage {n} complete',flush=True)
        selected=land[land.replicate.le(n)&land.H_target.isin(c['extension_h_targets'])]
        assert len(selected)==7*5*n and selected.n_clusters.ge(2).all()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(domain_task,SimpleNamespace(**row),c,domain,str(data),fp) for row in selected.to_dict(orient='records')]
            for i,f in enumerate(as_completed(futures),1):
                f.result()
                if i%10==0:print(f'Domain stage {n}: {i}/{len(futures)} snapshots',flush=True)
        decision=domain_summary(c,data,out,n)
        print('Domain',n,{k:decision[k] for k in ['half_distance_min','half_distance_max','log10_r_min','log10_r_max']},flush=True)
        if n==5:
            change=compare_stages(P.FINAL/c['pilot_results'],stage,c)
            checks=pd.read_csv(stage/'tables/13_domain_sensitivity.csv');omissions=checks[checks.check.eq('complete_seed_omission')]
            boundary_change=max((omissions.log10_r_min-decision['log10_r_min']).abs().max(),(omissions.log10_r_max-decision['log10_r_max']).abs().max())
            change['domain_omission_maximum_decades']=float(boundary_change)
            change['extend_to_eight']|=boundary_change>.25 or not omissions.half_distance_max.eq(decision['half_distance_max']).all()
            P.save(stage/'logs/13_stage_comparison.json',change)
            if not change['extend_to_eight']:
                P.save(out/'logs/13_compute_complete.json',dict(n_seeds=n,status='complete',fingerprint=fp));return
    compare_stages(out/'seeds005',out/'seeds008',c)
    P.save(out/'logs/13_compute_complete.json',dict(n_seeds=8,status='complete',fingerprint=fp))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['preflight','compute','audit','export'],default='compute');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    # One controller per output; automatically released on process exit.
    lock_path=P.FINAL/'data/processed'/REL/'controller.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c,d,ext,domain,land,data,out,fp=setup(args.config)
        if args.stage=='preflight':return
        if args.stage=='compute':compute(c,d,ext,domain,land,data,out,fp,args.workers);return
        n=json.loads((out/'logs/13_compute_complete.json').read_text())['n_seeds'];cc={**c,'replicates':list(range(1,n+1))};stage=out/f'seeds{n:03d}'
        if args.stage=='audit':V.audit(cc,d,land[land.replicate<=n],stage,view(data,n))
        if args.stage=='export':V.export(cc,d,stage)

if __name__=='__main__':main()
