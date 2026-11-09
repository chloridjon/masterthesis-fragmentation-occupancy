"""Paired q=0/q=1 screening for Results B-D; separate coarse q=1 range survey.

Run from repository root with final-self-colonization-pilot. Three complete
seed blocks, seven scenarios, prespecified H levels and a systematic subset of
the fixed conditional design. No bootstrap inference or new reference adoption.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-self-pilot-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.metapop.verified_batch import raw_ratio_group
from src.metapop.hanski import run_continuous_equilibrium_from_kernel
from src.metapop.domain_selection import normalized_colonization,scaled_ratio_sweep,capacity_bracket
from src.tools.weighted_rank_associations import balanced_spearman
from src.tools.final_plot_style import FOCAL_SCENARIOS

REL='02_metapop/02_04_robustness/self_colonization_pilot_20261015'
DEFAULT=FINAL/'config'/REL/'13_pilot.json'
METRICS=['edge_density_total','largest_cluster_fraction_forest','chi',
         'mean_nearest_neighbor_distance','mean_pairwise_distance','complete_mst_d50']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def inputs(config):
    c=json.loads(config.read_text())
    design=pd.read_csv(FINAL/c['parameter_design'],float_precision='round_trip')
    groups=np.rint(np.linspace(0,63,16)).astype(int)
    design=design[design.dispersion_group.isin(groups)|design.parameter_id.eq(c['reference'])].reset_index(drop=True)
    ext=pd.DataFrame([dict(parameter_id=f'extension_e{i:02d}_r{j:02d}',r=float(r),
        half_distance=float(ell),log_r=float(np.log(r)),alpha=float(np.log(2)/ell))
        for i,ell in enumerate(c['extension_half_distances']) for j,r in enumerate(c['extension_ratios'])])
    land=pd.read_csv(FINAL/c['landscape_data'],float_precision='round_trip')
    land=land[land.replicate.isin(c['replicates'])&land.H_target.isin(c['h_targets'])]
    assert len(design)==65 and len(land)==7*len(c['replicates'])*len(c['h_targets'])
    assert not land.duplicated(['scenario_id','replicate','H_target']).any()
    assert (land.n_random_cumulative+land.n_directed_cumulative).eq(land.event_count).all()
    np.testing.assert_allclose(design.r,np.exp(design.log_r),rtol=1e-14)
    np.testing.assert_allclose(design.alpha,np.log(2)/design.half_distance,rtol=1e-14)
    out=FINAL/'results'/REL;data=FINAL/'data/processed'/REL
    for p in [out/'logs',out/'tables',out/'figures',data/'snapshots']:p.mkdir(parents=True,exist_ok=True)
    sources=[config,Path(__file__),FINAL/c['parameter_design'],FINAL/c['landscape_data']]
    sources += [FINAL/'code/base/src/metapop'/f'{n}.py' for n in ['hanski','domain_selection','parameter_sweep','verified_batch']]
    sources += [FINAL/'code/base/src/tools/weighted_rank_associations.py']
    hashes={str(p.relative_to(FINAL)):sha(p) for p in sources}
    fp=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    manifest=dict(configuration=c,source_sha256=hashes,fingerprint=fp)
    path=data/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Pilot sources changed; choose a separate output path'
    else:save(path,manifest)
    design.to_csv(data/'13_matched_parameters.csv',index=False);ext.to_csv(data/'13_extension_parameters.csv',index=False)
    save(out/'logs/13_preflight.json',dict(fingerprint=fp,n_parameters=len(design),n_landscapes=len(land),
        n_extension_parameters=len(ext),r_range=[float(design.r.min()),float(design.r.max())],
        half_distance_range=[float(design.half_distance.min()),float(design.half_distance.max())],
        extension_r=c['extension_ratios'],extension_half_distance=c['extension_half_distances'],
        replication_unit='complete shared landscape seed; no profile resampling',
        uncertainty='descriptive pilot; no confidence intervals',status='passed'))
    return c,design,ext,land,out,data,fp

def solve_group(a,d,items,c):
    ell=float(items.half_distance.iloc[0]);ratios=items.r.to_numpy()
    p,reports,k=raw_ratio_group(a,d,ell,ratios,q=1,max_iter=c['max_iter'],tol=c['tol'])
    check=(len(a)<=200)
    independent=None
    if check:
        w,s=normalized_colonization(a,d,ell,q=1)
        independent=scaled_ratio_sweep(w,s,np.log(ratios),tol=1e-13,max_iter=c['continuation_max_iter'])
    for j,r in enumerate(ratios):
        report=reports[j]
        if report['needs_scalar_verification']:
            bracket=None
            try:bracket=capacity_bracket(a,d,ell,q=1,max_iter=10000)
            except ArithmeticError:pass
            if bracket is not None and r>bracket.upper:
                p[:,j]=0;report.update(converged=True,solver='capacity_verified_empty',n_iter=0)
            else:
                # The kernel already contains q on its diagonal. No second self term.
                result=run_continuous_equilibrium_from_kernel(a,k,r=r,b=1,x=.5,
                    initial=1,tol=c['tol']/10,max_iter=c['continuation_max_iter'])
                assert result.converged,('q=1 scalar did not converge',ell,r,len(a))
                p[:,j]=result.probabilities;report.update(converged=True,solver='continued_scalar',n_iter=result.n_iter)
        source=k@(a*p[:,j]);target=source/(source+r*a**(-.5))
        residual=float(np.max(np.abs(target-p[:,j])))
        assert residual<=c['tol']*1.1,(ell,r,residual)
        report['fixed_point_residual']=residual
        report['independent_difference']=None
        if independent is not None:
            other=independent[j];assert other.converged
            error=float(np.max(np.abs(other.probabilities-p[:,j])))
            assert error<=2e-7,(ell,r,error)
            report['independent_difference']=error
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    return p,reports

def task(scenario,rep,c,design,ext,land,data_name,fp):
    started=time.monotonic();data=Path(data_name);identity=f'{scenario}__rep{rep:03d}'
    baseline=FINAL/c['baseline_data']/'trajectories'/f'{identity}.csv'
    meta=json.loads(baseline.with_suffix('.json').read_text());assert meta['sha256']==sha(baseline)
    old=pd.read_csv(baseline,float_precision='round_trip')
    old=old[old.H_target.isin(c['h_targets'])&old.parameter_id.isin(design.parameter_id)]
    assert len(old)==len(c['h_targets'])*len(design)
    assert old.q.eq(0).all()
    selected=land[(land.scenario_id==scenario)&(land.replicate==rep)]
    hashes={str(baseline.relative_to(FINAL)):sha(baseline)};outputs=[]
    for row in selected.sort_values('H_target',ascending=False).itertuples(index=False):
        path=data/'snapshots'/f'{identity}__h{row.H_target:.3f}.csv'
        marker=path.with_suffix('.json')
        if marker.exists():
            receipt=json.loads(marker.read_text());assert receipt['fingerprint']==fp and sha(path)==receipt['csv_sha256']
            outputs.append(pd.read_csv(path));continue
        prior=old[old.H_target.eq(row.H_target)].set_index('parameter_id').loc[design.parameter_id]
        assert prior.grid_sha256.eq(row.grid_sha256).all() and prior.n_patches.eq(row.n_clusters).all()
        records=[];vectors=[];offsets=[0];geometry_hash=None
        items=design.copy();items['sample']='matched'
        if row.H_target in c['extension_h_targets']:
            extension=ext.copy();extension['sample']='extension';items=pd.concat([items,extension],ignore_index=True)
        if row.n_clusters>=2:
            geo=FINAL/c['geometry_cache']/'geometry'/f'{identity}__h{row.H_target:.3f}.npz'
            gm=json.loads(geo.with_suffix('.json').read_text());assert sha(geo)==gm['sha256'];geometry_hash=sha(geo)
            with np.load(geo) as z:
                a,d=z['areas'],z['distances'];assert str(z['grid_sha256'])==row.grid_sha256
            assert len(a)==row.n_clusters and a.sum()==round(40000*row.H_target)
            assert np.isfinite(d).all() and (d>=0).all() and np.array_equal(d,d.T)
            results={}
            for ell,group in items.groupby('half_distance',sort=True):
                p,reports=solve_group(a,d,group,c)
                for j,(index,item) in enumerate(group.iterrows()):results[index]=(p[:,j],reports[j])
        for index,item in items.iterrows():
            if row.n_clusters>=2:
                p,report=results[index];occupancy=float(p.mean());status='valid'
            else:p=np.empty(0);report={};occupancy=np.nan;status='excluded_fewer_than_two_patches'
            before=float(prior.loc[item.parameter_id].occupancy) if item['sample']=='matched' else np.nan
            if item['sample']=='matched' and status=='valid':assert occupancy>=before-2e-7
            records.append(dict(scenario_id=scenario,replicate=rep,H_target=row.H_target,seed=row.seed,
                grid_sha256=row.grid_sha256,n_patches=row.n_clusters,parameter_id=item.parameter_id,
                r=item.r,half_distance=item.half_distance,q=1,sample=item['sample'],occupancy=occupancy,
                baseline_occupancy=before,status=status,**report))
            vectors.append(p);offsets.append(offsets[-1]+len(p))
        frame=pd.DataFrame(records);frame.to_csv(path,index=False)
        np.savez_compressed(path.with_suffix('.npz'),probabilities=np.concatenate(vectors),offsets=offsets)
        save(marker,dict(fingerprint=fp,csv_sha256=sha(path),vectors_sha256=sha(path.with_suffix('.npz')),
            geometry_sha256=geometry_hash,grid_sha256=row.grid_sha256))
        outputs.append(frame)
    full=pd.concat(outputs,ignore_index=True)
    save(data/f'{identity}.json',dict(fingerprint=fp,baseline_sha256=hashes,n_rows=len(full),elapsed_seconds=time.monotonic()-started))
    print(identity,len(full),round(time.monotonic()-started,1),'seconds',flush=True)
    return len(full)

def summaries(c,design,land,out,data):
    full=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted((data/'snapshots').glob('*.csv'))],ignore_index=True)
    keys=['scenario_id','replicate','H_target','parameter_id']
    assert not full.duplicated(keys).any()
    matched=full[full['sample'].eq('matched')].copy()
    assert len(matched)==len(land)*len(design)
    base=matched.copy();base['q']=0;base['occupancy']=base.baseline_occupancy
    both=pd.concat([base,matched],ignore_index=True)
    summary=both.groupby(['q','parameter_id','scenario_id','H_target']).occupancy.agg(n_valid='count',median='median',q05=lambda x:x.quantile(.05),q95=lambda x:x.quantile(.95)).reset_index()
    summary.loc[summary.n_valid<c['minimum_valid_seeds'],['median','q05','q95']]=np.nan
    reference=summary[summary.scenario_id.eq('only_random_rho0')][['q','parameter_id','H_target','median']].rename(columns={'median':'baseline_median'})
    summary=summary.merge(reference,on=['q','parameter_id','H_target'],validate='many_to_one')
    summary['contrast']=summary['median']-summary.baseline_median
    summary.to_csv(out/'tables/13_B_C_medians_contrasts.csv',index=False)
    summary[summary.parameter_id.eq(c['reference'])].to_csv(out/'tables/13_B_reference.csv',index=False)
    associations=[]
    for (q,scenario,h),g in summary[~summary.scenario_id.eq('only_random_rho0')].groupby(['q','scenario_id','H_target']):
        g=g.merge(design[['parameter_id','r','half_distance']],on='parameter_id',validate='one_to_one')
        for parameter in ['r','half_distance']:
            valid=g.contrast.notna();v=g.loc[valid,'contrast']
            coefficient=float(spearmanr(np.log10(g.loc[valid,parameter]),v).statistic) if len(v)>=3 and np.ptp(v)>1e-10 else np.nan
            associations.append(dict(q=q,scenario_id=scenario,H_target=h,parameter=parameter,coefficient=coefficient,n_profiles=int(valid.sum())))
    pd.DataFrame(associations).to_csv(out/'tables/13_C_parameter_associations.csv',index=False)
    joined=both.merge(land[['scenario_id','replicate','H_target']+METRICS],on=['scenario_id','replicate','H_target'],validate='many_to_one')
    records=[]
    for (q,pid,h),g in joined.groupby(['q','parameter_id','H_target']):
        assert g.scenario_id.nunique()==7
        for metric in METRICS:
            coefficient=float(balanced_spearman(g[metric],g.occupancy,g.scenario_id,
                minimum_per_group=c['minimum_valid_seeds']))
            records.append(dict(q=q,parameter_id=pid,H_target=h,metric=metric,coefficient=coefficient))
    dd=pd.DataFrame(records);dd.to_csv(out/'tables/13_D_metric_associations.csv',index=False)
    ds=dd.groupby(['q','parameter_id','metric']).coefficient.agg(n_defined='count',median_signed='median',median_absolute=lambda x:x.abs().median()).reset_index()
    ds.to_csv(out/'tables/13_D_profile_summary.csv',index=False)
    extension=full[full['sample'].eq('extension')].groupby(['scenario_id','H_target','half_distance','r']).occupancy.agg(n_valid='count',median='median').reset_index()
    extension.loc[extension.n_valid<c['minimum_valid_seeds'],'median']=np.nan
    extension.to_csv(out/'tables/13_extension_survey.csv',index=False)
    coverage=[]
    for q,g in summary.groupby('q'):
        med=g['median'].dropna();coverage.append(dict(q=int(q),n_cells=len(med),floor=float((med<=.05).mean()),intermediate=float(((med>.05)&(med<.95)).mean()),ceiling=float((med>=.95).mean())))
    change=matched.occupancy-matched.baseline_occupancy
    b=summary[summary.parameter_id.eq(c['reference'])].pivot(index=['scenario_id','H_target'],columns='q',values='contrast')
    dp=dd.pivot(index=['parameter_id','H_target','metric'],columns='q',values='coefficient').dropna()
    audit=dict(status='numerically_checked_pilot',n_landscapes=len(land),n_profiles=len(design),n_matched_response_pairs=len(matched),
        n_excluded_landscapes=int((land.n_clusters<2).sum()),n_valid_pairs=int(change.notna().sum()),
        maximum_residual=float(full.fixed_point_residual.max()),
        n_independent_checks=int(full.independent_difference.notna().sum()),maximum_independent_difference=float(full.independent_difference.max()),
        occupancy_increase_median=float(change.median()),occupancy_increase_maximum=float(change.max()),
        response_coverage=coverage,B_maximum_contrast_change=float((b[1]-b[0]).abs().max()),
        D_n_paired_coefficients=len(dp),D_median_absolute_change=float((dp[1]-dp[0]).abs().median()),
        D_maximum_absolute_change=float((dp[1]-dp[0]).abs().max()),
        D_opposite_sign_count=int(((dp[0]*dp[1]<0)&(dp[0].abs()>.1)&(dp[1].abs()>.1)).sum()),
        limitations='Three seeds; systematic 65-profile subset; no confidence intervals; equal scenario mass; same multi-patch exclusions; extension is separate and provisional.')
    save(out/'logs/13_numeric_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--stage',choices=['preflight','compute','summary'],default='compute')
    args=parser.parse_args();c,design,ext,land,out,data,fp=inputs(args.config)
    if args.stage=='preflight':return
    if args.stage=='compute':
        jobs=[(s,r) for s in land.scenario_id.unique() for r in c['replicates']]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(task,s,r,c,design,ext,land,str(data),fp) for s,r in jobs]
            for f in as_completed(futures):f.result()
    summaries(c,design,land,out,data)

if __name__=='__main__':main()
