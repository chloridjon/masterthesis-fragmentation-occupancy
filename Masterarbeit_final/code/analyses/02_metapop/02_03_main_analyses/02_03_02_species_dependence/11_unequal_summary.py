"""Audit8/16 replication with explicit structural absence and cohort diagnostics."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,hashlib,importlib,json,os,sys,warnings
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
E=importlib.import_module('11_unequal_replication');M=E.M;R=E.R;V=E.V
S=importlib.import_module('11_confirmed_domain_summary')
from src.tools.final_plot_style import FOCAL_SCENARIOS

def stats(values,planned,minimum=3):
    assert values.shape[:2]==planned.shape and not np.isfinite(values[~planned]).any()
    count,q=S.statistics(values,minimum)
    exclusions=planned.sum(axis=0)[:,None]-count
    assert (exclusions>=0).all()
    return count,q,exclusions

def contrasts(q):return q[1]-q[1,0:1]

def independent_quantiles(frame,hs,parameter_ids,minimum=3):
    index=pd.MultiIndex.from_product([hs,parameter_ids],names=['H_target','parameter_id'])
    groups=frame.groupby(['H_target','parameter_id']).occupancy
    shape=(len(hs),len(parameter_ids))
    count=groups.count().reindex(index).to_numpy().reshape(shape)
    # pandas may return a read-only view; masking low-count cells needs owned storage.
    q=groups.quantile([.05,.5,.95]).unstack().reindex(index).to_numpy(copy=True).T.reshape((3,)+shape)
    q[:,count<minimum]=np.nan
    return count,q

def source_hashes(sid,rep,data):
    result={}
    p=data/'trajectories'/f'{sid}__rep{rep:03d}.csv'
    result[str(p.relative_to(FINAL))]=R.sha(p);result[str(p.with_suffix('.json').relative_to(FINAL))]=R.sha(p.with_suffix('.json'))
    for meta in (data/'snapshots').glob(f'{sid}__rep{rep:03d}__h*.json'):
        a=json.loads(meta.read_text());result[str(meta.relative_to(FINAL))]=R.sha(meta)
        for suffix,h in a['hashes'].items():result[str(meta.with_suffix(suffix).relative_to(FINAL))]=h
    return result

def audited(sid,rep,ctx,retained,out):
    identity=f'{sid}__rep{rep:03d}';cache=out/'logs/vector_audits'/f'{identity}.json';source=R.sha(Path(__file__))
    c,base,lc,design,bench,geo,gfp,data,oldout,fp=ctx
    if cache.exists():
        a=json.loads(cache.read_text());assert a['audit_source_sha256']==source and a['scientific_producer_fingerprint']==fp
        for name,h in a['input_sha256'].items():assert R.sha(FINAL/name)==h,name
        return pd.read_csv(data/'trajectories'/f'{identity}.csv',float_precision='round_trip'),a
    if rep<=5:
        a=retained[identity].copy();file=data/'trajectories'/f'{identity}.csv';assert R.sha(file)==a['trajectory_sha256']
        full=pd.read_csv(file,float_precision='round_trip')
        reference=bench[bench.scenario_id.eq(sid)&bench.replicate.eq(rep)]
        for h in base['h_targets']:
            land=reference[reference.H_target.eq(h)].iloc[0];p=data/'snapshots'/f'{identity}__h{h:.3f}.csv'
            table=V.snapshot_cache(p,fp,land.grid_sha256,design.parameter_id.tolist());assert table is not None
            pd.testing.assert_frame_equal(full[full.H_target.eq(h)].reset_index(drop=True),table.reindex(columns=full.columns),check_exact=True,check_dtype=False)
        mode='retained_full_numerical_audit_with_revalidated_hashes_offsets_means'
    else:
        full,a=S.audit_trajectory(sid,rep,ctx);mode='new_independent_full_vector_residual_identity_monotonicity_audit'
    a.update(audit_mode=mode,audit_source_sha256=source,canonical_audit_source_sha256=R.sha(Path(S.__file__)),scientific_producer_fingerprint=fp,input_sha256=source_hashes(sid,rep,data),completed_at_utc=datetime.now(timezone.utc).isoformat())
    R.save_json(cache,a);return full,a

def comparison_rows(oldq,newq,oldcounts,newcounts,scenarios,hs,label):
    oldc,newc=contrasts(oldq),contrasts(newq)
    rows=[]
    for j,s in enumerate(scenarios):
        dm=np.abs(newq[1,j]-oldq[1,j]);dc=np.abs(newc[j]-oldc[j]);finite=np.isfinite(dm);fc=np.isfinite(dc)
        rows.append(dict(comparison=label,scenario_id=s['scenario_id'],maximum_median_change=float(np.nanmax(dm)),maximum_q05_change=float(np.nanmax(abs(newq[0,j]-oldq[0,j]))),maximum_q95_change=float(np.nanmax(abs(newq[2,j]-oldq[2,j]))),maximum_contrast_change=float(np.nanmax(dc)) if j else np.nan,median_fraction_above_05=float((dm[finite]>.05).mean()),contrast_fraction_above_05=float((dc[fc]>.05).mean()) if j else np.nan,median_cells_newly_evaluable=int((~np.isfinite(oldq[1,j])&np.isfinite(newq[1,j])).sum()),median_cells_losing_evaluability=int((np.isfinite(oldq[1,j])&~np.isfinite(newq[1,j])).sum())))
    return rows

def run(workers):
    plan=json.loads(E.CONFIG.read_text());ctx=M.inputs();c,base,lc,design,bench,geo,gfp,data,oldout,fp=ctx
    assert fp==plan['scientific_producer_fingerprint'] and plan['scenario_seed_targets']==E.targets(lc['scenarios'])
    extension=FINAL/plan['results_directory'];out=extension/'summary_008_016'
    for p in [out/'tables',out/'logs/vector_audits']:p.mkdir(parents=True,exist_ok=True)
    marker=out/'11_numerical_audit.json'
    if marker.exists():
        a=json.loads(marker.read_text())
        for name,h in {**a['input_sha256'],**a['output_sha256']}.items():assert R.sha(FINAL/name)==h,name
        assert a['summary_source_sha256']==R.sha(Path(__file__));print('Previously audited8/16 summary retained.');return
    previous=json.loads((FINAL/plan['retained_five_seed_audit']).read_text());assert previous['visual_review_confirmed']
    assert previous['summary_source_sha256']==R.sha(Path(S.__file__)),'Changed retained-audit source; revalidation required'
    for name,h in {**previous['input_sha256'],**previous['output_sha256']}.items():assert R.sha(FINAL/name)==h,name
    prior=FINAL/plan['retained_five_seed_audit'];retained_table=pd.read_csv(prior.parent/'tables/11_vector_audit_by_trajectory.csv')
    retained={r['trajectory']:r for r in retained_table.to_dict('records')};scenarios=list(FOCAL_SCENARIOS);ids=[s['scenario_id'] for s in scenarios];hs=base['h_targets'];planned=E.planned_mask(scenarios)
    values=np.full((16,7,88,256),np.nan);records=[];inputs={str((data/'manifest.json').relative_to(FINAL)):R.sha(data/'manifest.json'),str(E.CONFIG.relative_to(FINAL)):R.sha(E.CONFIG),str(prior.relative_to(FINAL)):R.sha(prior)}
    keys=[(sid,rep) for sid,n in E.targets(scenarios).items() for rep in range(1,n+1)]
    assert len(keys)==104
    for sid,rep in keys:assert (data/'trajectories'/f'{sid}__rep{rep:03d}.json').exists()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(audited,sid,rep,ctx,retained,out):(sid,rep) for sid,rep in keys}
        for f in as_completed(tasks):
            full,a=f.result();sid,rep=tasks[f];values[rep-1,ids.index(sid)]=full.pivot(index='H_target',columns='parameter_id',values='occupancy').reindex(index=hs,columns=design.parameter_id).to_numpy()
            inputs.update(a['input_sha256']);records.append({k:v for k,v in a.items() if k!='input_sha256'})
            print(f'Audited {len(records)}/104: {sid} rep{rep:03d}',flush=True)
    count,q,excluded=stats(values,planned);med=q[1];delta=contrasts(q);paired=(np.isfinite(values[:8])&np.isfinite(values[:8,0:1])).sum(axis=0)
    rows=[]
    for j,sid in enumerate(ids):
        for k,h in enumerate(hs):
            for i,item in design.iterrows():rows.append(dict(scenario_id=sid,H_target=h,parameter_id=item.parameter_id,n_planned=int(planned[:,j].sum()),n_valid=int(count[j,k]),n_excluded=int(excluded[j,k]),n_structurally_absent=int((~planned[:,j]).sum()),n_paired_valid=int(paired[j,k,i]),q05=q[0,j,k,i],median=med[j,k,i],q95=q[2,j,k,i],contrast_vs_only_random=delta[j,k,i]))
    summaries=pd.DataFrame(rows);summaries.to_csv(out/'tables/11_response_summaries.csv',index=False)
    # Independent pandas grouping from the immutable saved trajectory rows, all cells.
    error=0.;independent_medians=[]
    for j,sid in enumerate(ids):
        frame=pd.concat([pd.read_csv(data/'trajectories'/f'{sid}__rep{rep:03d}.csv',usecols=['H_target','parameter_id','occupancy'],float_precision='round_trip') for rep in range(1,E.targets(scenarios)[sid]+1)],ignore_index=True)
        co,expected=independent_quantiles(frame,hs,design.parameter_id)
        assert np.array_equal(co,np.repeat(count[j,:,None],256,axis=1));assert np.array_equal(np.isnan(expected),np.isnan(q[:,j]))
        finite=np.isfinite(expected);error=max(error,float(np.abs(expected[finite]-q[:,j][finite]).max()));independent_medians.append(expected[1])
    assert error<1e-14
    ind=np.array(independent_medians);np.testing.assert_allclose(ind-ind[0:1],delta,atol=1e-14,rtol=0,equal_nan=True)
    # Omit complete seed clusters, retaining the planned partially paired cohorts.
    stability=[]
    for omit in range(16):
        co,other,ex=stats(np.delete(values,omit,axis=0),np.delete(planned,omit,axis=0));d=contrasts(other)
        for j,sid in enumerate(ids):
            dm=abs(other[1,j]-med[j]);dc=abs(d[j]-delta[j]);finite=np.isfinite(dm);fc=np.isfinite(dc)
            informative=finite&(med[j]>.05)&(med[j]<.95);informative_c=fc&(abs(delta[j])>.01)
            stability.append(dict(omitted_seed=omit+1,cohort='shared_1_to_8' if omit<8 else 'directed_only_9_to_16',scenario_id=sid,scenario_seed_removed=bool(planned[omit,j]),maximum_median_change=float(np.nanmax(dm)),maximum_q05_change=float(np.nanmax(abs(other[0,j]-q[0,j]))),maximum_q95_change=float(np.nanmax(abs(other[2,j]-q[2,j]))),maximum_contrast_change=float(np.nanmax(dc)) if j else np.nan,median_fraction_above_05=float((dm[finite]>.05).mean()),contrast_fraction_above_05=float((dc[fc]>.05).mean()) if j else np.nan,informative_median_fraction_above_05=float((dm[informative]>.05).mean()) if informative.any() else np.nan,informative_contrast_fraction_above_05=float((dc[informative_c]>.05).mean()) if j and informative_c.any() else np.nan,median_cells_losing_evaluability=int((np.isfinite(med[j])&~np.isfinite(other[1,j])).sum())))
    st=pd.DataFrame(stability);st.to_csv(out/'tables/11_cohort_seed_omission.csv',index=False)
    co5,q5,ex5=stats(values[:5],planned[:5]);co8,q8,ex8=stats(values[:8],planned[:8])
    expansion=comparison_rows(q5,q,co5,count,scenarios,hs,'initial5_to_8_16')+comparison_rows(q8,q,co8,count,scenarios,hs,'shared8_to_8_16')
    pd.DataFrame(expansion).to_csv(out/'tables/11_replication_expansion.csv',index=False)
    pd.DataFrame(records).to_csv(out/'tables/11_vector_audit_by_trajectory.csv',index=False)
    pd.DataFrame([dict(scenario_id=sid,H_target=h,n_planned=int(planned[:,j].sum()),n_valid=int(count[j,k]),n_excluded=int(excluded[j,k]),n_structurally_absent=int((~planned[:,j]).sum())) for j,sid in enumerate(ids) for k,h in enumerate(hs)]).to_csv(out/'tables/11_landscape_exclusions.csv',index=False)
    span=np.nanmax(med.reshape(-1,256),axis=0)-np.nanmin(med.reshape(-1,256),axis=0)
    table=design.copy();table['median_response_span']=span;table['min_landscape_q05']=np.nanmin(q[0].reshape(-1,256),axis=0);table['max_landscape_q95']=np.nanmax(q[2].reshape(-1,256),axis=0);table.to_csv(out/'tables/11_profile_response_diagnostics.csv',index=False)
    np.savez_compressed(out/'tables/11_response_arrays.npz',values=values,quantiles=q,valid_counts=count,planned_seed_mask=planned,n_planned_by_scenario=planned.sum(axis=0))
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);ensemble=np.nanquantile(med,[.05,.5,.95],axis=2)
    ens=pd.DataFrame([dict(scenario_id=sid,H_target=h,parameter_q05=ensemble[0,j,k],parameter_median=ensemble[1,j,k],parameter_q95=ensemble[2,j,k]) for j,sid in enumerate(ids) for k,h in enumerate(hs)]);ens.to_csv(out/'tables/11_parameter_ensemble_curves.csv',index=False)
    total=104*88*256;valid=sum(a['valid_vectors'] for a in records);missing=sum(a['exclusions'] for a in records);assert valid+missing==total
    note=f'''# Conditional-domain replication:8/16 seeds

The fixed 256-profile design was evaluated with eight Only-Random trajectories
and 16 trajectories in each of the six other scenarios, at all 88 habitat
levels. This is 104 complete trajectories and{total:,} response rows:
{valid:,} valid vectors and {missing:,} excluded landscape responses. The
initial five matched blocks are preserved. The model, tolerances, landscape
seeds, full kernels and parameter-design weights 1/256 remain unchanged.
Independent sampling units are whole trajectories, not individual snapshots.

Seeds 1–8 are shared across all seven scenarios. Seeds 9–16 supply additional
matched blocks across the six directed scenarios. The absent Only-Random
trajectories at these latter seeds were never planned; they are structural
absence, not ecological exclusions. Exclusions and valid counts therefore
use denominators 8 and 16, respectively. Contrasts are differences between
scenario medians at fixed H and use each scenario's full requested sample;
paired counts refer only to overlapping seeds. No pooling by the unequal
number of scenario replicates is used.

New trajectories received independent vector, offset, mean, raw-equation
residual, source, raster/seed/parameter identity and parameter-order audits.
The 35 original trajectories retain their previously completed full audits,
with source, cache hashes, offsets, vector means and saved-row identity
revalidated. These retained residual audits were not presented as newly
recomputed. All summary cells were independently reconstructed with pandas
from saved trajectory rows; maximum error{error:.3g}.

Seed-omission diagnostics separately identify the shared 1–8 and directed-only
9–16 cohorts. Initial 5-to-final and matched 8-to-final changes are reported
separately, including changing evaluability. Maximum omission median change
is {st.maximum_median_change.max():.3f}; maximum contrast change is
{st.maximum_contrast_change.max():.3f}. These are descriptive sensitivity
diagnostics, not confidence intervals or proof of sufficient replication.
Profiles with weak or saturated responses remain in the fixed design.

No bootstrap confidence intervals are calculated here. A future bootstrap
must resample the eight shared seed clusters together across seven scenarios,
and separately resample the eight additional directed seed clusters together
across six scenarios, keeping both cohort sizes fixed. The former balanced
16-seed procedure cannot be applied unchanged. Landscape 5th–95th percentiles
are replicate variation, while parameter-ensemble bands describe variation
across equally weighted parameter profiles. Neither is a confidence interval.
Undefined single-patch landscapes remain missing; high-H values describe the
remaining multi-patch subset. Curves along H jointly vary habitat amount
and configuration; pointwise contrasts compare scenarios at fixed H.

The previous five-seed figures and bird surfaces, and the separately audited
20-seed Result-B reference, remain preserved. No new biological calibration,
Result-D inference or manuscript edits are implied. Numerical summary is
complete; exports and actual visual review are separate required steps.
'''
    (out/'README.md').write_text(note)
    inputs[str(Path(__file__).relative_to(FINAL))]=R.sha(Path(__file__))
    files=[p for p in out.rglob('*') if p.is_file() and p!=marker]
    a=dict(status='independent_numeric_audit_passed_export_pending',n_parameters=256,n_trajectories=104,scenario_seed_targets=E.targets(scenarios),n_common_matched_seeds=8,n_additional_directed_seeds=8,n_response_rows=total,n_valid_vectors=valid,n_exclusions=missing,n_retained_numerical_trajectory_audits=35,n_new_full_trajectory_audits=69,maximum_original_equation_residual=max(a['maximum_residual'] for a in records),maximum_parameter_monotonicity_violation=max(a['maximum_parameter_monotonicity_violation'] for a in records),maximum_independent_summary_error=error,maximum_seed_omission_median=float(st.maximum_median_change.max()),maximum_seed_omission_contrast=float(st.maximum_contrast_change.max()),bootstrap_intervals_calculated=False,scientific_producer_fingerprint=fp,summary_source_sha256=R.sha(Path(__file__)),input_sha256=inputs,output_sha256={str(p.relative_to(FINAL)):R.sha(p) for p in files},completed_at_utc=datetime.now(timezone.utc).isoformat())
    R.save_json(marker,a);print(json.dumps({k:v for k,v in a.items() if k not in ['input_sha256','output_sha256']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=6);a=p.parse_args();run(a.workers)
