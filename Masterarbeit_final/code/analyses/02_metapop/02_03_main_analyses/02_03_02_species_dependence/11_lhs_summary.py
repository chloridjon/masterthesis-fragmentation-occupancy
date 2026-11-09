"""Complete-run summaries and matched-seed stability, separate from inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


def summary_arrays(values, minimum):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        count=np.isfinite(values).sum(axis=0)
        median=np.nanmedian(values,axis=0)
        q05=np.nanquantile(values,.05,axis=0)
        q95=np.nanquantile(values,.95,axis=0)
    median[count<minimum]=np.nan
    q05[count<minimum]=np.nan;q95[count<minimum]=np.nan
    return count,median,q05,q95


def finish(c,data,results,fp):
    final=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
    lc=json.loads((final/c['landscape_config']).read_text())
    lhs=pd.read_csv(final/c['lhs_design'])
    scenarios=[s['scenario_id'] for s in lc['scenarios']]
    hs=c['h_targets'];reps=c['replicates'];ids=lhs.parameter_id.tolist()
    assert len(reps)==20 and len(scenarios)==7 and len(ids)==128
    frames=[]
    for rep in reps:
        for scenario in scenarios:
            path=data/'trajectories'/f'{scenario}__rep{rep:03d}.csv'
            if not path.exists(): raise RuntimeError(f'Missing complete trajectory: {path}')
            part=pd.read_csv(path)
            assert len(part)==len(hs)*128 and not part.duplicated(['H_target','parameter_id']).any()
            assert part.producer_fingerprint.eq(fp).all()
            assert part.scenario_id.eq(scenario).all() and part.replicate.eq(rep).all()
            assert set(part.H_target)==set(hs) and set(part.parameter_id)==set(ids)
            if part.status.eq('nonconverged').any():
                raise RuntimeError(f'Unconverged equilibrium retained in {path}; summaries withheld.')
            frames.append(part)
    table=pd.concat(frames,ignore_index=True)
    expected=20*7*len(hs)*128
    assert len(table)==expected
    assert not table.duplicated(['scenario_id','replicate','H_target','parameter_id']).any()
    valid=table.status.eq('valid')
    assert table.loc[valid,'occupancy'].between(0,1).all()
    assert table.loc[~valid,'occupancy'].isna().all()
    assert table.loc[valid,'n_patches'].ge(2).all()
    assert table.loc[~valid,'n_patches'].lt(2).all()
    assert table.loc[valid,'fixed_point_residual'].max()<=c['solver']['tol']*1.1
    tmp=data/'11_lhs_occupancy.csv.tmp';table.to_csv(tmp,index=False);tmp.replace(tmp.with_suffix(''))
    # Every cell retains its full matched-seed axis, including missing responses.
    mappings=[{v:i for i,v in enumerate(axis)} for axis in (reps,scenarios,hs,ids)]
    indices=tuple(table[col].map(mapping).to_numpy() for col,mapping in
                  zip(['replicate','scenario_id','H_target','parameter_id'],mappings))
    values=np.full((20,7,len(hs),128),np.nan)
    values[indices]=table.occupancy.to_numpy()
    minimum=c['summary']['minimum_valid_seeds']
    count,median,q05,q95=summary_arrays(values,minimum)
    keys=pd.MultiIndex.from_product([scenarios,hs,ids],names=['scenario_id','H_target','parameter_id']).to_frame(index=False)
    keys['n_valid']=count.ravel();keys['n_excluded']=20-count.ravel()
    keys['median']=median.ravel();keys['q05']=q05.ravel();keys['q95']=q95.ravel()
    contrast=median-median[0][None,:,:]
    keys['contrast_vs_only_random']=contrast.ravel()
    keys['n_paired_valid']=((np.isfinite(values)) & np.isfinite(values[:,0:1])).sum(axis=0).ravel()
    keys.to_csv(results/'tables/11_lhs_response_summaries.csv',index=False)
    landscapes=table.drop_duplicates(['scenario_id','replicate','H_target'])
    exclusions=landscapes.assign(excluded=landscapes.n_patches.lt(2)).groupby(['scenario_id','H_target']).agg(
        n_landscapes=('replicate','size'),n_excluded=('excluded','sum'))
    exclusions['excluded_fraction']=exclusions.n_excluded/exclusions.n_landscapes
    exclusions.to_csv(results/'tables/11_analysis_domain_exclusions.csv')
    informative=np.isfinite(median) & (median>c['summary']['floor']) & (median<c['summary']['ceiling'])
    informative_contrast=np.isfinite(contrast) & (np.abs(contrast)>c['summary']['informative_contrast'])
    stability=[];cell_changes=[]
    def compare(label,estimate,delta):
        changes={'median':np.abs(estimate[1]-median),'q05':np.abs(estimate[2]-q05),
                 'q95':np.abs(estimate[3]-q95),'contrast':np.abs(delta-contrast)}
        for name,diff in changes.items():
            relevant=informative_contrast if name=='contrast' else informative
            allowed=c['summary']['contrast_tolerance' if name=='contrast' else 'occupancy_tolerance']
            finite=np.isfinite(diff)
            relevant_finite=finite & relevant
            n=int(finite.sum());nr=int(relevant_finite.sum())
            stability.append(dict(check=label,statistic=name,n_comparable_cells=n,
                n_final_cells_unavailable_in_check=int((np.isfinite(contrast if name=='contrast' else median) & ~finite).sum()),
                n_informative_comparable_cells=nr,maximum_change=float(diff[finite].max()) if n else np.nan,
                fraction_exceeding_tolerance=float((diff[finite]>allowed).mean()) if n else np.nan,
                informative_fraction_exceeding_tolerance=float((diff[relevant_finite]>allowed).mean()) if nr else np.nan,
                tolerance=allowed))
            if label.startswith('first_'):
                part=keys[['scenario_id','H_target','parameter_id']].copy()
                part['check']=label;part['statistic']=name;part['absolute_change']=diff.ravel()
                part['informative_final_cell']=relevant.ravel();cell_changes.append(part)
    for n in c['summary']['checkpoints']:
        if n==20: continue
        sub=summary_arrays(values[:n],minimum)
        compare(f'first_{n}_vs_20',sub,sub[1]-sub[1][0][None,:,:])
    for i,rep in enumerate(reps):
        sub=summary_arrays(np.delete(values,i,axis=0),minimum)
        compare(f'omit_seed_block_{rep}',sub,sub[1]-sub[1][0][None,:,:])
    pd.DataFrame(stability).to_csv(results/'tables/11_seed_stability.csv',index=False)
    pd.concat(cell_changes,ignore_index=True).to_csv(results/'tables/11_seed_extension_cell_changes.csv',index=False)
    stable=pd.DataFrame(stability)
    important=stable[(stable.check.eq('first_15_vs_20')|stable.check.str.startswith('omit_seed_block_'))]
    adequate=bool(len(important) and important.n_informative_comparable_cells.gt(0).all()
                  and important.informative_fraction_exceeding_tolerance.le(c['summary']['maximum_sensitive_fraction']).all()
                  and important.n_final_cells_unavailable_in_check.eq(0).all())
    status=dict(status='completed_independent_vector_audit_pending',n_response_rows=len(table),
        n_valid_responses=int(valid.sum()),n_excluded_responses=int((~valid).sum()),
        n_landscapes=len(landscapes),n_h_levels=len(hs),n_seed_blocks=20,n_parameters=128,
        producer_fingerprint=fp,maximum_fixed_point_residual=float(table.fixed_point_residual.max()),
        n_independent_solver_checks=int(table.independent_probability_difference.notna().sum()),
        maximum_independent_solver_difference=float(table.independent_probability_difference.max()),
        provisional_20_seed_stability_passed=adequate,
        uncertainty_scope='q05/q95 are landscape variation; no inferential bootstrap intervals yet',
        reference_profile_adopted=False,
        summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (results/'logs/11_run_status.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    print(json.dumps(status,indent=2),flush=True)
