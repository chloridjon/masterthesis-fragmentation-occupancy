"""Audit complete LHS vectors, provenance and independently recomputed summaries."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
REL=Path('02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds')
DATA=FINAL/'data/processed'/REL
RESULTS=FINAL/'results'/REL


def check_supplementary_records(table,c,fp,records):
    """Accept tighter references only for individually registered source/trajectory pairs."""
    registered={}
    for repair,source in records:
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest==repair['supplementary_verification_source_sha256']
        assert repair['status']=='trajectory_resumed_and_verified'
        assert repair['original_primary_producer_fingerprint']==fp
        assert repair['primary_solver_and_tolerance_unchanged']
        assert repair['independent_reference_tolerance']==1e-12<c['solver']['tol']
        assert repair['independent_probability_comparison_tolerance_unchanged']==c['solver']['independent_probability_tolerance']
        registered.setdefault(digest,set()).add(repair['trajectory'])
    if 'supplementary_verification_source_sha256' not in table:
        assert not records
        return 0
    supplemented=table[table.supplementary_verification_source_sha256.notna()]
    for digest,part in supplemented.groupby('supplementary_verification_source_sha256'):
        assert digest in registered, 'Unregistered supplementary source'
        assert set(part.trajectory_id)<=registered[digest], 'Source applied to an unregistered trajectory'
        assert part.independent_reference_tolerance.eq(1e-12).all()
        assert part.independent_probability_difference.dropna().le(c['solver']['independent_probability_tolerance']).all()
    return len(supplemented)


def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for relative,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((FINAL/relative).read_bytes()).hexdigest()==digest,relative
    fp=manifest['producer_fingerprint'];c=manifest['configuration']
    table=pd.read_csv(DATA/'11_lhs_occupancy.csv')
    expected=128*7*20*len(c['h_targets'])
    assert len(table)==expected and table.producer_fingerprint.eq(fp).all()
    records=[]
    repair_path=RESULTS/'logs/11_supplementary_verification.json'
    if repair_path.exists():
        repair=json.loads(repair_path.read_text())
        source=Path(__file__).with_name('11_resume_verified_trajectory.py')
        records.append((repair,source))
    for path in sorted((RESULTS/'logs/targeted_repairs').glob('*__verification.json')):
        repair=json.loads(path.read_text())
        source=Path(__file__).with_name('11_targeted_lhs_repair.py')
        diagnosis_path=path.with_name(repair['trajectory']+'__diagnosis.json')
        assert repair['diagnosis_path']==str(diagnosis_path.relative_to(FINAL))
        assert hashlib.sha256(diagnosis_path.read_bytes()).hexdigest()==repair['diagnosis_sha256']
        diagnosis=json.loads(diagnosis_path.read_text())
        assert diagnosis['status']=='independent_reference_precision_rejection_verified'
        assert diagnosis['supplementary_verification_source_sha256']==repair['supplementary_verification_source_sha256']
        assert diagnosis['original_primary_producer_fingerprint']==fp
        assert diagnosis['primary_solver_and_tolerance_unchanged']
        assert diagnosis['original_difference']>c['solver']['independent_probability_tolerance']
        assert diagnosis['tighter_difference']<=c['solver']['independent_probability_tolerance']
        assert diagnosis['primary_residual']<=1.1*c['solver']['tol']
        fixture=path.with_name(repair['trajectory']+'__fixture.npz')
        assert hashlib.sha256(fixture.read_bytes()).hexdigest()==diagnosis['fixture_sha256']
        records.append((repair,source))
    supplementary_rows=check_supplementary_records(table,c,fp,records)
    assert not table.duplicated(['scenario_id','replicate','H_target','parameter_id']).any()
    csvs=sorted((DATA/'snapshots').glob('*.csv'))
    assert len(csvs)==7*20*len(c['h_targets'])
    valid_rows=0;excluded_rows=0;maximum_mean_error=0.;n_pilot_checks=0
    for i,path in enumerate(csvs,1):
        marker=json.loads(path.with_suffix('.json').read_text())
        assert marker['producer_fingerprint']==fp
        for saved in (path,path.with_suffix('.npz')):
            assert hashlib.sha256(saved.read_bytes()).hexdigest()==marker['hashes'][saved.suffix]
        part=pd.read_csv(path)
        assert len(part)==128 and part.parameter_id.nunique()==128
        assert part.n_patches.nunique()==1 and part.grid_sha256.eq(marker['grid_sha256']).all()
        n=int(part.n_patches.iloc[0])
        with np.load(path.with_suffix('.npz')) as z:
            probabilities=z['probabilities'];offsets=z['offsets']
            assert len(offsets)==129
            assert np.isfinite(probabilities).all() and (probabilities>=0).all() and (probabilities<=1).all()
            if n>=2:
                assert len(probabilities)==n*128 and np.array_equal(offsets,np.arange(129)*n)
                assert part.status.eq('valid').all() and part.converged.all()
                means=probabilities.reshape(128,n).mean(axis=1)
                error=float(np.max(np.abs(means-part.occupancy.to_numpy())))
                assert error<1e-14;maximum_mean_error=max(maximum_mean_error,error)
                assert part.fixed_point_residual.max()<=c['solver']['tol']*1.1
                assert part.relative_rate_balance.max()<=c['solver']['tol']*1.1
                zero=part.solver.eq('capacity_verified_empty_equilibrium')
                assert (part.loc[zero,'r']>part.loc[zero,'capacity_upper']).all()
                valid_rows+=128
            else:
                assert len(probabilities)==0 and (offsets==0).all()
                assert part.status.eq('excluded_fewer_than_two_patches').all()
                assert part.occupancy.isna().all();excluded_rows+=128
        if i%1000==0: print(f'Audited {i}/{len(csvs)} saved snapshots.',flush=True)
    for path in (DATA/'progress').glob('*.json'):
        progress=json.loads(path.read_text())
        assert progress['completed_snapshots']==len(c['h_targets'])
        n_pilot_checks+=len(progress.get('pilot_geometry_checks',[]))
    keys=['scenario_id','H_target','parameter_id'];valid=table[table.status.eq('valid')]
    independent=valid.groupby(keys).occupancy.agg(['count','median'])
    independent=independent.join(valid.groupby(keys).occupancy.quantile([.05,.95]).unstack().rename(columns={.05:'q05',.95:'q95'}))
    stored=pd.read_csv(RESULTS/'tables/11_lhs_response_summaries.csv').set_index(keys)
    check=stored.join(independent,rsuffix='_independent')
    assert np.array_equal(check.n_valid,check['count'].fillna(0).astype(int))
    errors=[]
    for column in ['median','q05','q95']:
        expected_values=check[column+'_independent'].where(check.n_valid>=c['summary']['minimum_valid_seeds'])
        assert np.array_equal(check[column].isna(),expected_values.isna())
        difference=np.abs(check[column]-expected_values).dropna()
        errors.append(float(difference.max()))
        assert difference.max()<1e-14
    baseline=check.xs('only_random_rho0').median_independent
    for scenario in check.index.get_level_values(0).unique():
        part=check.xs(scenario)
        expected_contrast=part['median']-check.xs('only_random_rho0')['median']
        assert np.allclose(part.contrast_vs_only_random,expected_contrast,equal_nan=True,atol=1e-14,rtol=0)
    stability=pd.read_csv(RESULTS/'tables/11_seed_stability.csv')
    for label,subset in [('first_15_vs_20',valid[valid.replicate.le(15)]),
                         ('omit_seed_block_20',valid[valid.replicate.ne(20)])]:
        grouped=subset.groupby(keys).occupancy.agg(['count','median'])
        sub=grouped['median'].where(grouped['count']>=c['summary']['minimum_valid_seeds']).reindex(stored.index)
        diff=(sub-stored['median']).abs()
        row=stability[stability.check.eq(label)&stability.statistic.eq('median')].iloc[0]
        assert int(diff.notna().sum())==row.n_comparable_cells
        assert abs(diff.max()-row.maximum_change)<1e-14
    status_path=RESULTS/'logs/11_run_status.json';status=json.loads(status_path.read_text())
    summary_path=Path(__file__).with_name('11_lhs_summary.py')
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest()==status['summary_source_sha256']
    audit=dict(status='passed',n_vector_rows_checked=valid_rows,n_excluded_rows_checked=excluded_rows,
        n_snapshot_caches=len(csvs),n_response_rows=len(table),maximum_vector_mean_error=maximum_mean_error,
        maximum_independent_summary_error=max(errors),n_original_pilot_geometry_comparisons=n_pilot_checks,
        source_hashes_verified=True,producer_fingerprint=fp,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_supplementary_reference_rows_verified=supplementary_rows,
        registered_supplementary_trajectories=[r['trajectory'] for r,_ in records])
    (RESULTS/'logs/11_independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    status.update(status='completed_and_audited',independent_audit_status='passed')
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
