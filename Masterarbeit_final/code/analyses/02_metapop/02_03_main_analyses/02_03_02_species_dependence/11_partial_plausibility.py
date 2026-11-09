"""Read-only plausibility and conditional timing assessment of saved snapshots."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import heapq
import json
import time
import numpy as np
import pandas as pd
from src.landscape.generation import run_sequential_deforestation
from src.landscape.landscape import Landscape

FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
REL=Path('02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds')
DATA=FINAL/'data/processed'/REL
RESULTS=FINAL/'results'/REL


def verify_replayed_pilot_geometry(grid, expected_hash, areas, distances):
    """Check actual raster geometry rather than an overwritten progress receipt."""
    assert hashlib.sha256(grid.tobytes()).hexdigest()==expected_hash
    landscape=Landscape(grid)
    landscape.ensure_clusters(edge_depth=0,connectivity='4',boundary_mode='open',
                              distance_mode='voronoi_dt',voronoi_depth=1)
    a=np.asarray(landscape.cluster_areas,float)
    d=np.asarray(landscape.cluster_dist_matrix,float)
    np.testing.assert_array_equal(a,areas)
    np.testing.assert_allclose(d,distances,atol=1e-12,rtol=0)
    return a,d


def main():
    started=time.time();manifest=json.loads((DATA/'manifest.json').read_text());c=manifest['configuration']
    fp=manifest['producer_fingerprint']
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((FINAL/name).read_bytes()).hexdigest()==digest,name
    lhs=pd.read_csv(FINAL/c['lhs_design']).set_index('parameter_id')
    benchmark=pd.read_csv(FINAL/c['landscape_data']).set_index(['scenario_id','replicate','H_target'])
    lc=json.loads((FINAL/c['landscape_config']).read_text())
    scenarios=[s['scenario_id'] for s in lc['scenarios']]
    paths=[p for p in sorted((DATA/'snapshots').glob('*.csv')) if p.with_suffix('.json').exists()]
    frames=[];metas=[]
    for i,path in enumerate(paths,1):
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['producer_fingerprint']==fp
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['hashes']['.csv']
        t=pd.read_csv(path,float_precision='round_trip')
        assert len(t)==128 and set(t.parameter_id)==set(lhs.index)
        assert t.producer_fingerprint.eq(fp).all()
        assert t.grid_sha256.eq(meta['grid_sha256']).all()
        row=t.iloc[0];old=benchmark.loc[(row.scenario_id,row.replicate,row.H_target)]
        assert row.grid_sha256==old.grid_sha256 and row.n_patches==old.n_clusters
        assert t.seed.eq(lc['base_seed']+row.replicate-1).all()
        assert t.event_count.eq(old.event_count).all() and t.n_random_cumulative.eq(old.n_random_cumulative).all()
        assert (t.n_random_cumulative+t.n_directed_cumulative).eq(t.event_count).all()
        design=lhs.loc[t.parameter_id]
        assert np.allclose(t.log_r,design.log_r,atol=1e-13,rtol=0)
        assert np.allclose(t.r,np.exp(t.log_r),rtol=1e-14,atol=0)
        assert np.allclose(t.half_distance,design.half_distance,rtol=1e-14,atol=0)
        assert np.allclose(t.alpha,np.log(2)/t.half_distance,rtol=1e-13,atol=0)
        assert t.q.eq(0).all()
        frames.append(t);metas.append(meta)
        if i%1000==0:print(f'Checked {i}/{len(paths)} saved snapshot tables.',flush=True)
    table=pd.concat(frames,ignore_index=True)
    assert not table.duplicated(['scenario_id','replicate','H_target','parameter_id']).any()
    good=table.status.eq('valid');excluded=table.n_patches.lt(2)
    assert np.array_equal(excluded,table.status.eq('excluded_fewer_than_two_patches'))
    assert table.loc[excluded,'occupancy'].isna().all()
    assert table.loc[good,'occupancy'].between(0,1).all() and table.loc[good,'converged'].all()
    assert table.loc[good,'fixed_point_residual'].max()<=1.1e-10
    assert table.loc[good,'relative_rate_balance'].max()<=1.1e-10
    assert table.loc[good,'max_delta'].max()<=1.1e-10
    cov=table.solver.eq('capacity_verified_empty_equilibrium')
    assert (table.loc[cov,'r']>table.loc[cov,'capacity_upper']).all()
    assert table.loc[cov,'occupancy'].eq(0).all()
    # Maximal equilibria are ordered when one parameter combination has
    # both lower extinction ratio and longer dispersal on the same landscape.
    ids=lhs.index.tolist();rs=lhs.r.to_numpy();ells=lhs.half_distance.to_numpy()
    dominant=(rs[:,None]<=rs[None,:]) & (ells[:,None]>=ells[None,:])
    np.fill_diagonal(dominant,False)
    left,right=np.where(dominant)
    pivot=table[good].pivot(index=['scenario_id','replicate','H_target'],columns='parameter_id',values='occupancy').reindex(columns=ids)
    v=pivot.to_numpy()
    violation=v[:,right]-v[:,left]
    maximum_order_violation=float(np.nanmax(np.maximum(violation,0)))
    assert maximum_order_violation<=2e-7
    n_order_checks=int(np.isfinite(violation).sum())
    vector_indices=np.unique(np.linspace(0,len(paths)-1,min(256,len(paths))).astype(int))
    vector_rows=0;max_mean_error=0.
    for number,index in enumerate(vector_indices,1):
        path=paths[index];t=frames[index];n=int(t.n_patches.iloc[0]);npz=path.with_suffix('.npz')
        assert hashlib.sha256(npz.read_bytes()).hexdigest()==metas[index]['hashes']['.npz']
        with np.load(npz) as z:
            p=z['probabilities'];offsets=z['offsets']
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
            assert len(offsets)==129
            if n>=2:
                assert np.array_equal(offsets,np.arange(129)*n)
                means=p.reshape(128,n).mean(axis=1)
                error=float(np.max(np.abs(means-t.occupancy)))
                assert error<1e-14;max_mean_error=max(max_mean_error,error);vector_rows+=128
            else:assert len(p)==0 and (offsets==0).all()
        if number%64==0:print(f'Checked {number}/{len(vector_indices)} probability-vector caches.',flush=True)
    # Recompute the original unnormalised equilibrium equation on actual,
    # independently matched pilot geometries, including large Only-Random cases.
    model_checks=[];geometry_checks=[]
    for sid in scenarios:
        scenario=next(s for s in lc['scenarios'] if s['scenario_id']==sid)
        pars=lc['access_reference'];seed=lc['base_seed']
        replay=run_sequential_deforestation(np.ones((200,200),dtype=np.int8),np.random.default_rng(seed),
            rho=scenario['rho'],directed_mode=scenario['mode'],target_habitat_fraction=0.,
            generation_boundary=lc['generation_boundary'],eta=pars['eta'],k=pars['k'],
            lambda_decay=pars['lambda'],save_habitat_fractions=[.325,.5],seed=seed)
        grids={float(s.requested_habitat_fractions[0]):s.grid for s in replay.snapshots}
        for h in [.325,.5]:
            identity=f'{sid}__rep001__h{h:.3f}'
            path=DATA/'snapshots'/f'{identity}.csv'
            geo=FINAL/'data/processed/02_metapop/02_02_species_parameter_selection/limit_based/geometry'/f'{identity}.npz'
            if not path.exists() or not geo.exists():continue
            t=pd.read_csv(path,float_precision='round_trip')
            if t.n_patches.iloc[0]<2:continue
            with np.load(geo) as z:
                a=z['areas'];d=z['distances'];assert str(z['grid_sha256'])==t.grid_sha256.iloc[0]
                a,d=verify_replayed_pilot_geometry(grids[h],t.grid_sha256.iloc[0],a,d)
            assert len(a)==t.n_patches.iloc[0] and a.sum()==round(40000*h)
            geometry_checks.append(dict(scenario_id=sid,H=h,n_patches=len(a),
                grid_sha256=t.grid_sha256.iloc[0],pilot_geometry_sha256=hashlib.sha256(geo.read_bytes()).hexdigest()))
            with np.load(path.with_suffix('.npz')) as z:
                ps=z['probabilities'].reshape(128,len(a))
            choices=np.unique([int(t.occupancy.idxmin()),int(t.occupancy.idxmax()),int((t.occupancy-.5).abs().idxmin())])
            for index in choices:
                row=t.iloc[index];p=ps[index];k=np.exp(-row.alpha*d);np.fill_diagonal(k,0)
                source=k@(a*p);rate=row.r*a**(-.5)
                target=source/(source+rate)
                residual=float(np.max(np.abs(target-p)))
                assert residual<=1.1e-10
                model_checks.append(dict(scenario_id=sid,H=h,n_patches=len(a),parameter_id=row.parameter_id,
                                         occupancy=row.occupancy,recomputed_residual=residual))
    pd.DataFrame(model_checks).to_csv(RESULTS/'tables/11_partial_independent_model_checks.csv',index=False)
    pd.DataFrame(geometry_checks).to_csv(RESULTS/'tables/11_replayed_pilot_geometry_checks.csv',index=False)
    # Estimate the remaining six-worker schedule from scenario-specific
    # completed trajectories; observed H timing accounts for active progress.
    progress=[(p,json.loads(p.read_text())) for p in (DATA/'progress').glob('*.json')]
    durations={};profiles={}
    for sid in scenarios:
        complete=[x for p,x in progress if x['trajectory'].startswith(sid+'__') and x['completed_snapshots']==len(c['h_targets'])]
        durations[sid]=[x['elapsed_seconds'] for x in complete]
        if not complete:raise RuntimeError(f'Insufficient timing observations for {sid}')
        steps=[]
        for x in complete:
            identity=x['trajectory'];marks=[]
            for h in reversed(c['h_targets']):
                marks.append((DATA/'snapshots'/f'{identity}__h{h:.3f}.json').stat().st_mtime)
            steps.append(np.r_[0,np.diff(marks)])
        profiles[sid]=np.median(np.array(steps),axis=0)
    active=[];seen=set();now=time.time()
    for path,x in progress:
        seen.add(x['trajectory'])
        if x['completed_snapshots']<len(c['h_targets']):
            sid=x['trajectory'].rsplit('__rep',1)[0]
            steps=profiles[sid][x['completed_snapshots']:]
            remaining=float(steps.sum())
            if len(steps):remaining-=min(max(0,now-path.stat().st_mtime),.5*steps[0])
            active.append(max(0,remaining))
    queued=[sid for rep in c['replicates'] for sid in scenarios if f'{sid}__rep{rep:03d}' not in seen]
    med={s:float(np.median(durations[s])) for s in scenarios}
    def predict(factor):
        heap=[v*factor for v in active]
        while len(heap)<6:heap.append(0.)
        heapq.heapify(heap)
        for sid in queued:
            elapsed=heapq.heappop(heap);heapq.heappush(heap,elapsed+factor*med[sid])
        return max(heap)/3600+.25
    timing=pd.DataFrame([dict(scenario_id=s,n_completed=len(durations[s]),median_hours=med[s]/3600,
                             min_hours=min(durations[s])/3600,max_hours=max(durations[s])/3600) for s in scenarios])
    timing.to_csv(RESULTS/'tables/11_partial_runtime_by_scenario.csv',index=False)
    checked_categories=table[good].assign(near_zero=table.loc[good,'occupancy'].le(.05),
        intermediate=table.loc[good,'occupancy'].between(.05,.95,inclusive='neither'),near_one=table.loc[good,'occupancy'].ge(.95))
    fractions=checked_categories[['near_zero','intermediate','near_one']].mean().to_dict()
    report=dict(status='partial_checks_passed',as_of_utc=datetime.now(timezone.utc).isoformat(),
        n_snapshots_checked=len(paths),n_response_rows_checked=len(table),n_valid_rows=int(good.sum()),
        n_excluded_rows=int(excluded.sum()),n_nonconverged_rows=int(table.status.eq('nonconverged').sum()),
        maximum_fixed_point_residual=float(table.fixed_point_residual.max()),
        n_independent_solver_checks=int(table.independent_probability_difference.notna().sum()),
        maximum_independent_solver_difference=float(table.independent_probability_difference.max()),
        n_parameter_dominance_checks=n_order_checks,maximum_parameter_order_violation=maximum_order_violation,
        n_probability_caches_checked=len(vector_indices),n_probability_rows_checked=vector_rows,maximum_vector_mean_error=max_mean_error,
        n_recomputed_original_equation_checks=len(model_checks),
        n_replayed_pilot_geometry_checks=len(geometry_checks),
        maximum_recomputed_original_equation_residual=max(x['recomputed_residual'] for x in model_checks),
        saved_response_fractions=fractions,
        estimated_remaining_hours=predict(1.),heuristic_remaining_hours_range=[predict(.75),predict(1.8)],
        timing_is_conditional_not_a_confidence_interval=True,
        timing_limitations='Only a few complete Only-Random runs; rare near-critical cases and shared CPU/memory load may prolong completion.',
        interpretation='Saved snapshots are uneven across scenarios and habitat levels; fractions are partial diagnostics, not final scenario comparisons.',
        elapsed_check_seconds=time.time()-started,producer_fingerprint=fp,
        assessment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS/'logs/11_partial_plausibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
