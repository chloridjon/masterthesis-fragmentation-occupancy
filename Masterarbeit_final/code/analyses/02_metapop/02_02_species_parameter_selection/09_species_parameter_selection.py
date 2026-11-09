"""Section 2.2: broad species pilot, range decision, independent LHS and profiles.

Run from the repository root, using its existing .venv. All landscape snapshots
are replayed and checked against Result A hashes and patch counts. The shared
full depth-1 Voronoi hybrid matrix is used without kernel truncation. Saved NPZ
files retain patch probabilities for every parameter row (ragged offsets).
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

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == 'Masterarbeit_final')
REL = Path('02_metapop/02_02_species_parameter_selection')
sys.path.insert(0, str(FINAL / 'code/base'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ufz-species-mpl')
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
import pandas as pd
from scipy.stats import qmc
from src.landscape.generation import run_sequential_deforestation
from src.landscape.landscape import Landscape
from src.metapop.hanski import prepare_dispersal_kernel, run_continuous_equilibrium_from_kernel
from src.metapop.parameter_sweep import continuous_ratio_sweep

CONFIG = FINAL / 'config' / REL / '09_species_parameter_selection.json'
DATA = FINAL / 'data/processed' / REL
RESULTS = FINAL / 'results' / REL


def source_fingerprint(c):
    digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode())
    sources = [Path(__file__), FINAL / c['landscape_config']]
    sources += [FINAL / 'code/base/src' / name for name in (
        'landscape/generation.py', 'landscape/deforestation.py', 'landscape/landscape.py',
        'landscape/boundary.py', 'landscape/complete_edge_distances.py',
        'metapop/hanski.py', 'metapop/parameter_sweep.py')]
    for path in sources:
        digest.update(str(path.relative_to(FINAL)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_config(c, lc):
    if (c['b'], c['x'], c['self_colonization'], c['initial_occupancy'], c['kernel_epsilon']) != (1., .5, 0., 1., 0.):
        raise ValueError('Species pilot must use the selected untruncated baseline model.')
    for name in ('r_values', 'half_distance_values'):
        v = np.asarray(c[name], float)
        if len(v) < 3 or np.any(~np.isfinite(v)) or v[0] <= 0 or np.any(np.diff(v) <= 0):
            raise ValueError('Invalid positive parameter gradient.')
    if not set(c['h_targets']).issubset(lc['h_targets']):
        raise ValueError('H targets must have a Result A identity benchmark.')
    if len(c['replicates']) != len(set(c['replicates'])) or len(c['replicates']) < 3:
        raise ValueError('At least three unique complete seed blocks required.')
    if (lc['distance_mode'], lc['voronoi_depth'], lc['analysis_boundary'], lc['grid_size_cells']) != ('voronoi_dt', 1, 'open', 200):
        raise ValueError('Landscape representation differs from the analysis plan.')
    if not 0 < c['selection']['floor'] < c['selection']['ceiling'] < 1:
        raise ValueError('Invalid occupancy categories.')


def run_trajectory(scenario, rep, c, lc, benchmark, fingerprint, phase, profiles):
    started = time.monotonic()
    seed = lc['base_seed'] + rep - 1
    params = lc['access_reference']
    run = run_sequential_deforestation(
        np.ones((200,200), dtype=np.int8), np.random.default_rng(seed),
        rho=scenario['rho'], directed_mode=scenario['mode'], target_habitat_fraction=0,
        generation_boundary=lc['generation_boundary'], eta=params['eta'], k=params['k'],
        lambda_decay=params['lambda'], save_habitat_fractions=c['h_targets'], seed=seed)
    if run.n_events != 40000 or run.n_random_forced_initial != 1 or run.n_random_total + run.n_directed != 40000:
        raise AssertionError('Unexpected event trace.')
    random_counts = np.r_[0, np.cumsum(np.asarray(run.event_modes) == 'random')]
    rows, probabilities, offsets = [], [], [0]
    for snap in run.snapshots:
        h = float(snap.requested_habitat_fractions[0])
        identity = hashlib.sha256(snap.grid.tobytes()).hexdigest()
        old = benchmark.loc[benchmark.H_target.eq(h)].iloc[0]
        if identity != old.grid_sha256 or snap.habitat_fraction != h:
            raise AssertionError('Replayed grid differs from Result A.')
        landscape = Landscape(snap.grid)
        if h > 0:
            landscape.ensure_clusters(edge_depth=0, connectivity='4', boundary_mode='open',
                                      distance_mode='voronoi_dt', voronoi_depth=1)
            a = np.asarray(landscape.cluster_areas, float)
        else:
            a = np.empty(0)
        n = len(a)
        if n != old.n_clusters or a.sum() != round(40000*h):
            raise AssertionError('Recomputed patch geometry or areas differ.')
        event = int(snap.event_count)
        common = dict(scenario_id=scenario['scenario_id'], replicate=rep, seed=seed,
                      trajectory_id=f"{scenario['scenario_id']}__rep{rep:03d}", H_target=h,
                      n_patches=n, grid_sha256=identity, event_count=event,
                      n_random_cumulative=int(random_counts[event]),
                      n_directed_cumulative=event-int(random_counts[event]),
                      generator_version=run.generator_version, producer_fingerprint=fingerprint)
        groups = ([(ell, [(f'r{i:02d}_ell{j:02d}', r) for i,r in enumerate(c['r_values'])])
                   for j,ell in enumerate(c['half_distance_values'])] if phase == 'pilot' else
                  [(row['half_distance'], [(row['profile'], row['r'])]) for row in profiles])
        for ell, items in groups:
            kernel = prepare_dispersal_kernel(landscape.cluster_dist_matrix,
                        alpha=np.log(2)/ell, kernel_epsilon=0, dtype=np.float64) if n>=2 else None
            solved = continuous_ratio_sweep(a, kernel, [r for _,r in items], b=c['b'], x=c['x'],
                       max_iter=c['solver']['max_iter'], tol=c['solver']['tol']) if n>=2 else [None]*len(items)
            for (param_id,r), result in zip(items, solved):
                retried = False
                if result is not None and not result.converged:
                    retried = True
                    result = run_continuous_equilibrium_from_kernel(a, kernel, r=r, b=c['b'], x=c['x'],
                        max_iter=c['solver']['retry_max_iter'], tol=c['solver']['tol'], initial=1.)
                p = result.probabilities if result is not None else np.empty(0)
                residual = (float(np.max(np.abs(kernel@(a*p)/(kernel@(a*p)+r*a**(-.5))-p)))
                            if n>=2 else np.nan)
                rows.append({**common, 'parameter_id':param_id, 'r':r, 'half_distance':ell,
                    'alpha':np.log(2)/ell, 'occupancy':result.occupancy_fraction if result is not None and result.converged else np.nan,
                    'last_iterate_mean':result.occupancy_fraction if result is not None else np.nan,
                    'converged':result.converged if result is not None else False,
                    'n_iter':result.n_iter if result is not None else 0,
                    'max_delta':result.max_delta if result is not None else np.nan,
                    'fixed_point_residual':residual, 'retried':retried,
                    'status':'excluded_fewer_than_two_patches' if n<2 else ('valid' if result.converged else 'nonconverged')})
                probabilities.append(p)
                offsets.append(offsets[-1]+len(p))
    destination = DATA / phase
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / f"{scenario['scenario_id']}__rep{rep:03d}"
    frame = pd.DataFrame(rows)
    np.savez_compressed(stem.with_suffix('.npz'), probabilities=np.concatenate(probabilities),
                        offsets=np.asarray(offsets), areas_unit=np.array(c['area_units']))
    temporary = stem.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, stem.with_suffix('.csv'))
    return len(rows), int(frame.status.eq('nonconverged').sum()), time.monotonic()-started


def run_phase(c, lc, benchmark, fingerprint, phase, profiles, workers, limit):
    jobs=[]
    expected_count = len(c['h_targets'])*(len(c['r_values'])*len(c['half_distance_values']) if phase=='pilot' else len(profiles))
    for scenario in lc['scenarios']:
        for rep in c['replicates']:
            path = DATA / phase / f"{scenario['scenario_id']}__rep{rep:03d}.csv"
            if path.exists():
                old=pd.read_csv(path)
                if len(old)!=expected_count or not old.producer_fingerprint.eq(fingerprint).all() or not path.with_suffix('.npz').exists():
                    raise RuntimeError(f'Stale or incomplete cache: {path}')
            else:
                b=benchmark[benchmark.scenario_id.eq(scenario['scenario_id']) & benchmark.replicate.eq(rep)]
                jobs.append((scenario, rep, c, lc, b, fingerprint, phase, profiles))
    if limit:
        jobs=jobs[:limit]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_trajectory,*job):job[:2] for job in jobs}
        for number,future in enumerate(as_completed(futures),1):
            count,failed,seconds=future.result()
            scenario,rep=futures[future]
            print(f'{phase}: {number}/{len(jobs)} trajectories; {scenario["scenario_id"]} rep {rep}; {count} runs, {failed} nonconverged; {seconds:.1f}s',flush=True)
    paths=sorted((DATA/phase).glob('*.csv'))
    if len(paths)!=len(lc['scenarios'])*len(c['replicates']):
        return None
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    if len(data)!=expected_count*len(paths) or data.duplicated(['scenario_id','seed','H_target','parameter_id']).any():
        raise AssertionError('Unexpected sample counts or duplicate keys.')
    data.to_csv(DATA/f'09_{phase}_occupancy.csv',index=False)
    return data


def diagnostics(data,c):
    groups=['scenario_id','H_target','r','half_distance']
    rows=[]
    for key,part in data.groupby(groups,sort=True):
        v=part.loc[part.status.eq('valid'),'occupancy'].to_numpy(float)
        rows.append(dict(zip(groups,key)) | dict(n_total=len(part),n_valid=len(v),
            n_excluded=int(part.n_patches.lt(2).sum()),n_nonconverged=int(part.status.eq('nonconverged').sum()),
            median=np.median(v) if len(v) else np.nan, q05=np.quantile(v,.05) if len(v) else np.nan,
            q95=np.quantile(v,.95) if len(v) else np.nan,
            floor_fraction=np.mean(v<=c['selection']['floor']) if len(v) else np.nan,
            ceiling_fraction=np.mean(v>=c['selection']['ceiling']) if len(v) else np.nan,
            intermediate_fraction=np.mean((v>c['selection']['floor'])&(v<c['selection']['ceiling'])) if len(v) else np.nan))
    return pd.DataFrame(rows)


def select_ranges(cells,c, minimum_seeds=None):
    minimum=c['selection']['minimum_valid_seeds_per_scenario_h'] if minimum_seeds is None else minimum_seeds
    eligible=cells[cells.n_valid.ge(minimum)]
    if eligible.empty:
        raise RuntimeError('Insufficient valid seed blocks for range selection.')
    marginal=[]
    selected={}
    for axis,field in [('r','r_values'),('half_distance','half_distance_values')]:
        table=eligible.groupby(axis)[['floor_fraction','ceiling_fraction','intermediate_fraction']].mean().reset_index()
        table['eligible']=table.intermediate_fraction.ge(c['selection']['minimum_marginal_intermediate_fraction'])
        table['axis']=axis
        table['value']=table[axis]
        marginal.append(table.drop(columns=axis))
        values=table.loc[table.eligible,axis].to_numpy(float)
        if len(values)<2:
            raise RuntimeError(f'Exploratory grid requires expansion for {axis}.')
        selected[axis]=[float(values.min()),float(values.max())]
    return selected,pd.concat(marginal,ignore_index=True)


def profile_table(ranges):
    points={axis:np.exp(np.log(bounds[0])+np.array([.25,.5,.75])*np.log(bounds[1]/bounds[0])) for axis,bounds in ranges.items()}
    rows=[]
    for name,i,j in [('Reference',1,1),('Low r, short dispersal',0,0),('Low r, long dispersal',0,2),
                     ('High r, short dispersal',2,0),('High r, long dispersal',2,2)]:
        r,ell=points['r'][i],points['half_distance'][j]
        rows.append(dict(profile=name,r=r,half_distance=ell,alpha=np.log(2)/ell,b=1.,x=.5,self_colonization=0.))
    return pd.DataFrame(rows)


def seed_audit(data,c,full_cells,ranges):
    rows=[]
    keys=['scenario_id','H_target','r','half_distance']
    baseline=full_cells.set_index(keys)
    subsets=[('leave_one_out',seed,data[data.seed.ne(seed)]) for seed in sorted(data.seed.unique())]
    first=sorted(data.seed.unique())[:3]
    subsets.append(('first_three',-1,data[data.seed.isin(first)]))
    for kind,seed,part in subsets:
        cells=diagnostics(part,c)
        merged=cells.set_index(keys).join(baseline[['median','n_valid']],rsuffix='_full')
        valid=merged.n_valid.ge(3)&merged.n_valid_full.ge(3)
        changes=np.abs(merged.loc[valid,'median']-merged.loc[valid,'median_full'])
        chosen,_=select_ranges(cells,c)
        steps=[]
        for axis,field in [('r','r_values'),('half_distance','half_distance_values')]:
            grid=list(c[field])
            steps.extend(abs(grid.index(a)-grid.index(b)) for a,b in zip(chosen[axis],ranges[axis]))
        fraction=float(np.mean(changes>c['selection']['seed_influence_occupancy_tolerance']))
        rows.append(dict(check=kind,omitted_seed=seed,n_seed_blocks=part.seed.nunique(),
            n_comparable_cells=len(changes),max_median_change=float(changes.max()),
            fraction_median_changes_exceeding_tolerance=fraction,max_range_endpoint_grid_steps=max(steps),
            r_low=chosen['r'][0],r_high=chosen['r'][1],half_distance_low=chosen['half_distance'][0],half_distance_high=chosen['half_distance'][1],
            stable=fraction<=c['selection']['maximum_sensitive_fraction'] and max(steps)<=c['selection']['range_endpoint_tolerance_grid_steps']))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-trajectories',type=int,default=0)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    c=json.loads(CONFIG.read_text())
    lc=json.loads((FINAL/c['landscape_config']).read_text())
    validate_config(c,lc)
    fingerprint=source_fingerprint(c)
    for directory in (DATA,RESULTS/'tables',RESULTS/'logs'):
        directory.mkdir(parents=True,exist_ok=True)
    manifest={'configuration':c,'producer_fingerprint':fingerprint,
              'source_landscape_table_sha256':hashlib.sha256((FINAL/c['landscape_data']).read_bytes()).hexdigest()}
    path=DATA/'run_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Configuration, code, or landscape benchmark changed; do not adopt stale cached runs.')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    grid=pd.DataFrame([dict(r=r,half_distance=ell,alpha=np.log(2)/ell) for r in c['r_values'] for ell in c['half_distance_values']])
    grid.to_csv(RESULTS/'tables/09_exploratory_parameter_grid.csv',index=False)
    print(f'Grid audited: {len(grid)} combinations, r={c["r_values"]}, half-distance={c["half_distance_values"]}; {len(c["replicates"])} seed blocks, 7 scenarios, {len(c["h_targets"])} H levels.',flush=True)
    if args.validate_only:
        return
    benchmark=pd.read_csv(FINAL/c['landscape_data'])
    pilot=run_phase(c,lc,benchmark,fingerprint,'pilot',[],args.workers,args.max_trajectories)
    if pilot is None:
        print('Incomplete development run; parameter decision withheld.',flush=True)
        return
    cells=diagnostics(pilot,c)
    cells.to_csv(RESULTS/'tables/09_parameter_scenario_h_diagnostics.csv',index=False)
    ranges,marginal=select_ranges(cells,c)
    marginal.to_csv(RESULTS/'tables/09_range_selection_marginals.csv',index=False)
    audit=seed_audit(pilot,c,cells,ranges)
    audit.to_csv(RESULTS/'tables/09_seed_influence.csv',index=False)
    profile=profile_table(ranges)
    profile.to_csv(RESULTS/'tables/09_generic_profiles.csv',index=False)
    unit=qmc.LatinHypercube(d=2,seed=c['final_lhs']['seed']).random(c['final_lhs']['n_samples'])
    lo=np.log([ranges['r'][0],ranges['half_distance'][0]])
    hi=np.log([ranges['r'][1],ranges['half_distance'][1]])
    sample=np.exp(qmc.scale(unit,lo,hi))
    pd.DataFrame(dict(parameter_id=[f'lhs{i:03d}' for i in range(1,len(sample)+1)],r=sample[:,0],
        half_distance=sample[:,1],alpha=np.log(2)/sample[:,1],unit_log_r=unit[:,0],unit_log_half_distance=unit[:,1]))\
        .to_csv(RESULTS/'tables/09_independent_final_lhs.csv',index=False)
    status='selected' if audit.stable.all() and not pilot.status.eq('nonconverged').any() else 'provisional_sensitive_or_nonconverged'
    decision=dict(status=status,r_range=ranges['r'],half_distance_range=ranges['half_distance'],
                  reference_profile=profile.iloc[0].to_dict(),selection_rule=c['selection'],
                  area_units=c['area_units'],distance_units=c['distance_units'],
                  lhs=c['final_lhs'],producer_fingerprint=fingerprint,
                  conditional_on='Five pilot seeds and the sampled scenario-H domain; downstream analyses use the full landscape sample.')
    (RESULTS/'logs/09_parameter_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    profiles=run_phase(c,lc,benchmark,fingerprint,'profiles',profile.to_dict('records'),args.workers,args.max_trajectories)
    if profiles is None:
        return
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('species_summary',Path(__file__).with_name('09_species_parameter_summary.py'))
    summary=module_from_spec(spec)
    spec.loader.exec_module(summary)
    summary.summarise(pilot,profiles,c,cells,decision,audit)
    print(f'Completed parameter selection: {status}; ranges={ranges}',flush=True)


if __name__=='__main__':
    main()
