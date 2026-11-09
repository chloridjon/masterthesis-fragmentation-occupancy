"""Extend the species pilot; preserve and audit the original trajectories."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-species-expanded-mpl')
import numpy as np
import pandas as pd
from scipy.stats import qmc

FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
REL=Path('02_metapop/02_02_species_parameter_selection')
CONFIG=FINAL/'config'/REL/'expanded_grid/09_expanded_species_parameter_selection.json'
DATA=FINAL/'data/processed'/REL/'expanded_grid'
RESULTS=FINAL/'results'/REL/'expanded_grid'


def load(name):
    path=Path(__file__).with_name(name)
    spec=importlib.util.spec_from_file_location(name.removesuffix('.py'),path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def worker(scenario,rep,c,lc,benchmark,fingerprint,destination):
    producer=load('09_species_parameter_selection.py')
    producer.DATA=Path(destination)
    return producer.run_trajectory(scenario,rep,c,lc,benchmark,fingerprint,'pilot',[])


def verify_source(source,c,producer):
    manifest=json.loads((source/'run_manifest.json').read_text())
    previous=manifest['configuration']
    if producer.source_fingerprint(previous)!=manifest['producer_fingerprint'] or manifest['producer_fingerprint']!=c['extension']['source_fingerprint']:
        raise RuntimeError('Original producer or model sources changed; source reuse is not authorised by provenance.')
    for field in previous:
        if field not in ('r_values','half_distance_values','final_lhs') and previous[field]!=c[field]:
            raise RuntimeError(f'The extension changed the baseline setting {field}.')
    if not set(previous['r_values']).issubset(c['r_values']) or not set(previous['half_distance_values']).issubset(c['half_distance_values']):
        raise ValueError('Expanded grid must contain the complete original grid.')
    return previous,manifest


def combine(parts,destination,c,fingerprint):
    frames=[]; vectors=[]; offsets=[0]
    for stem in parts:
        frame=pd.read_csv(stem.with_suffix('.csv'))
        with np.load(stem.with_suffix('.npz')) as saved:
            p=saved['probabilities']; ix=saved['offsets']
            if len(ix)!=len(frame)+1 or ix[-1]!=len(p) or not np.array_equal(np.diff(ix),np.where(frame.n_patches.ge(2),frame.n_patches,0)):
                raise AssertionError('Invalid source probability offsets.')
            if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
                raise AssertionError('Invalid source probabilities.')
            for i,row in frame.iterrows():
                v=p[ix[i]:ix[i+1]].copy()
                if row.status=='valid' and not np.isclose(v.mean(),row.occupancy,atol=1e-12,rtol=0):
                    raise AssertionError('Source means and probability vectors differ.')
                vectors.append(v); offsets.append(offsets[-1]+len(v))
        frame['source_producer_fingerprint']=frame.producer_fingerprint
        frame['producer_fingerprint']=fingerprint
        frame['parameter_id']=[f"r{c['r_values'].index(r):02d}_ell{c['half_distance_values'].index(ell):02d}" for r,ell in zip(frame.r,frame.half_distance)]
        frames.append(frame)
    merged=pd.concat(frames,ignore_index=True)
    if merged.duplicated(['H_target','r','half_distance']).any() or len(merged)!=len(c['h_targets'])*len(c['r_values'])*len(c['half_distance_values']):
        raise AssertionError('The expanded trajectory grid is incomplete or duplicated.')
    for h,part in merged.groupby('H_target'):
        if part.grid_sha256.nunique()!=1 or part.n_patches.nunique()!=1:
            raise AssertionError('Parameter stages do not share the identical landscape.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    merged.to_csv(destination.with_suffix('.csv'),index=False)
    np.savez_compressed(destination.with_suffix('.npz'),probabilities=np.concatenate(vectors),offsets=np.asarray(offsets),areas_unit=np.array(c['area_units']))
    return merged


def boundary_diagnostics(cells,c):
    rows=[]
    eligible=cells[cells.n_valid.ge(c['selection']['minimum_valid_seeds_per_scenario_h'])]
    for axis,levels,keys in [('r',c['r_values'],['scenario_id','H_target','half_distance']),
                            ('half_distance',c['half_distance_values'],['scenario_id','H_target','r'])]:
        for side,a,b in [('lower',levels[0],levels[1]),('upper',levels[-2],levels[-1])]:
            left=eligible[eligible[axis].eq(a)].set_index(keys)['median']
            right=eligible[eligible[axis].eq(b)].set_index(keys)['median']
            delta=(right-left).abs().dropna()
            fraction=float((delta>c['extension']['boundary_occupancy_change_tolerance']).mean())
            rows.append(dict(axis=axis,side=side,first_value=a,second_value=b,n_cells=len(delta),
                             max_median_change=float(delta.max()),fraction_changes_above_tolerance=fraction,
                             plateau=fraction<=c['extension']['plateau_max_fraction_exceeding_tolerance']))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    producer=load('09_species_parameter_selection.py')
    c=json.loads(CONFIG.read_text()); lc=json.loads((FINAL/c['landscape_config']).read_text())
    producer.validate_config(c,lc)
    source=FINAL/c['extension']['source_data']
    old,old_manifest=verify_source(source,c,producer)
    fingerprint=hashlib.sha256((producer.source_fingerprint(c)+Path(__file__).read_text()).encode()).hexdigest()
    manifest=dict(configuration=c,producer_fingerprint=fingerprint,reused_source_manifest=old_manifest,
                  source_landscape_table_sha256=hashlib.sha256((FINAL/c['landscape_data']).read_bytes()).hexdigest())
    DATA.mkdir(parents=True,exist_ok=True)
    path=DATA/'run_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Expanded-run provenance changed; refusing stale cache.')
    if not path.exists(): path.write_text(json.dumps(manifest,indent=2)+'\n')
    for directory in (RESULTS/'tables',RESULTS/'logs'):
        directory.mkdir(parents=True,exist_ok=True)
    grid=pd.DataFrame([dict(r=r,half_distance=ell,alpha=np.log(2)/ell) for r in c['r_values'] for ell in c['half_distance_values']])
    grid.to_csv(RESULTS/'tables/09_exploratory_parameter_grid.csv',index=False)
    print(f'Expanded grid: {len(c["r_values"])} × {len(c["half_distance_values"])} = {len(grid)} combinations; alpha {grid.alpha.min():.8g}–{grid.alpha.max():.8g}.',flush=True)
    stages={'new_r':{**old,'r_values':[r for r in c['r_values'] if r not in old['r_values']]},
            'new_half_distance':{**old,'r_values':c['r_values'],'half_distance_values':[ell for ell in c['half_distance_values'] if ell not in old['half_distance_values']]}}
    benchmark=pd.read_csv(FINAL/c['landscape_data'])
    jobs=[]
    for stage,settings in stages.items():
        stage_dir=DATA/'parts'/stage
        for scenario in lc['scenarios']:
            for rep in c['replicates']:
                path=stage_dir/'pilot'/f"{scenario['scenario_id']}__rep{rep:03d}.csv"
                if path.exists():
                    cached=pd.read_csv(path)
                    if not cached.producer_fingerprint.eq(fingerprint).all() or not path.with_suffix('.npz').exists():
                        raise RuntimeError('Invalid extension cache.')
                else:
                    b=benchmark[benchmark.scenario_id.eq(scenario['scenario_id'])&benchmark.replicate.eq(rep)]
                    jobs.append((scenario,rep,settings,lc,b,fingerprint,str(stage_dir)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,*job):job for job in jobs}
        for i,future in enumerate(as_completed(futures),1):
            count,failed,seconds=future.result(); job=futures[future]
            print(f'Extension {i}/{len(jobs)}: {job[0]["scenario_id"]} rep {job[1]}, {count} rows, {failed} nonconverged, {seconds:.1f}s',flush=True)
    frames=[]
    for scenario in lc['scenarios']:
        for rep in c['replicates']:
            name=f"{scenario['scenario_id']}__rep{rep:03d}"
            parts=[source/'pilot'/name]+[DATA/'parts'/stage/'pilot'/name for stage in stages]
            frames.append(combine(parts,DATA/'pilot'/name,c,fingerprint))
    pilot=pd.concat(frames,ignore_index=True)
    pilot.to_csv(DATA/'09_pilot_occupancy.csv',index=False)
    cells=producer.diagnostics(pilot,c)
    cells.to_csv(RESULTS/'tables/09_parameter_scenario_h_diagnostics.csv',index=False)
    ranges,marginal=producer.select_ranges(cells,c)
    marginal.to_csv(RESULTS/'tables/09_range_selection_marginals.csv',index=False)
    audit=producer.seed_audit(pilot,c,cells,ranges)
    audit.to_csv(RESULTS/'tables/09_seed_influence.csv',index=False)
    boundaries=boundary_diagnostics(cells,c)
    boundaries.to_csv(RESULTS/'tables/09_boundary_response_changes.csv',index=False)
    profile=producer.profile_table(ranges)
    profile.to_csv(RESULTS/'tables/09_generic_profiles.csv',index=False)
    unit=qmc.LatinHypercube(d=2,seed=c['final_lhs']['seed']).random(c['final_lhs']['n_samples'])
    points=np.exp(qmc.scale(unit,np.log([ranges['r'][0],ranges['half_distance'][0]]),np.log([ranges['r'][1],ranges['half_distance'][1]])))
    pd.DataFrame(dict(parameter_id=[f'lhs{i:03d}' for i in range(1,len(unit)+1)],r=points[:,0],half_distance=points[:,1],
                      alpha=np.log(2)/points[:,1],unit_log_r=unit[:,0],unit_log_half_distance=unit[:,1]))\
        .to_csv(RESULTS/'tables/09_independent_final_lhs.csv',index=False)
    open_axes=[axis for axis,field in [('r','r_values'),('half_distance','half_distance_values')]
               if ranges[axis][0]==c[field][0] or ranges[axis][1]==c[field][-1]]
    decision=dict(status='candidate_boundary_limited' if open_axes else 'selected',r_range=ranges['r'],half_distance_range=ranges['half_distance'],
                  open_boundary_axes=open_axes,reference_profile=profile.iloc[0].to_dict(),selection_rule=c['selection'],
                  lhs=c['final_lhs'],producer_fingerprint=fingerprint,seed_audit_passed=bool(audit.stable.all()),
                  boundary_assessment='Retention at a grid edge is not evidence that the informative parameter region ends there.',
                  area_units=c['area_units'],distance_units=c['distance_units'])
    if not audit.stable.all() or pilot.status.eq('nonconverged').any(): decision['status']='provisional_sensitive_or_nonconverged'
    (RESULTS/'logs/09_parameter_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    producer.DATA=DATA; producer.RESULTS=RESULTS
    profiles=producer.run_phase(c,lc,benchmark,fingerprint,'profiles',profile.to_dict('records'),args.workers,0)
    summary=load('09_species_parameter_summary.py'); summary.RESULTS=RESULTS
    summary.summarise(pilot,profiles,c,cells,decision,audit,processed_dir=DATA)
    print(f'Expanded pilot completed: {decision["status"]}, ranges={ranges}',flush=True)


if __name__=='__main__':
    main()
