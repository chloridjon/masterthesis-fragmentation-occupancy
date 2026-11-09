"""Reproducible limit-based species pilot for q=0 and q=1; run from repo root."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-limit-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
REL=Path('02_metapop/02_02_species_parameter_selection')
FOLDER=Path(__file__).parent
sys.path.insert(0,str(FOLDER)); sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
from scipy.stats import qmc
from src.metapop.domain_selection import half_distance_bounds,scientific_ratio
CONFIG=FINAL/'config'/REL/'limit_based/09_limit_based_species_parameters.json'
DATA=FINAL/'data/processed'/REL/'limit_based'
RESULTS=FINAL/'results'/REL/'limit_based'


def load(name):
    spec=importlib.util.spec_from_file_location(name.removesuffix('.py'),FOLDER/name)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


def write_manifest(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value: raise RuntimeError(f'Stale cache manifest: {path}')
    else: path.write_text(json.dumps(value,indent=2)+'\n')


def run_geometry(c,lc,geo,workers):
    destination=DATA/'geometry'; destination.mkdir(parents=True,exist_ok=True)
    fp=geo.fingerprint(c)
    write_manifest(destination/'manifest.json',dict(configuration={k:c[k] for k in ('landscape_config','landscape_data','replicates','h_targets')},geometry_fingerprint=fp))
    benchmark=pd.read_csv(FINAL/c['landscape_data']); jobs=[]
    for scenario in lc['scenarios']:
        for rep in c['replicates']:
            path=destination/f"{scenario['scenario_id']}__rep{rep:03d}.csv"
            if path.exists():
                old=pd.read_csv(path)
                if len(old)!=len(c['h_targets']) or not old.geometry_fingerprint.eq(fp).all() or not all((destination/f).exists() for f in old.geometry_file):
                    raise RuntimeError('Stale or incomplete geometry cache.')
            else:
                b=benchmark[benchmark.scenario_id.eq(scenario['scenario_id'])&benchmark.replicate.eq(rep)]
                jobs.append((scenario,rep,c,lc,b,str(destination),fp))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(geo.replay,*job):job for job in jobs}
        for i,f in enumerate(as_completed(futures),1):
            n,seconds=f.result(); job=futures[f]
            print(f"Geometry {i}/{len(jobs)}: {job[0]['scenario_id']} rep {job[1]}, {seconds:.1f}s",flush=True)
    table=pd.concat([pd.read_csv(p) for p in sorted(destination.glob('*.csv'))],ignore_index=True)
    if len(table)!=7*len(c['replicates'])*len(c['h_targets']) or table.duplicated(['scenario_id','replicate','H_target']).any():
        raise AssertionError('Invalid landscape sample.')
    table.to_csv(DATA/'09_landscape_geometry.csv',index=False)
    return table,fp


def run_bounds(g,c,ell,fp,gfp,worker,workers):
    destination=DATA/'bounds'; destination.mkdir(parents=True,exist_ok=True); jobs=[]
    for (_,rep),frame in g.groupby(['scenario_id','replicate']):
        path=destination/f"{frame.iloc[0].scenario_id}__rep{int(rep):03d}.csv"
        if path.exists():
            old=pd.read_csv(path)
            if len(old)!=len(frame[frame.n_patches.ge(2)])*2 or not old.producer_fingerprint.eq(fp).all() or not path.with_suffix('.npz').exists():
                raise RuntimeError('Stale or incomplete boundary cache.')
        else: jobs.append((frame,c,ell,str(DATA/'geometry'),str(destination),fp,gfp))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker.boundaries,*job):job for job in jobs}
        for i,f in enumerate(as_completed(futures),1):
            n,seconds=f.result(); job=futures[f]
            print(f"Bounds {i}/{len(jobs)}: {job[0].iloc[0].scenario_id} rep {int(job[0].iloc[0].replicate)}, {seconds:.1f}s",flush=True)
    table=pd.concat([pd.read_csv(p) for p in sorted(destination.glob('*.csv'))],ignore_index=True)
    if len(table)!=int(g.n_patches.ge(2).sum())*2: raise AssertionError('Incomplete boundary table.')
    table.to_csv(DATA/'09_boundary_landscapes.csv',index=False)
    return table


def domains_from_bounds(boundaries,c,ell):
    levels=np.geomspace(*ell,c['pilot_grid']['n_half_distances']); domains=[]
    for q in c['model_formulations']:
        part=boundaries[boundaries.q.eq(q)]
        lrmin=float(part.log_r_feasible.min()); upper=float(part.capacity_upper.max())
        lrmax=float(np.log(upper*(1+c['upper_capacity_margin'])))
        rs=np.linspace(lrmin,lrmax,c['pilot_grid']['n_r_values'])
        grid=[dict(parameter_id=f'r{i:02d}_ell{j:02d}',ell_id=j,log_r=float(lr),half_distance=float(e)) for j,e in enumerate(levels) for i,lr in enumerate(rs)]
        profiles=[]
        for name,ri,ei in [('Reference',.5,.5),('Low r, short dispersal',.25,.25),('Low r, long dispersal',.25,.75),('High r, short dispersal',.75,.25),('High r, long dispersal',.75,.75)]:
            j=int(round(ei*(len(levels)-1)))
            profiles.append(dict(parameter_id=name,ell_id=j,log_r=float(lrmin+ri*(lrmax-lrmin)),half_distance=float(levels[j])))
        controlling=part.loc[part.log_r_feasible.idxmin()]
        largest=part.loc[part.capacity_upper.idxmax()]
        domains.append(dict(q=q,log_r_min=lrmin,log_r_max=lrmax,r_min=scientific_ratio(lrmin),r_max=scientific_ratio(lrmax),
            maximum_capacity_upper=upper,lower_controlling_landscape={k:controlling[k].item() if isinstance(controlling[k],np.generic) else controlling[k] for k in ['scenario_id','replicate','H_target']},
            upper_controlling_landscape={k:largest[k].item() if isinstance(largest[k],np.generic) else largest[k] for k in ['scenario_id','replicate','H_target']},
            half_distance_values=levels.tolist(),grid=grid,profiles=profiles))
    return domains


def save_design(domains,c):
    for domain in domains:
        q=int(domain['q']); lo,hi=domain['half_distance_values'][0],domain['half_distance_values'][-1]
        for name,items in [('pilot_grid',domain['grid']),('generic_profiles',domain['profiles'])]:
            table=pd.DataFrame(items)
            table['q']=q; table['r']=[scientific_ratio(lr) for lr in table.log_r]
            table['log10_r']=table.log_r/np.log(10); table['alpha']=np.log(2)/table.half_distance
            table.to_csv(RESULTS/'tables'/f'09_q{q}_{name}.csv',index=False)
        unit=qmc.LatinHypercube(d=2,seed=c['final_lhs']['seeds'][str(q)]).random(c['final_lhs']['n_samples'])
        lr=domain['log_r_min']+unit[:,0]*(domain['log_r_max']-domain['log_r_min'])
        ell=np.exp(np.log(lo)+unit[:,1]*np.log(hi/lo))
        pd.DataFrame(dict(parameter_id=[f'lhs{i:03d}' for i in range(1,len(lr)+1)],q=q,log_r=lr,log10_r=lr/np.log(10),r=[scientific_ratio(v) for v in lr],half_distance=ell,alpha=np.log(2)/ell,unit_log_r=unit[:,0],unit_log_half_distance=unit[:,1]))\
            .to_csv(RESULTS/'tables'/f'09_q{q}_independent_lhs.csv',index=False)


def run_responses(g,c,domains,fp,gfp,worker,workers):
    destination=DATA/'responses'; destination.mkdir(parents=True,exist_ok=True); jobs=[]
    expected=len(c['h_targets'])*(len(domains[0]['grid'])+5)
    for domain in domains:
        for (_,rep),frame in g.groupby(['scenario_id','replicate']):
            path=destination/f"q{int(domain['q'])}__{frame.iloc[0].scenario_id}__rep{int(rep):03d}.csv"
            if path.exists():
                old=pd.read_csv(path)
                if len(old)!=expected or not old.producer_fingerprint.eq(fp).all() or not path.with_suffix('.npz').exists():
                    raise RuntimeError('Stale or incomplete response cache.')
            else: jobs.append((frame,c,domain,str(DATA/'geometry'),str(destination),fp,gfp))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker.responses,*job):job for job in jobs}
        for i,f in enumerate(as_completed(futures),1):
            n,failed,seconds=f.result(); job=futures[f]
            print(f"Responses {i}/{len(jobs)}: q={int(job[2]['q'])} {job[0].iloc[0].scenario_id} rep {int(job[0].iloc[0].replicate)}, {n} rows, {failed} nonconverged, {seconds:.1f}s",flush=True)
    frames=[pd.read_csv(p,dtype={'r':str}) for p in sorted(destination.glob('*.csv'))]
    data=pd.concat(frames,ignore_index=True)
    if len(data)!=70*expected or data.duplicated(['q','scenario_id','replicate','H_target','phase','parameter_id']).any():
        raise AssertionError('Invalid response sample.')
    data.to_csv(DATA/'09_occupancy.csv',index=False)
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--stage',choices=['all','geometry','bounds','responses','summary'],default='all')
    args=parser.parse_args(); c=json.loads(CONFIG.read_text()); lc=json.loads((FINAL/c['landscape_config']).read_text())
    if (c['b'],c['x'],c['initial_occupancy'],c['kernel_epsilon'])!=(1.,.5,1.,0.) or c['model_formulations']!=[0.,1.]:
        raise ValueError('Model differs from the analysis protocol.')
    if (lc['grid_size_cells'],lc['distance_mode'],lc['voronoi_depth'],lc['analysis_boundary'])!=(200,'voronoi_dt',1,'open'):
        raise ValueError('Landscape representation differs from the protocol.')
    for name in ('tables','logs','figures'): (RESULTS/name).mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    geo=load('09_species_geometry.py'); worker=load('09_species_domain_workers.py')
    g,gfp=run_geometry(c,lc,geo,args.workers)
    if args.stage=='geometry': return
    ell=half_distance_bounds(g.d_min_positive.min(),g.d_max.max(),c['kernel_tolerance']); fp=worker.source_fingerprint(c,gfp)
    manifest=dict(configuration=c,geometry_fingerprint=gfp,producer_fingerprint=fp,half_distance_bounds=list(ell),distance_min_positive=float(g.d_min_positive.min()),distance_max=float(g.d_max.max()))
    write_manifest(DATA/'run_manifest.json',manifest)
    boundaries=run_bounds(g,c,ell,fp,gfp,worker,args.workers)
    domains=domains_from_bounds(boundaries,c,ell)
    value=dict(status='domains_computed_responses_pending',half_distance_bounds=list(ell),kernel_tolerance=c['kernel_tolerance'],lower_occupancy_target=c['lower_occupancy_target'],domains=domains)
    (RESULTS/'logs/09_parameter_domains.json').write_text(json.dumps(value,indent=2)+'\n')
    for d in domains: print(f"q={int(d['q'])}: r {d['r_min']}–{d['r_max']}; ell {ell[0]:.8g}–{ell[1]:.8g}",flush=True)
    save_design(domains,c)
    if args.stage=='bounds': return
    response_fp=hashlib.sha256((fp+json.dumps(domains,sort_keys=True)).encode()).hexdigest()
    write_manifest(DATA/'response_manifest.json',dict(configuration=c,domains=domains,producer_fingerprint=response_fp,boundary_fingerprint=fp))
    if args.stage=='summary': data=pd.read_csv(DATA/'09_occupancy.csv',dtype={'r':str})
    else: data=run_responses(g,c,domains,response_fp,gfp,worker,args.workers)
    if args.stage=='responses': return
    summary=load('09_limit_based_species_summary.py')
    summary.summarise(g,boundaries,data,c,value,RESULTS,DATA)
    print('Limit-based preliminary analysis completed.',flush=True)

if __name__=='__main__': main()
