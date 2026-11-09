"""Refine q=0 transitions, freeze a main domain, validate profiles and save LHS."""
from __future__ import annotations
import argparse,hashlib,importlib,json,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-transition-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
FOLDER=Path(__file__).parent
sys.path.insert(0,str(FOLDER)); sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
from src.metapop.domain_selection import scientific_ratio
WORKER=importlib.import_module('09_species_domain_workers')
GEO=importlib.import_module('09_species_geometry')
REL=Path('02_metapop/02_02_species_parameter_selection')
CONFIG=FINAL/'config'/REL/'transition_focused/09_transition_species_parameters.json'
DATA=FINAL/'data/processed'/REL/'transition_focused'
RESULTS=FINAL/'results'/REL/'transition_focused'
OLD=FINAL/'data/processed'/REL/'limit_based'


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_geometry(c):
    fp=GEO.fingerprint(c)
    manifest=json.loads((OLD/'geometry/manifest.json').read_text())
    if manifest['geometry_fingerprint']!=fp: raise RuntimeError('Geometry sources changed; replay required.')
    g=pd.read_csv(OLD/'09_landscape_geometry.csv')
    benchmark=pd.read_csv(FINAL/c['landscape_data'])
    keys=['scenario_id','replicate','H_target']
    matched=g.merge(benchmark[keys+['grid_sha256','n_clusters']],on=keys,validate='one_to_one',suffixes=('','_benchmark'))
    assert len(g)==350 and len(matched)==350 and not g.duplicated(keys).any()
    assert g.geometry_fingerprint.eq(fp).all()
    assert matched.grid_sha256.eq(matched.grid_sha256_benchmark).all()
    assert matched.n_patches.eq(matched.n_clusters).all()
    assert (g.n_random_cumulative+g.n_directed_cumulative).eq(g.event_count).all()
    for _,row in g.iterrows():
        a,d=WORKER.geometry(row,OLD/'geometry',fp)
        assert a.sum()==round(40000*row.H_target)
        assert np.isfinite(d).all() and (d>=0).all() and np.array_equal(d,d.T)
    g.to_csv(DATA/'09_landscape_geometry.csv',index=False)
    return g,fp


def pilot_domain(c):
    setting=c['pilot_grid']; lo=setting['log10_r_min']; hi=setting['log10_r_max']; step=setting['log10_r_step']
    n=round((hi-lo)/step)+1
    rs=np.linspace(lo,hi,n)*np.log(10); ell=setting['half_distances']
    return dict(q=0.,half_distance_values=ell,profiles=[],
        grid=[dict(parameter_id=f'r{i:03d}_ell{j:02d}',ell_id=j,log_r=float(lr),half_distance=e)
              for j,e in enumerate(ell) for i,lr in enumerate(rs)])


def response_run(g,c,domain,gfp,workers,phase):
    fp=WORKER.source_fingerprint(c,gfp)
    fp=hashlib.sha256((fp+json.dumps(domain,sort_keys=True)).encode()+Path(__file__).read_bytes()).hexdigest()
    directory=DATA/phase; directory.mkdir(exist_ok=True)
    manifest=dict(configuration=c,domain=domain,geometry_fingerprint=gfp,producer_fingerprint=fp)
    path=directory/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError(f'Stale {phase} cache.')
    save_json(path,manifest)
    jobs=[]; expected=10*(len(domain['grid'])+len(domain['profiles']))
    for (_,rep),frame in g.groupby(['scenario_id','replicate']):
        p=directory/f'q0__{frame.iloc[0].scenario_id}__rep{int(rep):03d}.csv'
        if p.exists():
            table=pd.read_csv(p)
            if len(table)!=expected or not table.producer_fingerprint.eq(fp).all() or not p.with_suffix('.npz').exists():
                raise RuntimeError('Incomplete or stale response cache.')
        else: jobs.append((frame,c,domain,str(OLD/'geometry'),str(directory),fp,gfp))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(WORKER.responses,*job):job for job in jobs}
        for i,f in enumerate(as_completed(futures),1):
            n,failed,seconds=f.result(); frame=futures[f][0]
            print(f'{phase} {i}/{len(jobs)}: {frame.iloc[0].scenario_id} rep {int(frame.iloc[0].replicate)}, {seconds:.1f}s, {failed} nonconverged',flush=True)
    table=pd.concat([pd.read_csv(p,dtype={'r':str}) for p in sorted(directory.glob('*.csv'))],ignore_index=True)
    assert len(table)==35*expected and not table.duplicated(['scenario_id','replicate','H_target','phase','parameter_id']).any()
    table.to_csv(DATA/f'09_{phase}_occupancy.csv',index=False)
    if table.status.eq('nonconverged').any(): raise RuntimeError('Unconverged responses; do not freeze the main domain.')
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--stage',choices=['all','preflight','pilot','select','profiles','summary'],default='all')
    args=parser.parse_args(); c=json.loads(CONFIG.read_text())
    for d in [DATA,RESULTS/'tables',RESULTS/'logs',RESULTS/'figures']: d.mkdir(parents=True,exist_ok=True)
    g,gfp=validate_geometry(c); domain=pilot_domain(c)
    save_json(RESULTS/'logs/09_preflight.json',dict(configuration=c,geometry_fingerprint=gfp,
        n_landscapes=len(g),n_valid_landscapes=int(g.n_patches.ge(2).sum()),
        n_excluded_landscapes=int(g.n_patches.lt(2).sum()),n_pilot_combinations=len(domain['grid']),
        ratio_adjacent_factor=10**c['pilot_grid']['log10_r_step'],geometry_cache_revalidated=True))
    print(f'Preflight: {len(g)} matched landscapes; {len(domain["grid"])} combinations; r step factor {10**c["pilot_grid"]["log10_r_step"]:.4f}',flush=True)
    if args.stage=='preflight': return
    if args.stage in ['all','pilot']: data=response_run(g,c,domain,gfp,args.workers,'pilot')
    else: data=pd.read_csv(DATA/'09_pilot_occupancy.csv',dtype={'r':str})
    if args.stage=='pilot': return
    summary=importlib.import_module('09_transition_species_summary')
    decision=summary.select(data,c,RESULTS)
    save_json(RESULTS/'logs/09_main_parameter_decision.json',decision)
    print(f'Main candidate: r={decision["r_min"]}–{decision["r_max"]}; ell={decision["half_distance_min"]}–{decision["half_distance_max"]}',flush=True)
    if args.stage=='select': return
    profile_domain=summary.profile_domain(decision)
    if args.stage in ['all','profiles']:
        profiles=response_run(g,c,profile_domain,gfp,args.workers,'profiles')
    else: profiles=pd.read_csv(DATA/'09_profiles_occupancy.csv',dtype={'r':str})
    if args.stage=='profiles': return
    summary.finish(g,data,profiles,c,decision,RESULTS,DATA)
    print('Transition-domain refinement completed.',flush=True)

if __name__=='__main__': main()
