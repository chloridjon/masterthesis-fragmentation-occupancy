"""Verified Result A replay and full Voronoi-hybrid geometry for the species pilot."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
from src.landscape.generation import run_sequential_deforestation
from src.landscape.landscape import Landscape


def fingerprint(c):
    settings={k:c[k] for k in ('landscape_config','landscape_data','replicates','h_targets')}
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True).encode())
    sources=[Path(__file__),FINAL/c['landscape_config'],FINAL/c['landscape_data']]
    sources += [FINAL/'code/base/src/landscape'/f'{name}.py' for name in
                ('generation','deforestation','landscape','boundary','complete_edge_distances')]
    for p in sources:
        digest.update(str(p.relative_to(FINAL)).encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def replay(scenario,rep,c,lc,benchmark,destination,fp):
    started=time.monotonic(); destination=Path(destination)
    seed=lc['base_seed']+rep-1; params=lc['access_reference']
    run=run_sequential_deforestation(np.ones((200,200),dtype=np.int8),np.random.default_rng(seed),
        rho=scenario['rho'],directed_mode=scenario['mode'],target_habitat_fraction=0,
        generation_boundary=lc['generation_boundary'],eta=params['eta'],k=params['k'],
        lambda_decay=params['lambda'],save_habitat_fractions=c['h_targets'],seed=seed)
    assert run.n_events==40000 and run.n_random_forced_initial==1
    assert run.n_random_total+run.n_directed==40000
    counts=np.r_[0,np.cumsum(np.asarray(run.event_modes)=='random')]
    rows=[]
    for snap in run.snapshots:
        h=float(snap.requested_habitat_fractions[0]); old=benchmark[benchmark.H_target.eq(h)].iloc[0]
        identity=hashlib.sha256(snap.grid.tobytes()).hexdigest()
        assert identity==old.grid_sha256 and snap.habitat_fraction==h
        land=Landscape(snap.grid)
        if h>0:
            land.ensure_clusters(edge_depth=0,connectivity='4',boundary_mode='open',distance_mode='voronoi_dt',voronoi_depth=1)
            a=np.asarray(land.cluster_areas,float); d=np.asarray(land.cluster_dist_matrix,float)
        else: a=np.empty(0); d=np.empty((0,0))
        n=len(a); assert n==old.n_clusters and a.sum()==round(40000*h)
        assert d.shape==(n,n) and np.isfinite(d).all() and (d>=0).all()
        np.testing.assert_allclose(d,d.T,atol=1e-12,rtol=0)
        if n>=2:
            off=np.triu(np.ones((n,n),dtype=bool),1)
            values=d[off]; pos=values[values>0]
            dmin=float(pos.min()) if len(pos) else np.nan; dmax=float(values.max())
            zeros=int((values==0).sum())
        else: dmin=dmax=np.nan; zeros=0
        name=f"{scenario['scenario_id']}__rep{rep:03d}__h{h:.3f}.npz"
        np.savez_compressed(destination/name,areas=a,distances=d,grid_sha256=np.array(identity),geometry_fingerprint=np.array(fp))
        event=int(snap.event_count)
        rows.append(dict(scenario_id=scenario['scenario_id'],replicate=rep,seed=seed,H_target=h,
                         n_patches=n,grid_sha256=identity,event_count=event,n_random_cumulative=int(counts[event]),
                         n_directed_cumulative=event-int(counts[event]),generator_version=run.generator_version,
                         geometry_file=name,d_min_positive=dmin,d_max=dmax,n_zero_distance_pairs=zeros,
                         geometry_fingerprint=fp))
    pd.DataFrame(rows).to_csv(destination/f"{scenario['scenario_id']}__rep{rep:03d}.csv",index=False)
    return len(rows),time.monotonic()-started
