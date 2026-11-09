"""Resume a rejected independent comparison with a tighter reference solver."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import time
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent))
import pandas as pd
worker=importlib.import_module('11_lhs_workers')
REL=Path('02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds')
DATA=FINAL/'data/processed'/REL
RESULTS=FINAL/'results'/REL


def main():
    m=json.loads((DATA/'manifest.json').read_text());c=m['configuration']
    for name,digest in m['source_hashes'].items():
        assert hashlib.sha256((FINAL/name).read_bytes()).hexdigest()==digest,name
    lc=json.loads((FINAL/c['landscape_config']).read_text())
    sid='access_weighted_rho0999';rep=12
    s=next(x for x in lc['scenarios'] if x['scenario_id']==sid)
    benchmark=pd.read_csv(FINAL/c['landscape_data'])
    benchmark=benchmark[benchmark.scenario_id.eq(sid)&benchmark.replicate.eq(rep)]
    lhs=pd.read_csv(FINAL/c['lhs_design'])
    original=worker.scaled_ratio_sweep
    def accurate_reference(*args,**kwargs):
        kwargs['tol']=min(kwargs['tol'],1e-12)
        return original(*args,**kwargs)
    worker.scaled_ratio_sweep=accurate_reference
    original_solve=worker.solve
    def recorded_solve(*args,**kwargs):
        probabilities,report=original_solve(*args,**kwargs)
        report['independent_reference_tolerance']=1e-12
        report['supplementary_verification_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return probabilities,report
    worker.solve=recorded_solve
    result=worker.trajectory(s,rep,c,lc,benchmark,lhs,str(DATA),m['producer_fingerprint'])
    note=dict(status='trajectory_resumed_and_verified',trajectory=f'{sid}__rep{rep:03d}',
        original_primary_producer_fingerprint=m['producer_fingerprint'],
        supplementary_verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary_solver_and_tolerance_unchanged=True,independent_reference_tolerance=1e-12,
        independent_probability_comparison_tolerance_unchanged=2e-7,
        initial_rejected_case=dict(H=.275,parameter_id='lhs043',primary_occupancy=0.,
            capacity_upper=30557.243504179296,r=30572.229757387093,
            reference_maximum_difference_at_tol_1e10=2.0375958194635763e-7),
        result=list(result))
    (RESULTS/'logs/11_supplementary_verification.json').write_text(json.dumps(note,indent=2)+'\n')
    print(json.dumps(note,indent=2),flush=True)
    # The original executor drains already queued work after a failed future.
    # Resume the withheld summary once all original and repaired files exist;
    # the already active completion stage then performs the final audit.
    while len(list((DATA/'trajectories').glob('*.csv')))<140:
        time.sleep(5)
    summary=importlib.import_module('11_lhs_summary')
    summary.finish(c,DATA,RESULTS,m['producer_fingerprint'])
    print('Full summary restored; existing completion stage will audit it.',flush=True)


if __name__=='__main__':main()
