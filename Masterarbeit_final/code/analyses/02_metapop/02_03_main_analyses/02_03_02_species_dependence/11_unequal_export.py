"""Export the audited8/16 sample; confirm only after actual PDF sight review."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib,json,sys,os,subprocess
os.environ.setdefault('MPLCONFIGDIR','/tmp/ufz-unequal-export-mpl')
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(FINAL/'code/base'))
import numpy as np
import pandas as pd
E=importlib.import_module('11_unequal_replication');S=importlib.import_module('11_confirmed_domain_summary');R=E.R

def setup():
    plan=json.loads(E.CONFIG.read_text());out=FINAL/plan['results_directory']/'summary_008_016';p=out/'11_numerical_audit.json';a=json.loads(p.read_text())
    assert a['status']=='independent_numeric_audit_passed_export_pending' and a['n_trajectories']==104
    for name,h in {**a['input_sha256'],**a['output_sha256']}.items():assert R.sha(FINAL/name)==h,name
    return plan,out,p,a

def export():
    plan,out,p,a=setup();marker=out/'11_export_audit.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        for name,h in old['output_sha256'].items():assert R.sha(FINAL/name)==h,name
        assert old['export_source_sha256']==R.sha(Path(__file__));print('Checked exports retained; inspect/confirm them.');return
    (out/'figures').mkdir(exist_ok=True)
    design=pd.read_csv(out/'tables/11_profile_response_diagnostics.csv');t=pd.read_csv(out/'tables/11_parameter_ensemble_curves.csv')
    ids=[s['scenario_id'] for s in S.FOCAL_SCENARIOS];hs=sorted(t.H_target.unique())
    ensemble=np.array([[t[t.scenario_id.eq(sid)].set_index('H_target').reindex(hs)[name].to_numpy() for sid in ids] for name in ['parameter_q05','parameter_median','parameter_q95']])
    S.render(design,design.median_response_span.to_numpy(),ensemble,hs,out,16)
    files=list((out/'figures').glob('*'));fonts=[]
    for f in sorted((out/'figures').glob('*.pdf')):fonts.append(dict(file=str(f.relative_to(FINAL)),fonts=subprocess.run(['pdffonts',str(f)],capture_output=True,text=True,check=True).stdout))
    captions='''# Figure captions: unequal8/16 replication

The parameter diagnostic shows all256 fixed sampled profiles, coloured by the
span of median equilibrium patch occupancy across evaluable scenario–H cells.
Medians use up to eight Only-Random or16 other-scenario landscapes, requiring
at least three valid multi-patch landscapes. No profile was replaced or
excluded based on its response. Parameter axes are logarithmic; dispersal
half-distance is in dimensionless raster-cell coordinates.

The parameter-ensemble figure shows the seven fixed scenarios. Each curve is
the median across256 landscape-median profile responses, with equal design
weights1/256. Ribbons show5th–95th percentiles across parameter profiles,
not landscape variation or confidence intervals. Landscape medians use eight
Only-Random seeds and16 seeds in each of the other six scenarios. Missing
landscapes and cells with fewer than three valid seeds remain unavailable.
The eight shared seed blocks and eight additional directed-only blocks are
identified separately in the numerical report. Habitat proportion decreases
left to right. Colours identify rho, Access curves are dashed, and other
curves are solid. Along-H curves jointly vary amount and configuration.

These figures are descriptive. No bootstrap intervals, biological calibration
or completed Result-D inference are claimed. Actual standalone visual review
is a separate confirmation; review in the final manuscript remains necessary.
'''
    (out/'11_figure_captions.md').write_text(captions);files.append(out/'11_figure_captions.md')
    R.save_json(marker,dict(status='exports_checked_actual_visual_review_pending',numeric_audit_sha256=R.sha(p),export_source_sha256=R.sha(Path(__file__)),output_sha256={str(f.relative_to(FINAL)):R.sha(f) for f in files},font_checks=fonts,visual_review_confirmed=False))

def confirm():
    plan,out,p,a=setup();marker=out/'11_export_audit.json';b=json.loads(marker.read_text())
    assert R.sha(p)==b['numeric_audit_sha256'] and R.sha(Path(__file__))==b['export_source_sha256']
    for name,h in b['output_sha256'].items():assert R.sha(FINAL/name)==h,name
    b.update(status='independently_and_visually_checked_8_16',visual_review_confirmed=True,confirmed_at_utc=datetime.now(timezone.utc).isoformat());R.save_json(marker,b)
    print('Independent8/16 numerical and actual visual checks confirmed.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--confirm-visual-review',action='store_true');a=p.parse_args();confirm() if a.confirm_visual_review else export()
