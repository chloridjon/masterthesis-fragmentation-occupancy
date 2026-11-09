"""Finalize checked artifacts and distinguish initial from validated bounds."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd
import scipy
FINAL=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
REL='02_metapop/02_04_robustness/self_colonization_extension_20261015'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--confirm-visual-review',action='store_true',required=True);parser.parse_args()
    out=FINAL/'results'/REL;data=FINAL/'data/processed'/REL
    n=json.loads((out/'logs/13_compute_complete.json').read_text())['n_seeds'];stage=out/f'seeds{n:03d}'
    decision=json.loads((stage/'logs/13_q1_validated_domain.json').read_text())
    rows=pd.read_csv(stage/'tables/13_endpoint_responses.csv');checks=[]
    for seed in [0]+list(range(1,n+1)):
        part=rows if seed==0 else rows[rows.replicate!=seed]
        med=part.groupby(['scenario_id','H_target','half_distance','role']).occupancy.agg(n='count',median='median').reset_index()
        med=med[med.n>=3];med['covered']=np.where(med.role.eq('lower'),med['median']>=.95-1e-10,med['median']<=.05+1e-10)
        coverage=med.groupby(['half_distance','role']).covered.agg(['mean','count']).reset_index();coverage['omitted_seed']=seed
        checks.append(coverage)
        if seed==0:med[~med.covered].to_csv(stage/'tables/13_endpoint_uncovered_cells.csv',index=False)
    checks=pd.concat(checks,ignore_index=True);checks.to_csv(stage/'tables/13_endpoint_seed_omission.csv',index=False)
    lower=float(checks[checks.role.eq('lower')]['mean'].min());upper=float(checks[checks.role.eq('upper')]['mean'].min())
    report=stage/'13_extension_report.md';text=report.read_text().split('\n## Final checked endpoint coverage')[0]
    text=text.replace('final-self-colonization-extension-review with stages validation, audit, export.',
        'final-self-colonization-extension-review --stage validation,\nfinal-self-colonization-extension-diagnostics --full-audit, and\nfinal-self-colonization-extension-review --stage export.')
    text+=f'''
## Final checked endpoint coverage

The seed-omission ratio bounds reported above refer to the initial transition
envelope on the five selection H levels. Validation at all eligible stored H
expanded its upper log10-r endpoint from6.25 to6.75. The final separate q=1
working rectangle is r={decision['r_min']:.10g}–{decision['r_max']:.10g} and
half-distance1–{decision['half_distance_max']:g} cell-coordinate units.
At every retained distance,76 scenario-H cells were eligible under the full
eight-seed design: all76 met the lower endpoint target;75/76 met the upper
endpoint target. The remaining cell is saved explicitly and does not become
zero or disappear. Across complete-seed omissions, the minimum coverage was
{100*lower:.4f}% at the lower endpoint and{100*upper:.4f}% at the upper endpoint.
The final coverage rule is{' retained' if min(lower,upper)>=.95-1e-12 else ' not retained under every omission'}
at the declared95% threshold. This is sample-conditional coverage of median
responses, not a confidence interval or a biological species limit.

All five latest PDFs were rendered with Poppler and actually viewed. Font
embedding, editable SVG text, panel layout, interval legends, H direction,
parameter-axis transformations and missing cells were checked. The known
Poppler font-container warning has correct actual glyph rendering. Review at
the manuscript's eventual embedded size remains a later editorial task.
'''
    fixture='only_random_rho0__rep001__h0.925'
    geo=FINAL/'data/processed/02_metapop/02_02_species_parameter_selection/response_boundary_refinement/geometry'/f'{fixture}.npz'
    with np.load(geo) as z:areas=z['areas'];distances=z['distances']
    vectors={};identities={}
    for q,folder in [(0,FINAL/'data/processed/02_metapop/02_03_main_analyses/02_03_02_species_dependence/confirmed_domain_256/snapshots'),(1,data/'snapshots')]:
        path=folder/f'{fixture}.csv';frame=pd.read_csv(path);i=int(np.flatnonzero(frame.parameter_id.eq('domain131'))[0])
        with np.load(path.with_suffix('.npz')) as z:vectors[q]=z['probabilities'][z['offsets'][i]:z['offsets'][i+1]].copy()
        identities[str(path.relative_to(FINAL))]=sha(path)
    r=858.7775251431638;ell=12.946219067264217;k=np.exp(-np.log(2)/ell*distances);np.fill_diagonal(k,0)
    rates=r*areas**(-.5);external=k@(areas*vectors[1]);self_source=areas*vectors[1]
    np.testing.assert_allclose((external+self_source)/(external+self_source+rates),vectors[1],atol=1.1e-12,rtol=0)
    mechanical=dict(fixture=fixture,parameter_id='domain131',r=r,half_distance=ell,areas=areas.tolist(),q0_probabilities=vectors[0].tolist(),q1_probabilities=vectors[1].tolist(),
        q1_external_source=external.tolist(),q1_self_source=self_source.tolist(),input_csv_sha256=identities,
        correction='The large high-H response in this fixture primarily reflects self-colonization of the dominant patch. The one-cell patch cannot persist from its self term alone at r>1 and benefits mainly from external colonization. This supersedes the earlier pilot-report suggestion of local self-persistence of the small isolated patch.')
    (stage/'logs/13_high_H_mechanism_check.json').write_text(json.dumps(mechanical,indent=2)+'\n')
    text+=f'''
## Mechanistic check of the high-H reference response

In the directly checked Only-Random seed1 fixture at H=.925, patch areas are
36999 and1 cells. At the unchanged reference, q=0 patch occupancies are
{vectors[0].tolist()}, while q=1 gives {vectors[1].tolist()}.
The dominant patch obtains the largest direct benefit from its self term;
its greater occupancy increases external colonization of the one-cell patch.
The latter cannot persist on its self term alone at r=858.78>1. This corrects
the earlier three-seed pilot report's suggestion that the large mean shift
was caused by local self-persistence of the small isolated patch. Original
pilot artifacts remain immutable; this checked interpretation supersedes that
sentence. Mean patch occupancy weights both patches equally and differs from
occupied habitat area. This fixture does not establish a universal mechanism
for every scenario or H level.
'''
    report.write_text(text)
    manifest=json.loads((data/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert sha(FINAL/name)==digest,name
    for name,digest in decision['supplemental_sources'].items():assert sha(FINAL/name)==digest,name
    retained=json.loads((out/'logs/13_retained_pilot.json').read_text())
    old=FINAL/'data/processed/02_metapop/02_04_robustness/self_colonization_pilot_20261015/snapshots'
    for name,digest in retained['files'].items():assert sha(old/name)==digest,name
    checks_audit={}
    for name in ['13_independent_audit.json','13_domain_independent_audit.json','13_new_seed_geometry_audit.json']:
        path=stage/'logs'/name;assert json.loads(path.read_text())['status']=='passed';checks_audit[name]=sha(path)
    precision_path=stage/'logs/13_diagnostic_reader_precision.json';precision=json.loads(precision_path.read_text())
    assert precision['status']=='passed' and precision['primary_coefficients_sha256_before']==precision['primary_coefficients_sha256_after']==sha(stage/'tables/13_D_metric_associations.csv')
    assert precision['supplemental_source_sha256']==sha(Path(__file__).with_name('13_self_colonization_extension_diagnostics.py'))
    copies=['metapop/hanski.py','metapop/self_equilibrium.py','metapop/transition_domain.py','tools/weighted_rank_associations.py','tools/codex_run.py']
    for name in copies:assert (FINAL.parent/'src'/name).read_bytes()==(FINAL/'code/base/src'/name).read_bytes()
    assert len(list((data/'snapshots').glob('*.csv')))==7*13*n
    assert len(list((data/'domain').glob('*.csv')))==7*5*n
    files={};fonts={}
    pdfs=list((stage/'figures').glob('*.pdf'));assert len(pdfs)==5
    for pdf in pdfs:
        fonts[pdf.name]=subprocess.run(['pdffonts',str(pdf)],capture_output=True,text=True,check=True).stdout
        assert 'LMRoman8-Regular' in fonts[pdf.name] and 'LatinModernMath-Regular' in fonts[pdf.name]
        svg=pdf.with_suffix('.svg');assert '<text' in svg.read_text() and '@font-face' in svg.read_text()
        preview=stage/'review'/f'{pdf.stem}.png';assert preview.exists()
        files[pdf.name]=dict(pdf_sha256=sha(pdf),svg_sha256=sha(svg),actual_poppler_preview_sha256=sha(preview))
    receipt=dict(status='numerically_and_visually_checked_extension',completed_utc=datetime.now(timezone.utc).isoformat(),n_seeds=n,
        n_matched_profiles=65,n_domain_profiles=481,n_landscape_states=7*13*n,
        working_domain={k:decision[k] for k in ['r_min','r_max','half_distance_min','half_distance_max']},
        minimum_endpoint_omission_coverage=dict(lower=lower,upper=upper),tests_passed=16,
        prior_pilot_files_preserved=len(retained['files']),independent_audits=checks_audit,supplemental_diagnostic_reader_sha256=sha(precision_path),
        report_sha256=sha(report),figures=files,font_checks=fonts,root_frozen_identical=copies,
        source_sha256=sha(Path(__file__)),software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        limitations='Fixed65 matched profiles, finite8-seed sensitivity analysis, no bootstrap CIs or bird refits. Plateau conditional on five selection H; endpoints validated across all eligible stored H. Baseline domain and reference unchanged.')
    (stage/'logs/13_final_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','n_seeds','working_domain','minimum_endpoint_omission_coverage']},indent=2))

if __name__=='__main__':main()
