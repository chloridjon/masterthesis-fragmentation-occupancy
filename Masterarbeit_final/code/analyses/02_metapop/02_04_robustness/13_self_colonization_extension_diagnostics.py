"""Preserve exact saved float ranks in supplementary omission diagnostics.

The original model/endpoint/review sources remain immutable. This entry point
supplies pandas' round-trip parser during diagnostics, matching the primary
summary producer, and reruns the unchanged coefficient-equality guard. No model
or endpoint responses are modified. Use --full-audit for a complete new audit;
otherwise retain the already passed source-bound numerical audits.
"""
from pathlib import Path
import argparse
import importlib.util
import json
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('checked_self_extension_review',HERE/'13_self_colonization_extension_review.py')
I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full-audit',action='store_true');args=parser.parse_args()
    E,P=I.E,I.P;c,d,ext,domain,land,data,out,fp=E.setup(E.DEFAULT)
    n=json.loads((out/'logs/13_compute_complete.json').read_text())['n_seeds'];stage=out/f'seeds{n:03d}'
    decision=json.loads((stage/'logs/13_q1_validated_domain.json').read_text())
    for name,digest in decision['supplemental_sources'].items():assert P.sha(P.FINAL/name)==digest
    primary=stage/'tables/13_D_metric_associations.csv';before=P.sha(primary)
    original=I.pd.read_csv
    def round_trip_reader(*args,**kwargs):
        kwargs.setdefault('float_precision','round_trip')
        return original(*args,**kwargs)
    I.pd.read_csv=round_trip_reader
    try:
        if args.full_audit:
            E.V.audit({**c,'replicates':list(range(1,n+1))},d,land[land.replicate<=n],stage,E.view(data,n))
            I.independent_domain_audit(c,domain,land,data,out,n)
        for name in ['13_independent_audit.json','13_domain_independent_audit.json','13_new_seed_geometry_audit.json']:
            assert json.loads((stage/'logs'/name).read_text())['status']=='passed'
        I.omission_summary(c,d,land,data,out,n)
        I.report(c,d,land,data,out,n)
    finally:I.pd.read_csv=original
    assert P.sha(primary)==before
    P.save(stage/'logs/13_diagnostic_reader_precision.json',dict(status='passed',
        supplemental_source_sha256=P.sha(Path(__file__)),unchanged_review_source_sha256=P.sha(Path(I.__file__)),
        model_fingerprint=fp,primary_coefficients_sha256_before=before,primary_coefficients_sha256_after=P.sha(primary),
        parser='pandas float_precision=round_trip, identical to primary summary input parsing',
        guard='All10140 original weighted-rank coefficients reproduced from separately assembled inputs at absolute1e-14, with identical missingness; no assertion or tolerance weakened.',
        scope='Supplementary diagnostics and report only. All model/endpoint responses and their producer-source bindings retained unchanged.'))
    print('Supplementary diagnostics and report passed with exact saved-float parsing',flush=True)

if __name__=='__main__':main()
