"""Apply the saved-response plausibility checks to the complete audited initial batch."""
from datetime import datetime,timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
FOLDER=Path(__file__).parent;sys.path.insert(0,str(FOLDER))
P=importlib.import_module('11_partial_plausibility')


def main():
    base=P.RESULTS;state=json.loads((base/'logs/11_run_status.json').read_text())
    independent=json.loads((base/'logs/11_independent_audit.json').read_text())
    assert state['status']=='completed_and_audited' and independent['status']=='passed'
    assert len(list((P.DATA/'trajectories').glob('*.csv')))==140
    P.RESULTS=base/'full_plausibility'
    for path in [P.RESULTS/'tables',P.RESULTS/'logs']:path.mkdir(parents=True,exist_ok=True)
    # A separate destination preserves the original measured runtime table.
    P.main()
    report=json.loads((P.RESULTS/'logs/11_partial_plausibility.json').read_text())
    assert report['n_response_rows_checked']==128*7*20*88
    assert report['n_snapshots_checked']==7*20*88 and report['n_nonconverged_rows']==0
    assert report['producer_fingerprint']==independent['producer_fingerprint']
    report.update(status='complete_initial_batch_plausibility_passed',
        all_probability_vectors_audited_separately=True,
        interpretation='Complete balanced initial design; undefined multi-patch cases retained. Parameter-order and original-equation checks passed; no empirical biological validation.',
        full_check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full_check_completed_at_utc=datetime.now(timezone.utc).isoformat())
    for key in ['estimated_remaining_hours','heuristic_remaining_hours_range','timing_limitations','timing_is_conditional_not_a_confidence_interval']:report.pop(key,None)
    (base/'logs/11_full_plausibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
