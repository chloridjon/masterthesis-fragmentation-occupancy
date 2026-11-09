"""Supplementary precision corrections must not weaken provenance checks."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if ROOT.name=='Masterarbeit_final': ROOT=ROOT.parent
PATH=ROOT/'Masterarbeit_final/code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence/11_lhs_audit.py'
SPEC=importlib.util.spec_from_file_location('lhs_supplementary_audit',PATH)
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


class SupplementaryAudit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.source=Path(self.tmp.name)/'verified.py';self.source.write_text('verified source\n')
        self.digest=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.config={'solver':{'tol':1e-10,'independent_probability_tolerance':2e-7}}
        self.receipt=dict(status='trajectory_resumed_and_verified',
            supplementary_verification_source_sha256=self.digest,
            original_primary_producer_fingerprint='original',primary_solver_and_tolerance_unchanged=True,
            independent_reference_tolerance=1e-12,independent_probability_comparison_tolerance_unchanged=2e-7,
            trajectory='declared__rep019')
        self.table=pd.DataFrame([dict(supplementary_verification_source_sha256=self.digest,
            trajectory_id='declared__rep019',independent_reference_tolerance=1e-12,
            independent_probability_difference=1e-8)])

    def tearDown(self):self.tmp.cleanup()

    def check(self):
        return AUDIT.check_supplementary_records(self.table,self.config,'original',[(self.receipt,self.source)])

    def test_verified_pair_accepted(self):self.assertEqual(self.check(),1)

    def test_changed_source_rejected(self):
        self.source.write_text('changed\n')
        with self.assertRaises(AssertionError):self.check()

    def test_different_trajectory_rejected(self):
        self.table.loc[0,'trajectory_id']='undeclared__rep020'
        with self.assertRaises(AssertionError):self.check()

    def test_primary_change_rejected(self):
        self.receipt['primary_solver_and_tolerance_unchanged']=False
        with self.assertRaises(AssertionError):self.check()


if __name__=='__main__':unittest.main()
