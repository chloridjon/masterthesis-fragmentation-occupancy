"""Cohort preservation, unequal replication, and independently repeated ranks."""
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT if ROOT.name=='Masterarbeit_final' else ROOT/'Masterarbeit_final'
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_03_main_analyses/02_03_03_metric_associations'))
D=importlib.import_module('12_result_d_metric_associations')


class ResultDTests(unittest.TestCase):
    def test_bootstrap_matches_literal_seed_replication_and_chunk_invariance(self):
        rng=np.random.default_rng(17)
        x=rng.integers(1,8,(2,6,112)).astype(float)
        y=rng.random((2,112));y[:,8:16]=np.nan;x[:,:,8:16]=np.nan
        # Preserve a second H-dependent ecological exclusion separately.
        x[1,:,19]=np.nan;y[1,19]=np.nan
        shared,extra=D.two_cohort_indices(7,19)
        c=dict(minimum_valid_per_scenario=3,minimum_response_range=1e-10,bootstrap=dict(processing_chunk=3))
        a=D.bootstrap_block(x,y,shared,extra,c)
        c['bootstrap']['processing_chunk']=2
        np.testing.assert_array_equal(a,D.bootstrap_block(x,y,shared,extra,c))
        for draw in range(7):
            first=shared[draw].astype(int);second=extra[draw].astype(int)+8
            literal=np.concatenate([first]+[np.r_[first,second]+16*j for j in range(1,7)])
            f=np.bincount(literal,minlength=112)
            self.assertEqual(f[:16].sum(),8)
            self.assertEqual(f[8:16].sum(),0)
            for k in range(2):
                for m in range(6):
                    np.testing.assert_allclose(a[draw,k,m],D.independent(x[k,m],y[k],f,c),atol=1e-13,rtol=0,equal_nan=True)

    def test_summary_preserves_sign_changes_zeros_and_informative_denominators(self):
        r=np.array([1.,-1,0,np.nan])[:,None,None]
        absolute,signed,consistency,n=D.summaries(r)
        self.assertEqual(absolute[0,0],1)
        self.assertEqual(signed[0,0],0)
        self.assertEqual(consistency[0,0],1/3)
        self.assertEqual(n[0,0],3)
        a=D.summaries(r,np.array([1,0,0,0],bool)[:,None])
        self.assertEqual(a[3][0,0],1)


if __name__=='__main__':unittest.main()
