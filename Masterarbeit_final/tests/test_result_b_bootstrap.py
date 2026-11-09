"""Paired dependence, missingness and withholding of unstable intervals."""
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT if ROOT.name=='Masterarbeit_final' else ROOT/'Masterarbeit_final'
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_03_main_analyses/02_03_01_scenario_occupancy'))
B=importlib.import_module('10_result_b_bootstrap')


class PairedBootstrapTests(unittest.TestCase):
    def settings(self):
        return dict(initial_draws=100,batch_draws=100,maximum_draws=500,
                    consecutive_stable_batches=3,endpoint_tolerance=.002,central_tolerance=.002,seed=42)

    def test_identical_matched_responses_have_exact_zero_contrast_uncertainty(self):
        x=np.array([.1,.2,.5,.7,.9]);a=np.stack([x,x],axis=1)[:,:,None]
        result=B.paired_contrast_bootstrap(a,self.settings())
        self.assertTrue(result['converged'][0,0])
        self.assertEqual(result['low'][0,0],0.)
        self.assertEqual(result['high'][0,0],0.)
        self.assertEqual(result['draws_used'][0,0],400)

    def test_undefined_baseline_is_not_zero_occupancy(self):
        a=np.full((5,2,1),.5);a[:,0,:]=np.nan
        result=B.paired_contrast_bootstrap(a,self.settings())
        self.assertFalse(result['eligible'][0,0])
        self.assertTrue(np.isnan(result['low'][0,0]))
        self.assertEqual(result['status'][0,0],'undefined_scenario_or_baseline')

    def test_budget_exhaustion_does_not_publish_a_confidence_interval(self):
        a=np.zeros((5,2,1));settings=self.settings();settings['maximum_draws']=200
        result=B.paired_contrast_bootstrap(a,settings)
        self.assertFalse(result['converged'][0,0])
        self.assertTrue(np.isnan(result['low'][0,0]))
        self.assertEqual(result['status'][0,0],'budget_exhausted')

    def test_joint_seed_draws_preserve_dependence_across_habitat_levels(self):
        x=np.linspace(.01,.99,17)
        one=np.stack([x,x*x],axis=1)
        a=np.stack([one,one],axis=2)
        result=B.paired_contrast_bootstrap(a,self.settings())
        first=[(r['draws'],r['low'],r['centre'],r['high']) for r in result['history'] if r['h_index']==0]
        second=[(r['draws'],r['low'],r['centre'],r['high']) for r in result['history'] if r['h_index']==1]
        self.assertEqual(first,second)


if __name__=='__main__':unittest.main()
