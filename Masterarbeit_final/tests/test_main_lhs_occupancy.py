"""Known equilibrium, continuation and missing-domain checks for the main run."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT if ROOT.name=='Masterarbeit_final' else ROOT/'Masterarbeit_final'
FOLDER=FINAL/'code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence'
sys.path.insert(0,str(FOLDER))
worker=importlib.import_module('11_lhs_workers')
summary=importlib.import_module('11_lhs_summary')


class MainLHSChecks(unittest.TestCase):
    def config(self):
        return json.loads((FINAL/'config/02_metapop/02_03_main_analyses/02_03_02_species_dependence/lhs_20_seeds/11_lhs_occupancy.json').read_text())

    def test_known_equilibrium_and_analytically_empty_case(self):
        c=self.config();a=np.array([4.,4.]);d=np.array([[0.,1.],[1.,0.]])
        for r,expected in [(1.,.75),(8.,0.)]:
            item=SimpleNamespace(alpha=np.log(2),log_r=np.log(r),half_distance=1.,parameter_id='lhs001')
            p,report=worker.solve(a,d,item,c)
            np.testing.assert_allclose(p,expected,atol=1e-8)
            self.assertEqual(report['status'],'valid')
            self.assertLessEqual(report['fixed_point_residual'],1.1e-10)

    def test_same_iterate_is_continued_when_first_budget_is_insufficient(self):
        c=self.config();c['solver']['max_iter']=1
        item=SimpleNamespace(alpha=np.log(2),log_r=0.,half_distance=1.,parameter_id='lhs001')
        p,report=worker.solve(np.array([4.,4.]),np.array([[0.,1.],[1.,0.]]),item,c)
        np.testing.assert_allclose(p,.75,atol=1e-8)
        self.assertGreater(report['n_iter'],1)
        self.assertIn('continued',report['solver'])
        self.assertTrue(report['converged'])

    def test_excluded_domain_and_true_extinction_are_distinct(self):
        x=np.array([[np.nan,0.,.2],[np.nan,0.,.5],[np.nan,0.,.8]])
        count,median,low,high=summary.summary_arrays(x,3)
        self.assertEqual(count.tolist(),[0,3,3])
        self.assertTrue(np.isnan(median[0]))
        self.assertEqual(median[1],0.)
        self.assertEqual(median[2],.5)
        self.assertAlmostEqual(low[2],.23)
        self.assertAlmostEqual(high[2],.77)


if __name__=='__main__':unittest.main()
