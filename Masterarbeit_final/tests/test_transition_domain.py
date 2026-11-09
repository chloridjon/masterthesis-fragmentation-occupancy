import unittest
import numpy as np
import pandas as pd
from src.metapop.transition_domain import transition_crossing,select_transition_rectangle


def fixture():
    rows=[]
    for ell,shift in [(1,0),(2,1),(4,2),(8,2.02),(16,2.03)]:
        for i in range(20):
            v=shift+i/20
            rows.append(dict(scenario_id=str(i),H_target=.5,half_distance=ell,
                p50=v,p95_low=v-1,p05_high=v+1))
    return pd.DataFrame(rows)


class TransitionSelectionTests(unittest.TestCase):
    def test_crossing_retains_bracket_and_censoring(self):
        c=transition_crossing([0,1,2],[1,.75,0],.5)
        self.assertAlmostEqual(c['value'],4/3)
        self.assertEqual((c['low'],c['high']),(1,2))
        self.assertEqual(transition_crossing([0,1],[.2,0],.95)['status'],'below_grid')
        with self.assertRaises(ValueError): transition_crossing([0,1],[.1,.8],.5)

    def test_selection_plateau_and_outside_brackets(self):
        result=select_transition_rectangle(fixture())
        self.assertEqual(result['half_distance_max'],4)
        self.assertAlmostEqual(result['log10_r_min'],-1.2)
        self.assertAlmostEqual(result['log10_r_max'],4.15)

    def test_no_silent_loss_of_censored_cells(self):
        f=fixture(); f.loc[(f.half_distance==1)&f.scenario_id.isin(['0','1','2']),'p95_low']=-np.inf
        with self.assertRaisesRegex(ValueError,'censored'): select_transition_rectangle(f)
        f=fixture(); f.loc[f.half_distance==16,'p50']+=1
        with self.assertRaisesRegex(ValueError,'Plateau'): select_transition_rectangle(f)

if __name__=='__main__': unittest.main()
