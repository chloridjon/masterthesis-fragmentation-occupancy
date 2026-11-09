import importlib
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/analyses/02_metapop/02_02_species_parameter_selection'))
D=importlib.import_module('09_confirmed_response_sampling')


class ConditionalDensityTests(unittest.TestCase):
    def test_area_mapping_allocates_wide_strips_proportionally(self):
        # First x strip: width1,height1,area1. Second: width1,height3,area3.
        s=D.pd.DataFrame(dict(log_ell_low=[0,1],log_ell_high=[1,2],log_r_low=[0,0],log_r_high=[1,3],log_area=[1,3]))
        x,r,index=D.area_map(s,.125,.5)
        self.assertEqual((x,r,index),(.5,.5,0))
        x,r,index=D.area_map(s,.625,.5)
        self.assertEqual((x,r,index),(1.5,1.5,1))

    def test_unresolved_region_is_not_filled_by_interpolation(self):
        # The only retained certificate lies at ell2; left/right are
        # bracketed by separate classifications, yielding no positive area.
        with self.assertRaises(ValueError):D.inner_strips([1,2,4],[0,1,3],[0,1,2],[1,4],[-1,4])

    def test_complementary_certificates_define_a_rectangular_inner_region(self):
        s=D.inner_strips([1,4],[2,0],[1,1],[1,4],[-1,4])
        self.assertEqual(len(s),1)
        self.assertAlmostEqual(s.log_area.sum(),2*np.log(4))
        self.assertEqual((s.log_r_low.iloc[0],s.log_r_high.iloc[0]),(0,2))


if __name__=='__main__':unittest.main()
