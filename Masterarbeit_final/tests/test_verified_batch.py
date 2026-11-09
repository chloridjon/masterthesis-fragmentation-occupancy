import unittest
import numpy as np
from src.metapop.verified_batch import raw_ratio_group
from src.metapop.hanski import run_continuous_equilibrium_from_kernel


class VerifiedGroupTests(unittest.TestCase):
    def test_known_two_patch_equilibria(self):
        p,reports,k=raw_ratio_group([4,4],[[0,1],[1,0]],1,[.1,1,8])
        np.testing.assert_allclose(p,[[.975,.75,0],[.975,.75,0]],atol=1e-11)
        self.assertTrue(all(not r['needs_scalar_verification'] for r in reports))
        self.assertTrue(all(r['fixed_point_residual']<=1e-12 for r in reports))

    def test_unequal_areas_agree_with_scalar_original_equation(self):
        a=np.array([1.,9.,3.]);d=np.array([[0,1,5],[1,0,2],[5,2,0]])
        ratios=[.01,1,10,100]
        p,reports,k=raw_ratio_group(a,d,2,ratios)
        for i,r in enumerate(ratios):
            ref=run_continuous_equilibrium_from_kernel(a,k,r=r,b=1,x=.5,initial=1.,tol=1e-13,max_iter=200000)
            self.assertTrue(ref.converged)
            np.testing.assert_allclose(p[:,i],ref.probabilities,atol=1e-10,rtol=0)

    def test_insufficient_iteration_budget_requires_scalar_check(self):
        p,reports,k=raw_ratio_group([4,4],[[0,1],[1,0]],1,[1],max_iter=1)
        self.assertTrue(reports[0]['needs_scalar_verification'])

    def test_single_patch_is_undefined(self):
        with self.assertRaises(ValueError):raw_ratio_group([1],[[0]],1,[1])


if __name__=='__main__':unittest.main()
