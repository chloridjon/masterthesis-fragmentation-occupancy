import unittest
import numpy as np
from src.metapop.domain_selection import half_distance_bounds,normalized_colonization,scaled_ratio_sweep,capacity_bracket,lower_occupancy_bracket,scientific_ratio
from src.metapop.hanski import run_continuous_equilibrium_from_kernel

class DomainSelectionTests(unittest.TestCase):
    def test_kernel_endpoint_definitions(self):
        lo,hi=half_distance_bounds(2,200)
        self.assertAlmostEqual(2**(-2/lo),.01)
        self.assertAlmostEqual(2**(-200/hi),.99)

    def test_scaled_solver_preserves_unnormalised_model_for_both_q(self):
        a=np.array([1.,4.,9.]); d=np.array([[0.,2.,8.],[2.,0.,3.],[8.,3.,0.]])
        k=np.exp(-np.log(2)*d/4); np.fill_diagonal(k,0)
        for q in [0.,1.]:
            w,s=normalized_colonization(a,d,4,q=q)
            for lr in np.log([.01,1.,10.,100.]):
                result=scaled_ratio_sweep(w,s,[lr])[0]
                scalar=run_continuous_equilibrium_from_kernel(a,k,r=np.exp(lr),self_colonization=q,b=1,x=.5,max_iter=100000,tol=1e-10)
                self.assertTrue(result.converged and scalar.converged)
                np.testing.assert_allclose(result.probabilities,scalar.probabilities,atol=3e-9)

    def test_underflowing_raw_kernel_has_known_equilibrium(self):
        a=np.ones(2); d=np.array([[0.,2000.],[2000.,0.]])
        w,s=normalized_colonization(a,d,1.)
        lr=-np.log(2)*2000+np.log(.25)
        result=scaled_ratio_sweep(w,s,[lr])[0]
        np.testing.assert_allclose(result.probabilities,.75,atol=1e-9)
        self.assertTrue(result.converged)

    def test_capacity_and_lower_bound_for_two_equal_patches(self):
        a=np.array([4.,4.]); d=np.array([[0.,2.],[2.,0.]])
        for q in [0.,1.]:
            cap=capacity_bracket(a,d,2,q=q)
            critical=8*(.5+q)
            self.assertAlmostEqual(cap.value,critical)
            self.assertLessEqual(cap.lower,critical+1e-12)
            self.assertGreaterEqual(cap.upper,critical-1e-12)
            w,s=normalized_colonization(a,d,2,q=q)
            report,lo,hi=lower_occupancy_bracket(w,s)
            self.assertLessEqual(report['log_r_feasible'],np.log(.01*critical)+1e-7)
            self.assertGreaterEqual(report['log_r_infeasible'],np.log(.01*critical)-1e-7)
            self.assertGreaterEqual(lo.mean(),.99); self.assertLess(hi.mean(),.99)

    def test_self_colonization_survives_short_dispersal_limit(self):
        a=np.array([1.,4.]); d=np.array([[0.,2000.],[2000.,0.]])
        w,s=normalized_colonization(a,d,.1,q=1)
        result=scaled_ratio_sweep(w,s,[np.log(.2)])[0]
        np.testing.assert_allclose(result.probabilities,[.8,.975],atol=1e-9)

    def test_scientific_ratio_preserves_values_beyond_float_range(self):
        value=scientific_ratio(-1000.)
        self.assertIn('e-435',value)

if __name__=='__main__': unittest.main()
