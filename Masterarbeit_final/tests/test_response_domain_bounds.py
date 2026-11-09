"""Guard sound interval certificates used by adaptive boundary refinement."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.metapop.response_domain import parameter_order,quantile_bounds,envelope_certificate,cells_requiring_evaluation,boundary_bracket


class BoundTests(unittest.TestCase):
    def test_kernel_order_holds_at_all_distances(self):
        e=np.array([1.,4.,16.]);r=np.log([.1,10,1000])
        for ell in [1,2,8,32]:
            for lr in np.log([1e-8,.1,1000,1e8]):
                more,less=parameter_order(e,r,ell,lr,282.9)
                d=np.linspace(0,282.9,1001)
                difference=np.log(2)*d[:,None]*(1/e-1/ell)+r-lr
                self.assertTrue((difference[:,more]>=-1e-12).all())
                self.assertTrue((difference[:,less]<=1e-12).all())

    def test_complements_need_separate_witnesses(self):
        lower=np.array([[0,.98],[0,.98]])
        upper=np.array([[.9,1],[.9,1]])
        self.assertEqual(envelope_certificate(lower,upper)['category'],1)
        self.assertFalse(cells_requiring_evaluation(lower,upper).any())

    def test_all_saturation_cells_must_be_resolved(self):
        lo=np.array([[.94,.97],[.96,.98]]);hi=np.array([[.99,.98],[1,1]])
        self.assertEqual(envelope_certificate(lo,hi)['category'],3)
        np.testing.assert_array_equal(cells_requiring_evaluation(lo,hi),[True,False])

    def test_both_quantile_bounds_are_preserved(self):
        q=np.array([[[.6,.8],[.9,1]],[[.1,.2],[.3,.4]]])
        lo,hi=quantile_bounds([4,4],np.log([1,100]),q,q,4,np.log(10),300)
        np.testing.assert_array_equal(lo,q[1]);np.testing.assert_array_equal(hi,q[0])

    def test_contradiction_is_not_a_waiver(self):
        q=np.array([[[.2],[.3]],[[.8],[.9]]])
        with self.assertRaises(ValueError):quantile_bounds([4,4],np.log([1,100]),q,q,4,np.log(10),300)

    def test_window_endpoint_is_not_observed_boundary(self):
        result=boundary_bracket([1,2],[0,1],[1,1],4,'upper',[-12,7])
        self.assertEqual(result,(1.,7.,True,False))

    def test_quantile_bounds_contain_known_two_patch_equilibria(self):
        # Equal-area, two-patch systems have p*=max(1-r/(A^1.5 K),0).
        rng=np.random.default_rng(20261013)
        distances=rng.uniform(1,280,(20,4));areas=rng.uniform(1,1000,(20,4))
        def exact_quantiles(ell,lr):
            log_capacity=1.5*np.log(areas)-np.log(2)*distances/ell
            p=-np.expm1(np.minimum(lr-log_capacity,0))
            return np.quantile(p,[.05,.95],axis=0,method='linear')
        e=np.exp(rng.uniform(0,np.log(4096),25));r=rng.uniform(-12,7,25)*np.log(10)
        q=np.asarray([exact_quantiles(ell,lr) for ell,lr in zip(e,r)])
        for _ in range(100):
            ell=float(np.exp(rng.uniform(0,np.log(4096))));lr=float(rng.uniform(-12,7)*np.log(10))
            lo,hi=quantile_bounds(e,r,q,q,ell,lr,282.9);truth=exact_quantiles(ell,lr)
            self.assertTrue((truth>=lo-1e-14).all());self.assertTrue((truth<=hi+1e-14).all())


if __name__=='__main__':unittest.main()
