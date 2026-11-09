import numpy as np
import unittest
from scipy.stats import spearmanr, rankdata
from src.tools.weighted_rank_associations import balanced_spearman, weighted_midranks


def test_uniform_matches_scipy_with_ties_and_missing():
    x=np.array([1.,2,2,4,5,7,8,9]);y=np.array([4.,3,3,1,5,8,7,np.nan])
    rho=balanced_spearman(x,y,np.zeros(8))
    assert abs(rho-spearmanr(x[:-1],y[:-1]).statistic)<1e-14


def test_integer_weights_equal_literal_replication_and_equal_scenarios():
    x=np.array([1,2,2,3,4,5,6,7.]);y=np.array([8,4,4,7,3,1,2,6.]);g=np.repeat([0,1],4)
    # Valid totals 4 and 8: multiply the first group's frequencies by 2.
    f=np.array([1,2,1,0,1,3,2,2]);ix=np.repeat(np.arange(8),f*np.where(g==0,2,1))
    assert abs(balanced_spearman(x,y,g,f)-spearmanr(x[ix],y[ix]).statistic)<1e-14
    assert abs(balanced_spearman(x,y,g,f)-balanced_spearman(x,y,g,f*np.where(g==0,9,1)))<1e-14


def test_undefined_scenario_constant_response_and_zero_weight_extremes():
    x=np.arange(8.);g=np.repeat([0,1],4);y=x.copy()
    assert np.isnan(balanced_spearman(x,np.ones(8),g))
    f=np.array([1,1,0,0,1,1,1,1]);assert np.isnan(balanced_spearman(x,y,g,f))
    y[:]=0;y[-1]=1;f=np.ones(8);f[-1]=0
    assert np.isnan(balanced_spearman(x,y,g,f))
    y=np.arange(8.)*1e-12;assert np.isnan(balanced_spearman(x,y,g))


def test_broadcast_missing_ties_and_rank_definition():
    x=np.array([[2,1,1,4],[1,2,3,4.]])
    ranks=weighted_midranks(x,np.ones(4))
    np.testing.assert_allclose(ranks,rankdata(x,axis=-1)-.5)
    y=np.array([4,3,2,1.]);rho=balanced_spearman(x,y,np.zeros(4),minimum_per_group=3)
    np.testing.assert_allclose(rho,[spearmanr(z,y).statistic for z in x],atol=1e-14)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(f) for n,f in globals().items() if n.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
