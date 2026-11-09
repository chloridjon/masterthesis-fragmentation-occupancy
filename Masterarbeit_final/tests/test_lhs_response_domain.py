"""Protect candidate-domain inference and landscape missingness semantics."""
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

FINAL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FINAL/'code/analyses/02_metapop/02_02_species_parameter_selection'))
D=importlib.import_module('09_lhs_response_domain')


class CandidateDomainChecks(unittest.TestCase):
    def test_plateaus_and_full_or_partial_responses(self):
        # Flat intermediate responses qualify; this rule is not a variation test.
        np.testing.assert_array_equal(D.classify([.96,0.,0.,.5],[1.,.04,1.,.5],.05),[0,2,1,1])

    def test_partial_order_leaves_unsupported_gap_unknown(self):
        points=np.array([[.5,.2],[.5,.8]])
        lower=np.array([.99,0.]);upper=np.array([1.,.01])
        targets=np.array([[.7,.1],[.3,.9],[.5,.5]])
        np.testing.assert_array_equal(D.monotone_classes(points,lower,upper,targets,.05),[0,2,3])

    def test_conflicting_dominance_is_rejected(self):
        points=np.array([[.2,.8],[.8,.2]])
        with self.assertRaises(ValueError):
            D.monotone_classes(points,np.array([.99,0.]),np.array([1.,.01]),np.array([[.5,.5]]),.05)

    def test_missing_landscapes_are_not_extinction(self):
        values=np.array([.2,.5,.8,np.nan]).reshape(4,1,1,1)
        count,q=D.quantiles_by_valid_pattern(values,3)
        np.testing.assert_array_equal(count,[[3]])
        np.testing.assert_allclose(q.ravel(),[.23,.77])
        _,missing=D.quantiles_by_valid_pattern(values,4)
        self.assertTrue(np.isnan(missing).all())

    def test_missingness_cannot_depend_on_species_parameters(self):
        values=np.ones((3,1,1,2));values[0,0,0,1]=np.nan
        with self.assertRaises(AssertionError):D.quantiles_by_valid_pattern(values,3)

    def test_unbracketed_crossings_remain_missing(self):
        points=np.array([[.5,.3]])
        brackets=D.boundary_brackets(points,np.array([.99]),np.array([1.]),[.2,.8],.05,[0.,0.],[1.,1.])
        self.assertFalse(brackets.lower_boundary_bracket_complete.any())
        self.assertFalse(brackets.upper_boundary_bracket_complete.any())
        self.assertTrue(brackets.upper_boundary_bracket_high.isna().all())


if __name__=='__main__':unittest.main()
