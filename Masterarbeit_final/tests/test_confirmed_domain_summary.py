"""Summary guards for matched landscapes and undefined model domains."""
import importlib.util
from pathlib import Path
import numpy as np
import unittest


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'Masterarbeit_final').is_dir())
SOURCE = ROOT / 'Masterarbeit_final/code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence/11_confirmed_domain_summary.py'
spec = importlib.util.spec_from_file_location('confirmed_summary_under_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_landscape_quantiles_and_minimum_count():
    values = np.broadcast_to(np.arange(5)[:, None, None, None] / 5, (5, 2, 2, 2)).copy()
    values[:2, 1, 0] = np.nan
    values[:3, 1, 1] = np.nan
    counts, quantiles = module.statistics(values)
    np.testing.assert_array_equal(counts, [[5, 5], [3, 2]])
    np.testing.assert_allclose(quantiles[:, 0, 0, 0], [.04, .4, .76], atol=1e-15)
    np.testing.assert_allclose(quantiles[:, 1, 0, 1], [.42, .6, .78], atol=1e-15)
    assert np.isnan(quantiles[:, 1, 1]).all()


def test_extinction_is_zero_and_undefined_landscapes_stay_missing():
    values = np.zeros((5, 1, 2, 2))
    values[:, :, 1] = np.nan
    counts, quantiles = module.statistics(values)
    np.testing.assert_array_equal(counts, [[5, 0]])
    assert (quantiles[:, 0, 0] == 0).all()
    assert np.isnan(quantiles[:, 0, 1]).all()


def test_inconsistent_parameter_missingness_is_rejected():
    values = np.zeros((5, 1, 1, 2))
    values[0, 0, 0, 1] = np.nan
    with unittest.TestCase().assertRaises(AssertionError):
        module.statistics(values)


class ConfirmedDomainSummaryTests(unittest.TestCase):
    test_landscape_quantiles_and_minimum_count = staticmethod(test_landscape_quantiles_and_minimum_count)
    test_extinction_is_zero_and_undefined_landscapes_stay_missing = staticmethod(test_extinction_is_zero_and_undefined_landscapes_stay_missing)
    test_inconsistent_parameter_missingness_is_rejected = staticmethod(test_inconsistent_parameter_missingness_is_rejected)


if __name__ == "__main__":
    unittest.main()
