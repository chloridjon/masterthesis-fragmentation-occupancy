"""Scientific comparison fixtures for parameter-space bird matching."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence/11_bird_parameter_surfaces.py'
spec = importlib.util.spec_from_file_location('bird_surfaces', PATH)
bird = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bird)


class BirdParameterSurfaceTests(unittest.TestCase):
    def test_curve_score_cannot_switch_scenario_at_each_H(self):
        # Each scenario disagrees at one H, despite a perfect pointwise envelope.
        score = bird.curve_rmse([[0, 1], [1, 0]], [0, 0])
        np.testing.assert_allclose(score, np.sqrt(.5))
        self.assertGreater(float(score.min()), 0)

    def test_unavailable_response_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            bird.curve_rmse([0, np.nan], [0, 0])

    def test_scenario_interpolation_precedes_minimum_and_has_no_extrapolation(self):
        points = np.array([[0, 0], [1, 0], [0, 1]])
        scenarios = np.array([[0, 1, 1], [1, 0, 0]])
        fields, envelope, _ = bird.interpolate_scenarios(points, scenarios, np.array([[.25, .25], [.9, .9]]))
        np.testing.assert_allclose(fields[:, 0], [.5, .5])
        self.assertAlmostEqual(envelope[0], .5)
        self.assertTrue(np.isnan(envelope[1]))
        # Interpolating the sampled minima instead would incorrectly return zero.
        self.assertTrue(np.all(scenarios.min(axis=0) == 0))

    def test_log_coordinates_use_both_declared_design_ranges(self):
        profiles = pd.DataFrame({'half_distance': [1, 10, 100], 'r': [1, 100, 10000]})
        points, _, _ = bird.normalized_coordinates(profiles,
            {'half_distance_min': 1, 'half_distance_max': 100, 'log10_r_min': 0, 'log10_r_max': 4})
        np.testing.assert_allclose(points, [[0, 0], [.5, .5], [1, 1]])
