"""Scientific fixtures for the empirical fallback reconstruction."""
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence/11_reconstruct_bird_curves.py"
spec = importlib.util.spec_from_file_location("bird_reconstruction", PATH)
bird = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bird)


class BirdReconstructionTests(unittest.TestCase):
    def test_known_logit_curve_recovers_coefficients_and_threshold(self):
        # logit(psi)=-2+5H crosses psi=.5 at H=.4.
        b0, b1 = bird.coefficients(bird.expit(-1), 0.5)
        self.assertAlmostEqual(b0, -2, delta=1e-13)
        self.assertAlmostEqual(b1, 5, delta=1e-13)
        self.assertAlmostEqual(bird.threshold(bird.expit(-1), 0.5, 0.5), 0.4, delta=1e-13)

    def test_boundary_and_non_increasing_anchors_are_rejected(self):
        for p20, p40 in [(0, 0.5), (0.2, 1), (0.4, 0.4), (0.6, 0.4)]:
            with self.subTest(p20=p20, p40=p40), self.assertRaises(ValueError):
                bird.coefficients(p20, p40)

    def test_rounding_rectangle_bounds_cover_internal_thresholds(self):
        for p20, p40 in [(0.05, 0.13), (0.15, 0.6), (0.42, 0.98)]:
            lower, upper = bird.rounding_threshold_bounds(p20, p40, 0.005)
            # Check inside and beyond-anchor cases over the full rectangle.
            for i in range(21):
                for j in range(21):
                    h = bird.threshold(p20 - 0.005 + i * 0.0005, p40 - 0.005 + j * 0.0005)
                    self.assertGreaterEqual(h, lower - 1e-14)
                    self.assertLessEqual(h, upper + 1e-14)

    def test_rounding_rectangle_cannot_cross_zero_slope(self):
        with self.assertRaises(ValueError):
            bird.rounding_threshold_bounds(0.50, 0.51, 0.005)
