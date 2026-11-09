"""Source-reading fixtures, independent of the real figure's extracted answers."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "code/analyses/02_metapop/02_03_main_analyses/02_03_02_species_dependence/11_digitize_bird_figure2.py"
spec = importlib.util.spec_from_file_location("bird_digitization", PATH)
bird = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bird)


class BirdFigureDigitizationTests(unittest.TestCase):
    def test_probability_axis_is_inverted_and_coordinates_recover(self):
        result = bird.calibrate([0, .2, .4, .6, .8, 1], [210, 170, 130, 90, 50, 10])
        self.assertAlmostEqual(result["slope_pixels_per_unit"], -200)
        self.assertAlmostEqual((90 - result["intercept_pixels"]) / result["slope_pixels_per_unit"], .6)
        self.assertLess(result["maximum_tick_fit_residual_pixels"], 1e-12)

    def test_decorative_green_component_does_not_replace_curve(self):
        image = np.full((70, 100, 3), 244, dtype=np.uint8)
        image[45, :, :] = [0, 128, 0]
        image[5:20, 5:30, :] = [0, 128, 0]
        mask = bird.curve_mask(image)
        self.assertEqual(int(mask.sum()), 100)
        self.assertTrue(mask[45].all())
        self.assertFalse(mask[5:20].any())

    def test_thick_green_line_does_not_split_band(self):
        column = np.full((100, 3), 244., dtype=float)
        column[20:81] = 214
        column[40:61] = [0, 128, 0]
        band = bird.band_at_column(column, np.arange(100), np.arange(40, 61),
                                   {"grey_chroma_max": 12, "grey_intensity_min": 150,
                                    "grey_intensity_max": 225, "merge_vertical_gaps_pixels": 8})
        self.assertEqual(band["upper_y"], 20)
        self.assertEqual(band["lower_y"], 80)
        self.assertEqual(band["neutral_pixels"], 40)

    def test_grid_sampling_preserves_occlusion_and_refuses_extrapolation(self):
        trace = {"H_source_pixel": .2, "H_per_pixel": .005, "psi_per_pixel": .005,
                 "psi_figure2_curve": .4, "lower_available": True,
                 "psi_ci_lower_digitized": .2, "lower_status": "source_trace",
                 "lower_axis_resolution_snap": False, "upper_available": False,
                 "psi_ci_upper_digitized": "", "upper_status": "potential_illustration_occlusion",
                 "upper_axis_resolution_snap": False}
        source = {"species_id": "fixture", "species_as_published": "Fixture species", "H": .2, "psi_reconstructed": .41}
        cfg = {"nominal_coverage_assumed": .95, "band_definition_verified": False}
        result = bird.nearest_grid_row([trace], source, cfg)
        self.assertTrue(result["lower_available"])
        self.assertFalse(result["upper_available"])
        self.assertEqual(result["psi_ci_upper_digitized"], "")
        self.assertFalse(result["band_definition_verified"])
        outside = bird.nearest_grid_row([trace], {**source, "H": .21}, cfg)
        self.assertFalse(outside["figure2_curve_available"])
        self.assertEqual(outside["psi_figure2_curve"], "")
        self.assertEqual(outside["upper_status"], "outside_native_pixel_support")
