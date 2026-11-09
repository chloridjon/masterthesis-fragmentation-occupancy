from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import sparse

from src.metapop.hanski import (
    HanskiConfig,
    metapopulation_capacity_and_mean_si_from_kernel,
    rates_from_ratio_speed,
    ratio_speed_from_rates,
    resolve_rate_parameters,
    run_continuous_equilibrium_from_kernel,
    run_discrete_equilibrium_from_kernel,
    run_expectation_from_kernel,
)


class MetapopParameterizationTests(unittest.TestCase):
    def test_ratio_speed_round_trip(self) -> None:
        c, e0 = rates_from_ratio_speed(r=0.25, s=0.8)
        self.assertAlmostEqual(c, 0.8)
        self.assertAlmostEqual(e0, 0.2)
        recovered_r, recovered_s = ratio_speed_from_rates(c, e0)
        self.assertAlmostEqual(recovered_r, 0.25)
        self.assertAlmostEqual(recovered_s, 0.8)

    def test_hanski_config_from_ratio_speed_matches_direct_config(self) -> None:
        ratio_speed = HanskiConfig.from_ratio_speed(r=0.2, s=1.0, alpha=0.3, x=0.7)
        direct = HanskiConfig(c=1.0, e0=0.2, alpha=0.3, x=0.7)

        self.assertEqual(ratio_speed, direct)
        self.assertAlmostEqual(ratio_speed.r, 0.2)
        self.assertAlmostEqual(ratio_speed.s, 1.0)

    def test_ratio_speed_and_direct_rates_produce_same_dynamics(self) -> None:
        areas = np.array([1.0, 2.0, 3.0])
        kernel = np.array(
            [
                [0.0, 0.4, 0.2],
                [0.4, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ]
        )
        common = {"b": 1.0, "x": 0.5, "steps": 20, "initial": 0.5}

        from_ratio_speed = run_expectation_from_kernel(areas, kernel, r=0.25, s=0.8, **common)
        from_direct_rates = run_expectation_from_kernel(areas, kernel, c=0.8, e0=0.2, **common)

        self.assertEqual(from_ratio_speed, from_direct_rates)

    def test_rate_parameterizations_cannot_be_mixed_or_incomplete(self) -> None:
        with self.assertRaisesRegex(ValueError, "either r/s or c/e0"):
            resolve_rate_parameters(c=1.0, e0=0.2, r=0.2, s=1.0)
        with self.assertRaisesRegex(ValueError, "requires both r and s"):
            resolve_rate_parameters(r=0.2)

    def test_continuous_equilibrium_satisfies_rate_balance(self) -> None:
        areas = np.array([1.0, 2.0, 3.0])
        kernel = np.array(
            [
                [0.0, 0.4, 0.2],
                [0.4, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ]
        )
        result = run_continuous_equilibrium_from_kernel(
            areas,
            kernel,
            r=0.25,
            b=1.0,
            x=0.5,
            tol=1e-12,
        )
        connectivity = kernel @ (areas * result.probabilities)
        residual = (1.0 - result.probabilities) * connectivity - (
            0.25 * areas ** -0.5 * result.probabilities
        )

        self.assertTrue(result.converged)
        self.assertLess(float(np.max(np.abs(residual))), 1e-10)

    def test_discrete_equilibrium_matches_long_expectation_run(self) -> None:
        areas = np.array([1.0, 2.0, 3.0])
        kernel = np.array(
            [
                [0.0, 0.4, 0.2],
                [0.4, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ]
        )
        equilibrium = run_discrete_equilibrium_from_kernel(
            areas,
            kernel,
            r=0.25,
            s=0.01,
            b=1.0,
            x=0.5,
            self_colonization=1.0,
            tol=1e-12,
        )
        occupancy, _ = run_expectation_from_kernel(
            areas,
            kernel,
            r=0.25,
            s=0.01,
            b=1.0,
            x=0.5,
            self_colonization=1.0,
            steps=100_000,
            initial=1.0,
            min_steps=1,
            tol=1e-12,
        )

        self.assertTrue(equilibrium.converged)
        self.assertAlmostEqual(equilibrium.occupancy_fraction, occupancy, places=9)

    def test_schnell_self_colonization_supports_a_single_patch(self) -> None:
        areas = np.array([4.0])
        kernel = np.zeros((1, 1), dtype=float)

        without_self = run_continuous_equilibrium_from_kernel(
            areas,
            kernel,
            r=1.0,
            b=1.0,
            x=0.5,
            self_colonization=0.0,
            tol=1e-12,
        )
        with_self = run_continuous_equilibrium_from_kernel(
            areas,
            kernel,
            r=1.0,
            b=1.0,
            x=0.5,
            self_colonization=1.0,
            tol=1e-12,
        )

        self.assertAlmostEqual(without_self.occupancy_fraction, 0.0)
        self.assertTrue(with_self.converged)
        self.assertAlmostEqual(with_self.occupancy_fraction, 0.875, places=10)

    def test_schnell_capacity_diagonal_is_area_to_x_plus_b(self) -> None:
        areas = np.array([4.0])
        kernel = np.zeros((1, 1), dtype=float)

        capacity, mean_si, matrix = metapopulation_capacity_and_mean_si_from_kernel(
            areas,
            kernel,
            b=1.0,
            x=0.5,
            self_colonization=1.0,
            return_matrix=True,
        )
        symmetric, symmetric_mean_si, _ = metapopulation_capacity_and_mean_si_from_kernel(
            areas,
            kernel,
            b=1.0,
            x=0.5,
            self_colonization=1.0,
            method="symmetric",
        )
        sparse_capacity, sparse_mean_si, sparse_matrix = metapopulation_capacity_and_mean_si_from_kernel(
            areas,
            sparse.csr_matrix(kernel),
            b=1.0,
            x=0.5,
            self_colonization=1.0,
            return_matrix=True,
        )

        self.assertAlmostEqual(capacity.lambda_max, 8.0, places=10)
        self.assertAlmostEqual(mean_si, 4.0, places=10)
        np.testing.assert_allclose(matrix, np.array([[8.0]]))
        self.assertAlmostEqual(symmetric.lambda_max, capacity.lambda_max, places=10)
        self.assertAlmostEqual(symmetric_mean_si, mean_si, places=10)
        self.assertAlmostEqual(sparse_capacity.lambda_max, capacity.lambda_max, places=10)
        self.assertAlmostEqual(sparse_mean_si, mean_si, places=10)
        np.testing.assert_allclose(sparse_matrix.toarray(), np.array([[8.0]]))

    def test_self_colonization_rejects_negative_or_nonfinite_values(self) -> None:
        for value in (-0.1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "self_colonization"):
                    HanskiConfig(self_colonization=value)

    def test_ratio_speed_rejects_invalid_values(self) -> None:
        for r, s in [(-0.1, 1.0), (0.2, 0.0), (0.2, -1.0), (math.nan, 1.0), (0.2, math.inf)]:
            with self.subTest(r=r, s=s):
                with self.assertRaises(ValueError):
                    rates_from_ratio_speed(r, s)

    def test_direct_conversion_requires_positive_c(self) -> None:
        with self.assertRaisesRegex(ValueError, "c must be finite and > 0"):
            ratio_speed_from_rates(0.0, 0.2)


if __name__ == "__main__":
    unittest.main()
