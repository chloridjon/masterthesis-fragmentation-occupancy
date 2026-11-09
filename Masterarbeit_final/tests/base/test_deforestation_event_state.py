from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from src.landscape.deforestation import (
    FOREST,
    FOUR_NEIGHBOR_KERNEL,
    DeforestationEventState,
    _neighbor_count,
    candidates_edge,
    weights_accessibility_front,
)


def _fragmented_grid(
    shape: tuple[int, int],
    habitat_fraction: float,
    seed: int,
    sigma: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = rng.standard_normal(shape)
    if sigma > 0.0:
        field = gaussian_filter(field, sigma=sigma, mode="wrap")
    threshold = np.quantile(field, 1.0 - habitat_fraction)
    return (field >= threshold).astype(np.int8)


class DeforestationEventStateTests(unittest.TestCase):
    def assert_state_matches_full(self, state: DeforestationEventState) -> None:
        expected_neighbors = _neighbor_count(
            state.grid,
            FOUR_NEIGHBOR_KERNEL,
            boundary_mode=state.boundary_mode,
        )
        np.testing.assert_array_equal(state.neighbor_count, expected_neighbors)

        expected_edge = candidates_edge(
            state.grid,
            boundary_mode=state.boundary_mode,
            neighbor_count=expected_neighbors,
            count_canvas_boundary_as_edge=state.count_canvas_boundary_as_edge,
        )
        np.testing.assert_array_equal(state.edge_candidate_mask, expected_edge)
        self.assertEqual(
            set(map(int, state.forest_indices)),
            set(map(int, np.flatnonzero(state.grid == FOREST))),
        )
        self.assertEqual(
            set(map(int, state.edge_candidate_indices)),
            set(map(int, np.flatnonzero(expected_edge))),
        )

        if state.accessibility_enabled:
            expected_weights = weights_accessibility_front(
                state.grid,
                candidates=expected_edge,
                k=state.k,
                lambda_decay=state.lambda_decay,
                eta=state.eta,
                boundary_mode=state.boundary_mode,
                neighbor_count=expected_neighbors,
            )
            np.testing.assert_allclose(
                state.accessibility_weights,
                expected_weights,
                rtol=1e-12,
                atol=1e-12,
            )
            expected_total = float(
                expected_weights[np.isfinite(expected_weights) & (expected_weights > 0.0)].sum()
            )
            self.assertAlmostEqual(
                state.weighted_sampler.total,
                expected_total,
                delta=1e-9 * max(1.0, expected_total),
            )

    def test_periodic_200x200_incremental_sequences_match_full_recomputation(self) -> None:
        # Together these cases exercise all requested k, lambda and eta values,
        # several habitat fractions, and both fine and clustered fragmentation.
        cases = (
            (0.25, 0.0, 1, 0.1, 0.5),
            (0.55, 1.5, 3, 0.5, 1.0),
            (0.80, 4.0, 5, 1.0, 2.0),
        )
        for case_index, (habitat, sigma, k, decay, eta) in enumerate(cases):
            with self.subTest(habitat=habitat, sigma=sigma, k=k, decay=decay, eta=eta):
                grid = _fragmented_grid((200, 200), habitat, 100 + case_index, sigma)
                state = DeforestationEventState(
                    grid,
                    mode="access_weighted",
                    boundary_mode="periodic",
                    k=k,
                    lambda_decay=decay,
                    eta=eta,
                )
                rng = np.random.default_rng(1000 + case_index)
                for event_index in range(200):
                    before = int((state.grid == FOREST).sum())
                    event_mode = ("random", "edge_random", "access_weighted")[
                        event_index % 3
                    ]
                    changed = state.deforest_one(rng, mode=event_mode)
                    self.assertIsNotNone(changed)
                    self.assertEqual(int((state.grid == FOREST).sum()), before - 1)
                    self.assert_state_matches_full(state)

    def test_parameter_matrix_and_mixed_mechanisms(self) -> None:
        # A denser cross-product on smaller grids catches interactions without
        # making the normal unit-test suite prohibitively slow.
        for k in (1, 3, 5):
            for decay in (0.1, 0.5, 1.0):
                for eta in (0.5, 1.0, 2.0):
                    with self.subTest(k=k, decay=decay, eta=eta):
                        state = DeforestationEventState(
                            _fragmented_grid((32, 37), 0.62, 10 + k, 1.2),
                            mode="random",
                            boundary_mode="periodic",
                            k=k,
                            lambda_decay=decay,
                            eta=eta,
                        )
                        rng = np.random.default_rng(500 + k)
                        for event_index in range(12):
                            mode = ("random", "edge_random", "access_weighted")[event_index % 3]
                            self.assertIsNotNone(state.deforest_one(rng, mode=mode))
                            self.assert_state_matches_full(state)
                        self.assertEqual(state.mode, "random")
                        self.assertTrue(state.accessibility_enabled)

    def test_open_boundary_and_no_candidate_result(self) -> None:
        full = np.ones((9, 11), dtype=np.int8)
        state = DeforestationEventState(full, mode="edge_random", boundary_mode="open")
        self.assertIsNone(state.deforest_one(np.random.default_rng(1)))
        self.assert_state_matches_full(state)

        changed = state.deforest_one(np.random.default_rng(2), mode="random")
        self.assertIsNotNone(changed)
        self.assert_state_matches_full(state)
        self.assertIsNotNone(state.deforest_one(np.random.default_rng(3), mode="edge_random"))
        self.assert_state_matches_full(state)

    def _assert_empirical_distribution(
        self,
        grid: np.ndarray,
        mode: str,
        expected_weights: np.ndarray,
        draws: int,
    ) -> None:
        expected_flat = np.flatnonzero(expected_weights > 0.0)
        probabilities = expected_weights.reshape(-1)[expected_flat].astype(float)
        probabilities /= probabilities.sum()
        lookup = {int(flat): position for position, flat in enumerate(expected_flat)}
        counts = np.zeros(expected_flat.size, dtype=int)

        for seed in range(draws):
            state = DeforestationEventState(
                grid,
                mode=mode,
                boundary_mode="periodic",
                k=3,
                lambda_decay=0.5,
                eta=1.0,
            )
            changed = state.deforest_one(np.random.default_rng(seed))
            self.assertIsNotNone(changed)
            counts[lookup[int(changed)]] += 1

        expected_counts = draws * probabilities
        standardized = np.abs(counts - expected_counts) / np.sqrt(
            expected_counts * (1.0 - probabilities)
        )
        self.assertLess(float(standardized.max()), 5.0)

    def test_one_draw_selection_frequencies(self) -> None:
        grid = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 0, 0, 1, 1, 1],
                [1, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int8,
        )
        edge = candidates_edge(
            grid,
            boundary_mode="periodic",
            count_canvas_boundary_as_edge=False,
        )
        self._assert_empirical_distribution(grid, "edge_random", edge.astype(float), 8_000)

        access = weights_accessibility_front(
            grid,
            candidates=edge,
            k=3,
            lambda_decay=0.5,
            eta=1.0,
            boundary_mode="periodic",
        )
        self._assert_empirical_distribution(grid, "access_weighted", access, 8_000)


if __name__ == "__main__":
    unittest.main()
