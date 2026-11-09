from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from src.landscape.deforestation import DeforestationEventState
from src.landscape.generation import (
    SEQUENTIAL_DEFORESTATION_GENERATOR_VERSION,
    SequentialDeforestationBlockedError,
    SequentialDeforestationController,
    run_sequential_deforestation,
)


class SequentialDeforestationTests(unittest.TestCase):
    def test_incremental_controller_matches_batch_trajectory_exactly(self) -> None:
        shape = (31, 37)
        seed = 2048
        n_events = 417
        kwargs = {
            "rho": 0.99,
            "directed_mode": "access_weighted",
            "generation_boundary": "periodic",
            "k": 4,
            "lambda_decay": 0.7,
            "eta": 1.2,
            "seed": seed,
        }
        batch = run_sequential_deforestation(
            np.ones(shape, dtype=np.int8),
            np.random.default_rng(seed),
            target_habitat_fraction=(np.prod(shape) - n_events) / np.prod(shape),
            **kwargs,
        )
        controller = SequentialDeforestationController(
            np.ones(shape, dtype=np.int8),
            np.random.default_rng(seed),
            **kwargs,
        )
        events = [controller.step() for _ in range(n_events)]

        np.testing.assert_array_equal(controller.grid, batch.grid)
        np.testing.assert_array_equal(
            np.asarray([event.changed_flat_index for event in events]),
            batch.changed_flat_indices,
        )
        self.assertEqual(tuple(event.mode for event in events), batch.event_modes)
        self.assertEqual(controller.n_random_forced_initial, batch.n_random_forced_initial)
        self.assertEqual(controller.n_random_stochastic, batch.n_random_stochastic)
        self.assertEqual(controller.n_directed, batch.n_directed)
        self.assertEqual(controller.metadata()["event_count"], n_events)
        self.assertEqual(controller.metadata()["H"], batch.final_habitat_fraction)

    def test_exact_event_count_on_200x200_grid(self) -> None:
        target = 0.73125
        total = 200 * 200
        expected_final = int(np.floor(target * total + 0.5))
        result = run_sequential_deforestation(
            np.ones((200, 200), dtype=np.int8),
            np.random.default_rng(100),
            rho=0.9,
            directed_mode="edge_random",
            target_habitat_fraction=target,
        )
        self.assertEqual(result.final_forest_cells, expected_final)
        self.assertEqual(result.n_events, total - expected_final)
        self.assertEqual(result.changed_flat_indices.size, result.n_events)
        self.assertEqual(np.unique(result.changed_flat_indices).size, result.n_events)
        self.assertEqual(result.L, 200)
        self.assertEqual(result.grid_shape, (200, 200))
        self.assertEqual(result.generation_boundary, "periodic")
        self.assertEqual(result.generator_version, SEQUENTIAL_DEFORESTATION_GENERATOR_VERSION)
        self.assertTrue(result.forced_initial_random)

    def test_first_event_is_forced_random_for_all_requested_rho_values(self) -> None:
        for rho in (0.0, 0.9, 0.99, 0.999, 1.0):
            with self.subTest(rho=rho):
                result = run_sequential_deforestation(
                    np.ones((20, 20), dtype=np.int8),
                    np.random.default_rng(11),
                    rho=rho,
                    directed_mode="access_weighted",
                    target_habitat_fraction=0.98,
                )
                self.assertEqual(result.event_modes[0], "random")
                self.assertEqual(result.n_random_forced_initial, 1)
                self.assertEqual(
                    result.n_random_total,
                    result.n_random_forced_initial + result.n_random_stochastic,
                )

    def test_limiting_cases_and_initial_nucleation(self) -> None:
        random_only = run_sequential_deforestation(
            np.ones((25, 25), dtype=np.int8),
            np.random.default_rng(7),
            rho=0.0,
            directed_mode="edge_random",
            target_habitat_fraction=0.8,
        )
        self.assertEqual(set(random_only.event_modes), {"random"})
        self.assertEqual(random_only.n_directed, 0)
        self.assertEqual(random_only.n_random_stochastic, random_only.n_events - 1)
        self.assertEqual(random_only.expected_events_per_stochastic_random, 1.0)

        directed_only = run_sequential_deforestation(
            np.ones((25, 25), dtype=np.int8),
            np.random.default_rng(8),
            rho=1.0,
            directed_mode="access_weighted",
            target_habitat_fraction=0.8,
        )
        self.assertEqual(directed_only.event_modes[0], "random")
        self.assertEqual(set(directed_only.event_modes[1:]), {"access_weighted"})
        self.assertEqual(directed_only.n_random_stochastic, 0)
        self.assertEqual(directed_only.n_directed, directed_only.n_events - 1)
        self.assertTrue(math.isinf(directed_only.expected_events_per_stochastic_random))

        state = DeforestationEventState(
            np.ones((30, 30), dtype=np.int8),
            mode="edge_random",
            boundary_mode="periodic",
        )
        self.assertEqual(len(state.edge_candidate_indices), 0)
        self.assertIsNotNone(state.deforest_one(np.random.default_rng(9), mode="random"))
        self.assertGreater(len(state.edge_candidate_indices), 0)

    def test_mechanism_frequencies_follow_binomial_rho(self) -> None:
        # Pool independent trajectories. The tolerance is five binomial
        # standard deviations plus one discrete observation.
        for rho in (0.9, 0.99, 0.999):
            directed = 0
            post_initial = 0
            for replicate in range(20):
                result = run_sequential_deforestation(
                    np.ones((20, 20), dtype=np.int8),
                    np.random.default_rng(10_000 + replicate),
                    rho=rho,
                    directed_mode="edge_random",
                    target_habitat_fraction=0.0,
                )
                directed += result.n_directed
                post_initial += result.n_events - 1
            observed = directed / post_initial
            standard_error = math.sqrt(rho * (1.0 - rho) / post_initial)
            self.assertLessEqual(abs(observed - rho), 5.0 * standard_error + 1.0 / post_initial)

    def test_snapshots_are_exact_nested_and_from_one_trajectory(self) -> None:
        requested = [0.9, 0.8, 0.7, 0.6]
        result = run_sequential_deforestation(
            np.ones((40, 50), dtype=np.int8),
            np.random.default_rng(1234),
            rho=0.99,
            directed_mode="access_weighted",
            target_habitat_fraction=0.6,
            save_habitat_fractions=requested,
            include_initial_snapshot=True,
        )
        requested_snapshots = [
            snapshot
            for snapshot in result.snapshots
            if "requested_habitat_fraction" in snapshot.reasons
        ]
        self.assertEqual(len(requested_snapshots), len(requested))
        self.assertEqual([snapshot.habitat_fraction for snapshot in requested_snapshots], requested)
        for earlier, later in zip(requested_snapshots, requested_snapshots[1:]):
            self.assertTrue(np.all((earlier.grid == 0) <= (later.grid == 0)))
            self.assertGreater(earlier.forest_cells, later.forest_cells)
            self.assertLess(earlier.event_count, later.event_count)

    def test_reproducibility_includes_trajectory_and_counters(self) -> None:
        kwargs = {
            "rho": 0.99,
            "directed_mode": "access_weighted",
            "target_habitat_fraction": 0.55,
            "save_habitat_fractions": [0.9, 0.7, 0.55],
            "k": 3,
            "lambda_decay": 0.5,
            "eta": 1.0,
            "seed": 555,
        }
        a = run_sequential_deforestation(
            np.ones((35, 41), dtype=np.int8), np.random.default_rng(555), **kwargs
        )
        b = run_sequential_deforestation(
            np.ones((35, 41), dtype=np.int8), np.random.default_rng(555), **kwargs
        )
        np.testing.assert_array_equal(a.grid, b.grid)
        np.testing.assert_array_equal(a.changed_flat_indices, b.changed_flat_indices)
        self.assertEqual(a.event_modes, b.event_modes)
        self.assertEqual(a.n_random_total, b.n_random_total)
        self.assertEqual(a.n_random_stochastic, b.n_random_stochastic)
        self.assertEqual(a.n_directed, b.n_directed)
        for snapshot_a, snapshot_b in zip(a.snapshots, b.snapshots):
            np.testing.assert_array_equal(snapshot_a.grid, snapshot_b.grid)

    def test_selected_mode_failure_is_exposed_without_fallback(self) -> None:
        with patch.object(DeforestationEventState, "deforest_one", side_effect=[0, None]):
            with self.assertRaises(SequentialDeforestationBlockedError) as caught:
                run_sequential_deforestation(
                    np.ones((10, 10), dtype=np.int8),
                    np.random.default_rng(91),
                    rho=1.0,
                    directed_mode="edge_random",
                    target_habitat_fraction=0.98,
                )
        self.assertEqual(caught.exception.diagnostics["event_number"], 2)
        self.assertEqual(caught.exception.diagnostics["selected_mode"], "edge_random")


if __name__ == "__main__":
    unittest.main()
