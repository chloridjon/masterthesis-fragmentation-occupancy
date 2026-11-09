from __future__ import annotations

import unittest

import numpy as np

from src.landscape.deforestation import weights_accessibility_front


class AccessWeightSemanticsTests(unittest.TestCase):
    def test_open_path_ring_counts_and_parameter_roles(self) -> None:
        grid = np.ones((5, 5), dtype=np.int8)
        grid[2, 1] = grid[2, 0] = grid[1, 1] = 0
        candidates = np.zeros_like(grid, dtype=bool)
        candidates[2, 2] = True

        # One direct open neighbour and two distinct open cells at distance 2.
        # At lambda=ln(2), A_2=1+2*0.5=2, so eta=2 gives q=4.
        weights = weights_accessibility_front(
            grid, candidates, k=2, lambda_decay=np.log(2), eta=2,
            boundary_mode="open",
        )
        self.assertAlmostEqual(weights[2, 2], 4.0)
        self.assertEqual(int(np.count_nonzero(weights)), 1)

        # At k=1, lambda has no mathematical role.
        for decay in (0.1, 0.5, 1.0):
            weights = weights_accessibility_front(
                grid, candidates, k=1, lambda_decay=decay, eta=2,
                boundary_mode="open",
            )
            self.assertAlmostEqual(weights[2, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
