from __future__ import annotations

import unittest

import numpy as np

from src.landscape.landscape import Landscape
from src.landscape.landscape_stats import (
    compute_adjacency_metrics,
    compute_cluster_stats,
    compute_landscape_metrics,
    compute_landscape_metrics_from_landscape,
    susceptibility_chi,
)


class MetricDefinitionSemanticsTests(unittest.TestCase):
    def test_edge_density_counts_unlike_adjacencies(self) -> None:
        grid = np.array([[1, 0], [0, 0]], dtype=np.int8)
        metrics = compute_adjacency_metrics(grid, boundary_mode="open")
        self.assertAlmostEqual(metrics["edge_density_total"], 2 / 4)
        self.assertAlmostEqual(metrics["edge_density"], 2 / 4)

    def test_chi_excludes_one_largest_even_when_another_patch_percolates(self) -> None:
        grid = np.zeros((9, 9), dtype=np.int8)
        grid[1:5, 1:5] = 1  # largest patch, area 16; does not percolate
        grid[:, 8] = 1      # smaller patch, area 9; percolates vertically
        grid[7, 3] = 1      # smallest patch, area 1
        expected = (9 * 9 + 1) / (9 + 1)

        direct = compute_landscape_metrics(
            grid, metric_keys=["chi", "S2_size", "n_clusters"],
            boundary_mode="open",
        )
        self.assertEqual(direct["n_clusters"], 3)
        self.assertAlmostEqual(direct["chi"], expected)
        self.assertEqual(direct["S2_size"], 9)
        self.assertAlmostEqual(susceptibility_chi(np.array([16, 9, 1])), expected)

        landscape = Landscape(grid)
        landscape.ensure_clusters(
            connectivity="4", boundary_mode="open", distance_mode="centroid",
            rebuild=True,
        )
        cached = compute_landscape_metrics_from_landscape(
            landscape, metric_keys=["chi", "S2_size"]
        )
        bundle = compute_cluster_stats(
            landscape.core_cluster_labels, landscape.n_clusters,
            boundary_mode="open",
        )
        self.assertAlmostEqual(cached["chi"], expected)
        self.assertEqual(cached["S2_size"], 9)
        self.assertAlmostEqual(bundle["chi"], expected)

    def test_chi_is_undefined_without_another_patch(self) -> None:
        for grid in (np.zeros((5, 5), dtype=np.int8), np.ones((5, 5), dtype=np.int8)):
            self.assertTrue(np.isnan(compute_landscape_metrics(grid, metric_keys=["chi"])["chi"]))


if __name__ == "__main__":
    unittest.main()
