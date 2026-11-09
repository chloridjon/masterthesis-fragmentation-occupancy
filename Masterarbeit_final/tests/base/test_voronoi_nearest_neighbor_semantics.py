from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from src.landscape.landscape import Landscape
from src.landscape.landscape_stats import (
    compute_landscape_metrics,
    compute_landscape_metrics_from_landscape,
    mean_nearest_neighbor_distance_from_values,
    nearest_neighbor_distances_from_centroids,
    nearest_neighbor_distances_from_matrix,
    nearest_neighbor_distances_from_voronoi_graph,
)


class VoronoiNearestNeighborSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = np.zeros((12, 12), dtype=np.int8)
        self.grid[2:5, 1:4] = 1
        self.grid[2:5, 7:10] = 1

    def test_canonical_metric_uses_exact_voronoi_edges_and_centroid_is_explicit(self) -> None:
        landscape = Landscape(self.grid)
        landscape.ensure_clusters(
            connectivity="4",
            boundary_mode="open",
            distance_mode="voronoi_dt",
            voronoi_depth=1,
            rebuild=True,
        )
        metrics = compute_landscape_metrics_from_landscape(
            landscape,
            metric_keys=[
                "mean_nearest_neighbor_distance",
                "mean_centroid_nearest_neighbor_distance",
            ],
        )
        expected_voronoi = mean_nearest_neighbor_distance_from_values(
            nearest_neighbor_distances_from_voronoi_graph(landscape.voronoi_neighbor_graph)
        )
        expected_centroid = mean_nearest_neighbor_distance_from_values(
            nearest_neighbor_distances_from_centroids(
                landscape.cluster_centroids,
                shape=self.grid.shape,
                boundary_mode="open",
            )
        )
        self.assertAlmostEqual(metrics["mean_nearest_neighbor_distance"], expected_voronoi)
        self.assertAlmostEqual(
            metrics["mean_centroid_nearest_neighbor_distance"], expected_centroid
        )
        self.assertNotAlmostEqual(expected_voronoi, expected_centroid)

    def test_array_api_defaults_to_voronoi_nn(self) -> None:
        metrics = compute_landscape_metrics(
            self.grid,
            metric_keys=[
                "mean_nearest_neighbor_distance",
                "mean_centroid_nearest_neighbor_distance",
            ],
        )
        self.assertLess(
            metrics["mean_nearest_neighbor_distance"],
            metrics["mean_centroid_nearest_neighbor_distance"],
        )

    def test_centroid_fallback_cannot_become_a_nearest_neighbour(self) -> None:
        # Three nested patches share a centroid. The outer and inner patches
        # are not Voronoi neighbours; their hybrid fallback is zero, whereas
        # every exact direct-neighbour edge has length four cells.
        grid = np.zeros((21, 21), dtype=np.int8)
        grid[:2, :] = grid[-2:, :] = grid[:, :2] = grid[:, -2:] = 1
        grid[5, 5:16] = grid[15, 5:16] = 1
        grid[5:16, 5] = grid[5:16, 15] = 1
        grid[9:12, 9:12] = 1
        landscape = Landscape(grid)
        landscape.ensure_clusters(
            connectivity="4", boundary_mode="open", distance_mode="voronoi_dt",
            voronoi_depth=1, rebuild=True,
        )
        self.assertEqual(landscape.n_clusters, 3)
        self.assertAlmostEqual(
            mean_nearest_neighbor_distance_from_values(
                nearest_neighbor_distances_from_matrix(landscape.cluster_dist_matrix)
            ),
            4 / 3,
        )
        self.assertAlmostEqual(
            compute_landscape_metrics_from_landscape(
                landscape, metric_keys=["mean_nearest_neighbor_distance"]
            )["mean_nearest_neighbor_distance"],
            4.0,
        )
        self.assertAlmostEqual(
            compute_landscape_metrics(
                grid, metric_keys=["mean_nearest_neighbor_distance"]
            )["mean_nearest_neighbor_distance"],
            4.0,
        )

    def test_nearest_neighbour_is_undefined_without_two_patches(self) -> None:
        for grid in (np.zeros((5, 5), dtype=np.int8), np.ones((5, 5), dtype=np.int8)):
            metrics = compute_landscape_metrics(
                grid, metric_keys=["mean_nearest_neighbor_distance"]
            )
            self.assertTrue(np.isnan(metrics["mean_nearest_neighbor_distance"]))

    def test_missing_graph_edge_is_not_a_zero_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "Every patch needs"):
            nearest_neighbor_distances_from_voronoi_graph(sparse.csr_matrix((2, 2)))


if __name__ == "__main__":
    unittest.main()
