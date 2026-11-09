"""Protect the figure-only ED conversion from double scaling and data loss."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.tools.final_figure_data import read_metric_plot_table


class FigureDataTests(unittest.TestCase):
    def sources(self, root, path, denominator=79600):
        return {"grid_cell_count": 40000, "tables": {str(path.relative_to(root)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "edge_density_source_denominator": denominator}}}

    def test_counts_convert_to_cell_area_with_zero_and_missing_preserved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "observed.csv"
            pd.DataFrame({"edge_density_total": [2 / 79600, 0, np.nan],
                          "chi": [12, 0, np.nan]}).to_csv(path, index=False)
            original = path.read_bytes()
            table = read_metric_plot_table(path, root, self.sources(root, path), (), wide=True)
            np.testing.assert_allclose(table.edge_density_total, [2 / 40000, 0, np.nan], equal_nan=True)
            np.testing.assert_allclose(table.chi, [12, 0, np.nan], equal_nan=True)
            self.assertEqual(path.read_bytes(), original)

    def test_quantiles_scale_only_for_ed_and_keep_interval_membership(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "summary.csv"
            pd.DataFrame({"metric": ["edge_density_total", "chi"], "median": [.3, 12],
                          "q05": [.1, 3], "q95": [.5, 24]}).to_csv(path, index=False)
            result = read_metric_plot_table(path, root, self.sources(root, path), ("median", "q05", "q95"))
            self.assertAlmostEqual(result.iloc[0]["median"], .597)
            self.assertEqual(result.iloc[1]["median"], 12)
            points = np.array([0, .1, .3, .5, .8])
            np.testing.assert_array_equal((points >= .1) & (points <= .5),
                                          (points * 1.99 >= result.iloc[0].q05) &
                                          (points * 1.99 <= result.iloc[0].q95))

    def test_changed_source_rejected_before_a_second_conversion(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "observed.csv"
            path.write_text("edge_density_total\n0.3\n")
            sources = self.sources(root, path)
            path.write_text("edge_density_total\n0.597\n")
            with self.assertRaisesRegex(ValueError, "Figure source changed"):
                read_metric_plot_table(path, root, sources, (), wide=True)


if __name__ == "__main__":
    unittest.main()
