"""Read pinned metric tables with an explicit edge-density unit conversion."""
from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Sequence

import pandas as pd


def read_metric_plot_table(path: Path, final_root: Path, sources: dict,
                           value_columns: Sequence[str], *, wide: bool = False) -> pd.DataFrame:
    """Convert only ED from a documented source denominator to raster-cell area.

    Source hashes prevent applying the legacy conversion to a regenerated table
    that already uses the canonical definition. Other metrics and missing values
    are preserved; source files are never modified.
    """
    spec = sources["tables"][str(path.relative_to(final_root))]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Figure source changed; review its ED denominator before plotting: {path}")
    frame = pd.read_csv(path)
    factor = spec["edge_density_source_denominator"] / sources["grid_cell_count"]
    if wide:
        frame["edge_density_total"] *= factor
    else:
        rows = frame["metric"].eq("edge_density_total")
        frame.loc[rows, list(value_columns)] *= factor
    return frame
