"""Render two selected real windows with fixed OpenStreetMap context lines.

The examples illustrate distinct causes of low largest-cluster fraction at
high forest cover. They were selected for explanation, not to estimate the
frequency of either corridor type in the balanced real-window sample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from pyproj import CRS, Transformer
from scipy import ndimage


FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_02_real_landscape_coverage")
DATA = FINAL / "data/processed" / REL / "08_lcf_window_examples"
FIGURES = FINAL / "results" / REL / "figures"
sys.path.insert(0, str(FINAL / "code/base"))

from src.tools.final_plot_style import (  # noqa: E402
    LANDSCAPE_COLORS,
    apply_final_style,
    export_figure,
    figure_size,
)


LINE_STYLES = {
    "gas_pipeline": ("#CC6677", 1.1),
    "power_line": ("#E69F00", 1.1),
    "waterway": ("#0072B2", 1.5),
}
PANEL_LABELS = (
    "(a) Utility corridor",
    "(b) Lena River corridor",
)
REGION_LABELS = (
    "Khanty–Mansi Autonomous Okrug, Russia",
    "Irkutsk Oblast, Russia",
)


def load_sample(case: dict) -> tuple[np.ndarray, dict]:
    path = DATA / case["copied_npz"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["npz_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        grid = archive["grid"]
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["candidate_id"] == case["sample_id"]
    assert grid.shape == (200, 200)
    assert set(np.unique(grid)).issubset({0, 1})
    assert hashlib.sha256(grid.tobytes()).hexdigest() == case["grid_sha256"]
    assert np.isclose(grid.mean(), case["H"], atol=1e-10)
    labels, count = ndimage.label(grid, structure=ndimage.generate_binary_structure(2, 1))
    assert count >= 2
    areas = np.bincount(labels.ravel())[1:]
    assert np.isclose(areas.max() / areas.sum(), case["LCF"], atol=1e-10)
    assert int(metadata["extent_m"]) == case["extent_m"]
    assert np.isclose(float(metadata["lat"]), case["lat"])
    assert np.isclose(float(metadata["lon"]), case["lon"])
    return grid, metadata


def main() -> None:
    manifest = json.loads((DATA / "08_lcf_window_examples_manifest.json").read_text())
    features = json.loads((DATA / "08_lcf_window_osm_features.geojson").read_text())
    assert len(manifest["cases"]) == 2
    assert features["type"] == "FeatureCollection"
    assert {feature["properties"]["sample_id"] for feature in features["features"]} == {
        case["sample_id"] for case in manifest["cases"]
    }

    apply_final_style()
    fig = plt.figure(figsize=figure_size(125))
    axes = [fig.add_axes((.045, .28, .425, .545)),
            fig.add_axes((.53, .28, .425, .545))]
    palette = ListedColormap([
        LANDSCAPE_COLORS["non_habitat"],
        LANDSCAPE_COLORS["habitat"],
    ])

    for index, (ax, case) in enumerate(zip(axes, manifest["cases"], strict=True)):
        grid, metadata = load_sample(case)
        extent = case["extent_m"]
        ax.imshow(
            grid, cmap=palette, vmin=0, vmax=1, origin="upper",
            extent=(-extent / 2, extent / 2, -extent / 2, extent / 2),
            interpolation="nearest", rasterized=True,
        )
        transformer = Transformer.from_crs(
            "EPSG:4326", CRS.from_wkt(metadata["local_crs"]), always_xy=True
        )
        drawn_ids = set()
        for feature in features["features"]:
            props = feature["properties"]
            if props["sample_id"] != case["sample_id"]:
                continue
            kind = props["feature_type"]
            assert kind in LINE_STYLES
            coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=float)
            x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
            colour, width = LINE_STYLES[kind]
            ax.plot(x, y, color=colour, linewidth=width, solid_capstyle="round")
            drawn_ids.add(props["osm_way_id"])
        assert drawn_ids == set(case["osm_way_ids"])
        ax.set_xlim(-extent / 2, extent / 2)
        ax.set_ylim(-extent / 2, extent / 2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(.5)
            spine.set_edgecolor("#737373")

        left = ax.get_position().x0
        fig.text(left, .86, PANEL_LABELS[index], ha="left", va="bottom", fontsize=9.5)
        fig.text(left, .253, f'{extent // 1000} km window · {case["sample_id"]}',
                 ha="left", va="top", fontsize=8.5)
        fig.text(left, .222, REGION_LABELS[index], ha="left", va="top", fontsize=8.5)
        fig.text(left, .191,
                 f'{case["lat"]:.5f}° N, {case["lon"]:.5f}° E',
                 ha="left", va="top", fontsize=8.5)
        fig.text(left, .16,
                 rf'$H={case["H"]:.3f}$ · LCF $={case["LCF"]:.3f}$',
                 ha="left", va="top", fontsize=8.5)

    legend = [
        Patch(facecolor=LANDSCAPE_COLORS["habitat"], label="Forest"),
        Patch(facecolor=LANDSCAPE_COLORS["non_habitat"],
              edgecolor="#737373", linewidth=.4, label="Non-forest"),
        Line2D([], [], color=LINE_STYLES["gas_pipeline"][0],
               linewidth=1.5, label="Mapped gas pipeline"),
        Line2D([], [], color=LINE_STYLES["power_line"][0],
               linewidth=1.5, label="Mapped power line"),
        Line2D([], [], color=LINE_STYLES["waterway"][0],
               linewidth=1.8, label="Mapped watercourse"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=3, frameon=False)
    export_figure(fig, FIGURES / "08_lcf_window_examples")
    fig.savefig(FIGURES / "08_lcf_window_examples.png", dpi=350,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
