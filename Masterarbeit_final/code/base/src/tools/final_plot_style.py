"""Shared print settings for final thesis figures."""

from __future__ import annotations

import base64
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


FULL_WIDTH_MM = 160.0
HALF_WIDTH_MM = 77.0
ROMAN_FONT_FAMILY = "Latin Modern Roman 8"
ROMAN_FONT_FILES = {
    ("normal", "normal"): "lmroman8-regular.otf",
    ("normal", "bold"): "lmroman8-bold.otf",
    ("italic", "normal"): "lmroman8-italic.otf",
}
NEUTRAL = "#404040"
MISSING = "#D9D9D9"
LANDSCAPE_COLORS = {
    "habitat": "#315C42",
    "non_habitat": "#E8E8E8",
}
METHOD_COLORS = {
    "only_random": "#B24A35",
    "edge_random": "#377A9F",
    "access_weighted": "#6A5795",
}

# One palette and legend order for the seven prespecified thesis scenarios.
# Matched rho levels share colours; the method is encoded by line style.
FOCAL_SCENARIOS = (
    {"scenario_id": "only_random_rho0", "label": r"Only-Random ($\rho=0$)",
     "color": "#B24A35", "linestyle": "-"},
    {"scenario_id": "edge_random_rho09", "label": r"Edge-Random ($\rho=0.9$)",
     "color": "#C47723", "linestyle": "-"},
    {"scenario_id": "access_weighted_rho09", "label": r"Access-based ($\rho=0.9$)",
     "color": "#C47723", "linestyle": "--"},
    {"scenario_id": "edge_random_rho099", "label": r"Edge-Random ($\rho=0.99$)",
     "color": "#247D83", "linestyle": "-"},
    {"scenario_id": "access_weighted_rho099", "label": r"Access-based ($\rho=0.99$)",
     "color": "#247D83", "linestyle": "--"},
    {"scenario_id": "edge_random_rho0999", "label": r"Edge-Random ($\rho=0.999$)",
     "color": "#455B9B", "linestyle": "-"},
    {"scenario_id": "access_weighted_rho0999", "label": r"Access-based ($\rho=0.999$)",
     "color": "#455B9B", "linestyle": "--"},
)

# Match the cluster/Voronoi display in applications/defragmentation_live.py.
CLUSTER_PALETTE_RGB = (
    (0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167),
    (213, 94, 0), (240, 228, 66), (51, 34, 136), (136, 34, 85),
    (17, 119, 51), (221, 204, 119), (170, 68, 153), (68, 170, 153),
    (102, 17, 0), (136, 204, 238), (153, 153, 51), (170, 68, 0),
    (68, 119, 170), (238, 102, 119), (34, 136, 51), (102, 204, 238),
)
VORONOI_BACKGROUND_RGB = (239, 233, 220)
VORONOI_BOUNDARY_RGB = (58, 58, 58)
VORONOI_BOUNDARY_ALPHA = 0.34
VORONOI_REGION_COLOR_WEIGHT = 0.48
VORONOI_NETWORK_COLORS = {
    "edge": "#5C656D",
    "mst": "#404040",
    "non_tree": "#DCE0E3",
}


@lru_cache(maxsize=1)
def roman_font_files() -> dict[tuple[str, str], Path]:
    """Register explicit Roman 8 faces under a distinct Matplotlib family."""
    resolved = {}
    for face, filename in ROMAN_FONT_FILES.items():
        entries = [
            entry for entry in font_manager.fontManager.ttflist
            if Path(entry.fname).name == filename and Path(entry.fname).is_file()
        ]
        if not entries:
            raise RuntimeError(
                f"Required Latin Modern Roman 8 font is unavailable: {filename}. "
                "Install the Latin Modern OpenType fonts; do not silently substitute."
            )
        entry = replace(sorted(entries, key=lambda item: item.fname)[0], name=ROMAN_FONT_FAMILY)
        if not any(
            existing.name == ROMAN_FONT_FAMILY and existing.fname == entry.fname
            for existing in font_manager.fontManager.ttflist
        ):
            font_manager.fontManager.ttflist.append(entry)
        resolved[face] = Path(entry.fname)
    font_manager.fontManager._findfont_cached.cache_clear()
    return resolved


def apply_final_style() -> None:
    roman_font_files()
    mpl.rcParams.update(
        {
            "font.family": ROMAN_FONT_FAMILY,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Latin Modern Math",
            "mathtext.it": "Latin Modern Math:italic",
            "mathtext.bf": f"{ROMAN_FONT_FAMILY}:bold",
            "mathtext.cal": f"{ROMAN_FONT_FAMILY}:italic",
            "mathtext.sf": ROMAN_FONT_FAMILY,
            "mathtext.tt": "Latin Modern Mono",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "figure.labelsize": 9.5,
            "figure.labelweight": "normal",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.columnspacing": 1.2,
            "legend.handletextpad": 0.6,
            "legend.labelspacing": 0.4,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def figure_size(height_mm: float, width_mm: float = FULL_WIDTH_MM) -> tuple[float, float]:
    return width_mm / 25.4, height_mm / 25.4


def _embed_roman8_in_svg(path: Path) -> None:
    """Keep SVG text editable while preserving the selected optical-size face."""
    rules = []
    for (style, weight), font_path in roman_font_files().items():
        encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
        rules.append(
            "@font-face {"
            f"font-family: '{ROMAN_FONT_FAMILY}'; font-style: {style}; "
            f"font-weight: {weight}; "
            f"src: url('data:font/otf;base64,{encoded}') format('opentype');"
            "}"
        )
    text = path.read_text(encoding="utf-8")
    if "<defs>" not in text:
        raise RuntimeError(f"SVG export has no definitions element: {path}")
    font_css = '<style type="text/css">' + "\n".join(rules) + "</style>"
    text = text.replace("<defs>", "<defs>\n" + font_css, 1)
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def compact_shared_labels(figure: mpl.figure.Figure, pad_points: float = 7.0) -> None:
    """Place shared labels beside outer tick labels, independently of figure size."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [axis.get_tightbbox(renderer) for axis in figure.axes if axis.get_visible()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return
    pad = pad_points * figure.dpi / 72.0
    width, height = figure.bbox.width, figure.bbox.height
    ylabel = getattr(figure, "_supylabel", None)
    if ylabel is not None:
        box = ylabel.get_window_extent(renderer)
        x, y = ylabel.get_position()
        ylabel.set_position((x + (min(b.x0 for b in boxes) - pad - box.x1) / width, y))
    xlabel = getattr(figure, "_supxlabel", None)
    if xlabel is not None:
        box = xlabel.get_window_extent(renderer)
        x, y = xlabel.get_position()
        xlabel.set_position((x, y + (min(b.y0 for b in boxes) - pad - box.y1) / height))
        figure.canvas.draw()
        box = xlabel.get_window_extent(figure.canvas.get_renderer())
        # Current final figures have one common legend beneath the panel grid.
        if len(figure.legends) == 1:
            legend = figure.legends[0]
            legend.set_loc("upper center")
            legend.set_bbox_to_anchor((0.5, (box.y0 - 6.0 * figure.dpi / 72.0) / height), transform=figure.transFigure)


def export_figure(figure: mpl.figure.Figure, stem: Path) -> None:
    compact_shared_labels(figure)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".pdf"):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        if suffix == ".svg":
            _embed_roman8_in_svg(path)
