"""Compare accessibility rings in simple and structured illustrative landscapes.

The fixed 13 x 13 habitat masks are illustrations, not empirical observations.
Breadth-first traversal enters non-habitat on the first step and never crosses
habitat. Its ring counts are checked against the model's accessibility weights.
"""

from __future__ import annotations

import csv
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch, Rectangle

THESIS_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(THESIS_ROOT / "code" / "base"))
from src.landscape.deforestation import FOUR_NEIGHBOR_OFFSETS, weights_accessibility_front
from src.tools.final_plot_style import (
    LANDSCAPE_COLORS, apply_final_style, export_figure, figure_size,
)

OUTPUT_STEM = (THESIS_ROOT / "latex" / "figures" / "02_theoretical_framework"
               / "access_rings_schematic")
MAX_RING = 5
GRID_RADIUS = 6
FOCAL_CELL = (GRID_RADIUS, GRID_RADIUS)
HABITAT_COLOUR = LANDSCAPE_COLORS["habitat"]
OUTSIDE_COLOUR = LANDSCAPE_COLORS["non_habitat"]


def illustrative_landscapes() -> tuple[np.ndarray, np.ndarray]:
    """Return the original isolated cell and a fixed irregular patch mosaic."""
    simple = np.zeros((2 * GRID_RADIUS + 1,) * 2, dtype=np.int8)
    simple[FOCAL_CELL] = 1
    structured = np.array([
        [int(cell) for cell in row] for row in (
            "1111000001100",
            "1111100011110",
            "0111100011110",
            "0011000001100",
            "0000111000000",
            "0011111000100",
            "0111111001110",
            "0111100011110",
            "0011000001100",
            "0000000000000",
            "0111000111000",
            "1111001111100",
            "0110000111000",
        )
    ], dtype=np.int8)
    return simple, structured


def accessibility_rings(habitat: np.ndarray) -> np.ndarray:
    """Assign shortest non-habitat path lengths up to k (0 at x, -1 elsewhere).

    The focal cell is central, so a path of at most five steps cannot touch
    the window boundary; open and periodic boundaries give identical rings.
    """
    assert habitat[FOCAL_CELL] == 1
    rings = np.full(habitat.shape, -1, dtype=np.int8)
    rings[FOCAL_CELL] = 0
    queue = deque([FOCAL_CELL])
    while queue:
        row, col = queue.popleft()
        depth = int(rings[row, col]) + 1
        if depth > MAX_RING:
            continue
        for dy, dx in FOUR_NEIGHBOR_OFFSETS:
            ny, nx = row + dy, col + dx
            if (0 <= ny < habitat.shape[0] and 0 <= nx < habitat.shape[1]
                    and not habitat[ny, nx] and rings[ny, nx] == -1):
                rings[ny, nx] = depth
                queue.append((ny, nx))

    candidates = np.zeros_like(habitat, dtype=bool)
    candidates[FOCAL_CELL] = True
    for max_depth in range(1, MAX_RING + 1):
        weights = weights_accessibility_front(
            habitat, candidates, k=max_depth, lambda_decay=0.5,
            eta=1.0, boundary_mode="periodic",
        )
        expected = sum(np.count_nonzero(rings == depth) * np.exp(-0.5 * (depth - 1))
                       for depth in range(1, max_depth + 1))
        np.testing.assert_allclose(weights[FOCAL_CELL], expected, rtol=1e-12)
    return rings


def main() -> None:
    apply_final_style()
    blue = np.asarray(to_rgb("#377A9F"))
    # Opacity decreases outward; numbers retain ring identity.
    ring_colours = [to_hex(alpha * blue + (1 - alpha) * np.ones(3))
                    for alpha in np.linspace(0.90, 0.22, MAX_RING)]
    fig = plt.figure(figsize=figure_size(108))

    records = []
    labels = ("(a) Isolated forest cell", "(b) Structured patch landscape")
    for panel, (habitat, label, left) in enumerate(
            zip(illustrative_landscapes(), labels, (0.018, 0.517)), start=1):
        ax = fig.add_axes((left, 0.255, 0.465, 0.689))
        rings = accessibility_rings(habitat)
        for row, col in np.ndindex(habitat.shape):
            y, x = GRID_RADIUS - row, col - GRID_RADIUS
            ring = int(rings[row, col])
            colour = (HABITAT_COLOUR if habitat[row, col] else
                      ring_colours[ring - 1] if ring > 0 else OUTSIDE_COLOUR)
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                  facecolor=colour, edgecolor="white", linewidth=0.45))
            if ring > 0:
                ax.text(x, y, str(ring), ha="center", va="center", fontsize=8.5,
                        color="white" if ring <= 2 else "#202020")
            records.append({"panel": panel, "row_offset": y, "column_offset": x,
                            "habitat": int(habitat[row, col]),
                            "focal_cell": int((row, col) == FOCAL_CELL),
                            "ring": ring if ring >= 0 else ""})

        # Orange contrasts with both the green forest and blue rings.
        ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, facecolor="#D55E00",
                               edgecolor="white", linewidth=.7))
        # An inset white outline singles out x within the continuous green patch.
        ax.add_patch(Rectangle((-0.43, -0.43), 0.86, 0.86, fill=False,
                               edgecolor="white", linewidth=0.9))
        ax.text(0, 0, r"$x$", ha="center", va="center", color="white", fontsize=10)
        limit = GRID_RADIUS + 0.5
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0, 1.035, label, transform=ax.transAxes, ha="left", va="bottom")
        counts = [np.count_nonzero(rings == depth) for depth in range(1, MAX_RING + 1)]
        ax.text(0.5, -0.042,
                "$" + r",\quad ".join(fr"o_{depth}={count}" for depth, count
                                        in enumerate(counts, start=1)) + "$",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        print(f"Panel {panel}: non-habitat ring counts = {counts}")

    handles = [Patch(facecolor=HABITAT_COLOUR, label="Forest"),
               Patch(facecolor=OUTSIDE_COLOUR, label="Non-habitat outside rings")]
    handles += [Patch(facecolor=colour, label=fr"Ring $r={index}$")
                for index, colour in enumerate(ring_colours, start=1)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.095),
               ncols=4, frameon=False, handlelength=1.35, handleheight=1.0,
               labelspacing=0.6, borderaxespad=0)

    export_figure(fig, OUTPUT_STEM)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with OUTPUT_STEM.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
