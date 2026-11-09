"""Exact area contrast between Only-Random and Access-based NN distributions.

Supplementary illustration in Appendix B1. For each H, every one of 30
prespecified landscapes contributes one empirical patch-distance CDF.
The scenario curves are arithmetic means of these landscape CDFs, so their
signed area difference equals the arithmetic mean of paired landscape-mean
NN differences. A 0--20 cell view shows readable geometry; bars integrate
the complete observed distance domain, including the tails beyond 20 cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

FINAL = next(p for p in Path(__file__).resolve().parents if p.name == "Masterarbeit_final")
REL = Path("01_landscape/01_03_result_a/01_03_01_structural_scenario_responses")
sys.path.insert(0, str(FINAL / "code/base"))
from src.tools.final_plot_style import apply_final_style, export_figure, figure_size  # noqa: E402

ONLY = "only_random_rho0"
ACCESS = "access_weighted_rho0999"
H_LEVELS = (0.125, 0.025)
VIEW_MAX = 20.0
INPUT_PATCHES = FINAL / "results" / REL / "tables/08_nn_multiseed_patch_distances.csv"
INPUT_LANDSCAPES = FINAL / "results" / REL / "tables/08_nn_multiseed_landscape_summaries.csv"
ARCHIVED = FINAL / "data/processed" / REL / "landscape_metrics.csv"
STEM = FINAL / "results" / REL / "figures/09_nn_only_random_access_area_contrast"
INTERVAL_TABLE = FINAL / "results" / REL / "tables/09_nn_scenario_area_intervals.csv"
SUMMARY_TABLE = FINAL / "results" / REL / "tables/09_nn_scenario_area_summary.csv"
LOG = FINAL / "results" / REL / "logs/09_nn_scenario_area_contrast.json"
ONLY_COLOR = "#B24A35"
ACCESS_COLOR = "#455B9B"
POSITIVE_COLOR = "#B8D6B8"  # Access has greater mean NN.
NEGATIVE_COLOR = "#E9BBA9"  # Only-Random has greater mean NN.


def mean_cdf(samples: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    assert len(samples) == 30
    return np.mean([np.searchsorted(values, x, side="right") / len(values)
                    for values in samples], axis=0)


def main() -> None:
    patches = pd.read_csv(INPUT_PATCHES)
    landscapes = pd.read_csv(INPUT_LANDSCAPES)
    archived = pd.read_csv(ARCHIVED)
    selected = patches[patches.scenario_id.isin((ONLY, ACCESS)) & patches.H_target.isin(H_LEVELS)]
    assert not selected.duplicated(["scenario_id", "replicate", "H_target", "patch_id"]).any()
    groups = {}
    for (sid, h), group in selected.groupby(["scenario_id", "H_target"]):
        groups[sid, h] = {int(rep): np.sort(sample.nn_distance_cells.to_numpy())
                          for rep, sample in group.groupby("replicate")}
    all_reps = set(groups[ONLY, H_LEVELS[0]])
    assert len(all_reps) == 30
    assert all(set(group) == all_reps for group in groups.values())
    summary = []
    interval_rows = []
    curves = {}
    for h in H_LEVELS:
        samples = {sid: [groups[sid, h][rep] for rep in sorted(all_reps)]
                   for sid in (ONLY, ACCESS)}
        x = np.unique(np.r_[0., *(v for sid in (ONLY, ACCESS) for v in samples[sid])])
        f_only = mean_cdf(samples[ONLY], x)
        f_access = mean_cdf(samples[ACCESS], x)
        gap = f_only[:-1] - f_access[:-1]
        widths = np.diff(x)
        pieces = gap * widths
        positive = float(np.maximum(pieces, 0).sum())
        negative = float(np.minimum(pieces, 0).sum())
        net = positive + negative
        means = {sid: float(np.mean([v.mean() for v in samples[sid]])) for sid in (ONLY, ACCESS)}
        assert np.isclose(net, means[ACCESS] - means[ONLY], atol=1e-11)
        paired = landscapes[landscapes.H_target.eq(h) & landscapes.scenario_id.isin((ONLY, ACCESS))]
        paired = paired.pivot(index="replicate", columns="scenario_id", values="mean_nn_cells")
        assert set(paired.index) == all_reps
        assert np.isclose(net, (paired[ACCESS] - paired[ONLY]).mean(), atol=1e-11)
        # Archived 200-trajectory comparison checks the subsample direction.
        full = archived[archived.H_target.eq(h) & archived.scenario_id.isin((ONLY, ACCESS))]
        full = full.pivot(index="replicate", columns="scenario_id", values="mean_nearest_neighbor_distance")
        assert len(full) == 200 and full.notna().all().all()
        full_difference = full[ACCESS] - full[ONLY]
        clipped_widths = np.clip(np.minimum(x[1:], VIEW_MAX) - x[:-1], 0, None)
        visible_pieces = gap * clipped_widths
        visible_positive = float(np.maximum(visible_pieces, 0).sum())
        visible_negative = float(np.minimum(visible_pieces, 0).sum())
        summary.append({
            "H_target": h, "n_paired_trajectories_ecdf": 30,
            "mean_nn_only_cells_30": means[ONLY], "mean_nn_access_cells_30": means[ACCESS],
            "positive_area_access_longer_cells": positive,
            "negative_area_only_longer_cells": negative,
            "net_access_minus_only_cells": net,
            "positive_area_visible_to20_cells": visible_positive,
            "negative_area_visible_to20_cells": visible_negative,
            "net_tail_beyond20_cells": net - visible_positive - visible_negative,
            "n_archived_trajectories": 200,
            "mean_access_minus_only_cells_200": float(full_difference.mean()),
            "median_paired_access_minus_only_cells_200": float(full_difference.median()),
        })
        interval_rows.extend({
            "H_target": h, "left_distance_cells": float(left),
            "right_distance_cells": float(right), "F_only_random": float(fo),
            "F_access_based": float(fa), "signed_area_cells": float(piece),
        } for left, right, fo, fa, piece in zip(x[:-1], x[1:], f_only[:-1], f_access[:-1], pieces))
        curves[h] = (x, f_only, f_access)

    summary_df = pd.DataFrame(summary)
    interval_df = pd.DataFrame(interval_rows)
    SUMMARY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_TABLE, index=False)
    interval_df.to_csv(INTERVAL_TABLE, index=False)

    apply_final_style()
    fig = plt.figure(figsize=figure_size(132))
    grid = fig.add_gridspec(2, 2, height_ratios=(3.5, 1.0),
                           left=.105, right=.985, top=.91, bottom=.23,
                           wspace=.16, hspace=.39)
    top_axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    bar_axes = [fig.add_subplot(grid[1, i]) for i in range(2)]
    for i, h in enumerate(H_LEVELS):
        ax = top_axes[i]
        bar = bar_axes[i]
        x, f_only, f_access = curves[h]
        row = summary_df[summary_df.H_target.eq(h)].iloc[0]
        for left, right, fo, fa in zip(x[:-1], x[1:], f_only[:-1], f_access[:-1]):
            if left >= VIEW_MAX:
                break
            right = min(right, VIEW_MAX)
            if right <= left or np.isclose(fo, fa, atol=1e-14):
                continue
            ax.add_patch(Rectangle((left, min(fo, fa)), right - left, abs(fo - fa),
                                   facecolor=POSITIVE_COLOR if fo > fa else NEGATIVE_COLOR,
                                   edgecolor="none", zorder=1))
        # Exact step curves evaluated at all observed breakpoints inside the view.
        visible_x = np.r_[x[x < VIEW_MAX], VIEW_MAX]
        visible_x = np.unique(visible_x)
        visible_only = np.searchsorted(x, visible_x, side="right") - 1
        ax.step(visible_x, f_only[visible_only], where="post", color=ONLY_COLOR,
                linestyle="-", linewidth=1.9, zorder=3)
        ax.step(visible_x, f_access[visible_only], where="post", color=ACCESS_COLOR,
                linestyle="--", linewidth=1.9, zorder=3)
        ax.set(xlim=(0, VIEW_MAX), ylim=(0, 1.02),
               xlabel="Patch nearest-neighbour distance (cells)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis="y", color="#D9D9D9", linewidth=.45, zorder=0)
        ax.set_title(f"({chr(97+i)}) $H={h:.3f}$", loc="left", fontsize=9.5, pad=5)
        if i == 0:
            ax.set_ylabel("Mean cumulative fraction of patches")
        else:
            ax.tick_params(labelleft=False)
        pos = float(row.positive_area_access_longer_cells)
        neg = float(row.negative_area_only_longer_cells)
        bar.barh(0, pos, left=0, height=.47, color=POSITIVE_COLOR)
        bar.barh(0, neg, left=0, height=.47, color=NEGATIVE_COLOR)
        bar.axvline(0, color="#404040", linewidth=.7)
        bar.set(xlim=(-1.45, 1.45), ylim=(-.45, .45), xlabel="Full-range area (cells)")
        bar.set_yticks([])
        bar.set_xticks([-1, 0, 1])
        bar.spines[["left", "right", "top"]].set_visible(False)
        bar.text(pos + .04, 0, f"+{pos:.3f}", va="center", ha="left", fontsize=8.5)
        bar.text(neg - .04, 0, f"{neg:.3f}", va="center", ha="right", fontsize=8.5)
        bar.text(.98, .92, f"Net: {row.net_access_minus_only_cells:+.3f} cells",
                 transform=bar.transAxes, ha="right", va="top", fontsize=8.5)
    fig.legend(handles=[
        Line2D([0], [0], color=ONLY_COLOR, lw=1.9, label="Only-Random"),
        Line2D([0], [0], color=ACCESS_COLOR, lw=1.9, linestyle="--",
               label=r"Access-based ($\rho=0.999$)"),
        Patch(facecolor=POSITIVE_COLOR, label="Access-based farther"),
        Patch(facecolor=NEGATIVE_COLOR, label="Only-Random farther"),
    ], loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False)
    export_figure(fig, STEM)
    fig.savefig(STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps({
        "purpose": "Supplementary Only-Random versus Access-based area comparison at two H levels; Appendix B1",
        "H_levels": H_LEVELS, "scenarios": [ONLY, ACCESS], "view_max_cells": VIEW_MAX,
        "n_trajectories_for_ecdfs": 30,
        "identity": "mean_nn_access - mean_nn_only = integral_0^infinity [F_only(x)-F_access(x)] dx",
        "source_patch_table": str(INPUT_PATCHES),
        "all_area_integrals_equal_mean_of_paired_landscape_mean_differences": True,
        "figure_area_bars_include_full_tails_beyond_view": True,
    }, indent=2) + "\n")
    print(summary_df.to_string(index=False))
    print(f"Saved {STEM}")


if __name__ == "__main__":
    main()
