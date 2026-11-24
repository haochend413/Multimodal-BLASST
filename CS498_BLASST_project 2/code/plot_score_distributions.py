"""Plot causal score and tile distributions from a score study JSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score_distribution_study import SCHEMA

LABELS = {"T->T": "T→T", "T->V": "T→V", "V->V": "V→V", "mixed": "Mixed"}
COLORS = {"T->T": "#2259b5", "T->V": "#087970", "V->V": "#803aac", "mixed": "#586579"}


def _tail(ax, data, edges, metric, limit):
    for group, label in LABELS.items():
        record = data.get(group, {})
        if metric not in record:
            continue
        bins = np.asarray(record[metric])
        y = 100 * np.cumsum(bins[::-1])[::-1][1:] / bins.sum()
        ax.step(edges, y, where="post", label=f"{label} · {bins.sum():,}", color=COLORS[group], lw=2)
        ax.scatter([0], [y[list(edges).index(0)]], color=COLORS[group], s=16, zorder=3)
    ax.axvline(0, ls="--", lw=1.2, color="#364252")
    ax.set(xlim=(-12, min(10, max(1, math.ceil(limit)))), ylim=(0, 103),
           xlabel="Score minus row cutoff (logits)", ylabel="At or above x (%)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", frameon=False, fontsize=9)


def _support(ax, records, metric, xlabel):
    series = [(group, records[group][metric]) for group in LABELS if metric in records.get(group, {})]
    if not series:
        ax.text(.5, .5, "No exact tile scores available", ha="center", transform=ax.transAxes)
        return
    labels = ("0", ">0–<10", "10–<25", "25–<50", "50–<75", "75–<100", "100")
    width = .8 / len(series)
    for index, (group, raw) in enumerate(series):
        bins = np.asarray(raw)
        x = np.arange(len(labels)) + (index - (len(series) - 1) / 2) * width
        ax.bar(x, bins / bins.sum() * 100, width, label=f"{LABELS[group]} · {bins.sum():,}",
               color=COLORS[group])
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels, xlabel=xlabel, ylabel="Tiles (%)", ylim=(0, 103))
    ax.legend(frameon=False, fontsize=9)


def plot_report(report, path, phase="prefill"):
    if report["schema"] != SCHEMA:
        raise ValueError("Expected a causal score distribution report")
    sources = report["sources"]
    physical = sources.get("physical_tiles", {}).get(phase, {})
    tiles = sources.get("saved_tile_margins", {}).get(phase, physical)
    regions = sources.get("token_regions", {}).get(phase, {})
    n_exact = len(report["exact_score_cases"])
    exact_label = f"{n_exact} exact {'case' if n_exact == 1 else 'cases'}"
    n_margins = report.get("saved_tile_margin_cases") or n_exact
    desired = ("T->T", "T->V", "V->V") if phase == "prefill" else ("T->T", "T->V")
    pure_scores = all(group in physical for group in desired)
    lam = report["lambda_by_phase"][phase]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), layout="constrained")
    fig.suptitle(f"Causal BLASST · {phase.capitalize()} · λ = {report['lambda_by_phase'][phase]:.3g}",
                 x=.02, ha="left", fontsize=18, weight="bold")
    for ax in axes.flat:
        ax.grid(axis="y", color="#e2e8f0")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title(f"Best score per physical tile · {n_margins} cases", loc="left", weight="bold")
    _tail(axes[0, 0], tiles, report["bins"]["excess"], "tile_excess", -math.log(lam))
    axes[0, 1].set_title(
        f"Individual scores · {exact_label}\n"
        + ("Grouped by physical tile" if pure_scores else "Grouped by token pair, including mixed tiles"),
        loc="left", weight="bold")
    _tail(axes[0, 1], physical if pure_scores else regions, report["bins"]["excess"], "score_excess", -math.log(lam))
    axes[1, 0].set_title(f"Rows reaching the cutoff · {exact_label}", loc="left", weight="bold")
    _support(axes[1, 0], physical, "row_support", "Valid query rows reaching cutoff per tile (%)")
    axes[1, 1].set_title(f"Scores reaching the cutoff · {exact_label}", loc="left", weight="bold")
    _support(axes[1, 1], physical, "score_support", "Valid scores reaching cutoff per tile (%)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--phase", default="prefill", choices=("prefill", "decode"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot_report(json.loads(args.report.read_text()), args.output, args.phase)
