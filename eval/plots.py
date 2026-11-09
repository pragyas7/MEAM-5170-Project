"""Small, fixed-palette plots for the status report and the poster."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#c4492c", "#1d4e89", "#3f7d5a", "#6e4a7c", "#b0892e", "#2f5d62"]


def plot_planner_bars(summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    planners = list(summary["planners"].keys())
    rates = [summary["planners"][name]["success_rate"] for name in planners]
    calls = [summary["planners"][name]["mean_optimizer_calls"] or 0.0 for name in planners]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=140)
    axes[0].bar(planners, rates, color=COLORS[: len(planners)])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("success rate")
    axes[0].set_title(f"{summary.get('name', '')} · {summary.get('split', '')} · n={summary.get('n', '')}")
    axes[1].bar(planners, calls, color=COLORS[: len(planners)])
    axes[1].set_ylabel("mean optimizer calls")
    axes[1].set_title("calls per layout")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_failures(summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    planners = list(summary["planners"].keys())
    reasons: list[str] = []
    for name in planners:
        for reason in summary["planners"][name].get("failure_counts", {}):
            if reason not in reasons:
                reasons.append(reason)
    if not reasons:
        reasons = ["(none)"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    import numpy as np

    x = np.arange(len(planners))
    bottoms = np.zeros(len(planners))
    for i, reason in enumerate(reasons):
        heights = []
        for name in planners:
            counts = summary["planners"][name].get("failure_counts", {})
            heights.append(counts.get(reason, 0) if reason != "(none)" else 0)
        ax.bar(x, heights, bottom=bottoms, color=COLORS[i % len(COLORS)], label=reason, width=0.7)
        bottoms = bottoms + np.asarray(heights, dtype=float)
    ax.set_xticks(x, planners)
    ax.set_ylabel("failed layouts")
    ax.set_title("failure reasons")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_id_ood(id_summary: dict, ood_summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    planners = [name for name in id_summary["planners"] if name in ood_summary["planners"]]
    import numpy as np

    x = np.arange(len(planners))
    width = 0.36
    id_rates = [id_summary["planners"][name]["success_rate"] for name in planners]
    ood_rates = [ood_summary["planners"][name]["success_rate"] for name in planners]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    ax.bar(x - width / 2, id_rates, width, label="test-ID", color="#1d4e89")
    ax.bar(x + width / 2, ood_rates, width, label="test-OOD", color="#c4492c")
    ax.set_xticks(x, planners)
    ax.set_ylim(0, 1)
    ax.set_ylabel("success rate")
    ax.set_title("Q1: in-distribution vs held-out layouts")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
