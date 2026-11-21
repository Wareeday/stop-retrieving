"""Figures for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .evaluation import upper_hull  # noqa: E402

COLORS = {"gbm_giveup": "#7b2cbf", "fixed": "#555555", "coverage": "#c9a227", "top_score": "#b5651d",
          "logreg": "#3b7dd8", "gbm": "#d6336c", "oracle": "#2b9348"}
LABELS = {"fixed": "fixed number of hops", "coverage": "stop on question coverage",
          "top_score": "stop on top-score margin", "logreg": "learned (logistic regression)",
          "gbm": "learned (gradient boosting)",
          "gbm_giveup": "learned (gradient boosting) + give-up rule"}


def _line(ax, points, key, style="-", label=None, marker=None):
    hull = upper_hull(points)
    xs, ys = zip(*hull, strict=True)
    ax.plot(xs, ys, style, color=COLORS[key], label=label or LABELS[key], marker=marker,
            linewidth=2, markersize=6)


def plot_hotpotqa(res: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    _line(ax, res["fixed"]["points"], "fixed", marker="o")
    for h in ("coverage", "top_score"):
        _line(ax, res["heuristics"][h]["points"], h, "--")
    for kind in ("logreg", "gbm"):
        _line(ax, res["in_domain_cv"][kind]["points"], kind)
    # At the default rho the give-up variant is identical to plain gradient boosting on HotpotQA (it
    # never fires there), so it is left out of this figure to keep the lines readable.
    ax.plot([res["oracle"]["cost"]], [res["oracle"]["recall"]], "*", color=COLORS["oracle"],
            markersize=14, label="oracle (knows the gold passages)")
    ax.set_xlabel("average number of hops (retrieval calls per question)")
    ax.set_ylabel("share of questions with all gold passages retrieved")
    ax.set_title("HotpotQA: quality vs. cost of stopping policies\n(5-fold cross-validation, 1,000 questions)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_all_datasets(metrics: dict, path: Path) -> None:
    names = list(metrics["datasets"])
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 4.2), sharey=False)
    for ax, name in zip(axes, names, strict=True):
        res = metrics["datasets"][name]
        _line(ax, res["fixed"]["points"], "fixed", marker="o")
        gu = res["in_domain_cv"]["gbm"]["giveup"]
        _line(ax, res["in_domain_cv"]["gbm"]["points"], "gbm", label="learned (CV)")
        _line(ax, gu["by_rho"][str(gu["main_rho"])]["points"], "gbm_giveup",
              label="learned + give-up (CV)")
        if name in metrics["transfer_from_hotpotqa"]:
            tr = metrics["transfer_from_hotpotqa"][name]["gbm"]
            _line(ax, tr["points"], "gbm", ":", label="learned, trained on HotpotQA only")
            _line(ax, tr["giveup"]["by_rho"][str(tr["giveup"]["main_rho"])]["points"],
                  "gbm_giveup", ":", label="learned + give-up, trained on HotpotQA only")
        ax.plot([res["oracle"]["cost"]], [res["oracle"]["recall"]], "*", color=COLORS["oracle"],
                markersize=13, label="oracle")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("average hops")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share with all gold passages retrieved")
    # the middle panel has every curve (solid = trained in-domain, dotted = trained on HotpotQA)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ablation(ablation: dict, path: Path) -> None:
    items = sorted(ablation.items(), key=lambda kv: kv[1]["frontier_auc"])
    labels = [k for k, _ in items]
    values = [v["frontier_auc"] for _, v in items]
    errors = [v["frontier_auc_std"] for _, v in items]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    colors = ["#d6336c" if k == "all features" else "#9aa5b1" for k in labels]
    ax.barh(labels, values, xerr=errors, color=colors, capsize=2)
    ax.set_xlim(min(values) - 0.02, max(values) + 0.015)
    ax.set_xlabel("mean recall over budgets of 1 to 4 hops (higher is better)")
    ax.set_title("HotpotQA feature-group ablation (gradient boosting)\nerror bars: std over 5 CV splits, not over questions", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
