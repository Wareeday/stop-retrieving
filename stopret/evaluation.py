"""Metrics for stopping policies.

Cost is the average number of hops (= retrieval calls; every hop also adds k passages to
what a reader would have to process). Quality is the share of questions whose retrieved
evidence contains *all* gold supporting passages at the moment the policy stops.

Two policies at different operating points can always be mixed by flipping a biased coin
per question, so the achievable region of a policy is the upper convex hull of its
(cost, recall) points. All policies, including the fixed-hop baseline, are compared
through that hull, which keeps the comparison fair.
"""
from __future__ import annotations

import numpy as np

from .policies import fixed_scores, stop_hops


def outcome(sufficient: np.ndarray, stops: np.ndarray) -> tuple[float, float]:
    """(average hops, full-support recall) of a policy that stops at `stops`."""
    rows = np.arange(len(stops))
    return float(stops.mean()), float(sufficient[rows, stops - 1].mean())


def sweep(scores: np.ndarray, sufficient: np.ndarray, taus: np.ndarray) -> list[tuple[float, float]]:
    return [outcome(sufficient, stop_hops(scores, t)) for t in taus]


def default_taus(n: int = 101) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def fixed_points(sufficient: np.ndarray) -> list[tuple[float, float]]:
    n, max_hops = sufficient.shape
    return [outcome(sufficient, stop_hops(fixed_scores(n, max_hops, h), 0.5))
            for h in range(1, max_hops + 1)]


def upper_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper convex hull, cut off after its highest point (spending more never has to hurt)."""
    pts = sorted({(float(x), float(y)) for x, y in points})  # also accepts JSON lists
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (ox, oy), (ax, ay) = hull[-2], hull[-1]
            if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    best = max(range(len(hull)), key=lambda i: hull[i][1])
    return hull[: best + 1]


def recall_at_budget(points: list[tuple[float, float]], budget: float) -> float:
    """Best recall reachable with at most `budget` average hops (nan if even the cheapest
    point costs more)."""
    hull = upper_hull(points)
    xs, ys = zip(*hull)
    if budget < xs[0] - 1e-12:
        return float("nan")
    if budget >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(budget, xs, ys))


def frontier_auc(points: list[tuple[float, float]], max_hops: int) -> float:
    """Mean recall over budgets from 1 to max_hops hops (area under the hull, normalised)."""
    budgets = np.linspace(1.0, float(max_hops), 200)
    return float(np.nanmean([recall_at_budget(points, b) for b in budgets]))


def paired_bootstrap_vs_fixed(stop_fn, sufficient: np.ndarray, n_boot: int = 1000,
                              seed: int = 0) -> dict[str, float]:
    """How much better is a policy's recall than the fixed-hop baseline *at the same average
    cost*? `stop_fn(idx)` returns the stopping hops of the questions `idx`. Questions are
    resampled with replacement and the baseline is re-derived on every resample."""
    rng = np.random.default_rng(seed)
    n = len(sufficient)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cost, recall = outcome(sufficient[idx], stop_fn(idx))
        diffs.append(recall - recall_at_budget(fixed_points(sufficient[idx]), cost))
    cost, recall = outcome(sufficient, stop_fn(np.arange(n)))
    base = recall_at_budget(fixed_points(sufficient), cost)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"cost": cost, "recall": recall, "fixed_recall_same_cost": base,
            "diff": recall - base, "diff_lo": float(lo), "diff_hi": float(hi)}
