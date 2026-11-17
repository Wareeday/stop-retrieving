"""Stopping policies.

Every policy is expressed the same way: a matrix of "stop scores" with one row per
question and one column per hop, where a higher score means "I'm more ready to stop".
A policy with threshold tau stops at the first hop whose score reaches tau, and at the last
hop if none does. Sweeping tau traces out the cost/quality trade-off of that policy.

fixed(h)        stop after exactly h hops
coverage        stop when the retrieved passages cover enough of the question's terms
top_score       stop when the best new passage scores far above the rest of the candidates
learned         stop when a classifier thinks the evidence is sufficient (see models.py)
learned+giveup  additionally stop when a second classifier thinks the question will not become
                sufficient within the hop budget anyway, so more hops would only waste cost
oracle          stop at the first hop that is actually sufficient (uses gold, upper bound)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_matrix(table: pd.DataFrame, column: str, max_hops: int) -> np.ndarray:
    """Reshape a per-(question, hop) column into an (n_questions, max_hops) matrix."""
    hops = table["hop"].to_numpy().reshape(-1, max_hops)
    if not (hops == np.arange(1, max_hops + 1)).all():
        raise ValueError("table must have exactly max_hops consecutive rows per question")
    return table[column].to_numpy().reshape(-1, max_hops)


def stop_hops(scores: np.ndarray, tau: float) -> np.ndarray:
    """1-indexed hop at which each question stops."""
    reached = scores >= tau
    reached[:, -1] = True
    return reached.argmax(axis=1) + 1


def fixed_scores(n_questions: int, max_hops: int, hops: int) -> np.ndarray:
    scores = np.zeros((n_questions, max_hops))
    scores[:, hops - 1:] = 1.0
    return scores


def oracle_stops(sufficient: np.ndarray) -> np.ndarray:
    """First sufficient hop; questions that never become sufficient stop at hop 1."""
    stops = sufficient.argmax(axis=1) + 1
    stops[~sufficient.any(axis=1)] = 1
    return stops


def stop_hops_giveup(p_now: np.ndarray, p_ever: np.ndarray, tau: float, rho: float) -> np.ndarray:
    """Stop at the first hop where the evidence looks sufficient (p_now >= tau) or the question
    looks hopeless (p_ever < rho); otherwise run to the last hop."""
    reached = (p_now >= tau) | (p_ever < rho)
    reached[:, -1] = True
    return reached.argmax(axis=1) + 1
