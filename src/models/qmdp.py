"""QMDP approximation for POMDPs (Littman, Cassandra, Kaelbling 1995).

Given a fully-observable optimal Q-function Q*(s, a) and a belief b over states,
    pi_QMDP(b) = argmax_a sum_s b(s) Q*(s, a).

We also implement the standard Bayesian belief filter:
    b_t(s') ∝ O(o_t | s') * sum_s T(s' | s) b_{t-1}(s).
"""
from __future__ import annotations

import numpy as np


def qmdp_action(belief: np.ndarray, Q_star: np.ndarray) -> int:
    """QMDP greedy action given current belief.

    belief: (S,) summing to 1.
    Q_star: (S, A).
    Returns argmax_a sum_s belief[s] Q_star[s, a].
    """
    return int((belief @ Q_star).argmax())


def update_belief(
    belief: np.ndarray,
    T: np.ndarray,
    obs_likelihood: np.ndarray,
) -> np.ndarray:
    """One step of Bayesian belief filter.

    belief: (S,) prior over states at t-1, summing to 1.
    T: (S, S) transition matrix T[s, s'] = p(s' | s). Regime is exogenous to action.
    obs_likelihood: (S,) p(o_t | s) under each state's emission model.

    Returns: (S,) posterior belief at t.
    """
    predictive = belief @ T                # (S,), predictive prior at t
    unnorm = obs_likelihood * predictive    # (S,), joint with observation
    Z = unnorm.sum()
    if Z <= 0 or not np.isfinite(Z):
        # No usable emission information: preserve the predictive prior.
        return predictive / predictive.sum()
    return unnorm / Z


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Unique stationary distribution, without eigenvector sign ambiguity."""
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or not np.isfinite(T).all() or (T < 0).any() or not np.allclose(T.sum(1), 1):
        raise ValueError("Expected a square stochastic matrix")
    n = len(T)
    A = np.vstack([T.T - np.eye(n), np.ones(n)])
    if np.linalg.matrix_rank(A) < n:
        raise ValueError("Stationary distribution is not unique")
    v = np.linalg.lstsq(A, np.r_[np.zeros(n), 1.0], rcond=None)[0]
    v = np.maximum(v, 0)
    return v / v.sum()


def filter_log_emissions(log_emissions, T, initial):
    """Forward-only filter. The first row uses initial directly, without transition."""
    from scipy.special import logsumexp
    log_emissions = np.asarray(log_emissions, dtype=float)
    T, initial = np.asarray(T, dtype=float), np.asarray(initial, dtype=float)
    if log_emissions.ndim != 2 or T.shape != (len(initial), len(initial)) or log_emissions.shape[1] != len(initial):
        raise ValueError("Incompatible filter shapes")
    if (T < 0).any() or (initial < 0).any() or not np.allclose(T.sum(1), 1) or not np.isclose(initial.sum(), 1) or not np.isfinite(log_emissions).all():
        raise ValueError("Invalid probabilities or log emissions")
    belief = initial.copy()
    rows = []
    for i, emission in enumerate(log_emissions):
        prior = belief @ T if i else belief
        with np.errstate(divide="ignore"):
            joint = np.log(prior) + emission
        belief = np.exp(joint - logsumexp(joint))
        rows.append(belief.copy())
    return np.asarray(rows)
