"""Finite-state infinite-horizon discounted MDP solvers used by QMDP.

Implements:
    - value_iteration(P, R, lam, eps): Lec-7 Bellman iteration with
      stopping criterion `eps*(1-lam)/(2*lam)`.
    - policy_iteration(P, R, lam): exact PI via linear solve.
    - extract_greedy_policy(P, R, V, lam): argmax operator.

Shapes:
    P: (A, S, S), transition tensor, P[a, s, s'] = p(s' | s, a)
                   (NOTE: in our problem regimes are exogenous to action, so
                    P[a, s, s'] = T_HMM[s, s'] for every action a.)
    R: (S, A)   , expected reward per (state, action).
    V: (S,)     , value function.
"""
from __future__ import annotations

import numpy as np


def _validate(P, R, lam):
    P, R = np.asarray(P, dtype=float), np.asarray(R, dtype=float)
    if R.ndim != 2 or min(R.shape) == 0 or P.shape != (R.shape[1], R.shape[0], R.shape[0]):
        raise ValueError("Expected P(A,S,S), R(S,A)")
    if not np.isfinite(P).all() or not np.isfinite(R).all() or (P < 0).any() or not np.allclose(P.sum(2), 1):
        raise ValueError("Finite rewards and stochastic transitions required")
    if not 0 <= lam < 1:
        raise ValueError("Discount must be in [0,1)")
    return P, R


def value_iteration(
    P: np.ndarray, R: np.ndarray, lam: float = 0.95, eps: float = 1e-4, max_iter: int = 10_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Quiz-3 formula-sheet VI.

    Returns
    -------
    V : (S,) optimal value
    pi : (S,) optimal action index per state
    n : number of iterations
    """
    P, R = _validate(P, R, lam)
    if eps <= 0 or not np.isfinite(eps) or max_iter < 1:
        raise ValueError("Positive tolerance and iteration budget required")
    S = R.shape[0]
    V = np.zeros(S)
    tol = eps * (1 - lam)
    for n in range(1, max_iter + 1):
        # Q[s, a] = R[s, a] + lam * sum_{s'} P[a, s, s'] V[s']
        Q = R + lam * np.einsum("ast,t->sa", P, V)
        V_next = Q.max(axis=1)
        if np.max(np.abs(V_next - V)) < tol:
            V = V_next
            break
        V = V_next
    else:
        raise RuntimeError("Value iteration did not converge")
    pi = (R + lam * np.einsum("ast,t->sa", P, V)).argmax(axis=1)
    return V, pi, n


def policy_evaluation_exact(
    P: np.ndarray, R: np.ndarray, pi: np.ndarray, lam: float
) -> np.ndarray:
    """Closed-form policy eval: v = (I - lam * P_pi)^-1 r_pi."""
    P, R = _validate(P, R, lam)
    pi = np.asarray(pi)
    if pi.shape != (R.shape[0],) or not np.issubdtype(pi.dtype, np.integer) or (pi < 0).any() or (pi >= R.shape[1]).any():
        raise ValueError("Invalid policy")
    S = R.shape[0]
    P_pi = np.array([P[pi[s], s, :] for s in range(S)])   # (S, S)
    r_pi = np.array([R[s, pi[s]] for s in range(S)])      # (S,)
    return np.linalg.solve(np.eye(S) - lam * P_pi, r_pi)


def policy_iteration(
    P: np.ndarray, R: np.ndarray, lam: float = 0.95, max_iter: int = 1_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Exact policy iteration with greedy improvement."""
    P, R = _validate(P, R, lam)
    if max_iter < 1:
        raise ValueError("Positive iteration budget required")
    S, A = R.shape
    pi = np.zeros(S, dtype=int)
    for n in range(1, max_iter + 1):
        V = policy_evaluation_exact(P, R, pi, lam)
        Q = R + lam * np.einsum("ast,t->sa", P, V)
        pi_new = Q.argmax(axis=1)
        if np.array_equal(pi_new, pi):
            return V, pi, n
        pi = pi_new
    raise RuntimeError("Policy iteration did not converge")


def q_function(P: np.ndarray, R: np.ndarray, V: np.ndarray, lam: float) -> np.ndarray:
    """Q*(s, a) = R(s, a) + lam * sum_{s'} P(s' | s, a) V(s'). Shape (S, A)."""
    P, R = _validate(P, R, lam)
    if np.shape(V) != (R.shape[0],) or not np.isfinite(V).all():
        raise ValueError("Invalid value vector")
    return R + lam * np.einsum("ast,t->sa", P, V)
