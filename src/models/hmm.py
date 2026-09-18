"""Gaussian HMM calibration and model selection.

Implements:
    - fit_hmm(obs, n_states): Baum-Welch (EM) via hmmlearn, with restarts.
    - select_n_states(obs, candidates): BIC + held-out LL grid search.
    - bic(model, obs): BIC for a fitted GaussianHMM.

Notation matches Quiz 3 formula sheet (Module 2) and HW2/HW3 derivations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HMMFit:
    n_states: int
    model: object              # hmmlearn.GaussianHMM
    train_ll: float
    test_ll: float
    bic: float


def fit_hmm(
    obs: np.ndarray,
    n_states: int,
    n_restarts: int = 5,
    covariance_type: str = "full",
    n_iter: int = 200,
    seed: int = 0,
):
    """Fit a Gaussian HMM with EM restarts. Returns the best-likelihood model."""
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import-not-found]

    best_model, best_ll = None, -np.inf
    rng = np.random.default_rng(seed)
    for r in range(n_restarts):
        m = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=1e-4,
            random_state=int(rng.integers(0, 1 << 31)),
        )
        try:
            m.fit(obs)
            ll = m.score(obs)
        except Exception as e:  # noqa: BLE001, hmmlearn raises a few different things
            print(f"  restart {r} failed: {e}")
            continue
        if ll > best_ll:
            best_ll, best_model = ll, m
    if best_model is None:
        raise RuntimeError(f"All restarts failed for n_states={n_states}.")
    return best_model, best_ll


def bic(model, obs: np.ndarray) -> float:
    """Bayesian Information Criterion for a fitted GaussianHMM with full covariances.

    BIC = -2 ln L + k ln n.
    Free parameters k for Gaussian HMM with full covariance:
        starts: n_states - 1
        transitions: n_states * (n_states - 1)
        means: n_states * d
        covs: n_states * d * (d + 1) / 2   (full)
    """
    n = obs.shape[0]
    d = obs.shape[1]
    K = model.n_components
    k_params = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) // 2
    ll = model.score(obs)
    return -2.0 * ll + k_params * np.log(n)


def select_n_states(
    train_obs: np.ndarray,
    test_obs: np.ndarray,
    candidates: tuple[int, ...] = (2, 3, 4),
) -> list[HMMFit]:
    """Fit HMMs of several sizes on train, evaluate on test. Returns list sorted by BIC ascending."""
    fits: list[HMMFit] = []
    for K in candidates:
        model, train_ll = fit_hmm(train_obs, K)
        test_ll = model.score(test_obs)
        b = bic(model, train_obs)
        fits.append(HMMFit(n_states=K, model=model, train_ll=train_ll, test_ll=test_ll, bic=b))
        print(f"  K={K}: train_ll={train_ll:.2f}  test_ll={test_ll:.2f}  bic={b:.2f}")
    return sorted(fits, key=lambda f: f.bic)


def standardize_with_train_stats(
    train: pd.DataFrame, full: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize `full[cols]` using mean/std from `train[cols]`. Returns (train_array, full_array)."""
    mu = train[cols].mean()
    sd = train[cols].std(ddof=0).replace(0, 1.0)
    return ((train[cols] - mu) / sd).values, ((full[cols] - mu) / sd).values


# Train window used at fit time in 02_hmm_calibration.py. Any downstream call that
# feeds observations to a pickled HMM MUST standardize with these same statistics,
# because the pickled model's emission means/covariances live in z-scored space.
HMM_TRAIN_END = "2014-12-31"


def standardized_obs(
    df: pd.DataFrame, cols: list[str], train_end: str = HMM_TRAIN_END
) -> np.ndarray:
    """Return df[cols] standardized with the training-window (<= train_end) mean/std.

    This matches the fit-time standardization in 02_hmm_calibration.py exactly, so a
    pickled HMM's predict / predict_proba / emission-likelihood calls are evaluated in
    the same space the model was trained in. Feeding raw observations instead collapses
    the belief onto a single regime (the standardization bug fixed in this revision).
    """
    train = df.loc[:train_end]
    mu = train[cols].mean()
    sd = train[cols].std(ddof=0).replace(0, 1.0)
    return ((df[cols] - mu) / sd).values
