"""Alternative asset-allocation baselines for the QMDP comparison study.

Each function follows the same signature:

    def baseline_xxx(asset_rets: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
        return weights  # (T, n_assets) with index = asset_rets.index

The returned DataFrame has one row per rebalance date and one column per asset
(stock, bond). All rows must sum to 1 (long-only, fully invested unless the
strategy explicitly allows cash; see notes per function).

Strategies implemented:
  1. static_60_40             , textbook benchmark (Bogleheads, target-date funds).
  2. equal_weight             , 1/N naive diversification (DeMiguel/Garlappi/Uppal 2009).
  3. inverse_volatility       , "risk parity light"; weight by 1/sigma normalised.
  4. risk_parity_2asset       , Maillard/Roncalli/Teiletche 2010 equal-risk-contribution
                                  closed form for 2 assets.
  5. mean_variance_lw         , Markowitz with Ledoit-Wolf shrinkage covariance
                                  (Ledoit & Wolf 2003, 2004), long-only no-leverage,
                                  rolling-window estimation.
  6. faber_10mo_sma           , Faber 2007 "Quantitative Approach to TAA": hold asset
                                  if price > 10-month MA, else move to the other asset
                                  (we don't have cash, so bonds are the safe leg).
  7. ts_momentum_12mo         , Moskowitz/Ooi/Pedersen 2012 time-series momentum:
                                  long the asset(s) with positive trailing-12-month
                                  excess return; if both negative, weight by sign+inverse-vol.
  8. vol_target_60_40         , Hurst/Ooi/Pedersen volatility-targeting: scale the 60/40
                                  exposure each month to hit an annualised vol target.
  9. myopic_12mo              , Already in the main backtester but reproduced here for
                                  parity (12-month rolling mean return scoring).
 10. hmm_conditional_mv       , Guidolin/Timmermann-style HMM-conditional mean-variance:
                                  use the current belief over regimes to mix per-regime
                                  mean/covariance, then solve MV in closed form.
 11. black_litterman_hmm_views, Black-Litterman 1992 with views derived from the HMM
                                  regime-conditional expected returns weighted by belief.

References cited in docstrings; see the project's references.bib for full citations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. Static 60/40
# ----------------------------------------------------------------------
def static_60_40(asset_rets: pd.DataFrame) -> pd.DataFrame:
    """Textbook 60/40 stock/bond. The benchmark every regime paper tries to beat."""
    n = len(asset_rets)
    return pd.DataFrame(
        np.tile([0.6, 0.4], (n, 1)),
        index=asset_rets.index, columns=asset_rets.columns,
    )


# ----------------------------------------------------------------------
# 2. Equal weight (1/N)
# ----------------------------------------------------------------------
def equal_weight(asset_rets: pd.DataFrame) -> pd.DataFrame:
    """Equal weights across all assets. DeMiguel/Garlappi/Uppal (2009) show 1/N is
    surprisingly hard to beat for small universes due to estimation error.
    """
    n = len(asset_rets)
    k = asset_rets.shape[1]
    return pd.DataFrame(
        np.full((n, k), 1.0 / k),
        index=asset_rets.index, columns=asset_rets.columns,
    )


# ----------------------------------------------------------------------
# 3. Inverse-volatility weights ("risk parity light")
# ----------------------------------------------------------------------
def inverse_volatility(asset_rets: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """w_i ∝ 1 / sigma_i, with sigma_i estimated from a rolling lookback window.

    This is the cheapest "risk parity" approximation and is equivalent to true
    risk parity when asset correlations are zero. For SPY/AGG (correlation
    historically near zero) this is essentially equivalent to ERC.
    """
    sigma = asset_rets.rolling(lookback, min_periods=12).std()
    inv = 1.0 / sigma.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    # Backfill the first lookback-1 months with 50/50 fallback
    w = w.fillna(0.5)
    return w


# ----------------------------------------------------------------------
# 4. Risk parity (equal risk contribution), 2-asset closed form
# ----------------------------------------------------------------------
def risk_parity_2asset(asset_rets: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """Equal-risk-contribution weights for the 2-asset case (Maillard/Roncalli/Teiletche 2010).

    For two assets with vols (s1, s2) and correlation rho, the ERC weight on
    asset 1 is the root in [0, 1] of a quadratic that simplifies. We compute
    numerically per row via the analytic solution

        w1 = sigma2 / (sigma1 + sigma2)    when rho = 0   (matches inverse-vol)

    and use a bisection search for rho != 0. For SPY/AGG, rho is small enough
    that the inverse-vol form is within a few basis points of true ERC, but we
    implement the proper version.
    """
    cols = asset_rets.columns
    n_assets = len(cols)
    assert n_assets == 2, "risk_parity_2asset requires exactly 2 assets"
    weights = []
    for i in range(len(asset_rets)):
        if i < lookback:
            weights.append([0.5, 0.5])
            continue
        window = asset_rets.iloc[i - lookback:i]
        cov = window.cov().values
        w = _erc_2asset(cov)
        weights.append(w)
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


def _erc_2asset(cov: np.ndarray) -> np.ndarray:
    """Bisection solver for the 2-asset ERC weights."""
    s1 = np.sqrt(cov[0, 0]); s2 = np.sqrt(cov[1, 1])
    rho = cov[0, 1] / max(s1 * s2, 1e-12)
    # Risk contribution of asset 1: w1 * (w1 sigma1^2 + w2 sigma1 sigma2 rho)
    # Set RC1 = RC2 and solve for w1 ∈ (0, 1).
    def f(w1):
        w2 = 1 - w1
        rc1 = w1 * (w1 * s1**2 + w2 * s1 * s2 * rho)
        rc2 = w2 * (w2 * s2**2 + w1 * s1 * s2 * rho)
        return rc1 - rc2
    lo, hi = 1e-6, 1 - 1e-6
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    w1 = 0.5 * (lo + hi)
    return np.array([w1, 1 - w1])


# ----------------------------------------------------------------------
# 5. Mean-variance with Ledoit-Wolf shrinkage covariance
# ----------------------------------------------------------------------
def mean_variance_lw(
    asset_rets: pd.DataFrame, lookback: int = 60, risk_aversion: float = 2.0,
    long_only: bool = True,
) -> pd.DataFrame:
    """Long-only Markowitz mean-variance with Ledoit-Wolf shrunk covariance.

    Solves at each rebalance:
        max_w   mu^T w  -  (gamma/2) w^T Sigma_shrunk w
        s.t.    sum(w) = 1,  w >= 0
    where mu and Sigma are computed from a trailing `lookback`-month window.

    For 2 assets and long-only, the unconstrained optimum is

        w_star = (Sigma^-1 mu) / (1^T Sigma^-1 mu)

    rescaled by gamma. We clip and renormalise to satisfy long-only.

    References: Markowitz (1952), Ledoit & Wolf (2003, 2004). The shrinkage
    estimator is the constant-correlation target convex combination from L&W.
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        LedoitWolf = None

    weights = []
    cols = asset_rets.columns
    fallback = np.array([0.6, 0.4])
    for i in range(len(asset_rets)):
        if i < lookback:
            weights.append(fallback)
            continue
        window = asset_rets.iloc[i - lookback:i].values
        mu = window.mean(axis=0)
        if LedoitWolf is not None:
            try:
                Sigma = LedoitWolf().fit(window).covariance_
            except Exception:
                Sigma = np.cov(window.T)
        else:
            Sigma = np.cov(window.T)
        try:
            inv = np.linalg.inv(Sigma + 1e-8 * np.eye(Sigma.shape[0]))
            w_raw = inv @ mu / risk_aversion
            if long_only:
                w_raw = np.clip(w_raw, 0.0, None)
            if w_raw.sum() <= 0:
                w = fallback
            else:
                w = w_raw / w_raw.sum()
        except np.linalg.LinAlgError:
            w = fallback
        weights.append(w)
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


# ----------------------------------------------------------------------
# 6. Faber 10-month moving-average rule
# ----------------------------------------------------------------------
def faber_10mo_sma(
    asset_rets: pd.DataFrame, lookback: int = 10, defensive_asset: str = "agg_ret",
) -> pd.DataFrame:
    """Mebane Faber's 2007 "Quantitative Approach to Tactical Asset Allocation"

    Per asset, hold if price > N-month moving average, else move to the
    defensive asset. For SPY/AGG, the typical implementation holds SPY when
    SPY > SMA10(SPY), else holds AGG.

    Returns a 2-asset weight vector (allocation to stock vs bond) each month.
    """
    # Reconstruct synthetic price series from log returns
    price = np.exp(asset_rets.cumsum())
    sma = price.rolling(lookback, min_periods=lookback).mean()
    above_sma = price > sma

    weights = []
    cols = list(asset_rets.columns)
    spy_col = cols[0]  # convention: first column is risky (stock)
    bond_col = cols[1]
    for i in range(len(asset_rets)):
        if i < lookback:
            weights.append([0.5, 0.5])
            continue
        in_stock = bool(above_sma.iloc[i][spy_col])
        in_bond = bool(above_sma.iloc[i][bond_col])
        # If risky is above its MA, hold it; else go defensive (bond).
        # If bond is also below MA, still hold bond (cash-equivalent).
        if in_stock:
            w = [1.0, 0.0]
        else:
            w = [0.0, 1.0]
        weights.append(w)
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


# ----------------------------------------------------------------------
# 7. Time-series momentum (Moskowitz/Ooi/Pedersen 2012)
# ----------------------------------------------------------------------
def ts_momentum_12mo(asset_rets: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Per-asset sign-of-12-month-return rule (Moskowitz/Ooi/Pedersen 2012).

    Long an asset if its trailing 12-month return is positive; short / out
    otherwise. We translate "out" to weight 0 and renormalise; if both assets
    are negative, fall back to equal weight (so we don't have a 0/0 portfolio).
    """
    cum = asset_rets.rolling(lookback, min_periods=lookback).sum()
    signal = (cum > 0).astype(float)
    # Renormalise rows
    row_sums = signal.sum(axis=1)
    fallback = np.where(row_sums == 0)[0]
    signal_renorm = signal.div(row_sums.replace(0, np.nan), axis=0).fillna(0.5)
    return signal_renorm


# ----------------------------------------------------------------------
# 8. Volatility-targeting overlay on 60/40
# ----------------------------------------------------------------------
def vol_target_60_40(
    asset_rets: pd.DataFrame, target_vol: float = 0.10, lookback: int = 36,
) -> pd.DataFrame:
    """Scale 60/40 stock exposure to hit an annualised target vol.

    Compute realised vol of the 60/40 portfolio over the last `lookback` months,
    scale the stock weight by min(target/realised, cap=1.5). The remaining
    weight goes to bonds. Long-only and capped to keep within sensible weights.
    """
    base_w = np.array([0.6, 0.4])
    base_ret = asset_rets @ base_w
    realised = base_ret.rolling(lookback, min_periods=12).std() * np.sqrt(12)
    scale = (target_vol / realised).clip(upper=1.5)

    weights = []
    cols = asset_rets.columns
    for i in range(len(asset_rets)):
        if np.isnan(scale.iloc[i]):
            w = base_w.copy()
        else:
            s = float(scale.iloc[i])
            w_stock = 0.6 * s
            w_stock = float(np.clip(w_stock, 0.0, 1.0))
            w = np.array([w_stock, 1.0 - w_stock])
        weights.append(w)
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


# ----------------------------------------------------------------------
# 9. Myopic 12-month linear scoring (parity baseline)
# ----------------------------------------------------------------------
def myopic_12mo(
    asset_rets: pd.DataFrame, action_grid: np.ndarray | None = None, lookback: int = 12,
) -> pd.DataFrame:
    """Myopic argmax: pick the action that maximises trailing-mean portfolio return.

    Same baseline already implemented in experiments/05_backtest_compare.py.
    Replicated here so all baselines live in one module for the
    08_baselines_comparison run. Discrete action grid; default 6 weights from
    100/0 to 0/100.
    """
    if action_grid is None:
        action_grid = np.array([
            [0.0, 1.0], [0.2, 0.8], [0.4, 0.6],
            [0.6, 0.4], [0.8, 0.2], [1.0, 0.0],
        ])
    mu = asset_rets.rolling(lookback, min_periods=lookback).mean()
    weights = []
    cols = asset_rets.columns
    fallback = np.array([0.6, 0.4])
    for i in range(len(asset_rets)):
        if mu.iloc[i].isna().any():
            weights.append(fallback)
            continue
        scores = action_grid @ mu.iloc[i].values
        weights.append(action_grid[int(np.argmax(scores))])
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


# ----------------------------------------------------------------------
# 10. HMM-conditional mean-variance (Guidolin / Timmermann style)
# ----------------------------------------------------------------------
def hmm_conditional_mv(
    asset_rets: pd.DataFrame,
    belief_series: pd.DataFrame,       # (T, K) belief at each date
    regime_means: list[np.ndarray],    # per-regime mean asset returns, length K
    regime_covs: list[np.ndarray],     # per-regime cov matrices, length K
    risk_aversion: float = 2.0,
    long_only: bool = True,
) -> pd.DataFrame:
    """Per-date MV optimisation against belief-weighted regime moments.

    At each t, given belief b_t over K regimes:
        mu_t = sum_k b_t(k) * mu_k
        Sigma_t = sum_k b_t(k) * (Sigma_k + mu_k mu_k^T) - mu_t mu_t^T
    Then solve the same QP as `mean_variance_lw` (long-only, fully-invested).

    This is the natural "regime-conditional Markowitz" that Guidolin and
    Timmermann (2007) study at the multi-asset MS-VAR level.
    """
    weights = []
    cols = asset_rets.columns
    fallback = np.array([0.6, 0.4])
    for i in range(len(asset_rets)):
        b = belief_series.iloc[i].values
        if np.isnan(b).any():
            weights.append(fallback); continue
        K = len(regime_means)
        # mixture mean
        mu = sum(b[k] * regime_means[k] for k in range(K))
        # mixture covariance (sum of conditional second moments minus outer product of mixture mean)
        Sigma = sum(b[k] * (regime_covs[k] + np.outer(regime_means[k], regime_means[k])) for k in range(K))
        Sigma = Sigma - np.outer(mu, mu)
        # Solve unconstrained MV, then long-only clip
        try:
            inv = np.linalg.inv(Sigma + 1e-8 * np.eye(Sigma.shape[0]))
            w_raw = inv @ mu / risk_aversion
            if long_only:
                w_raw = np.clip(w_raw, 0.0, None)
            if w_raw.sum() <= 0:
                w = fallback
            else:
                w = w_raw / w_raw.sum()
        except np.linalg.LinAlgError:
            w = fallback
        weights.append(w)
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


# ----------------------------------------------------------------------
# 11. Black-Litterman with HMM-derived views
# ----------------------------------------------------------------------
def black_litterman_hmm_views(
    asset_rets: pd.DataFrame,
    belief_series: pd.DataFrame,
    regime_means: list[np.ndarray],
    market_weights: np.ndarray = np.array([0.6, 0.4]),
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    view_confidence: float = 0.3,
    lookback: int = 60,
) -> pd.DataFrame:
    """Black-Litterman (1992) with views derived from the HMM belief.

    Steps each month:
      1. Compute the prior (equilibrium) expected returns from market weights
         and a sample covariance: Pi = lambda * Sigma * w_mkt.
      2. Form K absolute views, one per regime, asserting that under regime k
         the per-asset means are mu_k (HMM-estimated). The view-uncertainty
         matrix Omega is diagonal with entries scaled by the inverse confidence.
      3. Weight the views by the current belief b_t(k) so the BL posterior pulls
         toward the regimes the model currently believes in.
      4. Posterior mean: mu_BL = [(tau Sigma)^-1 + P^T Omega^-1 P]^-1
                                 [(tau Sigma)^-1 Pi + P^T Omega^-1 Q].
      5. Solve max w^T mu_BL - (lambda/2) w^T Sigma w long-only.

    See: Black & Litterman (1990, 1992), He & Litterman (1999) for the standard
    derivation.
    """
    weights = []
    cols = asset_rets.columns
    fallback = market_weights.copy()
    K = len(regime_means)
    n = asset_rets.shape[1]
    for i in range(len(asset_rets)):
        if i < lookback:
            weights.append(fallback); continue
        window = asset_rets.iloc[i - lookback:i].values
        Sigma = np.cov(window.T)
        Pi = risk_aversion * Sigma @ market_weights

        # Construct P (K views x n assets) and Q (K views) from regime-conditional means
        P = np.tile(np.eye(n), (K, 1))[:K * n].reshape(K * n, n)[:K]  # one-asset-per-view structure
        # Actually we want one view per regime asserting the asset return vector
        # equals mu_k. That's K vector views, each of length n. Standard BL does
        # scalar views, so we stack one view per (regime, asset) pair, K * n total.
        P = np.zeros((K * n, n))
        Q = np.zeros(K * n)
        b = belief_series.iloc[i].values
        if np.isnan(b).any():
            weights.append(fallback); continue
        for k in range(K):
            for j in range(n):
                row = k * n + j
                P[row, j] = 1.0
                Q[row] = regime_means[k][j]
        # Omega: diagonal, view variance scaled by belief, sure views (high belief)
        # get low Omega so they pull the posterior toward themselves
        omega_diag = np.zeros(K * n)
        for k in range(K):
            for j in range(n):
                row = k * n + j
                # higher belief -> lower variance -> stronger view
                conf = max(b[k], 1e-3) * view_confidence
                omega_diag[row] = tau * Sigma[j, j] / conf
        Omega = np.diag(omega_diag)

        try:
            tauSigma_inv = np.linalg.inv(tau * Sigma + 1e-8 * np.eye(n))
            Omega_inv = np.linalg.inv(Omega + 1e-12 * np.eye(K * n))
            A = tauSigma_inv + P.T @ Omega_inv @ P
            B = tauSigma_inv @ Pi + P.T @ Omega_inv @ Q
            mu_bl = np.linalg.solve(A, B)
            # Mean-variance under the posterior
            inv = np.linalg.inv(Sigma + 1e-8 * np.eye(n))
            w_raw = inv @ mu_bl / risk_aversion
            w_raw = np.clip(w_raw, 0.0, None)
            if w_raw.sum() <= 0:
                w = fallback
            else:
                w = w_raw / w_raw.sum()
        except np.linalg.LinAlgError:
            w = fallback
        weights.append(w)
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


# ----------------------------------------------------------------------
# Backtest helper: apply weights to returns with transaction cost
# ----------------------------------------------------------------------
def backtest_from_weights(
    weights: pd.DataFrame, asset_rets: pd.DataFrame, txcost_bps: float = 5.0,
) -> pd.Series:
    """End-of-month signals, applied to NEXT month's LOG asset returns.

    Converts each asset to simple returns before portfolio aggregation.
    Cost is an approximate L1 target-weight-change cost (not drift-adjusted);
    initial allocation from cash is charged. First row stays in cash.
    """
    if txcost_bps < 0 or not np.isfinite(txcost_bps):
        raise ValueError("Nonnegative finite costs required")
    if not weights.index.equals(asset_rets.index) or not weights.columns.equals(asset_rets.columns):
        raise ValueError("Weights and returns must have identical indexes and columns")
    if not asset_rets.index.is_unique or not asset_rets.index.is_monotonic_increasing:
        raise ValueError("Chronological unique dates required")
    if not np.isfinite(weights.values).all() or not np.isfinite(asset_rets.values).all() or (weights.values < 0).any() or (weights.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("Finite long-only weights and returns required")
    aligned_w = weights.shift(1).fillna(0)
    gross = (aligned_w * np.expm1(asset_rets)).sum(axis=1)
    turnover = aligned_w.sub(aligned_w.shift(1).fillna(0)).abs().sum(axis=1)
    cost = turnover * (txcost_bps / 1e4)
    return gross - cost
