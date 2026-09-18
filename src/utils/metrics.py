"""Performance metrics for monthly return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12  # monthly rebalancing


def cagr(returns: pd.Series) -> float:
    cum = (1.0 + returns).prod()
    yrs = len(returns) / PERIODS_PER_YEAR
    return cum ** (1.0 / yrs) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    return returns.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / PERIODS_PER_YEAR
    if excess.std(ddof=0) == 0:
        return 0.0
    return excess.mean() / excess.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax().clip(lower=1.0)
    return (cum / running_max - 1.0).min()  # negative number


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return cagr(returns) / abs(mdd)


def turnover(weights: pd.DataFrame) -> float:
    """Average monthly L1 turnover."""
    return (weights.diff().abs().sum(axis=1)).mean()


def downside_vol(returns: pd.Series, mar: float = 0.0) -> float:
    """Annualised downside (semi-)deviation. mar = monthly minimum acceptable return."""
    diff = (returns - mar).clip(upper=0.0)
    return np.sqrt((diff ** 2).mean()) * np.sqrt(PERIODS_PER_YEAR)


def sortino_ratio(returns: pd.Series, mar: float = 0.0, rf: float = 0.0) -> float:
    """Sortino ratio: (annual excess return) / (annual downside vol). Penalises only downside variability."""
    dv = downside_vol(returns, mar)
    if dv == 0:
        return 0.0
    excess = (returns - rf / PERIODS_PER_YEAR).mean() * PERIODS_PER_YEAR
    return excess / dv


def omega_ratio(returns: pd.Series, mar: float = 0.0) -> float:
    """Omega ratio: ratio of probability-weighted gains above MAR to losses below MAR.

    Equivalent to: sum(max(r - mar, 0)) / sum(max(mar - r, 0)). Treats every distributional
    moment, not just mean/variance.
    """
    diff = returns - mar
    pos = diff[diff > 0].sum()
    neg = -diff[diff < 0].sum()
    if neg == 0:
        return np.inf
    return float(pos / neg)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """IR: annualised mean of (strategy - benchmark) divided by tracking error.

    Standard performance attribution metric for active managers.
    """
    active = (returns - benchmark).dropna()
    te = active.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)
    if te == 0:
        return 0.0
    return float(active.mean() * PERIODS_PER_YEAR / te)


def hit_rate(returns: pd.Series, benchmark: pd.Series | None = None) -> float:
    """Fraction of months with positive return (or, if benchmark given, fraction beating benchmark)."""
    if benchmark is None:
        return float((returns > 0).mean())
    return float((returns > benchmark).mean())


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer index: RMS of percentage drawdowns. Penalises depth AND duration of drawdowns."""
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax().clip(lower=1.0)
    dd_pct = (cum / running_max - 1.0) * 100.0  # percent
    return float(np.sqrt((dd_pct ** 2).mean()))


def tail_ratio(returns: pd.Series, q: float = 0.05) -> float:
    """Tail ratio: |q-th right tail| / |q-th left tail|. Higher = more positively skewed."""
    lower = returns.quantile(q)
    upper = returns.quantile(1 - q)
    if lower == 0:
        return np.nan
    return float(abs(upper) / abs(lower))


def summarize(returns: pd.Series, weights: pd.DataFrame | None = None,
              benchmark: pd.Series | None = None, name: str = "") -> dict:
    out = {
        "name": name,
        "cagr": cagr(returns),
        "vol": annualized_vol(returns),
        "downside_vol": downside_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "omega": omega_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns),
        "ulcer": ulcer_index(returns),
        "tail_ratio": tail_ratio(returns),
        "hit_rate": hit_rate(returns),
    }
    if weights is not None:
        out["turnover_avg"] = turnover(weights)
    if benchmark is not None:
        out["info_ratio_vs_bench"] = information_ratio(returns, benchmark)
        out["hit_rate_vs_bench"] = hit_rate(returns, benchmark)
    return out
