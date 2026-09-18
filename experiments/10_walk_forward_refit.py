"""Experiment 10: Walk-forward HMM refitting (Nystrup, Madsen, Lindström 2018).

Original headline backtest fits the HMM ONCE on 2003-2014 and holds it fixed.
That's a mild source of look-ahead bias because the HMM "knows" about regimes
it has not yet observed. Nystrup et al. (2018) argue this matters and recommend
annual or quarterly refitting.

This experiment compares:
  Variant A, Fixed HMM (current report baseline)
  Variant B, Annual refit (fit on rolling 5y window, applied for next 12 months)
  Variant C, Quarterly refit (same window, faster updates)
  Variant D, Expanding-window refit (use ALL prior data, refit yearly)

For each variant we backtest QMDP and the HMM-conditional MV baseline (the most
HMM-dependent strategies). Static 60/40 is the no-HMM benchmark.

Outputs:
  results/walk_forward_metrics.csv
  figures/walk_forward_equity.{pdf,png}
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402

from src.models.baselines import (  # noqa: E402
    static_60_40, hmm_conditional_mv, backtest_from_weights,
)
from src.models.mdp import value_iteration, q_function  # noqa: E402
from src.models.hmm import fit_hmm  # noqa: E402
from src.models.qmdp import update_belief, stationary_distribution  # noqa: E402
from src.utils.metrics import summarize  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
FIG = REPO / "figures"

N_STATES = 2
LAMBDA = 0.95
GAMMA = 2.0
TXCOST_BPS = 5.0
N_SIM_REWARD = 3000
ACTIONS = np.array([
    [0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0],
])
OBS_COLS = ["vix", "term_spread"]


def standardize_with(train: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0); sd = train.std(axis=0); sd[sd == 0] = 1.0
    return (train - mu) / sd, (full - mu) / sd


def _gauss_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    d = x.shape[0]; diff = x - mu
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    norm = 1.0 / np.sqrt((2 * np.pi) ** d * max(np.linalg.det(cov), 1e-30))
    return float(norm * np.exp(-0.5 * diff @ inv @ diff))


def _build_R(regime_returns: dict[int, np.ndarray], gamma: float = GAMMA,
             rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    R = np.zeros((len(regime_returns), len(ACTIONS)))
    for s in range(len(regime_returns)):
        rets = regime_returns[s]
        idx = rng.integers(0, len(rets), size=N_SIM_REWARD)
        sampled = rets[idx]
        for a_idx, a in enumerate(ACTIONS):
            wealth = np.maximum(1.0 + sampled @ a, 1e-12)
            if abs(gamma - 1.0) < 1e-12:
                util = np.log(wealth)
            else:
                util = (np.power(wealth, 1 - gamma) - 1.0) / (1 - gamma)
            R[s, a_idx] = util.mean()
    return R


def run_walk_forward(
    df: pd.DataFrame, refit_every: int, train_window: int, expanding: bool = False,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Walk-forward QMDP + HMM-MV backtest with refit every `refit_every` months.

    Returns: (qmdp_returns, hmm_mv_returns, weights_qmdp_df).
    """
    asset_rets = df[["spy_ret", "agg_ret"]]
    obs_all = df[OBS_COLS].values
    n = len(df)
    if n < train_window:
        raise ValueError(f"Not enough data: need {train_window}, have {n}")

    # Pre-fit on the first training window so we have a model from t=train_window
    qmdp_w_rows, mv_w_rows = [], []
    rng = np.random.default_rng(42)
    current_model = None
    current_Q = None
    current_means_rets = None
    current_covs_rets = None
    current_belief = None
    current_T = None
    current_emiss_means = None
    current_emiss_covs = None
    last_refit_t = -1

    for t in range(n):
        if t < train_window:
            qmdp_w_rows.append([0.6, 0.4])
            mv_w_rows.append([0.6, 0.4])
            continue
        # Decide whether to refit
        if (t - train_window) % refit_every == 0 or current_model is None:
            train_start = 0 if expanding else (t - train_window)
            train_obs_raw = obs_all[train_start:t]
            _, full_std = standardize_with(train_obs_raw, obs_all)
            train_std = full_std[train_start:t]
            # Keep emission coordinates fixed until the NEXT successful refit.
            mu_t = train_obs_raw.mean(axis=0)
            sd_t = train_obs_raw.std(axis=0); sd_t[sd_t == 0] = 1
            # Fail explicitly; silently retaining a model with new scaling is invalid.
            current_model, _ = fit_hmm(train_std, n_states=N_STATES, n_restarts=3, seed=t)
            current_T = current_model.transmat_
            current_emiss_means = list(current_model.means_)
            current_emiss_covs = list(current_model.covars_)
            # Regime-conditional asset return moments from training window
            states_pred = current_model.predict(train_std)
            # Order by mean SPY return so that 0=bull, 1=bear
            train_rets = np.expm1(asset_rets.iloc[train_start:t].values)
            means = [(train_rets[states_pred == k][:, 0].mean() if (states_pred == k).sum() > 0
                     else train_rets[:, 0].mean())
                     for k in range(N_STATES)]
            order = sorted(range(N_STATES), key=lambda k: means[k], reverse=True)
            current_T = current_T[np.ix_(order, order)]
            current_emiss_means = [current_emiss_means[k] for k in order]
            current_emiss_covs = [current_emiss_covs[k] for k in order]
            states_pred = np.array([order.index(s) for s in states_pred])

            # Per-regime asset return arrays
            regime_returns = {}
            regime_covs = []
            regime_means_rets_list = []
            for k in range(N_STATES):
                mask = states_pred == k
                if mask.sum() < 5:
                    rr = train_rets
                else:
                    rr = train_rets[mask]
                regime_returns[k] = rr
                regime_means_rets_list.append(rr.mean(axis=0))
                regime_covs.append(np.cov(rr.T))
            current_means_rets = regime_means_rets_list
            current_covs_rets = regime_covs

            # Build R, solve MDP
            R = _build_R(regime_returns, gamma=GAMMA, rng=rng)
            P = np.array([current_T for _ in range(len(ACTIONS))])
            V, pi_idx, _ = value_iteration(P, R, lam=LAMBDA, eps=1e-4)
            current_Q = q_function(P, R, V, LAMBDA)
            current_belief = stationary_distribution(current_T)
            last_refit_t = t

        # Standardize this month's obs using THIS refit's training stats
        o_std = (obs_all[t] - mu_t) / sd_t

        # Belief update
        likelihood = np.array([_gauss_pdf(o_std, current_emiss_means[k], current_emiss_covs[k])
                               for k in range(N_STATES)])
        current_belief = update_belief(current_belief, current_T, likelihood)

        # QMDP action
        next_belief = current_belief @ current_T
        a_idx = int((next_belief @ current_Q).argmax())
        qmdp_w_rows.append(ACTIONS[a_idx].tolist())

        # HMM-conditional MV
        K = N_STATES
        mu_mv = sum(next_belief[k] * current_means_rets[k] for k in range(K))
        Sigma_mv = sum(next_belief[k] * (current_covs_rets[k]
                       + np.outer(current_means_rets[k], current_means_rets[k]))
                       for k in range(K))
        Sigma_mv = Sigma_mv - np.outer(mu_mv, mu_mv)
        try:
            inv = np.linalg.inv(Sigma_mv + 1e-8 * np.eye(2))
            w_raw = inv @ mu_mv / GAMMA
            w_raw = np.clip(w_raw, 0.0, None)
            w_mv = w_raw / w_raw.sum() if w_raw.sum() > 0 else np.array([0.6, 0.4])
        except np.linalg.LinAlgError:
            w_mv = np.array([0.6, 0.4])
        mv_w_rows.append(w_mv.tolist())

    qmdp_w = pd.DataFrame(qmdp_w_rows, index=df.index, columns=["spy_ret", "agg_ret"])
    mv_w   = pd.DataFrame(mv_w_rows,   index=df.index, columns=["spy_ret", "agg_ret"])

    qmdp_rets = backtest_from_weights(qmdp_w, asset_rets, TXCOST_BPS)
    mv_rets   = backtest_from_weights(mv_w,   asset_rets, TXCOST_BPS)
    return qmdp_rets, mv_rets, qmdp_w


def main() -> None:
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)
    asset_rets = df[["spy_ret", "agg_ret"]]

    # ---- Static benchmark ----
    bench = backtest_from_weights(static_60_40(asset_rets), asset_rets, TXCOST_BPS)

    # ---- Variant A: fixed HMM (the existing report baseline) ----
    # Use existing baselines pipeline with the saved 2-state HMM. For consistency
    # with the other variants here, we just *run with refit_every=10000* on the
    # full dataset, which is functionally equivalent (no refit ever).
    print("\n=== Variant A: fixed HMM (no refit, train 2003-2014) ===")
    train_window_A = 135  # months from 2003-10 to 2014-12
    qmdp_A, mv_A, _ = run_walk_forward(df, refit_every=10000, train_window=train_window_A,
                                        expanding=False)

    print("\n=== Variant B: annual refit, rolling 5y window ===")
    qmdp_B, mv_B, _ = run_walk_forward(df, refit_every=12, train_window=60)

    print("\n=== Variant C: quarterly refit, rolling 5y window ===")
    qmdp_C, mv_C, _ = run_walk_forward(df, refit_every=3,  train_window=60)

    print("\n=== Variant D: expanding window, annual refit ===")
    qmdp_D, mv_D, _ = run_walk_forward(df, refit_every=12, train_window=60, expanding=True)

    variants = {
        "static_60_40":    bench,
        "qmdp_fixed":      qmdp_A,
        "qmdp_annual":     qmdp_B,
        "qmdp_quarterly":  qmdp_C,
        "qmdp_expanding":  qmdp_D,
        "hmm_mv_fixed":    mv_A,
        "hmm_mv_annual":   mv_B,
        "hmm_mv_quarterly": mv_C,
        "hmm_mv_expanding": mv_D,
    }

    rows = []
    for name, r in variants.items():
        rows.append(summarize(r, name=name, benchmark=bench))
    metrics = pd.DataFrame(rows).set_index("name").round(4)
    metrics.to_csv(RES / "walk_forward_metrics.csv")
    print("\n=== WALK-FORWARD METRICS ===")
    cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "calmar"]
    print(metrics[cols])
    print(f"\nWrote {RES / 'walk_forward_metrics.csv'}")

    # ---- Equity curves ----
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, r in variants.items():
        cum = (1 + r).cumprod()
        lw = 2.0 if name == "static_60_40" else 1.2
        ls = "-" if name == "static_60_40" else "--" if "fixed" in name else "-"
        ax.plot(cum.index, cum.values, lw=lw, label=name, linestyle=ls)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.set_title("Walk-forward refit: fixed vs annual vs quarterly vs expanding HMM")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "walk_forward_equity.pdf")
    fig.savefig(FIG / "walk_forward_equity.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG / 'walk_forward_equity.pdf'}")


if __name__ == "__main__":
    main()
