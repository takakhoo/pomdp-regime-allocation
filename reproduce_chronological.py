"""Offline, fixed-split diagnostic. No downloads, trading, or model selection on test."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from src.models.hmm import fit_hmm, standardize_with_train_stats
from src.models.mdp import value_iteration, q_function
from src.models.qmdp import filter_log_emissions
from src.models.baselines import backtest_from_weights
from src.utils.metrics import cagr, sharpe_ratio, max_drawdown

ROOT = Path(__file__).resolve().parent
CUTOFF = "2014-12-31"
COLS = ["vix", "term_spread"]
ASSETS = ["spy_ret", "agg_ret"]
ACTIONS = np.c_[np.linspace(0, 1, 6), np.linspace(1, 0, 6)]


def fit_policy(df):
    train = df.loc[:CUTOFF]
    z_train, _ = standardize_with_train_stats(train, train, COLS)
    model, ll = fit_hmm(z_train, 2, n_restarts=5, seed=7)
    # Smoothed state responsibilities are permitted INSIDE the training window.
    resp = model.predict_proba(z_train)
    simple = np.expm1(train[ASSETS].values)
    wealth = 1 + simple @ ACTIONS.T
    utility = 1 - 1 / wealth  # CRRA gamma=2, empirical training returns only
    R = (resp.T @ utility) / resp.sum(0)[:, None]
    P = np.repeat(model.transmat_[None], len(ACTIONS), axis=0)
    V, _, _ = value_iteration(P, R, eps=1e-9)
    Q = q_function(P, R, V, .95)
    return model, Q, R, float(ll)


def signals(df, model, Q):
    _, z = standardize_with_train_stats(df.loc[:CUTOFF], df, COLS)
    logp = np.column_stack([multivariate_normal.logpdf(z, model.means_[s], model.covars_[s]) for s in range(2)])
    beliefs = filter_log_emissions(logp, model.transmat_, model.startprob_)
    # At month t close, allocate for t+1 using its predicted regime distribution.
    forecast = beliefs @ model.transmat_
    actions = (forecast @ Q).argmax(1)
    return pd.DataFrame(ACTIONS[actions], index=df.index, columns=ASSETS), beliefs, forecast


def run(out=ROOT / "results/chronological"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/monthly.csv"
    df = pd.read_csv(source, index_col=0, parse_dates=True)[COLS + ASSETS].dropna()
    model, Q, R, ll = fit_policy(df)
    allocation, beliefs, forecast = signals(df, model, Q)
    train_end = df.index.searchsorted(pd.Timestamp(CUTOFF), side="right") - 1
    # Common cash start, first allocation at December 2014 close; entry cost included.
    evaluation = df.iloc[train_end:]
    strategies = {"QMDP": allocation.loc[evaluation.index],
                  "Static 60/40": pd.DataFrame(np.tile([.6,.4], (len(evaluation),1)), index=evaluation.index, columns=ASSETS)}
    prices = np.exp(df[ASSETS].cumsum())
    trend = (prices > prices.rolling(10).mean()).astype(float) * .5
    strategies["Lagged 10-month trend"] = trend.loc[evaluation.index]
    net = pd.DataFrame({name: backtest_from_weights(w, evaluation[ASSETS], 5).iloc[1:] for name,w in strategies.items()})
    result = {"data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "train": [str(df.index[0].date()), CUTOFF],
              "evaluation": [str(net.index[0].date()), str(net.index[-1].date())],
              "seed": 7, "hmm_states": 2, "restarts": 5, "train_log_likelihood": ll,
              "discount": .95, "crra_gamma": 2, "cost_bps_per_L1_target_change": 5,
              "qmdp_equals_myopic": bool(np.array_equal((forecast @ Q).argmax(1), (forecast @ R).argmax(1))),
              "metrics": {name: {"cagr": float(cagr(net[name])), "sharpe_zero_rf": float(sharpe_ratio(net[name])), "max_drawdown": float(max_drawdown(net[name]))} for name in net}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    net.to_csv(out / "monthly_net_returns.csv", float_format="%.12g")
    allocation.assign(regime_0=beliefs[:,0], regime_1=beliefs[:,1]).loc[evaluation.index].to_csv(out / "end_of_month_signals.csv", float_format="%.12g")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios":[2,1]})
    wealth = pd.concat([pd.DataFrame(1., index=[evaluation.index[0]], columns=net.columns), (1+net).cumprod()])
    wealth.plot(ax=axes[0], linewidth=1.8)
    axes[0].set(ylabel="Growth of 1", title="Fixed training window, next-month execution")
    allocation.loc[net.index, "spy_ret"].plot(ax=axes[1], color="#586d82", drawstyle="steps-post")
    axes[1].set(ylabel="Next-month equity weight", xlabel="Month-end decision date", ylim=(-.05,1.05))
    fig.text(.5,.01,"Historical snapshot, not point-in-time data • approximate costs • not investment advice",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(out / "chronological.png", dpi=160)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
