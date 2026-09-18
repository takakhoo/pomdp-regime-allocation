import numpy as np
import pandas as pd
import pytest
from src.models.mdp import value_iteration, policy_iteration, q_function
from src.models.qmdp import stationary_distribution, filter_log_emissions, update_belief
from src.models.baselines import backtest_from_weights
from src.utils.metrics import max_drawdown, ulcer_index
from reproduce_chronological import ROOT, fit_policy, signals


def test_cross_state_bellman_and_pi_agree():
    P = np.array([[[0,1],[1,0]], [[1,0],[0,1]]], float)
    R = np.array([[0,.3],[1,0]], float)
    V, pi, _ = value_iteration(P, R, .9, 1e-10)
    exact, policy, _ = policy_iteration(P,R,.9)
    np.testing.assert_allclose(V, exact, atol=1e-9)
    np.testing.assert_array_equal(pi, policy)
    assert V[0] > 4  # diagonal-only contraction gives .3/.1 = 3
    np.testing.assert_allclose(q_function(P,R,V,.9), R + .9 * np.stack([p @ V for p in P],1))
    np.testing.assert_allclose(value_iteration(P,R,0)[0],R.max(1))
    with pytest.raises(RuntimeError): value_iteration(P,R,.99,max_iter=1)
    with pytest.raises(ValueError): value_iteration(P,R,1)


def test_stationary_and_underflow():
    T = np.array([[.9,.1],[.2,.8]])
    np.testing.assert_allclose(stationary_distribution(T),[2/3,1/3])
    with pytest.raises(ValueError): stationary_distribution(np.eye(2))
    b = np.array([.8,.2])
    np.testing.assert_allclose(update_belief(b,T,np.zeros(2)),b@T)
    base = np.array([[-1,-2],[-2,-1],[-4,-3.]])
    np.testing.assert_allclose(filter_log_emissions(base,T,b),filter_log_emissions(base-10000,T,b),atol=1e-12)


def test_lag_log_conversion_and_initial_cost():
    ix = pd.date_range("2020-01-31", periods=3, freq="ME")
    rets = pd.DataFrame(np.log1p([[.1,.2],[.2,.3],[.3,.4]]), index=ix,columns=["a","b"])
    w = pd.DataFrame([[1,0],[0,1],[1,0]],index=ix,columns=rets.columns)
    np.testing.assert_allclose(backtest_from_weights(w,rets,10),[0,.2-.001,.4-.002])
    changed=w.copy();changed.iloc[-1]=[0,1]
    np.testing.assert_array_equal(backtest_from_weights(w,rets),backtest_from_weights(changed,rets))


def test_initial_wealth_drawdown():
    r = pd.Series([-.2,0])
    assert max_drawdown(r) == pytest.approx(-.2)
    assert ulcer_index(r) == pytest.approx(20)


def test_future_data_cannot_change_prior_signals_or_fit():
    df = pd.read_csv(ROOT / "data/processed/monthly.csv",index_col=0,parse_dates=True)
    model,Q,R,_=fit_policy(df)
    w,b,forecast=signals(df,model,Q)
    changed=df.copy()
    changed.loc["2020":,["vix","term_spread","spy_ret","agg_ret"]] += 2
    model2,Q2,R2,_=fit_policy(changed)
    np.testing.assert_allclose(Q,Q2)
    np.testing.assert_allclose(R,R2)
    w2,b2,_=signals(changed,model2,Q2)
    np.testing.assert_array_equal(w.loc[:"2019"],w2.loc[:"2019"])
    np.testing.assert_allclose(b[df.index.year<2020],b2[df.index.year<2020])
    np.testing.assert_array_equal((forecast@Q).argmax(1),(forecast@R).argmax(1))
