"""Proof that features (and the primary signal) at bar t use only data up to t.

Two independent checks at many random t:
  1. truncation: features computed on bars[:t+1] equal row t of features computed on all bars;
  2. future perturbation: scrambling every bar after t leaves rows <= t unchanged.
"""
import numpy as np
import pandas as pd
import pytest

from ml_risk.features import compute_market_features
from ml_risk.primary import crossover_signals

RNG = np.random.default_rng(0)


def _check_equal(a: pd.DataFrame, b: pd.DataFrame):
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


@pytest.fixture(scope="module")
def full(bars5, cfg):
    return compute_market_features(bars5, cfg["features"])


@pytest.mark.parametrize("t", sorted(RNG.integers(300, 5000, 25)))
def test_truncation_invariance(bars5, cfg, full, t):
    trunc = compute_market_features(bars5.iloc[: t + 1], cfg["features"])
    _check_equal(trunc.iloc[[t]], full.iloc[[t]])


def _perturb_after(bars: pd.DataFrame, t: int) -> pd.DataFrame:
    b = bars.copy()
    n = len(b) - t - 1
    shock = np.exp(RNG.normal(0, 0.02, n))
    for c in ["open", "high", "low", "close", "bid_open", "bid_high", "bid_low", "bid_close",
              "ask_open", "ask_high", "ask_low", "ask_close"]:
        b.iloc[t + 1:, b.columns.get_loc(c)] = b[c].iloc[t + 1:].to_numpy() * shock
    b.iloc[t + 1:, b.columns.get_loc("spread_close")] = RNG.uniform(0.05, 3.0, n)
    return b


@pytest.mark.parametrize("t", sorted(RNG.integers(300, 5000, 10)))
def test_future_perturbation_does_not_change_past_features(bars5, cfg, full, t):
    pert = compute_market_features(_perturb_after(bars5, t), cfg["features"])
    _check_equal(pert.iloc[: t + 1], full.iloc[: t + 1])
    # sanity: the perturbation does change the future, so the test has power
    assert not np.allclose(pert["bb_pct_b"].iloc[t + 30:t + 60], full["bb_pct_b"].iloc[t + 30:t + 60])


@pytest.mark.parametrize("t", sorted(RNG.integers(400, 5000, 10)))
def test_primary_signal_is_causal(bars5, cfg, t):
    s_full = crossover_signals(bars5, cfg["primary"])
    s_pert = crossover_signals(_perturb_after(bars5, t), cfg["primary"])
    cut = bars5.index[t]
    _check_equal(s_pert[s_pert.index <= cut], s_full[s_full.index <= cut])


def test_decision_time_is_bar_close(bars5):
    assert ((bars5["close_time"] - bars5.index) == pd.Timedelta(minutes=5)).all()
