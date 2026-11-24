import numpy as np
import pandas as pd

from ml_risk.backtest import kelly_fraction, max_drawdown, simulate_equity
from ml_risk.primary import simulate_trades

P = {"min_risk_spreads": 1.0, "max_entry_delay_minutes": 5, "max_hold_minutes": 60, "tp_r": 2.5}
COSTS = {"commission_per_side_per_lot": 2.25, "contract_size": 100, "extra_slippage_per_side": 0.0}


def _m1(bid_paths):
    """bid_paths: list of (open, high, low, close); ask = bid + 0.2."""
    idx = pd.date_range("2024-01-02 10:00", periods=len(bid_paths), freq="1min", tz="UTC")
    a = np.array(bid_paths, dtype=float)
    df = pd.DataFrame(a, index=idx, columns=["bid_open", "bid_high", "bid_low", "bid_close"])
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[f"bid_{c}"] + 0.2
    return df


def _sig(side, stop, at="2024-01-02 09:55"):
    t = pd.Timestamp(at, tz="UTC")
    return pd.DataFrame({"side": [side], "close_time": [t + pd.Timedelta(minutes=5)],
                         "signal_close": [100.0], "stop": [stop]}, index=[t])


def test_long_take_profit():
    # entry at ask_open 100.2; stop 98.2 -> risk 2.0; tp 105.2 hit on bid in minute 3
    m1 = _m1([(100, 100.5, 99.5, 100.3), (100.3, 103, 100, 102), (102, 105.3, 101.9, 105)])
    tr = simulate_trades(_sig(1, 98.2), m1, P, COSTS)
    r = tr.iloc[0]
    assert r.exit_reason == "tp" and np.isclose(r.entry, 100.2) and np.isclose(r.risk, 2.0)
    assert np.isclose(r.r_gross, 2.5)
    assert np.isclose(r.cost_r, 4.5 / 100 / 2.0)
    assert r.exit_time == m1.index[2] + pd.Timedelta(minutes=1)


def test_stop_wins_ties_and_gap_fills_at_open():
    # same minute touches both stop and target -> stop
    m1 = _m1([(100, 106, 98.0, 100)])
    r = simulate_trades(_sig(1, 98.2), m1, P, COSTS).iloc[0]
    assert r.exit_reason == "sl" and np.isclose(r.r_gross, -1.0)
    # gap below the stop -> filled at the (worse) open
    m1 = _m1([(100, 100.1, 99.9, 100), (97.0, 97.5, 96.5, 97)])
    r = simulate_trades(_sig(1, 98.2), m1, P, COSTS).iloc[0]
    assert r.exit_reason == "sl" and np.isclose(r.exit, 97.0) and r.r_gross < -1.0


def test_short_uses_ask_for_exits():
    # short entry at bid_open 100; stop 101 -> risk 1; ask high 101.0 hits stop (bid high 100.8)
    m1 = _m1([(100, 100.8, 99.9, 100.1)])
    r = simulate_trades(_sig(-1, 101.0), m1, P, COSTS).iloc[0]
    assert r.exit_reason == "sl" and np.isclose(r.entry, 100.0)


def test_entry_delay_and_min_risk_filters():
    m1 = _m1([(100, 100.5, 99.5, 100.3)])
    late = _sig(1, 98.2, at="2024-01-02 09:40")  # next minute 15 min after decision
    assert simulate_trades(late, m1, P, COSTS).empty
    tight = _sig(1, 100.05)  # risk 0.15 < 1 x spread 0.2
    assert simulate_trades(tight, m1, P, COSTS).empty


def test_equity_pnl_includes_commission_and_lot_rounding():
    t0 = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    df = pd.DataFrame({"entry_time": [t0], "exit_time": [t0 + pd.Timedelta(hours=1)],
                       "side": [1], "entry": [2000.0], "exit": [2005.0], "risk": [2.0], "r_net": [2.4]})
    s = {"initial_equity": 10000, "round_lots": True, "lot_step": 0.01}
    curve, pnl, lots = simulate_equity(df, np.array([0.01]), s, COSTS)
    assert np.isclose(lots[0], 0.5)  # 100 USD / (100 oz * 2.0)
    assert np.isclose(pnl[0], 0.5 * (100 * 5.0 - 4.5))
    assert np.isclose(curve.iloc[-1], 10000 + pnl[0])


def test_kelly_matches_bot_formula_and_cap():
    # bot: kelly = (b p - q) / b ; risk = kelly * fraction, capped
    p, b = np.array([0.32, 0.25, 0.60]), np.array([2.5, 2.5, 2.5])
    k = kelly_fraction(p, b, 0.25, 0.02)
    assert np.isclose(k[0], 0.25 * (2.5 * 0.32 - 0.68) / 2.5)  # = 1.2% risk
    assert k[1] == 0.0
    assert k[2] == 0.02


def test_max_drawdown():
    assert np.isclose(max_drawdown(np.array([100, 120, 90, 130, 65])), 0.5)
