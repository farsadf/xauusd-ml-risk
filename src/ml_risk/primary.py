"""TOY primary model used only to generate a public trade stream for meta-labeling.

    20/50 EMA crossover on 5m mid closes. Long when EMA20 crosses above EMA50 at
    the close of bar t (short: below). Stop at the 10-bar swing low/high (bars
    t-9..t), take-profit at 2.5R. Entry at the first 1m bar open after the signal
    bar closes (ask for longs, bid for shorts). Exits are simulated on 1m bid/ask
    bars; if stop and target are both touched inside one minute the STOP is
    assumed to fill first (conservative). A gap through the stop fills at the
    minute's open. Signals are independent events and may overlap in time.

This is deliberately simple and is NOT a production strategy. The ML layer only
grades its trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def crossover_signals(bars: pd.DataFrame, p: dict) -> pd.DataFrame:
    c = bars["close"]
    fast = c.ewm(span=p["ema_fast"], adjust=False).mean()
    slow = c.ewm(span=p["ema_slow"], adjust=False).mean()
    diff = fast - slow
    up = (diff > 0) & (diff.shift(1) <= 0)
    dn = (diff < 0) & (diff.shift(1) >= 0)
    side = pd.Series(0, index=bars.index, dtype=np.int8)
    side[up] = 1
    side[dn] = -1
    n = p["swing_lookback"]
    swing_low = bars["low"].rolling(n).min()
    swing_high = bars["high"].rolling(n).max()
    sig = pd.DataFrame({
        "side": side,
        "close_time": bars["close_time"],
        "signal_close": c,
        "stop": np.where(side > 0, swing_low, swing_high),
    }, index=bars.index)
    sig = sig.iloc[p["warmup_bars"]:]
    return sig[sig["side"] != 0].dropna()


def simulate_trades(signals: pd.DataFrame, m1: pd.DataFrame, p: dict, costs: dict) -> pd.DataFrame:
    """Simulate each signal on 1m bid/ask bars. Returns one row per resolved trade."""
    ts = m1.index.asi8  # ns since epoch, UTC
    bo, bh, bl, bc = (m1[f"bid_{x}"].to_numpy() for x in ("open", "high", "low", "close"))
    ao, ah, al, ac = (m1[f"ask_{x}"].to_numpy() for x in ("open", "high", "low", "close"))
    minute = np.int64(60_000_000_000)
    max_delay = p["max_entry_delay_minutes"] * minute
    max_hold = p["max_hold_minutes"] * minute
    slip = costs.get("extra_slippage_per_side", 0.0)
    cost_per_unit = 2.0 * costs["commission_per_side_per_lot"] / costs["contract_size"]  # USD per oz round trip

    rows = []
    for sig_time, s in signals.iterrows():
        side, stop = int(s["side"]), float(s["stop"])
        dec = pd.Timestamp(s["close_time"]).value
        j = int(np.searchsorted(ts, dec, side="left"))
        if j >= len(ts) or ts[j] - dec > max_delay:
            continue
        spread = ao[j] - bo[j]
        if side > 0:
            entry = ao[j] + slip
            risk = entry - stop
        else:
            entry = bo[j] - slip
            risk = stop - entry
        if not np.isfinite(risk) or risk <= 0 or risk < p["min_risk_spreads"] * spread:
            continue
        tp = entry + side * p["tp_r"] * risk
        end = int(np.searchsorted(ts, ts[j] + max_hold, side="left"))
        sl_ = slice(j, end)
        if side > 0:
            sl_hit = bl[sl_] <= stop
            tp_hit = bh[sl_] >= tp
        else:
            sl_hit = ah[sl_] >= stop
            tp_hit = al[sl_] <= tp
        hit = sl_hit | tp_hit
        if hit.any():
            k = int(np.argmax(hit))
            idx = j + k
            if sl_hit[k]:
                reason = "sl"
                exit_px = (min(stop, bo[idx]) - slip) if side > 0 else (max(stop, ao[idx]) + slip)
            else:
                reason = "tp"
                exit_px = tp
        else:
            if end >= len(ts):
                continue  # unresolved at end of data
            idx = end - 1
            reason = "timeout"
            exit_px = (bc[idx] - slip) if side > 0 else (ac[idx] + slip)
        r_gross = side * (exit_px - entry) / risk
        cost_r = cost_per_unit / risk
        rows.append({
            "signal_time": sig_time,
            "decision_time": pd.Timestamp(dec, tz="UTC"),
            "entry_time": pd.Timestamp(ts[j], tz="UTC"),
            "exit_time": pd.Timestamp(ts[idx] + minute, tz="UTC"),
            "side": side,
            "entry": entry, "stop": stop, "tp": tp, "risk": risk,
            "spread_entry": spread,
            "exit": exit_px, "exit_reason": reason,
            "r_gross": r_gross, "cost_r": cost_r, "r_net": r_gross - cost_r,
        })
    return pd.DataFrame(rows)
