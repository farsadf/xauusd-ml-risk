"""Meta-label construction and dataset assembly.

Label: y = 1 if the primary trade's NET R-multiple (after spread, which is in the
bid/ask fills, and Fusion Zero commission) is > 0, else 0. The label is resolved at
`exit_time`; each sample therefore spans [decision_time, exit_time], which is what
the purged walk-forward uses to keep labels out of the training window.

Features are joined on the signal bar (known at decision_time = bar close), so no
feature can see the trade's outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_bars, list_input_files, resample_bars
from .features import compute_market_features, feature_names
from .primary import crossover_signals, simulate_trades


def make_labels(trades: pd.DataFrame) -> pd.Series:
    return (trades["r_net"] > 0).astype(np.int8)


def assemble(trades: pd.DataFrame, feats: pd.DataFrame, cfg_features: dict) -> pd.DataFrame:
    df = trades.join(feats, on="signal_time", how="left", rsuffix="_f")
    if cfg_features.get("include_trade_context", True):
        df["side"] = trades["side"].astype(float)
        df["stop_atr"] = (df["signal_close"] - df["stop"]).abs() / df["_atr"]
    df["y"] = make_labels(df)
    names = feature_names(cfg_features)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=names)
    return df.sort_values("decision_time").reset_index(drop=True)


def build_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (dataset, 5m bars, input files). Dataset = one row per primary trade."""
    d = cfg["data"]
    files = list_input_files(d["path"])
    m1 = load_bars(d["path"], d.get("start"), d.get("end"), d.get("spread_model"))
    bars = resample_bars(m1, d["bar_minutes"])
    feats = compute_market_features(bars, cfg["features"])
    sig = crossover_signals(bars, cfg["primary"])
    trades = simulate_trades(sig, m1, cfg["primary"], cfg["costs"])
    trades = trades.join(sig[["signal_close"]], on="signal_time")
    ds = assemble(trades, feats, cfg["features"])
    # Non-ML baseline score, known at the signal-bar close: minus the round-trip cost
    # (spread + commission per oz) as a fraction of the stop distance. Higher = cheaper trade.
    c = cfg["costs"]
    comm_per_oz = 2.0 * c["commission_per_side_per_lot"] / c["contract_size"]
    ds["cost_score"] = -(ds["_spread"] + comm_per_oz) / (ds["signal_close"] - ds["stop"]).abs()
    ref =feats["atr_pct"].rolling(cfg["sizing"]["atr"]["ref_window_bars"], min_periods=50).median().shift(1)
    ds["atr_pct_ref"] = ds["signal_time"].map(ref)
    return ds, bars, files
