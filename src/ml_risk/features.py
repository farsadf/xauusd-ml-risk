"""Generic market-context features, computed causally on closed bars.

Every feature at bar t uses only bars <= t (rolling/ewm windows that end at t,
and the previous close for true range). `tests/test_features_no_lookahead.py`
proves this by recomputing on truncated and future-perturbed data.

Re-implemented from the bot's Go feature extractor, keeping only strategy-agnostic
features. Deliberate changes vs. the bot: ATR is expressed as a fraction of price
(atr_pct) instead of absolute pips, because XAU roughly doubled over the sample and
absolute ATR is non-stationary; strategy-coupled features (signal strength, signal
streaks, HTF momentum, strategy EMA distance) are not included.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURES = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "atr_pct", "range_atr",
    "upper_wick_ratio", "lower_wick_ratio", "wick_to_range_ratio", "clv",
    "bb_pct_b", "stoch_k", "stoch_d", "stoch_diff", "adx",
]
SPREAD_FEATURES = ["spread_atr"]
TRADE_CONTEXT_FEATURES = ["side", "stop_atr"]


def feature_names(cfg_features: dict) -> list[str]:
    names = list(MARKET_FEATURES)
    if cfg_features.get("use_spread", True):
        names += SPREAD_FEATURES
    if cfg_features.get("include_trade_context", True):
        names += TRADE_CONTEXT_FEATURES
    return sorted(names)  # sorted: identical ordering contract to the Go inference schema


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    """Wilder's ADX (recursive smoothing with alpha = 1/n)."""
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    a = 1.0 / n
    atr_w = true_range(h, l, c).ewm(alpha=a, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr_w
    mdi = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def compute_market_features(bars: pd.DataFrame, cfg_features: dict) -> pd.DataFrame:
    """Features for every bar, indexed like `bars`. Requires mid OHLC + close_time (+ spread_close)."""
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    f = pd.DataFrame(index=bars.index)

    t = pd.DatetimeIndex(bars["close_time"])  # the decision time = bar close
    f["hour_sin"] = np.sin(2 * np.pi * t.hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * t.hour / 24.0)
    f["minute_sin"] = np.sin(2 * np.pi * t.minute / 60.0)
    f["minute_cos"] = np.cos(2 * np.pi * t.minute / 60.0)

    atr = true_range(h, l, c).rolling(cfg_features["atr_window"]).mean()
    f["atr_pct"] = atr / c
    rng = h - l
    f["range_atr"] = rng / atr

    safe_rng = rng.replace(0, np.nan)
    f["upper_wick_ratio"] = ((h - np.maximum(o, c)) / safe_rng).fillna(0.0)
    f["lower_wick_ratio"] = ((np.minimum(o, c) - l) / safe_rng).fillna(0.0)
    f["wick_to_range_ratio"] = np.maximum(f["upper_wick_ratio"], f["lower_wick_ratio"])
    f["clv"] = ((c - l) / safe_rng).fillna(0.5)

    n, k = cfg_features["bb_window"], cfg_features["bb_k"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    width = (2 * k * sd).replace(0, np.nan)
    f["bb_pct_b"] = (c - (ma - k * sd)) / width

    kn = cfg_features["stoch_k"]
    ll = l.rolling(kn).min()
    hh = h.rolling(kn).max()
    raw_k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    stoch_k = raw_k.rolling(cfg_features["stoch_smooth"]).mean()
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_k.rolling(cfg_features["stoch_d"]).mean()
    f["stoch_diff"] = f["stoch_k"] - f["stoch_d"]

    f["adx"] = adx(h, l, c, cfg_features["adx_window"])

    if cfg_features.get("use_spread", True):
        f["spread_atr"] = bars["spread_close"] / atr

    f["_atr"] = atr  # helper columns (price units), not model features
    f["_spread"] = bars["spread_close"] if "spread_close" in bars else 0.0
    return f
