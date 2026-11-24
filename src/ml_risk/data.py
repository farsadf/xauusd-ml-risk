"""Bar loading, cleaning and resampling.

Input format (one or more .parquet / .csv / .csv.gz files, or a directory of them):

    timestamp   UTC bar-open time of a 1-minute bar (ISO-8601 string or datetime)
    bid_open, bid_high, bid_low, bid_close
    ask_open, ask_high, ask_low, ask_close
    volume      optional, informational only

Bid-only files (no ask_* columns, e.g. HistData) are accepted when a `spread_model`
is given; the ask side is then synthesised as bid + spread. Supported models:
    {"type": "flat_usd", "usd": 0.20}                       constant spread in price units
    {"type": "hourly_usd", "usd": [24 values, UTC hours]}   hour-of-day profile

Broker tick data can be converted to this format by aggregating ticks into 1m
bid/ask OHLC. Bars are labelled by their OPEN time; a 5m bar labelled 10:00 covers
[10:00, 10:05) and is only known at 10:05 (its `close_time`).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

PRICE_COLS = ["bid_open", "bid_high", "bid_low", "bid_close",
              "ask_open", "ask_high", "ask_low", "ask_close"]


def list_input_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")) +
                       glob.glob(os.path.join(path, "*.csv")) +
                       glob.glob(os.path.join(path, "*.csv.gz")))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no bar files found at {path}")
    return files


def _read_one(p: str) -> pd.DataFrame:
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def apply_spread_model(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Synthesise ask_* = bid_* + spread for bid-only data."""
    t = model["type"]
    if t == "flat_usd":
        spread = np.full(len(df), float(model["usd"]))
    elif t == "hourly_usd":
        prof = np.asarray(model["usd"], dtype=float)
        if prof.shape != (24,):
            raise ValueError("hourly_usd needs 24 values")
        spread = prof[pd.DatetimeIndex(df["timestamp"]).hour]
    else:
        raise ValueError(f"unknown spread model {t}")
    for f in ("open", "high", "low", "close"):
        df[f"ask_{f}"] = df[f"bid_{f}"].astype(float) + spread
    return df


def load_bars(path: str, start: str | None = None, end: str | None = None,
              spread_model: dict | None = None) -> pd.DataFrame:
    """Load 1m bid/ask bars, validate, clean. Returns a DataFrame indexed by UTC timestamp."""
    df = pd.concat([_read_one(p) for p in list_input_files(path)], ignore_index=True)
    if spread_model is not None:
        if "ask_close" in df.columns:
            raise ValueError("spread_model given but the data already has ask prices")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = apply_spread_model(df, spread_model)
    missing = [c for c in ["timestamp", *PRICE_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return clean_bars(df)


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with non-positive prices, crossed quotes, or inconsistent OHLC."""
    df = df.copy()
    df[PRICE_COLS] = df[PRICE_COLS].astype(np.float64)
    ok = (df[PRICE_COLS] > 0).all(axis=1)
    ok &= df["ask_close"] >= df["bid_close"]
    ok &= df["ask_open"] >= df["bid_open"]
    for side in ("bid", "ask"):
        ok &= df[f"{side}_high"] >= df[[f"{side}_open", f"{side}_close"]].max(axis=1) - 1e-9
        ok &= df[f"{side}_low"] <= df[[f"{side}_open", f"{side}_close"]].min(axis=1) + 1e-9
    return df[ok]


def resample_bars(m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1m bid/ask bars to `minutes` bars (left-labelled, left-closed).

    Adds mid OHLC, `spread_close` (ask_close - bid_close of the last minute) and
    `close_time` (label + minutes) — the earliest time the bar is observable.
    """
    agg = {}
    for side in ("bid", "ask"):
        agg[f"{side}_open"] = "first"
        agg[f"{side}_high"] = "max"
        agg[f"{side}_low"] = "min"
        agg[f"{side}_close"] = "last"
    out = m1[list(agg)].resample(f"{minutes}min", label="left", closed="left").agg(agg).dropna()
    for f in ("open", "high", "low", "close"):
        out[f] = (out[f"bid_{f}"] + out[f"ask_{f}"]) / 2.0
    out["spread_close"] = out["ask_close"] - out["bid_close"]
    out["close_time"] = out.index + pd.Timedelta(minutes=minutes)
    return out
