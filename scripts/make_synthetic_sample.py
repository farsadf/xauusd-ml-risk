#!/usr/bin/env python3
"""Generate the bundled SYNTHETIC 1m bid/ask sample (data/sample/).

A driftless random walk with GARCH(1,1)-style volatility clustering, an intraday
volatility profile, a weekend gap and a 1-hour daily break, at gold-like price and
spread levels. It contains NO real market data and NO price signal — it exists only
so the full pipeline (load -> features -> primary -> labels -> walk-forward -> report)
runs end to end from a fresh clone. It does contain a cost signal (net win probability
falls as the stop shrinks relative to spread + commission), so meta-model AUC is ~0.6
on it and the non-ML cost-rule baseline performs about as well.

    python scripts/make_synthetic_sample.py --out data/sample/xauusd_synthetic_1m.csv.gz
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample/xauusd_synthetic_1m.csv.gz")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start = pd.Timestamp(args.start, tz="UTC")
    idx = pd.date_range(start, start + pd.DateOffset(months=args.months), freq="1min", inclusive="left")
    idx = idx[(idx.weekday < 5) & (idx.hour != 21)]  # weekdays, daily break 21:00-22:00 UTC
    n = len(idx)

    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    season = 0.6 + 0.8 * np.exp(-((hours - 14.5) ** 2) / 8) + 0.3 * np.exp(-((hours - 8) ** 2) / 4)
    omega, alpha, beta = 2e-10, 0.05, 0.93
    var = np.empty(n)
    r = np.empty(n)
    v = omega / (1 - alpha - beta)
    z = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    for i in range(n):
        var[i] = v
        r[i] = np.sqrt(v) * season[i] * z[i]
        v = omega + alpha * r[i] ** 2 + beta * v
    close = 2000.0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[2000.0], close[:-1]])
    vol_px = np.sqrt(var) * season * close
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.6, n)) * vol_px
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.6, n)) * vol_px
    spread = np.clip(0.18 + 0.08 * (vol_px / np.median(vol_px) - 1) + rng.normal(0, 0.02, n), 0.08, 1.5)

    r3 = lambda x: np.round(x, 3)  # noqa: E731
    df = pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bid_open": r3(open_), "bid_high": r3(high), "bid_low": r3(low), "bid_close": r3(close),
        "ask_open": r3(open_ + spread), "ask_high": r3(high + spread),
        "ask_low": r3(low + spread), "ask_close": r3(close + spread),
        "volume": np.round(rng.gamma(2.0, 0.01, n) * season, 4),
    })
    df.to_csv(args.out, index=False, compression="gzip")
    print(f"wrote {args.out}: {n:,} rows {idx[0]} -> {idx[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
