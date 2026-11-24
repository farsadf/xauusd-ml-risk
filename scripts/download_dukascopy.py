#!/usr/bin/env python3
"""Download public XAUUSD 1-minute BID and ASK candles from Dukascopy.

Source: Dukascopy Bank public historical datafeed
  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/{SIDE}_candles_min_1.bi5
Each file is LZMA-compressed; each record is 24 bytes big-endian:
  uint32 seconds-from-midnight-UTC, uint32 open, uint32 close, uint32 low,
  uint32 high, float32 volume.  Prices are integers scaled by `--point` (1000 for XAUUSD).

Output: one Parquet file per year in --out, in the repo's documented bar format
(see README "Data format"): timestamp (UTC), bid_{o,h,l,c}, ask_{o,h,l,c}, volume.
Zero-volume (no-trade) minutes are dropped.  Already-downloaded years are skipped.

Usage:
  python scripts/download_dukascopy.py --start 2019-01-01 --end 2026-11-24 --out ../xauusd-ml-data/raw
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import lzma
import os
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
REC = struct.Struct(">IIIIIf")


CACHE_DIR: str | None = None


def fetch(sym: str, day: dt.date, side: str, retries: int = 12) -> bytes:
    """Fetch one day/side file; cached on disk under CACHE_DIR so interrupted runs resume."""
    url = URL.format(sym=sym, y=day.year, m=day.month - 1, d=day.day, side=side)
    cpath = None
    if CACHE_DIR:
        cpath = os.path.join(CACHE_DIR, sym, f"{day.isoformat()}_{side}.bi5")
        if os.path.exists(cpath):
            with open(cpath, "rb") as f:
                return f.read()
    data = _fetch_url(url, retries)
    if cpath is not None:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        tmp = cpath + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, cpath)
    return data


def _fetch_url(url: str, retries: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data fetch)"})
    delay = 2.0
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"failed after {retries} retries: {url}")


def decode(raw: bytes, day: dt.date, point: float) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    b = lzma.decompress(raw)
    a = np.frombuffer(b, dtype=np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"),
                                         ("l", ">u4"), ("h", ">u4"), ("v", ">f4")]))
    base = pd.Timestamp(day, tz="UTC")
    return pd.DataFrame({
        "timestamp": base + pd.to_timedelta(a["t"].astype(np.int64), unit="s"),
        "o": a["o"] / point, "h": a["h"] / point, "l": a["l"] / point, "c": a["c"] / point,
        "v": a["v"].astype(np.float64),
    })


def fetch_day(sym: str, day: dt.date, point: float) -> pd.DataFrame | None:
    """None = download failed (retried in a later pass); empty frame = no trading that day."""
    try:
        bid = decode(fetch(sym, day, "BID"), day, point)
        ask = decode(fetch(sym, day, "ASK"), day, point)
    except RuntimeError as e:
        print(f"[warn] {e}", flush=True)
        return None
    if bid.empty or ask.empty:
        return pd.DataFrame()
    df = bid.merge(ask, on="timestamp", suffixes=("_b", "_a"))
    df = df[(df["v_b"] > 0) & (df["v_a"] > 0)]
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "bid_open": df["o_b"], "bid_high": df["h_b"], "bid_low": df["l_b"], "bid_close": df["c_b"],
        "ask_open": df["o_a"], "ask_high": df["h_a"], "ask_low": df["l_a"], "ask_close": df["c_a"],
        "volume": df["v_b"],
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="inclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--point", type=float, default=1000.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default=None, help="raw .bi5 cache dir (default: <out>/bi5)")
    ap.add_argument("--passes", type=int, default=3, help="re-attempt failed days this many times")
    args = ap.parse_args()
    global CACHE_DIR
    CACHE_DIR = args.cache or os.path.join(args.out, "bi5")

    os.makedirs(args.out, exist_ok=True)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    for year in range(start.year, end.year + 1):
        path = os.path.join(args.out, f"{args.symbol}_{year}.parquet")
        y0, y1 = max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31))
        if os.path.exists(path) and year != end.year:
            print(f"[skip] {path}", flush=True)
            continue
        days = [y0 + dt.timedelta(days=i) for i in range((y1 - y0).days + 1)]
        days = [d for d in days if d.weekday() != 5]  # Saturday: market closed
        got: dict[dt.date, pd.DataFrame] = {}
        todo = days
        for attempt in range(args.passes):
            with cf.ThreadPoolExecutor(args.workers) as ex:
                for i, (d, df) in enumerate(zip(todo, ex.map(lambda d: fetch_day(args.symbol, d, args.point), todo))):
                    if df is not None:
                        got[d] = df
                    if i % 25 == 0:
                        print(f"[{year}] pass {attempt + 1}: {i}/{len(todo)} days", flush=True)
            todo = [d for d in days if d not in got]
            if not todo:
                break
        if todo:
            print(f"[error] {year}: {len(todo)} days still failing, e.g. {todo[:3]}; re-run to resume", flush=True)
            continue
        frames = [got[d] for d in days]
        out = pd.concat([f for f in frames if not f.empty]).sort_values("timestamp").reset_index(drop=True)
        out.to_parquet(path, index=False)
        print(f"[done] {path}: {len(out):,} bars {out.timestamp.min()} -> {out.timestamp.max()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
