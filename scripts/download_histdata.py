#!/usr/bin/env python3
"""Download free XAUUSD 1-minute BID bars from HistData.com and convert to Parquet.

Source: https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xauusd
  * ASCII "M1" bars: `YYYYMMDD HHMMSS;open;high;low;close;volume` (BID prices, volume is 0)
  * HistData documents its timestamps as "EST without daylight saving" (fixed UTC-5).
    For this XAUUSD data that is wrong, and so is plain America/New_York: timestamps
    are UTC-5 shifted +1h while EUROPEAN summer time is in force (rule "est_eu_dst").
    Evidence (scripts/check_timezone.py): 08:30-New-York release spikes (NFP, CPI, ...)
    land at the correct UTC minute in all regimes only under this rule; fixed UTC-5 is
    1h late through the summer, America/New_York is 1h off in the March/October-November
    weeks when US and EU clocks disagree. DST changes happen on Sundays at 01:00 UTC
    while gold is closed, so the conversion is unambiguous.
  * full years are one zip per year; the current year is one zip per month

Output: <out>/XAUUSD_<year>.parquet with columns
  timestamp (UTC), bid_open, bid_high, bid_low, bid_close
These files have no ask side; the loader then builds ask = bid + modelled spread from
`data.spread_model` in the config (see ml_risk.data). Raw zips are cached in <out>/zips.

Usage:
  python scripts/download_histdata.py --start-year 2019 --end-year 2026 --out ../xauusd-ml-data/histdata
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

BASE = "https://www.histdata.com"
PAGE = BASE + "/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{period}"
UA = "Mozilla/5.0 (research data fetch)"


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def fetch_zip(pair: str, year: int, month: int | None, cache: str) -> bytes | None:
    tag = f"{year}" if month is None else f"{year}{month:02d}"
    path = os.path.join(cache, f"{pair}_{tag}.zip")
    if os.path.exists(path):
        return open(path, "rb").read()
    period = f"{year}" if month is None else f"{year}/{month}"
    page_url = PAGE.format(pair=pair.lower(), period=period)
    html = _get(page_url)
    fields = dict(re.findall(r'name="(tk|date|datemonth|platform|timeframe|fxpair)" id="\w+" value="([^"]*)"', html))
    if "tk" not in fields:
        return None  # period not published (e.g. future month)
    body = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(BASE + "/get.php", data=body, headers={
        "User-Agent": UA, "Referer": page_url, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if not data.startswith(b"PK"):
        return None
    os.makedirs(cache, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    time.sleep(2)  # be polite
    return data


def to_utc(t: pd.Series, rule: str) -> pd.Series:
    """Convert naive HistData timestamps to UTC."""
    if rule == "est_eu_dst":
        base = (t + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
        # European summer time in force? (Europe/London follows EU transition instants)
        eu_dst = base.dt.tz_convert("Europe/London").map(lambda x: bool(x.dst()))
        return base - pd.to_timedelta(eu_dst.astype(int), unit="h")
    if rule == "fixed_est":
        return (t + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
    loc = t.dt.tz_localize(rule, ambiguous="NaT", nonexistent="shift_forward")  # an IANA zone name
    return loc.dt.tz_convert("UTC")


def parse_zip(data: bytes, rule: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(name), sep=";", header=None,
                         names=["ts", "bid_open", "bid_high", "bid_low", "bid_close", "volume"])
    t = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    df["timestamp"] = to_utc(t, rule)
    df = df.dropna(subset=["timestamp"])
    return df[["timestamp", "bid_open", "bid_high", "bid_low", "bid_close"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="XAUUSD")
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tz-rule", default="est_eu_dst",
                    help="est_eu_dst (default, verified) | fixed_est | an IANA zone such as America/New_York")
    args = ap.parse_args()

    cache = os.path.join(args.out, "zips")
    this_year = dt.date.today().year
    for year in range(args.start_year, args.end_year + 1):
        frames = []
        z = fetch_zip(args.pair, year, None, cache)
        if z is not None:
            frames.append(parse_zip(z, args.tz_rule))
        elif year >= this_year - 1:
            for m in range(1, 13):
                zm = fetch_zip(args.pair, year, m, cache)
                if zm is not None:
                    frames.append(parse_zip(zm, args.tz_rule))
        if not frames:
            print(f"[warn] no data for {year}", flush=True)
            continue
        df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        path = os.path.join(args.out, f"{args.pair}_{year}.parquet")
        df.to_parquet(path, index=False)
        days = df["timestamp"].dt.normalize().nunique()
        print(f"[done] {path}: {len(df):,} bars, {days} trading days, "
              f"{df.timestamp.min()} -> {df.timestamp.max()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
