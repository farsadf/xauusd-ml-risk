#!/usr/bin/env python3
"""Timezone sanity check for 1m XAUUSD bars (UTC parquet files in the repo format).

Major US releases (NFP, CPI, retail sales, jobless claims, ...) print at 08:30 New
York time = 13:30 UTC outside US daylight saving, 12:30 UTC during it. On weekdays
whose largest 1-minute range in 11:00-15:00 UTC is a clear spike (>= 8x the window
median) at minute :30, that spike must sit in the expected hour. Results are split by
regime, including the March / Oct-Nov weeks when US and EU clocks disagree.

    python scripts/check_timezone.py ../xauusd-ml-data/histdata
"""
import glob
import os
import sys

import pandas as pd


def main(path: str) -> int:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    df = pd.concat(pd.read_parquet(f) for f in files).set_index("timestamp").sort_index()
    rng = (df["bid_high"] - df["bid_low"])
    win = rng[(rng.index.hour >= 11) & (rng.index.hour < 15)]
    day = win.index.normalize()
    rows = []
    for d, w in win.groupby(day):
        if d.weekday() > 4 or len(w) < 200:
            continue
        t = w.idxmax()
        if t.minute != 30 or w.max() < 8 * w.median():
            continue
        noon = d + pd.Timedelta(hours=12)
        us = bool(noon.tz_convert("America/New_York").dst())
        eu = bool(noon.tz_convert("Europe/London").dst())
        regime = "both_dst" if us and eu else "neither" if not (us or eu) else "US_only" if us else "EU_only"
        rows.append({"date": d.date(), "regime": regime, "spike_utc": t.strftime("%H:%M"),
                     "expected": "12:30" if us else "13:30"})
    r = pd.DataFrame(rows)
    r["match"] = r["spike_utc"] == r["expected"]
    print(r.groupby("regime")["match"].agg(["mean", "sum", "size"])
          .rename(columns={"mean": "share_ok", "sum": "ok", "size": "spikes"}))
    bad = r[~r["match"]]
    print(f"\nmismatches: {len(bad)} / {len(r)}")
    if len(bad):
        print(bad.to_string(index=False))
    return 0 if len(bad) <= 0.02 * len(r) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
