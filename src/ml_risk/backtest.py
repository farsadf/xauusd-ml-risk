"""Sizing policies on a fixed trade stream, equity simulation, and risk metrics.

All four policies trade the SAME primary signals with the SAME stops/targets; only
the risk fraction per trade differs:

  fixed       1% of equity at risk on every trade
  atr         1% x clip(ref_ATR% / ATR%_t, 0.5, 2): risk down-weighted when current
              5m ATR% is above its trailing 20-day median (causal), up-weighted when below
  cost_rule   NON-ML filter: 1% if cost_score = -(spread + commission)/stop_distance
              >= a threshold chosen on the training window, else skip. Tests whether
              the model beats "just avoid trades whose costs are large vs. the stop".
  meta_filter 1% if P(win) >= threshold (threshold chosen in-train), else skip
  kelly       quarter-Kelly from P(win): f = 0.25 * (b p - (1-p)) / b, capped at 2%,
              skip if <= 0. b = net payoff ratio of that trade after costs:
              (tp_r - cost_r) / (1 + cost_r). Mirrors the bot's allocator formula
              (which uses b = TP/SL while bootstrapping) but makes b cost-aware.

Position size: lots = f * equity / (contract_size * stop_distance), rounded to the
lot step. P&L = lots * contract_size * side * (exit - entry) - 2 * commission * lots.
Spread is already in the bid/ask fills. Equity is realised (closed-trade) equity;
concurrent trades are sized off the equity at their entry time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = ["fixed", "atr", "cost_rule", "meta_filter", "kelly"]


def kelly_fraction(p: np.ndarray, b: np.ndarray, frac: float, cap: float) -> np.ndarray:
    k = (b * p - (1.0 - p)) / b
    return np.clip(frac * k, 0.0, cap)


def risk_fractions(df: pd.DataFrame, policy: str, s: dict, tp_r: float) -> np.ndarray:
    base = s["fixed_risk"]
    n = len(df)
    if policy == "fixed":
        return np.full(n, base)
    if policy == "atr":
        a = s["atr"]
        mult = (df["atr_pct_ref"] / df["atr_pct"]).to_numpy()
        mult = np.where(np.isfinite(mult), mult, 1.0)
        return base * np.clip(mult, a["min_mult"], a["max_mult"])
    if policy == "cost_rule":
        return np.where(df["cost_score"].to_numpy() >= df["cost_thr"].to_numpy(), base, 0.0)
    if policy == "meta_filter":
        return np.where(df["p"].to_numpy() >= df["threshold"].to_numpy(), base, 0.0)
    if policy == "kelly":
        c = df["cost_r"].to_numpy()
        b = (tp_r - c) / (1.0 + c)
        return kelly_fraction(df["p"].to_numpy(), b, s["kelly"]["fraction"], s["kelly"]["max_risk"])
    raise ValueError(policy)


def simulate_equity(df: pd.DataFrame, fractions: np.ndarray, s: dict, costs: dict
                    ) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Return (equity curve indexed by exit time, per-trade P&L, per-trade lots)."""
    contract = costs["contract_size"]
    comm_rt = 2.0 * costs["commission_per_side_per_lot"]
    n = len(df)
    entry_t = df["entry_time"].to_numpy()
    exit_t = df["exit_time"].to_numpy()
    move = (df["side"] * (df["exit"] - df["entry"])).to_numpy()
    risk = df["risk"].to_numpy()
    # events: (time, type, i) with exits (0) processed before entries (1) at equal timestamps
    ev = sorted([(exit_t[i], 0, i) for i in range(n)] + [(entry_t[i], 1, i) for i in range(n)])
    equity = float(s["initial_equity"])
    lots = np.zeros(n)
    pnl = np.zeros(n)
    curve_t, curve_v = [], []
    for t, typ, i in ev:
        if typ == 1:
            if fractions[i] <= 0 or equity <= 0:
                continue
            q = fractions[i] * equity / (contract * risk[i])
            if s.get("round_lots", True):
                q = np.floor(q / s["lot_step"] + 1e-9) * s["lot_step"]
            lots[i] = q
        else:
            if lots[i] > 0:
                pnl[i] = lots[i] * (contract * move[i] - comm_rt)
                equity += pnl[i]
                curve_t.append(t)
                curve_v.append(equity)
    curve = pd.Series(curve_v, index=pd.DatetimeIndex(curve_t), dtype=float)
    return curve, pnl, lots


def daily_equity(curve: pd.Series, start: pd.Timestamp, end: pd.Timestamp, initial: float) -> pd.Series:
    idx = pd.bdate_range(start.normalize(), end.normalize(), tz="UTC")
    if curve.empty:
        return pd.Series(initial, index=idx)
    c = curve.groupby(curve.index.normalize()).last()
    return c.reindex(idx).ffill().fillna(initial)


def max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    return float(np.max((peak - values) / peak))


def perf_metrics(curve: pd.Series, pnl: np.ndarray, lots: np.ndarray, df: pd.DataFrame,
                 start: pd.Timestamp, end: pd.Timestamp, initial: float,
                 fractions: np.ndarray | None = None) -> dict:
    """`risk_weighted_r` = sum(f_i * R_i) / sum(f_i) over intended risk fractions: a
    scale-free measure of whether a policy puts more risk on better trades (equals the
    plain average R for fixed sizing). Unlike equity metrics it does not reward simply
    betting smaller on a negative-expectancy stream."""
    taken = lots > 0
    deq = daily_equity(curve, start, end, initial)
    rets = deq.pct_change().fillna(deq.iloc[0] / initial - 1.0)
    years = max((end - start).days / 365.25, 1e-9)
    final = float(deq.iloc[-1])
    sd = rets.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2))
    path = np.concatenate([[initial], curve.to_numpy()]) if len(curve) else np.array([initial])
    mdd = max_drawdown(path)
    cagr = (final / initial) ** (1 / years) - 1 if final > 0 else -1.0
    r = df["r_net"].to_numpy()[taken]
    rw = float("nan")
    if fractions is not None and fractions.sum() > 0:
        rw = float((fractions * df["r_net"].to_numpy()).sum() / fractions.sum())
    return {
        "trades": int(taken.sum()),
        "risk_weighted_r": rw,
        "avg_risk_frac": float(fractions[fractions > 0].mean()) if fractions is not None and (fractions > 0).any() else float("nan"),
        "win_rate": float((pnl[taken] > 0).mean()) if taken.any() else float("nan"),
        "avg_r_net": float(r.mean()) if taken.any() else float("nan"),
        "total_return": final / initial - 1.0,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe": float(np.sqrt(252) * rets.mean() / sd) if sd > 0 else float("nan"),
        "sortino": float(np.sqrt(252) * rets.mean() / downside) if downside > 0 else float("nan"),
        "calmar": float(cagr / mdd) if mdd > 0 else float("nan"),
        "final_equity": final,
    }


def block_bootstrap_sharpe_diff(ra: pd.Series, rb: pd.Series, block: int, n_boot: int, seed: int = 0) -> dict:
    """Moving-block bootstrap of Sharpe(a) - Sharpe(b) on aligned daily returns."""
    a, b = ra.to_numpy(), rb.to_numpy()
    n = len(a)
    rng = np.random.default_rng(seed)

    def sh(x):
        s = x.std(ddof=1)
        return np.sqrt(252) * x.mean() / s if s > 0 else 0.0

    nb = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), nb)
        ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        diffs[k] = sh(a[ix]) - sh(b[ix])
    return {"observed": float(sh(a) - sh(b)), "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)), "p_le_0": float((diffs <= 0).mean())}
