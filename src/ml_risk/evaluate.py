"""Walk-forward evaluation of the meta-model vs. sizing baselines.

    python -m ml_risk.evaluate --config configs/default.yaml [--report-dir reports/<run_name>]

For every seed: for every outer fold (time-ordered, purged + embargoed), select and
fit the model using the training window only, then predict the following test window.
Out-of-sample predictions are pooled and used for
  * prediction quality: ROC-AUC, Brier, log-loss vs. a base-rate predictor, calibration
  * trading impact of 4 sizing policies on the same trades (fixed 1%, ATR, meta filter, quarter-Kelly)
  * SHAP (TreeSHAP via LightGBM pred_contrib) on out-of-sample rows
Aggregated outputs go to --report-dir; per-trade predictions stay in output_dir.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

from .backtest import (POLICIES, block_bootstrap_sharpe_diff, daily_equity, perf_metrics,  # noqa: E402
                       risk_fractions, simulate_equity)
from .config import embargo_minutes, load_config, run_info, set_seeds, write_json  # noqa: E402
from .features import feature_names  # noqa: E402
from .model import choose_score_threshold, select_and_fit  # noqa: E402
from .pipeline import dataset_cached  # noqa: E402
from .splits import outer_folds  # noqa: E402

POLICY_LABEL = {"fixed": "Fixed 1% risk", "atr": "ATR-scaled risk", "cost_rule": "Cost-rule filter (no ML)",
                "meta_filter": "Meta-model filter", "kelly": "Meta-model ¼-Kelly"}


def walk_forward(ds, names, cfg, seed, folds, emb, want_shap=False):
    X = ds[names].to_numpy(np.float64)
    parts, info, shap = [], [], []
    for f in folds:
        fm = select_and_fit(ds, f.train_idx, names, cfg, seed, emb)
        p = fm.booster.predict(X[f.test_idx])
        base = float(ds["y"].iloc[f.train_idx].mean())
        parts.append(pd.DataFrame({"row": f.test_idx, "p": p, "threshold": fm.threshold,
                                   "fold": f.fold, "base_rate": base}))
        y = ds["y"].to_numpy()[f.test_idx]
        info.append({"seed": seed, "fold": f.fold, "test_start": str(f.test_start.date()),
                     "test_end": str(f.test_end.date()), "n_train": int(len(f.train_idx)),
                     "n_test": int(len(f.test_idx)), "train_base_rate": base,
                     "test_base_rate": float(y.mean()),
                     "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
                     "threshold": fm.threshold, "n_rounds": fm.n_rounds,
                     "max_depth": fm.params["max_depth"], "min_data_in_leaf": fm.params["min_data_in_leaf"],
                     "kept_frac": float((p >= fm.threshold).mean())})
        if want_shap:
            shap.append(fm.booster.predict(X[f.test_idx], pred_contrib=True)[:, :-1])
    oos = pd.concat(parts, ignore_index=True)
    return oos, info, (np.vstack(shap) if want_shap else None)


def _logreg_reference_auc(ds, names, folds) -> float:
    """Pooled OOS AUC of an L2 logistic regression on the same features and folds
    (scaler fitted on each training window only) — checks the result is not LightGBM-specific."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = ds[names].to_numpy(np.float64), ds["y"].to_numpy()
    ps, ys = [], []
    for f in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[f.train_idx], y[f.train_idx])
        ps.append(clf.predict_proba(X[f.test_idx])[:, 1])
        ys.append(y[f.test_idx])
    return float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))


def prediction_metrics(y, p, base):
    eps = 1e-6
    out = {"auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, p)),
           "brier_base_rate": float(brier_score_loss(y, base)),
           "logloss": float(log_loss(y, np.clip(p, eps, 1 - eps), labels=[0, 1])),
           "logloss_base_rate": float(log_loss(y, np.clip(base, eps, 1 - eps), labels=[0, 1]))}
    out["brier_skill"] = 1 - out["brier"] / out["brier_base_rate"]
    return out


def calibration_table(y, p, bins=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    return pd.DataFrame({"b": b, "y": y, "p": p}).groupby("b").agg(
        p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)


def fmt_pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def fmt(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--report-dir", default=None, help="default: reports/<run_name>")
    args = ap.parse_args(argv)

    t_start = time.time()
    cfg = load_config(args.config)
    rep = Path(args.report_dir or Path("reports") / cfg["run_name"])
    rep.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, files = dataset_cached(cfg)
    names = feature_names(cfg["features"])
    emb = embargo_minutes(cfg)
    folds = outer_folds(ds, cfg["validation"], emb)
    if not folds:
        raise SystemExit("no walk-forward folds — not enough data for min_train_months + test_months")
    print(f"{len(ds):,} trades, {len(folds)} folds, features={names}", flush=True)

    s, costs, tp_r = cfg["sizing"], cfg["costs"], cfg["primary"]["tp_r"]
    init = float(s["initial_equity"])
    seeds = cfg["validation"]["seeds"]

    oos_rows = np.concatenate([f.test_idx for f in folds])
    test = ds.iloc[oos_rows].copy().reset_index(drop=True)
    test["row"] = oos_rows
    # non-ML baseline threshold, chosen on each fold's training window only
    thr_map = {}
    for f in folds:
        tr = ds.iloc[f.train_idx]
        thr = choose_score_threshold(tr["cost_score"].to_numpy(), tr["r_net"].to_numpy(), cfg["sizing"]["threshold"])
        thr_map.update({int(i): thr for i in f.test_idx})
    test["cost_thr"] = test["row"].map(thr_map)
    cost_rule_auc = float(roc_auc_score(test["y"], test["cost_score"]))
    logreg_auc = _logreg_reference_auc(ds, names, folds)
    ev_start = folds[0].test_start
    ev_end = min(folds[-1].test_end, test["exit_time"].max())

    # ---- primary stream (same for every seed) ----
    yr = test.groupby(test["decision_time"].dt.year).agg(
        trades=("y", "size"), win_rate=("y", "mean"), avg_r_gross=("r_gross", "mean"),
        avg_cost_r=("cost_r", "mean"), avg_r_net=("r_net", "mean"))
    primary_stats = {
        "all_trades": int(len(ds)), "oos_trades": int(len(test)),
        "oos_win_rate": float(test["y"].mean()), "oos_avg_r_gross": float(test["r_gross"].mean()),
        "oos_avg_cost_r": float(test["cost_r"].mean()), "oos_avg_r_net": float(test["r_net"].mean()),
        "timeout_rate": float((ds["exit_reason"] == "timeout").mean()),
        "median_hold_minutes": float(((ds["exit_time"] - ds["entry_time"]).dt.total_seconds() / 60).median()),
        "max_hold_minutes": float(((ds["exit_time"] - ds["entry_time"]).dt.total_seconds() / 60).max()),
        "data_start": str(ds["decision_time"].min()), "data_end": str(ds["decision_time"].max()),
        "oos_start": str(ev_start.date()), "oos_end": str(ev_end.date()),
        "cost_score_auc": cost_rule_auc,
        "logreg_auc": logreg_auc,
    }

    # ---- seeds ----
    per_seed, fold_info, curves, dailies, shap_vals, calib = [], [], {}, {}, None, None
    base_curves = {}
    for pol in ("fixed", "atr", "cost_rule"):
        fr = risk_fractions(test, pol, s, tp_r)
        c, pnl, lots = simulate_equity(test, fr, s, costs)
        base_curves[pol] = (c, perf_metrics(c, pnl, lots, test, ev_start, ev_end, init, fr))
    for k, seed in enumerate(seeds):
        set_seeds(seed)
        t0 = time.time()
        oos, info, sv = walk_forward(ds, names, cfg, seed, folds, emb, want_shap=(k == 0))
        fold_info += info
        m = test.merge(oos[["row", "p", "threshold", "fold", "base_rate"]], on="row", how="left")
        assert m["p"].notna().all()
        row = {"seed": seed, **prediction_metrics(m["y"].to_numpy(), m["p"].to_numpy(), m["base_rate"].to_numpy())}
        for pol in POLICIES:
            if pol in base_curves:
                c, pm = base_curves[pol]
            else:
                fr = risk_fractions(m, pol, s, tp_r)
                c, pnl, lots = simulate_equity(m, fr, s, costs)
                pm = perf_metrics(c, pnl, lots, m, ev_start, ev_end, init, fr)
            row.update({f"{pol}.{kk}": vv for kk, vv in pm.items()})
            if k == 0:
                curves[pol] = c
            dailies.setdefault(pol, []).append(daily_equity(c, ev_start, ev_end, init))
        per_seed.append(row)
        if k == 0:
            shap_vals = sv
            calib = calibration_table(m["y"].to_numpy(), m["p"].to_numpy())
            m[["decision_time", "exit_time", "side", "r_net", "y", "p", "threshold", "fold"]].to_parquet(
                out_dir / "oos_predictions_seed0.parquet", index=False)
        print(f"seed {seed}: AUC={row['auc']:.4f} brier_skill={row['brier_skill']:+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    ps = pd.DataFrame(per_seed)
    fi = pd.DataFrame(fold_info)

    # ---- bootstrap Sharpe differences vs fixed (per seed) ----
    boot = []
    pairs = [("atr", "fixed"), ("cost_rule", "fixed"), ("meta_filter", "fixed"), ("kelly", "fixed"),
             ("meta_filter", "cost_rule"), ("kelly", "cost_rule")]
    for pol, ref in pairs:
        for k, seed in enumerate(seeds):
            r = dailies[pol][k].pct_change().fillna(0.0)
            rb = dailies[ref][k].pct_change().fillna(0.0)
            b = block_bootstrap_sharpe_diff(r, rb, cfg["report"]["bootstrap_block_days"],
                                            cfg["report"]["bootstrap_samples"], seed=seed)
            boot.append({"policy": pol, "vs": ref, "seed": seed, **b})
            if pol in ("atr", "cost_rule"):
                break  # non-ML policies do not depend on the seed
    bt = pd.DataFrame(boot)

    # ---- SHAP ----
    shap_imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=names).sort_values(ascending=False)

    # ---- outputs ----
    summary = {"primary": primary_stats, "n_folds": len(folds), "seeds": seeds,
               "per_seed": per_seed,
               "mean": ps.drop(columns="seed").mean().to_dict(),
               "std": ps.drop(columns="seed").std(ddof=1).to_dict() if len(ps) > 1 else {},
               "primary_by_year": yr.reset_index().to_dict(orient="records"),
               "folds": fold_info, "bootstrap_sharpe_vs_fixed": boot,
               "calibration_seed0": calib.to_dict(orient="records"),
               "shap_mean_abs_seed0": shap_imp.to_dict(),
               "runtime_seconds": round(time.time() - t_start, 1)}
    write_json(summary, rep / "metrics.json")
    write_json(run_info(cfg, files), rep / "run_info.json")
    fi.to_csv(rep / "folds.csv", index=False)
    _plots(rep, curves, dailies, ev_start, ev_end, init, calib, shap_imp, fi)
    _markdown(rep, cfg, summary, ps, bt, yr, shap_imp, fi)
    print(f"report -> {rep}  ({time.time() - t_start:.0f}s)")
    return 0


def _plots(rep, curves, dailies, start, end, init, calib, shap_imp, fi):
    colors = {"fixed": "#4c72b0", "atr": "#55a868", "cost_rule": "#dd8452", "meta_filter": "#c44e52", "kelly": "#8172b3"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for pol in POLICIES:
        d = pd.concat(dailies[pol], axis=1)
        mid = d.median(axis=1)
        ax1.plot(mid.index, mid.values, color=colors[pol], lw=1.4, label=POLICY_LABEL[pol])
        if d.shape[1] > 1 and pol in ("meta_filter", "kelly"):
            ax1.fill_between(d.index, d.min(axis=1), d.max(axis=1), color=colors[pol], alpha=0.15, lw=0)
        dd = mid / mid.cummax() - 1
        ax2.plot(dd.index, 100 * dd.values, color=colors[pol], lw=1.0)
    ax1.axhline(init, color="grey", lw=0.6, ls=":")
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (USD, log)")
    ax1.set_title("Out-of-sample walk-forward equity (median across seeds; band = seed range)")
    ax1.legend(frameon=False, fontsize=9)
    ax2.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    fig.savefig(rep / "equity_drawdown.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8)
    ax.plot(calib["p_mean"], calib["y_rate"], "o-", color="#c44e52")
    lo, hi = 0.0, max(calib["p_mean"].max(), calib["y_rate"].max()) * 1.15
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted P(win) (decile mean)")
    ax.set_ylabel("Realised win rate")
    ax.set_title("Calibration, out-of-sample (seed 0)")
    fig.tight_layout()
    fig.savefig(rep / "calibration.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    si = shap_imp.iloc[::-1]
    ax.barh(si.index, si.values, color="#4c72b0")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("Feature attribution, out-of-sample (seed 0)")
    fig.tight_layout()
    fig.savefig(rep / "shap_importance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for seed, g in fi.groupby("seed"):
        ax.plot(pd.to_datetime(g["test_start"]), g["auc"], marker="o", lw=1, ms=3, label=f"seed {seed}")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Per-fold out-of-sample AUC")
    ax.legend(frameon=False, fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(rep / "fold_auc.png", dpi=130)
    plt.close(fig)


def _markdown(rep, cfg, S, ps, bt, yr, shap_imp, fi):
    mean, std = S["mean"], S.get("std", {})
    pm = lambda k, f=fmt: f(mean[k]) + (f" ± {f(std[k])}" if k in std and np.isfinite(std.get(k, np.nan)) else "")  # noqa: E731
    pp = lambda k: pm(k, fmt_pct)  # noqa: E731
    P = S["primary"]
    L = [f"# Results — `{cfg['run_name']}`", "",
         f"Data {P['data_start'][:10]} → {P['data_end'][:10]}; out-of-sample {P['oos_start']} → {P['oos_end']}, "
         f"{S['n_folds']} walk-forward folds ({cfg['validation']['scheme']}, test {cfg['validation']['test_months']} mo), "
         f"seeds {S['seeds']}. Values are mean ± std across seeds.", "",
         "## Primary (toy) trade stream", "",
         f"- trades: {P['all_trades']:,} total, {P['oos_trades']:,} out-of-sample",
         f"- OOS win rate {fmt_pct(P['oos_win_rate'])}; avg R gross {fmt(P['oos_avg_r_gross'], 3)}, "
         f"cost {fmt(P['oos_avg_cost_r'], 3)} R, net {fmt(P['oos_avg_r_net'], 3)} R",
         f"- time-exit rate {fmt_pct(P['timeout_rate'])}; median hold {P['median_hold_minutes']:.0f} min", "",
         "## Prediction quality (pooled OOS)", "",
         "| metric | meta-model | base-rate predictor |", "|---|---|---|",
         f"| ROC-AUC | {pm('auc', lambda x: fmt(x, 4))} | 0.5 (cost-rule score: {fmt(P['cost_score_auc'], 4)}; "
         f"logistic regression: {fmt(P['logreg_auc'], 4)}) |",
         f"| Brier | {pm('brier', lambda x: fmt(x, 4))} | {pm('brier_base_rate', lambda x: fmt(x, 4))} |",
         f"| Log-loss | {pm('logloss', lambda x: fmt(x, 4))} | {pm('logloss_base_rate', lambda x: fmt(x, 4))} |",
         f"| Brier skill score | {pm('brier_skill', lambda x: fmt(x, 4))} | 0 |", "",
         "## Trading impact (same trades, Fusion Zero costs)", "",
         "| policy | trades | avg risk | win rate | avg R net | risk-weighted R | total return | CAGR | max DD | Sharpe | Sortino | Calmar |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pol in POLICIES:
        L.append(f"| {POLICY_LABEL[pol]} | {pm(pol + '.trades', lambda x: f'{x:,.0f}')} | {pp(pol + '.avg_risk_frac')} | "
                 f"{pp(pol + '.win_rate')} | {pm(pol + '.avg_r_net', lambda x: fmt(x, 3))} | "
                 f"{pm(pol + '.risk_weighted_r', lambda x: fmt(x, 3))} | {pp(pol + '.total_return')} | {pp(pol + '.cagr')} | "
                 f"{pp(pol + '.max_drawdown')} | {pm(pol + '.sharpe')} | {pm(pol + '.sortino')} | {pm(pol + '.calmar')} |")
    L += ["", "## Sharpe differences (moving-block bootstrap of daily returns)", "",
          "| policy | vs | seed | ΔSharpe | 95% CI | share of resamples ≤ 0 |", "|---|---|---|---|---|---|"]
    for _, r in bt.iterrows():
        L.append(f"| {POLICY_LABEL[r['policy']]} | {POLICY_LABEL[r['vs']]} | {r['seed']} | {r['observed']:+.2f} | "
                 f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | {r['p_le_0']:.2f} |")
    seed_cols = ["seed", "auc", "brier_skill"] + [f"{p}.{m}" for p in ("meta_filter", "kelly")
                                                  for m in ("trades", "total_return", "max_drawdown", "sharpe")]
    L += ["", "## Per-seed", "", _md_table(ps[seed_cols]), "",
          "## Primary stream by year (OOS)", "", _md_table(yr.reset_index()), "",
          "## Per-fold (seed 0)", "", _md_table(fi[fi["seed"] == fi["seed"].min()].drop(columns="seed")), "",
          "## SHAP mean |value| (seed 0, OOS)", "", _md_table(shap_imp.rename("mean_abs_shap").reset_index()
                                                                .rename(columns={"index": "feature"})), ""]
    (rep / "metrics.md").write_text("\n".join(L))


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
