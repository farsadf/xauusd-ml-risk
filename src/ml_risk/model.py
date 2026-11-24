"""LightGBM meta-model: nested (in-train-only) hyper-parameter, round-count and
threshold selection, then a refit on the full training window.

Selection protocol for one training window W (all inside W, never touching the test):
  1. Inner walk-forward folds over W (see splits.inner_folds), purged + embargoed.
  2. For each candidate (max_depth, min_data_in_leaf): fit on `fit`, early-stop on
     `es`, score log-loss on `val`.
  3. Pick the candidate with the lowest mean val log-loss; n_rounds = median of its
     early-stopped best iterations.
  4. Threshold for the filter policy: from that candidate's pooled inner-val
     predictions, the threshold maximising total net R of kept trades, subject to
     keeping >= min_keep_frac of trades.
  5. Refit on all of W with the chosen params and n_rounds.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .splits import inner_folds


def make_params(base: dict, depth: int, min_leaf: int, seed: int, from_depth: bool = True) -> dict:
    p = dict(base)
    p.update(max_depth=int(depth), min_data_in_leaf=int(min_leaf), seed=int(seed),
             bagging_seed=int(seed), feature_fraction_seed=int(seed), data_random_seed=int(seed))
    if from_depth:
        p["num_leaves"] = max(2, 2 ** int(depth) - 1)
    return p


def fit_booster(X: np.ndarray, y: np.ndarray, params: dict, rounds: int,
                X_es: np.ndarray | None = None, y_es: np.ndarray | None = None,
                es_rounds: int | None = None) -> lgb.Booster:
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    if X_es is None:
        return lgb.train(params, dtrain, num_boost_round=rounds)
    dval = lgb.Dataset(X_es, label=y_es, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(es_rounds, verbose=False)])


def choose_threshold(p: np.ndarray, r: np.ndarray, t: dict) -> float:
    grid = np.round(np.arange(t["grid_min"], t["grid_max"] + 1e-9, t["grid_step"]), 4)
    best_thr, best_val = float(grid[0]), -np.inf
    n = len(p)
    for thr in grid:
        keep = p >= thr
        if keep.sum() < max(1, t["min_keep_frac"] * n):
            continue
        val = r[keep].sum()
        if val > best_val:
            best_thr, best_val = float(thr), val
    return best_thr


def choose_score_threshold(score: np.ndarray, r: np.ndarray, t: dict) -> float:
    """Threshold on an arbitrary score (non-ML baseline): candidates are quantiles of the
    training-window score; same objective and min-keep constraint as choose_threshold."""
    qs = np.arange(0.0, 1.0 - t["min_keep_frac"] + 1e-9, 0.02)
    best_thr, best_val = -np.inf, -np.inf
    for thr in np.unique(np.quantile(score, qs)):
        keep = score >= thr
        val = r[keep].sum()
        if val > best_val:
            best_thr, best_val = float(thr), val
    return best_thr


@dataclass
class FoldModel:
    booster: lgb.Booster
    params: dict
    n_rounds: int
    threshold: float
    selection: list = field(default_factory=list)


def select_and_fit(ds: pd.DataFrame, train_idx: np.ndarray, names: list[str], cfg: dict,
                   seed: int, embargo_min: int) -> FoldModel:
    m, v = cfg["model"], cfg["validation"]
    X = ds[names].to_numpy(np.float64)
    y = ds["y"].to_numpy()
    r = ds["r_net"].to_numpy()
    inner = inner_folds(ds, train_idx, v["inner_folds"], v["early_stop_frac"], embargo_min)
    if not inner:
        raise RuntimeError("training window too small for inner folds")

    cands = list(itertools.product(m["search"]["max_depth"], m["search"]["min_data_in_leaf"]))
    results = []
    for depth, leaf in cands:
        params = make_params(m["base_params"], depth, leaf, seed, m.get("num_leaves_from_depth", True))
        losses, iters, preds, rets = [], [], [], []
        for f in inner:
            b = fit_booster(X[f["fit"]], y[f["fit"]], params, m["max_rounds"],
                            X[f["es"]], y[f["es"]], m["early_stopping_rounds"])
            pv = b.predict(X[f["val"]], num_iteration=b.best_iteration or None)
            losses.append(log_loss(y[f["val"]], np.clip(pv, 1e-6, 1 - 1e-6), labels=[0, 1]))
            iters.append(b.best_iteration or m["max_rounds"])
            preds.append(pv)
            rets.append(r[f["val"]])
        results.append({"max_depth": depth, "min_data_in_leaf": leaf,
                        "val_logloss": float(np.mean(losses)), "n_rounds": int(np.median(iters)),
                        "_p": np.concatenate(preds), "_r": np.concatenate(rets)})
    best = min(results, key=lambda d: d["val_logloss"])
    threshold = choose_threshold(best["_p"], best["_r"], cfg["sizing"]["threshold"])
    params = make_params(m["base_params"], best["max_depth"], best["min_data_in_leaf"], seed,
                         m.get("num_leaves_from_depth", True))
    n_rounds = max(1, best["n_rounds"])
    booster = fit_booster(X[train_idx], y[train_idx], params, n_rounds)
    sel = [{k: v for k, v in d.items() if not k.startswith("_")} for d in results]
    return FoldModel(booster, params, n_rounds, threshold, sel)
