"""Time-ordered walk-forward splits with purging and embargo.

Each sample i spans [t0_i, t1_i] = [decision_time, exit_time]. For a validation /
test window starting at `a`, a training sample is kept only if

    t1_i < a                 (purge: its label is fully resolved before the window)
    t0_i < a - embargo       (embargo: drop samples whose features sit right next to
                              the window, guarding against serially correlated features)

Training data never comes from after the test window. The embargo defaults to the
maximum possible trade duration (max_hold_minutes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_before(t0: pd.Series, t1: pd.Series, idx: np.ndarray, boundary: pd.Timestamp,
                  embargo: pd.Timedelta) -> np.ndarray:
    keep = (t1.iloc[idx] < boundary).to_numpy() & (t0.iloc[idx] < boundary - embargo).to_numpy()
    return idx[keep]


def outer_folds(ds: pd.DataFrame, v: dict, embargo_min: int) -> list[Fold]:
    t0, t1 = ds["decision_time"], ds["exit_time"]
    emb = pd.Timedelta(minutes=embargo_min)
    first = t0.min()
    start = (first.normalize() - pd.offsets.MonthBegin(1)) if first.day != 1 else first.normalize()
    test_start = start + pd.DateOffset(months=v["min_train_months"])
    if test_start <= first:
        test_start = first.normalize() + pd.offsets.MonthBegin(1) + pd.DateOffset(months=v["min_train_months"])
    last = t0.max()
    all_idx = np.arange(len(ds))
    folds, k = [], 0
    while test_start <= last:
        test_end = test_start + pd.DateOffset(months=v["test_months"])
        test_idx = all_idx[((t0 >= test_start) & (t0 < test_end)).to_numpy()]
        cand = all_idx[(t0 < test_start).to_numpy()]
        if v.get("scheme", "expanding") == "rolling":
            lo = test_start - pd.DateOffset(months=v["rolling_train_months"])
            cand = cand[(t0.iloc[cand] >= lo).to_numpy()]
        train_idx = purged_before(t0, t1, cand, test_start, emb)
        if len(test_idx) and len(train_idx):
            folds.append(Fold(k, test_start, test_end, train_idx, test_idx))
            k += 1
        test_start = test_end
    return folds


def inner_folds(ds: pd.DataFrame, train_idx: np.ndarray, n_inner: int, es_frac: float,
                embargo_min: int) -> list[dict]:
    """Inner walk-forward inside a training window.

    Returns a list of {fit, es, val} index arrays. `val` is used only for model
    selection and threshold choice; `es` (the purged tail of the inner-train slice)
    only for early stopping, so selection is not biased by early stopping.
    """
    t0, t1 = ds["decision_time"], ds["exit_time"]
    emb = pd.Timedelta(minutes=embargo_min)
    idx = train_idx[np.argsort(t0.iloc[train_idx].to_numpy(), kind="stable")]
    blocks = np.array_split(idx, n_inner + 1)
    out = []
    for k in range(n_inner):
        val = blocks[k + 1]
        if len(val) == 0:
            continue
        val_start = t0.iloc[val].min()
        inner = purged_before(t0, t1, np.concatenate(blocks[: k + 1]), val_start, emb)
        if len(inner) < 20:
            continue
        es_start = t0.iloc[inner[int(len(inner) * (1 - es_frac))]]
        es = inner[(t0.iloc[inner] >= es_start).to_numpy()]
        fit = purged_before(t0, t1, inner, es_start, emb)
        if len(fit) < 20 or len(es) < 5:
            continue
        out.append({"fit": fit, "es": es, "val": val})
    return out
