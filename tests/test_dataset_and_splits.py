import numpy as np
import pandas as pd
import pytest

from ml_risk.config import embargo_minutes
from ml_risk.features import feature_names
from ml_risk.labels import build_dataset
from ml_risk.splits import inner_folds, outer_folds


@pytest.fixture(scope="module")
def ds(cfg):
    d, _, _ = build_dataset(cfg)
    return d


def test_dataset_shapes_and_types(ds, cfg):
    names = feature_names(cfg["features"])
    assert names == sorted(names)
    assert len(ds) > 300
    assert set(names) <= set(ds.columns)
    assert not ds[names].isna().any().any()
    assert np.isfinite(ds[names].to_numpy()).all()
    assert set(ds["y"].unique()) <= {0, 1}
    assert ds["exit_reason"].isin(["sl", "tp", "timeout"]).all()


def test_trade_timing(ds):
    # features at decision_time (signal-bar close); entry strictly later or equal; exit after entry
    assert (ds["decision_time"] == ds["signal_time"] + pd.Timedelta(minutes=5)).all()
    assert (ds["entry_time"] >= ds["decision_time"]).all()
    assert (ds["exit_time"] > ds["entry_time"]).all()


def test_label_matches_net_r(ds):
    assert ((ds["r_net"] > 0).astype(int) == ds["y"]).all()
    tp = ds[ds["exit_reason"] == "tp"]
    assert np.allclose(tp["r_gross"], 2.5)
    sl = ds[ds["exit_reason"] == "sl"]
    assert (sl["r_gross"] <= -1 + 1e-9).all()


def test_outer_folds_are_purged_and_embargoed(ds, cfg):
    emb = embargo_minutes(cfg)
    folds = outer_folds(ds, cfg["validation"], emb)
    assert len(folds) >= 2
    prev_end = None
    for f in folds:
        tr, te = ds.iloc[f.train_idx], ds.iloc[f.test_idx]
        assert (tr["exit_time"] < f.test_start).all(), "label of a training sample leaks into test window"
        assert (tr["decision_time"] < f.test_start - pd.Timedelta(minutes=emb)).all(), "embargo violated"
        assert (te["decision_time"] >= f.test_start).all() and (te["decision_time"] < f.test_end).all()
        assert len(np.intersect1d(f.train_idx, f.test_idx)) == 0
        if prev_end is not None:
            assert f.test_start >= prev_end
        prev_end = f.test_end


def test_inner_folds_are_purged(ds, cfg):
    emb = embargo_minutes(cfg)
    f = outer_folds(ds, cfg["validation"], emb)[-1]
    inner = inner_folds(ds, f.train_idx, 2, 0.15, emb)
    assert inner
    for k in inner:
        val_start = ds["decision_time"].iloc[k["val"]].min()
        es_start = ds["decision_time"].iloc[k["es"]].min()
        fit = ds.iloc[k["fit"]]
        assert (fit["exit_time"] < es_start).all()
        assert (ds["exit_time"].iloc[k["es"]] < val_start).all()
        assert set(k["val"]) <= set(f.train_idx)
