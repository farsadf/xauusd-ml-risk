"""Inference contract (identical to the bot's Go filter) + end-to-end smoke test on the sample."""
import json

import numpy as np
import pytest

from ml_risk import evaluate, train
from ml_risk.inference import MetaModel, features_for_signal, kelly_risk_fraction


@pytest.fixture(scope="module")
def smoke(cfg, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    c = dict(cfg)
    c["output_dir"] = str(tmp / "out")
    c["validation"] = {**cfg["validation"], "seeds": [0]}
    c["report"] = {**cfg["report"], "bootstrap_samples": 50}
    cfg_path = tmp / "cfg.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in c.items() if not k.startswith("_")}))
    art = tmp / "artifacts"
    assert train.main(["--config", str(cfg_path), "--out", str(art), "--name", "m"]) == 0
    rep = tmp / "report"
    assert evaluate.main(["--config", str(cfg_path), "--report-dir", str(rep)]) == 0
    return {"art": art, "rep": rep, "cfg": c}


def test_train_exports_bot_artifacts(smoke):
    art = smoke["art"]
    assert (art / "m.txt").exists() and (art / "m_features.json").exists() and (art / "m_meta.json").exists()
    schema = json.loads((art / "m_features.json").read_text())
    assert schema == sorted(schema)
    head = (art / "m.txt").read_text().splitlines()[0]
    assert head == "tree"  # LightGBM text model, v3 format readable by dmitryikh/leaves


def test_evaluate_writes_report(smoke):
    rep = smoke["rep"]
    for f in ("metrics.json", "metrics.md", "run_info.json", "folds.csv", "equity_drawdown.png",
              "calibration.png", "shap_importance.png", "fold_auc.png"):
        assert (rep / f).exists(), f
    m = json.loads((rep / "metrics.json").read_text())
    assert 0.0 <= m["mean"]["auc"] <= 1.0
    assert m["run_info"] if "run_info" in m else True


def test_predict_semantics_match_go(smoke, bars5, cfg):
    mm = MetaModel.load(smoke["art"] / "m.txt")
    feats = features_for_signal(bars5.iloc[:2000], side=1, stop=float(bars5["low"].iloc[1990:2000].min()),
                                cfg_features=cfg["features"])
    p = mm.predict(feats)
    assert 0.0 < p < 1.0
    # identical to a batch prediction on the schema-ordered vector
    vec = np.array([[feats[k] for k in mm.schema]])
    assert np.isclose(p, mm.booster.predict(vec)[0])
    # missing key -> 0.0 (Go behaviour), extra keys ignored
    drop = dict(feats)
    drop.pop("adx")
    drop["unused_extra"] = 123.0
    vec0 = vec.copy()
    vec0[0, mm.schema.index("adx")] = 0.0
    assert np.isclose(mm.predict(drop), mm.booster.predict(vec0)[0])
    # strict mode surfaces the mismatch instead
    with pytest.raises(KeyError):
        MetaModel(mm.booster, mm.schema, strict=True).predict(drop)


def test_kelly_risk_fraction_matches_allocator():
    assert kelly_risk_fraction(0.25, 2.5) == 0.0
    assert np.isclose(kelly_risk_fraction(0.32, 2.5), 0.25 * (2.5 * 0.32 - 0.68) / 2.5)
    assert kelly_risk_fraction(0.9, 2.5) == 0.02


def test_seeded_training_is_reproducible(smoke):
    c = smoke["cfg"]
    from ml_risk.config import embargo_minutes
    from ml_risk.features import feature_names
    from ml_risk.model import select_and_fit
    from ml_risk.pipeline import dataset_cached
    ds, _ = dataset_cached(c)
    names = feature_names(c["features"])
    idx = np.arange(len(ds))
    a = select_and_fit(ds, idx, names, c, 0, embargo_minutes(c))
    b = select_and_fit(ds, idx, names, c, 0, embargo_minutes(c))
    X = ds[names].to_numpy()
    assert np.array_equal(a.booster.predict(X), b.booster.predict(X))


def test_go_leaves_parity(smoke, bars5, cfg, tmp_path):
    """The bot's Go loader (dmitryikh/leaves) must produce the same P(win) as Python."""
    import shutil
    import subprocess
    from pathlib import Path
    if shutil.which("go") is None:
        pytest.skip("Go toolchain not installed")
    tool = Path(__file__).resolve().parents[1] / "tools" / "go_parity"
    binary = tmp_path / "go_parity"
    build = subprocess.run(["go", "build", "-o", str(binary), "."], cwd=tool, capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"go build failed (offline?): {build.stderr[-300:]}")
    mm = MetaModel.load(smoke["art"] / "m.txt")
    rows = []
    for end in range(400, 4000, 90):
        b = bars5.iloc[:end]
        rows.append(features_for_signal(b, side=1 if end % 2 else -1,
                                        stop=float(b["low"].iloc[-10:].min()), cfg_features=cfg["features"]))
    rows.append({})  # all missing -> zeros
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    out = subprocess.run([str(binary), "-model", str(smoke["art"] / "m.txt"),
                          "-features", str(smoke["art"] / "m_features.json"),
                          "-input", str(tmp_path / "rows.json")], capture_output=True, text=True, check=True)
    go = np.array([float(x) for x in out.stdout.split()])
    py = np.array([mm.predict(r) for r in rows])
    assert go.shape == py.shape
    assert np.max(np.abs(go - py)) < 1e-9
