"""Inference API — same contract as the bot's Go `ml_inference` filter.

Bot contract (internal/ml/inference.go + internal/portfolio/allocator.go):
  * artifacts: `<name>.txt` (LightGBM v3 text model, loadable by dmitryikh/leaves)
               `<name>_features.json` (JSON list of feature names, sorted)
  * input:     a mapping feature_name -> float (TradeSignal.Features)
  * vector:    values in schema order; a key missing from the mapping becomes 0.0
  * output:    P(win) in [0, 1] (sigmoid-transformed raw score), written to
               TradeSignal.MLProbability and consumed by the Kelly allocator.

`MetaModel.predict` reproduces that exactly (including the silent 0.0 default).
Pass `strict=True` to raise on missing keys instead — recommended, because the
silent default can hide a train/serve feature-name mismatch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import compute_market_features


class MetaModel:
    def __init__(self, booster: lgb.Booster, schema: list[str], strict: bool = False):
        self.booster = booster
        self.schema = list(schema)
        self.strict = strict

    @classmethod
    def load(cls, model_path: str | Path, schema_path: str | Path | None = None,
             strict: bool = False) -> "MetaModel":
        model_path = Path(model_path)
        if schema_path is None:
            schema_path = model_path.with_name(model_path.stem + "_features.json")
        schema = json.loads(Path(schema_path).read_text())
        return cls(lgb.Booster(model_file=str(model_path)), schema, strict)

    def save(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(model_path))
        model_path.with_name(model_path.stem + "_features.json").write_text(json.dumps(self.schema))

    def vector(self, features: Mapping[str, float]) -> np.ndarray:
        missing = [k for k in self.schema if k not in features]
        if missing and self.strict:
            raise KeyError(f"missing features: {missing}")
        return np.array([float(features.get(k, 0.0)) for k in self.schema], dtype=np.float64)

    def predict(self, features: Mapping[str, float]) -> float:
        """P(win) for one trade signal. Same semantics as Go `PredictSingle`."""
        return float(self.booster.predict(self.vector(features)[None, :])[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        X = np.column_stack([df[k].to_numpy(np.float64) if k in df else np.zeros(len(df))
                             for k in self.schema])
        return self.booster.predict(X)


def kelly_risk_fraction(p: float, b: float, fraction: float = 0.25, max_risk: float = 0.02) -> float:
    """The bot allocator's sizing rule: fraction * (b p - q) / b, 0 if no edge, capped."""
    k = (b * p - (1.0 - p)) / b
    return 0.0 if k <= 0 else min(fraction * k, max_risk)


def features_for_signal(bars: pd.DataFrame, side: int, stop: float, cfg_features: dict) -> dict:
    """Feature dict for a signal on the LAST bar of `bars` (5m mid OHLC + spread_close + close_time).

    Convenience for live use: computes the same features as training, for the
    most recent closed bar only.
    """
    f = compute_market_features(bars, cfg_features).iloc[-1]
    out = {k: float(v) for k, v in f.items() if not k.startswith("_")}
    if cfg_features.get("include_trade_context", True):
        out["side"] = float(side)
        out["stop_atr"] = abs(float(bars["close"].iloc[-1]) - stop) / float(f["_atr"])
    return out
