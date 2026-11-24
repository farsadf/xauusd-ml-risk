"""Shared helpers: cached dataset build."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import file_digest
from .data import list_input_files
from .labels import build_dataset

CACHE_VERSION = 2  # bump when dataset construction code changes


def dataset_cached(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build (or load from cache) the per-trade dataset. Cache key = input digest + relevant config."""
    files = list_input_files(cfg["data"]["path"])
    key_src = json.dumps({"v": CACHE_VERSION, "digest": file_digest(files),
                          **{k: cfg[k] for k in ("data", "primary", "costs", "features")},
                          "atr_ref": cfg["sizing"]["atr"]}, sort_keys=True, default=str)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache = Path(cfg["output_dir"]) / "cache" / f"dataset_{key}.parquet"
    if cache.exists():
        return pd.read_parquet(cache), files
    ds, _, files = build_dataset(cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(cache, index=False)
    return ds, files
