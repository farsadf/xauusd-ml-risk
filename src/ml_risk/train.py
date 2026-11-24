"""Train the deployable meta-model on the full configured period.

    python -m ml_risk.train --config configs/default.yaml

Uses the same nested, purged selection as each walk-forward fold (params, rounds
and filter threshold chosen inside the data), then refits on everything and exports
bot-compatible artifacts:  <out>/<name>.txt, <out>/<name>_features.json, <out>/<name>_meta.json.
This model is for deployment/inference; performance claims come only from
`ml_risk.evaluate` (walk-forward, out-of-sample).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import embargo_minutes, load_config, run_info, set_seeds, write_json
from .features import feature_names
from .inference import MetaModel
from .model import select_and_fit
from .pipeline import dataset_cached


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="xauusd_meta")
    ap.add_argument("--out", default=None, help="default: <output_dir>/artifacts/<run_name>")
    ap.add_argument("--seed", type=int, default=None, help="default: first seed in config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg["validation"]["seeds"][0] if args.seed is None else args.seed
    set_seeds(seed)
    ds, files = dataset_cached(cfg)
    names = feature_names(cfg["features"])
    all_idx = np.arange(len(ds))
    fm = select_and_fit(ds, all_idx, names, cfg, seed, embargo_minutes(cfg))

    out = Path(args.out or Path(cfg["output_dir"]) / "artifacts" / cfg["run_name"])
    MetaModel(fm.booster, names).save(out / f"{args.name}.txt")
    meta = {
        "name": args.name, "seed": seed, "n_samples": int(len(ds)),
        "train_period": [str(ds["decision_time"].min()), str(ds["decision_time"].max())],
        "base_rate": float(ds["y"].mean()), "features": names,
        "params": fm.params, "n_rounds": fm.n_rounds, "threshold": fm.threshold,
        "selection": fm.selection, "run_info": run_info(cfg, files),
    }
    write_json(meta, out / f"{args.name}_meta.json")
    print(f"trained on {len(ds):,} trades ({meta['train_period'][0]} -> {meta['train_period'][1]}), "
          f"base rate {meta['base_rate']:.3f}, rounds={fm.n_rounds}, threshold={fm.threshold:.2f}")
    print(f"artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
