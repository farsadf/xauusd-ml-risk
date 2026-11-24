"""Config loading, seeding and run provenance."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def embargo_minutes(cfg: dict) -> int:
    e = cfg["validation"].get("embargo_minutes")
    return int(cfg["primary"]["max_hold_minutes"] if e is None else e)


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (out.stdout.strip() + ("-dirty" if dirty else "")) if out.returncode == 0 else None
    except Exception:
        return None


def file_digest(paths: list[str]) -> str:
    """sha256 over the (name, size, first/last 1MB) of each input file — cheap and stable."""
    h = hashlib.sha256()
    for p in sorted(paths):
        st = os.stat(p)
        h.update(os.path.basename(p).encode())
        h.update(str(st.st_size).encode())
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if st.st_size > (2 << 20):
                f.seek(-(1 << 20), os.SEEK_END)
                h.update(f.read())
    return h.hexdigest()


def run_info(cfg: dict, input_files: list[str]) -> dict[str, Any]:
    import lightgbm
    import pandas
    import sklearn
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "pandas": pandas.__version__,
                     "lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__},
        "input_files": [os.path.basename(p) for p in input_files],
        "input_digest": file_digest(input_files) if input_files else None,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }


def write_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
