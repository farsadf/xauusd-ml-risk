from pathlib import Path

import pytest

from ml_risk.config import load_config
from ml_risk.data import load_bars, resample_bars

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "xauusd_synthetic_1m.csv.gz"


@pytest.fixture(scope="session")
def cfg():
    c = load_config(ROOT / "configs" / "sample.yaml")
    c["data"]["path"] = str(SAMPLE)
    return c


@pytest.fixture(scope="session")
def m1():
    return load_bars(str(SAMPLE), "2024-01-01", "2024-01-31")


@pytest.fixture(scope="session")
def bars5(m1):
    return resample_bars(m1, 5)
