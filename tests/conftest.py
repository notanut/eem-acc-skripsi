"""Shared test helpers."""

import importlib.util
from pathlib import Path

from eem_acc import scenarios as S

_spec = importlib.util.spec_from_file_location("run_rq2_for_tests",
                                               Path(__file__).resolve().parents[1] / "experiments" / "run_rq2.py")


def tiny_v2(monkeypatch, tmp_path, run_rq2, scenarios=("CAP-steady",), horizon=20.0):
    """A tiny but complete v2 design (seconds to run): 1 loaded cell (CAP-steady, M=8), 2 ACC configs of which one is
    feasible, 2 fixed windows (one feasible), 1 token-bucket rate, dev seeds 0-1, test seeds 100-101."""
    monkeypatch.setattr(S, "SCENARIOS", {k: S.SCENARIOS[k] for k in scenarios})
    monkeypatch.setattr(S, "M_LEVELS", (8,))
    monkeypatch.setattr(S, "DEV_SEEDS", (0, 1))
    monkeypatch.setattr(S, "TEST_SEEDS", (100, 101))
    monkeypatch.setattr(S, "HORIZON", horizon)
    monkeypatch.setattr(S, "SUCCESS_GRACE", 3.0)
    monkeypatch.setattr(S, "ACC_GRID", dict(alpha=(0.5, 0.75), beta=(0.5,), epsilon=(0.05,), n_c=(8,), kappa=(0.8,)))
    monkeypatch.setattr(S, "FIXED_W_DEV", (2, 8))
    monkeypatch.setattr(S, "PACER_RATES_DEV", (14.0,))
    monkeypatch.setattr(run_rq2, "FROZEN_PATH", tmp_path / "frozen.json")
