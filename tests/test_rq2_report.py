"""Chapter IV tables/figures on a tiny real pipeline (dev -> select -> test), so every column exists as in real data."""

import json

import matplotlib
matplotlib.use("Agg")
import pytest

from eem_acc import rq2_report as R
from eem_acc import scenarios as S
from eem_acc.analysis import load_results


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    import importlib.util
    from pathlib import Path
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("rq2rep")
    spec = importlib.util.spec_from_file_location("run_rq2_rep", Path(__file__).resolve().parents[1] / "experiments" / "run_rq2.py")
    run_rq2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_rq2)
    from conftest import tiny_v2
    tiny_v2(mp, tmp, run_rq2, scenarios=("CAP-steady", "CAP-drop"), horizon=60.0)
    mp.setattr(S, "M_LEVELS", (8, 32))
    mp.setattr(S, "SCENARIOS", {**S.SCENARIOS, "CAP-drop": {**S.SCENARIOS["CAP-drop"],
               "capacity_events": ((20.0, 1 / 3), (40.0, 1.0)),
               "analysis": S.AnalysisConfig(windows=(("pre", 5.0, 20.0), ("during", 20.0, 40.0), ("post", 40.0, 55.0)),
                                            event=(20.0, 40.0), pre=(5.0, 20.0))}})
    mp.setattr(S, "FIXED_W_DEV", (2, 4, 8, 12))
    # the report code is under test here, not the selection: a hand-written frozen file (structure as select writes it)
    loaded = S.loaded_cells()
    frozen = {"acc": dict(alpha=0.25, beta=0.25, epsilon=0.05, n_c=8, kappa=0.8),
              "acc_kappa1": {"config": dict(alpha=0.75, beta=0.5, epsilon=0.05, n_c=8, kappa=1.0), "feasible": False,
                             "score": 0.9, "n_feasible_cells": 1, "n_loaded_cells": len(loaded)},
              "best_fixed_w": 4, "best_pacer_rate": 14.0,
              "baseline_fallback": {"fixed": {"value": 4, "n_feasible_cells": 2, "n_loaded_cells": len(loaded)}},
              "secondary_global_baselines": {"fixed": {"value": 8, "n_feasible_scenarios": 1, "n_scenarios": 2}},
              "best_fixed_w_per_scenario": {"CAP-steady": 4, "CAP-drop": 12},
              "criterion": {"loaded_cells": [f"{s}|{m}" for s, m in loaded]},
              "denominators": {f"{s}|{m}": 20.0 for s, m in loaded},
              "loso": {"CAP-drop": {"acc": {"tag": "x", "feasible_on_training": True, "feasible_on_left_out": True,
                                            "score_on_left_out": 0.9, "mean_slo_violation": 0.01, "mean_throttle_rate": 0.0}}}}
    assert loaded
    run_rq2.run_phase("test", tmp, frozen=frozen)
    yield tmp, frozen, load_results(tmp / "results" / "rq2" / "test" / "units"), run_rq2
    mp.undo()


def test_key_methods_are_unique_and_start_with_acc(tiny_run):
    _, frozen, _, _ = tiny_run
    km = R.key_methods(frozen)
    assert list(km)[0] == "ACC (frozen)" and len(set(km.values())) == len(km)


def test_cell_table_columns_and_feasibility(tiny_run):
    _, frozen, test, _ = tiny_run
    ct = R.cell_table(test, frozen)
    assert {"e2e_p95_censored", "queue_mean", "J_05", "J_10", "goodput_ci95", "feasible"} <= set(ct.columns)
    assert (ct["n_seeds"] == len(S.TEST_SEEDS)).all()
    row = ct.iloc[0]
    ok = (row.slo_violation_rate <= 0.05) and (row.call_success_rate >= 0.99) and (row.throttle_rate <= 0.10)
    assert bool(row.feasible) == ok
    # J consistent with its components (lambda = gamma = 0.5)
    assert row.J_05 <= row.call_success_rate + 1e-12


def test_event_and_recovery_tables(tiny_run):
    _, frozen, test, _ = tiny_run
    et = R.event_table(test, frozen)
    assert set(et["window"]) == {"pre", "during", "post"} and set(et["scenario"]) == {"CAP-drop"}
    rt = R.recovery_table(test, frozen)
    assert (rt[["recovered", "no_degradation", "censored", "no_reference"]].sum(axis=1) == rt["n_seeds"]).all()


def test_markdown_renders_bools_and_nan():
    import pandas as pd
    md = R.to_markdown(pd.DataFrame({"a": [True, False], "b": [1.23456, float("nan")]}))
    assert "| yes | 1.235 |" in md and "| no |  |" in md


def test_analyze_script_writes_everything(tiny_run, monkeypatch):
    tmp, frozen, _, _ = tiny_run
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("analyze_rq2", Path(__file__).resolve().parents[1] / "experiments" / "analyze_rq2.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    w = A.analyze(tmp, frozen, figures=True)
    for k in ("primary_comparisons", "cell_table", "event_windows", "recovery", "test_descriptive", "robustness_matrix",
              "loso_dev", "markdown"):
        assert w[k].exists() and w[k].stat().st_size > 50
    figs = sorted(p.name for p in (tmp / "analysis" / "figures").iterdir())
    assert figs == sorted(f"fig_rq2_{n}.{e}" for n in ("timeseries", "frontier", "robustness") for e in ("pdf", "png"))
