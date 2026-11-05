"""Tests for the RQ2 design wiring, runner resume and dev selection.
Synthetic data only: nothing here is a result."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from eem_acc import scenarios as S
from eem_acc.agent import PacerConfig, TokenBucketPacer
from eem_acc.analysis import cell_means, load_results, score_methods, select_frozen
from eem_acc.env import run_virtual
from eem_acc.sim import run_scenario
from eem_acc.utils import ResultStore

_spec = importlib.util.spec_from_file_location(
    "run_rq2", Path(__file__).resolve().parents[1] / "experiments" / "run_rq2.py")
run_rq2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_rq2)


# ---------- pacer ----------
def _times(cfg, n):
    async def main():
        p = TokenBucketPacer(cfg)
        loop = asyncio.get_running_loop()
        out = []
        for _ in range(n):
            await p.acquire()
            out.append(loop.time())
        return out
    return run_virtual(main())


def test_pacer_spaces_calls_at_rate():
    t = _times(PacerConfig(rate=10.0, burst=1), 5)
    assert t == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_pacer_allows_burst_then_paces():
    t = _times(PacerConfig(rate=10.0, burst=3), 6)
    assert t[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert t[3:] == pytest.approx([0.1, 0.2, 0.3])


def test_pacer_fifo_under_concurrency():
    async def main():
        p = TokenBucketPacer(PacerConfig(rate=10.0, burst=1))
        loop = asyncio.get_running_loop()
        order = []

        async def w(i):
            await p.acquire(); order.append((i, round(loop.time(), 6)))

        await asyncio.gather(*[w(i) for i in range(4)])
        return order

    assert run_virtual(main()) == [(0, 0.0), (1, 0.1), (2, 0.2), (3, 0.3)]


def test_tokenbucket_scenario_respects_rate():
    sc = S.build("RATE-steady", 4, S.pacer_method(10.0), seed=0)
    # re-run through the public API and look at attempt dispatch times via metrics: goodput <= rate
    m = run_scenario(sc)["metrics"]
    assert m["goodput"] <= 10.0 + 1e-6
    assert m["n_429"] == 0 and m["n_calls_ok"] == m["n_calls"]


# ---------- design wiring ----------
def test_design_counts_match_approved_proposal():
    assert len(S.acc_grid_methods()) == 81
    dev = S.units(S.dev_methods(), S.DEV_SEEDS)
    assert len(dev) == (81 + 7 + 4) * 4 * 3 * 5 == 5520
    assert len({u[0] for u in dev}) == len(dev)                     # unique unit ids


def test_every_dev_unit_builds():
    for uid, scn, m, meth, seed in S.units(S.dev_methods(), S.DEV_SEEDS)[::37]:
        sc = S.build(scn, m, meth, seed)
        assert sc.acc.l_target == 0.45 and sc.workload.n_agents == m


def test_test_grid_size_and_ablation_flags():
    frozen = {"acc": dict(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16), "best_fixed_w": 8, "best_pacer_rate": 40.0}
    ms = S.test_methods(frozen)
    assert len(ms) == 9                                             # acc + 3 ablations + 3 fixed + best fixed + tb
    assert len(S.units(ms, S.TEST_SEEDS)) == 4 * 3 * 9 * 10 == 1080
    assert S.build("CAP-steady", 4, ms[1], 100).acc.use_throttle_branch is False
    assert S.build("CAP-steady", 4, ms[2], 100).acc.use_latency_branch is False
    assert S.build("CAP-steady", 4, ms[3], 100).acc.multiplicative_decrease is False


def test_best_fixed_w_not_duplicated_when_already_reported():
    frozen = {"acc": dict(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16), "best_fixed_w": 16, "best_pacer_rate": 40.0}
    assert len(S.test_methods(frozen)) == 8


def test_scenarios_have_the_approved_events():
    drop = S.build("CAP-drop", 16, S.fixed_method(8), 0).server
    assert drop.capacity_events == ((20.0, 0.5), (40.0, 1.0))
    burst = S.build("CAP-burst", 16, S.fixed_method(8), 0).workload
    assert burst.phases == ((0.0, 0.1), (10.0, 0.0), (15.0, 0.1))
    assert S.build("RATE-steady", 4, S.fixed_method(8), 0).server.rate_limit == 40.0


# ---------- runner: resume ----------
def test_runner_resumes_without_rerunning(tmp_path, monkeypatch):
    calls = []
    real = run_rq2.run_scenario
    monkeypatch.setattr(run_rq2, "run_scenario", lambda sc: (calls.append(1), real(sc))[1])
    run_rq2.run_phase("dev", tmp_path, limit=6)
    assert len(calls) == 6
    run_rq2.run_phase("dev", tmp_path, limit=10)                    # extends: only 4 new units
    assert len(calls) == 10
    df = load_results(tmp_path / "results" / "rq2" / "dev" / "units")
    assert len(df) == 10 and df["unit_id"].is_unique
    assert (tmp_path / "logs" / "rq2_dev.log").exists()


def test_runner_survives_a_crash_and_continues(tmp_path, monkeypatch):
    real = run_rq2.run_scenario
    n = {"i": 0}

    def flaky(sc):
        n["i"] += 1
        if n["i"] == 4:
            raise RuntimeError("colab disconnected")
        return real(sc)

    monkeypatch.setattr(run_rq2, "run_scenario", flaky)
    with pytest.raises(RuntimeError):
        run_rq2.run_phase("dev", tmp_path, limit=8)
    monkeypatch.setattr(run_rq2, "run_scenario", real)
    run_rq2.run_phase("dev", tmp_path, limit=8)
    assert len(load_results(tmp_path / "results" / "rq2" / "dev" / "units")) == 8


def test_select_refuses_incomplete_dev(tmp_path, capsys):
    run_rq2.run_phase("dev", tmp_path, limit=3)
    assert run_rq2.do_select(tmp_path) == 2
    assert "refusing" in capsys.readouterr().out


# ---------- selection logic (synthetic) ----------
def _rows(tag, kind, per_cell, seeds=(0, 1), **meta):
    """per_cell: {(scenario, M): (goodput, slo_violation, success)}"""
    out = []
    for (scn, m), (g, v, ok) in per_cell.items():
        for s in seeds:
            out.append({"unit_id": f"{scn}{m}{tag}{s}", "scenario": scn, "M": m, "seed": s, "tag": tag,
                        "m_kind": kind, **{f"m_{k}": x for k, x in meta.items()},
                        "goodput": g, "slo_violation_rate": v, "call_success_rate": ok, "throttle_rate": 0.0})
    return out


CELLS = [("A", 4), ("A", 16)]


def _synthetic():
    rows = []
    # accA: best goodput but violates SLO in one cell (mean 0.10 > 0.05) -> infeasible
    rows += _rows("accA", "acc", {("A", 4): (10, 0.0, 1.0), ("A", 16): (40, 0.20, 1.0)}, alpha=.5, beta=.5, epsilon=.1, n_c=16)
    # accB: slightly less goodput, feasible -> expected ACC pick
    rows += _rows("accB", "acc", {("A", 4): (10, 0.0, 1.0), ("A", 16): (36, 0.04, 1.0)}, alpha=.25, beta=.5, epsilon=.1, n_c=8)
    # accC: feasible but low goodput
    rows += _rows("accC", "acc", {("A", 4): (8, 0.0, 1.0), ("A", 16): (20, 0.0, 1.0)}, alpha=.75, beta=.5, epsilon=.1, n_c=32)
    rows += _rows("fixed_w4", "fixed", {("A", 4): (9, 0.0, 1.0), ("A", 16): (25, 0.0, 1.0)}, w=4)
    rows += _rows("fixed_w64", "fixed", {("A", 4): (10, 0.0, 1.0), ("A", 16): (30, 0.5, 0.9)}, w=64)
    rows += _rows("tb_r40", "tokenbucket", {("A", 4): (10, 0.0, 1.0), ("A", 16): (30, 0.0, 1.0)}, rate=40.0)
    return pd.DataFrame(rows)


def test_selection_respects_constraints_and_normalisation():
    res = select_frozen(_synthetic())
    assert res["ok"]
    assert res["acc"] == dict(alpha=0.25, beta=0.5, epsilon=0.1, n_c=8)    # accB: best feasible, accA infeasible
    assert res["best_fixed_w"] == 4                                        # w64 violates SLO and success
    assert res["best_pacer_rate"] == 40.0
    sc = score_methods(_synthetic()).set_index("tag")
    assert sc.loc["accA", "feasible"] == False                             # noqa: E712
    assert sc.loc["accA", "score"] == pytest.approx(1.0)                   # best in both cells
    assert sc.loc["accB", "score"] == pytest.approx((1.0 + 36 / 40) / 2)


def test_selection_reports_infeasible_instead_of_relaxing():
    df = _synthetic()
    df.loc[df["tag"].str.startswith("acc"), "slo_violation_rate"] = 0.9
    res = select_frozen(df)
    assert res["ok"] is False and "acc" not in res                         # nothing silently relaxed
    assert res["top"]["acc"]


def test_incomplete_candidate_is_not_eligible():
    df = _synthetic()
    df = df[~((df["tag"] == "accB") & (df["M"] == 16))]
    sc = score_methods(df).set_index("tag")
    assert sc.loc["accB", "complete"] == False and sc.loc["accB", "feasible"] == False   # noqa: E712


def test_missing_slo_counts_as_worst_case():
    df = _synthetic()
    df.loc[df["tag"] == "accB", "slo_violation_rate"] = None
    assert cell_means(df).query("tag == 'accB'")["slo_violation_rate"].eq(1.0).all()
