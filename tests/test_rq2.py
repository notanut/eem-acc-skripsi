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
    sc = S.build("RATE-steady", 8, S.pacer_method(10.0), seed=0)
    m = run_scenario(sc)["metrics"]
    assert m["goodput"] <= 10.0 + 1e-6
    assert m["n_429"] == 0 and m["n_calls_ok"] == m["n_calls_finished"]


# ---------- design wiring (v2, docs/rq2_design_v2.md) ----------
def test_design_counts_match_approved_v2():
    assert len(S.acc_grid_methods()) == 81 * 3                      # kappa {0.8, 0.9, 1.0} (D22)
    assert len(S.FIXED_W_DEV) == 12 and {12, 20} <= set(S.FIXED_W_DEV) and len(S.PACER_RATES_DEV) == 15
    dev = S.units(S.dev_methods(), S.DEV_SEEDS)
    assert len(dev) == (243 + 12 + 15) * 4 * 3 * 5 == 16200
    assert len({u[0] for u in dev}) == len(dev)                     # unique unit ids


def test_loaded_cells_match_the_spec_table():
    assert S.loaded_cells() == [("CAP-steady", 8), ("CAP-steady", 32), ("CAP-steady", 96), ("RATE-steady", 96),
                                ("CAP-drop", 32), ("CAP-drop", 96), ("CAP-burst", 32), ("CAP-burst", 96)]


def test_every_dev_unit_builds():
    for uid, scn, m, meth, seed in S.units(S.dev_methods(), S.DEV_SEEDS)[::97]:
        sc = S.build(scn, m, meth, seed)
        assert sc.acc.l_target == 0.45 and sc.workload.n_agents == m and sc.workload.horizon == 120.0
        assert sc.retry.timeout == 1.0 and sc.analysis.success_grace == 20.0


FROZEN = {"acc": dict(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, kappa=0.8), "best_fixed_w": 4,
          "best_pacer_rate": 14.0,
          "best_fixed_w_per_scenario": {"CAP-steady": 4, "RATE-steady": 32, "CAP-drop": 8, "CAP-burst": 24}}


def test_test_grid_and_ablation_flags():
    ms = S.test_methods(FROZEN)
    tags = [S.method_tag(m) for m in ms]
    assert tags == ["acc_a0.5_b0.5_e0.1_n16_k0.8", "acc_a0.5_b0.5_e0.1_n16_k1",
                    "acc_a0.5_b0.5_e0.1_n16_k0.8_no429", "acc_a0.5_b0.5_e0.1_n16_k0.8_nolat",
                    "acc_a0.5_b0.5_e0.1_n16_k0.8_additive",
                    "fixed_w4", "fixed_w16", "fixed_w64", "fixed_w8", "fixed_w24", "fixed_w32", "tb_r14"]
    assert S.build("CAP-steady", 8, ms[2], 100).acc.use_throttle_branch is False
    assert S.build("CAP-steady", 8, ms[3], 100).acc.use_latency_branch is False
    assert S.build("CAP-steady", 8, ms[4], 100).acc.multiplicative_decrease is False
    assert S.build("CAP-steady", 8, ms[1], 100).acc.kappa == 1.0


def test_kappa_one_frozen_config_not_duplicated():
    f = {**FROZEN, "acc": {**FROZEN["acc"], "kappa": 1.0}}
    tags = [S.method_tag(m) for m in S.test_methods(f)]
    assert len(tags) == len(set(tags)) == 11


def test_scenarios_have_the_approved_events():
    drop = S.build("CAP-drop", 32, S.fixed_method(8), 0)
    assert drop.server.capacity_events == ((40.0, 1 / 3), (80.0, 1.0)) and drop.server.capacity == 24
    burst = S.build("CAP-burst", 32, S.fixed_method(8), 0).workload.burst
    assert (burst.t_start, burst.t_end, burst.rate) == (40.0, 60.0, 80.0)
    rate = S.build("RATE-steady", 8, S.fixed_method(8), 0).server
    assert (rate.rate_limit, rate.burst, rate.capacity) == (100.0, 20, 48)
    assert S.build("CAP-steady", 8, S.fixed_method(8), 0).server.capacity == 4


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


# ---------- selection logic (synthetic, design v2) ----------
def _rows(tag, kind, per_cell, seeds=(0, 1), **meta):
    """per_cell: {(scenario, M): (goodput, slo_violation, success, throttle)}"""
    out = []
    for (scn, m), vals in per_cell.items():
        g, v, ok, thr = (*vals, 0.0) if len(vals) == 3 else vals
        for s in seeds:
            out.append({"unit_id": f"{scn}{m}{tag}{s}", "scenario": scn, "M": m, "seed": s, "tag": tag,
                        "m_kind": kind, **{f"m_{k}": x for k, x in meta.items()},
                        "goodput": g, "slo_violation_rate": v, "call_success_rate": ok, "throttle_rate": thr})
    return out


LOADED = [("A", 16), ("A", 64), ("B", 16)]       # ("A", 4) is a light cell


@pytest.fixture
def synthetic_design(monkeypatch):
    monkeypatch.setattr(S, "loaded_cells", lambda: list(LOADED))
    monkeypatch.setattr(S, "SCENARIOS", {"A": {}, "B": {}})


def acc(a=.5, b=.5, e=.1, n=16, k=1.0):
    return dict(alpha=a, beta=b, epsilon=e, n_c=n, kappa=k)


def _synthetic():
    rows = []
    good = {("A", 4): (10, 0, 1), ("A", 16): (40, 0.02, 1), ("A", 64): (40, 0.02, 1), ("B", 16): (20, 0.01, 1)}
    # accA: highest goodput but violates the SLO in ONE loaded cell (scenario-mean would hide it) -> infeasible
    rows += _rows("accA", "acc", {**good, ("A", 64): (48, 0.09, 1.0)}, **acc(a=.75))
    rows += _rows("accB", "acc", good, **acc(a=.25))                           # feasible -> expected pick
    rows += _rows("accC", "acc", {**good, ("B", 16): (15, 0.0, 1.0)}, **acc(a=.5))
    # accD: fine on SLO but throttles 15% in one cell -> infeasible (throttle constraint)
    rows += _rows("accD", "acc", {**good, ("B", 16): (25, 0.0, 1.0, 0.15)}, **acc(a=.5, n=8))
    rows += _rows("fixed_w4", "fixed", {("A", 4): (9, 0, 1), ("A", 16): (20, 0, 1), ("A", 64): (20, 0, 1), ("B", 16): (10, 0, 1)}, w=4)
    rows += _rows("fixed_w64", "fixed", {("A", 4): (10, 0, 1), ("A", 16): (44, .5, .9), ("A", 64): (44, .5, .9), ("B", 16): (30, 0, 1)}, w=64)
    rows += _rows("tb_r40", "tokenbucket", {("A", 4): (10, 0, 1), ("A", 16): (30, 0, 1), ("A", 64): (30, 0, 1), ("B", 16): (20, 0, 1)}, rate=40.0)
    return pd.DataFrame(rows)


def test_selection_uses_per_cell_feasibility_and_throttle_constraint(synthetic_design):
    res = select_frozen(_synthetic())
    assert res["ok"]
    assert res["acc"] == acc(a=.25)                                     # accA (one-cell SLO) and accD (throttle) out
    assert res["best_fixed_w"] == 4 and res["best_pacer_rate"] == 40.0
    sc = score_methods(_synthetic(), LOADED).set_index("tag")
    assert not sc.loc["accA", "feasible"] and not sc.loc["accD", "feasible"] and sc.loc["accB", "feasible"]


def test_denominators_come_from_feasible_methods_only(synthetic_design):
    res = select_frozen(_synthetic())
    # A|64: accA has 48 but is infeasible there (SLO 0.09) -> denominator 40; B|16: fixed_w64 30 is feasible there
    assert res["denominators"] == {"A|16": 40.0, "A|64": 40.0, "B|16": 30.0}
    sc = score_methods(_synthetic(), LOADED, res["denominators"]).set_index("tag")
    assert sc.loc["accB", "score"] == pytest.approx((1 + 1 + 20 / 30) / 3)


def test_cell_without_feasible_method_is_excluded_from_scoring(synthetic_design):
    df = _synthetic()
    df.loc[(df["scenario"] == "B"), "slo_violation_rate"] = 0.5
    res = select_frozen(df)
    assert res["unscored_cells"] == ["B|16"] and "B|16" not in res["denominators"]
    assert res["ok"] is False and "acc" not in res                      # nobody is feasible in B|16: never relaxed


def test_light_cells_do_not_count(synthetic_design):
    df = _synthetic()
    df.loc[(df["scenario"] == "A") & (df["M"] == 4), "slo_violation_rate"] = 0.9   # light cell
    assert select_frozen(df)["acc"] == acc(a=.25)


def test_ties_prefer_lower_slo_then_conservative_order(synthetic_design):
    df = _synthetic()
    twin = df[df["tag"] == "accB"].copy()
    twin["tag"], twin["m_kappa"] = "accB_k08", 0.8
    twin["goodput"] *= 0.997                                            # within the 0.5% tie tolerance
    twin["slo_violation_rate"] = 0.0                                     # but lower SLO violation
    res = select_frozen(pd.concat([df, twin]))
    assert res["acc"] == acc(a=.25, k=0.8)
    twin["slo_violation_rate"] = df[df["tag"] == "accB"]["slo_violation_rate"].values
    res = select_frozen(pd.concat([df, twin]))                          # equal SLO -> kappa descending (1.0 first)
    assert res["acc"] == acc(a=.25, k=1.0)


def test_per_scenario_best_fixed_w_and_loso(synthetic_design):
    res = select_frozen(_synthetic())
    assert res["best_fixed_w_per_scenario"] == {"A": 4, "B": 64}       # w64 is feasible and best in B only
    lo = res["loso"]
    assert set(lo) == {"A", "B"}
    assert lo["B"]["fixed"]["config"] == 4                              # chosen on A alone
    assert lo["B"]["fixed"]["score_on_left_out"] == pytest.approx(10 / 30)


def test_pareto_front(synthetic_design):
    front = {p["tag"] for p in select_frozen(_synthetic())["pareto_front"]}
    assert "accA" in front and "tb_r40" in front                        # highest score / zero SLO with best score
    assert "fixed_w4" not in front                                       # same zero SLO as tb_r40, lower score
    assert "accC" not in front                                           # accD: same mean SLO, higher score


def test_selection_reports_infeasible_instead_of_relaxing(synthetic_design):
    df = _synthetic()
    df.loc[df["tag"].str.startswith("acc"), "slo_violation_rate"] = 0.9
    res = select_frozen(df)
    assert res["ok"] is False and "acc" not in res
    assert res["top"]["acc"]


def test_incomplete_candidate_is_not_eligible(synthetic_design):
    df = _synthetic()
    df = df[~((df["tag"] == "accB") & (df["scenario"] == "B"))]
    sc = score_methods(df, LOADED).set_index("tag")
    assert not sc.loc["accB", "complete"] and not sc.loc["accB", "feasible"]


def test_missing_values_count_as_worst_case():
    df = _synthetic()
    df.loc[df["tag"] == "accB", ["slo_violation_rate", "throttle_rate", "call_success_rate"]] = None
    c = cell_means(df).query("tag == 'accB'")
    assert c["slo_violation_rate"].eq(1.0).all() and c["throttle_rate"].eq(1.0).all() and c["call_success_rate"].eq(0.0).all()


# ---------- statistics ----------
def test_sign_flip_test_exact_values():
    from eem_acc.analysis import sign_flip_test
    assert sign_flip_test([1.0] * 10) == pytest.approx(2 / 1024)        # only all-plus and all-minus are as extreme
    assert sign_flip_test([1.0, -1.0]) == 1.0
    assert sign_flip_test([3.0, 1.0, 2.0]) == pytest.approx(2 / 8)


def test_holm_adjustment():
    from eem_acc.analysis import holm
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_t_ci_matches_known_value():
    from eem_acc.analysis import t_ci
    lo, hi = t_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (lo, hi) == pytest.approx((3 - 2.776 * (2.5 ** 0.5) / 5 ** 0.5, 3 + 2.776 * (2.5 ** 0.5) / 5 ** 0.5))


def test_primary_comparisons_on_synthetic_test_data(monkeypatch):
    from eem_acc.analysis import primary_comparisons, robustness_matrix
    frozen = {"acc": acc(), "best_fixed_w": 4, "best_pacer_rate": 40.0,
              "best_fixed_w_per_scenario": {"A": 64, "B": 4},
              "criterion": {"loaded_cells": ["A|16", "B|16"]}, "denominators": {"A|16": 40.0, "B|16": 20.0}}
    monkeypatch.setattr(S, "SCENARIOS", {"A": {}, "B": {}})
    rows = []
    seeds = tuple(range(100, 110))
    for s in seeds:
        bump = 0.1 * (s - 100)
        rows += _rows(S.method_tag(S.acc_method(**acc())), "acc", {("A", 16): (36 + bump, 0, 1), ("B", 16): (20, 0, 1)}, seeds=(s,))
        rows += _rows("fixed_w4", "fixed", {("A", 16): (20, 0, 1), ("B", 16): (18, 0, 1)}, seeds=(s,), w=4)
        rows += _rows("fixed_w64", "fixed", {("A", 16): (40, .5, .9), ("B", 16): (20, 0, 1)}, seeds=(s,), w=64)
        rows += _rows("tb_r40", "tokenbucket", {("A", 16): (30, 0, 1), ("B", 16): (20, 0, 1)}, seeds=(s,), rate=40.0)
    test = pd.DataFrame(rows)
    pc = primary_comparisons(test, frozen).set_index("endpoint")
    p1a = pc.loc["P1a ACC - global fixed W"]
    assert p1a["mean_diff"] > 0 and p1a["p_value"] == pytest.approx(2 / 1024) and p1a["n_seeds"] == 10
    p2a = pc.loc["P2 A: ACC - best fixed W=64"]
    assert p2a["mean_diff"] < 0
    assert (pc["p_holm"] >= pc["p_value"] - 1e-12).all()
    rm = robustness_matrix(test, frozen)
    assert set(rm["method"]) >= {"ACC (frozen)", "fixed W=64 (best for A)"}


def test_loso_with_a_single_scenario_is_skipped_not_crashing(monkeypatch):
    monkeypatch.setattr(S, "loaded_cells", lambda: [("A", 16), ("A", 64)])
    monkeypatch.setattr(S, "SCENARIOS", {"A": {}})
    df = _synthetic()
    res = select_frozen(df[df["scenario"] == "A"])
    assert res["loso"] == {"A": {"skipped": "no loaded cells left to train on"}}


def test_score_methods_refuses_empty_input():
    with pytest.raises(ValueError):
        score_methods(_synthetic(), [])



def test_primary_comparisons_pair_by_seed(monkeypatch):
    """Baseline varies strongly by seed, ACC = baseline + 1 in every seed: paired differences are all +1 (p = 2/1024).
    Any mis-pairing across seeds would mix signs and destroy significance."""
    from eem_acc.analysis import primary_comparisons
    frozen = {"acc": acc(), "best_fixed_w": 4, "best_pacer_rate": 40.0, "best_fixed_w_per_scenario": {"A": 4},
              "criterion": {"loaded_cells": ["A|16"]}, "denominators": {"A|16": 1.0}}
    monkeypatch.setattr(S, "SCENARIOS", {"A": {}})
    rows = []
    for s in range(100, 110):
        base = 10.0 * ((s * 7) % 10)                              # 0..90, scrambled order
        rows += _rows(S.method_tag(S.acc_method(**acc())), "acc", {("A", 16): (base + 1, 0, 1)}, seeds=(s,))
        rows += _rows("fixed_w4", "fixed", {("A", 16): (base, 0, 1)}, seeds=(s,), w=4)
        rows += _rows("tb_r40", "tokenbucket", {("A", 16): (base, 0, 1)}, seeds=(s,), rate=40.0)
    pc = primary_comparisons(pd.DataFrame(rows), frozen).set_index("endpoint")
    r = pc.loc["P1a ACC - global fixed W"]
    assert r["mean_diff"] == pytest.approx(1.0) and r["p_value"] == pytest.approx(2 / 1024)
    assert r["ci95_low"] == pytest.approx(1.0) and r["ci95_high"] == pytest.approx(1.0)


def test_select_rerun_with_identical_result_is_not_refused(tmp_path, monkeypatch):
    from conftest import tiny_v2
    tiny_v2(monkeypatch, tmp_path, run_rq2)
    run_rq2.run_phase("dev", tmp_path)
    assert run_rq2.do_select(tmp_path) == 0
    assert run_rq2.do_select(tmp_path) == 0



def test_best_kappa1_config_is_added_to_the_test_methods():
    f = {**FROZEN, "acc_kappa1": {"config": dict(alpha=0.75, beta=0.25, epsilon=0.05, n_c=32, kappa=1.0),
                                  "feasible": False, "score": 0.9}}
    tags = [S.method_tag(m) for m in S.test_methods(f)]
    assert tags[:3] == ["acc_a0.5_b0.5_e0.1_n16_k0.8", "acc_a0.5_b0.5_e0.1_n16_k1", "acc_a0.75_b0.25_e0.05_n32_k1"]
    assert len(tags) == len(set(tags)) == 13


def test_select_reports_best_kappa1_config(synthetic_design):
    df = _synthetic()
    k08 = df[df["tag"] == "accB"].copy()
    k08["tag"], k08["m_kappa"], k08["goodput"] = "accB_k08", 0.8, k08["goodput"] * 1.05
    res = select_frozen(pd.concat([df, k08]))
    assert res["acc"]["kappa"] == 0.8                                  # the kappa 0.8 variant wins overall
    k1 = res["acc_kappa1"]
    assert k1["config"] == acc(a=.25) and k1["feasible"] is True
    sc = score_methods(pd.concat([df, k08]), LOADED, res["denominators"]).set_index("tag")
    assert k1["score"] == pytest.approx(sc.loc["accB", "score"])


def test_best_kappa1_falls_back_to_best_score_when_infeasible(synthetic_design):
    df = _synthetic()
    df.loc[df["tag"].str.startswith("acc"), "slo_violation_rate"] = 0.2
    df.loc[(df["tag"] == "accC") & ~((df["scenario"] == "A") & (df["M"] == 64)), "slo_violation_rate"] = 0.0
    res = select_frozen(df)                         # accA scores highest but is infeasible in all cells; accC in one
    assert res["acc_kappa1"]["feasible"] is False and res["acc_kappa1"]["config"] == acc(a=.5)


def test_tost_exact_values():
    from eem_acc.analysis import tost_sign_flip
    assert tost_sign_flip([0.0] * 10, 0.05) == pytest.approx(1 / 1024)       # clearly inside the margin
    assert tost_sign_flip([0.04] * 10, 0.05) == pytest.approx(1 / 1024)
    assert tost_sign_flip([0.1] * 10, 0.05) == 1.0                           # clearly outside
    assert tost_sign_flip([0.2, -0.2] * 5, 0.05) > 0.05                      # noisy: equivalence not shown


def test_about_equal_endpoint_uses_tost(monkeypatch):
    from eem_acc.analysis import primary_comparisons
    frozen = {"acc": acc(), "best_fixed_w": 4, "best_pacer_rate": 40.0, "best_fixed_w_per_scenario": {"RATE-steady": 32},
              "criterion": {"loaded_cells": ["RATE-steady|96"]}, "denominators": {"RATE-steady|96": 100.0}}
    monkeypatch.setattr(S, "SCENARIOS", {"RATE-steady": {}})
    rows = []
    for s in range(100, 110):
        jitter = 0.5 if s % 2 else -0.5
        rows += _rows(S.method_tag(S.acc_method(**acc())), "acc", {("RATE-steady", 96): (97 + jitter, 0, 1, 0.05)}, seeds=(s,))
        rows += _rows("fixed_w32", "fixed", {("RATE-steady", 96): (97, 0, 1, 0.0)}, seeds=(s,), w=32)
        rows += _rows("fixed_w4", "fixed", {("RATE-steady", 96): (19, 0, 1)}, seeds=(s,), w=4)
        rows += _rows("tb_r40", "tokenbucket", {("RATE-steady", 96): (40, 0, 1)}, seeds=(s,), rate=40.0)
    pc = primary_comparisons(pd.DataFrame(rows), frozen).set_index("endpoint")
    r = pc.loc["P2 RATE-steady: ACC - best fixed W=32"]
    assert r["test"] == "TOST +-0.05" and r["p_value"] < 0.01         # differences +-0.005 << 0.05 margin
    assert r["t_tost_equivalent"] and -0.05 < r["ci90_low"] < r["ci90_high"] < 0.05
    assert pc.loc["P1a ACC - global fixed W", "test"] == "two-sided"
