"""Rule-B tests: determinism, interrupted-vs-uninterrupted equality, edge cases, tiny end-to-end smoke.
Synthetic/tiny configurations only. Nothing here is a result."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from eem_acc import scenarios as S
from eem_acc.acc import ACCConfig, ACCController
from eem_acc.agent import Gateway, RetryConfig, WorkloadConfig
from eem_acc.env import MockServer, ServerConfig
from eem_acc.metrics import compute_metrics
from eem_acc.sim import Scenario, run_scenario
from eem_acc.utils import ResultStore, run_resumable

_spec = importlib.util.spec_from_file_location(
    "run_rq2", Path(__file__).resolve().parents[1] / "experiments" / "run_rq2.py")
run_rq2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_rq2)


def _all(units_dir):
    return dict(ResultStore(units_dir).iter_results())


# ------------------------------------------------------------------ determinism
@pytest.mark.parametrize("scn,meth", [
    ("CAP-drop", S.acc_method(**S.ACC_START)),
    ("CAP-burst", S.pacer_method(20.0)),
    ("RATE-steady", S.fixed_method(8)),
    ("CAP-steady", S.acc_method(**S.ACC_START, multiplicative_decrease=False)),
])
def test_same_seed_same_result_different_seed_differs(scn, meth):
    a = run_scenario(S.build(scn, 4, meth, 7))
    b = run_scenario(S.build(scn, 4, meth, 7))
    c = run_scenario(S.build(scn, 4, meth, 8))
    assert a == b
    assert a["metrics"] != c["metrics"]


def test_scenarios_differ_only_in_method_for_the_same_seed():
    """Config-level pairing check (the behavioural CRN check is in test_review_fixes.py)."""
    a = S.build("CAP-steady", 4, S.fixed_method(4), 5)
    b = S.build("CAP-steady", 4, S.acc_method(**S.ACC_START), 5)
    assert a.seed == b.seed and a.server == b.server and a.workload == b.workload and a.retry == b.retry


# ------------------------------------------------------------------ interrupted == uninterrupted
def test_rq2_runner_interrupted_then_resumed_equals_uninterrupted(tmp_path, monkeypatch):
    a_root, b_root = tmp_path / "A", tmp_path / "B"
    run_rq2.run_phase("dev", a_root, limit=10)                       # uninterrupted

    real, n = run_rq2.run_scenario, {"i": 0}

    def crash_on_fifth(sc):
        n["i"] += 1
        if n["i"] == 5:
            raise RuntimeError("colab disconnected")
        return real(sc)

    monkeypatch.setattr(run_rq2, "run_scenario", crash_on_fifth)
    with pytest.raises(RuntimeError):
        run_rq2.run_phase("dev", b_root, limit=10)                   # interrupted midway
    monkeypatch.setattr(run_rq2, "run_scenario", real)
    run_rq2.run_phase("dev", b_root, limit=10)                       # resumed
    ua, ub = (r / "results" / "rq2" / "dev" / "units" for r in (a_root, b_root))
    assert _all(ua) == _all(ub) and len(_all(ua)) == 10


def test_stateful_resume_equals_uninterrupted(tmp_path):
    """State carried across units (like the EEM store): resumed run == uninterrupted run, exactly."""
    ids = [str(i) for i in range(9)]

    def make(state):
        def run_unit(u):
            state["acc"] = state["acc"] * 31 + int(u) + 1            # order- and history-dependent
            return {"v": state["acc"]}
        return run_unit

    kw = dict(config={"c": 1}, stateful=True, checkpoint_every=2)
    s1 = {"acc": 7}
    run_resumable(ids, make(s1), tmp_path / "A", get_state=lambda: s1["acc"],
                  set_state=lambda v: s1.update(acc=v), **kw)

    s2 = {"acc": 7}
    base = make(s2)

    def crashing(u):
        if u == "6":
            raise RuntimeError("disconnect")
        return base(u)

    with pytest.raises(RuntimeError):
        run_resumable(ids, crashing, tmp_path / "B", get_state=lambda: s2["acc"],
                      set_state=lambda v: s2.update(acc=v), **kw)
    s2["acc"] = -1                                                   # fresh process: memory gone
    run_resumable(ids, base, tmp_path / "B", get_state=lambda: s2["acc"],
                  set_state=lambda v: s2.update(acc=v), **kw)
    assert _all(tmp_path / "A" / "units") == _all(tmp_path / "B" / "units")


def test_truncated_unit_file_is_recomputed_on_resume(tmp_path):
    run_rq2.run_phase("dev", tmp_path, limit=4)
    units = tmp_path / "results" / "rq2" / "dev" / "units"
    victim = sorted(units.glob("*.json"))[1]
    good = victim.read_text()
    victim.write_text(good[: len(good) // 2])                        # truncated by a disconnect
    run_rq2.run_phase("dev", tmp_path, limit=4)
    assert victim.read_text() == good                                # recomputed identically


# ------------------------------------------------------------------ edge cases
def _server(**kw):
    base = dict(base_latency=0.1, latency_sigma=0.0, capacity=4, degradation=1.0, hard_cap=8,
                rate_limit=1000.0, burst=1000, retry_after=0.2, error_latency=0.01)
    base.update(kw)
    return ServerConfig(**base)


def _acc(**kw):
    base = dict(alpha=0.5, beta=0.5, epsilon=0.1, w0=4, w_min=1, w_max=16, n_c=4, l_target=0.15)
    base.update(kw)
    return ACCConfig(**base)


RETRY = RetryConfig(max_retries=3, backoff_base=0.05, backoff_cap=0.5)


def test_metrics_on_an_empty_gateway_do_not_crash():
    gw = Gateway(MockServer(_server(), 0), ACCController(_acc()), RETRY, seed=0)
    m = compute_metrics(gw, 0.15)
    assert m["n_calls"] == 0 and m["n_attempts"] == 0
    assert m["goodput"] is None and m["throttle_rate"] is None and m["lat_p95"] is None
    assert m["makespan"] == 0.0


def test_zero_latency_server_has_zero_makespan_and_no_division_error():
    sc = Scenario(_server(base_latency=0.0, error_latency=0.0),
                  WorkloadConfig(n_agents=2, calls_per_agent=5, phases=((0.0, 0.0),), start_jitter=0.0),
                  RETRY, _acc(), seed=0)
    m = run_scenario(sc)["metrics"]
    assert m["n_calls_ok"] == 10 and m["makespan"] == 0.0 and m["goodput"] is None
    assert m["slo_violation_rate"] == 0.0


def test_acc_n_c_one_updates_on_every_attempt():
    c = ACCController(_acc(n_c=1, w0=4))
    for i in range(3):
        assert c.record(200, 0.05, now=float(i)) is True
    assert c.window == 7 and c.n_updates == 3


def test_acc_tolerates_non_positive_latency_without_crashing():
    c = ACCController(_acc(n_c=2))
    c.record(200, 0.0); c.record(200, -1.0)
    assert c.window == 5                                             # dL <= 0 -> healthy -> +1


def test_acc_decrease_factor_exactly_zero_or_negative_clamps_to_w_min():
    c = ACCController(_acc(beta=1.0, w0=6, w_min=2))
    for _ in range(4):
        c.record(429, 0.0)                                           # F = 1, beta = 1 -> factor 0
    assert c.window == 2


def test_retries_exhausted_counts_as_one_call_with_all_attempts_logged():
    sc = Scenario(_server(burst=1, rate_limit=0.0001, retry_after=0.01),
                  WorkloadConfig(n_agents=1, calls_per_agent=2, phases=((0.0, 0.0),), start_jitter=0.0),
                  RetryConfig(max_retries=2, backoff_base=0.01, backoff_cap=0.02), _acc(frozen=True), seed=0)
    m = run_scenario(sc)["metrics"]
    assert m["n_calls"] == 2 and m["n_calls_ok"] == 1 and m["n_exhausted_calls"] == 1
    assert m["n_attempts"] == 1 + 3 and m["n_429"] == 3              # ok call + (1 + 2 retries)


# ------------------------------------------------------------------ tiny end-to-end smoke (dev -> select -> test)
def test_end_to_end_pipeline_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ACC_GRID", dict(alpha=(0.25, 0.5), beta=(0.5,), epsilon=(0.1,), n_c=(8,)))
    monkeypatch.setattr(S, "FIXED_W_DEV", (2, 8))
    monkeypatch.setattr(S, "PACER_RATES_DEV", (20.0,))
    monkeypatch.setattr(S, "DEV_SEEDS", (0, 1))
    monkeypatch.setattr(S, "TEST_SEEDS", (100,))
    monkeypatch.setattr(S, "M_LEVELS", (2,))
    monkeypatch.setattr(S, "CALLS_PER_AGENT", 12)
    monkeypatch.setattr(run_rq2, "FROZEN_PATH", tmp_path / "frozen.json")

    run_rq2.run_phase("dev", tmp_path)
    assert run_rq2.do_select(tmp_path) == 0
    frozen = json.loads((tmp_path / "frozen.json").read_text())
    assert frozen["ok"] and frozen["dev_seeds"] == [0, 1]
    assert set(frozen["acc"]) == {"alpha", "beta", "epsilon", "n_c"}

    run_rq2.run_phase("test", tmp_path, frozen=frozen)
    test_units = _all(tmp_path / "results" / "rq2" / "test" / "units")
    expected = len(S.units(S.test_methods(frozen), S.TEST_SEEDS))
    assert len(test_units) == expected > 0
    assert all("window_trace" in r for r in test_units.values())     # traces are kept in the test phase
    assert {r["meta"]["seed"] for r in test_units.values()} == {100}   # test uses ONLY test seeds
    dev_seeds_used = {r["meta"]["seed"] for r in _all(tmp_path / "results" / "rq2" / "dev" / "units").values()}
    assert dev_seeds_used == {0, 1} and not dev_seeds_used & {100}     # no seed overlap dev/test
