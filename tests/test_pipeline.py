"""Pipeline-level tests: interrupted-vs-uninterrupted equality, tiny end-to-end run, edge cases,
determinism. Synthetic / tiny configurations only: nothing here is a result."""

import asyncio
import importlib.util
import json
import math
from pathlib import Path

import pytest

from eem_acc import scenarios as S
from eem_acc.acc import ACCConfig, ACCController
from eem_acc.agent import Gateway, RetryConfig, WorkloadConfig
from eem_acc.env import MockServer, ServerConfig, run_virtual
from eem_acc.metrics import compute_metrics
from eem_acc.sim import run_scenario
from eem_acc.utils import ResultStore

_spec = importlib.util.spec_from_file_location(
    "run_rq2", Path(__file__).resolve().parents[1] / "experiments" / "run_rq2.py")
run_rq2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_rq2)


def _all_results(root, phase="dev"):
    store = ResultStore(root / "results" / "rq2" / phase / "units")
    return {uid: res for uid, res in store.iter_results()}


# ---------- checkpoint / resume: interrupted == uninterrupted ----------
def test_interrupted_then_resumed_equals_uninterrupted(tmp_path, monkeypatch):
    a, b = tmp_path / "uninterrupted", tmp_path / "interrupted"
    run_rq2.run_phase("dev", a, limit=14)

    real = run_rq2.run_scenario
    n = {"i": 0}

    def crash_on_6th(sc):
        n["i"] += 1
        if n["i"] == 6:
            raise RuntimeError("colab disconnected")
        return real(sc)

    monkeypatch.setattr(run_rq2, "run_scenario", crash_on_6th)
    with pytest.raises(RuntimeError):
        run_rq2.run_phase("dev", b, limit=14)
    assert len(_all_results(b)) == 5                      # partial results were kept
    monkeypatch.setattr(run_rq2, "run_scenario", real)
    run_rq2.run_phase("dev", b, limit=14)                 # resume
    ra, rb = _all_results(a), _all_results(b)
    assert ra.keys() == rb.keys() and len(ra) == 14
    assert ra == rb                                       # identical metrics, unit by unit


def test_resume_after_partial_write_leftover(tmp_path):
    """A *.tmp leftover from a write interrupted mid-way must not break or pollute the resume."""
    run_rq2.run_phase("dev", tmp_path, limit=4)
    units = tmp_path / "results" / "rq2" / "dev" / "units"
    (units / "CAP-steady__M4__junk.json.abc123.tmp").write_bytes(b'{"unit_id": "x", "res')
    run_rq2.run_phase("dev", tmp_path, limit=8)
    assert len(_all_results(tmp_path)) == 8
    assert not list(units.glob("*.tmp"))


def test_corrupted_unit_file_is_rerun(tmp_path):
    run_rq2.run_phase("dev", tmp_path, limit=3)
    units = sorted((tmp_path / "results" / "rq2" / "dev" / "units").glob("*.json"))
    units[0].write_text("{not json")                      # corrupted result file
    run_rq2.run_phase("dev", tmp_path, limit=3)
    assert len(_all_results(tmp_path)) == 3               # the unreadable unit was recomputed


# ---------- tiny end-to-end: dev -> select -> test ----------
@pytest.fixture
def tiny_design(monkeypatch, tmp_path):
    from conftest import tiny_v2
    tiny_v2(monkeypatch, tmp_path, run_rq2)
    return tmp_path


def test_tiny_end_to_end_pipeline(tiny_design):
    root = tiny_design / "out"
    run_rq2.run_phase("dev", root)
    n_dev = (2 + 2 + 1) * 1 * 1 * 2
    assert len(_all_results(root)) == n_dev == 10
    assert run_rq2.do_select(root) == 0
    frozen = json.loads((tiny_design / "frozen.json").read_text())
    assert frozen["ok"] and set(frozen["acc"]) == {"alpha", "beta", "epsilon", "n_c", "kappa"}
    run_rq2.run_phase("test", root, frozen=frozen)
    test_res = _all_results(root, "test")
    expected = len(S.units(S.test_methods(frozen), S.TEST_SEEDS))
    assert len(test_res) == expected > 0
    assert all("window_trace" in r for r in test_res.values())          # traces stored for the test phase
    seeds = {r["meta"]["seed"] for r in test_res.values()}
    assert seeds == {100, 101}
    assert seeds.isdisjoint({r["meta"]["seed"] for r in _all_results(root).values()})   # no dev/test seed overlap


def test_tiny_pipeline_is_deterministic(tiny_design):
    outs = []
    for name in ("r1", "r2"):
        root = tiny_design / name
        run_rq2.run_phase("dev", root)
        assert run_rq2.do_select(root) == 0
        outs.append(((tiny_design / "frozen.json").read_text(), _all_results(root)))
    assert outs[0] == outs[1]


def test_test_phase_refuses_without_frozen_config():
    with pytest.raises(AssertionError):
        run_rq2.phase_units("test", None)


# ---------- edge cases ----------
def test_metrics_on_empty_gateway():
    async def main():
        srv = ServerConfig(base_latency=0.1, latency_sigma=0.0, capacity=4, degradation=1.0, hard_cap=8,
                           rate_limit=100.0, burst=10, retry_after=0.1, error_latency=0.01)
        c = ACCController(ACCConfig(alpha=.5, beta=.5, epsilon=.1, w0=2, w_min=1, w_max=4, n_c=4, l_target=0.2))
        return Gateway(MockServer(srv, 0), c, RetryConfig(2, 0.01, 0.1), 0)

    m = compute_metrics(run_virtual(main()), 0.2)
    assert m["n_calls"] == 0 and m["n_attempts"] == 0
    assert m["goodput"] is None and m["lat_p95"] is None and m["throttle_rate"] is None


@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.parametrize("latency", [0.0, -1.0, float("inf")])
def test_acc_tolerates_degenerate_latencies(latency):
    c = ACCController(ACCConfig(alpha=.5, beta=.5, epsilon=.1, w0=4, w_min=1, w_max=8, n_c=4, l_target=1.0))
    for i in range(4):
        c.record(200, latency, float(i))
    assert 1 <= c.window <= 8


def test_acc_nan_latency_holds_window():
    c = ACCController(ACCConfig(alpha=.5, beta=.5, epsilon=.1, w0=4, w_min=1, w_max=8, n_c=4, l_target=1.0))
    for i in range(4):
        c.record(200, float("nan"), float(i))
    assert c.window == 4                                   # no evidence either way -> hold, never crash


def test_zero_agents_or_calls_runs_cleanly():
    from eem_acc.agent import run_agents

    async def main():
        srv = ServerConfig(base_latency=0.1, latency_sigma=0.0, capacity=4, degradation=1.0, hard_cap=8,
                           rate_limit=100.0, burst=10, retry_after=0.1, error_latency=0.01)
        c = ACCController(ACCConfig(alpha=.5, beta=.5, epsilon=.1, w0=2, w_min=1, w_max=4, n_c=4, l_target=0.2))
        gw = Gateway(MockServer(srv, 0), c, RetryConfig(2, 0.01, 0.1), 0)
        await run_agents(gw, WorkloadConfig(n_agents=3, calls_per_agent=0, phases=((0.0, 0.1),), start_jitter=0.0), 0)
        return gw

    assert run_virtual(main()).calls == []


@pytest.mark.parametrize("scn,meth", [
    ("CAP-drop", S.pacer_method(20.0)),
    ("CAP-burst", S.acc_method(**S.ACC_START, use_latency_branch=False)),
    ("RATE-steady", S.fixed_method(16)),
])
def test_same_seed_same_result_across_scenario_kinds(scn, meth):
    a = run_scenario(S.build(scn, 4, meth, 7))
    b = run_scenario(S.build(scn, 4, meth, 7))
    assert a == b
    assert run_scenario(S.build(scn, 4, meth, 8)) != a


# ---------- slow: execute the Colab notebook headlessly (not in the fast suite) ----------
@pytest.mark.notebook
def test_live_notebook_executes_headless(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "02_live.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)
    for c in nb.cells:
        c.source = c.source.replace("LIMIT = None          # e.g. 60 for a quick trial", "LIMIT = 20")
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(nb_path.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
