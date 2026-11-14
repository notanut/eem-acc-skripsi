"""Tests for the review fixes: CRN, shrink-before-release, grant-time logging, J metrics, result stamping."""

import asyncio
import json

import numpy as np

import pytest

from eem_acc import scenarios as S
from eem_acc.agent import SlotPool
from eem_acc.env import MockServer, run_virtual
from eem_acc.metrics import compute_metrics, objective_J
from eem_acc.sim import SIM_VERSION, run_scenario, simulate
import experiments.run_rq2 as run_rq2  # noqa: E402  (conftest/pyproject put repo root on path via test_reproducibility usage)


def _server(seed=3):
    return MockServer(S.build("CAP-steady", 4, S.fixed_method(4), seed).server, seed)


def test_noise_is_keyed_by_request_not_by_call_order():
    s1, s2 = _server(), _server()
    reqs = [(0, 0), (1, 0), (2, 5), (3, 1)]
    a = [s1._noise(r) for r in reqs]
    b = [s2._noise(r) for r in reversed(reqs)][::-1]
    assert a == b                                   # order independent
    assert len(set(a)) == len(a)                    # different requests -> different noise
    assert s1._noise((0, 0)) == s1._noise((0, 0))   # same request -> same noise


def test_crn_same_call_same_latency_across_methods():
    """The per-request pairing (not just the multiset) must hold: below capacity, logical call (agent, k) gets
    the same latency under two methods even though the calls are served in a different order and with a
    different number of rejected attempts before admission."""
    def lat_by_call(meth, seed=7):
        sc = S.build("CAP-steady", 4, meth, seed)
        seen = {}
        orig = MockServer.handle
        async def spy(self, request=None):
            r = await orig(self, request)
            if r.status == 200:
                seen[request] = r.latency
            return r
        MockServer.handle = spy
        try:
            simulate(sc)
        finally:
            MockServer.handle = orig
        return seen
    a, b = lat_by_call(S.fixed_method(1)), lat_by_call(S.acc_method(**S.ACC_START))
    assert len(a) == len(b) == 4 * 150 and set(a) == set(b)
    assert all(a[k] == pytest.approx(b[k]) for k in a)
    c = lat_by_call(S.fixed_method(1), seed=8)
    assert sum(a[k] != pytest.approx(c[k]) for k in a) > 500


def test_crn_noise_ignores_number_of_rejected_attempts():
    """Same logical call: noise must not depend on how many 429/503 came before admission."""
    s = _server()
    z0 = s._noise((1, 4))
    for r in [(0, 0), (1, 3), (2, 2)] * 3:
        s._noise(r)
    assert s._noise((1, 4)) == z0 == _server()._noise((1, 4))


def test_slot_granted_window_logged_at_grant_time():
    async def go():
        p = SlotPool(1)
        first = await p.acquire()
        t = asyncio.ensure_future(p.acquire())
        await asyncio.sleep(0)
        p.set_limit(3)
        p.release()
        return first, await t
    first, second = run_virtual(go())
    assert first == (1, 1)
    assert second[0] == 3 and second[1] >= 1


def test_cancelled_waiter_does_not_leak_a_slot():
    async def go():
        p = SlotPool(1)
        await p.acquire()
        t = asyncio.ensure_future(p.acquire())
        await asyncio.sleep(0)
        p.release()          # slot handed to the waiter's future
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return p.in_use
    assert run_virtual(go()) == 0


def test_objective_J_matches_eq1_and_handles_none():
    m = {"call_success_rate": 0.9, "mean_latency_excess_all_attempts": 0.2, "throttle_rate": 0.1}
    assert objective_J(m, 0.5, 0.5) == pytest.approx(0.9 - 0.1 - 0.05)
    assert objective_J(m, 1.0, 1.0) == pytest.approx(0.9 - 0.2 - 0.1)
    assert objective_J({**m, "throttle_rate": None}, 0.5, 0.5) is None


def test_metrics_all_attempts_excess_and_waste_consistent():
    r = run_scenario(S.build("RATE-steady", 16, S.fixed_method(16), 2))["metrics"]
    assert r["wasted_attempts_per_ok_call"] == pytest.approx((r["n_attempts"] - r["n_calls_ok"]) / r["n_calls_ok"])
    # error attempts are fast (excess 0), so averaging over all attempts can only dilute the success-only excess
    assert 0 <= r["mean_latency_excess_all_attempts"] <= r["mean_latency_excess"] + 1e-12


def test_spec_hash_changes_with_config_and_is_stable():
    u = ("CAP-steady", 4, S.fixed_method(4), 1)
    assert run_rq2.spec_hash(*u) == run_rq2.spec_hash(*u)
    assert run_rq2.spec_hash(*u) != run_rq2.spec_hash("CAP-steady", 4, S.fixed_method(8), 1)
    assert run_rq2.spec_hash(*u) != run_rq2.spec_hash("CAP-steady", 4, S.fixed_method(4), 2)


def test_resume_refuses_results_from_other_simulator_version(tmp_path, monkeypatch):
    run_rq2.run_phase("dev", tmp_path, limit=2)
    monkeypatch.setattr(run_rq2, "SIM_VERSION", "other")
    with pytest.raises(ValueError, match="do not match the current configuration"):
        run_rq2.run_phase("dev", tmp_path, limit=2)


def test_stamps_present_in_results(tmp_path):
    run_rq2.run_phase("dev", tmp_path, limit=1)
    f = next((tmp_path / "results/rq2/dev/units").glob("*.json"))
    d = json.loads(f.read_text())
    meta = (d.get("result") or d)["meta"]
    assert meta["sim_version"] == SIM_VERSION and len(meta["spec_hash"]) == 16


def test_select_refuses_when_test_results_exist(tmp_path):
    (tmp_path / "results/rq2/test/units").mkdir(parents=True)
    (tmp_path / "results/rq2/test/units/x.json").write_text("{}")
    assert run_rq2.do_select(tmp_path) == 4


# ------------------------------------------------------------------------------------------------ gateway
from eem_acc.acc import ACCConfig, ACCController
from eem_acc.agent import Gateway, RetryConfig, WorkloadConfig
from eem_acc.agent.scripted import _agent
from eem_acc.env.server import Response


class FakeServer:
    """Scripted server: status/latency per request; records dispatch times."""
    def __init__(self, plan=None, raise_on=None):
        self.plan, self.raise_on = plan or {}, raise_on
        self.dispatch, self.max_inflight_seen, self.inflight = {}, 0, 0

    async def handle(self, request=None):
        loop = asyncio.get_running_loop()
        self.dispatch.setdefault(request, []).append(loop.time())
        if request == self.raise_on:
            raise RuntimeError("server blew up")
        status, lat, ra = self.plan.get(request, (200, 1.0, None))
        self.inflight += 1
        self.max_inflight_seen = max(self.max_inflight_seen, self.inflight)
        try:
            await asyncio.sleep(lat)
        finally:
            self.inflight -= 1
        return Response(status, lat, retry_after=ra)


def _acc(**kw):
    base = dict(alpha=0.5, beta=0.5, epsilon=0.1, w0=2, w_min=1, w_max=8, n_c=1, l_target=10.0)
    base.update(kw)
    return ACCController(ACCConfig(**base))


RETRY = RetryConfig(max_retries=2, backoff_base=0.05, backoff_cap=1.0)


def test_shrink_applies_before_the_slot_is_released():
    """W=2, n_c=1: call A's 429 at t=0.01 shrinks W to 1. Releasing A's slot must NOT admit the queued call C
    (in_use stays 1 = B); C may only start when B finishes at t=1.0. Releasing before shrinking admits C at 0.01."""
    srv = FakeServer({(0, 0): (429, 0.01, 5.0)})
    gw = Gateway(srv, _acc(), RETRY, seed=0)

    async def go():
        await asyncio.gather(*(gw.call(request=(a, 0)) for a in range(3)))
    run_virtual(go())
    assert srv.dispatch[(2, 0)][0] >= 1.0 - 1e-9
    assert srv.max_inflight_seen <= 2


def test_attempt_logs_window_at_grant_time_not_at_resume():
    """The limit is changed between the grant and the waiter's resumption; the record must show the grant-time limit."""
    srv = FakeServer()
    gw = Gateway(srv, _acc(w0=1, frozen=True), RETRY, seed=0)
    real_release = gw.slots.release

    def release_then_change_limit():
        real_release()
        gw.slots.limit = 99            # after the grant, before the granted waiter runs
    gw.slots.release = release_then_change_limit

    async def go():
        await asyncio.gather(gw.call(request=(0, 0)), gw.call(request=(1, 0)))
    run_virtual(go())
    assert [a.window for a in gw.attempts] == [1, 1]


@pytest.mark.parametrize("where", ["server", "controller"])
def test_slot_released_when_server_or_controller_raises(where):
    srv = FakeServer(raise_on=(0, 0) if where == "server" else None)
    ctrl = _acc()
    if where == "controller":
        def boom(*a, **k):
            raise RuntimeError("controller blew up")
        ctrl.record = boom
    gw = Gateway(srv, ctrl, RETRY, seed=0)

    async def go():
        with pytest.raises(RuntimeError):
            await gw.call(request=(0, 0))
        return gw.slots.in_use
    assert run_virtual(go()) == 0


def test_backoff_jitter_is_keyed_by_call_and_retry_not_by_order():
    mk = lambda: Gateway(FakeServer(), _acc(), RETRY, seed=11)
    g1, g2 = mk(), mk()
    keys = [(0, (3, 7)), (1, (3, 7)), (0, (0, 0)), (1, (5, 2))]
    v1 = {kk: g1._backoff(*kk) for kk in keys}
    v2 = {kk: g2._backoff(*kk) for kk in reversed(keys)}
    assert v1 == v2
    assert v1[(0, (3, 7))] != v1[(1, (3, 7))] / 2      # retries of one call get distinct jitter
    assert len({round(v / min(1.0, 0.05 * 2 ** k), 12) for (k, _), v in v1.items()}) == len(keys)


def test_think_time_draws_do_not_depend_on_timing():
    """Two runs with different service times cross a zero-think phase after different numbers of calls. The
    k-th think time (where nonzero in both) must still be the same variate: one draw per call, always."""
    phases = ((0.0, 0.1), (1.0, 0.0), (1.2, 0.1))
    wl = WorkloadConfig(n_agents=1, calls_per_agent=60, phases=phases, start_jitter=0.0)

    class G:
        def __init__(self, d):
            self.d, self.log = d, []
        async def call(self, request=None):
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await asyncio.sleep(self.d)
            self.log.append((t0, loop.time()))

    def thinks(d):
        g = G(d)
        run_virtual(_agent(g, wl, np.random.default_rng(5), 0))
        prev, out = 0.0, []
        for t0, t1 in g.log:
            out.append(t0 - prev)
            prev = t1
        return out
    a, b = thinks(0.01), thinks(0.05)
    both = [k for k in range(60) if a[k] > 1e-12 and b[k] > 1e-12]
    assert len(both) > 20 and any(a[k] <= 1e-12 for k in range(60)) and any(b[k] <= 1e-12 for k in range(60))
    assert all(a[k] == pytest.approx(b[k]) for k in both)


def test_floor_epsilon_recovers_exact_integer():
    """12*(1 - 0.5*25/30) is exactly 7 but evaluates to 6.999999999999999 in floats."""
    ctrl = _acc(w0=12, w_max=64, n_c=30, beta=0.5)
    for i in range(30):
        ctrl.record(429 if i < 25 else 200, 0.1, float(i))
    assert ctrl.window == 7


# ------------------------------------------------------------------------------------------- result store
from eem_acc.utils import ResultStore


def test_sanitised_ids_do_not_collide(tmp_path):
    st = ResultStore(tmp_path)
    st.save("a/b", {"v": 1})
    st.save("a_b", {"v": 2})
    assert st.load("a/b") == {"v": 1} and st.load("a_b") == {"v": 2}


def test_file_of_another_unit_is_not_accepted(tmp_path):
    st = ResultStore(tmp_path)
    st.save("x", {"v": 1})
    st.path("x").replace(st.path("y"))
    assert not st.is_valid("y")
    with pytest.raises(ValueError):
        st.load("y")


# ------------------------------------------------------------------------------------- runner guards
def _tiny(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "ACC_GRID", dict(alpha=(0.25, 0.5), beta=(0.5,), epsilon=(0.1,), n_c=(8,)))
    monkeypatch.setattr(S, "FIXED_W_DEV", (2, 8))
    monkeypatch.setattr(S, "PACER_RATES_DEV", (20.0,))
    monkeypatch.setattr(S, "DEV_SEEDS", (0, 1))
    monkeypatch.setattr(S, "TEST_SEEDS", (100,))
    monkeypatch.setattr(S, "M_LEVELS", (2,))
    monkeypatch.setattr(S, "CALLS_PER_AGENT", 12)
    monkeypatch.setattr(run_rq2, "FROZEN_PATH", tmp_path / "repo_frozen.json")


def test_select_refuses_stale_dev_results(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    monkeypatch.setattr(run_rq2, "SIM_VERSION", "older")
    assert run_rq2.do_select(tmp_path) == 5


def test_select_refuses_foreign_leftover_file(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    ResultStore(tmp_path / "results/rq2/dev/units").save("leftover_from_old_grid", {"meta": {}, "metrics": {}})
    assert run_rq2.do_select(tmp_path) == 5


def test_frozen_records_version_and_digest_and_test_phase_checks_them(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    assert run_rq2.do_select(tmp_path) == 0
    fz = json.loads(run_rq2.FROZEN_PATH.read_text())
    assert fz["sim_version"] == SIM_VERSION and len(fz["dev_digest"]) == 16
    frozen, msg = run_rq2.load_frozen_for_test(tmp_path)
    assert frozen == fz, msg
    # repo copy edited -> copies differ -> refuse
    run_rq2.FROZEN_PATH.write_text(json.dumps({**fz, "dev_digest": "x"}))
    assert run_rq2.load_frozen_for_test(tmp_path)[0] is None
    run_rq2.FROZEN_PATH.write_text(json.dumps(fz))
    # simulator version changed -> refuse
    monkeypatch.setattr(run_rq2, "SIM_VERSION", "newer")
    assert run_rq2.load_frozen_for_test(tmp_path)[0] is None


def test_frozen_selected_on_another_root_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    a, b = tmp_path / "A", tmp_path / "B"
    run_rq2.run_phase("dev", a)
    assert run_rq2.do_select(a) == 0
    monkeypatch.setattr(S, "CALLS_PER_AGENT", 13)       # B's dev results differ from A's
    run_rq2.run_phase("dev", b)
    (b / "results/rq2").mkdir(parents=True, exist_ok=True)
    # B has no own selection; only the repo copy (selected on A) exists -> digest mismatch -> refuse
    assert run_rq2.load_frozen_for_test(b)[0] is None


def test_select_refuses_to_overwrite_a_different_frozen_file(tmp_path, monkeypatch):
    _tiny(monkeypatch, tmp_path)
    run_rq2.run_phase("dev", tmp_path)
    run_rq2.FROZEN_PATH.write_text(json.dumps({"something": "else"}))
    assert run_rq2.do_select(tmp_path) == 6
    assert run_rq2.do_select(tmp_path, force=True) == 0
    assert run_rq2.do_select(tmp_path) == 0             # identical content: allowed


def test_iter_results_skips_result_copied_under_another_name(tmp_path):
    st = ResultStore(tmp_path)
    st.save("x", {"v": 1})
    import shutil
    shutil.copy(st.path("x"), tmp_path / "copy_of_x.json")
    assert list(st.iter_results()) == [("x", {"v": 1})]
