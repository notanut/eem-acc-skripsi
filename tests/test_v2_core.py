"""Design v2 simulator core: kappa, client timeout as overload, client timeouts, horizon runs, open-loop burst."""

import asyncio
import dataclasses

import pytest

from eem_acc.status import CLIENT_TIMEOUT as TO

from eem_acc.acc import ACCConfig, ACCController
from eem_acc.agent import BurstConfig, Gateway, RetryConfig, WorkloadConfig
from eem_acc.agent.scripted import run_agents
from eem_acc.env import MockServer, ServerConfig, run_virtual
from eem_acc.env.server import Response
from eem_acc.sim import Scenario, simulate


def acfg(**kw):
    base = dict(alpha=0.5, beta=0.5, epsilon=0.1, w0=4, w_min=1, w_max=64, n_c=8, l_target=0.5)
    base.update(kw)
    return ACCConfig(**base)


def scfg(**kw):
    base = dict(base_latency=0.2, latency_sigma=0.0, capacity=1000, degradation=0.0, hard_cap=10_000,
                rate_limit=1e6, burst=10**6, retry_after=0.5, error_latency=0.02)
    base.update(kw)
    return ServerConfig(**base)


# ------------------------------------------------------------------ controller
def test_kappa_scales_the_latency_target():
    c1, c8 = ACCController(acfg()), ACCController(acfg(kappa=0.8))
    buf = [(200, 0.45)] * 20
    assert c1.latency_degradation(buf) == pytest.approx((0.45 - 0.5) / 0.5)
    assert c8.latency_degradation(buf) == pytest.approx((0.45 - 0.4) / 0.4)
    assert c1.next_window(10, 0.0, c1.latency_degradation(buf)) == 11        # below 0.5: increase
    assert c8.next_window(10, 0.0, c8.latency_degradation(buf)) < 10         # above 0.8*0.5: decrease


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01])
def test_kappa_validated(bad):
    with pytest.raises(ValueError):
        acfg(kappa=bad)


def test_client_timeout_counts_as_overload_signal():
    c = ACCController(acfg(w0=10, n_c=4, beta=0.5))
    for s in (TO, 200, 200, 200):
        c.record(s, 0.1)
    assert c.window == 8                       # floor(10 * (1 - 0.5 * 1/4)) = 8: the throttle branch fired


# ------------------------------------------------------------------ client timeouts
class SlowFirstServer:
    """First attempt of every call takes `slow` s, later ones `fast` s. Tracks in-flight work."""
    def __init__(self, slow=3.0, fast=0.1):
        self.slow, self.fast, self.seen, self.inflight, self.max_inflight_seen = slow, fast, {}, 0, 0

    async def handle(self, request=None):
        n = self.seen.get(request, 0)
        self.seen[request] = n + 1
        self.inflight += 1
        self.max_inflight_seen = max(self.max_inflight_seen, self.inflight)
        try:
            lat = self.slow if n == 0 else self.fast
            await asyncio.sleep(lat)
            return Response(200, lat)
        finally:
            self.inflight -= 1


def test_timeout_records_client_timeout_retries_and_server_keeps_working():
    srv = SlowFirstServer()
    gw = Gateway(srv, ACCController(acfg(frozen=True)), RetryConfig(3, 0.05, 1.0, timeout=1.0), seed=0)

    async def go():
        rec = await gw.call(request=(0, 0))
        busy_after_call = srv.inflight              # the abandoned first attempt is still running on the server
        await gw.shutdown()
        return rec, busy_after_call
    rec, busy = run_virtual(go())
    assert [a.status for a in gw.attempts] == [TO, 200]
    assert gw.attempts[0].latency == pytest.approx(1.0)
    assert rec.ok and rec.attempts == 2
    assert busy == 1
    assert srv.inflight == 0 and not gw._abandoned     # shutdown cancelled the abandoned work


def test_no_timeout_configured_means_no_client_timeout():
    srv = SlowFirstServer()
    gw = Gateway(srv, ACCController(acfg(frozen=True)), RetryConfig(3, 0.05, 1.0), seed=0)
    run_virtual(gw.call(request=(0, 0)))
    assert [a.status for a in gw.attempts] == [200] and gw.attempts[0].latency == pytest.approx(3.0)


# ------------------------------------------------------------------ horizon runs
def horizon_scenario(m=4, T=10.0, burst=None, server=None, retry=None, seed=0, w=4):
    wl = WorkloadConfig(n_agents=m, calls_per_agent=None, phases=((0.0, 0.1),), start_jitter=1.0,
                        horizon=T, burst=burst)
    return Scenario(server=server or scfg(), workload=wl, retry=retry or RetryConfig(3, 0.05, 1.0),
                    acc=acfg(w0=w, w_max=max(64, w), frozen=True), seed=seed)


@pytest.mark.parametrize("bad", [dict(calls_per_agent=5, horizon=10.0), dict(calls_per_agent=None, horizon=None),
                                 dict(calls_per_agent=5, horizon=None, burst=BurstConfig(1, 2, 3.0))])
def test_workload_mode_validation(bad):
    with pytest.raises(ValueError):
        WorkloadConfig(n_agents=2, phases=((0.0, 0.1),), start_jitter=0.0, **bad)


def test_horizon_run_stops_at_T_and_keeps_unfinished_calls_open():
    # 1 slot, 8 agents, 0.2 s service: heavily queued, so calls are still waiting at T
    gw = simulate(horizon_scenario(m=8, T=10.0, w=1))
    assert all(a.t_dispatch < 10.0 for a in gw.attempts)
    assert all(c.t_done <= 10.0 + 1e-9 for c in gw.calls)
    assert len(gw.open_calls) >= 1                       # requested before T, unfinished
    assert all(t < 10.0 for t, _ in gw.open_calls.values())
    assert gw.slots.in_use == 0 and gw.server.inflight == 0 and gw.waiting == 0


def test_horizon_run_is_deterministic():
    sc = horizon_scenario(m=6, T=8.0, burst=BurstConfig(2.0, 4.0, 20.0),
                          server=scfg(latency_sigma=0.3, capacity=3, degradation=1.0, hard_cap=6),
                          retry=RetryConfig(3, 0.05, 1.0, timeout=0.5))
    a, b = simulate(sc), simulate(sc)
    key = lambda gw: ([dataclasses.astuple(x) for x in gw.attempts], [dataclasses.astuple(x) for x in gw.calls],
                      sorted(gw.open_calls.values()))
    assert key(a) == key(b)


def test_burst_calls_are_open_loop_and_only_inside_the_window():
    b = BurstConfig(2.0, 6.0, 50.0)
    gw = simulate(horizon_scenario(m=2, T=10.0, burst=b, w=1000))
    burst_calls = [c for c in gw.calls if c.request[0] == 2]           # source id = n_agents
    assert all(2.0 <= c.t_request < 6.0 for c in burst_calls)
    assert 150 <= len(burst_calls) <= 250                                # ~ 50/s x 4 s, Poisson
    ts = sorted(c.t_request for c in burst_calls)
    assert sum(1 for x, y in zip(ts, ts[1:]) if y - x < 0.2) > 50       # arrivals while earlier calls still run


def test_burst_does_not_change_agent_random_streams():
    """Unloaded, deterministic server: agents' request times must be identical with and without the burst."""
    plain = simulate(horizon_scenario(m=3, T=10.0, w=1000))
    burst = simulate(horizon_scenario(m=3, T=10.0, w=1000, burst=BurstConfig(2.0, 6.0, 30.0)))
    agent_times = lambda gw: sorted((c.request, c.t_request) for c in gw.calls if c.request[0] < 3)
    assert agent_times(plain) == agent_times(burst)


def test_timeouts_in_a_full_run_leave_no_dangling_work():
    sc = horizon_scenario(m=10, T=6.0, w=64, server=scfg(latency_sigma=0.8, capacity=2, degradation=3.0, hard_cap=50),
                          retry=RetryConfig(4, 0.05, 1.0, timeout=0.6))
    gw = simulate(sc)
    assert any(a.status == TO for a in gw.attempts)
    assert gw.server.inflight == 0 and not gw._abandoned and gw.slots.in_use == 0


def test_no_pending_task_destroyed_logs(caplog):
    import gc
    import logging
    caplog.set_level(logging.ERROR, logger="asyncio")
    gc.collect()                 # leftovers of EARLIER tests must not be attributed to this run
    caplog.clear()
    sc = horizon_scenario(m=10, T=6.0, w=64, burst=BurstConfig(1.0, 5.0, 40.0),
                          server=scfg(latency_sigma=0.8, capacity=2, degradation=3.0, hard_cap=50),
                          retry=RetryConfig(4, 0.05, 1.0, timeout=0.6))
    simulate(sc)
    gc.collect()
    assert not [r for r in caplog.records if "destroyed" in r.getMessage() or "never retrieved" in r.getMessage()]



class StatusServer:
    def __init__(self, statuses):
        self.statuses, self.inflight, self.max_inflight_seen = list(statuses), 0, 0

    async def handle(self, request=None):
        await asyncio.sleep(0.01)
        return Response(self.statuses.pop(0) if self.statuses else 200, 0.01)


@pytest.mark.parametrize("status,retried", [(400, False), (404, False), (501, False), (408, True), (429, True), (500, True),
                                            (502, True), (503, True), (504, True)])
def test_only_transient_statuses_are_retried(status, retried):
    """D29 + D33: 408/429/500/502/503/504/client timeout are retried; any other error goes straight back to the agent."""
    gw = Gateway(StatusServer([status]), ACCController(acfg(frozen=True)), RetryConfig(3, 0.05, 1.0), seed=0)
    rec = run_virtual(gw.call(request=(0, 0)))
    if retried:
        assert rec.ok and [a.status for a in gw.attempts] == [status, 200]
    else:
        assert rec.status == status and rec.attempts == 1


def test_client_timeout_code_is_not_an_http_status():
    assert not (100 <= TO <= 599)



def test_exhausted_408_chain_returns_408_after_all_retries():
    gw = Gateway(StatusServer([408] * 10), ACCController(acfg(frozen=True)), RetryConfig(3, 0.05, 1.0), seed=0)
    rec = run_virtual(gw.call(request=(0, 0)))
    assert rec.status == 408 and not rec.ok and rec.attempts == 4
