"""Server / gateway / simulation tests. All numeric values are TEST-ONLY placeholders,
NOT experimental choices, and nothing here is a result."""

import asyncio
import dataclasses
import time

import pytest

from eem_acc.acc import ACCConfig, ACCController
from eem_acc.agent import Gateway, RetryConfig, SlotPool, WorkloadConfig, run_agents
from eem_acc.env import MockServer, ServerConfig, run_virtual
from eem_acc.metrics import compute_metrics
from eem_acc.sim import Scenario, run_scenario


def scfg(**kw):
    base = dict(base_latency=0.1, latency_sigma=0.0, capacity=4, degradation=1.0, hard_cap=8,
                rate_limit=1000.0, burst=1000, retry_after=0.2, error_latency=0.01)
    base.update(kw)
    return ServerConfig(**base)


def acfg(**kw):
    base = dict(alpha=0.5, beta=0.5, epsilon=0.1, w0=4, w_min=1, w_max=16, n_c=8, l_target=0.15)
    base.update(kw)
    return ACCConfig(**base)


RETRY = RetryConfig(max_retries=5, backoff_base=0.05, backoff_cap=1.0)


# ---------- virtual time ----------
def test_virtual_time_is_fast_and_exact():
    async def main():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.sleep(3600)
        return loop.time() - t0

    start = time.time()
    assert run_virtual(main()) == pytest.approx(3600.0)
    assert time.time() - start < 1.0


def test_virtual_loop_detects_deadlock():
    async def main():
        await asyncio.get_running_loop().create_future()   # never resolved

    with pytest.raises(RuntimeError, match="deadlock"):
        run_virtual(main())


# ---------- server ----------
def test_rate_limit_returns_429_with_retry_after():
    async def main():
        s = MockServer(scfg(burst=2, rate_limit=1.0), seed=0)
        return await asyncio.gather(*[s.handle() for _ in range(4)])

    rs = run_virtual(main())
    assert [r.status for r in rs] == [200, 200, 429, 429]
    assert rs[2].retry_after == 0.2


def test_token_bucket_refills_over_time():
    async def main():
        s = MockServer(scfg(burst=1, rate_limit=1.0), seed=0)
        a = await s.handle()
        b = await s.handle()                  # bucket empty right after
        await asyncio.sleep(1.5)
        c = await s.handle()
        return a.status, b.status, c.status

    # 2nd call arrives 0.1 s after the 1st took the only token (0.1 < 1 token refilled)
    assert run_virtual(main()) == (200, 429, 200)


def test_overload_returns_503_above_hard_cap():
    async def main():
        s = MockServer(scfg(hard_cap=3), seed=0)
        return [r.status for r in await asyncio.gather(*[s.handle() for _ in range(5)])]

    assert run_virtual(main()) == [200, 200, 200, 503, 503]


def test_latency_grows_with_load():
    async def run(n):
        s = MockServer(scfg(capacity=2, degradation=1.0, hard_cap=100), seed=0)
        rs = await asyncio.gather(*[s.handle() for _ in range(n)])
        return max(r.latency for r in rs)

    low = run_virtual(run(2))
    high = run_virtual(run(8))
    assert low == pytest.approx(0.1)
    assert high > low * 2                     # inflight 8 / cap 2 -> factor 1 + 3 = 4


def test_capacity_event_reduces_hard_cap():
    async def main():
        s = MockServer(scfg(hard_cap=8, capacity_events=((5.0, 0.25),)), seed=0)
        before = [r.status for r in await asyncio.gather(*[s.handle() for _ in range(4)])]
        await asyncio.sleep(10)
        after = [r.status for r in await asyncio.gather(*[s.handle() for _ in range(4)])]
        return before, after

    before, after = run_virtual(main())
    assert before == [200] * 4
    assert after == [200, 200, 503, 503]      # hard_cap 8 * 0.25 = 2


# ---------- slot pool / gateway ----------
def test_slot_pool_limit_and_dynamic_resize():
    async def main():
        p = SlotPool(1)
        await p.acquire()
        woke = []

        async def waiter(i):
            await p.acquire(); woke.append(i)

        tasks = [asyncio.ensure_future(waiter(i)) for i in range(2)]
        await asyncio.sleep(0)
        assert woke == []
        p.set_limit(3)                         # growing the window wakes waiters (FIFO)
        await asyncio.sleep(0)
        assert woke == [0, 1] and p.in_use == 3
        await asyncio.gather(*tasks)

    run_virtual(main())


def test_gateway_never_exceeds_window():
    async def main():
        c = ACCController(acfg(frozen=True, w0=3, w_min=1, w_max=3))
        gw = Gateway(MockServer(scfg(), 0), c, RETRY, seed=1)
        await asyncio.gather(*[gw.call() for _ in range(30)])
        return gw

    gw = run_virtual(main())
    assert gw.server.max_inflight_seen <= 3
    assert max(a.in_use for a in gw.attempts) <= 3
    assert all(c.ok for c in gw.calls) and len(gw.calls) == 30


def test_gateway_retries_429_and_counts_attempts():
    async def main():
        c = ACCController(acfg(frozen=True))
        gw = Gateway(MockServer(scfg(burst=1, rate_limit=2.0), 0), c, RETRY, seed=1)
        calls = await asyncio.gather(*[gw.call() for _ in range(3)])
        return gw, calls

    gw, calls = run_virtual(main())
    assert all(c.ok for c in calls)
    assert sum(c.attempts for c in calls) == len(gw.attempts) > 3        # retries happened
    assert any(a.status == 429 for a in gw.attempts)


def test_gateway_gives_up_after_max_retries():
    async def main():
        c = ACCController(acfg(frozen=True))
        gw = Gateway(MockServer(scfg(burst=1, rate_limit=0.0001, retry_after=0.01), 0), c,
                     RetryConfig(max_retries=2, backoff_base=0.01, backoff_cap=0.02), seed=1)
        return await asyncio.gather(*[gw.call() for _ in range(2)])

    a, b = run_virtual(main())
    exhausted = b if a.ok else a
    assert not exhausted.ok and exhausted.status == 429 and exhausted.attempts == 3   # 1 + 2 retries


def test_acc_window_changes_reach_gateway_slots():
    async def main():
        c = ACCController(acfg(n_c=4, w0=8, w_max=8, beta=0.9))
        gw = Gateway(MockServer(scfg(burst=2, rate_limit=0.5), 0), c, RETRY, seed=1)
        await asyncio.gather(*[gw.call() for _ in range(12)])
        return gw

    gw = run_virtual(main())
    assert gw.controller.window < 8
    assert gw.slots.limit == gw.controller.window


# ---------- full scenario ----------
def make_scenario(seed=0, **acc_kw):
    return Scenario(
        server=scfg(),
        workload=WorkloadConfig(n_agents=12, calls_per_agent=15, phases=((0.0, 0.0),), start_jitter=0.5),
        retry=RETRY, acc=acfg(**acc_kw), seed=seed)


def test_scenario_is_deterministic():
    a = run_scenario(make_scenario(seed=3))
    b = run_scenario(make_scenario(seed=3))
    c = run_scenario(make_scenario(seed=4))
    assert a == b
    assert a != c


def test_scenario_metric_invariants():
    m = run_scenario(make_scenario())["metrics"]
    assert m["n_attempts"] == m["n_200"] + m["n_429"] + m["n_503"]
    assert m["n_calls"] == 12 * 15
    assert m["n_calls_ok"] + m["n_exhausted_calls"] == m["n_calls"]
    assert m["n_200"] >= m["n_calls_ok"]
    assert m["throttle_rate"] == pytest.approx((m["n_429"] + m["n_503"]) / m["n_attempts"])
    assert m["server_max_inflight"] <= m["window_max"]       # in-flight never exceeds the window
    assert 1 <= m["window_min"] <= m["window_max"] <= 16     # bounds from ACCConfig


def test_scenario_config_is_json_serialisable():
    import json
    json.dumps(dataclasses.asdict(make_scenario()))
