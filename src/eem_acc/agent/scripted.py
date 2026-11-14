"""Scripted (non-LLM) agents for isolating the ACC (RQ2) cheaply.

Each agent runs a sequential loop: think -> API call through the gateway -> repeat. Concurrency arises across
the M agents. `phases` gives the mean think time as a function of simulated time.

Two run modes:
* `calls_per_agent`: each agent makes exactly that many calls; the run ends when all are done (design v1).
* `horizon` (design v2, D21): agents keep calling until simulated time T = horizon; at T the run STOPS: every
  agent, burst call and in-flight attempt is cancelled (calls requested but unfinished stay in
  `Gateway.open_calls` and count as not successful), and abandoned server work is cancelled.
Optional open-loop `burst` (design v2, D18): an extra source issues calls at Poisson times with rate `rate`
during [t_start, t_end), regardless of completions. Its calls use request ids (n_agents, j).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from .gateway import Gateway


@dataclass(frozen=True)
class BurstConfig:
    t_start: float
    t_end: float
    rate: float              # extra calls per second during [t_start, t_end)

    def __post_init__(self) -> None:
        if not (0 <= self.t_start < self.t_end) or self.rate <= 0:
            raise ValueError("need 0 <= t_start < t_end and rate > 0")


@dataclass(frozen=True)
class WorkloadConfig:
    n_agents: int
    calls_per_agent: int | None   # v1 mode (None in horizon mode)
    phases: tuple                 # ((t_start, think_time_mean_s), ...) sorted by t_start
    start_jitter: float           # agents start at Uniform(0, start_jitter) s
    horizon: float | None = None  # v2 mode: run length T in simulated seconds
    burst: BurstConfig | None = None

    def __post_init__(self) -> None:
        if (self.calls_per_agent is None) == (self.horizon is None):
            raise ValueError("set exactly one of calls_per_agent (v1) or horizon (v2)")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.burst is not None and self.horizon is None:
            raise ValueError("an open-loop burst needs horizon mode")


def _think_mean(phases: tuple, now: float) -> float:
    mean = phases[0][1]
    for t, m in phases:
        if t <= now:
            mean = m
    return mean


async def _agent(gw: Gateway, wl: WorkloadConfig, rng: np.random.Generator, agent_id: int) -> None:
    loop = asyncio.get_running_loop()
    if wl.start_jitter > 0:
        await asyncio.sleep(float(rng.uniform(0, wl.start_jitter)))
    k = 0
    while wl.horizon is not None or k < wl.calls_per_agent:
        u = float(rng.exponential(1.0))                 # ALWAYS one draw per call: the stream never depends on timing
        mean = _think_mean(wl.phases, loop.time())
        if mean > 0:
            await asyncio.sleep(mean * u)
        if wl.horizon is not None and loop.time() >= wl.horizon:
            return
        await gw.call(request=(agent_id, k))
        k += 1


async def _burst_source(gw: Gateway, b: BurstConfig, rng: np.random.Generator, source_id: int,
                        spawned: list[asyncio.Task]) -> None:
    loop = asyncio.get_running_loop()
    if b.t_start > loop.time():
        await asyncio.sleep(b.t_start - loop.time())
    j = 0
    while True:
        await asyncio.sleep(float(rng.exponential(1.0 / b.rate)))
        if loop.time() >= b.t_end:
            return
        spawned.append(asyncio.ensure_future(gw.call(request=(source_id, j))))
        j += 1


async def run_agents(gw: Gateway, wl: WorkloadConfig, seed: int) -> None:
    # spawn(n_agents + 1): children 0..n-1 are the same streams as spawn(n_agents), child n feeds the burst
    children = np.random.SeedSequence(seed).spawn(wl.n_agents + 1)
    agents = [asyncio.ensure_future(_agent(gw, wl, np.random.default_rng(c), i))
              for i, c in enumerate(children[:wl.n_agents])]
    if wl.horizon is None:
        await asyncio.gather(*agents)
        return
    spawned: list[asyncio.Task] = []
    sources = []
    if wl.burst is not None:
        sources.append(asyncio.ensure_future(
            _burst_source(gw, wl.burst, np.random.default_rng(children[wl.n_agents]), wl.n_agents, spawned)))
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0.0, wl.horizon - loop.time()))
    everything = agents + sources + spawned
    for t in everything:
        t.cancel()
    results = await asyncio.gather(*everything, return_exceptions=True)
    for r in results:                                   # a real error must not be hidden by the cancellation
        if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
            raise r
    await gw.shutdown()
