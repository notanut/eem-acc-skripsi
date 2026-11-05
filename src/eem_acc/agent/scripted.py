"""Scripted (non-LLM) agents for isolating the ACC (RQ2) cheaply.

Each agent runs a sequential loop: think -> API call through the gateway ->
repeat, `calls_per_agent` times. Concurrency arises across the M agents.
`phases` gives the mean think time as a function of simulated time, which lets a
scenario switch between steady load and bursts (think time 0 = back-to-back).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from .gateway import Gateway


@dataclass(frozen=True)
class WorkloadConfig:
    n_agents: int
    calls_per_agent: int
    phases: tuple            # ((t_start, think_time_mean_s), ...) sorted by t_start
    start_jitter: float      # agents start at Uniform(0, start_jitter) s


def _think_mean(phases: tuple, now: float) -> float:
    mean = phases[0][1]
    for t, m in phases:
        if t <= now:
            mean = m
    return mean


async def _agent(gw: Gateway, wl: WorkloadConfig, rng: np.random.Generator) -> None:
    loop = asyncio.get_running_loop()
    if wl.start_jitter > 0:
        await asyncio.sleep(float(rng.uniform(0, wl.start_jitter)))
    for _ in range(wl.calls_per_agent):
        mean = _think_mean(wl.phases, loop.time())
        if mean > 0:
            await asyncio.sleep(float(rng.exponential(mean)))
        await gw.call()


async def run_agents(gw: Gateway, wl: WorkloadConfig, seed: int) -> None:
    children = np.random.SeedSequence(seed).spawn(wl.n_agents)
    await asyncio.gather(*[_agent(gw, wl, np.random.default_rng(c)) for c in children])
