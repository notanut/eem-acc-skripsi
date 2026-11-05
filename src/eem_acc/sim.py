"""One simulated scenario run: server + gateway(ACC) + scripted agents, virtual time.

`Scenario` is a plain dataclass tree, so `dataclasses.asdict(scenario)` is a JSON-able
config for `run_resumable` (one unit = one (scenario, method, seed) run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .acc import ACCConfig, ACCController
from .agent import Gateway, PacerConfig, RetryConfig, WorkloadConfig, run_agents
from .env import MockServer, ServerConfig, run_virtual
from .metrics import compute_metrics


@dataclass(frozen=True)
class Scenario:
    server: ServerConfig
    workload: WorkloadConfig
    retry: RetryConfig
    acc: ACCConfig
    seed: int
    pacer: PacerConfig | None = None   # client-side token-bucket baseline (None = off)


async def _run(sc: Scenario) -> dict[str, Any]:
    ss = np.random.SeedSequence(sc.seed).spawn(3)
    seeds = [int(s.generate_state(1)[0]) for s in ss]
    server = MockServer(sc.server, seeds[0])
    ctrl = ACCController(sc.acc)
    gw = Gateway(server, ctrl, sc.retry, seeds[1], pacer=sc.pacer)
    await run_agents(gw, sc.workload, seeds[2])
    return {
        "metrics": compute_metrics(gw, sc.acc.l_target),
        "window_trace": [list(x) for x in ctrl.trace],
    }


def run_scenario(sc: Scenario) -> dict[str, Any]:
    return run_virtual(_run(sc))
