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
from .metrics import AnalysisConfig, compute_metrics


# Bump whenever simulator/gateway/controller/metrics semantics change: it is part of every result's stamp,
# so results produced by different code can never be silently mixed in one results folder.
SIM_VERSION = "5"   # 2: shrink-before-release, grant-time window logging, keyed (CRN) randomness, J metrics
                    # 3: server noise keyed by logical call (agent, k) only (not the attempt index);
                    #    keyed draws served from per-agent cached tables (eem_acc/keyed.py)
                    # 4: design v2: horizon workload, open-loop burst, client timeouts (= overload),
                    #    kappa, noise keyed by (agent, k, admission), success over requested calls
                    # 5: client timeout = status.CLIENT_TIMEOUT (not 504); only status.RETRYABLE is retried;
                    #    timeouts are latency samples at T_out; no latency sample -> hold W (D29, D30)


@dataclass(frozen=True)
class Scenario:
    server: ServerConfig
    workload: WorkloadConfig
    retry: RetryConfig
    acc: ACCConfig
    seed: int
    pacer: PacerConfig | None = None   # client-side token-bucket baseline (None = off)
    analysis: AnalysisConfig | None = None   # v2 event windows / recovery (metrics only)


async def _simulate(sc: Scenario) -> Gateway:
    ss = np.random.SeedSequence(sc.seed).spawn(3)
    seeds = [int(s.generate_state(1)[0]) for s in ss]
    server = MockServer(sc.server, seeds[0])
    ctrl = ACCController(sc.acc)
    gw = Gateway(server, ctrl, sc.retry, seeds[1], pacer=sc.pacer)
    await run_agents(gw, sc.workload, seeds[2])
    return gw


async def _run(sc: Scenario) -> dict[str, Any]:
    gw = await _simulate(sc)
    return {
        "metrics": compute_metrics(gw, sc.acc.l_target, sc.workload.horizon, sc.analysis),
        "window_trace": [list(x) for x in gw.controller.trace],
    }


def simulate(sc: Scenario) -> Gateway:
    """Run a scenario and return the finished Gateway (attempt/call logs) for plotting."""
    return run_virtual(_simulate(sc))


def run_scenario(sc: Scenario) -> dict[str, Any]:
    return run_virtual(_run(sc))
