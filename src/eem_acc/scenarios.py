"""RQ2 scenarios, methods and grids exactly as approved by the user (2026-11-05,
see docs/stage2_design_proposal.md, D10-D15). Changing any value here changes the
experiment: do it only with the user's approval and record it in HANDOFF.md.

Implementation assumptions flagged to the user:
* token-bucket baseline burst = 1 (pure pacing at `rate`);
* all methods use the same retry policy and workload.
"""

from __future__ import annotations

import itertools
from typing import Any

from .acc import ACCConfig
from .agent import PacerConfig, RetryConfig, WorkloadConfig
from .env import ServerConfig
from .sim import Scenario

# ---- D12 -------------------------------------------------------------------
L_TARGET = 0.45                      # 1.5 x unloaded server p95 (0.303 s, measured)

# ---- D10 / D11 -------------------------------------------------------------
_COMMON = dict(base_latency=0.2, latency_sigma=0.25, degradation=1.5, retry_after=0.5, error_latency=0.02)
SERVER_REGIMES = {
    "CAP":  dict(capacity=8, hard_cap=16, rate_limit=200.0, burst=100),
    "RATE": dict(capacity=32, hard_cap=64, rate_limit=40.0, burst=20),
}
# name -> (regime, think-time phases, capacity events)
SCENARIOS = {
    "CAP-steady":  ("CAP", ((0.0, 0.1),), ()),
    "CAP-burst":   ("CAP", ((0.0, 0.1), (10.0, 0.0), (15.0, 0.1)), ()),
    "CAP-drop":    ("CAP", ((0.0, 0.1),), ((20.0, 0.5), (40.0, 1.0))),
    "RATE-steady": ("RATE", ((0.0, 0.1),), ()),
}
M_LEVELS = (4, 16, 64)
CALLS_PER_AGENT = 150
START_JITTER = 1.0
RETRY = RetryConfig(max_retries=6, backoff_base=0.1, backoff_cap=4.0)

# ---- D13 -------------------------------------------------------------------
ACC_FIXED = dict(w0=8, w_min=1, w_max=64)
ACC_GRID = dict(alpha=(0.25, 0.5, 0.75), beta=(0.25, 0.5, 0.75), epsilon=(0.05, 0.1, 0.2), n_c=(8, 16, 32))
ACC_START = dict(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16)     # start values (centre of the grid)

# ---- D14 -------------------------------------------------------------------
FIXED_W_REPORTED = (4, 16, 64)                 # low / mid / high
FIXED_W_DEV = (1, 2, 4, 8, 16, 32, 64)         # candidates for "best global fixed W"
PACER_RATES_DEV = (10.0, 20.0, 40.0, 80.0)     # candidates for the token-bucket baseline
PACER_BURST = 1                                # assumption (see module docstring)
ABLATIONS = {
    "acc_no429":    dict(use_throttle_branch=False),
    "acc_nolat":    dict(use_latency_branch=False),
    "acc_additive": dict(multiplicative_decrease=False),
}

# ---- D15 -------------------------------------------------------------------
DEV_SEEDS = tuple(range(0, 5))
TEST_SEEDS = tuple(range(100, 110))
MAX_SLO_VIOLATION = 0.05
MIN_CALL_SUCCESS = 0.99


# ---- method specs (plain dicts -> JSON-able, hashable via tag) -------------
def acc_method(alpha: float, beta: float, epsilon: float, n_c: int, **flags: Any) -> dict:
    return dict(kind="acc", alpha=alpha, beta=beta, epsilon=epsilon, n_c=n_c, **flags)


def fixed_method(w: int) -> dict:
    return dict(kind="fixed", w=int(w))


def pacer_method(rate: float) -> dict:
    return dict(kind="tokenbucket", rate=float(rate))


def method_tag(m: dict) -> str:
    k = m["kind"]
    if k == "fixed":
        return f"fixed_w{m['w']}"
    if k == "tokenbucket":
        return f"tb_r{m['rate']:g}"
    flags = "_no429" if m.get("use_throttle_branch") is False else ""
    flags += "_nolat" if m.get("use_latency_branch") is False else ""
    flags += "_additive" if m.get("multiplicative_decrease") is False else ""
    return f"acc_a{m['alpha']:g}_b{m['beta']:g}_e{m['epsilon']:g}_n{m['n_c']}{flags}"


def unit_id(scenario: str, m_agents: int, method: dict, seed: int) -> str:
    return f"{scenario}__M{m_agents}__{method_tag(method)}__s{seed}"


def build(scenario: str, m_agents: int, method: dict, seed: int) -> Scenario:
    regime, phases, events = SCENARIOS[scenario]
    server = ServerConfig(**_COMMON, **SERVER_REGIMES[regime], capacity_events=events)
    wl = WorkloadConfig(n_agents=m_agents, calls_per_agent=CALLS_PER_AGENT, phases=phases,
                        start_jitter=START_JITTER)
    k = method["kind"]
    pacer = None
    if k == "acc":
        flags = {f: method[f] for f in ("use_throttle_branch", "use_latency_branch", "multiplicative_decrease")
                 if f in method}
        acc = ACCConfig(alpha=method["alpha"], beta=method["beta"], epsilon=method["epsilon"],
                        n_c=method["n_c"], l_target=L_TARGET, **ACC_FIXED, **flags)
    elif k == "fixed":
        w = method["w"]
        acc = ACCConfig(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, l_target=L_TARGET,
                        w0=w, w_min=1, w_max=max(64, w), frozen=True)
    elif k == "tokenbucket":
        acc = ACCConfig(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, l_target=L_TARGET,
                        w0=64, w_min=1, w_max=64, frozen=True)   # window effectively unconstrained
        pacer = PacerConfig(rate=method["rate"], burst=PACER_BURST)
    else:
        raise ValueError(f"unknown method kind: {k}")
    return Scenario(server=server, workload=wl, retry=RETRY, acc=acc, seed=seed, pacer=pacer)


# ---- phase unit lists --------------------------------------------------------
def acc_grid_methods() -> list[dict]:
    keys = list(ACC_GRID)
    return [acc_method(**dict(zip(keys, vals))) for vals in itertools.product(*ACC_GRID.values())]


def dev_methods() -> list[dict]:
    return acc_grid_methods() + [fixed_method(w) for w in FIXED_W_DEV] + [pacer_method(r) for r in PACER_RATES_DEV]


def test_methods(frozen: dict) -> list[dict]:
    """Methods of the test phase, from the frozen dev selection (configs/rq2_frozen.json)."""
    a = frozen["acc"]
    ms = [acc_method(**a)]
    ms += [acc_method(**a, **flags) for flags in ABLATIONS.values()]
    ms += [fixed_method(w) for w in FIXED_W_REPORTED]
    best_w = frozen["best_fixed_w"]
    if best_w not in FIXED_W_REPORTED:
        ms.append(fixed_method(best_w))
    ms.append(pacer_method(frozen["best_pacer_rate"]))
    return ms


def units(methods: list[dict], seeds: tuple[int, ...]) -> list[tuple[str, str, int, dict, int]]:
    """(unit_id, scenario, M, method, seed) in a fixed order (cheap/light cells first)."""
    out = []
    for seed in seeds:
        for scn in SCENARIOS:
            for m in M_LEVELS:
                for meth in methods:
                    out.append((unit_id(scn, m, meth, seed), scn, m, meth, seed))
    return out
