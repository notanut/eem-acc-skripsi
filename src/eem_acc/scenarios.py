"""RQ2 scenarios, methods and grids: design v2, approved by the user 2026-11-14 (docs/rq2_design_v2.md,
decisions D16-D23). v1 (D10-D15) is in git history. Changing any value here changes the experiment: do it only
with the user's approval and record it in HANDOFF.md (SIM_VERSION in sim.py must be bumped for semantic changes).
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

from .acc import ACCConfig
from .agent import BurstConfig, PacerConfig, RetryConfig, WorkloadConfig
from .env import ServerConfig
from .metrics import AnalysisConfig
from .sim import Scenario

# ---- SLO and workload (D12, D21) -------------------------------------------
L_TARGET = 0.45                      # L_SLO: 1.5 x unloaded server p95 (0.303 s, measured in v1)
HORIZON = 120.0                      # s of simulated time per run
THINK_MEAN = 0.1                     # s, exponential
START_JITTER = 1.0                   # s
RETRY = RetryConfig(max_retries=6, backoff_base=0.1, backoff_cap=4.0, timeout=1.0)
SUCCESS_GRACE = 20.0                 # s; success over calls requested before T - 20 s (implementation finding,
                                     # pending the user's confirmation: see HANDOFF / report 2026-11-14)

# ---- server regimes and scenarios (D16, D18) ---------------------------------
_COMMON = dict(base_latency=0.2, latency_sigma=0.25, degradation=1.5, retry_after=0.5, error_latency=0.02)
REGIMES = {
    "CAP_SMALL": dict(capacity=4, hard_cap=8, rate_limit=1000.0, burst=500),
    "RATE":      dict(capacity=48, hard_cap=96, rate_limit=100.0, burst=20),
    "CAP_LARGE": dict(capacity=24, hard_cap=48, rate_limit=1000.0, burst=500),
}
SCENARIOS: dict[str, dict[str, Any]] = {
    "CAP-steady":  dict(regime="CAP_SMALL", capacity_events=(), burst=None, analysis=AnalysisConfig()),
    "RATE-steady": dict(regime="RATE", capacity_events=(), burst=None, analysis=AnalysisConfig()),
    "CAP-drop":    dict(regime="CAP_LARGE", capacity_events=((40.0, 1 / 3), (80.0, 1.0)), burst=None,
                        analysis=AnalysisConfig(windows=(("pre", 20.0, 40.0), ("during", 40.0, 80.0),
                                                         ("post", 80.0, 100.0)),
                                                event=(40.0, 80.0), pre=(20.0, 40.0))),
    "CAP-burst":   dict(regime="CAP_LARGE", capacity_events=(), burst=BurstConfig(40.0, 60.0, 80.0),
                        analysis=AnalysisConfig(windows=(("pre", 20.0, 40.0), ("during", 40.0, 60.0),
                                                         ("post", 60.0, 80.0)),
                                                event=(40.0, 60.0), pre=(20.0, 40.0))),
}
M_LEVELS = (8, 32, 96)

# ---- loaded cells, fixed a priori from offered load (D20, spec section 5) ---------
LOAD_MARGIN = 1.2


def offered_load(scenario: str, m_agents: int) -> dict[str, float]:
    """Peak offered concurrency c = M * L0 / (L0 + think) (+ burst rate * L0) and offered rate c / L0, together with
    the tightest capacity and the rate limit, over the run."""
    sc = SCENARIOS[scenario]
    reg = REGIMES[sc["regime"]]
    l0 = _COMMON["base_latency"]
    c = m_agents * l0 / (l0 + THINK_MEAN)
    if sc["burst"] is not None:
        c += sc["burst"].rate * l0
    scales = [1.0] + [s for _, s in sc["capacity_events"]]
    # peak burst load is combined with the tightest capacity: only valid while no scenario has both
    assert not (sc["burst"] is not None and sc["capacity_events"]), "offered_load: burst + capacity events"
    return {"concurrency": c, "rate": c / l0, "min_capacity": reg["capacity"] * min(scales),
            "rate_limit": reg["rate_limit"]}


def is_loaded(scenario: str, m_agents: int) -> bool:
    o = offered_load(scenario, m_agents)
    return o["concurrency"] >= LOAD_MARGIN * o["min_capacity"] or o["rate"] >= LOAD_MARGIN * o["rate_limit"]


def loaded_cells() -> list[tuple[str, int]]:
    return [(s, m) for s in SCENARIOS for m in M_LEVELS if is_loaded(s, m)]


# ---- methods (D13, D14, D21, D22) ----------------------------------------------
ACC_FIXED = dict(w0=8, w_min=1, w_max=128)
ACC_GRID = dict(alpha=(0.25, 0.5, 0.75), beta=(0.25, 0.5, 0.75), epsilon=(0.05, 0.1, 0.2), n_c=(8, 16, 32),
                kappa=(0.8, 0.9, 1.0))
ACC_START = dict(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, kappa=0.9)    # centre of the grid (illustrations only)
FIXED_W_REPORTED = (4, 16, 64)
FIXED_W_DEV = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96)
PACER_RATES_DEV = (10.0, 14.0, 15.0, 16.0, 17.0, 18.0, 22.0, 27.0, 33.0, 40.0, 50.0, 65.0, 80.0, 100.0, 130.0)
PACER_BURST = 1
UNCONSTRAINED_W = 128
ABLATIONS = {
    "acc_no429":    dict(use_throttle_branch=False),
    "acc_nolat":    dict(use_latency_branch=False),
    "acc_additive": dict(multiplicative_decrease=False),
}

# ---- protocol (D15, D20) -------------------------------------------------------
DEV_SEEDS = tuple(range(0, 5))
TEST_SEEDS = tuple(range(100, 110))
MAX_SLO_VIOLATION = 0.05
MIN_CALL_SUCCESS = 0.99
MAX_THROTTLE_RATE = 0.10
TIE_TOLERANCE = 0.005                # candidates with score >= (1 - tol) * best are tied


# ---- method specs (plain dicts -> JSON-able, named via tag) --------------------
def acc_method(alpha: float, beta: float, epsilon: float, n_c: int, kappa: float = 1.0, **flags: Any) -> dict:
    return dict(kind="acc", alpha=alpha, beta=beta, epsilon=epsilon, n_c=n_c, kappa=kappa, **flags)


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
    return (f"acc_a{m['alpha']:g}_b{m['beta']:g}_e{m['epsilon']:g}_n{m['n_c']}"
            f"_k{m.get('kappa', 1.0):g}{flags}")


def unit_id(scenario: str, m_agents: int, method: dict, seed: int) -> str:
    return f"{scenario}__M{m_agents}__{method_tag(method)}__s{seed}"


def build(scenario: str, m_agents: int, method: dict, seed: int) -> Scenario:
    sc = SCENARIOS[scenario]
    server = ServerConfig(**_COMMON, **REGIMES[sc["regime"]], capacity_events=sc["capacity_events"])
    wl = WorkloadConfig(n_agents=m_agents, calls_per_agent=None, phases=((0.0, THINK_MEAN),),
                        start_jitter=START_JITTER, horizon=HORIZON, burst=sc["burst"])
    k = method["kind"]
    pacer = None
    if k == "acc":
        flags = {f: method[f] for f in ("use_throttle_branch", "use_latency_branch", "multiplicative_decrease")
                 if f in method}
        acc = ACCConfig(alpha=method["alpha"], beta=method["beta"], epsilon=method["epsilon"], n_c=method["n_c"],
                        kappa=method.get("kappa", 1.0), l_target=L_TARGET, **ACC_FIXED, **flags)
    elif k == "fixed":
        w = method["w"]
        acc = ACCConfig(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, l_target=L_TARGET,
                        w0=w, w_min=1, w_max=max(UNCONSTRAINED_W, w), frozen=True)
    elif k == "tokenbucket":
        acc = ACCConfig(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, l_target=L_TARGET,
                        w0=UNCONSTRAINED_W, w_min=1, w_max=UNCONSTRAINED_W, frozen=True)
        pacer = PacerConfig(rate=method["rate"], burst=PACER_BURST)
    else:
        raise ValueError(f"unknown method kind: {k}")
    analysis = dataclasses.replace(sc["analysis"], success_grace=SUCCESS_GRACE)
    return Scenario(server=server, workload=wl, retry=RETRY, acc=acc, seed=seed, pacer=pacer, analysis=analysis)


# ---- phase unit lists -----------------------------------------------------------
def acc_grid_methods() -> list[dict]:
    keys = list(ACC_GRID)
    return [acc_method(**dict(zip(keys, vals))) for vals in itertools.product(*ACC_GRID.values())]


def dev_methods() -> list[dict]:
    return acc_grid_methods() + [fixed_method(w) for w in FIXED_W_DEV] + [pacer_method(r) for r in PACER_RATES_DEV]


def test_methods(frozen: dict) -> list[dict]:
    """Test-phase methods from the frozen dev selection: frozen ACC; the same ACC with kappa = 1 (the paper's base
    algorithm, D22); 3 ablations; fixed {4,16,64}; the global best fixed W; the per-scenario best fixed W values
    (robustness matrix, D16); the tuned token bucket. Duplicates are removed (order kept)."""
    a = frozen["acc"]
    ms = [acc_method(**a)]
    if a.get("kappa", 1.0) != 1.0:
        ms.append(acc_method(**{**a, "kappa": 1.0}))
    ms += [acc_method(**a, **flags) for flags in ABLATIONS.values()]
    ws = list(FIXED_W_REPORTED) + [frozen["best_fixed_w"]] + sorted(set(frozen["best_fixed_w_per_scenario"].values()))
    ms += [fixed_method(w) for w in ws]
    ms.append(pacer_method(frozen["best_pacer_rate"]))
    seen, out = set(), []
    for m in ms:
        t = method_tag(m)
        if t not in seen:
            seen.add(t)
            out.append(m)
    return out


def units(methods: list[dict], seeds: tuple[int, ...]) -> list[tuple[str, str, int, dict, int]]:
    """(unit_id, scenario, M, method, seed) in a fixed order (light cells first within a seed)."""
    out = []
    for seed in seeds:
        for m in M_LEVELS:
            for scn in SCENARIOS:
                for meth in methods:
                    out.append((unit_id(scn, m, meth, seed), scn, m, meth, seed))
    return out
