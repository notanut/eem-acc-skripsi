"""Stage-2 SMOKE TEST (not an experiment, not a result).

All parameter values are arbitrary PLACEHOLDERS used only to check that the
simulator, gateway and controller behave sensibly end to end. They are NOT the
hyperparameters or scenarios of the paper (those are still to be confirmed).
Do not cite these numbers anywhere.

Run:  PYTHONPATH=src python experiments/smoke_stage2.py
"""

from eem_acc.acc import ACCConfig
from eem_acc.agent import RetryConfig, WorkloadConfig
from eem_acc.env import ServerConfig
from eem_acc.sim import Scenario, run_scenario

SERVER = ServerConfig(base_latency=0.2, latency_sigma=0.2, capacity=8, degradation=1.5, hard_cap=16,
                      rate_limit=60.0, burst=20, retry_after=0.5, error_latency=0.02)
WORKLOAD = WorkloadConfig(n_agents=32, calls_per_agent=40, phases=((0.0, 0.1),), start_jitter=1.0)
RETRY = RetryConfig(max_retries=6, backoff_base=0.1, backoff_cap=4.0)
ACC = dict(alpha=0.5, beta=0.5, epsilon=0.1, w_min=1, w_max=32, n_c=16, l_target=0.3)

METHODS = {
    "fixed W=4":  dict(w0=4, frozen=True),
    "fixed W=16": dict(w0=16, frozen=True),
    "fixed W=32": dict(w0=32, frozen=True),
    "ACC (w0=8)": dict(w0=8),
}

cols = ["n_429", "n_503", "throttle_rate", "lat_p95", "slo_violation_rate", "goodput", "makespan",
        "window_mean", "window_final", "n_exhausted_calls"]
print(f"{'method':<12}" + "".join(f"{c:>19}" for c in cols))
for name, kw in METHODS.items():
    sc = Scenario(SERVER, WORKLOAD, RETRY, ACCConfig(**{**ACC, **kw}), seed=0)
    m = run_scenario(sc)["metrics"]
    print(f"{name:<12}" + "".join(f"{(round(m[c], 3) if isinstance(m[c], float) else m[c])!s:>19}" for c in cols))
