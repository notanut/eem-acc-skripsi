"""Design aid (NOT a result): check that a candidate server profile produces distinct load regimes.
Run: PYTHONPATH=src python experiments/calibrate_server_profile.py
Profiles are CANDIDATES pending user approval (see HANDOFF.md, open question 5)."""
from eem_acc.acc import ACCConfig
from eem_acc.agent import RetryConfig, WorkloadConfig
from eem_acc.env import ServerConfig
from eem_acc.sim import Scenario, run_scenario

PROFILES = {
    # capacity-bound: rate limit effectively non-binding
    "CAP":  ServerConfig(base_latency=0.2, latency_sigma=0.25, capacity=8, degradation=1.5, hard_cap=16,
                         rate_limit=200.0, burst=100, retry_after=0.5, error_latency=0.02),
    # rate-limit-bound: capacity high, 40 calls/s limit binding
    "RATE": ServerConfig(base_latency=0.2, latency_sigma=0.25, capacity=32, degradation=1.5, hard_cap=64,
                         rate_limit=40.0, burst=20, retry_after=0.5, error_latency=0.02),
}
RETRY = RetryConfig(max_retries=6, backoff_base=0.1, backoff_cap=4.0)

def acc(w):  # frozen window = effectively unconstrained when w=64
    return ACCConfig(alpha=.5, beta=.5, epsilon=.1, w0=w, w_min=1, w_max=64, n_c=16, l_target=0.45, frozen=True)

for name, srv in PROFILES.items():
    m = run_scenario(Scenario(srv, WorkloadConfig(1, 400, ((0.0, 0.0),), 0.0), RETRY, acc(1), 0))["metrics"]
    print(f"\n[{name}] unloaded latency p50={m['lat_p50']:.3f} p95={m['lat_p95']:.3f} p99={m['lat_p99']:.3f}")
    print(f"{'M':>4} {'n_429':>8} {'n_503':>8} {'thr':>7} {'lat_p95':>8} {'goodput':>8} {'inflight':>8}")
    for M in (4, 16, 64):
        m = run_scenario(Scenario(srv, WorkloadConfig(M, 100, ((0.0, 0.1),), 1.0), RETRY, acc(64), 0))["metrics"]
        print(f"{M:>4} {m['n_429']:>8} {m['n_503']:>8} {m['throttle_rate']:>7.3f} {m['lat_p95']:>8.3f} {m['goodput']:>8.1f} {m['server_max_inflight']:>8}")
