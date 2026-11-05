# RQ2 experiment design proposal (DRAFT, pending user approval)

Nothing here is decided. Calibration evidence: `experiments/calibrate_server_profile.py` (design aid, not a result).

## D10. Server regimes and scenarios
Both regimes: base latency 0.2 s, lognormal sigma 0.25 (measured unloaded server latency p50 = 0.200 s, p95 = 0.303 s), degradation 1.5, 429 retry-after 0.5 s, error latency 0.02 s.

| Regime | capacity C | hard cap (503) | rate limit | what it exercises |
|---|---|---|---|---|
| CAP (capacity-bound) | 8 | 16 | 200/s, burst 100 (non-binding) | latency branch, 503 |
| RATE (rate-limit-bound) | 32 | 64 | 40/s, burst 20 | 429 branch |

Calibration with an unconstrained window (think time 0.1 s, one seed; NOT a method comparison):
- CAP: M=4 clean; M=16 p95 latency 0.619 s, no throttling; M=64 73% throttled (all 503).
- RATE: M=4 clean; M=16 12.8% throttled (429); M=64 64.7% throttled (429).

Options:
- **A (recommended):** 4 scenarios: CAP-steady, CAP-burst, CAP-capacity-drop (x0.5 at t=20 s, recovery to x1.0 at t=40 s), RATE-steady; each with M in {4, 16, 64}.
- B: A plus RATE-burst (5 scenarios).
- C: one combined server (both limits binding) with the same three patterns. Simpler but cannot attribute effects to a branch.

## D11. Workload (scripted agents)
Mean think time 0.1 s (exponential); 150 calls per agent; start jitter 1 s; burst = think time 0 for t in [10, 15) s. Retry: max_retries 6, backoff base 0.1 s, cap 4 s (fixed, not tuned).
Alternatives: fewer calls per agent (shorter, noisier), or think time 0.05 s (heavier). Recommend the values above.

## D12. L_target
Recommend **0.45 s = 1.5 x unloaded server p95** (0.303 s). Alternatives: 1.25 x (0.38 s, stricter) or 2 x (0.61 s, looser).

## D13. ACC hyperparameters
Fixed (not tuned): W0 = 8, W_min = 1, W_max = 64 (equals the largest M, so M never caps the window).
Dev grid (start value in bold), 3^4 = 81 configurations:
- alpha in {0.25, **0.5**, 0.75}; beta in {0.25, **0.5**, 0.75}; epsilon in {0.05, **0.1**, 0.2}; n_c in {8, **16**, 32}.
Dev budget: 81 x 4 scenarios x 3 M x 5 seeds = 4,860 simulated runs (each well under a second; resumable).

## D14. Baselines
- Fixed W: low/mid/high = {4, 16, 64} (64 = effectively unconstrained), plus **best global fixed W** tuned on dev from {1, 2, 4, 8, 16, 32, 64} (a strong baseline, so ACC is not compared only against strawmen).
- Client-side token bucket + exponential backoff (window unconstrained), rate tuned on dev from {10, 20, 40, 80}/s. (Needs a small pacer added to the gateway.)
- ACC ablations via flags: no 429 branch (= latency-only), no latency branch (= 429-only), additive-only.
Test grid: 4 scenarios x 3 M x 9 methods x 10 seeds = 1,080 runs.

## D15. Protocol and selection criterion
- Dev seeds 0-4 for all tuning; test seeds 100-109; the selected configuration is committed to the repo (frozen) before any test run. One global ACC configuration for all scenarios (no per-scenario tuning).
- **Caveat on the objective J:** J penalises latency excess and throttling but does not reward throughput, so for scripted agents (success is about 100%) J alone is maximised by a tiny window. Selection options:
  - (a) maximise mean J only (risk: selects a collapsed window);
  - **(b, recommended)** maximise goodput subject to constraints: SLO-violation rate <= 5% and call success >= 99%, averaged over dev scenarios; report J (lambda = gamma = 0.5 as primary, 1.0 as sensitivity) and all components on the test seeds;
  - (c) report the whole Pareto front (goodput vs SLO violation vs throttle rate) with no single winner.
- RQ4 (sensitivity): one-factor sweeps of alpha, beta, epsilon, n_c, W0, W_max around the frozen configuration on the test seeds.

## Not decided here
EEM hyperparameters (k = 3, k_s, theta_link), LLM-agent task suite and fault rates (Stage 3).
