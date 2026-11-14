# RQ2 design v2 (revision of D10-D15 after review) — DRAFT for approval

_Status: proposal, 2026-11-14. Implements the user's decisions D16-D19. Nothing here is implemented yet.
Numbers in section 9 are CALIBRATION evidence from a scratch prototype on dev seeds 0-1 only (test seeds 100-109
were never used); they are design aids, not results, and must not be cited._

## 0. Why a v2
Review of v1 (2026-11-14) found: (H1) every regime had a throughput-optimal fixed window of about 8 = W0, so a fixed
W=8 won everywhere and the design could not show the value of adapting; (H2) the SLO constraint averaged over all
cells hid ~10% violation in loaded cells; (H3) M=4 and the closed-loop burst did not discriminate; (M1) events at
absolute times diluted unequally because run length depended on the method; (M2) coarse token-bucket grid;
(M3) normalisation by a possibly infeasible method; no timeouts although the paper's retry policy has them.
User decisions: D16 (different optimal W per scenario + cross-regime robustness test), D17 (constraints per scenario
over loaded cells + Pareto front), D18 (open-loop burst, M in {8,32,96}), D19 (event-window metrics, finer
token-bucket grid, normalise by best feasible method, client timeouts).

## 1. Workload: fixed run duration (NEW, decision Q1)
- Each run lasts a fixed horizon **T = 120 s** of simulated time instead of 150 calls per agent. Agents (closed
  loop: think, call, repeat) stop issuing new calls at T. Think time exponential, mean 0.1 s; start jitter U(0,1) s.
- Metrics count calls completed by T and attempts dispatched before T. Work still in flight at T is drained and
  excluded; abandoned server-side work (after client timeouts) is cancelled when the run ends.
- Why: every method and every M now sees the events at the same times and for the same share of the run (fixes M1),
  and goodput = successful calls / T is directly comparable across methods.

## 2. Server regimes and scenarios (D16, D18)
Common: base latency 0.2 s, lognormal sigma 0.25, degradation 1.5 (latency x (1 + 1.5 * max(0, inflight/capacity - 1))),
Retry-After 0.5 s on 429, error latency 0.02 s. Note: beyond capacity, throughput FALLS (congestion collapse), so
"more concurrency" is genuinely harmful in capacity-bound regimes.

| scenario | regime | capacity / hard cap | rate limit (burst) | event | best fixed W (calibration) |
|---|---|---|---|---|---|
| CAP-steady | small capacity | 4 / 8 | none (1000/s) | - | 4 |
| RATE-steady | rate-limited | 48 / 96 | 100/s (20) | - | 24-32 |
| CAP-drop | large capacity | 24 / 48 | none | capacity and hard cap x 1/3 during [40, 80) s | 24 before/after, 8 during; best feasible single W = 8 |
| CAP-burst | large capacity | 24 / 48 | none | open-loop Poisson burst, 80 extra calls/s during [40, 60) s | 24 |

- **Open-loop burst:** a separate source issues calls at Poisson times regardless of completions (each call goes
  through the same gateway, window and retry policy; request ids (M, j) keep CRN pairing). Burst calls count in
  goodput and all metrics.
- **M (number of agents) in {8, 32, 96}.** 12 cells = 4 scenarios x 3 M.
- The best fixed windows differ by scenario (4 / ~24-32 / 8 under a drop / 24), so no single fixed window is
  feasible and efficient everywhere: this is the condition ACC is designed for (fixes H1).

## 3. Client timeouts (D19, decision Q3)
- Per-attempt client timeout **T_out = 1.0 s** (about 2.2 x L_target). On timeout the client stops waiting and the
  attempt is recorded as status 504; the server keeps working on it (it still occupies server capacity).
- Timeouts are retried exactly like 429/503 (backoff with jitter, no Retry-After), up to r_max = 6.
- **Proposed:** timeouts count in the controller's throttle fraction F together with 429/503 (they are the same kind
  of overload signal, and Section III's retry policy already groups them). Calibration: timeouts are rare (only in
  heavy overload), so this choice changes little in practice.

## 4. Methods
- **ACC:** W0 = 8, W_min = 1, **W_max = 128** (was 64; M now reaches 96), dev grid unchanged: alpha, beta in
  {.25,.5,.75}, eps in {.05,.1,.2}, n_c in {8,16,32} (81 configs).
- **Controller target margin kappa (NEW, decision Q2):** the controller regulates p95 latency to kappa * L_target,
  while the SLO is still evaluated at L_target = 0.45 s. Proposed **kappa = 0.8** fixed (not tuned), and a kappa
  sensitivity (0.7 / 0.8 / 0.9 / 1.0) later in RQ4. Reason (calibration): with kappa = 1 the AIMD probing around the
  setpoint violates the 5% SLO in capacity-bound regimes for EVERY one of the 81 configs (best 9.4% in CAP-steady);
  with kappa = 0.8 ACC is feasible in all tested cells. A controller target below the SLO is standard practice, but
  it is a new hyperparameter and a Section III-C addition.
- **Fixed windows:** dev candidates {1, 2, 4, 8, 16, 24, 32, 48, 64, 96}; reported in test: {4, 16, 64}, the best
  GLOBAL fixed W (one value for all scenarios, chosen like ACC), and the per-scenario best fixed W (oracle, for
  reference and the robustness matrix).
- **Token bucket + backoff:** 12 rates {10, 14, 18, 22, 27, 33, 40, 50, 65, 80, 100, 130} calls/s, burst 1, window
  unconstrained. (D19 said "~12 rates 20-45/s", based on v1's 40/s limit; with the v2 regimes the useful range is
  10-130/s: CAP-steady is feasible only up to ~14/s, RATE-steady saturates at 100/s.)
- **Ablations (test):** ACC without the 429/503 branch, without the latency branch, additive-only.

## 5. Selection on dev (D17)
- Dev seeds 0-4 (calibration used only seeds 0-1 of these); test seeds 100-109, untouched until the frozen config is
  committed.
- **Loaded cell:** a cell where at least one fixed-window dev candidate violates a constraint (i.e. the choice of
  window matters there). Computed from dev results by a fixed rule, before any method is scored. Expected from
  calibration: all cells loaded except RATE-steady M=8 and CAP-drop M=8 (reported as light-load sanity cells).
- **Feasible (per family candidate):** for EVERY scenario, the mean over its loaded cells (seed-averaged) has SLO
  violation <= 5% and call success >= 99%.
- **Score:** mean over loaded cells of goodput / (best goodput of any FEASIBLE-in-that-cell dev method in that cell)
  (fixes M3; cell-level feasibility = SLO <= 5% and success >= 99% in that cell).
- **Choice:** per family (ACC, fixed W, token bucket) the feasible candidate with the highest score; ties within
  0.5% of the score are broken by lower mean SLO violation, then by the smaller (more conservative) parameter set
  in a fixed order. If no candidate of a family is feasible, nothing is chosen and the user is asked (constraints
  are never relaxed silently).
- **Pareto front** (mean normalised goodput vs mean SLO violation over loaded cells) is reported for all dev
  candidates.

## 6. Metrics
- Existing: goodput (successful calls / T), call success, throttle rate (429/503/504 per attempt), p50/p95/p99
  server latency of successes, SLO violation rate, latency excess (both variants), J (lambda = gamma = 0.5 primary,
  1.0 sensitivity), attempts per call, wasted attempts per successful call, e2e p95, window statistics.
- **Event windows (D19):** for CAP-drop: pre [20, 40), during [40, 80), post [80, 100); for CAP-burst: pre [20, 40),
  during [40, 60), post [60, 80). Goodput, SLO violation and throttle rate are reported per window.
- **Recovery time:** first t >= event end at which a 5-s sliding window has p95 latency <= L_target AND goodput >=
  0.9 x the pre-window goodput, minus the event end; censored at T (reported as ">= value").

## 7. Test phase and statistics
- Methods: frozen ACC, 3 ablations, fixed {4, 16, 64}, global best fixed W, per-scenario best fixed W values (at most
  4 distinct values), tuned token bucket. About 13 methods x 12 cells x 10 seeds ~ 1,560 runs.
- **Robustness matrix (D16):** each per-scenario best fixed W (chosen on dev) evaluated in every scenario on test
  seeds: goodput and SLO, next to ACC (one global config).
- Statistics: per cell mean +- 95% CI (t, 10 seeds); paired differences ACC - baseline per cell with a paired
  bootstrap over seeds (10,000 resamples; seeds are paired through common random numbers); a difference is called
  significant only if its 95% CI excludes 0. Results that favour a baseline are reported as such.

## 8. Cost
Dev: (81 ACC + 10 fixed + 12 token bucket) x 12 cells x 5 seeds = 6,180 runs. Calibration timing ~0.4 s per run on
average (heavier at M=96) -> about 40 min in the sandbox, roughly 1-1.5 h on a free Colab CPU. Test ~1,560 runs,
~10-15 min. Fully resumable as before.

## 9. Calibration evidence (scratch prototype; dev seeds 0-1; NOT results)
ACC = alpha .75, beta .5, eps .05, n_c 32, kappa 0.8, client timeouts on. Entries: goodput (calls/s) / SLO violation.

| scenario | M | W2 | W4 | W8 | W16 | W24 | W32 | W48 | W64 | W96 | ACC (kappa=0.8) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAP-steady | 8 | 10 / 0.00 | 19 / 0.00 | 15 / 0.33 | 15 / 0.33 | 15 / 0.33 | 15 / 0.33 | 15 / 0.33 | 15 / 0.33 | 15 / 0.33 | 17.6 / 0.020 |
| CAP-steady | 32 | 10 / 0.00 | 19 / 0.00 | 15 / 0.67 | 15 / 0.61 | 15 / 0.61 | 15 / 0.61 | 15 / 0.61 | 15 / 0.61 | 15 / 0.61 | 16.6 / 0.028 |
| CAP-steady | 96 | 10 / 0.00 | 19 / 0.00 | 15 / 0.67 | 15 / 0.64 | 15 / 0.64 | 15 / 0.64 | 15 / 0.64 | 15 / 0.64 | 15 / 0.64 | 15.6 / 0.034 |
| RATE-steady | 8 | 10 / 0.00 | 19 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 25.9 / 0.001 |
| RATE-steady | 32 | 10 / 0.00 | 19 / 0.00 | 38 / 0.00 | 77 / 0.00 | 100 / 0.00 | 100 / 0.00 | 100 / 0.00 | 100 / 0.00 | 100 / 0.00 | 97.6 / 0.001 |
| RATE-steady | 96 | 10 / 0.00 | 19 / 0.00 | 39 / 0.00 | 77 / 0.00 | 100 / 0.00 | 100 / 0.00 | 100 / 0.00 | 100 / 0.00 | 100 / 0.00 | 95.8 / 0.001 |
| CAP-drop | 8 | 10 / 0.00 | 19 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 26 / 0.00 | 25.9 / 0.001 |
| CAP-drop | 32 | 10 / 0.00 | 19 / 0.00 | 38 / 0.00 | 61 / 0.11 | 78 / 0.08 | 78 / 0.08 | 78 / 0.08 | 78 / 0.08 | 78 / 0.08 | 74.3 / 0.008 |
| CAP-drop | 96 | 10 / 0.00 | 19 / 0.00 | 39 / 0.00 | 61 / 0.11 | 87 / 0.08 | 79 / 0.13 | 70 / 0.65 | 72 / 0.63 | 72 / 0.63 | 78.4 / 0.008 |
| CAP-burst | 8 | 10 / 0.00 | 19 / 0.00 | 30 / 0.00 | 36 / 0.00 | 39 / 0.00 | 39 / 0.01 | 38 / 0.16 | 38 / 0.14 | 38 / 0.14 | 36.6 / 0.006 |
| CAP-burst | 32 | 10 / 0.00 | 19 / 0.00 | 39 / 0.00 | 77 / 0.00 | 105 / 0.00 | 102 / 0.01 | 100 / 0.11 | 101 / 0.10 | 101 / 0.10 | 83.9 / 0.007 |
| CAP-burst | 96 | 10 / 0.00 | 19 / 0.00 | 39 / 0.00 | 77 / 0.00 | 116 / 0.00 | 103 / 0.05 | 90 / 0.66 | 92 / 0.64 | 92 / 0.64 | 106.1 / 0.007 |

Other calibration findings (same conditions):
- kappa = 1.0, all 81 ACC configs: best SLO violation in CAP-steady M=32 was 0.094; kappa 0.9: 0.055; 0.8: 0.028;
  0.7: 0.018 (goodput 16.9 / 16.9 / 16.6 / 15.3 vs best fixed W=4: 19.3).
- Token bucket at M=96: CAP-steady feasible only at <= 14/s (SLO 0.47 at 18/s); RATE-steady goodput = rate up to
  100/s; CAP-drop SLO >= 0.08 from 40/s; CAP-burst feasible up to 100/s.
- Light-load cells: RATE-steady M=8 and CAP-drop M=8 give identical results for every W >= 8.

## 10. Decisions needed from the user
- **Q1** Fixed run duration T = 120 s instead of 150 calls per agent (section 1). Alternatives: keep calls per agent
  and accept unequal event exposure; or a longer T (cost grows linearly).
- **Q2** Controller target margin kappa = 0.8 (new hyperparameter, Section III-C text change). Alternatives: tune
  kappa in the dev grid (x3-4 cost); or keep kappa = 1 and accept that ACC is infeasible under a 5% SLO in
  capacity-bound cells (then select via the Pareto front).
- **Q3** Client timeouts counted in F (section 3). Alternative: retried but not counted in F (F stays 429/503 as in
  the current text).
- **Q4** Token-bucket grid 10-130/s (12 rates) instead of D19's 20-45/s.
- **Q5** Score over loaded cells only, with the loaded-cell rule of section 5.

## 11. Implementation notes (after approval)
SIM_VERSION -> 4; new workload (horizon + open-loop burst source), TimeoutServer-style client timeout in the
gateway (server work continues; cancelled at run end), F including 504 if Q3 approved, kappa in ACCConfig (default
1.0 so the paper's base algorithm is unchanged), event-window and recovery metrics, loaded-cell rule and new
selection in analysis.py, robustness matrix, fresh results folder (RUN_NAME v2). Tests for each piece (including
determinism, CRN pairing of burst calls, timeout semantics, event-window boundaries, selection edge cases) and the
mutation check, then review, then a short trial in Colab before the full run.

## 12. Section III impacts (text to update after approval)
III-C: controller target kappa * L_target (if Q2); timeouts in F (if Q3). Evaluation setup (Chapter IV): horizon-based
workload, four scenarios, loaded-cell constraint rule, robustness matrix.
