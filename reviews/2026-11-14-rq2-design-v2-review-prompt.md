# Review request: revised RQ2 experiment design (v2, revision 2), 2026-11-14

You are an independent senior reviewer in experimental methodology and distributed systems, acting like a strict
thesis examiner. Be skeptical and concrete; do not praise.

## Project
Thesis / IEEE-style paper "Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive
Concurrency Control" (Bina Nusantara University). RQ2 evaluates ACC, an AIMD controller of a global concurrency
window W shared by M LLM agents calling one API: on each control interval of n_c completed attempts,
if F (fraction of 429/503) > 0: W = floor(W(1 - beta F)); elif dL = (p95 latency of successes - L_target)/L_target
> eps: W = floor(W(1 - alpha dL)); elif dL <= 0: W + 1; else hold; clamp [W_min, W_max]. Retries with exponential
backoff + jitter honouring Retry-After. RQ2 runs in a deterministic virtual-time simulation (mock server with token
bucket -> 429, load-dependent latency with congestion collapse, hard cap -> 503) with scripted agents. The goal is to
show whether ACC protects the API (fewer 429/503, latency within SLO) without giving up goodput, versus fixed windows
and a client-side token bucket. Code: https://github.com/notanut/eem-acc-skripsi (v1 design implemented in
src/eem_acc/scenarios.py and analysis.py; v2 below is NOT implemented yet).

## Background you need
v1 (one set of scenarios all with an optimal fixed window of ~8 = ACC's start window) was reviewed and found unable
to show any benefit of adaptation; it was never run as an experiment. The author chose fixes (D16-D19). v2 r1 was
reviewed internally; r2 below incorporates that review. Calibration numbers come from a scratch prototype on dev
seeds only.

## The specification to review
# RQ2 design v2 (revision of D10-D15 after review) — DRAFT r2 for approval

_Status: proposal, 2026-11-14, revision 2 (after an independent review of r1). Implements the user's decisions D16-D19.
Nothing here is implemented yet. Numbers in section 11 are CALIBRATION evidence from a scratch prototype on dev seeds
0-4 only (test seeds 100-109 never used); they are design aids, not results, and must not be cited.
Items marked **[DECISION]** wait for the user; the recommended option is written in, alternatives are in section 12._

## 0. Why a v2, and disclosure
Review of v1 found: (H1) every regime had a throughput-optimal fixed window of about 8 = W0, so a fixed W=8 won
everywhere; (H2) the SLO constraint averaged over all cells hid ~10% violation in loaded cells; (H3) M=4 and the
closed-loop burst did not discriminate; (M1) events at absolute times diluted unequally; (M2) coarse token-bucket
grid; (M3) normalisation by a possibly infeasible method; no timeouts although the paper's retry policy has them.
**Disclosure for Chapter IV:** v1 was never run as an experiment, but its calibration (432 dev-seed runs) showed a
fixed W = 8 beating ACC almost everywhere. The scenarios below were redesigned AFTER seeing that. To keep this
defensible, v2 (a) states the expected direction of every comparison in advance (section 8), (b) reports ACC against
the per-scenario best fixed window (where ACC is expected to lose), and (c) adds a scenario designed so that ACC is
expected to lose [DECISION D-E].

## 1. Workload: fixed run duration
- Each run lasts **T = 120 s** of simulated time instead of 150 calls per agent. Agents (closed loop: think, call,
  repeat) stop issuing new calls at T. Think time exponential, mean 0.1 s; start jitter U(0,1) s.
- **Accounting (fixes review H4):** the population is every call REQUESTED before T. A call not finished by T counts
  as not successful, and the backlog at T (calls queued or in flight) is reported. Goodput = successful calls finished
  by T, divided by T. Attempts dispatched before T enter all attempt metrics. The run is not continued past T (no
  drain); abandoned server work is cancelled and awaited at the end.
- e2e latency percentiles include unfinished calls (their e2e is censored at T - t_request; reported as such).
- Why: every method and M sees the events at the same times (fixes M1), and goodput is comparable across methods.
- Warm-up is part of the run: ACC starts at W0 = 8 and needs time (about 500 completions) to grow, and this cost is
  counted in its goodput. Stated in the paper.

## 2. Server regimes and scenarios
Common: base latency 0.2 s, lognormal sigma 0.25, degradation 1.5 (latency x (1 + 1.5 * max(0, inflight/capacity - 1))),
Retry-After 0.5 s on 429, error latency 0.02 s. Beyond capacity, throughput FALLS (congestion collapse).

| scenario | capacity / hard cap | rate limit (burst) | event | best fixed W (calibration) | expected for ACC |
|---|---|---|---|---|---|
| CAP-steady | 4 / 8 | none (1000/s) | - | 4 | loses to best fixed W (probing cost) |
| RATE-steady | 48 / 96 | 100/s (20) | - | 24-32 | ~equal goodput, far fewer 429s |
| CAP-drop | 24 / 48 | none | capacity and hard cap x 1/3 during [40, 80) s | 8 (single W) | wins (adapts) |
| CAP-burst | 24 / 48 | none | open-loop Poisson burst, 80 extra calls/s during [40, 60) s | 24 | loses (slow recovery) |
| LAT-shift [D-E] | 24 / 48 | none | base latency x 2 during [40, 80) s, capacity unchanged | large W | loses (backs off for a latency rise it did not cause) |

- **Open-loop burst:** a separate source issues calls at Poisson times regardless of completions (same gateway,
  window and retry policy; its own random stream; request ids (M, j) keep CRN pairing). Burst calls count in all
  metrics. The gateway queue is unbounded; its length is reported per event window.
- **M in {8, 32, 96}**; 4 scenarios x 3 M = 12 cells (15 with LAT-shift).

## 3. Client timeouts
- Per-attempt client timeout **T_out = 1.0 s**. On timeout the client stops waiting; the attempt is recorded as
  status 504; the server keeps working on it (it still occupies capacity until it finishes or the run ends).
- Timeouts are retried like 429/503 (backoff with jitter), up to r_max = 6, and **count in F** and in f_j of Eq. (1)
  [DECISION, recommended]. A 504 attempt **counts as an SLO violation** (its latency 1.0 s > L_target).
- Server noise must be keyed by (agent, k, admission number), not (agent, k): after a timeout the retry is a new
  admission and must get its own noise (implementation bug found in review; CRN pairing is unaffected).
- With hard cap = 2 x capacity, server latency is bounded at about 2.5 x base, so timeouts are rare (0-51 per run of
  ~13k attempts in calibration). RQ2 therefore says almost nothing about timeouts; stated in the paper.

## 4. Methods
- **ACC:** W0 = 8, W_min = 1, W_max = 128, dev grid alpha, beta in {.25,.5,.75}, eps in {.05,.1,.2}, n_c in {8,16,32}.
- **Controller target margin kappa [DECISION D-C]:** the controller regulates p95 latency to L_ctrl = kappa * L_SLO
  (L_SLO = 0.45 s, the SLO used in evaluation); dL and eps are then relative to L_ctrl (so alpha's effective gain is
  scaled by 1/kappa). Recommended: kappa in {0.8, 0.9, 1.0} as a dev-grid dimension; kappa = 1 is the paper's base
  algorithm and is always reported. Calibration (single config): best SLO violation in CAP-steady M=32 is 0.094 /
  0.055 / 0.028 / 0.018 for kappa 1.0 / 0.9 / 0.8 / 0.7 on seeds 0-1, and 0.093 / 0.052 / 0.028 / 0.017 on seeds 2-4.
  The value 0.8 in r1 was itself chosen from these numbers, so a fixed 0.8 would be a tuned value and must be
  disclosed as such.
- **Fixed windows:** dev candidates {1, 2, 4, 8, 16, 24, 32, 48, 64, 96}.
- **Token bucket + backoff:** {10, 14, 15, 16, 17, 18, 22, 27, 33, 40, 50, 65, 80, 100, 130} calls/s (15 rates; the global
  choice is decided between 14 and 18/s, hence the fine steps there), burst 1, window unconstrained.
- **Tuned 429/503-only AIMD [DECISION D-E]:** ACC without the latency branch, tuned on dev as a baseline in its own
  right (beta x n_c = 9 configs), a classic adaptive baseline.
- **Ablations (test):** ACC without the 429/503 branch, without the latency branch, additive-only (frozen ACC values).

## 5. Loaded cells (fixed BEFORE any run; fixes review C1, M3)
A cell is loaded if its offered demand exceeds the server limit by at least 20% at some time, computed from the
parameters alone. Offered concurrency c = M * L0 / (L0 + think) with L0 = the scenario's current base latency (0.2 s, so c = 2M/3;
0.4 s during LAT-shift's shift, c = 0.8M), plus lambda_b * L0 during a burst. Loaded if c >= 1.2 * capacity(t) or c / L0 >= 1.2 * rate limit for some t.

| scenario | M=8 | M=32 | M=96 |
|---|---|---|---|
| CAP-steady (cap 4) | loaded (c=5.3) | loaded | loaded |
| RATE-steady (100/s) | light (27/s) | light (107/s < 120) | loaded (320/s) |
| CAP-drop (cap 8 during drop) | light (5.3) | loaded (21.3) | loaded |
| CAP-burst (cap 24) | light (5.3+16) | loaded (21.3+16) | loaded |
| LAT-shift [D-E] (cap 24) | light | light (25.6 during the shift, < 28.8) | loaded |

Light cells are reported as sanity cells (ACC should not hurt there) and are excluded from selection and scoring.
Every scenario has at least one loaded cell (8 loaded cells of 12; 9 of 15 with LAT-shift).

## 6. Selection on dev [DECISION D-A]
- Dev seeds 0-4; test seeds 100-109 untouched until the frozen configuration is committed.
- **Feasible:** in EVERY loaded cell (seed-averaged): SLO violation <= 5%, call success >= 99%, and throttle rate
  (429/503/504 per attempt) <= 10% [new constraint, justified by the gamma term of Eq. 1: protecting the API is the
  point of RQ2]. (Recommended over r1's per-scenario mean, which re-created v1's averaging problem: e.g. fixed W=32
  in CAP-burst has 5.3% violation at M=96 on all 5 seeds but passed on the scenario mean.)
- **Score:** mean over loaded cells of goodput / (best goodput of any feasible dev method in that cell). The
  denominators are computed from dev results and FROZEN with the configuration; test results are normalised by the
  same frozen denominators. A loaded cell in which no dev method is feasible is reported and excluded from scoring.
- **Choice per family** (ACC, fixed W, token bucket, 429-only AIMD): the feasible candidate with the highest score.
  Ties: all candidates with score >= 0.995 x max are tied; among them pick the lowest mean SLO violation, then the
  lowest mean throttle rate, then the first in a fixed lexicographic order (ACC: kappa desc, n_c desc, alpha asc,
  beta asc, eps asc; fixed: W asc; token bucket: rate asc). If a family has no feasible candidate, nothing is chosen
  and the user is asked (never relaxed silently).
- **Pareto front** (mean normalised goodput vs mean SLO violation over loaded cells) reported for all dev candidates.
- **Leave-one-scenario-out [DECISION D-E]:** the selection is repeated 4 (5) times, each time leaving one scenario
  out of the dev data, and the chosen configs are evaluated on the left-out scenario (analysis only, no extra runs
  beyond the test methods).

## 7. Metrics
- Goodput, call success (over calls requested before T), backlog at T, throttle rate (429/503/504 per attempt),
  p50/p95/p99 server latency of successes, SLO violation rate (successes + 504s), latency excess (both variants),
  J (lambda = gamma = 0.5 primary, 1.0 sensitivity; f_j includes 504), attempts per call, wasted attempts per
  successful call, e2e p95 (censored at T for unfinished calls), window statistics, gateway queue length.
- **Event windows:** an attempt belongs to the window of its DISPATCH time, a call to the window of its REQUEST time.
  CAP-drop and LAT-shift: pre [20, 40), during [40, 80), post [80, 100); CAP-burst: pre [20, 40), during [40, 60),
  post [60, 80). Per window: goodput, SLO violation, throttle rate, e2e p95, mean gateway queue length.
- **Recovery time:** evaluated on a 1-s grid with a TRAILING 5-s window. Recovered at the first t >= event end from
  which, for 5 consecutive grid points, p95 latency of the window <= max(L_SLO, the method's own pre-window p95) and
  goodput >= 0.9 x the method's own pre-window goodput; recovery time = that t - event end. Reported as "n/a
  (no degradation)" if the method never left those bounds during the event, and as ">= T - event end (censored)" if
  not recovered by T. Windows with fewer than 20 successful attempts use goodput only.

## 8. Test phase, comparisons and statistics (fixes review M2)
- Methods: frozen ACC (and ACC with kappa = 1 if kappa was tuned), 3 ablations, tuned 429-only AIMD, fixed {4, 16, 64},
  global best fixed W, per-scenario best fixed W values, tuned token bucket.
- **Pre-registered primary comparisons (expected direction stated now):**
  P1 ACC vs the best GLOBAL fixed window and the tuned token bucket, on the loaded-cell score: expected ACC higher.
  P2 ACC vs the per-scenario best fixed window (an oracle that knows the scenario), per scenario: expected ACC lower in
  CAP-steady, CAP-burst and LAT-shift, about equal goodput but lower throttle rate in RATE-steady, higher in CAP-drop.
  P3 ACC vs tuned 429-only AIMD on the loaded-cell score: no direction assumed.
- Primary endpoints are tested with a paired exact sign-flip (randomisation) test over the 10 test seeds (all 1,024
  sign patterns), Holm-corrected across the primary endpoints; effect sizes with 95% CIs from a paired t interval.
  Per-cell results are descriptive (mean and 95% t-CI), not significance-tested. Results contradicting the expected
  direction are reported as such.
- **Robustness matrix:** each per-scenario best fixed W (chosen on dev) evaluated in every scenario on test seeds.

## 9. Cost (measured: 0.14 / 0.36 / 0.49 s per run at M = 8 / 32 / 96, heaviest ~1.7 s)
- Dev with all recommended options: ACC 81 x 3 kappa = 243, fixed 10, token bucket 15, 429-only AIMD 9 = 277 methods
  x 15 cells x 5 seeds = 20,775 runs: about 2 h in the sandbox, roughly 4-6 h on a free Colab CPU (resumable).
- With kappa fixed at 0.8 (disclosed) and no LAT-shift: 115 x 12 x 5 = 6,900 runs, about 40 min (Colab ~1.5 h).
- Test: about 14 methods x 15 cells x 10 seeds = 2,100 runs, about 15 min.

## 10. Implementation notes (after approval)
SIM_VERSION -> 4; horizon workload and open-loop burst source; client timeout in the gateway (server work continues,
cancelled and awaited at run end); noise keyed by (agent, k, admission); F and f_j including 504; kappa in ACCConfig
(default 1.0); latency events in the server (LAT-shift); event-window, backlog, queue and recovery metrics; loaded
cells as a constant table; new selection (per-cell feasibility, throttle constraint, frozen denominators, ties,
LOSO); primary-comparison statistics; robustness matrix; fresh results folder (RUN_NAME "v2"). Tests for every piece
plus the mutation check, review, external review prompt, a short Colab trial, then the full run.

## 11. Calibration evidence (scratch prototype; dev seeds 0-1 unless stated; NOT results)
ACC = alpha .75, beta .5, eps .05, n_c 32, kappa 0.8, timeouts on; r1 accounting (completed calls only). Entries:
goodput (calls/s) / SLO violation.

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

Further calibration facts (independent review re-ran seeds 2-4): RATE-steady M=96: fixed W >= 24 has 51% throttled
attempts and call success 0.989-0.992 (retries exhausted), ACC 5% and 1.000. CAP-burst M=96: W=2 and token bucket
10/s leave ~1,000 burst calls unfinished at T (hidden under r1 accounting, fixed in section 1). ACC loses to the
per-scenario best fixed W by about 14% (CAP-steady), 3% (RATE-steady), 20% (CAP-burst M=32) and wins about 2x in
CAP-drop. Token bucket at M=96: CAP-steady feasible only up to 14/s.

## 12. Decisions needed from the user
- **D-A** Feasibility per loaded cell + throttle constraint <= 10% (recommended), or per-cell without throttle
  constraint, or keep D17's per-scenario mean.
- **D-B** Confirm: fixed horizon T = 120 s with the accounting of section 1; timeouts count in F/f_j and as SLO
  violations; token-bucket grid of 15 rates (10-130/s).
- **D-C** kappa: tune {0.8, 0.9, 1.0} on dev (recommended; ~2 h sandbox / 4-6 h Colab), or fix 0.8 and disclose it as
  calibrated, or kappa = 1 only (ACC then likely infeasible in capacity-bound cells).
- **D-E** Fairness additions: LAT-shift scenario (ACC expected to lose), leave-one-scenario-out analysis, tuned
  429-only AIMD baseline (all recommended; optional extra: CAP-steady capacity sweep 4/8/16 in RQ4).

## 13. Section III impacts
III-C: L_ctrl = kappa * L_SLO (if D-C); timeouts in F (III-C) and in f_j of Eq. 1 (III-A). Chapter IV setup:
horizon workload, five scenarios, a-priori loaded cells, per-cell constraints, pre-registered comparisons, v1
disclosure.


## Checklist
1. Is the design fair and defensible to an examiner, given the scenarios were redesigned after seeing v1's null
   result? Is the disclosure + pre-registration + "ACC expected to lose" scenario enough? What else?
2. Are the baselines strong enough (global fixed W, per-scenario oracle W, token bucket, tuned 429-only AIMD)? Any
   strawman left? Is it fair that a single global fixed W / rate must satisfy constraints in every loaded cell?
3. kappa (controller target below the SLO): fair? tune vs fix? Consistency with the controller spec.
4. Loaded-cell rule, per-cell feasibility, throttle constraint, frozen normalisation denominators, tie-breaking,
   leave-one-scenario-out: well-defined, gameable, circular?
5. Horizon accounting (unfinished calls, backlog, censoring), open-loop burst, timeouts (server work continues),
   event windows and recovery time: well-defined and measuring what they claim?
6. Statistics: pre-registered primary comparisons, sign-flip test with 10 seeds, Holm; enough seeds? power?
7. Dev/test leakage, unseeded randomness, CRN validity, anything that could bias or invalidate the results.
8. Cost/feasibility on a free Colab CPU.

## Output
Findings ranked Critical / High / Medium / Low with location (spec section), explanation and concrete fix; then your
answer to each of decisions D-A, D-B, D-C, D-E with reasons; then a verdict: "approve", "approve after fixes (list)"
or "rework".
