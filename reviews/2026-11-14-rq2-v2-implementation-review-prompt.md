# Review request: implementation of RQ2 design v2 (simulator, metrics, selection, statistics), 2026-11-14

You are an independent senior reviewer (distributed systems, simulation, experimental statistics). Be skeptical and
concrete; no praise. Repo: https://github.com/notanut/eem-acc-skripsi (this is the reviewed state).

## Project
Thesis "Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control". RQ2
evaluates ACC (an AIMD controller of a global concurrency window shared by M agents) in a deterministic virtual-time
asyncio simulation with scripted agents, against fixed windows and a client-side token bucket. The approved design
(including implementation notes, section 14, and open decisions, section 15) is reproduced in full below; the code
must implement it exactly.

## What to check
1. Does the code implement the spec exactly (every section)? List every mismatch.
2. Simulation correctness: client timeouts (504; server work continues; cancelled at the end), stopping at T
   (cancellation of agents, burst calls, slot waiters, pacer, in-flight attempts), slot pool and window updates,
   determinism, common-random-number pairing (noise keyed by (agent, k, admission), jitter keyed per retry, burst ids).
3. Metrics: success population (calls requested before T - 20 s), SLO violation including 504, throttle rate including
   504, objective J, censored e2e, time-weighted queue, event windows (completion-time goodput), recovery time rules.
4. Selection: a-priori loaded cells, per-cell feasibility (SLO <= 5%, success >= 99%, throttle <= 10%), denominators
   from cell-feasible methods, ties, Pareto front, per-scenario best fixed W, leave-one-scenario-out.
5. Statistics: primary endpoints, pairing by seed, exact sign-flip test, Holm, t CIs. Any multiplicity or power issue?
6. Anything that could bias results, leak test information into tuning, or make a baseline unfair.
7. The implementation choices I1-I10 and the open decisions Q-I1, Q-GRID, Q-EQ, Q-K1, Q-REC: your recommendation each.

## Output
Findings ranked Critical/High/Medium/Low with file:line (or spec section), explanation and concrete fix; then your
answers to the open decisions; then a verdict: "OK to run the dev phase" / "run after fixes (list)" / "do not run".

## Spec (docs/rq2_design_v2.md)
# RQ2 design v2 (revision of D10-D15 after review) — APPROVED 2026-11-14 (r2 + user decisions D20-D23)

_Status: APPROVED by the user 2026-11-14 (decisions in section 12). Revision 2 (after an independent review of r1). Implements the user's decisions D16-D19.
Nothing here is implemented yet. Numbers in section 11 are CALIBRATION evidence from a scratch prototype on dev seeds
0-4 only (test seeds 100-109 never used); they are design aids, not results, and must not be cited.
Decisions are recorded in section 12. Options the user declined are kept, marked NOT INCLUDED, so the record is complete._

## 0. Why a v2, and disclosure
Review of v1 found: (H1) every regime had a throughput-optimal fixed window of about 8 = W0, so a fixed W=8 won
everywhere; (H2) the SLO constraint averaged over all cells hid ~10% violation in loaded cells; (H3) M=4 and the
closed-loop burst did not discriminate; (M1) events at absolute times diluted unequally; (M2) coarse token-bucket
grid; (M3) normalisation by a possibly infeasible method; no timeouts although the paper's retry policy has them.
**Disclosure for Chapter IV:** v1 was never run as an experiment, but its calibration (432 dev-seed runs) showed a
fixed W = 8 beating ACC almost everywhere. The scenarios below were redesigned AFTER seeing that. To keep this
defensible, v2 (a) states the expected direction of every comparison in advance (section 8), (b) reports ACC against
the per-scenario best fixed window (where ACC is expected to lose in CAP-steady and CAP-burst), and (c) repeats the
selection leaving one scenario out (D23). (A dedicated ACC-expected-to-lose scenario, LAT-shift, was offered and
NOT INCLUDED by the user's choice.)

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
| ~~LAT-shift~~ (NOT INCLUDED) | 24 / 48 | none | base latency x 2 during [40, 80) s | large W | loses |

- **Open-loop burst:** a separate source issues calls at Poisson times regardless of completions (same gateway,
  window and retry policy; its own random stream; request ids (M, j) keep CRN pairing). Burst calls count in all
  metrics. The gateway queue is unbounded; its length is reported per event window.
- **M in {8, 32, 96}**; 4 scenarios x 3 M = 12 cells.

## 3. Client timeouts
- Per-attempt client timeout **T_out = 1.0 s**. On timeout the client stops waiting; the attempt is recorded as
  status 504; the server keeps working on it (it still occupies capacity until it finishes or the run ends).
- Timeouts are retried like 429/503 (backoff with jitter), up to r_max = 6, and **count in F** and in f_j of Eq. (1)
  (D21). A 504 attempt **counts as an SLO violation** (its latency 1.0 s > L_target).
- Server noise must be keyed by (agent, k, admission number), not (agent, k): after a timeout the retry is a new
  admission and must get its own noise (implementation bug found in review; CRN pairing is unaffected).
- With hard cap = 2 x capacity, server latency is bounded at about 2.5 x base, so timeouts are rare (0-51 per run of
  ~13k attempts in calibration). RQ2 therefore says almost nothing about timeouts; stated in the paper.

## 4. Methods
- **ACC:** W0 = 8, W_min = 1, W_max = 128, dev grid alpha, beta in {.25,.5,.75}, eps in {.05,.1,.2}, n_c in {8,16,32}.
- **Controller target margin kappa (D22):** the controller regulates p95 latency to L_ctrl = kappa * L_SLO
  (L_SLO = 0.45 s, the SLO used in evaluation); dL and eps are then relative to L_ctrl (so alpha's effective gain is
  scaled by 1/kappa). Decided: kappa in {0.8, 0.9, 1.0} as a dev-grid dimension (243 ACC configs); kappa = 1 is the paper's base
  algorithm and is always reported. Calibration (single config): best SLO violation in CAP-steady M=32 is 0.094 /
  0.055 / 0.028 / 0.018 for kappa 1.0 / 0.9 / 0.8 / 0.7 on seeds 0-1, and 0.093 / 0.052 / 0.028 / 0.017 on seeds 2-4.
  The value 0.8 in r1 was itself chosen from these numbers, so a fixed 0.8 would be a tuned value and must be
  disclosed as such.
- **Fixed windows:** dev candidates {1, 2, 4, 8, 16, 24, 32, 48, 64, 96}.
- **Token bucket + backoff:** {10, 14, 15, 16, 17, 18, 22, 27, 33, 40, 50, 65, 80, 100, 130} calls/s (15 rates; the global
  choice is decided between 14 and 18/s, hence the fine steps there), burst 1, window unconstrained.
- Tuned 429/503-only AIMD baseline: NOT INCLUDED (the 429/503-only variant remains a test ablation).
- **Ablations (test):** ACC without the 429/503 branch, without the latency branch, additive-only (frozen ACC values).

## 5. Loaded cells (fixed BEFORE any run; fixes review C1, M3)
A cell is loaded if its offered demand exceeds the server limit by at least 20% at some time, computed from the
parameters alone. Offered concurrency c = M * L0 / (L0 + think) with L0 = the scenario's current base latency (0.2 s, so c = 2M/3)
, plus lambda_b * L0 during a burst. Loaded if c >= 1.2 * capacity(t) or c / L0 >= 1.2 * rate limit for some t.

| scenario | M=8 | M=32 | M=96 |
|---|---|---|---|
| CAP-steady (cap 4) | loaded (c=5.3) | loaded | loaded |
| RATE-steady (100/s) | light (27/s) | light (107/s < 120) | loaded (320/s) |
| CAP-drop (cap 8 during drop) | light (5.3) | loaded (21.3) | loaded |
| CAP-burst (cap 24) | light (5.3+16) | loaded (21.3+16) | loaded |

Light cells are reported as sanity cells (ACC should not hurt there) and are excluded from selection and scoring.
Every scenario has at least one loaded cell (8 loaded cells of 12).

## 6. Selection on dev (D20)
- Dev seeds 0-4; test seeds 100-109 untouched until the frozen configuration is committed.
- **Feasible:** in EVERY loaded cell (seed-averaged): SLO violation <= 5%, call success >= 99%, and throttle rate
  (429/503/504 per attempt) <= 10% [new constraint, justified by the gamma term of Eq. 1: protecting the API is the
  point of RQ2]. (Recommended over r1's per-scenario mean, which re-created v1's averaging problem: e.g. fixed W=32
  in CAP-burst has 5.3% violation at M=96 on all 5 seeds but passed on the scenario mean.)
- **Score:** mean over loaded cells of goodput / (best goodput of any feasible dev method in that cell). The
  denominators are computed from dev results and FROZEN with the configuration; test results are normalised by the
  same frozen denominators. A loaded cell in which no dev method is feasible is reported and excluded from scoring.
- **Choice per family** (ACC, fixed W, token bucket): the feasible candidate with the highest score.
  Ties: all candidates with score >= 0.995 x max are tied; among them pick the lowest mean SLO violation, then the
  lowest mean throttle rate, then the first in a fixed lexicographic order (ACC: kappa desc, n_c desc, alpha asc,
  beta asc, eps asc; fixed: W asc; token bucket: rate asc). If a family has no feasible candidate, nothing is chosen
  and the user is asked (never relaxed silently).
- **Pareto front** (mean normalised goodput vs mean SLO violation over loaded cells) reported for all dev candidates.
- **Leave-one-scenario-out (D23):** the selection is repeated 4 times, each time leaving one scenario
  out of the dev data, and the chosen configs are evaluated on the left-out scenario (analysis only, no extra runs
  beyond the test methods).

## 7. Metrics
- Goodput, call success (over calls requested before T), backlog at T, throttle rate (429/503/504 per attempt),
  p50/p95/p99 server latency of successes, SLO violation rate (successes + 504s), latency excess (both variants),
  J (lambda = gamma = 0.5 primary, 1.0 sensitivity; f_j includes 504), attempts per call, wasted attempts per
  successful call, e2e p95 (censored at T for unfinished calls), window statistics, gateway queue length.
- **Event windows:** an attempt belongs to the window of its DISPATCH time, a call to the window of its REQUEST time.
  CAP-drop: pre [20, 40), during [40, 80), post [80, 100); CAP-burst: pre [20, 40), during [40, 60),
  post [60, 80). Per window: goodput, SLO violation, throttle rate, e2e p95, mean gateway queue length.
- **Recovery time:** evaluated on a 1-s grid with a TRAILING 5-s window. Recovered at the first t >= event end from
  which, for 5 consecutive grid points, p95 latency of the window <= max(L_SLO, the method's own pre-window p95) and
  goodput >= 0.9 x the method's own pre-window goodput; recovery time = that t - event end. Reported as "n/a
  (no degradation)" if the method never left those bounds during the event, and as ">= T - event end (censored)" if
  not recovered by T. Windows with fewer than 20 successful attempts use goodput only.

## 8. Test phase, comparisons and statistics (fixes review M2)
- Methods: frozen ACC (and ACC with kappa = 1 if kappa was tuned), 3 ablations, fixed {4, 16, 64},
  global best fixed W, per-scenario best fixed W values, tuned token bucket.
- **Pre-registered primary comparisons (expected direction stated now):**
  P1 ACC vs the best GLOBAL fixed window and the tuned token bucket, on the loaded-cell score: expected ACC higher.
  P2 ACC vs the per-scenario best fixed window (an oracle that knows the scenario), per scenario: expected ACC lower in
  CAP-steady and CAP-burst, about equal goodput but lower throttle rate in RATE-steady, higher in CAP-drop.
- Primary endpoints are tested with a paired exact sign-flip (randomisation) test over the 10 test seeds (all 1,024
  sign patterns), Holm-corrected across the primary endpoints; effect sizes with 95% CIs from a paired t interval.
  Per-cell results are descriptive (mean and 95% t-CI), not significance-tested. Results contradicting the expected
  direction are reported as such.
- **Robustness matrix:** each per-scenario best fixed W (chosen on dev) evaluated in every scenario on test seeds.

## 9. Cost (measured: 0.14 / 0.36 / 0.49 s per run at M = 8 / 32 / 96, heaviest ~1.7 s)
- Dev (as decided): ACC 81 x 3 kappa = 243, fixed 10, token bucket 15 = 268 methods x 12 cells x 5 seeds = 16,080
  runs: about 1.5 h in the sandbox, roughly 3-5 h on a free Colab CPU (resumable across disconnects).
- Test: about 12 methods x 12 cells x 10 seeds = 1,440 runs, about 10-15 min.

## 10. Implementation notes (after approval)
SIM_VERSION -> 4; horizon workload and open-loop burst source; client timeout in the gateway (server work continues,
cancelled and awaited at run end); noise keyed by (agent, k, admission); F and f_j including 504; kappa in ACCConfig
(default 1.0); event-window, backlog, queue and recovery metrics; loaded
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

## 12. Decisions (user, 2026-11-14)
- **D20 (was D-A):** feasible = in EVERY loaded cell SLO <= 5%, success >= 99% and throttle rate <= 10% (replaces D17's
  per-scenario mean).
- **D21 (was D-B):** fixed horizon T = 120 s with the accounting of section 1; client timeouts count in F, in f_j of
  Eq. 1 and as SLO violations; token-bucket grid of 15 rates (10-130/s).
- **D22 (was D-C):** kappa in {0.8, 0.9, 1.0} tuned on dev; kappa = 1 always reported.
- **D23 (was D-E):** leave-one-scenario-out analysis INCLUDED. NOT included: LAT-shift scenario, tuned 429-only AIMD
  baseline, capacity sweep.

## 13. Section III impacts
III-C: L_ctrl = kappa * L_SLO (D22); timeouts in F (III-C) and in f_j of Eq. 1 (III-A) (D21). Chapter IV setup:
horizon workload, four scenarios, a-priori loaded cells, per-cell constraints, pre-registered comparisons, v1
disclosure.

## 14. Implementation notes (2026-11-14, after implementation and its independent review)
Choices made while implementing; items marked PENDING need the user's confirmation before the dev run.
- **I1 Success population (PENDING):** call success is measured over calls requested before T - 20 s (a call requested
  before 100 s that has not finished by 120 s counts as failed); success over all requested calls is also stored.
  Reason: closed-loop agents almost always have one call running at T, so counting those as failures capped success
  at about 1 - M/requested for EVERY method (best fixed W in CAP-steady M=96: 0.961), making all methods infeasible.
  20 s exceeds the longest retry chain (~15 s).
- **I2** Attempt metrics use attempts completed by T (the run stops at T; 4-34 in-flight attempts per run, <= 0.4%, have
  no outcome; slow ones are slightly over-represented among them, so SLO violation is marginally underestimated).
- **I3 (PENDING, see 15)** "ACC with kappa = 1" in the test phase = the frozen ACC configuration with kappa set to 1 (a
  kappa ablation).
- **I4** Leave-one-scenario-out evaluates the left-out scenario on its DEV results (those results were not seen by that
  selection; evaluating on test seeds would need extra runs).
- **I5** The per-scenario best fixed W is skipped for a scenario without loaded cells (never happens in this design).
- **I6** Event-window goodput and recovery goodput = successful calls COMPLETED in the window / its length (throughput,
  like the headline goodput); arrival rate, e2e and success use calls REQUESTED in the window; attempt metrics use
  attempts DISPATCHED in the window. (Request-time goodput was an arrival rate in closed loop and hid throughput drops.)
- **I7** Recovery: grid points run from event start + 5 s to T; "no_degradation" only if no point is bad between event
  start + 5 s and the recovery point (damage that appears after the event counts). The 5-s trailing window adds up
  to 5 s of lag to every recovery time.
- **I8** Because feasibility requires EVERY loaded cell, a loaded cell in which no dev method is feasible makes the
  selection fail (ok = False, the user is asked); it cannot be silently "excluded from scoring".
- **I9** Normalisation denominator of a cell = best goodput among methods that are feasible IN THAT CELL.
- **I10** P1 (ACC vs the global fixed W / token bucket) is close to certain by construction: to be feasible in
  CAP-steady (capacity 4) a single global fixed window must be about 4 and a token bucket about 14/s. Chapter IV
  must say so; P2 (vs the per-scenario best fixed W) is the informative comparison.
- Measured cost: M=96 runs take 0.1-1.4 s; total dev ~2 h in the sandbox, ~4-6 h on free Colab.

## 15. Decisions needed before the dev run (2026-11-14)
- **Q-I1** Confirm I1 (success over calls requested before T - 20 s).
- **Q-GRID** Fixed-window grid has nothing between 16 and 24; in RATE-steady M=96 (calibration, seed 0): W=16 77.4 calls/s,
  W=20 (not in grid) 97.0 calls/s with 0% throttling, ACC 96.4. W >= 24 is infeasible (51% throttled). Without W=20
  the P2 RATE-steady result would be an ACC win caused by the grid gap. Proposed: add W = 12 and 20 (+120 dev runs).
- **Q-EQ** P2 RATE-steady expects "about equal" goodput; a non-significant two-sided test does not show equivalence.
  Options: pre-register an equivalence margin (e.g. +-5% normalised goodput, TOST) or report that endpoint as
  descriptive only.
- **Q-K1** Besides the kappa ablation (I3), also select the best kappa = 1 ACC configuration on dev and run it on test
  (no extra dev runs, +120 test runs), so the paper's base algorithm is represented at its best.
- **Q-REC** Recovery uses server-side signals (p95 latency, completed goodput). In CAP-burst, too-small windows build a
  large client-side queue (e.g. fixed W=8: queue ~600, e2e p95 ~24 s after the burst) that these signals never see.
  Option: also require e2e p95 (of calls requested in the trailing window) <= 2 x the pre-window e2e p95 for recovery.


## Source code

### `src/eem_acc/scenarios.py`

```python
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
```

### `src/eem_acc/agent/scripted.py`

```python
"""Scripted (non-LLM) agents for isolating the ACC (RQ2) cheaply.

Each agent runs a sequential loop: think -> API call through the gateway -> repeat. Concurrency arises across
the M agents. `phases` gives the mean think time as a function of simulated time.

Two run modes:
* `calls_per_agent`: each agent makes exactly that many calls; the run ends when all are done (design v1).
* `horizon` (design v2, D21): agents keep calling until simulated time T = horizon; at T the run STOPS: every
  agent, burst call and in-flight attempt is cancelled (calls requested but unfinished stay in
  `Gateway.open_calls` and count as not successful), and abandoned server work is cancelled.
Optional open-loop `burst` (design v2, D18): an extra source issues calls at Poisson times with rate `rate`
during [t_start, t_end), regardless of completions. Its calls use request ids (n_agents, j).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from .gateway import Gateway


@dataclass(frozen=True)
class BurstConfig:
    t_start: float
    t_end: float
    rate: float              # extra calls per second during [t_start, t_end)

    def __post_init__(self) -> None:
        if not (0 <= self.t_start < self.t_end) or self.rate <= 0:
            raise ValueError("need 0 <= t_start < t_end and rate > 0")


@dataclass(frozen=True)
class WorkloadConfig:
    n_agents: int
    calls_per_agent: int | None   # v1 mode (None in horizon mode)
    phases: tuple                 # ((t_start, think_time_mean_s), ...) sorted by t_start
    start_jitter: float           # agents start at Uniform(0, start_jitter) s
    horizon: float | None = None  # v2 mode: run length T in simulated seconds
    burst: BurstConfig | None = None

    def __post_init__(self) -> None:
        if (self.calls_per_agent is None) == (self.horizon is None):
            raise ValueError("set exactly one of calls_per_agent (v1) or horizon (v2)")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.burst is not None and self.horizon is None:
            raise ValueError("an open-loop burst needs horizon mode")


def _think_mean(phases: tuple, now: float) -> float:
    mean = phases[0][1]
    for t, m in phases:
        if t <= now:
            mean = m
    return mean


async def _agent(gw: Gateway, wl: WorkloadConfig, rng: np.random.Generator, agent_id: int) -> None:
    loop = asyncio.get_running_loop()
    if wl.start_jitter > 0:
        await asyncio.sleep(float(rng.uniform(0, wl.start_jitter)))
    k = 0
    while wl.horizon is not None or k < wl.calls_per_agent:
        u = float(rng.exponential(1.0))                 # ALWAYS one draw per call: the stream never depends on timing
        mean = _think_mean(wl.phases, loop.time())
        if mean > 0:
            await asyncio.sleep(mean * u)
        if wl.horizon is not None and loop.time() >= wl.horizon:
            return
        await gw.call(request=(agent_id, k))
        k += 1


async def _burst_source(gw: Gateway, b: BurstConfig, rng: np.random.Generator, source_id: int,
                        spawned: list[asyncio.Task]) -> None:
    loop = asyncio.get_running_loop()
    if b.t_start > loop.time():
        await asyncio.sleep(b.t_start - loop.time())
    j = 0
    while True:
        await asyncio.sleep(float(rng.exponential(1.0 / b.rate)))
        if loop.time() >= b.t_end:
            return
        spawned.append(asyncio.ensure_future(gw.call(request=(source_id, j))))
        j += 1


async def run_agents(gw: Gateway, wl: WorkloadConfig, seed: int) -> None:
    # spawn(n_agents + 1): children 0..n-1 are the same streams as spawn(n_agents), child n feeds the burst
    children = np.random.SeedSequence(seed).spawn(wl.n_agents + 1)
    agents = [asyncio.ensure_future(_agent(gw, wl, np.random.default_rng(c), i))
              for i, c in enumerate(children[:wl.n_agents])]
    if wl.horizon is None:
        await asyncio.gather(*agents)
        return
    spawned: list[asyncio.Task] = []
    sources = []
    if wl.burst is not None:
        sources.append(asyncio.ensure_future(
            _burst_source(gw, wl.burst, np.random.default_rng(children[wl.n_agents]), wl.n_agents, spawned)))
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0.0, wl.horizon - loop.time()))
    everything = agents + sources + spawned
    for t in everything:
        t.cancel()
    results = await asyncio.gather(*everything, return_exceptions=True)
    for r in results:                                   # a real error must not be hidden by the cancellation
        if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
            raise r
    await gw.shutdown()
```

### `src/eem_acc/agent/gateway.py`

```python
"""API gateway: global concurrency window + retry policy (revised Section III-C).

* `AcquireSlot(W)`: at most W attempts in flight across ALL agents (FIFO waiters).
* Retry: 429 / 503 are retried by the gateway with truncated exponential backoff
  and jitter, honouring Retry-After; the slot is released while backing off.
  Retries do not consume agent steps. After `max_retries` the call is returned to
  the agent as an exhausted transient failure.
* Every attempt (including retries) is logged and fed to the controller.
* Optional client-side token-bucket pacer (baseline); queue_wait then includes pacing delay.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..keyed import KeyedUniforms

from ..acc import ACCController
from ..env import MockServer, Response


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    backoff_base: float     # s
    backoff_cap: float      # s
    timeout: float | None = None   # client per-attempt timeout (s): status 504, server keeps working (D21)


@dataclass
class AttemptRecord:
    t_dispatch: float
    status: int
    latency: float          # server-side latency of this attempt
    window: int             # limit W in force when the slot was granted
    in_use: int             # slots in use right after the grant (always <= window)


@dataclass
class CallRecord:
    t_request: float
    t_done: float
    status: int             # final status (200, or last 429/503 if retries exhausted)
    attempts: int
    queue_wait: float       # total time waiting for a slot, summed over attempts
    payload: Any = None
    request: Any = None     # (agent, k) id of the logical call

    @property
    def ok(self) -> bool:
        return self.status == 200


@dataclass(frozen=True)
class PacerConfig:
    """Client-side token-bucket pacer (baseline): at most `rate` attempts/s with bursts of `burst`."""
    rate: float
    burst: int


class TokenBucketPacer:
    """GCRA token bucket. Each acquire reserves its send time in FIFO order, so it is
    deterministic under the virtual clock. Retries are paced too (every attempt acquires)."""

    def __init__(self, cfg: PacerConfig):
        self.T = 1.0 / cfg.rate                  # emission interval
        self.tau = (cfg.burst - 1) * self.T      # burst tolerance
        self.tat = 0.0                           # theoretical arrival time

    async def acquire(self) -> None:
        now = asyncio.get_running_loop().time()
        tat = max(self.tat, now)
        t_allowed = max(now, tat - self.tau)
        self.tat = tat + self.T
        if t_allowed > now:
            await asyncio.sleep(t_allowed - now)


class SlotPool:
    """Counting semaphore whose limit can change at run time."""

    def __init__(self, limit: int):
        self.limit = limit
        self.in_use = 0
        self._waiters: deque[asyncio.Future] = deque()

    async def acquire(self) -> tuple[int, int]:
        """Wait for a slot. Returns (limit, in_use) as they were when the slot was GRANTED."""
        if self.in_use < self.limit and not self._waiters:
            self.in_use += 1
            return self.limit, self.in_use
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            return await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()          # the slot had already been granted to us: give it back
            raise

    def release(self) -> None:
        self.in_use -= 1
        self._wake()

    def set_limit(self, limit: int) -> None:
        self.limit = limit
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.in_use < self.limit:
            fut = self._waiters.popleft()
            if not fut.done():
                self.in_use += 1
                fut.set_result((self.limit, self.in_use))


class Gateway:
    def __init__(self, server: MockServer, controller: ACCController, retry: RetryConfig, seed: int,
                 pacer: PacerConfig | None = None):
        self.server = server
        self.controller = controller
        self.retry = retry
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)      # fallback stream when a call carries no request id
        self._jitter = KeyedUniforms(self.seed, tag=7)
        self.slots = SlotPool(controller.window)
        self.pacer = TokenBucketPacer(pacer) if pacer is not None else None
        self.attempts: list[AttemptRecord] = []
        self.calls: list[CallRecord] = []
        self.open_calls: dict[int, tuple[float, Any]] = {}   # requested, not finished: id -> (t_request, request)
        self._next_id = 0
        self.waiting = 0                                   # calls waiting for the pacer or a slot
        self.queue_trace: list[tuple[float, int]] = [(0.0, 0)]
        self._abandoned: set[asyncio.Task] = set()          # server work the client stopped waiting for

    def _queue(self, delta: int) -> None:
        self.waiting += delta
        self.queue_trace.append((asyncio.get_running_loop().time(), self.waiting))

    def _abandon(self, task: asyncio.Task) -> None:
        self._abandoned.add(task)

        def done(t: asyncio.Task) -> None:
            self._abandoned.discard(t)
            if not t.cancelled():
                t.exception()                              # retrieve, so it is never reported as unhandled
        task.add_done_callback(done)

    async def _send(self, request):
        if self.retry.timeout is None:
            return await self.server.handle(request)
        task = asyncio.ensure_future(self.server.handle(request))
        try:
            done, _ = await asyncio.wait({task}, timeout=self.retry.timeout)
        except BaseException:
            self._abandon(task)                            # the call itself was cancelled (end of run)
            raise
        if task in done:
            return task.result()
        self._abandon(task)                                # client gives up; the server keeps working
        return Response(504, self.retry.timeout)

    async def shutdown(self) -> None:
        """End of run: cancel server work that clients abandoned (timeouts / cancelled calls) and wait for it."""
        tasks = list(self._abandoned)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _backoff(self, k: int, request: tuple | None = None) -> float:
        r = self.retry
        base = min(r.backoff_cap, r.backoff_base * (2 ** k))
        # jitter keyed by (seed, request, retry index) when a request id is given, so that it does not
        # depend on the order in which other calls happen to run (common random numbers across methods)
        if request is None:
            u = float(self.rng.random())
        else:
            agent, call_k = request
            u = self._jitter.get(int(agent), int(call_k) * (r.max_retries + 1) + k)
        return base * (0.5 + 0.5 * u)

    async def call(self, request: tuple | None = None) -> CallRecord:
        """`request` = optional (agent, call_index) identifying the logical call; it keys the server noise
        and the backoff jitter so both are order-independent (common random numbers across methods)."""
        loop = asyncio.get_running_loop()
        t_request = loop.time()
        cid = self._next_id
        self._next_id += 1
        self.open_calls[cid] = (t_request, request)
        queue_wait = 0.0
        n = 0
        while True:
            t_wait = loop.time()
            self._queue(+1)
            try:
                if self.pacer is not None:
                    await self.pacer.acquire()
                window_now, in_use = await self.slots.acquire()     # grant-time limit and occupancy
            finally:
                self._queue(-1)
            t_dispatch = loop.time()
            queue_wait += t_dispatch - t_wait
            try:
                resp = await self._send(request)
                n += 1
                self.attempts.append(AttemptRecord(t_dispatch, resp.status, resp.latency, window_now, in_use))
                # Update the window BEFORE freeing this slot, so a shrink takes effect for the very slot being
                # released (otherwise it would be re-granted at the old limit; Algorithm 1 line 8 precedes
                # the next dispatch).
                if self.controller.record(resp.status, resp.latency, loop.time()):
                    self.slots.set_limit(self.controller.window)
            finally:
                self.slots.release()       # always freed, also if the server/controller raises
            if resp.status == 200:
                break
            if n > self.retry.max_retries:
                break
            delay = max(resp.retry_after or 0.0, self._backoff(n - 1, request))
            await asyncio.sleep(delay)
        rec = CallRecord(t_request, loop.time(), resp.status, n, queue_wait, resp.payload, request)
        self.calls.append(rec)
        del self.open_calls[cid]
        return rec
```

### `src/eem_acc/env/server.py`

```python
"""Mock API server model (in-process, virtual time).

Mechanisms (all parameters are explicit; there are NO defaults on purpose, because
the values are experimental design decisions that the user must confirm):

* Token-bucket rate limit -> HTTP 429 with a Retry-After hint.
* Load-dependent latency: service time grows once in-flight calls exceed `capacity`.
* Hard overload cap -> HTTP 503 when in-flight calls reach `hard_cap`.
* Optional capacity events: at time t, capacity and hard_cap are multiplied by a
  scale factor (e.g. a mid-run capacity drop).

Check order: rate limit (429) first, then overload (503), then service.
Latency is fixed at arrival time from the load at that moment (a documented
simplification).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..keyed import KeyedNormals


@dataclass(frozen=True)
class ServerConfig:
    base_latency: float          # s, unloaded median service time
    latency_sigma: float         # lognormal sigma of per-call noise (0 = deterministic)
    capacity: int                # in-flight level above which latency degrades
    degradation: float           # latency x (1 + degradation * max(0, inflight/capacity - 1))
    hard_cap: int                # in-flight >= hard_cap -> 503
    rate_limit: float            # token refill rate (calls / s)
    burst: int                   # token bucket size
    retry_after: float           # s, hint attached to 429 responses
    error_latency: float         # s, time to produce a 429/503
    capacity_events: tuple = ()  # ((t, scale), ...) sorted by t; scales capacity and hard_cap


@dataclass
class Response:
    status: int
    latency: float               # server-side time for this attempt (s)
    retry_after: float | None = None
    payload: Any = None


ADMISSION_STRIDE = 16   # max admissions of one logical call (>= max_retries + 1)


class MockServer:
    def __init__(self, cfg: ServerConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)      # fallback stream when no request id is given
        self._keyed = KeyedNormals(self.seed, tag=1)
        self._admissions: dict[tuple, int] = {}     # admissions so far per logical call
        self.inflight = 0
        self.max_inflight_seen = 0
        self._tokens = float(cfg.burst)
        self._last_refill = 0.0
        self._events = sorted(cfg.capacity_events)
        self.counts = {200: 0, 429: 0, 503: 0}

    def _scale(self, now: float) -> float:
        s = 1.0
        for t, scale in self._events:
            if t <= now:
                s = scale
        return s

    def _refill(self, now: float) -> None:
        dt = max(0.0, now - self._last_refill)
        self._tokens = min(float(self.cfg.burst), self._tokens + dt * self.cfg.rate_limit)
        self._last_refill = now

    def _noise(self, request: Any) -> float:
        """Lognormal multiplicative noise. With a request id (agent, k) the draw is keyed by
        (seed, agent, k, admission number): independent of serving order and of how many 429/503 attempts
        preceded an admission, so methods compared on the same seed see the same noise for the same logical
        call (common random numbers). A call is admitted more than once only after a client timeout (the
        retry is new work for the server) and then gets fresh noise."""
        if self.cfg.latency_sigma <= 0:
            return 1.0
        if request is None:
            z = self.rng.normal(0.0, 1.0)
        else:
            agent, k = request
            a = self._admissions.get(request, 0)
            self._admissions[request] = a + 1
            if a >= ADMISSION_STRIDE:
                raise ValueError(f"more than {ADMISSION_STRIDE} admissions of one call; raise ADMISSION_STRIDE")
            z = self._keyed.get(int(agent), int(k) * ADMISSION_STRIDE + a)
        return math.exp(self.cfg.latency_sigma * float(z))

    async def handle(self, request: Any = None) -> Response:
        cfg = self.cfg
        loop = asyncio.get_running_loop()
        now = loop.time()
        scale = self._scale(now)
        capacity = max(1, int(round(cfg.capacity * scale)))
        hard_cap = max(1, int(round(cfg.hard_cap * scale)))

        self._refill(now)
        if self._tokens < 1.0:
            self.counts[429] += 1
            await asyncio.sleep(cfg.error_latency)
            return Response(429, cfg.error_latency, retry_after=cfg.retry_after)
        if self.inflight >= hard_cap:
            self.counts[503] += 1
            await asyncio.sleep(cfg.error_latency)
            return Response(503, cfg.error_latency)

        self._tokens -= 1.0
        self.inflight += 1
        self.max_inflight_seen = max(self.max_inflight_seen, self.inflight)
        noise = self._noise(request)
        overload = max(0.0, self.inflight / capacity - 1.0)
        latency = cfg.base_latency * noise * (1.0 + cfg.degradation * overload)
        try:
            await asyncio.sleep(latency)
        finally:
            self.inflight -= 1
        self.counts[200] += 1
        return Response(200, latency)
```

### `src/eem_acc/acc/controller.py`

```python
"""Adaptive Concurrency Controller (AIMD), implementing revised Section III-C.

Once every `n_c` completed attempts (one control interval) the window W is updated
from that interval's attempts only:

    F   = fraction of attempts that returned 429 or 503
    dL  = (p95 server latency of the interval's *successful* attempts - L_target) / L_target
          (0 if the interval has no successful attempt)

    if use_throttle_branch and F > 0:          W <- max(W_min, floor(W * (1 - beta * F)))
    elif use_latency_branch and dL > epsilon:  W <- max(W_min, floor(W * (1 - alpha * dL)))
    elif (not use_latency_branch) or dL <= 0:  W <- min(W + 1, W_max)
    else:                                      hold   (tolerance band 0 < dL <= epsilon)

IMPLEMENTATION ASSUMPTION (not in the paper text, flagged in HANDOFF.md): the latency
percentile uses successful (HTTP 200) attempts only, because 429/503 responses carry
no service time.

Config flags exist so the same class also implements the ablations/baselines
(no-throttle-branch, no-latency-branch, additive-only, fixed window via `frozen`).
There are NO default values for the hyperparameters: they are experimental decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


OVERLOAD_STATUSES = (429, 503, 504)   # 504 = client timeout (D21): an overload signal like 429/503
_FLOOR_EPS = 1e-9   # guards floor() against float error, e.g. 12*(1-0.5*25/30) = 6.999999999999999 (exactly 7)


@dataclass(frozen=True)
class ACCConfig:
    alpha: float
    beta: float
    epsilon: float
    w0: int
    w_min: int
    w_max: int
    n_c: int
    l_target: float
    use_throttle_branch: bool = True
    use_latency_branch: bool = True
    multiplicative_decrease: bool = True   # False -> "additive-only" ablation (never decreases)
    frozen: bool = False                   # True -> fixed window baseline (W stays at w0)
    kappa: float = 1.0                     # controller target L_ctrl = kappa * l_target (l_target = the SLO); 1 = paper's base

    def __post_init__(self) -> None:
        if not (1 <= self.w_min <= self.w0 <= self.w_max):
            raise ValueError("need 1 <= w_min <= w0 <= w_max")
        if self.n_c < 1:
            raise ValueError("n_c must be >= 1")
        if self.l_target <= 0:
            raise ValueError("l_target must be > 0")
        if not (0 < self.kappa <= 1):
            raise ValueError("kappa must be in (0, 1]")

    @property
    def l_ctrl(self) -> float:
        return self.kappa * self.l_target


class ACCController:
    def __init__(self, cfg: ACCConfig):
        self.cfg = cfg
        self.window = int(cfg.w0)
        self._buf: list[tuple[int, float]] = []   # (status, server latency) of current interval
        self.n_updates = 0
        self.trace: list[tuple[float, int, float, float]] = []  # (t, W after update, F, dL)

    # -- signals -------------------------------------------------------------
    @staticmethod
    def throttle_rate(buf: list[tuple[int, float]]) -> float:
        return sum(1 for s, _ in buf if s in OVERLOAD_STATUSES) / len(buf)

    def latency_degradation(self, buf: list[tuple[int, float]]) -> float:
        ok = [lat for s, lat in buf if s == 200]
        if not ok:
            return 0.0
        p95 = float(np.percentile(ok, 95))
        return (p95 - self.cfg.l_ctrl) / self.cfg.l_ctrl

    # -- update --------------------------------------------------------------
    def next_window(self, w: int, f: float, dl: float) -> int:
        c = self.cfg
        if c.use_throttle_branch and f > 0:
            if not c.multiplicative_decrease:
                return w
            return max(c.w_min, math.floor(w * (1.0 - c.beta * f) + _FLOOR_EPS))
        if c.use_latency_branch and dl > c.epsilon:
            if not c.multiplicative_decrease:
                return w
            return max(c.w_min, math.floor(w * (1.0 - c.alpha * dl) + _FLOOR_EPS))
        if (not c.use_latency_branch) or dl <= 0:
            return min(w + 1, c.w_max)
        return w

    def record(self, status: int, latency: float, now: float = 0.0) -> bool:
        """Feed one completed attempt. Returns True if the window was updated."""
        self._buf.append((status, float(latency)))
        if len(self._buf) < self.cfg.n_c:
            return False
        buf, self._buf = self._buf, []
        f = self.throttle_rate(buf)
        dl = self.latency_degradation(buf)
        if not self.cfg.frozen:
            self.window = self.next_window(self.window, f, dl)
        self.n_updates += 1
        self.trace.append((float(now), self.window, f, dl))
        return True

    # -- checkpointing -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"window": self.window, "buf": list(self._buf), "n_updates": self.n_updates,
                "trace": list(self.trace)}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.window = int(s["window"])
        self._buf = [tuple(x) for x in s["buf"]]
        self.n_updates = int(s["n_updates"])
        self.trace = [tuple(x) for x in s["trace"]]
```

### `src/eem_acc/metrics.py`

```python
"""Infrastructure metrics from a finished simulation (RQ2).

Latency metrics use *successful* attempts' server-side latency (the quantity the
SLO and the ACC refer to). End-to-end metrics include queueing and retries.
The combined objective J (Eq. 1 of the paper) needs lambda/gamma (undecided), so only its
components are stored; `objective_J` combines them exactly as Eq. 1:
    J = call_success_rate - lambda * mean_latency_excess_all_attempts - gamma * throttle_rate
where the latency excess averages max(0, (L_j - L_target)/L_target) over ALL attempts j (a 429/503
contributes its short error latency, i.e. ~0 excess), as in the paper. `mean_latency_excess` is the
variant over successful attempts only (it does not dilute under heavy throttling); report both and say which is J.
`slo_violation_rate` and lat_p* use successful attempts only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agent.gateway import Gateway


def _pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if x else None


def compute_metrics(gw: Gateway, l_target: float, horizon: float | None = None,
                    analysis: "AnalysisConfig | None" = None) -> dict[str, Any]:
    """v1 (horizon None): as before. v2 (horizon T): see `compute_metrics_v2`, whose keys are added on top."""
    if horizon is not None:
        return compute_metrics_v2(gw, l_target, horizon, analysis)
    att, calls = gw.attempts, gw.calls
    n_att = len(att)
    n200 = sum(a.status == 200 for a in att)
    n429 = sum(a.status == 429 for a in att)
    n503 = sum(a.status == 503 for a in att)
    lat = [a.latency for a in att if a.status == 200]
    e2e = [c.t_done - c.t_request for c in calls]
    ok_calls = [c for c in calls if c.ok]
    makespan = (max(c.t_done for c in calls) - min(c.t_request for c in calls)) if calls else 0.0
    w = [a.window for a in att]
    excess_all = [max(0.0, (a.latency - l_target) / l_target) for a in att]
    return {
        "n_calls": len(calls),
        "n_calls_ok": len(ok_calls),
        "call_success_rate": len(ok_calls) / len(calls) if calls else None,
        "n_exhausted_calls": len(calls) - len(ok_calls),
        "n_attempts": n_att,
        "n_200": n200, "n_429": n429, "n_503": n503,
        "throttle_rate": (n429 + n503) / n_att if n_att else None,
        "attempts_per_call": n_att / len(calls) if calls else None,
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "slo_violation_rate": (sum(x > l_target for x in lat) / len(lat)) if lat else None,
        "mean_latency_excess": float(np.mean([max(0.0, (x - l_target) / l_target) for x in lat])) if lat else None,
        "mean_latency_excess_all_attempts": float(np.mean(excess_all)) if excess_all else None,   # J's definition
        "wasted_attempts_per_ok_call": (n_att - len(ok_calls)) / len(ok_calls) if ok_calls else None,
        "e2e_p50": _pct(e2e, 50), "e2e_p95": _pct(e2e, 95), "e2e_p99": _pct(e2e, 99),
        "mean_queue_wait": float(np.mean([c.queue_wait for c in calls])) if calls else None,
        "makespan": makespan,
        "goodput": len(ok_calls) / makespan if makespan > 0 else None,
        "window_mean": float(np.mean(w)) if w else None,
        "window_min": min(w) if w else None,
        "window_max": max(w) if w else None,
        "window_final": gw.controller.window,
        "server_max_inflight": gw.server.max_inflight_seen,
    }


def objective_J(m: dict[str, Any], lam: float, gam: float) -> float | None:
    """Eq. (1) of the paper from stored metrics (None if a component is undefined)."""
    parts = (m.get("call_success_rate"), m.get("mean_latency_excess_all_attempts"), m.get("throttle_rate"))
    if any(p is None for p in parts):
        return None
    return parts[0] - lam * parts[1] - gam * parts[2]



# ======================================================================================= design v2 (D21)
OVERLOAD = (429, 503, 504)


@dataclass(frozen=True)
class AnalysisConfig:
    """Event windows of a v2 scenario. `windows`: ((name, t0, t1), ...). `event`: (start, end) of the disturbance,
    `pre`: (t0, t1) of the reference window for recovery (None = no event)."""
    windows: tuple = ()
    event: tuple | None = None
    pre: tuple | None = None
    recovery_span: float = 5.0        # trailing window (s)
    recovery_grid: float = 1.0        # grid step (s)
    recovery_hold: int = 5            # consecutive good grid points
    recovery_min_ok: int = 20         # fewer successes in a window -> judge goodput only
    recovery_goodput_frac: float = 0.9
    success_grace: float = 0.0        # call success is measured over calls requested before T - grace (see below)


def _slo_bad(a, l_target: float) -> bool:
    return a.status == 504 or (a.status == 200 and a.latency > l_target)


def _calls_with_censoring(gw: Gateway, horizon: float):
    """[(t_request, ok, e2e, finished, t_done)] for every call REQUESTED before T; unfinished: e2e censored at T,
    t_done None."""
    out = [(c.t_request, c.ok, c.t_done - c.t_request, True, c.t_done) for c in gw.calls]
    out += [(t, False, horizon - t, False, None) for t, _ in gw.open_calls.values()]
    return out


def _queue_mean(trace: list[tuple[float, int]], t0: float, t1: float) -> float | None:
    """Time-weighted mean of a step function given as [(t, value)] (value holds from t on) over [t0, t1)."""
    if t1 <= t0:
        return None
    area, cur, last = 0.0, 0, t0
    for t, v in trace:
        if t <= t0:
            cur = v
            continue
        if t >= t1:
            break
        area += cur * (t - last)
        cur, last = v, t
    area += cur * (t1 - last)
    return area / (t1 - t0)


def _completed_ok(calls, t0: float, t1: float, closed_right: bool = False) -> int:
    if closed_right:
        return sum(1 for c in calls if c[1] and t0 < c[4] <= t1)
    return sum(1 for c in calls if c[1] and t0 <= c[4] < t1)


def _window_stats(gw: Gateway, calls, l_target: float, t0: float, t1: float) -> dict[str, Any]:
    """Goodput = successful calls COMPLETED in the window / its length (throughput, as the headline goodput);
    e2e and arrivals use calls REQUESTED in the window; attempt metrics use attempts DISPATCHED in the window."""
    att = [a for a in gw.attempts if t0 <= a.t_dispatch < t1]
    cw = [c for c in calls if t0 <= c[0] < t1]
    judged = [a for a in att if a.status in (200, 504)]
    e2e = [c[2] for c in cw]
    return {
        "goodput": _completed_ok(calls, t0, t1) / (t1 - t0),
        "arrival_rate": len(cw) / (t1 - t0),
        "slo_violation_rate": (sum(_slo_bad(a, l_target) for a in judged) / len(judged)) if judged else None,
        "throttle_rate": (sum(a.status in OVERLOAD for a in att) / len(att)) if att else None,
        "e2e_p95_censored": _pct(e2e, 95),
        "queue_mean": _queue_mean(gw.queue_trace, t0, t1),
        "n_calls": len(cw), "n_attempts": len(att),
    }


def recovery_time(gw: Gateway, calls, l_target: float, horizon: float, cfg: AnalysisConfig) -> dict[str, Any]:
    """Section 7 of docs/rq2_design_v2.md. Trailing windows (t - span, t] on a grid; good = p95 latency of successful
    attempts dispatched in the window <= max(L_SLO, own pre-window p95) [skipped if < min_ok successes] AND goodput
    (successful calls COMPLETED in the window / span) >= frac x own pre-window goodput. Grid points run from
    event start + span to T. Recovered at the first grid point t* >= event end from which `hold` consecutive points
    are good; time = t* - event end.
    status: 'no_degradation' (no bad point between event start + span and t*: damage AFTER the event counts too) |
    'recovered' | 'censored' (never `hold` good points after the end; time = lower bound T - end) | 'no_reference'
    (no success in the pre window). The trailing window adds up to `span` s of lag to every recovery time."""
    (e0, e1), (p0, p1) = cfg.event, cfg.pre
    pre_ok = [a.latency for a in gw.attempts if p0 <= a.t_dispatch < p1 and a.status == 200]
    pre_gp = _completed_ok(calls, p0, p1) / (p1 - p0)
    if not pre_ok or pre_gp <= 0:
        return {"status": "no_reference", "time": None}
    lat_bound = max(l_target, float(np.percentile(pre_ok, 95)))
    span = cfg.recovery_span

    def good(t: float) -> bool:
        lo = t - span
        ok_lat = [a.latency for a in gw.attempts if lo < a.t_dispatch <= t and a.status == 200]
        gp = _completed_ok(calls, lo, t, closed_right=True) / span
        if gp < cfg.recovery_goodput_frac * pre_gp:
            return False
        if len(ok_lat) >= cfg.recovery_min_ok and float(np.percentile(ok_lat, 95)) > lat_bound:
            return False
        return True

    step = cfg.recovery_grid
    grid = [float(t) for t in np.arange(e0 + span, horizon + 1e-9, step)]
    flags = [good(t) for t in grid]
    for i, t in enumerate(grid):
        if t < e1 - 1e-9 or i + cfg.recovery_hold > len(grid):
            continue
        if all(flags[i:i + cfg.recovery_hold]):
            if all(flags[:i]):
                return {"status": "no_degradation", "time": None}
            return {"status": "recovered", "time": t - e1}
    return {"status": "censored", "time": horizon - e1}


def compute_metrics_v2(gw: Gateway, l_target: float, horizon: float,
                       analysis: AnalysisConfig | None = None) -> dict[str, Any]:
    """Design v2 accounting (D21): success population = calls requested before T - grace (unfinished = not successful);
    attempts =
    those completed by T (the run stops at T, so in-flight attempts have no outcome); a 504 (client timeout) is an
    overload signal (throttle rate, f_j of Eq. 1) and an SLO violation."""
    att = gw.attempts
    calls = _calls_with_censoring(gw, horizon)
    n_req = len(calls)
    n_ok = sum(c[1] for c in calls)
    n_att = len(att)
    cnt = {s: sum(a.status == s for a in att) for s in (200, 429, 503, 504)}
    judged = [a for a in att if a.status in (200, 504)]
    lat = [a.latency for a in att if a.status == 200]
    w = [a.window for a in att]
    excess_all = [max(0.0, (a.latency - l_target) / l_target) for a in att]
    grace = analysis.success_grace if analysis is not None else 0.0
    pop = [c for c in calls if c[0] < horizon - grace]      # every call in `pop` had >= grace s to finish
    out = {
        "n_calls_requested": n_req, "n_calls_ok": n_ok, "n_calls_finished": len(gw.calls),
        "backlog_at_T": len(gw.open_calls),
        "backlog_older_than_grace": sum(1 for t, _ in gw.open_calls.values() if t < horizon - grace),
        "success_grace": grace, "n_calls_success_population": len(pop),
        # closed-loop agents almost always have a call running at T; counting those as failures would cap
        # success at ~1 - M/requested for EVERY method, so the population excludes the last `grace` seconds
        "call_success_rate": (sum(c[1] for c in pop) / len(pop)) if pop else None,
        "call_success_rate_all_requested": n_ok / n_req if n_req else None,
        "goodput": n_ok / horizon,
        "n_attempts": n_att, "n_200": cnt[200], "n_429": cnt[429], "n_503": cnt[503], "n_504": cnt[504],
        "throttle_rate": (cnt[429] + cnt[503] + cnt[504]) / n_att if n_att else None,
        "attempts_per_call": n_att / n_req if n_req else None,
        "wasted_attempts_per_ok_call": (n_att - n_ok) / n_ok if n_ok else None,
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "slo_violation_rate": (sum(_slo_bad(a, l_target) for a in judged) / len(judged)) if judged else None,
        "mean_latency_excess": float(np.mean([max(0.0, (x - l_target) / l_target) for x in lat])) if lat else None,
        "mean_latency_excess_all_attempts": float(np.mean(excess_all)) if excess_all else None,
        "e2e_p50_censored": _pct([c[2] for c in calls], 50), "e2e_p95_censored": _pct([c[2] for c in calls], 95),
        "queue_mean": _queue_mean(gw.queue_trace, 0.0, horizon),
        "queue_max": max(v for _, v in gw.queue_trace),
        "window_mean": float(np.mean(w)) if w else None,
        "window_min": min(w) if w else None, "window_max": max(w) if w else None,
        "window_final": gw.controller.window,
        "server_max_inflight": gw.server.max_inflight_seen,
        "horizon": horizon,
    }
    if analysis is not None:
        out["windows"] = {name: _window_stats(gw, calls, l_target, t0, t1) for name, t0, t1 in analysis.windows}
        if analysis.event is not None:
            out["recovery"] = recovery_time(gw, calls, l_target, horizon, analysis)
    return out
```

### `src/eem_acc/analysis.py`

```python
"""RQ2 analysis for design v2 (docs/rq2_design_v2.md sections 5-8; decisions D20-D23).

Dev selection (per family: ACC, fixed W, token bucket):
* loaded cells are fixed a priori (scenarios.loaded_cells());
* a cell result is feasible if SLO violation <= 5%, call success >= 99% and throttle rate <= 10% (seed means);
* a candidate is feasible if it is feasible in EVERY loaded cell;
* score = mean over scored loaded cells of goodput / denominator, denominator = best goodput of any feasible dev
  method in that cell (frozen with the selection; cells with no feasible method are excluded from scoring);
* ties: score >= (1 - TIE_TOLERANCE) * best; then lower mean SLO violation, lower mean throttle rate, then a fixed
  lexicographic order (conservative first). Constraints are never relaxed: no feasible candidate -> ok = False.
Test analysis: robustness matrix, pre-registered primary comparisons (exact paired sign-flip test over seeds, Holm),
leave-one-scenario-out (on dev data of the left-out scenario), Pareto front.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import scenarios as S
from .utils import ResultStore

METRICS = ["goodput", "slo_violation_rate", "call_success_rate", "throttle_rate"]
FAMILY_ORDER = {  # last-resort lexicographic tie-break: kappa desc (prefer the paper's base value), then n_c desc,
                  # alpha/beta/eps asc; fixed W and rate ascending (smaller first)
    "acc": [("m_kappa", False), ("m_n_c", False), ("m_alpha", True), ("m_beta", True), ("m_epsilon", True)],
    "fixed": [("m_w", True)],
    "tokenbucket": [("m_rate", True)],
}


# ------------------------------------------------------------------------------------------------ loading
def load_results(units_dir: str | Path) -> pd.DataFrame:
    rows = []
    for uid, res in ResultStore(units_dir).iter_results():
        meta, m = res["meta"], res["metrics"]
        row = {"unit_id": uid, "scenario": meta["scenario"], "M": meta["M"], "seed": meta["seed"],
               "tag": S.method_tag(meta["method"]), **{f"m_{k}": v for k, v in meta["method"].items()}}
        row.update({k: v for k, v in m.items() if not isinstance(v, dict)})
        for wname, ws in (m.get("windows") or {}).items():
            row.update({f"win_{wname}_{k}": v for k, v in ws.items()})
        if m.get("recovery"):
            row["recovery_status"] = m["recovery"]["status"]
            row["recovery_time"] = m["recovery"]["time"]
        rows.append(row)
    return pd.DataFrame(rows)


def _worst_fill(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["slo_violation_rate"] = d["slo_violation_rate"].fillna(1.0)   # nothing judged = worst case
    d["call_success_rate"] = d["call_success_rate"].fillna(0.0)
    d["throttle_rate"] = d["throttle_rate"].fillna(1.0)
    return d


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged core metrics per (scenario, M, tag), with the method parameters."""
    d = _worst_fill(df)
    keys = ["scenario", "M", "tag"]
    g = d.groupby(keys)
    out = g[METRICS].mean()
    out["n_seeds"] = g.size()
    params = [c for c in d.columns if c.startswith("m_")]
    if params:
        out = out.join(g[params].first())
    out = out.reset_index()
    out["cell_feasible"] = ((out["slo_violation_rate"] <= S.MAX_SLO_VIOLATION)
                            & (out["call_success_rate"] >= S.MIN_CALL_SUCCESS)
                            & (out["throttle_rate"] <= S.MAX_THROTTLE_RATE))
    return out


def _loaded_mask(cells: pd.DataFrame, loaded: list[tuple[str, int]]) -> pd.Series:
    keys = set(loaded)
    return pd.Series([(s, int(m)) in keys for s, m in zip(cells["scenario"], cells["M"])], index=cells.index)


def denominators(cells: pd.DataFrame, loaded: list[tuple[str, int]]) -> dict[str, float]:
    """Best goodput of any cell-feasible method per loaded cell ('scenario|M' -> value). Cells with no feasible
    method are absent (excluded from scoring and reported)."""
    out = {}
    for (s, m) in loaded:
        c = cells[(cells["scenario"] == s) & (cells["M"] == m) & cells["cell_feasible"]]
        if len(c):
            out[f"{s}|{int(m)}"] = float(c["goodput"].max())
    return out


def score_methods(df: pd.DataFrame, loaded: list[tuple[str, int]] | None = None,
                  denoms: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per method over the loaded cells: score (mean normalised goodput over scored cells), feasibility
    (feasible in every loaded cell, all loaded cells present), mean SLO violation / success / throttle."""
    loaded = S.loaded_cells() if loaded is None else loaded
    if df.empty or not loaded:
        raise ValueError("score_methods needs results and at least one loaded cell")
    cells = cell_means(df)
    cells = cells[_loaded_mask(cells, loaded)].copy()
    denoms = denominators(cells, loaded) if denoms is None else denoms
    cells["cell_key"] = cells["scenario"] + "|" + cells["M"].astype(int).astype(str)
    cells["denom"] = cells["cell_key"].map(denoms)
    cells["norm_goodput"] = cells["goodput"] / cells["denom"]
    g = cells.groupby("tag")
    out = pd.DataFrame({
        "score": g["norm_goodput"].mean(),             # NaN denominators (unscored cells) are skipped
        "mean_slo_violation": g["slo_violation_rate"].mean(),
        "mean_call_success": g["call_success_rate"].mean(),
        "mean_throttle_rate": g["throttle_rate"].mean(),
        "n_cells": g.size(),
        "n_feasible_cells": g["cell_feasible"].sum(),
    })
    params = [c for c in cells.columns if c.startswith("m_")]
    out = out.join(g[params].first()).reset_index()
    out["complete"] = out["n_cells"] == len(loaded)
    out["feasible"] = out["complete"] & (out["n_feasible_cells"] == len(loaded))
    return out.sort_values(["feasible", "score", "mean_slo_violation", "tag"],
                           ascending=[False, False, True, True]).reset_index(drop=True)


def choose(scored: pd.DataFrame, family: str) -> pd.Series | None:
    """Best feasible candidate of a family with the documented tie rules; None if none is feasible."""
    t = scored[(scored["m_kind"] == family) & scored["feasible"]]
    if t.empty:
        return None
    best = t["score"].max()
    tied = t[t["score"] >= (1.0 - S.TIE_TOLERANCE) * best]
    cols = ["mean_slo_violation", "mean_throttle_rate"] + [c for c, _ in FAMILY_ORDER[family]]
    asc = [True, True] + [a for _, a in FAMILY_ORDER[family]]
    cols_present = [c for c in cols if c in tied.columns]
    asc_present = [a for c, a in zip(cols, asc) if c in tied.columns]
    return tied.sort_values(cols_present, ascending=asc_present).iloc[0]


def _params(row: pd.Series, family: str) -> Any:
    if family == "acc":
        return {"alpha": float(row["m_alpha"]), "beta": float(row["m_beta"]), "epsilon": float(row["m_epsilon"]),
                "n_c": int(row["m_n_c"]), "kappa": float(row["m_kappa"])}
    if family == "fixed":
        return int(row["m_w"])
    return float(row["m_rate"])


def pareto_front(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """Non-dominated methods on (higher score, lower mean SLO violation), complete methods only."""
    t = scored[scored["complete"]].dropna(subset=["score"])
    pts = t[["tag", "score", "mean_slo_violation", "feasible"]].to_dict("records")
    front = [p for p in pts if not any(q["score"] >= p["score"] and q["mean_slo_violation"] <= p["mean_slo_violation"]
                                       and (q["score"] > p["score"] or q["mean_slo_violation"] < p["mean_slo_violation"])
                                       for q in pts)]
    return sorted(front, key=lambda p: -p["score"])


def _select(df: pd.DataFrame, loaded: list[tuple[str, int]]) -> dict[str, Any]:
    cells = cell_means(df)
    denoms = denominators(cells[_loaded_mask(cells, loaded)], loaded)
    scored = score_methods(df, loaded, denoms)
    res: dict[str, Any] = {"ok": True, "denominators": denoms,
                           "unscored_cells": [f"{s}|{m}" for s, m in loaded if f"{s}|{m}" not in denoms],
                           "top": {}}
    for fam, key in (("acc", "acc"), ("fixed", "best_fixed_w"), ("tokenbucket", "best_pacer_rate")):
        res["top"][fam] = scored[scored["m_kind"] == fam].head(5)[
            ["tag", "score", "mean_slo_violation", "mean_throttle_rate", "mean_call_success", "feasible"]].to_dict("records")
        row = choose(scored, fam)
        if row is None:
            res["ok"] = False
        else:
            res[key] = _params(row, fam)
    return res, scored


def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
    """Frozen configuration from DEV results (all loaded cells), plus the per-scenario best fixed W (oracle for the
    robustness matrix), the Pareto front and leave-one-scenario-out selections."""
    loaded = S.loaded_cells()
    res, scored = _select(df, loaded)
    res["criterion"] = {"objective": "maximise mean normalised goodput over loaded cells",
                        "max_slo_violation": S.MAX_SLO_VIOLATION, "min_call_success": S.MIN_CALL_SUCCESS,
                        "max_throttle_rate": S.MAX_THROTTLE_RATE, "tie_tolerance": S.TIE_TOLERANCE,
                        "loaded_cells": [f"{s}|{m}" for s, m in loaded]}
    res["dev_seeds"] = sorted(int(x) for x in df["seed"].unique())
    res["n_dev_runs"] = int(len(df))
    per = {}
    for scn in S.SCENARIOS:
        sub = [(s, m) for s, m in loaded if s == scn]
        if not sub:
            continue                  # no loaded cell: the window choice does not matter there (no oracle needed)
        r, _ = _select(df[df["scenario"] == scn], sub)
        if "best_fixed_w" in r:
            per[scn] = r["best_fixed_w"]
        else:
            res["ok"] = False
    res["best_fixed_w_per_scenario"] = per
    res["pareto_front"] = pareto_front(scored)
    res["loso"] = leave_one_scenario_out(df)
    return res


def leave_one_scenario_out(dev: pd.DataFrame) -> dict[str, Any]:
    """D23. For each scenario s: select on the other scenarios' loaded cells, then evaluate the chosen configs on s's
    loaded cells using s's DEV results (s was not seen by that selection; every dev candidate has dev results)."""
    out = {}
    loaded = S.loaded_cells()
    for s in S.SCENARIOS:
        train = [(a, m) for a, m in loaded if a != s]
        held = [(a, m) for a, m in loaded if a == s]
        if not train or not held:
            out[s] = {"skipped": "no loaded cells left to train on" if not train else "no loaded cells in this scenario"}
            continue
        sel, _ = _select(dev[dev["scenario"] != s], train)
        ev = {}
        held_df = dev[dev["scenario"] == s]
        cells = cell_means(held_df)
        cells = cells[_loaded_mask(cells, held)]
        dn = denominators(cells, held)
        for fam, key in (("acc", "acc"), ("fixed", "best_fixed_w"), ("tokenbucket", "best_pacer_rate")):
            if key not in sel:
                ev[fam] = None
                continue
            tag = S.method_tag(S.acc_method(**sel[key]) if fam == "acc" else
                               S.fixed_method(sel[key]) if fam == "fixed" else S.pacer_method(sel[key]))
            c = cells[cells["tag"] == tag].copy()
            c["norm"] = [g / dn.get(f"{a}|{int(m)}", np.nan) for g, a, m in zip(c["goodput"], c["scenario"], c["M"])]
            ev[fam] = {"config": sel[key], "tag": tag, "score_on_left_out": float(c["norm"].mean()) if len(c) else None,
                       "feasible_on_left_out": bool(c["cell_feasible"].all()) if len(c) else None,
                       "mean_slo_violation": float(c["slo_violation_rate"].mean()) if len(c) else None,
                       "mean_throttle_rate": float(c["throttle_rate"].mean()) if len(c) else None}
        out[s] = ev
    return out


# ------------------------------------------------------------------------------------------------ test phase
def robustness_matrix(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Each per-scenario best fixed W (and ACC, the global fixed W, the token bucket) evaluated in every scenario:
    mean over that scenario's loaded cells of goodput normalised by the frozen dev denominators, SLO, throttle."""
    loaded = [tuple(x.split("|")) for x in frozen["criterion"]["loaded_cells"]]
    loaded = [(s, int(m)) for s, m in loaded]
    cells = cell_means(test)
    cells = cells[_loaded_mask(cells, loaded)].copy()
    dn = frozen["denominators"]
    cells["norm"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(cells["goodput"], cells["scenario"], cells["M"])]
    rows = {"ACC (frozen)": S.method_tag(S.acc_method(**frozen["acc"])),
            f"fixed W={frozen['best_fixed_w']} (global)": S.method_tag(S.fixed_method(frozen["best_fixed_w"])),
            f"token bucket {frozen['best_pacer_rate']:g}/s": S.method_tag(S.pacer_method(frozen["best_pacer_rate"]))}
    for scn, w in frozen["best_fixed_w_per_scenario"].items():
        rows[f"fixed W={w} (best for {scn})"] = S.method_tag(S.fixed_method(w))
    out = []
    for label, tag in rows.items():
        for scn in S.SCENARIOS:
            c = cells[(cells["tag"] == tag) & (cells["scenario"] == scn)]
            if len(c):
                out.append({"method": label, "scenario": scn, "norm_goodput": c["norm"].mean(),
                            "slo_violation": c["slo_violation_rate"].mean(), "throttle_rate": c["throttle_rate"].mean(),
                            "feasible_all_loaded_cells": bool(c["cell_feasible"].all())})
    return pd.DataFrame(out)


def sign_flip_test(d: np.ndarray) -> float:
    """Exact two-sided paired randomisation test of mean(d) = 0: all 2^n sign patterns (n <= 20)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0 or n > 20:
        raise ValueError("need 1..20 paired differences")
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    means = np.abs((signs * d).mean(axis=1))
    return float(np.mean(means >= obs - 1e-12))


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 25: 2.060, 30: 2.042}


def t_ci(d: np.ndarray) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return (math.nan, math.nan)
    df = n - 1
    q = _T975.get(df) or (_T975[max(k for k in _T975 if k <= df)] if df < 30 else 1.96)
    h = q * d.std(ddof=1) / math.sqrt(n)
    return (float(d.mean() - h), float(d.mean() + h))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def _per_seed(test: pd.DataFrame, tag: str, cells_: list[tuple[str, int]], dn: dict, metric: str) -> pd.Series:
    t = _worst_fill(test[test["tag"] == tag])
    t = t[[(s, int(m)) in set(cells_) for s, m in zip(t["scenario"], t["M"])]].copy()
    if metric == "norm_goodput":
        t["v"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(t["goodput"], t["scenario"], t["M"])]
    else:
        t["v"] = t[metric]
    return t.groupby("seed")["v"].mean()


def primary_comparisons(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Pre-registered endpoints (spec section 8). Difference = ACC - baseline, per test seed (paired via CRN):
    P1a/P1b normalised goodput over all scored loaded cells vs the global fixed W / tuned token bucket;
    P2-<scenario> normalised goodput over that scenario's loaded cells vs that scenario's best fixed W;
    P2-RATE-steady-throttle: throttle rate in RATE-steady's loaded cells vs its best fixed W.
    Exact sign-flip p-values, Holm-adjusted over all endpoints; 95% paired t CI of the mean difference."""
    loaded = [(s, int(m)) for s, m in (x.split("|") for x in frozen["criterion"]["loaded_cells"])]
    scored = [c for c in loaded if f"{c[0]}|{c[1]}" in frozen["denominators"]]
    dn = frozen["denominators"]
    acc = S.method_tag(S.acc_method(**frozen["acc"]))
    ends = {
        "P1a ACC - global fixed W": (S.method_tag(S.fixed_method(frozen["best_fixed_w"])), scored, "norm_goodput", "ACC higher"),
        "P1b ACC - token bucket": (S.method_tag(S.pacer_method(frozen["best_pacer_rate"])), scored, "norm_goodput", "ACC higher"),
    }
    expect = {"CAP-steady": "ACC lower", "RATE-steady": "about equal", "CAP-drop": "ACC higher", "CAP-burst": "ACC lower"}
    for scn, w in frozen["best_fixed_w_per_scenario"].items():
        sub = [c for c in scored if c[0] == scn]
        ends[f"P2 {scn}: ACC - best fixed W={w}"] = (S.method_tag(S.fixed_method(w)), sub, "norm_goodput", expect.get(scn, ""))
    if "RATE-steady" in frozen["best_fixed_w_per_scenario"]:
        w = frozen["best_fixed_w_per_scenario"]["RATE-steady"]
        sub = [c for c in loaded if c[0] == "RATE-steady"]
        ends[f"P2 RATE-steady throttle: ACC - best fixed W={w}"] = (S.method_tag(S.fixed_method(w)), sub, "throttle_rate", "ACC lower")
    rows, pv = [], {}
    for name, (base, cells_, metric, exp) in ends.items():
        a, b = _per_seed(test, acc, cells_, dn, metric), _per_seed(test, base, cells_, dn, metric)
        d = (a - b).dropna()
        p = sign_flip_test(d.values) if len(d) else math.nan
        pv[name] = p
        lo, hi = t_ci(d.values)
        rows.append({"endpoint": name, "expected": exp, "n_seeds": len(d), "mean_diff": float(d.mean()) if len(d) else math.nan,
                     "ci95_low": lo, "ci95_high": hi, "p_signflip": p})
    adj = holm({k: v for k, v in pv.items() if not math.isnan(v)})
    for r in rows:
        r["p_holm"] = adj.get(r["endpoint"], math.nan)
    return pd.DataFrame(rows)
```

### `experiments/run_rq2.py`

```python
"""RQ2 runner (scripted agents, simulated server). Resumable: re-running continues.

Phases (approved design v2, docs/rq2_design_v2.md, D16-D23):
  dev     16,080 runs: 243 ACC configs (incl. kappa) + 10 fixed windows + 15 token-bucket rates,
          4 scenarios x M in {8, 32, 96}, dev seeds 0-4
  select  choose the frozen configuration from dev results (per-loaded-cell constraints) -> configs/rq2_frozen.json
          (also: per-scenario best fixed W, Pareto front, leave-one-scenario-out)
  test    frozen ACC (+ kappa = 1 variant) + 3 ablations + fixed {4,16,64} + global and per-scenario best fixed W
          + tuned token bucket, test seeds 100-109

Usage (repo root):
  PYTHONPATH=src python experiments/run_rq2.py --phase dev    --root <ROOT>
  PYTHONPATH=src python experiments/run_rq2.py --phase select --root <ROOT>
  PYTHONPATH=src python experiments/run_rq2.py --phase test   --root <ROOT>
<ROOT> is the Drive folder in Colab (e.g. /content/drive/MyDrive/Skripsi) or any local folder.
Outputs: <ROOT>/results/rq2/<phase>/{units,checkpoints}/, <ROOT>/logs/rq2_<phase>.log
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from eem_acc import scenarios as S
from eem_acc.analysis import load_results, score_methods, select_frozen
from eem_acc.sim import SIM_VERSION, run_scenario
from eem_acc.utils import ResultStore, atomic_write_json, run_resumable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FROZEN_PATH = REPO / "configs" / "rq2_frozen.json"


def phase_units(phase: str, frozen: dict | None):
    if phase == "dev":
        return S.units(S.dev_methods(), S.DEV_SEEDS), False
    if phase == "test":
        assert frozen is not None, "test phase needs configs/rq2_frozen.json (run --phase select first)"
        return S.units(S.test_methods(frozen), S.TEST_SEEDS), True
    raise ValueError(phase)


def spec_hash(scn: str, m: int, meth: dict, seed: int) -> str:
    """Hash of the FULL resolved configuration of one run + simulator version. Stored in every result;
    a resume that would mix results from a different configuration is refused."""
    blob = json.dumps({"sim": SIM_VERSION, "scenario": dataclasses.asdict(S.build(scn, m, meth, seed))},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def frozen_hash(frozen: dict | None) -> str | None:
    return None if frozen is None else hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]


_SRC_DIRS = ("sim.py", "metrics.py", "keyed.py", "scenarios.py", "env", "agent", "acc")


def source_hash() -> str:
    """Hash of the simulator source (informational stamp; SIM_VERSION is the enforced one)."""
    h = hashlib.sha256()
    pkg = REPO / "src" / "eem_acc"
    for name in _SRC_DIRS:
        p = pkg / name
        for f in sorted([p] if p.is_file() else p.rglob("*.py")):
            h.update(f.relative_to(pkg).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def check_results(phase: str, root: Path, frozen: dict | None) -> dict[str, str]:
    """ONE pass over every result file of the phase. Every file must belong to the phase's FULL unit list and
    carry that unit's current spec_hash; otherwise ValueError (mixed code/config versions, or leftovers of an
    older grid that load_results would silently read). Returns {unit_id: spec_hash} of the valid results."""
    units, _ = phase_units(phase, frozen)
    expected = {u[0]: spec_hash(*u[1:]) for u in units}
    store = ResultStore(root / "results" / "rq2" / phase / "units")
    ok, bad = {}, []
    for uid, res in store.iter_results():
        h = res.get("meta", {}).get("spec_hash") if isinstance(res, dict) else None
        if uid in expected and h == expected[uid]:
            ok[uid] = h
        else:
            bad.append(uid)
    if bad:
        raise ValueError(
            f"{len(bad)} result(s) in {store.root} do not match the current configuration/simulator version "
            f"(SIM_VERSION={SIM_VERSION}) or are not part of this phase's grid (e.g. {bad[0]}). "
            f"Use a new --root (recommended) or delete that folder.")
    return ok


def digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()[:16]


def run_phase(phase: str, root: Path, limit: int | None = None, frozen: dict | None = None):
    units, store_trace = phase_units(phase, frozen)
    if limit:
        units = units[:limit]
    index = {u[0]: u for u in units}
    fz = frozen_hash(frozen)

    check_results(phase, root, frozen)   # refuse to resume on top of results from other code/config
    stamp = {"sim_version": SIM_VERSION, "frozen_hash": fz, "numpy": np.__version__, "source_hash": source_hash()}

    def run_unit(uid: str) -> dict:
        _, scn, m, meth, seed = index[uid]
        r = run_scenario(S.build(scn, m, meth, seed))
        out = {"meta": {"scenario": scn, "M": m, "method": meth, "seed": seed,
                        "spec_hash": spec_hash(scn, m, meth, seed), **stamp},
               "metrics": r["metrics"]}
        if store_trace:
            out["window_trace"] = r["window_trace"]
        return out

    config = {"phase": phase, "design": "docs/rq2_design_v2.md@approved-2026-11-14", "sim_version": SIM_VERSION,
              "frozen": frozen}
    return run_resumable([u[0] for u in units], run_unit, root / "results" / "rq2" / phase,
                         config=config, stateful=False, checkpoint_every=50,
                         log_path=root / "logs" / f"rq2_{phase}.log", progress_interval_s=15.0)


def _canonical(obj) -> str:
    """Comparison form of a frozen config: the same serialiser as the file writer, NaN-safe."""
    from eem_acc.utils.atomic import _json_default
    return json.dumps(json.loads(json.dumps(obj, default=_json_default, sort_keys=True)), sort_keys=True)


def do_select(root: Path, force: bool = False) -> int:
    test_units = root / "results" / "rq2" / "test" / "units"
    if test_units.exists() and any(test_units.glob("*.json")):
        print("[select] refusing: test results already exist in this folder. Re-selecting after seeing test "
              "results would invalidate the dev/test split. Use a fresh --root for a new experiment.")
        return 4
    try:
        hashes = check_results("dev", root, None)
    except ValueError as e:
        print(f"[select] refusing: {e}")
        return 5
    expected = len(S.units(S.dev_methods(), S.DEV_SEEDS))
    if len(hashes) != expected:
        print(f"[select] refusing: {len(hashes)} of {expected} dev runs present. Finish --phase dev first.")
        return 2
    df = load_results(root / "results" / "rq2" / "dev" / "units")
    res = select_frozen(df)
    res = {**res, "sim_version": SIM_VERSION, "dev_digest": digest(hashes)}
    print(json.dumps(res, indent=2, default=str))
    if not res["ok"]:
        print("[select] NO feasible candidate in at least one family; nothing written. Ask the user how to proceed.")
        return 3
    if FROZEN_PATH.exists() and not force:
        old = json.loads(FROZEN_PATH.read_text())
        outdated = "sim_version" in old and old["sim_version"] != SIM_VERSION   # unusable anyway: test refuses it
        if outdated:
            print(f"[select] replacing {FROZEN_PATH}: it was selected with SIM_VERSION={old['sim_version']} (now {SIM_VERSION})")
        elif _canonical(res) != _canonical(old):
            print(f"[select] refusing: {FROZEN_PATH} already exists with DIFFERENT content (another root or "
                  f"an older run). Re-run with --force only if you really mean to replace it.")
            return 6
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(FROZEN_PATH, res)
    atomic_write_json(root / "results" / "rq2" / "dev_scores.json", score_methods(df).to_dict("records"))
    atomic_write_json(root / "results" / "rq2" / "rq2_frozen.json", res)   # Drive copy (Colab clones are ephemeral)
    print(f"[select] wrote {FROZEN_PATH} and {root / 'results' / 'rq2' / 'rq2_frozen.json'}. Locally: commit the repo "
          f"copy BEFORE the test phase. In Colab the clone cannot push: the Drive copy is the record (download it "
          f"or send it to be committed).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "select", "test"], required=True)
    ap.add_argument("--root", default="outputs", help="Drive folder (Colab) or local output folder")
    ap.add_argument("--limit", type=int, default=None, help="only the first N units (smoke tests)")
    ap.add_argument("--force", action="store_true", help="select: replace a different configs/rq2_frozen.json")
    args = ap.parse_args()
    root = Path(args.root)
    if args.phase == "select":
        return do_select(root, args.force)
    frozen = None
    if args.phase == "test":
        frozen, msg = load_frozen_for_test(root)
        if frozen is None:
            print(f"[test] refusing: {msg}")
            return 7
        print(f"[test] {msg}")
    run_phase(args.phase, root, args.limit, frozen)
    return 0


def load_frozen_for_test(root: Path) -> tuple[dict | None, str]:
    """The test phase uses the root's frozen copy; the repo copy (if any) must be identical, its simulator
    version must be the current one, and it must have been selected from THIS root's dev results."""
    drive = root / "results" / "rq2" / "rq2_frozen.json"
    copies = [p for p in (drive, FROZEN_PATH) if p.exists()]
    if not copies:
        return None, "no frozen configuration found; run --phase select first"
    frozen = json.loads(copies[0].read_text())
    if len(copies) == 2 and json.loads(copies[1].read_text()) != frozen:
        return None, f"{drive} and {FROZEN_PATH} differ; they must be the same selection"
    if frozen.get("sim_version") != SIM_VERSION:
        return None, f"frozen configuration was selected with SIM_VERSION={frozen.get('sim_version')}, now {SIM_VERSION}"
    try:
        dev = check_results("dev", root, None)
    except ValueError as e:
        return None, f"dev results invalid: {e}"
    if frozen.get("dev_digest") != digest(dev):
        return None, "frozen configuration was not selected from this root's (complete, current) dev results"
    return frozen, f"frozen configuration from {copies[0]} (hash {frozen_hash(frozen)}, dev digest ok)"


if __name__ == "__main__":
    sys.exit(main())
```
