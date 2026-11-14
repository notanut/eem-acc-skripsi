# Review request: second-opinion fixes (retry set, no-latency intervals, timeout code) + Methodology patch r5, 2026-11-14

You are an independent reviewer (distributed systems + IEEE methodology writing). Be concrete; no praise.

## Context
Thesis: AIMD concurrency controller (ACC) for LLM agents calling an API (https://github.com/notanut/eem-acc-skripsi).
After an external review the author decided: D29 the gateway retries only 408, 429, 503 and client timeouts, every
other response returns to the agent at once (408 retried but not an overload signal); D30 client timeouts are latency
samples (at T_out) in the interval p95, and an interval with no latency sample holds the window (previously it grew
it); D31 client timeouts use a non-HTTP status code instead of 504. The Methodology text patch (revision 5) below
describes the controller, retry policy and a new Algorithm 2.

## Checklist
1. Is the code diff correct for every controller flag combination (frozen, additive-only, no-429, no-latency)?
2. Does the text describe the code exactly (Algorithm 2 order: adapt window before releasing the slot; retry set;
   latency samples; hold rule; overload rate; L_ctrl = kappa * L_target; tolerance band; small-sample p95 bias bounds
   1/9, 1/17, 2/33 for Hyndman-Fan type 7 with n = 8, 16, 32)?
3. Open point: other 5xx (500, 502, a real 504) are returned to the agent at once and are not handled by EEM (whose
   trigger covers 4xx, schema errors and 200-with-error). Should they be retried, or returned as a transient notice?
4. Anything an examiner would challenge.

## Output
Findings by severity (location, explanation, fix), then a verdict.

## Code diff
```diff
diff --git a/src/eem_acc/acc/controller.py b/src/eem_acc/acc/controller.py
index 78ffe16..f65103c 100644
--- a/src/eem_acc/acc/controller.py
+++ b/src/eem_acc/acc/controller.py
@@ -3,17 +3,17 @@
 Once every `n_c` completed attempts (one control interval) the window W is updated
 from that interval's attempts only:
 
-    F   = fraction of attempts that returned 429 or 503 or timed out (504 = client timeout, design v2 / D21)
-    dL  = (p95 server latency of the interval's *successful* attempts - L_ctrl) / L_ctrl,
-          L_ctrl = kappa * L_target (kappa = 1: base algorithm; D22); 0 if the interval has no successful attempt
+    F   = fraction of attempts that returned 429 or 503 or timed out (status.CLIENT_TIMEOUT; D21)
+    dL  = (p95 latency of the interval's HTTP 200 attempts and timeouts (at T_out) - L_ctrl) / L_ctrl,
+          L_ctrl = kappa * L_target (kappa = 1: base algorithm; D22). No such attempt: no latency sample, and the
+          window is HELD unless the throttle branch fires (D30; before: dL = 0, which grew the window)
 
     if use_throttle_branch and F > 0:          W <- max(W_min, floor(W * (1 - beta * F)))
     elif use_latency_branch and dL > epsilon:  W <- max(W_min, floor(W * (1 - alpha * dL)))
     elif (not use_latency_branch) or dL <= 0:  W <- min(W + 1, W_max)
     else:                                      hold   (tolerance band 0 < dL <= epsilon)
 
-The latency percentile uses successful (HTTP 200) attempts only (429/503/504 carry no service time); this is
-stated in Section III-C (patch v2, docs/section3_patch_v2.md).
+429/503 carry no service time and are not latency samples. Stated in Section III-C (docs/section3_patch_v2.md).
 
 Config flags exist so the same class also implements the ablations/baselines
 (no-throttle-branch, no-latency-branch, additive-only, fixed window via `frozen`).
@@ -29,7 +29,7 @@ from typing import Any
 import numpy as np
 
 
-OVERLOAD_STATUSES = (429, 503, 504)   # 504 = client timeout (D21): an overload signal like 429/503
+from ..status import LATENCY_SAMPLE, OVERLOAD as OVERLOAD_STATUSES
 _FLOOR_EPS = 1e-9   # guards floor() against float error, e.g. 12*(1-0.5*25/30) = 6.999999999999999 (exactly 7)
 
 
@@ -77,25 +77,27 @@ class ACCController:
     def throttle_rate(buf: list[tuple[int, float]]) -> float:
         return sum(1 for s, _ in buf if s in OVERLOAD_STATUSES) / len(buf)
 
-    def latency_degradation(self, buf: list[tuple[int, float]]) -> float:
-        ok = [lat for s, lat in buf if s == 200]
+    def latency_degradation(self, buf: list[tuple[int, float]]) -> float | None:
+        ok = [lat for s, lat in buf if s in LATENCY_SAMPLE]
         if not ok:
-            return 0.0
+            return None                     # no latency sample in this interval
         p95 = float(np.percentile(ok, 95))
         return (p95 - self.cfg.l_ctrl) / self.cfg.l_ctrl
 
     # -- update --------------------------------------------------------------
-    def next_window(self, w: int, f: float, dl: float) -> int:
+    def next_window(self, w: int, f: float, dl: float | None) -> int:
         c = self.cfg
         if c.use_throttle_branch and f > 0:
             if not c.multiplicative_decrease:
                 return w
             return max(c.w_min, math.floor(w * (1.0 - c.beta * f) + _FLOOR_EPS))
+        if c.use_latency_branch and dl is None:
+            return w                        # no latency sample: hold (D30)
         if c.use_latency_branch and dl > c.epsilon:
             if not c.multiplicative_decrease:
                 return w
             return max(c.w_min, math.floor(w * (1.0 - c.alpha * dl) + _FLOOR_EPS))
-        if (not c.use_latency_branch) or dl <= 0:
+        if (not c.use_latency_branch) or dl <= 0:  # (dl is not None here when the latency branch is used)
             return min(w + 1, c.w_max)
         return w
 
diff --git a/src/eem_acc/agent/gateway.py b/src/eem_acc/agent/gateway.py
index 175ccf2..d5fa93c 100644
--- a/src/eem_acc/agent/gateway.py
+++ b/src/eem_acc/agent/gateway.py
@@ -1,9 +1,11 @@
 """API gateway: global concurrency window + retry policy (revised Section III-C).
 
 * `AcquireSlot(W)`: at most W attempts in flight across ALL agents (FIFO waiters).
-* Optional client timeout (design v2): an attempt not answered within `timeout` is recorded as 504; the slot is
+* Optional client timeout (design v2): an attempt not answered within `timeout` is recorded as status
+  CLIENT_TIMEOUT (not an HTTP code); the slot is
   released but the server keeps working on it (abandoned work is cancelled at the end of a run).
-* Retry: 429 / 503 / 504 (timeout) are retried by the gateway with truncated exponential backoff
+* Retry: 408 / 429 / 503 / client timeout (status.RETRYABLE, D29) are retried by the gateway with truncated
+  exponential backoff; any other status (200, other 4xx/5xx) is returned to the agent at once,
   and jitter, honouring Retry-After; the slot is released while backing off.
   Retries do not consume agent steps. After `max_retries` the call is returned to
   the agent as an exhausted transient failure.
@@ -24,6 +26,7 @@ from ..keyed import KeyedUniforms
 
 from ..acc import ACCController
 from ..env import MockServer, Response
+from ..status import CLIENT_TIMEOUT, RETRYABLE
 
 
 @dataclass(frozen=True)
@@ -31,7 +34,7 @@ class RetryConfig:
     max_retries: int
     backoff_base: float     # s
     backoff_cap: float      # s
-    timeout: float | None = None   # client per-attempt timeout (s): status 504, server keeps working (D21)
+    timeout: float | None = None   # client per-attempt timeout (s): CLIENT_TIMEOUT, server keeps working (D21)
 
 
 @dataclass
@@ -47,7 +50,7 @@ class AttemptRecord:
 class CallRecord:
     t_request: float
     t_done: float
-    status: int             # final status (200, or last 429/503/504 if retries exhausted)
+    status: int             # final status (200, a non-retryable status, or the last retryable one if exhausted)
     attempts: int
     queue_wait: float       # total time waiting for a slot, summed over attempts
     payload: Any = None
@@ -165,7 +168,7 @@ class Gateway:
         if task in done:
             return task.result()
         self._abandon(task)                                # client gives up; the server keeps working
-        return Response(504, self.retry.timeout)
+        return Response(CLIENT_TIMEOUT, self.retry.timeout)
 
     async def shutdown(self) -> None:
         """End of run: cancel server work that clients abandoned (timeouts / cancelled calls) and wait for it."""
@@ -213,13 +216,13 @@ class Gateway:
                 n += 1
                 self.attempts.append(AttemptRecord(t_dispatch, resp.status, resp.latency, window_now, in_use))
                 # Update the window BEFORE freeing this slot, so a shrink takes effect for the very slot being
-                # released (otherwise it would be re-granted at the old limit; Algorithm 1 line 8 precedes
+                # released (otherwise it would be re-granted at the old limit; Algorithm 2 line 5 precedes
                 # the next dispatch).
                 if self.controller.record(resp.status, resp.latency, loop.time()):
                     self.slots.set_limit(self.controller.window)
             finally:
                 self.slots.release()       # always freed, also if the server/controller raises
-            if resp.status == 200:
+            if resp.status not in RETRYABLE:     # success or a non-transient error: back to the agent (EEM)
                 break
             if n > self.retry.max_retries:
                 break
diff --git a/src/eem_acc/metrics.py b/src/eem_acc/metrics.py
index 72cc207..4dcbefe 100644
--- a/src/eem_acc/metrics.py
+++ b/src/eem_acc/metrics.py
@@ -35,6 +35,7 @@ def compute_metrics(gw: Gateway, l_target: float, horizon: float | None = None,
     n200 = sum(a.status == 200 for a in att)
     n429 = sum(a.status == 429 for a in att)
     n503 = sum(a.status == 503 for a in att)
+    n_to = sum(a.status == CLIENT_TIMEOUT for a in att)     # never > 0 in v1 runs (no client timeout)
     lat = [a.latency for a in att if a.status == 200]
     e2e = [c.t_done - c.t_request for c in calls]
     ok_calls = [c for c in calls if c.ok]
@@ -48,7 +49,7 @@ def compute_metrics(gw: Gateway, l_target: float, horizon: float | None = None,
         "n_exhausted_calls": len(calls) - len(ok_calls),
         "n_attempts": n_att,
         "n_200": n200, "n_429": n429, "n_503": n503,
-        "throttle_rate": (n429 + n503) / n_att if n_att else None,
+        "throttle_rate": (n429 + n503 + n_to) / n_att if n_att else None,
         "attempts_per_call": n_att / len(calls) if calls else None,
         "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
         "slo_violation_rate": (sum(x > l_target for x in lat) / len(lat)) if lat else None,
@@ -77,7 +78,7 @@ def objective_J(m: dict[str, Any], lam: float, gam: float) -> float | None:
 
 
 # ======================================================================================= design v2 (D21)
-OVERLOAD = (429, 503, 504)
+from .status import CLIENT_TIMEOUT, OVERLOAD
 
 
 @dataclass(frozen=True)
@@ -97,7 +98,7 @@ class AnalysisConfig:
 
 
 def _slo_bad(a, l_target: float) -> bool:
-    return a.status == 504 or (a.status == 200 and a.latency > l_target)
+    return a.status == CLIENT_TIMEOUT or (a.status == 200 and a.latency > l_target)
 
 
 def _calls_with_censoring(gw: Gateway, horizon: float):
@@ -136,7 +137,7 @@ def _window_stats(gw: Gateway, calls, l_target: float, t0: float, t1: float) ->
     e2e and arrivals use calls REQUESTED in the window; attempt metrics use attempts DISPATCHED in the window."""
     att = [a for a in gw.attempts if t0 <= a.t_dispatch < t1]
     cw = [c for c in calls if t0 <= c[0] < t1]
-    judged = [a for a in att if a.status in (200, 504)]
+    judged = [a for a in att if a.status in (200, CLIENT_TIMEOUT)]
     e2e = [c[2] for c in cw]
     return {
         "goodput": _completed_ok(calls, t0, t1) / (t1 - t0),
@@ -206,15 +207,15 @@ def compute_metrics_v2(gw: Gateway, l_target: float, horizon: float,
                        analysis: AnalysisConfig | None = None) -> dict[str, Any]:
     """Design v2 accounting (D21): success population = calls requested before T - grace (unfinished = not successful);
     attempts =
-    those completed by T (the run stops at T, so in-flight attempts have no outcome); a 504 (client timeout) is an
+    those completed by T (the run stops at T, so in-flight attempts have no outcome); a client timeout is an
     overload signal (throttle rate, f_j of Eq. 1) and an SLO violation."""
     att = gw.attempts
     calls = _calls_with_censoring(gw, horizon)
     n_req = len(calls)
     n_ok = sum(c[1] for c in calls)
     n_att = len(att)
-    cnt = {s: sum(a.status == s for a in att) for s in (200, 429, 503, 504)}
-    judged = [a for a in att if a.status in (200, 504)]
+    cnt = {s: sum(a.status == s for a in att) for s in (200, 429, 503, CLIENT_TIMEOUT)}
+    judged = [a for a in att if a.status in (200, CLIENT_TIMEOUT)]
     lat = [a.latency for a in att if a.status == 200]
     w = [a.window for a in att]
     excess_all = [max(0.0, (a.latency - l_target) / l_target) for a in att]
@@ -230,8 +231,9 @@ def compute_metrics_v2(gw: Gateway, l_target: float, horizon: float,
         "call_success_rate": (sum(c[1] for c in pop) / len(pop)) if pop else None,
         "call_success_rate_all_requested": n_ok / n_req if n_req else None,
         "goodput": n_ok / horizon,
-        "n_attempts": n_att, "n_200": cnt[200], "n_429": cnt[429], "n_503": cnt[503], "n_504": cnt[504],
-        "throttle_rate": (cnt[429] + cnt[503] + cnt[504]) / n_att if n_att else None,
+        "n_attempts": n_att, "n_200": cnt[200], "n_429": cnt[429], "n_503": cnt[503], "n_timeout": cnt[CLIENT_TIMEOUT],
+        # "overload rate" in the paper (429 + 503 + client timeouts per attempt); key kept for compatibility
+        "throttle_rate": (cnt[429] + cnt[503] + cnt[CLIENT_TIMEOUT]) / n_att if n_att else None,
         "attempts_per_call": n_att / n_req if n_req else None,
         "wasted_attempts_per_ok_call": (n_att - n_ok) / n_ok if n_ok else None,
         "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
diff --git a/src/eem_acc/sim.py b/src/eem_acc/sim.py
index 13f888e..b49a40c 100644
--- a/src/eem_acc/sim.py
+++ b/src/eem_acc/sim.py
@@ -19,11 +19,13 @@ from .metrics import AnalysisConfig, compute_metrics
 
 # Bump whenever simulator/gateway/controller/metrics semantics change: it is part of every result's stamp,
 # so results produced by different code can never be silently mixed in one results folder.
-SIM_VERSION = "4"   # 2: shrink-before-release, grant-time window logging, keyed (CRN) randomness, J metrics
+SIM_VERSION = "5"   # 2: shrink-before-release, grant-time window logging, keyed (CRN) randomness, J metrics
                     # 3: server noise keyed by logical call (agent, k) only (not the attempt index);
                     #    keyed draws served from per-agent cached tables (eem_acc/keyed.py)
-                    # 4: design v2: horizon workload, open-loop burst, client timeouts (504 = overload),
+                    # 4: design v2: horizon workload, open-loop burst, client timeouts (= overload),
                     #    kappa, noise keyed by (agent, k, admission), success over requested calls
+                    # 5: client timeout = status.CLIENT_TIMEOUT (not 504); only status.RETRYABLE is retried;
+                    #    timeouts are latency samples at T_out; no latency sample -> hold W (D29, D30)
 
 
 @dataclass(frozen=True)
diff --git a/src/eem_acc/status.py b/src/eem_acc/status.py
new file mode 100644
index 0000000..55aa476
--- /dev/null
+++ b/src/eem_acc/status.py
@@ -0,0 +1,10 @@
+"""Attempt outcome codes shared by the gateway, the controller and the metrics.
+
+CLIENT_TIMEOUT is deliberately NOT an HTTP status (review L6, 2026-11-14): a real HTTP 504 from a proxy in the
+LLM stage must not be confused with the client giving up after T_out.
+"""
+
+CLIENT_TIMEOUT = -1                              # the client abandoned the attempt after T_out
+OVERLOAD = (429, 503, CLIENT_TIMEOUT)            # f_j = 1 / counted in F (D21)
+RETRYABLE = (408, 429, 503, CLIENT_TIMEOUT)      # retried by the gateway (D29); everything else returns at once
+LATENCY_SAMPLE = (200, CLIENT_TIMEOUT)           # enter the interval p95; a timeout counts at T_out (D30)
diff --git a/src/eem_acc/viz.py b/src/eem_acc/viz.py
index 6f91713..09de302 100644
--- a/src/eem_acc/viz.py
+++ b/src/eem_acc/viz.py
@@ -11,6 +11,7 @@ from __future__ import annotations
 import numpy as np
 
 from . import scenarios as S
+from .status import OVERLOAD
 from .monitor import fmt_duration, summary_line
 from .sim import simulate
 
@@ -82,7 +83,7 @@ def demo_figure(scenario: str = "CAP-drop", m_agents: int = 32, seed: int = 0, f
         att = gw.attempts
         t = np.array([a.t_dispatch for a in att])
         axes[0].step(t, [a.window for a in att], where="post", color=c, linewidth=2)
-        xs, ys = _binned(t, [a.status in (429, 503, 504) for a in att], 1.0, t_end, lambda v: float(sum(v)) if v else 0.0)
+        xs, ys = _binned(t, [a.status in OVERLOAD for a in att], 1.0, t_end, lambda v: float(sum(v)) if v else 0.0)
         axes[1].plot(xs, ys, color=c, linewidth=2)
         ok = [(a.t_dispatch, a.latency) for a in att if a.status == 200]
         xs, ys = _binned(np.array([o[0] for o in ok]), [o[1] for o in ok], 2.5, t_end,
```

## Methodology patch (revision 5)
# Section III patch v2, revision 5 (follows RQ2 design v2, decisions D21-D22, D29-D30) — DRAFT for the user

_Applies to `docs/section3_revised.md` (and later the paper). History: r2 longer justification (user's request; notation
and tone unchanged; two checked references); r3 fixes from an internal review; **r4 (2026-11-14) fixes from a second
opinion (external LLM) accepted by the user**: the small-sample bias of the p95 estimate (the "about 5%" claim was
wrong), the band argument kappa <= 1/(1+eps), the window definition after a decrease, the retry set (D29), no-latency
intervals (D30), "overload rate", a non-HTTP timeout code, Algorithm 2, and a clause on the server-side SLO.
**r5:** review fixes (exact bounds 1/9, 1/17, 2/33; list append in Algorithm 2; 'in F' qualifier; band condition
reported, not imposed; full hyperparameter list).
Exact old/new text below. Nothing else in Section III changes._

---

## P13 (III-A, problem formulation): timeouts in $f_j$, server-side SLO

**Old** (paragraph before Eq. 1):
> Let $L_j$ denote the server round-trip latency of attempt $j$ (excluding gateway queue wait), $f_j \in \{0,1\}$
> indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable), and
> $L_{\text{target}}$ the p95 latency Service Level Objective (SLO).

**New:**
> Let $L_j$ denote the server round-trip latency of attempt $j$ (excluding gateway queue wait), $f_j \in \{0,1\}$
> indicate an *overload signal*, i.e. that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service
> Unavailable) or was abandoned by the client after the timeout $T_{\text{out}}$ (in which case
> $L_j = T_{\text{out}}$), and $L_{\text{target}}$ the p95 latency Service Level Objective (SLO). The SLO is
> server-side: it is judged on HTTP 200 attempts and timed-out attempts, and it excludes time spent waiting in the
> gateway queue; client-side waiting is therefore reported separately (end-to-end latency and queue length,
> Section IV).

**New sentences, appended to the paragraph that follows Eq. 1:**
> A timed-out attempt is an overload signal because, like a 429 or 503 response, it shows that the server could not
> serve the request within the time the client is willing to wait. TCP also treats a retransmission timeout as a
> congestion signal, and reacts to it more strongly than to other loss signals by reducing its congestion window to
> one segment [RFC5681, Sec. 3.1]; in $F_\tau$, ACC does not distinguish a timeout from a 429 or 503. For a timed-out attempt we
> set $L_j = T_{\text{out}}$, a lower bound on its true round-trip latency, so the latency term of $J$ understates
> the cost of a timeout; the attempt is additionally penalised through $f_j$. Since $T_{\text{out}} >
> L_{\text{target}}$, every timeout also counts as an SLO violation. In $J$, the latency term averages over all
> attempts, so 429/503 responses (which return almost immediately) contribute no excess; the SLO violation rate
> reported in Section IV is computed over HTTP 200 and timed-out attempts only.

---

## P14 (III-C, window, control interval and signals)

**Window. Old:**
> **Window.** $W$ is the maximum number of in-flight API calls, shared globally across all $M$ agents and enforced by
> a semaphore (`AcquireSlot`).

**New:**
> **Window.** $W$ limits the number of *outstanding* attempts (sent, and neither answered nor timed out), shared
> globally across all $M$ agents: an attempt is dispatched only when fewer than $W$ attempts are outstanding
> (`AcquireSlot`), and its slot is released when the attempt is answered or times out. A decrease of $W$ does not
> cancel outstanding attempts, so their number can exceed the new $W$ until enough of them finish. The server may
> continue an attempt the client has abandoned, so the server's concurrency can exceed $W$ until that work
> finishes; Section IV reports the maximum server concurrency.

(The rest of the paragraph is unchanged.)

**Control interval and signals. Old:**
> **Control interval and signals.** The window is updated once every $n_c$ completed attempts (a control interval
> $\tau$), using only the attempts of the current interval, so earlier events are not counted twice. Define
>
> - the throttle rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts returning 429 or 503), and
> - the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{target}}}{L_{\text{target}}}$, where
>   $\hat{L}^{95}_\tau$ is the p95 server latency of the attempts in interval $\tau$.

**New:**
> **Control interval and signals.** The window is updated once every $n_c$ attempts that have an outcome (a response
> or a client timeout), forming a control interval $\tau$; only the attempts of the current interval are used, so
> earlier events are not counted twice. Define
>
> - the overload rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts in interval $\tau$ that
>   returned 429 or 503 or timed out), and
> - the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{ctrl}}}{L_{\text{ctrl}}}$, where
>   $\hat{L}^{95}_\tau$ is the p95 latency of the latency samples of interval $\tau$: its HTTP 200 attempts and its
>   timed-out attempts, the latter counted at $T_{\text{out}}$. A 200 response whose payload EEM treats as an error
>   (Section III-B) still measures server latency and is a latency sample; 429/503 responses and other errors are
>   not. The controller's latency target is
>   $$L_{\text{ctrl}} = \kappa\, L_{\text{target}}, \qquad 0 < \kappa \le 1 .$$

**Update rule. Old** (lead-in and first bullet unchanged), **New: one added rule** after the throttled case:
> - Else if interval $\tau$ has no latency sample: $W_{\tau+1} = W_\tau$ (no evidence either way; for the full
>   controller this case does not arise from overload, because an interval of only 429/503/timeouts has
>   $F_\tau > 0$).

(The remaining three rules are unchanged; they now use $\Delta L_\tau$ relative to $L_{\text{ctrl}}$.)

**New paragraph, directly after the update rules:**
> *Controller target.* The controller regulates latency to $L_{\text{ctrl}} = \kappa L_{\text{target}}$ rather than to
> the SLO itself, for three reasons. First, with $\kappa = 1$ the tolerance band $(L_{\text{target}},
> (1+\epsilon)L_{\text{target}}]$ lies entirely above the SLO, so the controller would hold the window while the
> interval p95 violates the SLO; the band lies at or below the SLO only if $\kappa \le 1/(1+\epsilon)$. Second,
> $\hat{L}^{95}_\tau$ is estimated from at most $n_c$ values and is biased low: for independent latencies from any
> continuous distribution, the expected fraction of attempts above the sample p95 (Hyndman-Fan type 7, the common
> default) of $n_c \in \{8, 16, 32\}$ values is at least 1/9, 1/17 and 2/33 (11.1%, 5.9% and 6.1%), so holding the
> estimate at
> $L_{\text{target}}$ does not hold the violation rate at 5%. Third, additive increase probes past the server's
> capacity knee, and the interval that should detect a probe may consist partly, or when $n_c \le W$ entirely, of
> attempts dispatched before it. A tighter internal target provides headroom for these effects, analogous to keeping
> a safety margin between internal and user-facing objectives [SRE-SLO], at the cost of a smaller window. Because
> $\Delta L_\tau$ and $\epsilon$ are relative to $L_{\text{ctrl}}$, the effective gain of $\alpha$ with respect to
> $L_{\text{target}}$ is $\alpha/\kappa$. $\kappa$ is chosen on the development split together with the other
> controller hyperparameters, and $\kappa = 1$ is reported both with the same configuration and re-tuned
> (Section IV). The band condition $\kappa(1+\epsilon) \le 1$ is not imposed on the selection; whether the selected
> configuration satisfies it is reported.

---

## P15 (III-C, retry policy) — D29

**Old:**
> **Retry policy.** The gateway retries attempts that return 429, 503, or time out, using truncated exponential
> backoff with jitter (honouring `Retry-After` when present) for at most $r_{\max}$ retries.

**New:**
> **Retry policy.** The gateway retries attempts that return 408 (Request Timeout), 429 or 503, or that exceed the
> client timeout $T_{\text{out}}$, using truncated exponential backoff with jitter (honouring `Retry-After` when
> present) for at most $r_{\max}$ retries; no slot is held during backoff, and a retry after a timeout is a new
> request for the server. Any other response, successful or not, is returned to the agent at once (non-transient
> errors are handled by EEM, Section III-B). A 408 is retried but is not an overload signal.

(The rest of the paragraph is unchanged: every attempt enters the statistics; retries consume no steps; exhausted
retries give a transient-failure observation without reflection.)

---

## P16 (Algorithm 1, new Algorithm 2)

**Algorithm 1, input line. New:** `Input: Task T, Tools A, Memory M, controller parameters Theta_ACC = (alpha, beta, epsilon, kappa, n_c, W_min, W_max, W0), Target latency L_target, client timeout T_out`

**Algorithm 1, line 1. Old:** `1: W_current <- W0;  s_1 <- InitializeState(T)`
**New:** `1: s_1 <- InitializeState(T)      // the window W is shared by all agents, initialised once to W0`

**Algorithm 1, lines 6-8. Old:**
```
 6:   AcquireSlot(W_current)
 7:   (Status, Payload) <- DispatchWithRetry(Action_t, Args_t)   // attempts logged to ACC, Section III-C
 8:   ReleaseSlot();  if control interval complete: W_current <- AdaptWindow(W_current, F_tau, DeltaL_tau)
```
**New** (one line; the following lines are renumbered 7-18):
```
 6:   (Status, Payload) <- DispatchWithRetry(Action_t, Args_t)   // Algorithm 2
```

**New Algorithm 2** (after Algorithm 1):
```
Algorithm 2: DispatchWithRetry(a, x)        // shared by all agents: window W, interval buffer B
 1: for r = 0 to r_max do
 2:   AcquireSlot(W)                        // wait until fewer than W attempts are outstanding
 3:   (status, payload, L) <- Send(a, x) with timeout T_out   // on timeout: status <- TIMEOUT, L <- T_out
 4:   append (status, L) to B
 5:   if |B| = n_c then W <- AdaptWindow(W, F(B), ΔL(B));  B <- ∅    // before the slot is released
 6:   ReleaseSlot()
 7:   if status ∉ {408, 429, 503, TIMEOUT} then return (status, payload)
 8:   if r < r_max then Sleep(max(RetryAfter, Backoff(r)))         // no slot held while waiting
 9: end for
10: return (status, payload)                                       // retries exhausted
```

**Note under Algorithm 1. Old:** "the window update (line 8) is driven by ..." **New:** "the window update
(Algorithm 2, line 5) is driven by ..." (rest unchanged).

**Hyperparameter note** (last paragraph of Section III). **New:**
> Hyperparameter values are symbolic in Section III ($\lambda, \gamma, \alpha, \beta, \epsilon, \kappa, n_c, W_0,
> W_{\min}, W_{\max}, r_{\max}, T_{\text{out}}, L_{\text{target}}, \theta_{\text{link}}, k_s, k, \text{MaxSteps}$). The
> controller parameters $\alpha, \beta, \epsilon, \kappa, n_c$ are chosen on the development split; the others are
> fixed a priori ($\lambda, \gamma$ only weight the reported metric $J$). All values are reported in Section IV.
> (This replaces the first bullet of "Notes for the authors"; it stays an author note, not paper text.)

---

## References to add (checked 2026-11-14)
- [RFC5681] M. Allman, V. Paxson, and E. Blanton, "TCP Congestion Control," RFC 5681, IETF, Sep. 2009,
  doi:10.17487/RFC5681. Checked: Section 3.1, "upon a timeout ... cwnd MUST be set to no more than the loss window,
  LW, which equals 1 full-sized segment". https://www.rfc-editor.org/rfc/rfc5681.txt
- [SRE-SLO] C. Jones, J. Wilkes, N. Murphy, and C. Smith, "Service Level Objectives," in *Site Reliability
  Engineering*, B. Beyer, C. Jones, J. Petoff, and N. R. Murphy, Eds. O'Reilly Media, 2016, ch. 4. Checked: section
  "Objectives in Practice", subsection "SLOs Set Expectations", tactic "Keep a safety margin: Using a tighter internal
  SLO than the SLO advertised to users gives you room to respond to chronic problems...". Used only as an analogy.
  https://sre.google/sre-book/service-level-objectives/

## Evidence for the p95 bias statement (checked 2026-11-14)
For a continuous i.i.d. sample of size n, the k-th order statistic has expected CDF value k/(n+1). numpy's default
p95 interpolates between order statistics k = floor(0.95(n-1))+1 and k+1, so its expected CDF value is at most
(k+1)/(n+1): 8/9, 16/17, 31/33 for n = 8, 16, 32, i.e. at least 11.1%, 5.9%, 6.1% of attempts above it. (A simulation
with the RQ2 server's lognormal noise, sigma = 0.25, gives about 13.8%, 9.5%, 7.4%; not cited.)

