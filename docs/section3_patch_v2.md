# Section III patch v2, revision 6 (follows RQ2 design v2, decisions D21-D22, D29-D31, D33) — DRAFT for the user

_Applies to `docs/section3_revised.md` (and later the paper). History: r2 longer justification (user's request; notation
and tone unchanged; two checked references); r3 fixes from an internal review; **r4 (2026-11-14) fixes from a second
opinion (external LLM) accepted by the user**: the small-sample bias of the p95 estimate (the "about 5%" claim was
wrong), the band argument kappa <= 1/(1+eps), the window definition after a decrease, the retry set (D29), no-latency
intervals (D30), "overload rate", a non-HTTP timeout code, Algorithm 2, and a clause on the server-side SLO.
**r5:** review fixes (exact bounds 1/9, 1/17, 2/33; list append in Algorithm 2; 'in F' qualifier; band condition
reported, not imposed; full hyperparameter list). **r6:** 500/502/504 are retried too (D33); EEM trigger sentence and
Algorithm 1 line 14 comment updated accordingly (new P17).
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
> **Retry policy.** The gateway retries attempts that return 408 (Request Timeout), 429 (Too Many Requests), 500, 502,
> 503 or 504 (server-side and gateway errors that are usually transient), or that exceed the client timeout
> $T_{\text{out}}$, using truncated exponential backoff with jitter (honouring `Retry-After` when
> present) for at most $r_{\max}$ retries; no slot is held during backoff, and a retry after a timeout is a new
> request for the server. Any other response, successful or not, is returned to the agent at once (non-transient
> errors are handled by EEM, Section III-B). Only 429, 503 and timeouts are overload signals ($f_j = 1$); 408, 500,
> 502 and 504 are retried but do not enter $F_\tau$.

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
 7:   if status ∉ {408, 429, 500, 502, 503, 504, TIMEOUT} then return (status, payload)
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

## P17 (III-B EEM trigger; Algorithm 1 comment) — D33

**EEM trigger. Old:**
> **Trigger.** EEM handles only *non-transient* errors: HTTP 4xx other than 408 and 429, schema or validation errors,
> and HTTP 200 responses carrying an error payload. Errors 429, 503, and timeouts are handled by the gateway
> (Section III-C) and never generate reflections.

**New:**
> **Trigger.** EEM handles only *non-transient* errors: any HTTP error status that the gateway does not retry (4xx
> other than 408 and 429, and 5xx other than 500, 502, 503 and 504), schema or validation errors, and HTTP 200
> responses carrying an error payload. Errors that the gateway retries (408, 429, 500, 502, 503, 504 and client
> timeouts; Section III-C) never generate reflections; if their retries are exhausted, the agent receives a
> transient-failure observation.

**Algorithm 1, renumbered line 14 (old line 16). Old comment:** `// retries exhausted (429/503/timeout)`
**New comment:** `// retries exhausted (retried status or timeout)`

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
