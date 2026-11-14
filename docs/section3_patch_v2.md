# Section III patch v2, revision 3 (follows RQ2 design v2, decisions D21-D22) — DRAFT for the user

_Applies to `docs/section3_revised.md` (and later the paper). Revision 2 (2026-11-14): longer justification at the
user's request; notation and tone unchanged; two well-known references added. Revision 3: corrections from an
independent review (a conditionally false claim, "conservative" replaced by "lower bound", the Window definition and
Algorithm 1 lines 6-8 made consistent with timeouts and per-attempt slots, the TCP and SRE comparisons toned down).
Exact old/new text below. Nothing else in Section III changes._

---

## P13 (III-A, evaluation objective): client timeouts are overload signals (D21)

**Old** (definition of $f_j$, paragraph before Eq. 1):
> $f_j \in \{0,1\}$ indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable),

**New:**
> $f_j \in \{0,1\}$ indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable), or
> was abandoned by the client after the timeout $T_{\text{out}}$ (in which case $L_j = T_{\text{out}}$),

**New sentences, appended to the paragraph that follows Eq. 1:**
> A timed-out attempt is counted as an overload signal because, like a 429 or 503 response, it shows that the server
> could not serve the request within the time the client is willing to wait. TCP likewise treats a retransmission
> timeout as a congestion signal and reduces its congestion window to one segment [RFC5681, Sec. 3.1]; ACC weights a
> timeout exactly like a 429 or 503. For such an attempt we set $L_j = T_{\text{out}}$, a lower bound on its true
> service time, so the latency term of $J$ understates the cost of a timeout; the attempt is additionally penalised
> through $f_j$. We choose $T_{\text{out}} > L_{\text{target}}$, so every timeout also counts as an SLO violation: the
> SLO violation rate is measured over successful attempts and timed-out attempts.

---

## P14 (III-C, controller signals): timeouts in $F_\tau$ (D21) and a controller target below the SLO (D22)

**Old** (the two signals of the control interval):
> - the throttle rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts returning 429 or 503), and
> - the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{target}}}{L_{\text{target}}}$, where
>   $\hat{L}^{95}_\tau$ is the p95 server latency of the attempts in interval $\tau$.

**New:**
> - the throttle rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts in interval $\tau$ that
>   returned 429 or 503 or timed out), and
> - the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{ctrl}}}{L_{\text{ctrl}}}$, where
>   $\hat{L}^{95}_\tau$ is the p95 server latency of the successful (HTTP 200) attempts in interval $\tau$, and
>   $\Delta L_\tau = 0$ if the interval has none, so that latency alone never shrinks the window without a latency
>   measurement. An interval consisting only of 429, 503 and timeouts is handled by the throttle branch ($F_\tau > 0$).
>   A 200 response whose payload EEM treats as an error (Section III-B) still measures server latency and is counted
>   here. The controller's latency target is
>   $$L_{\text{ctrl}} = \kappa\, L_{\text{target}}, \qquad 0 < \kappa \le 1 .$$

**New paragraph, directly after the update rules:**
> *Controller target.* The controller regulates latency to $L_{\text{ctrl}}$ rather than to the SLO value
> $L_{\text{target}}$ itself, for two reasons. First, $L_{\text{target}}$ is a p95 objective: holding the interval
> p95 exactly at $L_{\text{target}}$ puts the long-run violation rate at about 5% at the set point, so any estimation
> noise (the p95 is estimated from only $n_c$ samples) or overshoot pushes it above the SLO. Second, the additive
> increase deliberately probes for spare capacity: the window grows by one after every healthy interval. A probe that
> crosses the server's capacity knee raises latency above the set point until a later interval triggers a decrease,
> and that interval still partly consists of attempts dispatched under the previous window; overshoots within the
> tolerance band are not corrected at all. The overshoot is largest when one extra concurrent call is a large fraction
> of the server's capacity. Regulating to a tighter internal target absorbs both effects, analogous to the common
> practice of keeping a safety margin between an internal objective and the objective promised to users [SRE-SLO].
> The margin trades throughput for compliance: a smaller $\kappa$ produces fewer SLO violations but holds a smaller
> window. Because $\Delta L_\tau$ and $\epsilon$ are relative to $L_{\text{ctrl}}$, the tolerance band in units of
> $L_{\text{target}}$ is $\left(\kappa L_{\text{target}},\, \kappa(1+\epsilon) L_{\text{target}}\right]$, and the
> effective gain of $\alpha$ with respect to $L_{\text{target}}$ grows by $1/\kappa$. $\kappa = 1$ recovers the base
> algorithm and is always reported alongside the tuned value; like the other controller hyperparameters, $\kappa$ is
> chosen on the development split (Section IV).

---

## P15 (III-C, window definition and retry policy)

**Window. Old:**
> **Window.** $W$ is the maximum number of in-flight API calls, shared globally across all $M$ agents and enforced by
> a semaphore (`AcquireSlot`).

**New:**
> **Window.** $W$ is the maximum number of attempts the client is waiting on at any time, shared globally across all
> $M$ agents and enforced by a semaphore (`AcquireSlot`); one slot is held per attempt and released when the attempt
> completes or times out. After a client timeout the server may still be processing the abandoned attempt, so the
> concurrency seen by the server can briefly exceed $W$.

(The rest of the paragraph is unchanged.)

**Retry policy. Old:**
> The gateway retries attempts that return 429, 503, or time out, using truncated exponential backoff with jitter
> (honouring `Retry-After` when present) for at most $r_{\max}$ retries.

**New:**
> The gateway retries attempts that return 429 or 503, or that exceed the client timeout $T_{\text{out}}$, using
> truncated exponential backoff with jitter (honouring `Retry-After` when present) for at most $r_{\max}$ retries. No
> slot is held during backoff, and a retry after a timeout is a new request for the server.

(The rest of the paragraph is unchanged.)

---

## P16 (Algorithm 1 and hyperparameter list)

**Input line. Old:** `Input: Task T, Tools A, Initial window W0, Memory M, Target latency L_target`
**New:** `Input: Task T, Tools A, Initial window W0, Memory M, Target latency L_target, margin kappa, client timeout T_out`

**Lines 6-8. Old:**
```
 6:   AcquireSlot(W_current)
 7:   (Status, Payload) <- DispatchWithRetry(Action_t, Args_t)   // attempts logged to ACC, Section III-C
 8:   ReleaseSlot();  if control interval complete: W_current <- AdaptWindow(W_current, F_tau, DeltaL_tau)
```
**New** (line numbers unchanged; makes the per-attempt slot explicit, as the gateway implements it):
```
 6:   (Status, Payload) <- DispatchWithRetry(Action_t, Args_t)   // Section III-C
 7:     // each attempt: AcquireSlot(W_current); send with timeout T_out; log outcome to ACC; if the control
 8:     // interval is complete: W_current <- AdaptWindow(...) BEFORE ReleaseSlot(); no slot is held during backoff
```

**Hyperparameter list** (last paragraph of Section III): add $\kappa$ and $T_{\text{out}}$ to the symbolic
hyperparameters fixed by the dev-split procedure and reported in Section IV.

---

## References to add (checked 2026-11-14)
- [RFC5681] M. Allman, V. Paxson, and E. Blanton, "TCP Congestion Control," RFC 5681, IETF, Sep. 2009. Checked:
  Section 3.1 states that "upon a timeout ... cwnd MUST be set to no more than the loss window, LW, which equals
  1 full-sized segment". https://www.rfc-editor.org/rfc/rfc5681.txt
- [SRE-SLO] C. Jones, J. Wilkes, N. Murphy, and C. Smith, "Service Level Objectives," in *Site Reliability
  Engineering*, B. Beyer, C. Jones, J. Petoff, and N. R. Murphy, Eds. O'Reilly Media, 2016, ch. 4. Checked: section
  "Objectives in Practice", tactic "Keep a safety margin: Using a tighter internal SLO than the SLO advertised to
  users gives you room to respond to chronic problems...". Used only as an analogy (its rationale is operational
  headroom, not control overshoot). https://sre.google/sre-book/service-level-objectives/

## Note for Chapter IV (not Section III)
Evaluation setup: 120-s horizon runs, four scenarios, loaded cells fixed a priori, per-cell constraints (SLO <= 5%,
success >= 99% over calls requested before T - 20 s, throttle <= 10%), kappa in {0.8, 0.9, 1.0} on the dev split,
T_out = 1.0 s (> L_target = 0.45 s), pre-registered comparisons, and the disclosure that the scenarios were
redesigned after v1's calibration showed no benefit of adaptation.
