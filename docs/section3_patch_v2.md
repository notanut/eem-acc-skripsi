# Section III patch v2, revision 2 (follows RQ2 design v2, decisions D21-D22) — DRAFT for the user

_Applies to `docs/section3_revised.md` (and later the paper). Revision 2 (2026-11-14): longer justification at the
user's request; notation and tone unchanged; two well-known references added (both checked against the sources,
see the end). Exact old/new text below. Nothing else in Section III changes._

---

## P13 (III-A, evaluation objective): client timeouts are overload signals (D21)

**Old** (definition of $f_j$, paragraph before Eq. 1):
> $f_j \in \{0,1\}$ indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable),

**New:**
> $f_j \in \{0,1\}$ indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable), or
> was abandoned by the client after the timeout $T_{\text{out}}$, in which case $L_j = T_{\text{out}}$,

**New sentences, appended to the paragraph that follows Eq. 1:**
> A timed-out attempt is counted as an overload signal because, like a 429 or 503 response, it shows that the server
> could not serve the request within the time the client is willing to wait; transport protocols treat timeouts the
> same way, and TCP reacts to a retransmission timeout with its strongest congestion response [RFC5681]. Setting
> $L_j = T_{\text{out}}$ for such an attempt is conservative: the true service time is at least $T_{\text{out}}$, and
> the server may still be processing the abandoned request, so it keeps consuming server capacity after the client
> has given up.

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
>   $\hat{L}^{95}_\tau$ is the p95 server latency of the *successful* attempts in interval $\tau$ ($\Delta L_\tau = 0$
>   if the interval has none; such an interval is handled by the throttle branch) and
>   $$L_{\text{ctrl}} = \kappa\, L_{\text{target}}, \qquad 0 < \kappa \le 1,$$
>   is the latency target of the controller.

**New paragraph, directly after the update rules:**
> *Controller target.* The controller regulates latency to $L_{\text{ctrl}}$ rather than to the SLO value
> $L_{\text{target}}$ itself, for two reasons. First, $L_{\text{target}}$ is a p95 objective: a controller that holds
> the p95 exactly at $L_{\text{target}}$ already lets about 5% of successful attempts exceed it at its set point, so
> it has no room left for error. Second, the additive increase deliberately probes for spare capacity: the window
> grows by one after every healthy interval, and a decrease can only follow once the next interval of $n_c$ attempts
> has completed. Each probe therefore overshoots the set point for a short time, and the overshoot is largest when one
> extra concurrent call is a large fraction of the server's capacity. Regulating to a tighter internal target absorbs
> both effects, in the same way as the common practice of keeping a safety margin between an internal objective and
> the objective promised to users [SRE-SLO]. The margin trades throughput for compliance: a smaller $\kappa$ produces
> fewer SLO violations but holds a smaller window. $\kappa = 1$ recovers the base algorithm. The tolerance $\epsilon$
> is relative to $L_{\text{ctrl}}$, so in units of $L_{\text{target}}$ the tolerance band is
> $\left(\kappa L_{\text{target}},\, \kappa(1+\epsilon) L_{\text{target}}\right]$. Like the other controller
> hyperparameters, $\kappa$ is chosen on the development split (Section IV).

---

## P15 (III-C retry policy, Algorithm 1, hyperparameter list)

**Retry policy. Old:**
> The gateway retries attempts that return 429, 503, or time out, using truncated exponential backoff with jitter
> (honouring `Retry-After` when present) for at most $r_{\max}$ retries.

**New:**
> The gateway retries attempts that return 429 or 503, or that exceed the client timeout $T_{\text{out}}$, using
> truncated exponential backoff with jitter (honouring `Retry-After` when present) for at most $r_{\max}$ retries. A
> timed-out attempt releases its slot in the window when the client gives up, although the server may still be
> processing it; its retry is a new request for the server.

(The rest of the paragraph is unchanged.)

**Algorithm 1, input line. Old:** `Input: Task T, Tools A, Initial window W0, Memory M, Target latency L_target`
**New:** `Input: Task T, Tools A, Initial window W0, Memory M, Target latency L_target, margin kappa, client timeout T_out`

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
  users gives you room to respond to chronic problems...". https://sre.google/sre-book/service-level-objectives/

## Note for Chapter IV (not Section III)
Evaluation setup: 120-s horizon runs, four scenarios, loaded cells fixed a priori, per-cell constraints (SLO <= 5%,
success >= 99% over calls requested before T - 20 s, throttle <= 10%), kappa in {0.8, 0.9, 1.0} on the dev split,
T_out = 1.0 s, pre-registered comparisons, and the disclosure that the scenarios were redesigned after v1's
calibration showed no benefit of adaptation.
