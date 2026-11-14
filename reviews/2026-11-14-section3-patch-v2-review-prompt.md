# Review request: Methodology text additions (Section III patch v2), 2026-11-14

You are a strict thesis examiner and IEEE reviewer. Be concrete; no praise.

## Context
Paper "Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control". ACC is an
AIMD controller of a global concurrency window shared by LLM agents calling an API. The experiment design added client
timeouts (counted as overload signals, like HTTP 429/503) and a controller latency target below the SLO,
L_ctrl = kappa * L_target. The patch below changes the Methodology text accordingly; the current Methodology text is
reproduced after it. Implementation: https://github.com/notanut/eem-acc-skripsi (src/eem_acc/acc/controller.py,
src/eem_acc/agent/gateway.py).

## Checklist
1. Is every statement true and precise (control-interval timing, overshoot argument, tolerance band, effective gain,
   lower-bound latency of timeouts, the window definition after timeouts, Algorithm 1 lines 6-8)?
2. Are the two references (RFC 5681 Sec. 3.1; Google SRE book ch. 4 "Keep a safety margin") used for claims they
   support, without overstating?
3. Consistency with the rest of the Methodology (notation, Eq. 1, Algorithm 1, EEM triggers).
4. Clarity and length for an IEEE-style paper; anything an examiner would challenge (e.g. is kappa a tuned fudge
   factor? is it fair to baselines?).

## Output
Findings ranked by severity with location and a concrete rewording; then a verdict.

## The patch
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


## Current Methodology text (docs/section3_revised.md)
# III. METHODOLOGY

The system couples symbolic cognitive reasoning with system-level execution management through four modules deployed together: the Agentic Planning and Execution Engine, the Execution Experience Memory (EEM), the Adaptive Concurrency Controller (ACC), and the API Gateway Interface. EEM and ACC are complementary: they address different failure classes (API-contract errors versus infrastructure overload) and share no runtime signal; their interaction is evaluated empirically through a factorial design (Section IV).

## A. Problem Formulation

Let $\mathcal{T}$ denote a high-level user task decomposed into $N$ sub-goals $(g_1, g_2, \dots, g_N)$. At step $t$, the agent receives an environment state $s_t \in \mathcal{S}$ and retrieves a set of candidate API tools $\mathcal{A}_t \subset \mathcal{A}$. It produces an action $a_t \in \mathcal{A}_t$ with input arguments $x_t \in \mathcal{X}$ that advances $s_t$ to $s_{t+1}$. One *step* consists of one LLM generation and one dispatched API call; retries performed by the gateway (Section III-C) do not consume steps. A task is *successful* if all sub-goals $g_1,\dots,g_N$ are verified by a programmatic checker against the ground-truth environment state within $\text{MaxSteps}$ steps.

Under $M$ concurrent agents sharing API endpoints, let $j$ index API call attempts dispatched through the gateway. Let $L_j$ denote the server round-trip latency of attempt $j$ (excluding gateway queue wait), $f_j \in \{0,1\}$ indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable), and $L_{\text{target}}$ the p95 latency Service Level Objective (SLO). For a set of tasks $\mathcal{T}_{\text{set}}$ and the set of attempts $\mathcal{J}$ made while solving them, we define the evaluation objective

$$
J = \frac{1}{|\mathcal{T}_{\text{set}}|}\sum_{\mathcal{T}\in\mathcal{T}_{\text{set}}}\mathbb{1}\big[\text{Success}_{\mathcal{T}}\big]
\;-\; \lambda\,\frac{1}{|\mathcal{J}|}\sum_{j\in\mathcal{J}}\max\!\Big(0,\frac{L_j-L_{\text{target}}}{L_{\text{target}}}\Big)
\;-\; \gamma\,\frac{1}{|\mathcal{J}|}\sum_{j\in\mathcal{J}} f_j ,
$$

where $\lambda,\gamma \ge 0$ are dimensionless weights balancing task success against infrastructure penalties. $J$ is a reported metric, not a quantity maximized by a solver: the memory-augmented prompt state $\theta$ and the concurrency window $W$ are adapted online by EEM and ACC, respectively.

## B. Execution Experience Memory (EEM)

EEM provides cognitive adaptation through a three-stage reflection pipeline building on verbal reinforcement principles [1], [4].

A single store $\mathcal{M}$ is shared by all $M$ agents, so a lesson learned by one agent is available to the others.

**Trigger.** EEM handles only *non-transient* errors: HTTP 4xx other than 408 and 429, schema or validation errors, and HTTP 200 responses carrying an error payload. Errors 429, 503, and timeouts are handled by the gateway (Section III-C) and never generate reflections. When an action $a_t$ yields such an exception state $e_t = (\text{StatusCode}, \text{ErrorPayload})$, the interceptor captures it. To avoid consuming context with raw tracebacks, $e_t$ is processed as follows.

1. **Failure trajectory parsing.** $e_t$ is converted into the structured tuple $k_t = (\text{API\_Name}, \text{Schema\_Expected}, \text{Schema\_Received}, \text{Error\_Type})$.
2. **Verbal reflection synthesis.** The same LLM that drives the agent produces a concise natural-language fix instruction $c_t = \text{LLM}_{\text{reflect}}(s_t, a_t, k_t)$.
3. **Memory storage and retrieval.** A note $n=(c_t,k_t,z)$ with embedding $z=\text{Emb}(c_t \,\Vert\, k_t)$ is inserted into a Zettelkasten-inspired store $\mathcal{M}$ [9]:
   - *Merge:* if a note with the same $(\text{API\_Name},\text{Error\_Type})$ key exists, it is updated with the newer reflection instead of duplicated.
   - *Link:* the note is linked to existing notes whose cosine similarity with $z$ exceeds $\theta_{\text{link}}$.
   - *Retrieve:* at step $t$ the query is $q_t=\text{Emb}(s_t \,\Vert\, \text{desc}(\mathcal{A}_t))$. The top-$k_s$ notes by cosine similarity are taken as seeds and expanded with their 1-hop linked neighbours (ranked by similarity to $q_t$) until $k$ notes are selected. The selected reflections $\mathcal{M}_{\text{rel}}$ enrich the prompt $\theta_{t+1}$ [8].

## C. Adaptive Concurrency Controller (ACC)

The ACC regulates execution dispatch between the planner and the API gateway, extending adaptive transaction management paradigms [18], [28].

**Window.** $W$ is the maximum number of in-flight API calls, shared globally across all $M$ agents and enforced by a semaphore (`AcquireSlot`). It is bounded by $W_{\min} \le W \le W_{\max}$ with $W_{\min}\ge 1$, integer-valued, initialized to $W_0$.

**Control interval and signals.** The window is updated once every $n_c$ completed attempts (a control interval $\tau$), using only the attempts of the current interval, so earlier events are not counted twice. Define

- the throttle rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts returning 429 or 503), and
- the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{target}}}{L_{\text{target}}}$, where $\hat{L}^{95}_\tau$ is the p95 server latency of the attempts in interval $\tau$.

**Update rule (AIMD).** At most one multiplicative decrease occurs per interval:

- If $F_\tau > 0$ (throttled): $W_{\tau+1} = \max\big(W_{\min}, \lfloor W_\tau (1-\beta F_\tau)\rfloor\big)$
- Else if $\Delta L_\tau > \epsilon$ (latency degradation): $W_{\tau+1} = \max\big(W_{\min}, \lfloor W_\tau (1-\alpha \Delta L_\tau)\rfloor\big)$
- Else if $\Delta L_\tau \le 0$ (healthy): $W_{\tau+1} = \min(W_\tau + 1, W_{\max})$
- Otherwise ($0<\Delta L_\tau\le\epsilon$, tolerance band): $W_{\tau+1}=W_\tau$

**Retry policy.** The gateway retries attempts that return 429, 503, or time out, using truncated exponential backoff with jitter (honouring `Retry-After` when present) for at most $r_{\max}$ retries. Every attempt, including retries, enters the statistics above. Retries do not consume agent steps. If retries are exhausted, the agent receives a transient-failure observation, which counts as one step and generates no reflection.

**Algorithm 1: Self-Reflective Agentic Execution with Adaptive Concurrency Control**

```
Input:  Task T, Tools A, Initial window W0, Memory M, Target latency L_target
Output: Task result R_final or Failure report
 1: W_current <- W0;  s_1 <- InitializeState(T)
 2: for step t = 1 to MaxSteps do
 3:   M_rel <- RetrieveNotes(M, s_t, A_t, k_s, k)
 4:   (Thought_t, Action_t, Args_t) <- LLM_Generate(ConstructPrompt(s_t, A, M_rel))
 5:   if Action_t == Finish then return Args_t
 6:   AcquireSlot(W_current)
 7:   (Status, Payload) <- DispatchWithRetry(Action_t, Args_t)   // attempts logged to ACC, Section III-C
 8:   ReleaseSlot();  if control interval complete: W_current <- AdaptWindow(W_current, F_tau, DeltaL_tau)
 9:   if Status == SUCCESS then
10:     s_{t+1} <- UpdateState(s_t, Payload)
11:   else if IsNonTransient(Status, Payload) then
12:     k_t <- ParseError(Payload)
13:     Reflection_t <- LLM_Reflect(s_t, Action_t, Args_t, k_t)
14:     StoreNote(M, Reflection_t, k_t)                              // merge + link
15:     s_{t+1} <- InjectErrorContext(s_t, Reflection_t)
16:   else                                                            // retries exhausted (429/503/timeout)
17:     s_{t+1} <- InjectTransientNotice(s_t)
18:   end
19: end
20: return ExecutionFailed("Max steps exceeded")
```

_Note:_ the window update (line 8) is driven by the shared gateway statistics of all agents, not by a single agent's call. In the multi-agent setting, $M$ copies of this loop run concurrently and share both the window $W$ and the EEM store $\mathcal{M}$.

---

## Notes for the authors (not part of the paper text)

- Hyperparameter values are intentionally symbolic in Section III ($\lambda,\gamma,\alpha,\beta,\epsilon,W_0,W_{\min},W_{\max},n_c,r_{\max},\theta_{\text{link}},k_s,k,\text{MaxSteps},L_{\text{target}}$); they are fixed by the dev-split procedure and reported in Section IV.
- Decided: the EEM store is shared across agents; a private-per-agent store is run only as a small ablation.

### Required edits elsewhere in the paper

1. **Abstract:** currently empty; the sentence "An Adaptive Concurrency Controller (ACC) that dynamically regulates request dispatch rates ..." is duplicated at the start of the Introduction and should be removed from there. The Abstract is written after the results exist.
2. **Introduction, "tightly coupled":** replace "two tightly coupled components" with "two complementary components".
3. **Introduction, last sentence:** "By unifying ... the proposed system achieves high task completion rates while ensuring underlying infrastructure stability" is an unverified outcome claim; rephrase as the goal (e.g. "aims to achieve ...") until Section IV supports it.
4. **Introduction, "(ATCC)":** ATCC is the name of method [18], not an abbreviation of "adaptive concurrency control"; reword.
5. **Section III opening line:** "four tightly coupled modules" -> "four modules deployed together" (done in the text above).
6. **Related Works / Introduction citations:** every cited work's claims must be verified against the source before submission (not done yet; Chapter IV will only cite verified papers).

