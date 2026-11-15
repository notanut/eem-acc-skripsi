# Review request: final Section III text (patch r7 applied) and HTTP 504 as an overload signal, 2026-11-15

You are a strict thesis examiner and IEEE reviewer. Be concrete; no praise.

## Context
Paper "Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control". ACC is an
AIMD controller of a global concurrency window shared by LLM agents calling an API; EEM turns non-transient API errors
into stored reflections. You (or another reviewer) previously reviewed patch revision 6 and asked for two fixes (M1:
retried non-overload errors 408/500/502 must be stated to count in the control interval with f_j = 0; M2: EEM
parameters and MaxSteps are not chosen on the RQ2 development split) and a decision on 504. Decision taken (D38): a
504 Gateway Timeout is an overload signal (f_j = 1), stays retryable, and is not a latency sample. Optional polish was
also applied: "typically return quickly ... little or no excess", the hold rule now says when it can happen, a
Hyndman-Fan reference for "sample p95", the third kappa reason reworded, Algorithm 2 has a Parameters line and an
atomicity note. After an internal review: a 504 clause in the J paragraph, "a 4xx other than 429", the label
"(overload)" instead of "(throttled)". The simulated server in the experiments never returns 504, so no experimental result changes (60
stored runs were re-simulated with the new code: identical).

Implementation: https://github.com/notanut/eem-acc-skripsi (src/eem_acc/status.py, src/eem_acc/acc/controller.py,
src/eem_acc/agent/gateway.py). The status sets in code are now:
```
OVERLOAD       = (429, 503, 504, CLIENT_TIMEOUT)          # f_j = 1
RETRYABLE      = (408, 429, 500, 502, 503, 504, CLIENT_TIMEOUT)
LATENCY_SAMPLE = (200, CLIENT_TIMEOUT)                    # timeouts counted at T_out
```

## Checklist
1. Were M1, M2 and the 504 decision applied fully and consistently (Section III-A f_j definition, III-B EEM trigger,
   III-C signals, hold rule, retry policy, Algorithms 1 and 2)? Any sentence that still contradicts them?
2. Is every statement true and precise? In particular: the hold rule ("happens only when every attempt of the interval
   is a non-overload error"), the p95 bias bounds (1/9, 1/17, 2/33 for n_c = 8, 16, 32, numpy default / Hyndman-Fan
   type 7), and "a 504 from an intermediate proxy reports the same about the upstream server".
3. Are the references ([HF96], [RFC5681] Sec. 3.1, [SRE-SLO]) used only for what they support?
4. Is the text now ready to go into the paper? Anything an examiner would still challenge?
5. The "Notes for the authors" part at the end is not paper text; check only that it is consistent.

## Output
Findings ranked by severity with location and a concrete rewording; then a verdict (approve / approve with fixes /
reject).

## Final Section III text (markdown with LaTeX math)

# III. METHODOLOGY

<!-- Section III patch v2 r7 (docs/section3_patch_v2.md) applied 2026-11-15 -->

The system couples symbolic cognitive reasoning with system-level execution management through four modules deployed together: the Agentic Planning and Execution Engine, the Execution Experience Memory (EEM), the Adaptive Concurrency Controller (ACC), and the API Gateway Interface. EEM and ACC are complementary: they address different failure classes (API-contract errors versus infrastructure overload) and share no runtime signal; their interaction is evaluated empirically through a factorial design (Section IV).

## A. Problem Formulation

Let $\mathcal{T}$ denote a high-level user task decomposed into $N$ sub-goals $(g_1, g_2, \dots, g_N)$. At step $t$, the agent receives an environment state $s_t \in \mathcal{S}$ and retrieves a set of candidate API tools $\mathcal{A}_t \subset \mathcal{A}$. It produces an action $a_t \in \mathcal{A}_t$ with input arguments $x_t \in \mathcal{X}$ that advances $s_t$ to $s_{t+1}$. One *step* consists of one LLM generation and one dispatched API call; retries performed by the gateway (Section III-C) do not consume steps. A task is *successful* if all sub-goals $g_1,\dots,g_N$ are verified by a programmatic checker against the ground-truth environment state within $\text{MaxSteps}$ steps.

Under $M$ concurrent agents sharing API endpoints, let $j$ index API call attempts dispatched through the gateway. Let $L_j$ denote the server round-trip latency of attempt $j$ (excluding gateway queue wait), $f_j \in \{0,1\}$ indicate an *overload signal*, i.e. that attempt $j$ returned HTTP 429 (Too Many Requests), 503 (Service Unavailable) or 504 (Gateway Timeout), or was abandoned by the client after the timeout $T_{\text{out}}$ (in which case $L_j = T_{\text{out}}$), and $L_{\text{target}}$ the p95 latency Service Level Objective (SLO). The SLO is server-side: it is judged on HTTP 200 attempts and timed-out attempts, and it excludes time spent waiting in the gateway queue; client-side waiting is therefore reported separately (end-to-end latency and queue length, Section IV). For a set of tasks $\mathcal{T}_{\text{set}}$ and the set of attempts $\mathcal{J}$ made while solving them, we define the evaluation objective

$$
J = \frac{1}{|\mathcal{T}_{\text{set}}|}\sum_{\mathcal{T}\in\mathcal{T}_{\text{set}}}\mathbb{1}\big[\text{Success}_{\mathcal{T}}\big]
\;-\; \lambda\,\frac{1}{|\mathcal{J}|}\sum_{j\in\mathcal{J}}\max\!\Big(0,\frac{L_j-L_{\text{target}}}{L_{\text{target}}}\Big)
\;-\; \gamma\,\frac{1}{|\mathcal{J}|}\sum_{j\in\mathcal{J}} f_j ,
$$

where $\lambda,\gamma \ge 0$ are dimensionless weights balancing task success against infrastructure penalties. $J$ is a reported metric, not a quantity maximized by a solver: the memory-augmented prompt state $\theta$ and the concurrency window $W$ are adapted online by EEM and ACC, respectively. A timed-out attempt is an overload signal because, like a 429 or 503 response, it shows that the server could not serve the request within the time the client is willing to wait; a 504 from an intermediate proxy reports the same about the upstream server. TCP also treats a retransmission timeout as a congestion signal, and reacts to it more strongly than to other loss signals by reducing its congestion window to one segment [RFC5681, Sec. 3.1]; in $F_\tau$, ACC does not distinguish a timeout from a 429, 503 or 504. For a timed-out attempt we set $L_j = T_{\text{out}}$, a lower bound on its true round-trip latency, so the latency term of $J$ understates the cost of a timeout; the attempt is additionally penalised through $f_j$. Since $T_{\text{out}} > L_{\text{target}}$, every timeout also counts as an SLO violation. In $J$, the latency term averages over all attempts, so 429/503 responses (which typically return quickly) contribute little or no excess, whereas a 504, which a proxy typically returns only after its own timeout, can contribute a large one; the SLO violation rate reported in Section IV is computed over HTTP 200 and timed-out attempts only.

## B. Execution Experience Memory (EEM)

EEM provides cognitive adaptation through a three-stage reflection pipeline building on verbal reinforcement principles [1], [4].

A single store $\mathcal{M}$ is shared by all $M$ agents, so a lesson learned by one agent is available to the others.

**Trigger.** EEM handles only *non-transient* errors: any HTTP error status that the gateway does not retry (4xx other than 408 and 429, and 5xx other than 500, 502, 503 and 504), schema or validation errors, and HTTP 200 responses carrying an error payload. Errors that the gateway retries (408, 429, 500, 502, 503, 504 and client timeouts; Section III-C) never generate reflections; if their retries are exhausted, the agent receives a transient-failure observation. When an action $a_t$ yields such an exception state $e_t = (\text{StatusCode}, \text{ErrorPayload})$, the interceptor captures it. To avoid consuming context with raw tracebacks, $e_t$ is processed as follows.

1. **Failure trajectory parsing.** $e_t$ is converted into the structured tuple $k_t = (\text{API\_Name}, \text{Schema\_Expected}, \text{Schema\_Received}, \text{Error\_Type})$.
2. **Verbal reflection synthesis.** The same LLM that drives the agent produces a concise natural-language fix instruction $c_t = \text{LLM}_{\text{reflect}}(s_t, a_t, k_t)$.
3. **Memory storage and retrieval.** A note $n=(c_t,k_t,z)$ with embedding $z=\text{Emb}(c_t \,\Vert\, k_t)$ is inserted into a Zettelkasten-inspired store $\mathcal{M}$ [9]:
   - *Merge:* if a note with the same $(\text{API\_Name},\text{Error\_Type})$ key exists, it is updated with the newer reflection instead of duplicated.
   - *Link:* the note is linked to existing notes whose cosine similarity with $z$ exceeds $\theta_{\text{link}}$.
   - *Retrieve:* at step $t$ the query is $q_t=\text{Emb}(s_t \,\Vert\, \text{desc}(\mathcal{A}_t))$. The top-$k_s$ notes by cosine similarity are taken as seeds and expanded with their 1-hop linked neighbours (ranked by similarity to $q_t$) until $k$ notes are selected. The selected reflections $\mathcal{M}_{\text{rel}}$ enrich the prompt $\theta_{t+1}$ [8].

## C. Adaptive Concurrency Controller (ACC)

The ACC regulates execution dispatch between the planner and the API gateway, extending adaptive transaction management paradigms [18], [28].

**Window.** $W$ limits the number of *outstanding* attempts (sent, and neither answered nor timed out), shared globally across all $M$ agents: an attempt is dispatched only when fewer than $W$ attempts are outstanding (`AcquireSlot`), and its slot is released when the attempt is answered or times out. A decrease of $W$ does not cancel outstanding attempts, so their number can exceed the new $W$ until enough of them finish. The server may continue an attempt the client has abandoned, so the server's concurrency can exceed $W$ until that work finishes; Section IV reports the maximum server concurrency. $W$ is bounded by $W_{\min} \le W \le W_{\max}$ with $W_{\min}\ge 1$, integer-valued, initialized to $W_0$.

**Control interval and signals.** The window is updated once every $n_c$ attempts that have an outcome (a response or a client timeout), forming a control interval $\tau$; only the attempts of the current interval are used, so earlier events are not counted twice. Define

- the overload rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts in interval $\tau$ that returned 429, 503 or 504 or timed out), and
- the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{ctrl}}}{L_{\text{ctrl}}}$, where $\hat{L}^{95}_\tau$ is the p95 latency of the latency samples of interval $\tau$: its HTTP 200 attempts and its timed-out attempts, the latter counted at $T_{\text{out}}$. A 200 response whose payload EEM treats as an error (Section III-B) still measures server latency and is a latency sample; 429, 503 and 504 responses and other errors are not. The controller's latency target is

$$
L_{\text{ctrl}} = \kappa\, L_{\text{target}}, \qquad 0 < \kappa \le 1 .
$$

**Update rule (AIMD).** At most one multiplicative decrease occurs per interval:

- If $F_\tau > 0$ (overload): $W_{\tau+1} = \max\big(W_{\min}, \lfloor W_\tau (1-\beta F_\tau)\rfloor\big)$
- Else if interval $\tau$ has no latency sample: $W_{\tau+1} = W_\tau$ (no evidence either way). This happens only when every attempt of the interval is a non-overload error (e.g. a 4xx other than 429, or a 500): an interval of only 429/503/504 responses has $F_\tau > 0$, and a timeout is a latency sample.
- Else if $\Delta L_\tau > \epsilon$ (latency degradation): $W_{\tau+1} = \max\big(W_{\min}, \lfloor W_\tau (1-\alpha \Delta L_\tau)\rfloor\big)$
- Else if $\Delta L_\tau \le 0$ (healthy): $W_{\tau+1} = \min(W_\tau + 1, W_{\max})$
- Otherwise ($0<\Delta L_\tau\le\epsilon$, tolerance band): $W_{\tau+1}=W_\tau$

*Controller target.* The controller regulates latency to $L_{\text{ctrl}} = \kappa L_{\text{target}}$ rather than to the SLO itself, for three reasons. First, with $\kappa = 1$ the tolerance band $(L_{\text{target}}, (1+\epsilon)L_{\text{target}}]$ lies entirely above the SLO, so the controller would hold the window while the interval p95 violates the SLO; the band lies at or below the SLO only if $\kappa \le 1/(1+\epsilon)$. Second, $\hat{L}^{95}_\tau$ is estimated from at most $n_c$ values and is biased low: for independent latencies from any continuous distribution, the expected fraction of attempts above the sample p95 (Hyndman-Fan type 7 [HF96], the common default) of $n_c \in \{8, 16, 32\}$ values is at least 1/9, 1/17 and 2/33 (11.1%, 5.9% and 6.1%), so holding the estimate at $L_{\text{target}}$ does not hold the violation rate at 5%. Third, additive increase probes past the server's capacity knee, and the interval that should detect a probe may consist partly, or when $n_c$ does not exceed the number of attempts still outstanding at the update entirely, of attempts dispatched before it. A tighter internal target provides headroom for these effects, analogous to keeping a safety margin between internal and user-facing objectives [SRE-SLO], at the cost of a smaller window. Because $\Delta L_\tau$ and $\epsilon$ are relative to $L_{\text{ctrl}}$, the effective gain of $\alpha$ with respect to $L_{\text{target}}$ is $\alpha/\kappa$. $\kappa$ is chosen on the development split together with the other controller hyperparameters, and $\kappa = 1$ is reported both with the same configuration and re-tuned (Section IV). The band condition $\kappa(1+\epsilon) \le 1$ is not imposed on the selection; whether the selected configuration satisfies it is reported.

**Retry policy.** The gateway retries attempts that return 408 (Request Timeout), 429 (Too Many Requests), 500, 502, 503 or 504 (server-side and gateway errors that are usually transient), or that exceed the client timeout $T_{\text{out}}$, using truncated exponential backoff with jitter (honouring `Retry-After` when present) for at most $r_{\max}$ retries; no slot is held during backoff, and a retry after a timeout is a new request for the server. Any other response, successful or not, is returned to the agent at once (non-transient errors are handled by EEM, Section III-B). Only 429, 503, 504 and timeouts are overload signals ($f_j = 1$); 408, 500 and 502 are retried and count as attempts of the control interval, but with $f_j = 0$, and they are not latency samples (nor is a 504). Every attempt, including retries, enters the statistics above. Retries do not consume agent steps. If retries are exhausted, the agent receives a transient-failure observation, which counts as one step and generates no reflection.

**Algorithm 1: Self-Reflective Agentic Execution with Adaptive Concurrency Control**

```
Input:  Task T, Tools A, Memory M, Target latency L_target, client timeout T_out,
        controller parameters Theta_ACC = (alpha, beta, epsilon, kappa, n_c, W_min, W_max, W0)
Output: Task result R_final or Failure report
 1: s_1 <- InitializeState(T)   // the window W is shared by all agents, initialised once to W0
 2: for step t = 1 to MaxSteps do
 3:   M_rel <- RetrieveNotes(M, s_t, A_t, k_s, k)
 4:   (Thought_t, Action_t, Args_t) <- LLM_Generate(ConstructPrompt(s_t, A, M_rel))
 5:   if Action_t == Finish then return Args_t
 6:   (Status, Payload) <- DispatchWithRetry(Action_t, Args_t)   // Algorithm 2
 7:   if Status == SUCCESS then
 8:     s_{t+1} <- UpdateState(s_t, Payload)
 9:   else if IsNonTransient(Status, Payload) then
10:     k_t <- ParseError(Payload)
11:     Reflection_t <- LLM_Reflect(s_t, Action_t, Args_t, k_t)
12:     StoreNote(M, Reflection_t, k_t)                         // merge + link
13:     s_{t+1} <- InjectErrorContext(s_t, Reflection_t)
14:   else                    // retries exhausted (retried status or timeout)
15:     s_{t+1} <- InjectTransientNotice(s_t)
16:   end
17: end
18: return ExecutionFailed("Max steps exceeded")
```

_Note:_ the window update (Algorithm 2, line 5) is driven by the shared gateway statistics of all agents, not by a single agent's call. In the multi-agent setting, $M$ copies of this loop run concurrently and share both the window $W$ and the EEM store $\mathcal{M}$.

**Algorithm 2: DispatchWithRetry**

```
Algorithm 2: DispatchWithRetry(a, x)   // shared by all agents: window W, interval buffer B
Parameters: n_c, r_max, T_out
 1: for r = 0 to r_max do
 2:   AcquireSlot(W)                   // wait until fewer than W attempts are outstanding
 3:   (status, payload, L) <- Send(a, x) with timeout T_out
                                        // on timeout: status <- TIMEOUT, L <- T_out
 4:   append (status, L) to B
 5:   if |B| = n_c then W <- AdaptWindow(W, F(B), ΔL(B)); B <- ∅   // before the slot is released
 6:   ReleaseSlot()
 7:   if status ∉ {408, 429, 500, 502, 503, 504, TIMEOUT} then return (status, payload)
 8:   if r < r_max then Sleep(max(RetryAfter, Backoff(r)))   // no slot held while waiting
 9: end for
10: return (status, payload)                             // retries exhausted
```

_Note:_ lines 4-6 execute without interleaving with other agents (in the asyncio implementation there is no suspension point between recording an outcome, adapting the window and releasing the slot), which is what makes the shared buffer B safe.

### References added in this revision

- [HF96] R. J. Hyndman and Y. Fan, "Sample quantiles in statistical packages," *The American Statistician*, vol. 50, no. 4, pp. 361-365, 1996, doi:10.1080/00031305.1996.10473566.
- [RFC5681] M. Allman, V. Paxson, and E. Blanton, "TCP Congestion Control," RFC 5681, IETF, Sep. 2009, doi:10.17487/RFC5681.
- [SRE-SLO] C. Jones, J. Wilkes, N. Murphy, and C. Smith, "Service Level Objectives," in *Site Reliability Engineering*, B. Beyer, C. Jones, J. Petoff, and N. R. Murphy, Eds. O'Reilly Media, 2016, ch. 4.

(These take the next free numbers in the paper's reference list.)

---

## Notes for the authors (not part of the paper text)

- Hyperparameter values are symbolic in Section III ($\lambda, \gamma, \alpha, \beta, \epsilon, \kappa, n_c, W_0, W_{\min}, W_{\max}, r_{\max}, T_{\text{out}}, L_{\text{target}}, \theta_{\text{link}}, k_s, k, \text{MaxSteps}$). The controller parameters $\alpha, \beta, \epsilon, \kappa, n_c$ are chosen on the development split; $\lambda, \gamma, W_0, W_{\min}, W_{\max}, r_{\max}, T_{\text{out}}, L_{\text{target}}$ are fixed a priori ($\lambda, \gamma$ only weight the reported metric $J$); the EEM parameters ($\theta_{\text{link}}, k_s, k$) and MaxSteps are set in Stage 3. All values are reported in Section IV.
- Decided: the EEM store is shared across agents; a private-per-agent store is run only as a small ablation.

### Required edits elsewhere in the paper

1. **Abstract:** currently empty; the sentence "An Adaptive Concurrency Controller (ACC) that dynamically regulates request dispatch rates ..." is duplicated at the start of the Introduction and should be removed from there. The Abstract is written after the results exist.
2. **Introduction, "tightly coupled":** replace "two tightly coupled components" with "two complementary components".
3. **Introduction, last sentence:** "By unifying ... the proposed system achieves high task completion rates while ensuring underlying infrastructure stability" is an unverified outcome claim; rephrase as the goal (e.g. "aims to achieve ...") until Section IV supports it.
4. **Introduction, "(ATCC)":** ATCC is the name of method [18], not an abbreviation of "adaptive concurrency control"; reword.
5. **Section III opening line:** "four tightly coupled modules" -> "four modules deployed together" (done in the text above).
6. **Related Works / Introduction citations:** every cited work's claims must be verified against the source before submission (not done yet; Chapter IV will only cite verified papers).
