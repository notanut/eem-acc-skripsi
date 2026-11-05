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
