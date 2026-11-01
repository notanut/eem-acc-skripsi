# Section III patch proposal (DRAFT - not applied, needs user approval)

Purpose: close the gaps in the Methodology that the experiments depend on. Each item lists the problem, options, and a recommendation. Nothing here is final until approved; the paper text is **not** edited until then. Hyperparameter *values* (alpha, beta, epsilon, W0, W_min, W_max, tau, n, lambda, gamma, MaxSteps) are decided separately (D7).

## ACC

**P1. Latency signal and Delta L (currently undefined).**
- A. EMA of per-call server round-trip latency, `Delta L = (L_ema - L_target) / L_target`.
- B. p95 over a sliding window of the last `n` completed calls, same normalization. L_target is then a p95 target.
- **Recommend B**: SLOs are normally tail-based, and the objective already penalizes tail excess. Cost: noisier with small `n` (n becomes a hyperparameter).

**P2. Dead zone between the latency branches.** As written, `0 < Delta L <= epsilon` matches no case.
- A. State it explicitly: hold W when `0 < Delta L <= epsilon` (hysteresis).
- B. Increase whenever `Delta L <= epsilon`.
- **Recommend A**: it matches the original intent (epsilon is a tolerance band) and is more stable.

**P3. Throttle rate F_tau.**
- A. Fraction of the last `n_f` completed calls returning 429 **or** 503.
- B. 429 only.
- **Recommend A (count-based window)**: Eq. (1) already defines R_t over both codes, and a count window is deterministic and reproducible (no wall-clock dependence).

**P4. Meaning and scope of W; "pacer" vs "window".**
- A. W = maximum number of in-flight API calls, **global and shared across all M agents**, enforced by a semaphore. Replace `AcquireTokenFromPacer(W)` with `AcquireSlot(W)`.
- B. A, plus a token-bucket rate derived from W (Little's law, W / latency).
- **Recommend A**: simplest, directly matches "concurrency window"; rate pacing would add a second mechanism to ablate. Add bounds: `W in [W_min, W_max]`, `W_min >= 1`, integer W (floor, then clamp).

**P5. Update cadence.**
- A. Update once per **control interval** (every `n_c` completed calls); at most one multiplicative decrease per interval.
- B. Update after every completed call.
- **Recommend A**: correlated 429s from one burst would otherwise trigger several back-to-back cuts and collapse W (standard AIMD reacts once per round trip).

**P6. Retry policy for 429/503 (currently unspecified).**
- A. Gateway retries transparently with truncated exponential backoff (honouring `Retry-After` if present), up to `max_retries`. The agent only sees the error if retries are exhausted, and retried calls do not consume agent steps.
- B. Return the 429/503 to the agent as an ordinary failed step.
- **Recommend A**: infrastructure faults are not reasoning errors. Define L_t as **server round-trip latency per attempt** (excludes gateway queue wait); report end-to-end latency (incl. queueing and retries) separately as an extra metric.

## EEM

**P7. Reflection trigger (Algorithm 1, line 12 reflects on any non-success).**
- A. Reflect only on non-transient errors: 4xx other than 408/429, schema/validation errors, and HTTP 200 with an error payload. Never on 429, 503, or timeouts.
- B. Reflect on every failure (as written).
- **Recommend A as the main method, and run B as a named ablation ("reflect-on-all")**: transient errors waste LLM calls and fill memory with lessons that are not about the API contract. The ablation tests that claim instead of assuming it.

**P8. Retrieval query (line 3 retrieves before the action is chosen, so `k_t` is unknown).**
- A. Query = embedding of `s_t` only.
- B. Query = embedding of `s_t` + names/descriptions of candidate tools `A_t`.
- **Recommend B**: tool names are available at that point and reflections are API-specific. Change line 3 to `RetrieveTopKReflections(M, s_t, A_t, k)`.

**P9. What "Zettelkasten-inspired" means concretely (unspecified).**
- A. Notes linked on insert (cosine > `theta_link` to existing notes); notes with the same `(API_Name, Error_Type)` key are merged/updated instead of duplicated; retrieval = top-k by cosine only.
- B. A, plus 1-hop expansion at retrieval (linked neighbours fill remaining slots, total still capped at k).
- **Recommend B**: otherwise the link structure never affects retrieval and the "flat store" ablation would differ only by deduplication. Define: embedding model all-MiniLM-L6-v2 over `c_t` plus the fields of `k_t`; reflection model = the same local Qwen2.5-7B-Instruct (4-bit) used by the agent (no extra cost, avoids a stronger-reflector confound).

## Objective, success, and framing

**P10. Eq. (1) and "max over theta, W".** Nothing optimizes theta or W jointly; they are adapted online by EEM and ACC.
- A. Reframe Eq. (1) as an **evaluation objective J** (reported, not maximized by a solver). Make it dimensionless: latency excess `max(0, (L_t - L_target) / L_target)`, `R_t in [0, 1]`, so lambda and gamma are unit-free weights.
- B. Keep as written.
- **Recommend A**: removes an unsupported optimization claim and the unit mismatch.

**P11. Success, steps, L_target.**
- Task success = every sub-goal `g_1..g_N` verified by a **programmatic checker against ground-truth mock-server state**, within MaxSteps. One step = one LLM generation plus one dispatched call (retries do not count; see P6).
- L_target = SLO on p95 server round-trip latency; its value is derived from the mock server's latency model (proposal: a fixed multiple of unloaded p95), fixed per scenario before any run.
- **Recommend as stated** (alternative: LLM-judge for success - rejected: costly and noisy).

**P12. "Tightly coupled" claim.** In Algorithm 1 the modules share no signal: EEM never sees ACC state and vice versa.
- A. Soften the wording to "complementary modules deployed together" and test the interaction empirically through the 2x2 factorial (RQ3).
- B. Add a real coupling (e.g. expose window saturation to the agent). More complexity and more ablations.
- **Recommend A**; list B as future work.

## Editorial fixes (no decision needed unless you object)
- Abstract text currently sits at the start of the Introduction ("An Adaptive Concurrency Controller (ACC) that ...") and the Abstract is empty; move/replace when the Abstract is written.
- "adaptive concurrency control (ATCC)": ATCC is the name of reference [18], not the abbreviation of the phrase; reword.
- `t` is used both for agent step and for call/latency time; use `t` for steps and `tau`/call index `j` for latency and throttle windows.
- Verify every reference before it is cited in Chapter IV (not done yet).

## Revised Algorithm 1 lines (if all recommendations are approved)
- 3: `M_rel <- RetrieveTopKReflections(M, s_t, A_t, k=3)`
- 6: `AcquireSlot(W_current)`
- 7: `(Status, Payload, ServerLatency) <- DispatchWithRetry(Action_t, Args_t)`   (P6)
- 8: `W_current <- AdaptConcurrencyWindow(...)` once per control interval (P5)
- 11-12: `else if IsNonTransient(Status, Payload) then  Reflection_t <- LLM_Reflect(...)`   (P7)

## Reply format
"approve all", or per item e.g. "P1 B, P9 A, rest as recommended".
