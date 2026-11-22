# S3.2b specification: RQ3 wiring (gateway, server, background load, shared store) (CPU only)

_Draft r1, 2026-11-22. Implements design r7 section 7 (RQ3) on top of S3.1 (suite) and S3.2a (agent loop, EEM),
reusing the RQ2 simulator unchanged (`Gateway`, `MockServer`, `ACCController`, frozen ACC config). Open choices are
marked **(Q)** and collected in section 9._

## 1. Run = one RQ3 cell on one seed

```python
res = run_rq3_unit(spec, condition, window, backend, embedder, tokenizer, cfg) -> dict
#   spec: an S3.1 rq3 stream (20 tasks; agents [[0,4,..],[1,5,..],[2,6,..],[3,7,..]])
#   condition: "C0" | "C2" (+ private-store variant of C2); window: "none" | "acc" | "fixed4"
```

Cells (design section 7): {C0, C2} x {no window, ACC on} plus C2 + fixed W = 4; descriptive: C2 + ACC with a
private store; the 2x2 with LLM times / 3 (sensitivity).

## 2. Simulator composition (virtual time, one event loop)

- **Server:** RQ2 CAP-steady server (`ServerConfig` from `scenarios.py`: base latency 0.2 s, capacity 4, hard cap 8,
  ...), wrapped by `SemanticServer`:
  - rate-limit and hard-cap checks first (429/503, unchanged);
  - at **admission** (the request passed those checks) a call from an LLM agent is executed by the S3.1 suite with
    `call_key = (agent, k)` (S3.1 D68: a retry after a client timeout gets the stored response; state changes happen
    once, before the latency delay);
  - a semantic **200** (including a silent 200 error payload) then takes the normal service latency (base x noise x
    degradation) and is a latency sample for ACC; a semantic **4xx** is returned after `error_latency` (0.02 s, like
    429/503) **(Q1)** and is neither an overload signal nor a latency sample (Section III-C);
  - calls from scripted background agents carry no semantics: admitted calls take the normal latency and return 200
    (load only) **(Q4)**.
- **Request ids:** the gateway and server key their randomness by `(agent, k)`; LLM agents use ids 1000-1003 (never
  colliding with scripted agents 0-31 or the burst source id 32); `k` is the agent's running call counter (S3.2a).
  An LLM request is a tuple subclass `(agent, k)` that also carries (op, args, task_index), so the RQ2 code is
  unchanged.
- **Gateway:** RQ2 `Gateway` with RQ2's retry policy (r_max 6, backoff 0.1-4 s, client timeout 1 s). Windows:
  `acc` = `configs/rq2_frozen.json` (alpha 0.25, beta 0.25, epsilon 0.05, n_c 32, kappa 0.8, W0 8, W 1-128);
  `none` = fixed W = 128 (RQ2's unconstrained window; any W >= 32 behaves the same with 36 agents, design section 7);
  `fixed4` = fixed W = 4.
- **Background load:** 32 scripted agents, RQ2 CAP-steady think times (exponential, mean 0.1 s, start jitter 1 s), in
  a new **until-done** mode: they run until all 4 LLM agents have finished their tasks or until T_max **(Q2)**; then
  every scripted call and in-flight attempt is cancelled and abandoned server work is shut down, as in RQ2's horizon
  mode. The RQ2 code paths are reused; the until-done loop is new code in `stage3/rq3.py` (RQ2 stays frozen).
- **LLM agents:** 4 S3.2a `Agent`s, each solving its 5 tasks in order; they start after a warm-up of **20 s** of
  background load **(Q3)**, so ACC's window has settled. The LLM backend call blocks the loop (virtual time stands
  still); the agent then sleeps its modelled t_llm (scaled by `llm_time_scale`, 1/3 in the sensitivity run).
- **Dispatcher:** `GatewayDispatcher.call(op, args, agent, k, task_index)` -> `gw.call(request)`; a final status in
  `status.RETRYABLE` (retries exhausted, incl. client timeouts) is a **transient** outcome (S3.2a: transient notice, no
  EEM); otherwise the suite's (status, body, truth, canonical op) from the admitted attempt is returned.

## 3. Store

- **Shared (C2):** one `EEMStore` used by the 4 agents; timing rules of S3.2a section 6.5 (prompt built when a
  generation starts; a note visible after its reflection's t_llm sleep; outcome update and re-check when the success
  response arrives; owner = (agent, task)). Reset at the start of every unit.
- **Private (descriptive):** one store per agent.

## 4. Hidden executions and drift visibility

- **Hidden execution:** a call the suite executed (`call_key` in the suite's store) whose final outcome reached the
  agent as transient (retries exhausted). Counted per unit; if the agent re-issues the call (a new `k`), a duplicate
  write may follow and fail an "exactly one" sub-goal; that is part of the no-window effect and is reported.
- **Overload hides drift** (design section 7): drift errors observed and notes created are counted per unit.

## 5. Results per unit

- Per LLM task: S3.2a task summary plus virtual start/finish times, completion time, and `unfinished_at_t_max`.
- Per unit: success over the 20 LLM tasks; RQ2 metrics for all attempts (`metrics.compute_metrics_v2` with
  horizon = the run's end and RQ2's 20 s grace) and for LLM attempts only (overload rate, SLO violation, e2e p95,
  attempts per call); hidden executions; drift errors observed; notes created; window trace; final notes and note
  events (C2); the stamp (S3.2a stamp + window, condition, store mode, llm_time_scale, T_max, warm-up, SIM_VERSION).
- Seeds: S3.1 rq3 seeds (test 4000-4009 planned); the scripted load's RNG seed is derived from the same seed, so all
  cells of a seed see the same background stream (common random numbers, as in RQ2).

## 6. Analysis hooks (frozen with the analysis code before test runs)

R1 (C2 vs C0 under ACC, NI margin one task per seed then superiority), R2 (ACC vs no window within C2), interaction
estimate with the floor rule, secondary metrics (design section 8). Implemented in S3.4/S3.5 with the RQ1 analysis;
S3.2b only produces the unit results.

## 7. Tests (pytest, CPU, ScriptedBackend)

- Semantic server: admission order (429/503 before semantics), idempotent retry after a client timeout (one state
  change), 4xx latency and controller accounting (not overload, not a latency sample), silent 200 is a latency sample,
  scripted calls load-only.
- Until-done: the run ends when the LLM agents finish; T_max cap marks unfinished tasks; no orphaned tasks.
- Transient path: with a tiny server (forced overload) LLM calls exhaust retries and give the transient notice; hidden
  executions are counted.
- Shared vs private store; S3.2a timing rules hold with 4 agents.
- Determinism: same unit twice -> identical results; different PYTHONHASHSEED; all cells of a seed see the same
  background request stream.
- Manipulation sanity (scripted backend): no window -> LLM-call success below ACC's (as in the RQ2 manipulation
  check, design section 7).

## 8. Out of scope

The llama.cpp backend, caching and resume (S3.3); the RQ3 analysis code (S3.4/S3.5).

## 9. Choices for the user

- **Q1 Semantic 4xx latency:** (a) `error_latency` (0.02 s; validation and business errors are fast, like 429/503)
  [recommended]; (b) the normal service latency.
- **Q2 T_max:** (a) 3,600 s of virtual time (far above the expected ~5-10 min; unfinished tasks reported)
  [recommended]; (b) 1,800 s.
- **Q3 Warm-up before the LLM agents start:** (a) 20 s (RQ2's success grace; ACC's window settles) [recommended];
  (b) none (LLM agents start with the scripted agents).
- **Q4 Background calls:** (a) load only (no semantics, always 200 when admitted) [recommended: the background is the
  RQ2 load generator]; (b) background agents also execute real operations on their own entities.
