# S3.2b specification: RQ3 wiring (gateway, server, background load, shared store) (CPU only)

_r4, 2026-11-22 (r3 + the re-review: D120-D121 and clear-cut fixes; r3 = the code review: D117-D119, section 10). Implements design r7 section 7 (RQ3) on top of S3.1 (suite) and S3.2a (agent loop, EEM), reusing
the RQ2 simulator unchanged (`Gateway`, `MockServer`, `ACCController`, frozen ACC config). r1 was reviewed by a
fresh-context subagent (2 High, 5 Medium, 9 Low); the user decided D107-D111 (section 9); the clear-cut findings are
applied below._

## 1. Run = one RQ3 cell on one seed

```python
res = run_rq3_unit(spec, condition, window, backend, embedder, tokenizer, cfg) -> dict
#   spec: an S3.1 rq3 stream (20 tasks; agents [[0,4,..],[1,5,..],[2,6,..],[3,7,..]])
#   condition: "C0" | "C2" | "C2-private"; window: "none" | "acc" | "fixed4"
```

Cells (design section 7): {C0, C2} x {no window, ACC on} plus C2 + fixed W = 4; descriptive: C2 + ACC with private
stores and the 2x2 with LLM times / 3 (both on the first 5 test seeds, 4000-4004). The same warm-up applies to every
cell. Seeds: rq3 test 4000-4009 (planned), rq3 **dev 5000-5099** (D111, an S3.1 change before the freeze: one unit
each of C2+ACC and C0+none in the pilot; only wiring bugs and T_max may change after it).

## 2. Simulator composition (virtual time, one event loop)

- **Configs from one source:** `ServerConfig`, retry policy and the ACC config are taken from
  `scenarios.build("CAP-steady", 32, method, seed)` with `method` = `acc_method(**frozen["acc"])` (ACC),
  `fixed_method(128)` (none; RQ2's unconstrained window, any W >= 32 behaves the same with 36 agents) or
  `fixed_method(4)`. A test asserts equality with those builds.
- **Seeds:** as `sim.py`: `SeedSequence(seed).spawn(3)` -> server, gateway, scripted agents; the scripted agents use
  `run_agents`' `spawn(n_agents + 1)` children. Nothing depends on the cell or condition. Common random numbers hold at
  the level of **draws** (per-agent think times; noise and jitter keyed by (agent, k, admission)); arrival times differ
  between cells because the scripted agents run in a closed loop (Section IV says so).
- **Server:** `SemanticServer(MockServer)` overrides `handle` and keeps RQ2's order: refill -> 429 -> 503 -> token ->
  then:
  - **scripted request** (plain `(agent, k)` tuple): inflight -> noise -> service latency -> 200, exactly as RQ2
    (load only, D110). A test asserts that a scripted-only run through `SemanticServer` gives attempts and metrics
    bit-identical to `MockServer` (`sim.run_scenario`, CAP-steady, M = 32, ACC and W = 128).
  - **LLM request:** executed by the S3.1 suite at admission with `call_key = (agent, k)` (idempotent: a retry after a
    client timeout gets the stored response; the state changes once, before any delay). A semantic **200** (including
    a silent error payload and an idempotent replay of a stored 200) then counts as inflight, draws noise and takes the
    service latency; it is a latency sample for ACC. A semantic **4xx** uses its rate token but is not inflight, draws
    no noise, returns after `error_latency` = 0.02 s (D107), and is neither an overload signal nor a latency sample
    (Section III-C); it has its own key in `counts`.
  - Every `handle` that completes is logged as (request, t_arrival, status, latency); an attempt the gateway abandoned
    at its client timeout (1 s) completes later with latency > T_out and counts as a client timeout. Attempts still in
    flight at the end are left out, as in the gateway. A test checks the log against `gw.attempts`.
- **Request ids:** an LLM request is a tuple subclass that stays exactly the 2-tuple `(agent, k)` (no `__eq__`/
  `__hash__` override; attributes op, args, task_index in `__dict__`), so the RQ2 gateway and server unpack it
  unchanged; the suite gets the plain tuple `(agent, k)`. LLM agents use ids 1000-1003. A test checks that its noise
  and jitter equal those of the plain tuple.
- **Gateway:** `TaggedGateway` (the RQ2 `Gateway` plus a per-attempt request log; no change to timing or draws), RQ2 retry policy (r_max 6, backoff 0.1-4 s, client timeout 1 s), windows as above.
- **Background load:** 32 scripted agents, CAP-steady think times (exponential, mean 0.1 s, start jitter 1 s), in a new
  **until-done** loop (`stage3/rq3.py`; RQ2 code untouched): they run until all 4 LLM agents have finished or until
  **T_max = 3,600 s** (D108); then every scripted call and in-flight attempt is cancelled and `gw.shutdown()` cancels
  abandoned server work (state changes made at admission stay).
- **LLM agents:** 4 S3.2a `Agent`s with ids 1000-1003, each solving its 5 tasks in order, starting after a **20 s
  warm-up** of background load (D109; RQ2 window traces show ACC's window settled by about 6 s, to be confirmed on the
  RQ2 dev seeds). `backend.generate` stays synchronous (no thread or executor), so the virtual clock stands still
  during a generation (test: `loop.time()` equal before and after). The agent then sleeps its t_llm scaled by
  `LoopConfig.llm_time_scale` (1.0; 1/3 in the sensitivity run; applied to action, fallback and reflection calls;
  stamped; RQ1 unchanged).
- **Dispatcher:** `GatewayDispatcher.call(op, args, agent, k, task_index)` -> `gw.call(request)`; a final status in
  `status.RETRYABLE` (exhausted, incl. client timeouts) is **transient** (S3.2a: transient notice, no EEM); otherwise
  the suite response of the admitted attempt (status, body, truth, suite op) is returned.
- **Partial logs:** each agent appends its step records to an external list as they are made, so a task cancelled at
  T_max keeps its steps; tasks never started get a summary with outcome failure (the denominator stays 20); completion
  times of tasks not finished by the end are censored at the end, from the task's start or, if never started, from
  the warm-up end, with `completion_censored` (D118). Never-started tasks carry `never_started` and are excluded from
  completion-time statistics, with their count reported per cell; they still count as failures for task success
  (D120). Cut-off tasks are summarised from their partial steps (drift, agent and format errors, API calls; the call
  number is recorded before the call); a generation cut short by T_max counts its full LLM time and its step counts
  as a step (D121, a caveat for capped units' secondary counts only).

## 3. Store

- **Shared (C2):** one `EEMStore` for the 4 agents, with S3.2a's timing rules (prompt built when a generation starts;
  a note visible after its reflection's t_llm sleep; outcome update and re-check when the success response arrives;
  owner = (agent, task)). The fallback generation after a malformed reply reuses the prompt built for the first
  generation, so a note that became visible in between is not shown to it (D119). A merge by another agent changes the owner, so the original agent's same-task outcome update
  no longer fires; reported as a caveat of the shared store.
- **Private (descriptive):** one store per agent; a test checks no note is visible across agents.

## 4. Hidden executions and drift visibility

- **Hidden execution:** a call the suite executed (`suite.executed(call_key)`, a public method) whose final outcome
  reached the agent as transient. Counted per unit, split into state-changing (stored 200 on a write) and others. In
  CAP-steady they should be near zero (with the hard cap at 8, a 1 s timeout needs noise z > 2.77); the test uses a
  slow server that forces them.
- **Overload hides drift:** drift errors observed and notes created are counted per unit.

## 5. Results per unit

- Per LLM task: the S3.2a summary plus virtual start/finish, completion time (censored), `unfinished_at_t_max`.
- **LLM call success** = final status not in `RETRYABLE` (a semantic 4xx is a delivered response, not an
  infrastructure failure); LLM-only overload rate (`status.OVERLOAD`; a client timeout counts), SLO violation, latency
  p95 over 200s, mean latency excess over all attempts, attempts per call and censored e2e p95, from the gateway's
  attempt records tagged with their request (`TaggedGateway`). Semantic 4xx attempts are in these denominators and in
  ACC's (stated).
- **J for RQ3** (Section III-A): LLM task success rate - 0.5 x mean latency excess - 0.5 x overload rate over the LLM
  attempts (lambda = gamma = 0.5, design section 8); the RQ2 objective J over the background attempts of the primary
  window beside it (D117).
- Infrastructure metrics (`metrics.compute_metrics_v2`, RQ2's 20 s grace) over the **background (scripted) traffic
  only** (D117), over **[warm-up, end]** through `AnalysisConfig.windows` (primary; plus call success with the grace,
  mean latency excess over all attempts, overload rate and the RQ2 objective J with lambda = gamma = 0.5) and over the
  whole run (secondary); goodput; window trace. The queue trace, window (trace, mean, final) and server max in-flight are shared by
  all traffic. The window's
  e2e p95 censors calls requested near the end with no grace (slightly low; stated).
- Hidden executions, drift errors observed, notes created, final notes and note events (C2), the stamp (S3.2a stamp +
  window, condition, store mode, llm_time_scale, T_max, warm-up, SIM_VERSION, **RQ3 wiring version**).

## 6. Analysis hooks

R1 (C2 vs C0 under ACC, NI margin one task per seed, then superiority), R2 (ACC vs no window within C2), interaction
with the floor rule, secondary metrics (design section 8): implemented with the RQ1 analysis in S3.4/S3.5.

## 7. Tests (pytest, CPU, ScriptedBackend)

- Configs equal the `scenarios.build` single source; `none`/`fixed4` frozen at W 128 / 4.
- Scripted-only bit-identity of `SemanticServer` vs `MockServer`.
- Semantic server: order (429/503 before semantics); idempotent retry after a forced client timeout (one state change;
  a retry while the first admission is in flight gets the stored response; both count inflight); 4xx bookkeeping;
  silent 200 is a latency sample; server log vs `gw.attempts`.
- Request subclass: same noise and jitter as the plain tuple; the dispatcher's status equals the suite's.
- Until-done and T_max: the run ends when the LLM agents finish; at T_max unfinished tasks are marked with partial
  logs; no pending tasks after the run; `shutdown` called; no LLM call before the warm-up ends; `get_api_docs` goes
  through the gateway.
- Transient path with a slow server: retries exhausted -> transient notice, hidden executions counted; the scripted
  backend re-issues the call unchanged after a transient notice.
- Shared vs private store; S3.2a timing rules with 4 agents.
- Determinism: same unit twice identical; different PYTHONHASHSEED; the scripted agents' draws identical across cells
  of a seed.
- Manipulation sanity (scripted backend): with no window, LLM call success is not above ACC's.

## 8. Out of scope

The llama.cpp backend, caching and resume (S3.3); the RQ3 analysis code (S3.4/S3.5).

## 9. Decisions (user, 2026-11-22)

| ID | Decision |
|---|---|
| D107 | Semantic 4xx returned after error_latency (0.02 s); its effect on completion time by error style is small and stated |
| D108 | T_max = 3,600 s of virtual time (expected 6-15 virtual minutes per unit; simulation cost is negligible) |
| D109 | LLM agents start after a 20 s warm-up (new relative to design r7; added to the design text) |
| D110 | Background calls are load only (no semantics) |
| D111 | RQ3 dev seeds 5000-5099; one unit each of C2+ACC and C0+none in the pilot; only wiring bugs and T_max may change after it |
| D117 | RQ3 infrastructure metrics over the background calls only; LLM calls are reported by the LLM-only metrics |
| D118 | Tasks not finished by the end: completion time censored at the end (never started: from the warm-up end) with a flag |
| D119 | The fallback generation reuses the prompt built for the first generation (it retries the same decision); with 4 agents a note may have become visible in between; documented |
| D120 | Never-started tasks are flagged and excluded from completion-time statistics (count reported per cell); they count as task failures |
| D121 | Partial summaries of cut-off tasks count a cut generation's full LLM time and its step; stated as a caveat |

## 10. Code review (2026-11-22): clear-cut changes

No High findings. Applied: agent errors end the run at once (`FIRST_EXCEPTION`), with `shutdown` always called
before re-raising (M4); the stamp carries `action_max_tokens`, `grammar_always`, the method, ACC, server and retry
configs and the SHA-256 of `rq2_frozen.json` (M5); LLM metrics from the tagged gateway attempts, so attempts
abandoned at shutdown are counted as the gateway counts them (M3, L6); the primary window gets call success, excess
and J (M2); `LLMRequest` copies and pickles (L4); embedding replay (`embedding_preload`, L5); notes created per unit
(L10); unused imports removed and the shared store built only when used (L12); RQ3 wiring version rq3-v2. Tests
added for every gap in section 7 that the review listed (M6, L11).
