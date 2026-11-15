# Review request: Stage 3 experiment design r3 (EEM, LLM agents, schema drift; RQ1, RQ3, RQ4), 2026-11-15

You are a strict thesis examiner with experience in LLM-agent evaluation and systems experiments. Be concrete; no
praise.

## Context
Paper "Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control". EEM
(Execution Experience Memory) turns non-transient API errors into a parsed tuple k_t, an LLM-written fix
instruction, and a Zettelkasten-style note store (merge on (API_Name, Error_Type), links above a similarity threshold,
top-k_s seeds + 1-hop neighbours up to k) shared by all agents. ACC is an AIMD concurrency window; its experiment
(RQ2) is finished with a frozen configuration. Constraints: zero cost (free Colab T4, local Qwen2.5-7B-Instruct 4-bit
via llama-cpp-python), never fabricate results, dev/test separation, resumable runs. The design below is revision 3,
after one internal review (4 High findings: RQ3 seed count too low for an exact sign-flip test; EEM store too small
for links/k to matter; repeated-error rate is a mechanism check, not an outcome; the Reflexion-style baseline lacked
cross-task memory). Code: https://github.com/notanut/eem-acc-skripsi (RQ2 virtual-time simulator in src/eem_acc).

## Checklist
1. Internal validity of RQ1: does the design isolate EEM's contribution (vs plain persistence of reflections, vs
   docs look-ups, vs prompt length)? Are the primary endpoints (success rate; API calls per solved task) and their
   comparisons the right ones? Any way the setup favours EEM by construction?
2. Baselines: are C0 (ReAct), C1 (cross-task FIFO of reflections), C3 (no links) and the ablations fair and
   sufficient for an IEEE paper on agent memory? Anything an examiner would ask for that is missing (e.g. ExpeL-style
   insight extraction, full-history memory)?
3. RQ3: is "32 scripted agents + 4 LLM agents on one gateway, in virtual time with a modelled LLM time" a defensible
   test of the combined system? Is the non-inferiority formulation of the interaction sensible?
4. Statistics: unit of analysis (stream), exact sign-flip tests with 10 streams/seeds, Holm, TOST margins, paired
   bootstrap CIs; power with ~20 tasks per stream.
5. Reproducibility: replay-from-cache resume and determinism claims with llama.cpp greedy decoding.
6. Feasibility of the budget on a free T4, and what you would cut first.
7. For each decision S1-S12, would you choose the recommended option? Why or why not?

## Output
Findings ranked by severity with section reference and a concrete fix; then your choice for S1-S12; then a verdict.

## What changed since r2 (which you may have reviewed)
r3 applies a second opinion on r2 and an internal review of the first r3 draft. The user decided: notes get an
"outcome update" (a deterministic "Last working change" line from the diff between the failed and the successful
arguments); C1 = reflection RAG (same k and memory budget, no parsing/merge/links/outcome update); E1/E3 tested
non-inferiority then superiority, E2/E4 on jointly solved tasks, Holm over six hypotheses with gating, stream count from
a power simulation, t intervals; RQ3 primary = EEM effect under ACC on, interaction descriptive; LLM time constants
measured on the T4. Please check especially sections 6 (outcome update), 8 (statistics) and 13 (Q1-Q4).

## The design (docs/stage3_design.md, revision 3)

# Stage 3 design proposal: API environment with schema drift, EEM, LLM agent loop (RQ1, RQ3, RQ4)

_Draft r3, 2026-11-15, NOT approved. Nothing here is implemented._

_History:_
- _r2 applied an internal review of r1._
- _r3 applies the user's second opinion on r2 (an external LLM, 4 High, 5 Medium, Low items). The user decided four
  points (section 13): H1 = outcome update of notes; C1 = reflection RAG; statistics changes (non-inferiority then
  superiority, jointly solved tasks, power simulation, t intervals, RQ3 primary endpoint); t_llm measured on the T4.
  The clear-cut points are folded into the text._
- _r3 also applies an internal review of the first r3 draft: a valid multiplicity scheme, a precise outcome-update
  rule, power-rule and memory-budget details, and budget additions. Four points from it are decisions for the user
  (section 13, Q1-Q4)._
- _Still to approve: the whole r3, i.e. S1-S12 as amended and Q1-Q4 in section 13, and the changes to
  D1/D3/D4/D5/D6/D9 in section 2._

## 1. Research questions

| RQ | Question | Where answered |
|----|----------|----------------|
| RQ1 | Does EEM improve recovery from API-contract errors (schema drift) across a stream of tasks? | Single LLM agent, no overload |
| RQ2 | Does ACC protect the API under concurrency? | Done (HANDOFF section 8) |
| RQ3 | How do EEM and ACC combine when drift and overload occur together? | LLM agents on a loaded API, factorial |
| RQ4 | How sensitive are the results to hyperparameters? | ACC: RQ2 dev grid (243 configs, already run) + boundary check. EEM: the dev selection sweep (k, k_s, theta_link) reported in full |

## 2. Earlier decisions: kept, and changed (changes need your OK)

**Kept:**
- D2/D9: Qwen2.5-7B-Instruct 4-bit GGUF via llama-cpp-python on a free T4; all-MiniLM-L6-v2 embeddings.
- D3/D6: baselines and ablations, re-shaped in section 7.
- D7: dev/test separation and freezing.
- Section III r7: EEM trigger set; shared store; merge, link and retrieve; transient-failure step.
- RQ2's frozen ACC configuration, used unchanged in RQ3.

**Would change (flagged):**
- **D1/D9 "FastAPI + uvicorn mock server":** experiments run in-process in virtual time (S1); FastAPI at most as a
  demo wrapper.
- **D1 "StableToolBench subset later":** dropped (S2), because it is not usable at zero cost as published (section 4).
- **D3 "ReAct + Reflexion-style last-failure reflection":** C1 becomes *reflection RAG* (user's decision); the
  last-failure buffer becomes the descriptive arm F.
- **D4 "paired bootstrap CIs":** at n = 10-14 streams, 95% t intervals instead, because percentile bootstrap intervals
  undercover at that size. RQ2 already reports t intervals.
- **D6 "k in {1,3,5}" as test arms:** dropped; the dev selection sweep over k is reported instead (RQ4).
- **D5 "fault rates p in {0.1, 0.3}":** drift is expressed as the share of endpoints that drift per stream (S3).
- **Section III-B (patch r8, S12):** notes get an *outcome update* when the fix is found (section 6, decided by the
  user as H1(a)).

## 3. Architecture

```
            +-------------------- one virtual-time event loop (deterministic) --------------------+
 LLM agents (Algorithm 1) --+                                                                      |
   prompt -> LLMBackend ----|--> Gateway (ACC window or baseline, retries) --> MockServer (RQ2:    |
   EEM store (shared) <-----|                                                   429/503/latency)   |
 scripted background agents |                                                   --> API semantics  |
   (RQ3 only) --------------+                                                   (state, schema,    |
                                                                                drift, 4xx / 200)  |
            +--------------------------------------------------------------------------------------+
 LLMBackend = CachedBackend(LlamaCppBackend | ScriptedBackend); cache per result unit, on Drive
```

- **Virtual time (S1).** The RQ2 simulator runs agents, gateway and server on a virtual clock. Stage 3 adds an
  API-semantics layer behind the existing `MockServer`: rate-limit and overload checks first, then schema validation
  and execution, as a real gateway would. An LLM call blocks the event loop, so virtual time stands still while the GPU
  works. Afterwards the agent is charged a **modelled** LLM time for every LLM call (actions and reflections):
  `t_llm = a + b_in * n_in + b_out * n_out`, with a, b_in, b_out **measured on the T4 in S3.0** (user's decision;
  it overrides the second opinion's "constants from a cited serving benchmark").
  Token counts come from the cache, so t_llm is deterministic on replay. Consequences: RQ3 reuses RQ2's calibrated
  server and frozen ACC; a slow T4 does not distort the API load; runs are reproducible by replay (section 9).
- **API semantics layer:** in-memory state (customers, products, orders, payments, shipments, tickets), handlers, a
  schema per operation and version, a drift injector, and a programmatic checker per task. Pure Python, CPU-only,
  unit-testable. Ground truth tags every error as *drift* or *agent/logic* (e.g. a 404 from a wrong id is not an
  "endpoint moved" drift).
- **LLM backend:** `generate(messages, grammar, params) -> (text, n_in, n_out)`. Implementations:
  - `LlamaCppBackend`: Colab T4; one `Llama` object behind a lock and a re-entry guard, because an interrupted Colab
    cell can leave a generation running in a daemon thread.
  - `ScriptedBackend`: deterministic rule-based policy for CPU tests and sanity runs.
  - `CachedBackend`: wraps either one.
- **Embeddings:** computed on CPU and cached by text hash. Cosine similarities are rounded to 1e-6 and ties broken by
  note id, so retrieval is identical on replay.

## 4. Task suite (S2)

- **Custom synthetic API suite:** about 16 operations in 3 related domains (customers/orders/payments,
  inventory/shipping, support tickets). They are documented to the agent as short OpenAPI-style summaries. About 25
  task templates need 2-6 calls each with data dependencies (e.g. find customer by email -> list cart -> create order
  -> pay -> request shipment). Instances are sampled with fresh entities, and success is checked against the final
  server state (Section III-A).
- **Freeze before any EEM dev run:** the suite, templates, error formats, drift pool and system prompt are frozen and
  committed before the first EEM dev run. Only the pilot's difficulty calibration (section 8) may change them, and
  only before that freeze.
- **Docs tool:** `get_api_docs(operation)` returns the **current** schema. It is an API call like any other: one step,
  through the gateway, counted in the load. It never drifts and never triggers EEM. Its system-prompt wording is
  chosen on dev **to maximise C0's dev success**, so the strongest memory-less agent is the baseline.
- **External benchmarks:**
  - StableToolBench uses gpt-4-turbo as its evaluator for Solvable Pass/Win Rate and to simulate uncached APIs
    [STB]. A later version replaces the simulator with trained LLMs (MirrorAPI) [MIR]. The second opinion reports
    released weights (MIT licence, a Qwen2.5-7B-Instruct fine-tune); I could not check that from here. The pass rate
    still uses a GPT judge, and a second 7B model would double T4 time.
  - BFCL v3 multi-turn: state- and response-based checks on executable backends, without a judge model, per a
    secondary source [BFCL-ES]. The second opinion reports Apache-2.0 for the dataset. The primary repository must be
    checked before option S2(b).

## 5. Schema drift and errors (S3)

- **Drift types:**
  1. renamed parameter
  2. type change
  3. new required field
  4. enum value change
  5. operation moved (404/410, with or without a pointer to the new route)
  6. "silent" error: HTTP 200 with `{"status": "error", ...}`

  Types 1-3 are D5; types 4-6 are additions.
- **Per stream:** a drift set is applied before the first task and stays in force; the agent's docs remain the
  **old** schema. 8 of 16 operations drift, 2 of them with two drift types, so the store holds about 10-14 notes. Dev
  and test streams draw drift sets from disjoint pools of (operation, drift type) pairs.
- **Error style:** informative (FastAPI-style 422 naming the bad field and expected type) or terse
  ("400 Bad Request"). It is randomised per (stream, operation), half and half, and results are reported per style.
- **Exposure balance:** every drifted operation gets at least 3 exposures per stream, the first in the first third
  of the stream.
- **Staleness probe (descriptive):** on 5 extra streams, one already-drifted operation drifts again at the stream
  midpoint (C0 and C2 only). It exercises stale notes and the merge/outcome update.
- **Reported in Section IV:** the share of all calls that hit a drifted operation. The drift density is stated as a
  limitation; the environment is denser in drift than typical production APIs.
- The EEM store is reset at the start of every stream.

## 6. Agent loop (Algorithm 1) and EEM details

- **Action format (S6):** one JSON object per step, `{"thought": str (at most 60 tokens), "action": str, "args":
  object}`. A grammar enforces the envelope only; `args` is free, so schema mistakes stay possible. A generation that
  still fails to parse counts as a step with the observation "invalid action format". It dispatches no API call, does
  not trigger EEM, and its rate is reported per condition.
- **Operations and API_Name:** the action names an operation id (e.g. `create_order`); `args` holds its parameters;
  API_Name = operation id. "Operation moved" makes the documented id fail with 404/410; the new id is learnt from the
  error pointer or from `get_api_docs`.
- **Observation after an API error (all conditions):** the raw status and error body, truncated to 400 characters, so
  informative 422 text is never hidden. EEM *adds* its note text; it does not replace the error text.
- **Outcome update (new; user's decision H1(a); Section III patch r8):**
  - **Open note:** each note keeps the arguments of the failed call that created or last refreshed it (its *trigger
    arguments*). It is open until an outcome is recorded.
  - **Success:** an HTTP 200 response without an error payload.
  - **Which success closes which note:**
    - A success of operation X closes the open notes keyed on X. The time window is decision Q2 (same task, or same
      stream).
    - For a 404/410 on X ("moved"), only a success of the operation id named in that error's pointer, or in a
      `get_api_docs(X)` response, closes the note.
  - **What is recorded:** one deterministic line, "Last working change: ...", built by diffing *that note's own*
    trigger arguments against the successful call's arguments (no LLM call):
    - always: keys that disappeared or appeared, and keys whose JSON type changed;
    - values only for fields named in the note's k_t or error payload, so unrelated changes (another entity id) do
      not leak in;
    - for moved operations, the new operation id;
    - with several open notes on X (an operation with two drift types), each note is diffed against its own trigger
      arguments.
  - **Wording:** "Last working change", not "Verified fix". The call worked, but the diff is not proven to be the
    cause.
  - **On a new error on the same key** (merge), the old line is removed and the note is open again (relevant to the
    staleness probe).
  - **Embedding:** the line is not part of z = Emb(c_t || k_t), so retrieval is unchanged by the update; the line is
    shown with the note in the prompt.
- **Tools:** A_t = A (16 operations, `get_api_docs`, `finish`). The retrieval query q_t = Emb(s_t || desc(A_t)) then
  has a constant tool part; Section IV says so.
- **Prompt layout:** static prefix (system rules + tool docs) -> memory section -> task -> trajectory.
- **Decoding:** temperature 0. Streams differ in task order, sampled entities, drift set and error styles.
- **MaxSteps:** fixed a priori as the template's minimal number of calls + 6, not tuned. The pilot and test report the
  share of failures due to budget exhaustion versus a wrong final state. Recovering from one drifted operation costs
  about 2 steps, so the budget binds more for C0.
- **Reflection prompt:** asks for one imperative fix instruction of at most 40 words, given (s_t, a_t, k_t).
- **k_t parser:** deterministic for 4xx/422 bodies, 404/410 and 200-with-error payloads; unrecoverable fields are
  `unknown`. The parser and the error formats are written by the same author, so parsing is accurate by construction.
  This is stated as a limitation; A1 (raw text) partly covers it.

## 7. Conditions

**RQ1** (single LLM agent, sequential stream of tasks, generous fixed window, no overload). Every memory condition
uses C2's frozen k and the same **memory budget of 400 prompt tokens**. Notes are added in rank order and the first note
that does not fit is dropped whole, with all lower-ranked notes. C1 has no merge, so duplicate reflections can crowd
its top-k; this is part of what C2 - C1 measures and is reported.

| ID | Condition | Memory | Role |
|----|-----------|--------|------|
| C0 | ReAct | none | primary comparison |
| C1 | Reflection RAG | reflections after non-transient errors (the reflection prompt sees the raw error, no k_t), embedded (reflection text) and retrieved top-k by cosine with the same query; no k_t, merge, links or outcome update | primary comparison (C2 - C1 = EEM's structure **and** outcome update; see C2-OU) |
| C2 | **EEM** | k_t + reflection + outcome update; Zettelkasten store; retrieval (Section III-B) | method |
| C3 | EEM flat | C2 without links | secondary (TOST) |
| F  | Reflection FIFO | the last k reflections, carried across tasks | descriptive |
| AN | All notes | C2's store, every note in the prompt (no retrieval) | descriptive |
| R  | Oracle docs | none; the current schema is shown | descriptive upper bound |
| A1 | EEM raw | raw error text stored and retrieved | descriptive |
| A2 | EEM no reflection | k_t + outcome update, template sentence instead of the LLM reflection | descriptive |
| C2-OU | EEM without outcome update | C2 as in Section III r7 | descriptive (Q1); separates the outcome update from EEM's structure |
| SP | Staleness probe | C0 and C2 with a mid-stream re-drift (section 5) | descriptive |

C1 - C0 measures cross-task memory as such (secondary). The k = 1/5 test arms of r2 are dropped; the dev sweep
is the EEM part of RQ4.

**RQ3:** {C0, C2} x {ACC off = no window (same retries), ACC on = RQ2 frozen config}, plus C2 with the fixed W = 4
window (RQ2's global fixed baseline): 5 cells.
- **Setting:** CAP-steady load; 32 scripted agents (RQ2's M = 32 cell) plus 4 LLM agents, i.e. 36 agents on one
  gateway. LLM request ids are offset so they never collide with scripted ids or the burst source.
- **Run length:** each LLM agent solves 5 tasks. The scripted load runs until all LLM agents finish (a new
  "until-done" workload mode), capped at T_max. Tasks unfinished at T_max count as failures and are reported.
- **Store:** the EEM store is shared across the LLM agents and reset at the start of every run (seed); a
  private-store run is descriptive (C2, ACC on, 5 seeds).
- **Manipulation check (done 2026-11-15 on existing RQ2 dev data, CAP-steady, M = 32, 5 dev seeds):** with no window
  (fixed W >= 32 behaves identically), call success is 0.854 (real retry exhaustion: no call older than the grace
  period is left unfinished), attempts per call are 3.34, the overload rate (metrics key `throttle_rate`) is 0.747,
  SLO violation is 0.605 and e2e p95 is 5.4 s. ACC gives call success 1.000 and e2e p95 1.9 s. Calls do fail without a
  window, so ACC can affect LLM-task success. In RATE-steady a missing window costs nothing (success 1.000 either way),
  which is why CAP-steady is used. On dev, fixed W = 4 matches ACC (success 1.000 both; e2e p95 1.72 s vs 1.89 s), so
  the C2 + W = 4 cell is expected to be close to C2 + ACC; it is the first cut candidate.
- **Overload hides drift:** a call whose retries are exhausted never reaches schema validation. Drift errors
  observed, and notes created, are counted per cell.
- **Sensitivity (descriptive):** the 2x2 is re-run on 5 seeds with LLM times 3x faster (a/3, b_in/3, b_out/3).

## 8. Metrics and statistics (frozen before any test stream)

**Per task:**
- success; steps; API calls (gateway attempts, including `get_api_docs`); LLM calls; prompt and output tokens;
  modelled LLM time; virtual completion time (RQ3)
- non-transient errors split into drift vs agent/logic; first-call outcome at each exposure to a drifted operation
- format errors; budget exhaustion

**Primary RQ1 endpoints** (unit = stream; paired across conditions: same tasks, order, drift and error styles):

| ID | Comparison | Measure | Test |
|----|------------|---------|------|
| E1 | C2 vs C0 | success rate | non-inferiority, margin 0.05; then superiority |
| E2 | C2 vs C0 | mean API calls on tasks **solved by both** in that stream | superiority (fewer) |
| E3 | C2 vs C1 | success rate | non-inferiority, margin 0.05; then superiority |
| E4 | C2 vs C1 | mean API calls on tasks solved by both | superiority (fewer) |

- **Tests:** exact paired sign-flip tests on the per-stream differences; the sign-flip test decides, the interval is
  reported only. All directional hypotheses are one-sided at family-wise alpha = 0.025:
  - NI1, NI3: non-inferiority of E1, E3 (test on difference + 0.05);
  - S1, S3: superiority of E1, E3;
  - E2, E4: fewer API calls on jointly solved tasks.
- **Multiplicity:** Holm across the six hypotheses {NI1, S1, E2, NI3, S3, E4}. Superiority S1 (S3) is claimed only
  if NI1 (NI3) is also rejected; this only removes rejections, so the family-wise error stays controlled.
- **Intervals:** 95% t intervals on the per-stream differences (D4 change).
- **Granularity and edge cases:**
  - With 20 tasks per stream the margin 0.05 equals one task, so a stream with difference -0.05 carries no weight in
    the NI test; ties reduce the effective n. Both are reported.
  - A stream with no jointly solved task has no E2/E4 difference and is left out of that test (reported).
  - "Solved by both" restricts E2/E4 to tasks the weaker condition can solve; this is stated.
- **Number of test streams (10 or 14):** fixed before the test phase by a power simulation.
  - Inputs: the SD of the per-stream differences C2 - C0 and C2 - C1 on the dev streams, i.e. the 4 selection streams
    with C0 and C1 also run (section 9).
  - Method: differences re-centred at 0 for NI (true difference 0) and at the observed dev mean for E2/E4; power at
    the Holm-adjusted level (0.025/6 for the first rejection).
  - Target: power 0.8 for NI1 and NI3. If 14 streams do not reach it, 14 are used and the achieved power is reported.
  - Dev values come from the selected configuration (a winner), so they are optimistic; stated.
**Secondary (outside Holm):**
- C1 vs C0 (cross-task memory); C3 vs C2 (TOST, margin 0.05)
- LLM calls and prompt tokens per solved task, to show EEM's own cost
- the ratio-of-sums version of E2/E4
- task-level mixed-effects logistic model: condition, position in stream, condition x position; random effects for
  stream and template
- per drift type, per error style; learning curves

**Mechanism checks (not endpoints):**
- **Repeat-exposure drift error rate:** among exposures to a drifted operation after its first exposure in the stream,
  the share whose first call fails with that operation's drift error. By construction this is about 1 for C0, unless
  it looks up the docs.
- **Note correctness:** whether the stored note names the new field, type, enum value or operation id. Checked
  automatically against ground truth, before and after the outcome update.
- **Retrieval recall:** when a drifted operation with an existing note is called, whether that note was in M_rel.
  Without this, a null result cannot be told apart from "learned but not retrieved".

**RQ3** (10 seeds, paired over seeds; Holm across {NI-R1, S-R1, R2}, one-sided alpha = 0.025, superiority gated on
non-inferiority as above):
- **R1:** success rate, C2 vs C0 under ACC on. Non-inferiority (margin 0.05), then superiority.
- **R2 (Q3):** success rate, ACC on vs no window, within C2, one-sided (ACC higher).
- No power check is possible before the runs; an inconclusive R1 or R2 is reported as inconclusive.
- **Secondary (Q4):** virtual task completion time (where ACC acts; it may be slower with ACC on, because LLM calls queue
  FIFO behind the scripted agents at W of about 4-5); fixed W = 4 vs ACC; RQ2 metrics per cell (overload rate, SLO
  violation, goodput, J with lambda = gamma = 0.5); drift errors observed and notes created.
- **Interaction:** (C2 - C0 | ACC on) - (C2 - C0 | off), reported as an estimate with a t interval only. **Floor
  rule:** if success under no window is below 0.2 for both C0 and C2, the interaction is reported as uninformative.

**Descriptive arms** (F, AN, R, A1, A2, C2-OU, SP, RQ3 sensitivity, private store) run on 5 streams or seeds,
without tests.

**Dev rules (written before the pilot):**
- *Pilot:* C0 and C2 (default parameters) on 5 dev streams; docs-tool wording variants are tried with C0 on 2 dev
  streams each before that.
- *Pilot acceptance:* C0 dev success between 0.2 and 0.8. Otherwise task difficulty is recalibrated, on dev only and
  before the freeze.
- *EEM selection:* on 4 dev streams, pick (k, k_s, theta_link) by the highest mean dev success rate. Ties within 0.05
  go to fewer API calls on jointly solved tasks, then to fewer prompt tokens. The whole sweep is reported (RQ4). C0 and
  C1 (with the selected k) are then run on the same 4 dev streams for the power simulation.
- All analysis code is frozen before test streams run, as in RQ2.

## 9. Cost, resume and reproducibility

- **Budget model:** about 9 LLM calls per task (failing tasks run to MaxSteps; reflections included). c_step is the
  wall time per LLM call on the T4 and is unknown until S3.0. The figures are an **illustration, not measured**.

| Block (20 tasks per stream) | Tasks | LLM calls | c_step 3 s | c_step 6 s |
|-------|------|-----------|-----------|-----------|
| Docs-wording variants (3 x C0 x 2 dev streams) | 120 | 1.1k | 0.9 h | 1.8 h |
| Pilot (C0, C2 x 5 dev streams) | 200 | 1.8k | 1.5 h | 3 h |
| EEM dev selection (6 configs x 4 streams) + C0, C1 on those streams | 640 | 5.8k | 4.8 h | 9.6 h |
| RQ1 primary (C0-C3 x 10 streams) | 800 | 7.2k | 6 h | 12 h |
| RQ1 descriptive (F, AN, R, A1, A2, C2-OU x 5 streams; SP 2 x 5) | 800 | 7.2k | 6 h | 12 h |
| RQ3 (5 cells x 10 seeds x 4 agents x 5 tasks) | 1,000 | 9k | 7.5 h | 15 h |
| RQ3 private store (5 seeds) + sensitivity (4 cells x 5 seeds) | 500 | 4.5k | 3.8 h | 7.5 h |
| **Total (10 streams)** | 4,060 | 36.5k | **~30 h** | **~61 h** |

  - 14 streams add 320 tasks (about 2.4 h at 3 s, 4.8 h at 6 s).
  - **Cut order if the budget is short (pre-registered, S8):** (0) the C2 + W = 4 RQ3 cell to 5 seeds; (1) R to 3 streams; (2) A1/A2 to 3 streams; (3) F/AN
    to 3 streams; (4) SP to 3 streams; (5) RQ3 sensitivity to 3 seeds; (6) descriptive arms to 15 tasks per stream.
    C0-C3 primary and the RQ3 2x2 are never cut.
  - KV-prefix reuse within one agent's trajectory helps; the response cache is per unit and gives no cross-condition
    saving.
- **Resume by replay:**
  - Every LLM output is appended (atomic, checksummed JSONL on Drive) to the cache **of its result unit**, together
    with n_in and n_out. The key is model file sha256, llama.cpp build, rendered chat template, grammar, sampling
    parameters, max_tokens, stop sequences, n_batch and the full prompt.
  - An interrupted unit restarts from the beginning: cached outputs replay instantly and the deterministic simulation
    reaches the same state, then continues.
  - **Strict replay:** a cache miss before the end of the unit's cache is an error, not a fresh generation (the replay
    has diverged).
  - Fresh greedy generations are not guaranteed bit-exact (KV reuse, batch size), so the claim is **replay
    reproducibility**. The caches are archived on Drive as primary data. Units never share caches, so a result does
    not depend on run order.
  - The "interrupted = uninterrupted" test covers the simulator and replay, not fresh generations.
- **Integrity:** spec hash, model sha256, llama-cpp-python version, prompt template version and sim version are stamped
  in every result; a resume with a different model or template is refused.

## 10. Milestones (each: code + tests -> review -> commit -> report)

- **S3.0 Feasibility on Colab T4** (you run a notebook, ~30-45 min). It measures:
  - download of the GGUF (~4.7 GB)
  - a, b_in, b_out (prefill and decode speed) with and without the grammar, and VRAM
  - JSON-envelope validity on 20 sample prompts
  - c_step on a toy task
  - optionally, the same for a 3B model

  Scale (S8) and the t_llm constants are fixed from these numbers.
- **S3.1** API semantics layer, drift injector, task sampler, checker (CPU; tests); freeze the suite.
- **S3.2** Agent loop with `ScriptedBackend`; EEM parser, store and outcome update; C0-C3 and the descriptive arms
  (CPU end-to-end tests).
- **S3.3** `LlamaCppBackend`, cache/replay, notebook `03_stage3.ipynb` (live progress, resume).
- **S3.4** Pilot on dev streams (difficulty, docs-tool wording), EEM dev selection, power simulation, freeze.
- **S3.5** RQ1 test; **S3.6** RQ3; **S3.7** descriptive arms; then Chapter IV.

## 11. Pre-registered expectations (written before any run)

**RQ1:**
- C2 is non-inferior to C0 and C1 in success.
- C2 needs fewer API calls than C0 and C1 on jointly solved tasks.
- The C2 - C0 advantage grows with position in the stream.
- With the outcome update, the advantage is at least as large for terse as for informative errors (descriptive; no
  test).
- No strong prior for C3 vs C2, or for C2 vs C1 in success.

**RQ3:**
- ACC on raises LLM-task success over no window (fewer exhausted retries); completion time may go either way.
- The EEM effect holds under ACC on.

**Limitations stated in advance:**
- The LLM agents generate under 1% of the RQ3 load, so "EEM reduces load" is not testable.
- Drift is denser than in typical production APIs.
- The k_t parser and the error formats are written by the same author.

## 12. Text impact

**Section IV (setup):**
- virtual time and t_llm; A_t = A; JSON envelope; `get_api_docs`; mixed population
- store reset per stream; raw error shown in all conditions; the drift density

**Section III, patch r8 (S12; written when the Stage 3 code is final):**
- III-A: a format-invalid generation also counts as a step (no API call).
- III-B: the observation keeps the raw error and adds the note text.
- III-B: a new "outcome update" step (section 6), and the matching line in Algorithm 1 after a success.

## 13. Decisions

**Decided by the user on 2026-11-15** (after the second opinion; part of r3, to be confirmed with the whole r3):
- H1 = outcome update (a)
- C1 = reflection RAG
- statistics changes (1)-(5)
- t_llm measured on the T4

**Decisions from the internal review of r3 (need your answer):**

| ID | Decision | Options | Recommendation |
|----|----------|---------|----------------|
| Q1 | Isolate the outcome update (C1 has none, so C2 - C1 mixes structure and update) | (a) descriptive arm C2-OU (EEM as in Section III r7), 5 streams, ~100 tasks; (b) (a) + "C1 + outcome update" arm; (c) only rename the contrast | (a) |
| Q2 | Time window of the outcome update | (a) same task only (attribution most reliable); (b) any later success of X in the same stream (also after a docs lookup in a later task; in RQ3 by any agent) | (a) |
| Q3 | RQ3 R2 (ACC on vs no window, within C2) as a primary test | (a) primary, inside Holm with R1; (b) secondary | (a) |
| Q4 | Virtual task completion time in RQ3 (the second opinion asked for an endpoint) | (a) secondary, reported with a t interval; (b) primary, inside Holm | (a) |

**Still to approve:**

| ID | Decision | Options | Recommendation (r3) |
|----|----------|---------|----------------|
| S1 | Execution environment (changes D1/D9 usage) | (a) in-process virtual time; FastAPI only as an optional demo wrapper; (b) real FastAPI + uvicorn, wall clock; (c) (a) + one FastAPI end-to-end demo run | (a) |
| S2 | Task suite (drops D1's StableToolBench) | (a) custom suite only, frozen before dev; (b) (a) + BFCL v3 subset with our drift, after checking the primary repo; (c) keep StableToolBench (paid judge) | (a); (b) only after RQ1 |
| S3 | Drift (replaces D5's p levels) | (a) 6 types, 8 of 16 operations (2 with two types), random error style, staleness probe descriptive; (b) (a) + a second drift wave in all streams; (c) two drift levels as a factor (~2x cost) | (a) |
| S4 | RQ3 setting | (a) section 7: 32 scripted + 4 LLM agents, CAP-steady, until-done with T_max, t_llm from the T4 + 3x-faster sensitivity; (b) all-LLM M = 8 on a scaled-down server (new calibration) | (a) |
| S5 | C1 | decided: reflection RAG; F and AN descriptive; within-task arm dropped | - |
| S6 | Action format | (a) grammar-enforced JSON envelope, thought capped, args free; (b) no grammar; (c) native tool-calling template | (a) |
| S7 | "ACC off" in RQ3 | (a) no window in the 2x2 + fixed W = 4 with C2 (5 cells); (b) fixed W = 4 as "off"; (c) no window only | (a); manipulation check done (section 7) |
| S8 | Scale | (a) section 9 with the pre-registered cut order; (b) smaller from the start; (c) decide after S3.0 | (c), target (a) |
| S9 | Model | (a) keep Qwen2.5-7B-Instruct Q4 (D9); (b) 3B; (c) compare in S3.0 | (a); (c) only if c_step > 6 s |
| S10 | MaxSteps | (a) a priori minimal calls + 6, with the budget-exhaustion share reported; (b) tuned on dev | (a) |
| S11 | Primary RQ1 endpoints | decided: E1-E4 as in section 8 | - |
| S12 | Section III edits | (a) patch r8 when the Stage 3 code is final (incl. the outcome update); (b) Section IV only | (a) |

## References (checked 2026-11-15)
- [STB] Z. Guo et al., "StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language
  Models," arXiv:2403.07714. Checked (v4 HTML): uncached calls are simulated by gpt-4-turbo; gpt-4-turbo is the
  evaluator for Solvable Pass/Win Rate; the paper notes the added cost. https://arxiv.org/abs/2403.07714
- [MIR] Z. Guo et al., "StableToolBench-MirrorAPI: Modeling Tool Environments as Mirrors of 7,000+ Real-World APIs,"
  arXiv:2503.20527, 2025. Checked (abstract): it trains specialised LLMs to simulate API responses and integrates them
  into StableToolBench. The model card (weights, licence) could not be opened from here.
  https://arxiv.org/abs/2503.20527
- [BFCL-ES] EvalScope documentation, "BFCL v3": multi-turn uses state-based and response-based checks; no judge model
  is mentioned. https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html (secondary source).
