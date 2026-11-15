# Stage 3 design proposal: API environment with schema drift, EEM, LLM agent loop (RQ1, RQ3, RQ4)

_**APPROVED 2026-11-15 (D39-D41): the user accepted all recommendations S1-S12 and the three changes to D1/D9/D5.** Draft r2, 2026-11-15. Nothing here is implemented. r2 applies an independent review of r1
(4 High, 9 Medium, 7 Low findings): clear-cut fixes are in the text; real choices are in section 13 (S1-S12) with
options and a recommendation. Section 2 lists what earlier decisions fix and which of them this proposal would change._

## 1. Research questions

| RQ | Question | Where answered |
|----|----------|----------------|
| RQ1 | Does EEM improve recovery from API-contract errors (schema drift) across a stream of tasks? | Single LLM agent, no overload |
| RQ2 | Does ACC protect the API under concurrency? | Done (HANDOFF section 8) |
| RQ3 | How do EEM and ACC combine when drift and overload occur together? | LLM agents on a loaded API, factorial |
| RQ4 | How sensitive are the results to hyperparameters? | ACC: RQ2 dev grid (243 configs, already run) + boundary check. EEM: k and theta_link on dev streams |

## 2. Earlier decisions: kept, and changed (changes need your OK)
**Kept:** D2/D9 (Qwen2.5-7B-Instruct 4-bit GGUF via llama-cpp-python on a free T4; all-MiniLM-L6-v2), D3/D6 baselines
and ablations (C1's exact form is S5), D4 metrics including paired bootstrap 95% CIs, D7 dev/test separation and
freezing; Section III r7 (EEM trigger set, shared store, merge/link/retrieve, transient-failure step); RQ2's frozen ACC
configuration used unchanged in RQ3.

**Would change (flagged):**
- **D1/D9 "FastAPI + uvicorn mock server"**: experiments run in-process in virtual time (S1); FastAPI at most as a
  demo wrapper.
- **D1 "StableToolBench subset later"**: dropped (S2), because it is not usable at zero cost as published (section 4).
- **D5 "fault rates p in {0.1, 0.3}"**: drift is expressed as the share of endpoints that drift per stream (S3).

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
  `t_llm = a + b_in * n_in + b_out * n_out`. This is deterministic given the cached output, and longer EEM prompts pay
  for their prefill. The basis for a, b_in and b_out is decision S4. Consequences: RQ3 reuses RQ2's calibrated server
  and frozen ACC; a slow T4 does not distort the API load; runs are reproducible by replay (section 9).
- **API semantics layer:** in-memory state (customers, products, orders, payments, shipments, tickets), endpoint
  handlers, a schema per endpoint and version, a drift injector, and a programmatic checker per task. Pure Python,
  CPU-only, unit-testable. Ground truth tags every error as *drift* or *agent/logic* (e.g. a 404 from a wrong id is
  not an "endpoint moved" drift).
- **LLM backend:** `generate(messages, grammar, params) -> (text, n_in, n_out)`. Implementations:
  - `LlamaCppBackend`: Colab T4; one `Llama` object guarded by a lock and a re-entry guard, because an interrupted
    Colab cell can leave a generation running in a daemon thread.
  - `ScriptedBackend`: deterministic rule-based policy for CPU tests and sanity runs.
  - `CachedBackend`: wraps either one.

## 4. Task suite (S2)

- **Custom synthetic API suite (recommended):** about 16 endpoints in 3 related domains (customers/orders/payments,
  inventory/shipping, support tickets), documented to the agent as short OpenAPI-style summaries. About 25 task
  templates need 2-6 calls each with data dependencies (e.g. find customer by email -> list cart -> create order ->
  pay -> request shipment). Instances are sampled with fresh entities. Success is checked against the final server
  state (Section III-A).
- **`get_api_docs(endpoint)`** returns the **current** schema. It is an API call like any other (one step, through the
  gateway, counted in the load), never drifts, and never triggers EEM. The system-prompt wording about it is fixed on
  dev before the pilot; it alone can drive C0's error rate down, so it is part of the frozen setup.
- **External benchmarks:**
  - StableToolBench uses gpt-4-turbo as its evaluator for Solvable Pass/Win Rate and to simulate uncached APIs
    [STB]. A later version replaces the simulator with trained LLMs (MirrorAPI) [MIR]; from its abstract I could not confirm
    released weights or a judge-free evaluation. As published [STB], it is not usable at zero cost without breaking
    comparability.
  - BFCL v3 multi-turn evaluates by state and response checks on executable backends, without a judge model, per a
    secondary source [BFCL-ES]. Injecting drift would be our own modification. Its primary repository and licence are
    not yet checked.

## 5. Schema drift and errors (S3)

- **Drift types:** (1) renamed parameter; (2) type change; (3) new required field; (4) enum value change; (5)
  endpoint moved (404/410, with or without a pointer); (6) "silent" error, i.e. HTTP 200 with
  `{"status": "error", ...}`. (1)-(3) are D5; (4)-(6) are proposed additions.
- **Per stream:** a drift set is applied before the first task and stays in force; the agent's docs remain the
  **old** schema. Recommended: 8 of 16 endpoints, 2 of them with two drift types, so the store holds about 10-14
  notes and k, links and C2-vs-C3 have room to matter (review H2). Dev and test streams draw drift sets from disjoint
  pools of (endpoint, drift type) pairs.
- **Error style:** informative (FastAPI-style 422 naming the bad field and expected type) or terse ("400 Bad
  Request"), **randomised per (stream, endpoint)**, half and half; results are reported per style.
- **Exposure balance:** the task sampler gives every drifted endpoint at least 3 exposures per stream, the first in
  the first third of the stream.
- **The EEM store is reset at the start of every stream**, so streams are independent.

## 6. Agent loop (Algorithm 1) details

- **Action format (S6):** one JSON object per step, `{"thought": str, "action": str, "args": object}`. A grammar
  enforces the envelope only; `args` is free, so schema mistakes stay possible. A generation that still fails to parse
  counts as a step with the observation "invalid action format". It dispatches no API call, does not trigger EEM, and
  is reported separately.
- **Observation after an API error (all conditions):** the raw status and error body (truncated to 400 characters),
  so informative 422 text is never hidden from any condition. EEM *adds* its reflection (Algorithm 1 line 13); it
  does not replace the error text.
- **Tools:** A_t = A (all 16 endpoints, `get_api_docs`, `finish`). The retrieval query q_t = Emb(s_t || desc(A_t))
  then has a constant tool part; this is stated in Section IV.
- **Prompt layout:** static prefix (system rules + tool docs) -> memory section -> task -> trajectory.
- **Decoding:** temperature 0. Streams differ in task order, sampled entities, drift set and error styles.
- **MaxSteps (fixed a priori, not tuned):** minimal number of calls of the template + 6.
- **Reflection prompt:** asks for one imperative fix instruction of at most 40 words, given (s_t, a_t, k_t).
- **k_t parser:** deterministic for 4xx/422 bodies, 404/410 and 200-with-error payloads; unrecoverable fields are
  `unknown`.

## 7. Conditions

**RQ1** (single LLM agent, sequential stream of tasks, generous fixed window, no overload):

| ID | Condition | Memory | Role |
|----|-----------|--------|------|
| C0 | ReAct | none | primary comparison |
| C1 | Reflexion-style (S5) | reflection after each non-transient error; the last k reflections are carried across tasks (FIFO; no parsing, retrieval or merge) | primary comparison |
| C2 | **EEM** | k_t + reflection, Zettelkasten store, retrieval (Section III-B) | method |
| C3 | EEM flat | C2 without links | secondary (TOST) |
| R  | Oracle docs | none; current schema shown | reference (upper bound), descriptive |
| A1 | EEM raw | raw error text stored and retrieved | ablation, descriptive |
| A2 | EEM no reflection | k_t stored with a template sentence | ablation, descriptive |
| A3 | EEM k=1 / k=5 | as C2 | RQ4, descriptive |
| C1w | within-task reflection | reflection kept until the task ends | optional extra arm |

**RQ3:** {C0, C2} x {ACC off, ACC on}. The concurrency arm is S7; population, scenario and t_llm basis are S4.
- The shared EEM store spans the LLM agents. A private-store ablation is descriptive.
- Recommended population: 32 scripted agents (RQ2's M = 32 load) plus 4 LLM agents, **36 agents on one gateway**.
  LLM request ids are offset so they never collide with scripted ids or the burst source.
- Each LLM agent solves a fixed number of tasks. The scripted load runs until all LLM agents finish (a new
  "until-done" workload mode, capped at T_max).
- Because the run length exceeds 120 s, a stationary scenario is used (S4); RQ2's timed events would only hit the
  first task.

## 8. Metrics and statistics

- **Per task:** success; steps; API calls including `get_api_docs`; LLM calls; non-transient errors split into drift
  vs agent/logic; first-call outcome at each exposure to a drifted endpoint; `get_api_docs` calls; prompt/output
  tokens; modelled LLM time; format errors.
- **Primary RQ1 endpoints** (unit = stream, paired across conditions):
  - E1: success rate, C2 vs C0
  - E2: API calls per solved task, C2 vs C0 (cost of recovery, incl. docs look-ups)
  - E3: success rate, C2 vs C1
  - E4: API calls per solved task, C2 vs C1
- **Tests:** exact paired sign-flip over 10 streams (smallest two-sided p = 2/1024), Holm across E1-E4, paired
  bootstrap 95% CIs (D4); TOST where "about equal" is the claim (margin 0.05 in success rate, 10% in calls per solved
  task). Ties reduce the effective n; this is reported.
- **Secondary:** C2 vs C3 (TOST); task-level mixed-effects logistic model (random effects for stream and template);
  per drift type, per error style, learning curves.
- **Mechanism check (not an endpoint):** repeat-exposure drift error rate. Among exposures to a drifted endpoint
  after its first exposure in the stream, the share whose first call to that endpoint fails with that endpoint's
  drift error. By construction this is about 1 for C0 unless it looks up the docs.
- **RQ3:** per-cell means, paired over seeds:
  - main effects of EEM and ACC on LLM-task success;
  - **non-inferiority of the EEM effect under ACC on**: (C2 - C0 | ACC on) >= (C2 - C0 | ACC off) - 0.05, on the
    difference scale fixed in advance, instead of a two-sided interaction p;
  - RQ2 metrics (overload rate, SLO violation, goodput, J with lambda = gamma = 0.5) per cell.

  10 seeds (smallest p = 2/1024; 5 seeds could never reach 0.05). Holm across the RQ3 primary set.
- **Ablations, oracle reference and RQ4 sweeps are descriptive** (5 streams; no tests).
- **Dev selection rule (written before the pilot):**
  - On 4 dev streams, pick (k, k_s, theta_link) by the highest mean dev success rate; ties within 0.02 go to fewer
    prompt tokens.
  - Pilot acceptance: C0 dev success between 0.2 and 0.8, otherwise task difficulty is recalibrated on dev only.
  - All analysis code is frozen before test streams run, as in RQ2.

## 9. Cost, resume and reproducibility

- **Budget model:** about 9 LLM calls per task (failing tasks run to MaxSteps; reflections included). c_step (wall
  seconds per LLM call on the T4) is unknown until S3.0. The figures below are an **illustration, not measured**.

| Block (20 tasks per stream) | Tasks | LLM calls | c_step 3 s | c_step 6 s |
|-------|------|-----------|-----------|-----------|
| Pilot (dev) | 200 | 1.8k | 1.5 h | 3 h |
| RQ1 dev selection (6 configs x 4 streams) | 480 | 4.3k | 3.6 h | 7.2 h |
| RQ1 test main (C0-C3 x 10 streams) | 800 | 7.2k | 6 h | 12 h |
| RQ1 descriptive (A1, A2, A3 x2, R x 5 streams) | 500 | 4.5k | 3.8 h | 7.5 h |
| RQ3 (4 cells x 10 seeds x 4 agents x 5 tasks) | 800 | 7.2k | 6 h | 12 h |
| **Total** | 2,780 | 25k | **~21 h** | **~42 h** |

  A free T4 gives a few hours per session and an unpredictable daily quota, so this spans days to weeks. S8 sets the
  scale after S3.0. Within one agent's trajectory, llama.cpp can reuse the KV cache for the static prefix. C0 and C2
  diverge from the first drifted call, so the response cache brings little cross-condition saving and is not used
  across conditions (below).
- **Resume by replay:**
  - Every LLM output is appended (atomic, checksummed JSONL on Drive) to the cache **of its result unit** (one
    stream-condition or one RQ3 run). The key is model file sha256, llama.cpp build, rendered chat template, grammar,
    sampling parameters, max_tokens, stop sequences, n_batch and the full prompt.
  - An interrupted unit restarts from the beginning: cached outputs replay instantly and the deterministic simulation
    reaches the same state, then continues.
  - Fresh greedy generations are not guaranteed bit-exact across runs (KV reuse, batch size), so the claim is
    **replay reproducibility**. The caches are archived on Drive as primary data, and units never share caches, so a
    result does not depend on run order.
- **Integrity:** spec hash, model sha256, llama-cpp-python version, prompt template version and sim version are stamped
  in every result; a resume with a different model or template is refused.

## 10. Milestones (each: code + tests -> review -> commit -> report)

- **S3.0 Feasibility on Colab T4** (you run a notebook, ~30-45 min):
  - download the GGUF (~4.7 GB);
  - measure prefill and decode tokens/s with and without the JSON grammar, and VRAM;
  - check JSON-envelope validity on 20 sample prompts and c_step on a toy task;
  - optionally compare a 3B model.
  
  The scale (S8) and the t_llm parameters are then fixed from these numbers.
- **S3.1** API semantics layer, drift injector, task sampler, checker (CPU; tests).
- **S3.2** Agent loop with `ScriptedBackend`; EEM parser and store; C0-C3, A1, A2 (CPU end-to-end tests).
- **S3.3** `LlamaCppBackend`, cache/replay, notebook `03_stage3.ipynb` (live progress, resume).
- **S3.4** Pilot on dev streams (difficulty calibration, prompt wording), dev selection, freeze.
- **S3.5** RQ1 test; **S3.6** RQ3; **S3.7** RQ4/descriptive arms; then Chapter IV.

## 11. Pre-registered expectations (written before any run)
- **RQ1:** C2 needs fewer API calls per solved task than C0 and C1, with at least equal success. The advantage is
  larger for terse errors and later in the stream. No strong prior for C3 vs C2.
- **RQ3:** ACC on raises LLM-task success under overload (fewer exhausted retries); EEM on raises it under drift; the
  EEM effect is not smaller under ACC on.
- **Limitation, stated in advance:** the LLM agents generate under 1% of the RQ3 load, so "EEM reduces load" is not
  testable in this design.

## 12. Text impact
- **Section IV (setup):** virtual time and t_llm; A_t = A; JSON envelope; `get_api_docs`; mixed population; store
  reset per stream; raw error shown in all conditions.
- **Section III (would need your OK, small):**
  - III-A, step definition: a format-invalid generation also counts as a step (no API call).
  - III-B, Algorithm 1 line 13: the observation keeps the raw error and adds the reflection.

## 13. Decisions needed (recommendation first)

| ID | Decision | Options | Recommendation |
|----|----------|---------|----------------|
| S1 | Execution environment (changes D1/D9 usage) | (a) in-process virtual time; FastAPI only as an optional demo wrapper; (b) real FastAPI + uvicorn, wall clock; (c) (a) + one FastAPI end-to-end demo run | (a) |
| S2 | Task suite (drops D1's StableToolBench) | (a) custom suite only; (b) (a) + BFCL v3 subset with our drift, after checking licence/format; (c) keep StableToolBench (paid judge) | (a); revisit (b) after RQ1 if time remains |
| S3 | Drift (replaces D5's p levels) | (a) 6 types, 8 of 16 endpoints (2 with two types), random error style, no second wave; (b) (a) + second drift wave mid-stream; (c) two drift levels (4 and 8 endpoints) as a factor (~2x cost) | (a) |
| S4 | RQ3 setting | (a) 32 scripted + 4 LLM agents, CAP-steady, until-done mode; t_llm from a served-model speed, with T4-measured values as a sensitivity run; (b) as (a) but t_llm = T4-measured; (c) all-LLM M = 8 on a scaled-down server (new calibration) | (a) |
| S5 | C1 definition | (a) cross-task FIFO of the last k reflections (C1w optional); (b) within-task reflection only; (c) both as arms (+25% cost) | (a) |
| S6 | Action format | (a) grammar-enforced JSON envelope, args free; (b) no grammar; (c) native tool-calling template | (a) |
| S7 | "ACC off" arm in RQ3 | (a) no window (same retries) in the 2x2, plus fixed W=4 with C2 only (5 cells); (b) fixed W=4 as "off"; (c) no window only | (a) |
| S8 | Scale | (a) section 9 (~25k LLM calls); (b) smaller: 15 tasks per stream, RQ1 descriptive arms on 3 streams; (c) decide after S3.0 | (c), target (a) |
| S9 | Model | (a) keep Qwen2.5-7B-Instruct Q4 (D9); (b) 3B; (c) compare in S3.0 | (a); (c) only if S3.0 shows c_step > 6 s |
| S10 | MaxSteps | (a) a priori minimal calls + 6; (b) tuned on dev | (a) |
| S11 | Primary RQ1 endpoints | (a) E1-E4 as in section 8; (b) success only (E1, E3); (c) add the mechanism check as primary | (a) |
| S12 | Section III edits of section 12 | (a) make them when Stage 3 code is final (patch r8); (b) Section IV only | (a) |

## References (checked 2026-11-15)
- [STB] Z. Guo et al., "StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language
  Models," arXiv:2403.07714. Checked (v4 HTML): uncached calls are simulated by gpt-4-turbo, gpt-4-turbo is the
  evaluator for Solvable Pass/Win Rate, and the paper notes the added cost. https://arxiv.org/abs/2403.07714
- [MIR] Z. Guo et al., "StableToolBench-MirrorAPI: Modeling Tool Environments as Mirrors of 7,000+ Real-World APIs,"
  arXiv:2503.20527, 2025. Checked (abstract): trains specialised LLMs to simulate API responses and integrates them
  into StableToolBench. The abstract does not say whether weights are released or how tasks are judged.
  https://arxiv.org/abs/2503.20527
- [BFCL-ES] EvalScope documentation, "BFCL v3": multi-turn uses state-based and response-based checks; no judge model
  mentioned. https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html (secondary source).
