# Stage 3 design proposal: API environment with schema drift, EEM, LLM agent loop (RQ1, RQ3, RQ4)

_Draft r1, 2026-11-15, for the user's decision. Nothing here is implemented yet. Every choice marked **(decide)** is
listed in section 13 with options and a recommendation. Section 2 lists what earlier decisions already fix._

## 1. Purpose and research questions

| RQ | Question | Where answered |
|----|----------|----------------|
| RQ1 | Does EEM improve recovery from API-contract errors (schema drift) across a stream of tasks? | Stage 3, single LLM agent, no overload |
| RQ2 | Does ACC protect the API under concurrency? | Done (frozen config, test results, section 8 of HANDOFF) |
| RQ3 | How do EEM and ACC combine when drift and overload occur together? | Stage 3, 2x2 factorial, LLM agents under load |
| RQ4 | How sensitive are the results to hyperparameters? | ACC: RQ2 dev grid (already run, 243 configs) + boundary check; EEM: k and theta_link on RQ1 dev streams |

## 2. Already fixed by earlier decisions (not reopened unless you want to)
- D2/D9: free local LLM on Colab T4: **Qwen2.5-7B-Instruct, 4-bit GGUF, llama-cpp-python**; embeddings
  **all-MiniLM-L6-v2**; D1/D9 said "FastAPI + uvicorn mock server" (see S1: I propose to change how it is used).
- D3/D6: EEM baselines (ReAct; Reflexion-style reflection; EEM; EEM with a flat store) and ablations (raw traceback
  instead of k_t; k_t without reflection; k in {1,3,5}; private store per agent as a small ablation).
- D5: drift types renamed parameter, type change, new required field; fault rates around 0.1-0.3.
- D7: hyperparameters chosen on a dev split with separate seeds, frozen before test runs.
- Section III (approved r7): EEM handles only non-transient errors (4xx other than 408/429, 5xx other than
  500/502/503/504, schema/validation errors, HTTP 200 with an error payload); shared store; merge on
  (API_Name, Error_Type); link above theta_link; retrieve k_s seeds + 1-hop neighbours up to k; exhausted retries give a
  transient-failure observation that costs one step.
- RQ2's frozen ACC configuration (`configs/rq2_frozen.json`) is used unchanged in RQ3 (no re-tuning).

## 3. Architecture

```
            +-------------------- one virtual-time event loop (deterministic) --------------------+
 LLM agents (Algorithm 1) --+                                                                      |
   prompt -> LLMBackend ----|--> Gateway (ACC window or baseline, retries) --> MockServer (RQ2:    |
   EEM store (shared) <-----|                                                   429/503/latency)   |
 scripted background agents |                                                   --> API semantics  |
   (RQ3 only, S4) ----------+                                                   (state, schema,    |
                                                                                drift, 422/404/200)|
            +--------------------------------------------------------------------------------------+
 LLMBackend = CachedBackend(LlamaCppBackend | ScriptedBackend)   cache on Drive (prompt hash -> output)
```

- **Virtual time (decide, S1).** The RQ2 simulator already runs agents, gateway and server on a virtual clock. Stage 3
  adds an API-semantics layer behind the existing `MockServer` (rate limit and overload checks first, then schema
  validation and execution, as a real gateway would). An LLM call blocks the event loop, so virtual time does not
  advance while the GPU works; the agent is then charged a modelled LLM time
  `t_llm = a + b * output_tokens` (a, b measured on the T4 in S3.0), which is deterministic given the cached output.
  Consequences: (i) RQ3 can use RQ2's calibrated server and frozen ACC unchanged; (ii) a slow T4 does not distort the
  API load pattern; (iii) runs are reproducible and can be resumed by replay (section 9).
- **API semantics layer.** In-memory state (users, products, orders, payments, shipments, tickets), endpoint
  handlers, a schema per endpoint and version, a drift injector, and a programmatic checker per task. Pure Python,
  CPU-only, unit-testable.
- **LLM backend interface.** `generate(messages, schema) -> (text, n_in, n_out)`. Implementations: `LlamaCppBackend`
  (Colab T4), `ScriptedBackend` (deterministic rule-based policy for CPU tests and sanity runs), `CachedBackend`
  (wraps either; sha256 of model id + quantisation + sampling parameters + full prompt -> output).

## 4. Task suite and environment (decide, S2)

- **Custom synthetic API suite (recommended)**: about 12 endpoints in 3 related domains (customers/orders/payments,
  inventory/shipping, support tickets), each with a documented schema shown to the agent as a short OpenAPI-style
  summary. About 25 task templates, each needing 2-6 calls with data dependencies (e.g. find customer by email -> list
  cart -> create order -> pay -> request shipment). Instances are sampled from templates with fresh entities; success
  is checked against the final server state (Section III-A: "verified by a programmatic checker against the
  ground-truth environment state").
- A tool `get_api_docs(endpoint)` returns the **current** schema (costs a step). It makes terse errors recoverable and
  gives EEM something to save: once a lesson is stored, later tasks need not look the schema up again.
- **External benchmark (optional, decide S2).** StableToolBench needs gpt-4-turbo as its evaluator (Solvable Pass
  Rate) and as the simulator of uncached APIs [STB], so it does not fit the zero-cost constraint (D2). BFCL v3
  multi-turn evaluates state- and response-based on executable backends without a judge model [BFCL-ES]; injecting
  drift into it would be our own modification, so it would add external flavour but not an external comparison.
  I have not checked its licence or task format in detail yet.

## 5. Schema drift (decide, S3)

- **Drift types:** (1) renamed parameter; (2) type change (e.g. amount number -> string with currency, id int -> "C-"
  prefixed string); (3) new required field; (4) enum value change; (5) endpoint moved (404 or 410 with a pointer or
  without); (6) "silent" error: HTTP 200 with `{"status": "error", ...}` payload. (1)-(3) are D5; (4)-(6) are proposed
  additions.
- **Per stream:** a drift set of 4 of the 12 endpoints (about 1/3; D5's "fault rate") is applied before the first
  task and stays in force; the agent's docs remain the **old** schema. Drift sets for dev and test streams come from
  disjoint pools of (endpoint, drift type) pairs, so EEM settings are not tuned on the test lessons.
- **Error messages:** each drifted endpoint is either *informative* (FastAPI-style 422 listing the bad field and the
  expected type) or *terse* ("400 Bad Request"), fixed per endpoint, half and half. Results are reported per style:
  with informative messages a memory-less agent can repair the call within the task, so EEM's value is mainly in
  avoiding the repeat; with terse messages repair needs `get_api_docs` or trial and error.
- Optional second drift wave at the middle of the stream (lessons become stale; tests the merge rule). Proposed as
  an RQ1 extension only if budget allows.

## 6. Agent loop (Algorithm 1) details

- **Action format:** one JSON object per step `{"thought": str, "action": str, "args": object}`, enforced by a JSON
  grammar for the envelope only (llama.cpp constrained decoding); `args` is free so the agent can still make schema
  mistakes. Formatting failures are counted separately and are not API errors (decide, S6).
- **Tools A_t:** all ~12 endpoints plus `get_api_docs` and `finish`; A_t = A (the suite is small enough for every
  prompt). Section III's "retrieves a set of candidate API tools A_t ⊂ A" then holds trivially; Section IV says so.
- **Prompt layout (for KV-cache reuse):** static prefix (system rules + tool docs, identical for all agents and
  conditions) -> memory section (EEM notes; empty section text is identical across conditions) -> task -> trajectory.
- **Temperature 0** (greedy). Variation between streams comes from task order, sampled entities and drift sets.
- **MaxSteps:** chosen on dev (e.g. minimal calls + 6); a transient-failure observation counts as one step.
- **Reflection prompt:** short, asks for one imperative fix instruction (<= 40 words) given (s_t, a_t, k_t).
- **Failure parsing k_t:** deterministic parser for 4xx/422 bodies, 404/410 and 200-with-error payloads; fields not
  recoverable from the message are `unknown`.

## 7. Conditions

**RQ1 (single LLM agent, sequential stream of tasks, generous fixed window, no overload)**

| ID | Condition | Memory |
|----|-----------|--------|
| C0 | ReAct | none |
| C1 | Reflexion-style | reflection after each non-transient error, kept only until the task ends |
| C2 | **EEM** | k_t + reflection, Zettelkasten store, retrieval (Section III-B) |
| C3 | EEM flat | as C2 without links (top-k by similarity only) |
| R  | Oracle docs (reference, decide S5) | none, but the agent sees the current schema (upper bound) |
| A1 | EEM raw | raw error text stored and retrieved, no parsing, no reflection |
| A2 | EEM no reflection | k_t stored, no LLM reflection (template sentence from k_t) |
| A3 | EEM k=1, k=5 | RQ4 |

**RQ3 (decide S4): 2x2 factorial {C0, C2} x {ACC off, ACC on}**, RQ2 scenario(s) as background, shared EEM store
across the LLM agents; plus "EEM private store" as a small ablation (D). **ACC off (decide S7):** no window limit, same
retry/backoff (what a plain agent framework does); best global fixed window W=4 from RQ2 as a secondary arm.
**Population (decide S4):** m = 4 LLM agents plus RQ2's scripted agents (M = 32 cell) sharing one gateway; the LLM
agents each solve a fixed number of tasks (e.g. 10), the scripted load runs until they finish.

## 8. Metrics and statistics

- **Per task:** success (checker), steps, API calls, invalid calls (non-transient errors), repeated errors (a
  non-transient error whose (API_Name, Error_Type) key already occurred earlier in the stream, in any agent),
  recovery (success after the first error), `get_api_docs` calls, prompt/output tokens, LLM time, format errors.
- **Per stream/run:** success rate, repeated-error rate, invalid-call rate, steps per solved task, tokens per task,
  learning curve over task index; RQ3 adds the RQ2 metrics (overload rate, SLO violation, goodput, J with
  lambda = gamma = 0.5).
- **Unit of analysis = stream (paired across conditions: same tasks, same order, same drift).** Primary RQ1
  endpoints (C2 vs C0: repeated-error rate, success rate; C2 vs C1: success rate; C2 vs C3: success rate), exact
  sign-flip test over streams, Holm across the primary set, TOST where "about equal" is the claim; secondary:
  task-level mixed-effects logistic model (random effects for stream and template), descriptive per drift type and
  error style. RQ3: per-cell means with paired sign-flip tests for the two main effects and the interaction contrast
  (C2/ACC on - C2/ACC off) - (C0/ACC on - C0/ACC off). Analysis code is written and frozen before test streams run,
  as in RQ2.

## 9. Cost, resume and reproducibility

- **Budget model.** Wall time ~= (LLM steps + reflections) x c_step; c_step is unknown until S3.0. Illustration
  only (NOT measured): with 30 tasks x ~6 steps per stream and c_step = 4 s, one stream-condition takes ~12 min.

| Block | Runs | at c_step = 3 s | at c_step = 6 s |
|-------|------|-----------------|-----------------|
| Pilot/calibration (dev) | ~10 stream-conditions | ~1.5 h | ~3 h |
| RQ1 dev selection (6 EEM configs x 3 streams x 20 tasks) | 18 | ~2 h | ~4 h |
| RQ1 test main (C0-C3 x 10 streams x 30 tasks) | 40 | ~6 h | ~12 h |
| RQ1 ablations + reference (A1, A2, A3 x2, R x 5 streams) | 25 | ~4 h | ~8 h |
| RQ3 (4 cells x 1 scenario x 5 seeds x 40 LLM tasks) | 20 | ~4 h | ~8 h |
| **Total** | | **~18 h** | **~35 h** |

  The free T4 gives a few hours per session and an unpredictable daily quota, so this spans several days; S8 offers
  smaller scales. Prompt-prefix reuse in llama.cpp and the response cache (C0 and C2 prompts are identical until the
  first error) reduce this; by how much is measured in the pilot.
- **Resume by replay.** Every LLM output is appended (atomic, checksummed JSONL on Drive) under its prompt hash. A run
  interrupted mid-stream is restarted from the beginning; cached outputs replay instantly and the deterministic
  simulation reaches the same state, then continues with new LLM calls. Each finished stream-condition is a result
  unit (`run_resumable`, as in RQ2). Embeddings are cached the same way.
- **Integrity:** spec hash, model file sha256, llama-cpp-python version, prompt template version and sim version are
  stamped in every result; a resume with a different model file or template is refused.

## 10. Milestones (each: code + tests -> review -> commit -> report)

- **S3.0 Feasibility on Colab T4 (you run a notebook, ~30 min):** download the GGUF (~4.7 GB), measure prefill and
  decode tokens/s, VRAM, JSON-envelope validity on 20 sample prompts, c_step on a toy task; optional 3B comparison.
  Go/no-go and final scale (S8) are decided on these numbers.
- **S3.1** API semantics layer, drift injector, task templates, checker (CPU; tests).
- **S3.2** Agent loop with `ScriptedBackend` (CPU end-to-end tests); EEM parser, store (merge, link, retrieve),
  baselines C0-C3, A1-A2.
- **S3.3** `LlamaCppBackend`, cache/replay, Colab notebook `03_stage3.ipynb` (live progress, resume).
- **S3.4** Pilot on dev streams: calibrate task difficulty (target: C0 success clearly between 0 and 1) and drift
  set; choose EEM parameters and MaxSteps on dev; freeze.
- **S3.5** RQ1 test; **S3.6** RQ3; **S3.7** RQ4 EEM sweep; then Chapter IV.

## 11. Pre-registered expectations (written before any run)
- RQ1: C2 has a lower repeated-error rate than C0 and C1, and at least equal success; the advantage is larger for
  terse error messages and late in the stream. C3 vs C2: no strong prior (links matter only when related notes exist).
- RQ3: ACC on raises LLM-task success under overload (fewer exhausted retries); EEM on raises success under drift;
  no prior on the sign of the interaction. With a mixed population the LLM agents' own calls are a small share of the
  load, so an "EEM reduces load" effect is not testable in this design (stated as a limitation).

## 12. Possible Section III / IV text impact
Section III: none required by the recommendations (A_t = A, JSON envelope, `get_api_docs` and virtual time are
experimental-setup details for Section IV). If you choose S4 option (b) or tool retrieval, a sentence in III-A changes.

## 13. Decisions needed (options; recommendation first)

| ID | Decision | Options | Recommendation |
|----|----------|---------|----------------|
| S1 | Execution environment | (a) in-process virtual time (RQ2 simulator + API layer), FastAPI only as a thin optional demo wrapper; (b) real FastAPI + uvicorn, wall clock, real LLM latency; (c) (a) for experiments + (b) for one end-to-end demo run | (a), or (c) if you want the FastAPI claim in the paper. Changes how D1/D9's FastAPI is used |
| S2 | Task suite | (a) custom synthetic suite only; (b) (a) + BFCL v3 multi-turn subset with our drift injected (after checking licence/format); (c) StableToolBench (needs paid gpt-4-turbo) | (a); revisit (b) after RQ1 if time remains |
| S3 | Drift and errors | (a) 6 drift types, 4 of 12 endpoints per stream, informative/terse half-half, no second wave; (b) as (a) + second wave; (c) D5 only (3 types), informative only | (a) |
| S4 | RQ3 population | (a) 4 LLM agents + RQ2 scripted load (M = 32 cell), frozen ACC unchanged; (b) all-LLM, M = 8, server scaled down (new calibration, ACC re-tuned or transferred) | (a) |
| S5 | RQ1 conditions | (a) C0-C3 main + A1, A2, A3 (5 streams) + oracle reference R; (b) (a) without R; (c) C0, C2, C3 main only | (a) |
| S6 | Action format | (a) JSON envelope by grammar, args free; (b) no grammar (format errors are part of the result); (c) native tool-calling chat template | (a) |
| S7 | "ACC off" in RQ3 | (a) no window + same retries, fixed W=4 as secondary; (b) fixed W=4 only; (c) no window only | (a) |
| S8 | Scale | (a) as section 9 (10 test streams x 30 tasks, RQ3 1 scenario x 5 seeds); (b) smaller: 6 streams x 25 tasks, RQ3 3 seeds; (c) decide after S3.0 numbers | (c), with (a) as the target |
| S9 | Model | (a) keep Qwen2.5-7B-Instruct Q4 (D9); (b) 3B for speed; (c) compare both in S3.0, then decide | (a), with (c) only if S3.0 shows c_step > 6 s |

## References (checked 2026-11-15)
- [STB] Z. Guo et al., "StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language
  Models," arXiv:2403.07714. Checked (v4 HTML): uncached calls simulated by gpt-4-turbo; gpt-4-turbo is the evaluator
  for Solvable Pass/Win Rate; the paper notes the added cost. https://arxiv.org/abs/2403.07714
- [BFCL-ES] EvalScope documentation, "BFCL v3": multi-turn uses state-based and response-based checks; no judge model
  mentioned. https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html (secondary source; the primary BFCL
  repository is not yet checked).
