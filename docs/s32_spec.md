# S3.2a specification: agent loop, EEM, conditions (CPU only)

_r4, 2026-11-22 (r3 + the user's second opinion on the implementation: D112-D114 and clear-cut fixes). Implements design r7 sections 3, 6 and 7 and Section III-B on top of the S3.1 suite
(`docs/s31_spec.md` r4). r1 was reviewed by a fresh-context subagent (4 High, 10 Medium, 9 Low); the user decided
D90-D101 (section 12); the clear-cut findings are applied below._

## 1. Scope (D90)

- **S3.2a (this spec):** the agent loop (Algorithm 1) as an asyncio coroutine; prompts; observations; the
  `LLMBackend` protocol with `ScriptedBackend`; the EEM pipeline (k_t parser, reflection, scrub, store with
  merge/link/retrieve, outcome update, possibly-outdated re-check); every condition (C0-C3, F, AN, R, A1, A2, C2-OU;
  SP = C0/C2 on SP streams); RQ1 units with a direct dispatcher (no overload); logs; CPU end-to-end tests.
- **S3.2b (own spec):** RQ3 wiring through the RQ2 `Gateway`/`MockServer` in virtual time (suite executed at
  admission), scripted background load in until-done mode, shared/private store across 4 LLM agents, hidden
  executions. S3.2a already fixes the shared-store timing rules (section 6.5) and tests them with two agents on a fake
  dispatcher.
- **S3.3:** `LlamaCppBackend`, the Qwen tokenizer (chat-template token counts), the MiniLM embedder, caches, notebook.

```python
backend.generate(messages, kind, grammar, max_tokens) -> Gen(text, n_in, n_out, finish_reason, grammar)
                       # grammar: None (free) | "envelope_cap300" (fallback) | "envelope" (grammar-always, no thought cap)
tokenizer.count(text) -> int;  tokenizer.count_messages(messages) -> int   # chat-template rendering, as llama.cpp n_in
embedder.embed(text) -> tuple[float, ...]       # cached by text hash; the cache is saved with each unit (S3.3 replay)
dispatcher.call(op_id, args, agent, k, task_index) -> Outcome(status, body, transient, truth, canonical_op)  (async)
memory = make_memory(condition, params, embedder, tokenizer)
result = await run_task(task, agent, memory, backend, dispatcher, cfg)
unit   = run_rq1_unit(spec, condition, backend, embedder, tokenizer, cfg) -> UnitResult
```

- **Call numbering (S3.1 D68 guard):** `k` is one running counter per agent across the whole stream (never reset per
  task), so `call_key = (agent, k)` is unique; both are ints.

## 2. Agent loop (Algorithm 1)

Per task, MaxSteps = min_calls + 1 + 6 (S3.1 reference plan):

1. `M_rel = memory.retrieve(s_t)` (section 5); the prompt is built at the start of the generation (section 3).
2. Context guard on the chat-template-rendered prompt: `tokenizer.count_messages(messages) + max_tokens <= 8,192`,
   else the task ends with reason `context_overflow`.
3. Generate freely (`action_max_tokens`, default 320). Strip one surrounding code fence (logged); the stripped text is
   what is parsed and, if it parses, appended as the assistant turn. Parse strictly (`feasibility.parse_action`). If
   it does not parse, regenerate once with the envelope grammar (fallback; thought cap 300 characters); both calls are
   charged and counted. If the fallback fails: format-error step, the fallback's raw text (400 characters) as the
   assistant turn, observation `invalid action format`, no API call, no EEM.
   - **Pilot switches as config (design section 8):** `grammar_always` (every action reply decoded with the envelope
     grammar, **no** thought cap, D65), `action_max_tokens` (640 switch), `docs_wording` (W1-W4).
4. `finish`: the task ends. **Success requires `finish` (D93)** and every sub-goal to hold on the final world state;
   the checker result is also logged at budget end (descriptive field `state_ok_at_end`).
5. Otherwise dispatch. **Success** = HTTP 200 without an error payload -> observation, `memory.on_success`.
   **Non-transient error** = any other non-retried status, or 200 with `{"status": "error"}` -> observation with the
   error, `memory.on_error` (EEM trigger), except for `get_api_docs` errors, which never create notes (D88).
   **Transient failure** (RQ3: retries exhausted) -> the transient notice; no EEM.
6. Every LLM call (action, fallback, reflection) is charged `t_llm = a + b_in*n_in + b_out*n_out` with the constants of
   its mode (D49) as a virtual-time sleep.
7. Reasons: `success`, `wrong_state` (finished, a sub-goal fails), `max_steps`, `context_overflow`.

A step = one action generation (+ fallback) and at most one API call; reflections are extra LLM calls, not steps.

## 3. Prompt layout

1. **system** (static prefix, byte-identical across steps and conditions except R's docs blocks; required by the
   D53.1 snapshot): fixed rules (one JSON object with exactly `thought` (one or two short sentences), `action`,
   `args`; `finish` with `{"summary": "..."}` ends the task; use ids returned by earlier calls; one API call per
   step), the docs-tool sentence (variant W1-W4, D94), and the tool docs: the 16 documented blocks
   (`docs.documented_all()`), then `get_api_docs` and `finish`. **R** shows the current blocks (re-rendered when the
   drift schedule changes) and keeps the same rules and docs-tool sentence (D97).
2. **user:** the memory section (fixed header `Notes from earlier API errors:` and one line per note; **omitted when
   there are no notes**) and the task instruction. With no notes, every memory condition's prompt is byte-identical to
   C0's (tested).
3. Trajectory: alternating **assistant** (section 2.3) and **user** (`Observation: ...`).

## 4. Observations

- Success: `Observation: 200 ` + the body as compact JSON, at most 1,500 characters (D100; asserted never truncated in
  the scripted runs; truncation is logged). **`get_api_docs`:** `Observation: 200` followed by the docs text verbatim
  (multi-line), identical in every condition.
- Non-transient error: `Observation: <status> ` + the body (`json.dumps` default spacing, or the plain text), cut at
  400 characters (the S3.1 bound). Then each condition that wrote a note for this error adds **its own just-written
  note** in its own format (D91): C2, C3, AN, C2-OU the EEM note; A2 the template sentence with k_t; C1 and F the
  reflection; A1 nothing (its note is the raw error already shown); C0 and R nothing.
- Transient: `Observation: the API is temporarily unavailable (the gateway retried and gave up). You may try again.`
- Format error: `Observation: invalid action format`.
- An action that is not an operation id is dispatched like any call (route 404 from the suite).

## 5. Retrieval query (D98)

`s_t` = the task instruction + the last action's name and argument keys (no values) + the first 200 characters of the
last observation's API part, without the note appended after an error (D105; it can contain ids; Section IV says so). q_t = Emb(s_t || desc(A_t)), desc(A_t) = the constant
list of the 18 tool names.

## 6. EEM

### 6.1 k_t parser (deterministic, version-stamped)

`k_t = (API_Name, Schema_Expected, Schema_Received, Error_Type)`. **API_Name = the id the agent called** (e.g. a
moved operation's old id, or `open_ticket` after the move); the suite's canonical operation is logged next to it and
used for the documented schema, exposures and retrieval recall.

Shapes (S3.1 `errors.py`, `suite.py`):

| Error | How recognised | Error_Type | Schema_Expected | Schema_Received |
|---|---|---|---|---|
| validation, informative | 422 with a list `detail` | from the **first** item's code (below) | per item, compact: `qty: missing`, `x: not allowed`, `amount: string`, `method: one of CARD|TRANSFER|WALLET`, `currency: "IDR"`, `amount: pattern` | sent keys with JSON types (`items[].qty: integer`); for enum errors also the sent value if it is a documented or current value of that field (D99) |
| validation, terse | 400 | `bad_request` | unknown | sent keys with types |
| moved, informative | 410 | `operation_removed` | `use '<new id>'` | `operation '<id>'` |
| route 404 | 404 with a plain-text body | `route_not_found` | unknown | `operation '<id>'` |
| resource 404 | 404 with a dict body | `resource_not_found` | `<resource> not found` | sent keys with types |
| business, informative | 409/422 with a string `detail` | `business_rule` | the sentence (value-free) | sent keys with types |
| business, terse | 409/422 with `error` | `business_rule` | unknown | sent keys with types |
| silent | 200 with `{"status": "error"}` | `silent_error` | informative: parsed from "Validation failed: field 'x': <msg>" as in the 422 row; terse: unknown | sent keys with types |
| other | - | `unknown` | unknown | unknown |

Item codes: `missing` -> missing_field; `extra_forbidden` -> unknown_field; `string_type`, `int_type`, `float_type`,
`bool_type`, `list_type`, `dict_type`, `model_type` -> type_error; `enum` -> enum_error; `literal_error` (also the
int01 "0 or 1") -> literal_error; `string_pattern_mismatch` -> format_error; `string_too_short`, `too_short`,
`finite_number` -> value_error. `error_keys` = the fields named by the items or the silent message (`loc[-1]` / `x`),
empty for terse errors.

- **Rendering:** the compact k_t rendering is capped at 30 tokens inside the note (design section 7). The reflection
  prompt and the embedding z use the **uncapped** rendering, so C2's reflection sees no less than C1's 400-character
  raw error. S3.3 tests with the Qwen tokenizer that the drift field and expected value survive the cap for every
  catalogue row x style.

### 6.2 Reflection and scrub

- Prompt (C2, C3, AN, C2-OU): the task instruction, the failed action and arguments, the uncapped k_t; one imperative
  fix instruction of at most 40 words. Free decoding, max_tokens 60, free constants.
- C1 and F: the same with the raw status and body (400 characters) instead of k_t.
- A2: no LLM call; the sentence `Fix the arguments of <API_Name> as stated above.` (D96).
- **Scrub (D92):** before storing, every task value in a reflection (and in A1's raw text) is replaced by a typed
  placeholder (`<customer_id>`, `<email>`, `<name>`, `<order_id>`, `<amount>`, `<text>`, ...): the task's entity ids,
  emails, names, product names, subjects and comment texts, amounts, and every scalar value from the task's arguments
  and observations, except enum, boolean and constant values of the schemas (longest match first, exact substring).
  Operation ids (documented and moved) are never scrub values, and `get_api_docs` arguments and bodies are not added.
  Numbers match on digit boundaries, text on word boundaries. **D103:** normalised forms are also matched: thousand
  separators (`125,000`, `125.000`), the number part of ids (`9761`), name tokens of 4+ letters, email local parts; S3.3
  adds an audit of real reflections for leftover entity values. The scrub rate is reported per condition (A1 counts
  its raw-text scrubs); a scripted mode whose reflections echo argument values tests it. **D104:** the note shown in
  the current task is the stored (scrubbed) text; steps whose arguments contain a `<...>` placeholder are counted per
  condition.
- A reflection that would overflow the context is skipped and logged; EEM stores the note without a reflection (as in
  A2); C1 and F store nothing.

### 6.3 Note and store (Section III-B)

Note: id (store counter), API_Name, canonical op, Error_Type, k_t, reflection (scrubbed), trigger args, error_keys,
outcome line (string + structured items), possibly_outdated, open, owner = (agent, task) of the last create/refresh,
links, z = Emb(reflection || uncapped k_t), history of line keys (for `known_keys`).

- **Merge:** same (API_Name, Error_Type) -> replace reflection, k_t, trigger args, error_keys and owner; remove the
  line (its keys stay in the history); reopen; recompute z and the note's links.
- **Link:** undirected links to every other note with cosine(z, z') > theta_link (cosines from the stored vectors,
  rounded to 1e-6).
- **Retrieve (C2, C2-OU, A1, A2):** seeds = top-k_s by cosine with q_t; then their 1-hop neighbours ranked by cosine
  with q_t, until k; then the next notes by cosine (fill, D98; added to the Section III patch r8 list). Ties by note id.
- **C3:** top-k by cosine, no links. **AN:** every note, ordered by note id (no budget; prompt tokens reported).
  **F:** the last k reflections, oldest first. **C1:** top-k reflections by cosine (no merge).
- **Note lines in the prompt**, in retrieval rank order, capped at 120 tokens each (k_t <= 30, reflection <= 60, line
  rest <= 30; cut from the end); budget k x 120:
  - EEM arms: `- <API_Name> (<Error_Type>): expected <...>; sent <...>. Fix: <reflection>. Last working change: <...>
    (possibly outdated)`;
  - A2: as EEM with the template sentence; A1: `- <API_Name>: <raw error>`;
  - C1 and F: `- <reflection>` (no k_t, design).

### 6.4 Outcome update (design section 6; S3.1 D76, D85, D88)

- **Same task** = the note's owner: open notes keyed on X are closed by a success of X by the same agent in the same
  task. A note left open after its task is closed only after a later refresh (new owner).
- **Line:** `leakcheck.last_working_items(...)` (new companion in S3.1 that returns the structured items;
  `last_working_change` renders the same items) with the note's trigger args, the successful args, the documented
  schema of the **canonical** op, the note's error_keys, and `known_keys` = every undocumented key ever introduced by
  any line on that operation, including lines a merge removed (the staleness probe needs this). S3.2 calls the S3.1
  reference, so it matches by construction; a test asserts every stored line equals it.
- **Moved:** a note with Error_Type `operation_removed` or `route_not_found` on X is also closed by a success, in its
  owner's task, of the id named in its pointer or in a `get_api_docs(X)` response, recording `use operation '<id>'`
  plus the diff against X's documented schema; a success of X itself closes it without a line.
- **New non-transient error on X:** the matching note merges; every other note on X keeps its line, marked possibly
  outdated. At the next success of X anywhere in the stream (any agent), each marked line is re-checked item by item:
  rename: new key present with its type and old key absent; removed: key absent; added: key present with its type
  (and value where recorded); type change: key has the recorded type; enum value: key has the recorded value;
  `use operation`: the call went to that id. Failing items are dropped; an empty line is removed; the mark is cleared.

### 6.5 Timing in a shared store (fixed now for S3.2b)

The prompt is built when the generation starts; a new or merged note becomes visible after its reflection's t_llm
sleep ends; the outcome update and the re-check run when the success response arrives. A two-agent test on a fake
dispatcher in S3.2a pins this down.

## 7. Conditions

| ID | Memory | Reflection | Store / retrieval | Outcome update | Shown after an error |
|---|---|---|---|---|---|
| C0 | none | - | - | - | - |
| C1 | reflection list (no merge) | raw-error prompt | embed reflection; top-k | no | reflection |
| C2 | EEM | k_t prompt | merge, links, seeds + neighbours + fill | yes | EEM note |
| C3 | EEM flat | k_t prompt | merge, top-k | yes | EEM note |
| F | last-k reflections (FIFO across tasks) | raw-error prompt | all k, oldest first | no | reflection |
| AN | EEM | k_t prompt | every note | yes | EEM note |
| R | none; current docs | - | - | - | - |
| A1 | raw error text notes (D101) | none | merge on (API_Name, Error_Type); z = Emb(scrubbed raw text); C2 retrieval | no | - |
| A2 | EEM | template sentence | as C2 | yes | note |
| C2-OU | EEM | k_t prompt | as C2 | no | EEM note |

SP = C0 and C2 on SP streams. The store is reset at the start of every unit.

## 8. ScriptedBackend (CPU tests and sanity runs)

- A **pure function** of (messages, task instance, strategy): everything it uses comes from the prompt (the task
  instance is passed as a side channel only to know the template and its reference `solve`).
- It plans the next call by replaying `solve` against the success bodies parsed from the observations so far
  (errors never change state; branches come from observed bodies; a silent 200 counts as a failure; it asserts that
  no success body is truncated).
- Strategies: `naive` (documented form), `docs` (after an error on X: `get_api_docs(X)` if no docs observation for X
  is in the prompt, then `oracle.docs_translate`), `memory` (apply the unmarked "Last working change" lines of notes
  on X in the prompt; possibly-outdated lines are ignored; otherwise `docs`).
- Reflections: a fixed template; modes inject malformed replies (fence, garbage) and reflections that echo argument
  values (scrub test).
- The `memory` strategy reads exactly the line format the code writes, so its results are **plumbing tests, never
  evidence**.
- The test embedder is a hashed bag of words (deterministic, meaningful overlap), so links and retrieval are not noise.

## 9. Logs

- Per step: agent, task, step, action, API_Name and canonical op, arg keys, status, truth (never shown), transient,
  LLM calls (kind, n_in, n_out, grammar, finish_reason, t_llm), thought length (D50), fence/fallback/format flags,
  context events, retrieved note ids (rank order), scrub count.
- **Note event log:** the full note at every create, merge, close, mark and re-check, with the trigger's truth
  `pair_ids`, virtual time, and for close/re-check the successful arguments, `moved_to` and `known_keys`, so every line
  can be recomputed independently (note correctness before and after the outcome update; note precision).
- Per step also: the notes visible at retrieval (id, canonical op; the retrieval-recall denominator), a hash of the
  rendered memory lines, the unparsed first reply when a fallback was needed, the grammar id of every call.
- Per task: success, reason, `state_ok_at_end`, steps, API calls (incl. `get_api_docs`), LLM calls, tokens, t_llm,
  drift vs agent errors, first-call outcome at each exposure to a drifted operation (planned vs experienced), skipped
  guessable reads, echo-trap tags (an error on an enum field whose sent value equals a documented value echoed in an
  earlier success body of the task), retrieval-recall records.
- Per unit: stamp (suite version, spec sha256, condition, params, parser and prompt-template versions, backend,
  tokenizer and embedder ids, prompt-constants hash, code commit), final notes, and the unit's own embedding cache
  (every vector it used, keyed by text sha256). JSONL with atomic writes on Drive comes in S3.3.

## 10. Parameters

Pilot defaults: k = 3, k_s = 2, theta_link = 0.65; selection grid k in {3, 5} (k_s 2 / 3) x theta_link in
{0.5, 0.65, 0.8}. Note cap 120 tokens, budget k x 120.

## 11. Tests (pytest, CPU)

- Parser: every S3.1 shape (all catalogue rows x both styles, business, resource/route 404, 410, silent) maps to 6.1;
  error_keys; no entity value in any k_t (property test over 200 streams).
- Store: merge, links, retrieval order (seeds, neighbours, fill, ties), C3, AN, F, C1; caps; empty-store prompt equals
  C0's byte for byte; system turn identical across conditions (except R).
- Outcome update: lines equal the S3.1 reference; owner rule; moved closing via pointer and via get_api_docs; moved +
  second drift (canonical op); possibly-outdated marking and item-wise re-check; known_keys history (SP); no notes from
  get_api_docs errors; reflection overflow; per-condition text shown after an error.
- Scrub: echoing reflections are scrubbed; no entity value in any stored or shown note field, all conditions, 200
  streams.
- Loop: fence, fallback, format error, grammar_always, action_max_tokens, context overflow, max_steps, finish
  required, transient notice, t_llm per mode, step and call counting, call numbering across tasks.
- Two agents on a fake dispatcher: the timing rules of 6.5.
- End to end (scripted, 20 dev streams; plumbing): no `wrong_state`; with `memory`, C2/C3 solve every task C0 with
  `docs` solves and make fewer repeat-exposure drift errors; `naive` C0 fails on drifted operations; every condition
  runs; determinism (same unit twice -> byte-identical logs; different PYTHONHASHSEED).

## 12. Decisions (user, 2026-11-22)

**r4 clear-cut changes (second opinion):** EEM uses only what the agents observed: the canonical operation of an id
comes from an alias map learned from 410 pointers, `get_api_docs` responses and moved items (the suite's operation is
logged only); the k_t enum whitelist is the documented values plus the values the error itself states; the scrub's
safe list is the documented vocabulary plus the schema vocabulary each agent has observed. k_t is scrubbed as defence
in depth. Retrieval recall is computed from the notes visible at retrieval. A unit can be replayed with its exported
embedding vectors (`embedding_preload`). **Analysis obligations:** C2 vs C0 mixes within-task help (reflection call
and the note shown after an error) with cross-task memory, so first-exposure and repeat-exposure outcomes are
reported separately; echo-trap counts are an upper bound (an echoed value can equal the documented value the agent
would send anyway); prompt tokens per condition are reported.


| ID | Decision |
|---|---|
| D90 | Split: S3.2a (RQ1 loop, EEM, conditions) and S3.2b (RQ3 wiring) |
| D91 | After an error each condition shows its own just-written note (C1/F reflection, A2 template + k_t, EEM arms the EEM note, A1 nothing) |
| D92 | Reflections (and A1 raw text) are scrubbed of task values before storing; scrub rate reported |
| D93 | Success requires `finish`; the state at budget end is logged descriptively |
| D94 | Docs-tool wording variants W1-W4 (W4 minimal: "get_api_docs(operation) returns the current schema of an operation.") |
| D96 | A2's template: "Fix the arguments of <API_Name> as stated above." |
| D97 | R keeps the same rules and docs-tool sentence; only the docs blocks differ |
| D98 | Query text = instruction + last action (keys) + 200 characters of the last observation; retrieval fills to k with the next most similar notes |
| D99 | Schema_Received = keys + JSON types; enum values only if they are documented or current values of that field |
| D100 | Success observations capped at 1,500 characters (asserted never truncated in scripted runs) |
| D101 | A1 = scrubbed raw error text, merged on (API_Name, Error_Type), C2 retrieval, no reflection or line |
| D103 | Scrub also matches normalised forms (separators, id number parts, name tokens, email local parts); S3.3 audits real reflections |
| D104 | The note shown in the current task is the stored (scrubbed) text; steps with placeholder arguments are counted per condition |
| D105 | The query's observation part excludes the note appended after an error |
| D106 | A long outcome line is shortened at item boundaries (whole items dropped from the end, "..." added); the "(possibly outdated)" mark is always kept |
| D112 | Note budget inside 120 tokens: k_t <= 30, the outcome line (label and mark included) reserved first at <= 40 tokens, the reflection gets the rest; line_shown and items shown/total logged per retrieved note |
| D113 | A merge keeps the note's old items as a possibly-outdated part, re-checked item by item at the next success; the new diff's items are added at close; items kept are logged |
| D114 | Only errors that can signal drift (validation 400/422, silent, route 404, 410) mark other notes on the operation; business-rule and resource-not-found errors do not |
| D102 | Enum remaps are recorded in the outcome line as old -> new (e.g. `'method': "card" -> "CARD"`; both schema values), not only the value used in one task |

Docs-tool variants: W1 "If a call fails because the API may have changed, call get_api_docs with the operation name to
see its current schema." W2 "The documentation above may be out of date; get_api_docs(operation) returns the current
schema of an operation." W3 "Before retrying a failed call, check get_api_docs(operation) for the current
parameters." W4 "get_api_docs(operation) returns the current schema of an operation." W3 may shrink the room for
E2/E4 (it pushes C0 to the docs); that follows from the design rule (maximise C0) and is reported.
