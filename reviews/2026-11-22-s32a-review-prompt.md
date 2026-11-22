# Review request: S3.2a implementation (agent loop, EEM, conditions), 2026-11-22

You are a strict reviewer of LLM-agent experiments and memory methods. Be concrete; no praise.

## Context
Thesis: LLM agents (Qwen2.5-7B, local, greedy) solve multi-call tasks against a synthetic API whose schema drifts
(S3.1, reviewed). S3.2a implements the agent loop (Algorithm 1), the EEM memory (k_t error parser, LLM reflection with
a scrub of task values, Zettelkasten-style store with merge/link/retrieve, a deterministic "Last working change"
outcome update from the S3.1 reference diff, possibly-outdated marking) and all conditions: C0 ReAct, C1 reflection
RAG, C2 EEM, C3 EEM without links, F FIFO reflections, AN all notes, R oracle docs, A1 raw error text, A2 template
reflection, C2-OU no outcome update. Primary endpoints: success (non-inferiority then superiority) and API calls on
jointly solved tasks, C2 vs C0 and C2 vs C1, 14 paired test streams. A scripted CPU stand-in for the LLM runs the
pipeline end to end; results with it are plumbing, not evidence. The spec (below) went through two internal reviews;
the code through a review and a re-review. 223 S3.2a tests pass (692 in the repo).

## Checklist
1. Fairness between conditions: anything other than the memory itself that differs between C0, C1 and C2 (prompt
   bytes, observation text, reflection prompts, note shown after an error, retrieval query, budgets)?
2. Is EEM as implemented a faithful, defensible operationalisation of Section III-B? Is anything likely to make C2 fail
   for reasons unrelated to the idea (note format, caps, retrieval query, merge key granularity, outcome-update rules)?
3. Leak safety of the shared store (k_t, reflections after the scrub, A1 text, outcome lines).
4. Measurement: do the logs support the planned mechanism checks (note correctness before/after the outcome update,
   note precision, retrieval recall, exposures planned vs experienced, echo traps, skipped reads)?
5. Readiness for S3.2b (4 concurrent LLM agents sharing the store through a gateway) and S3.3 (llama.cpp backend,
   Qwen tokenizer, MiniLM embedder, cache/replay).

## Output
Findings ranked by severity with concrete fixes; then a verdict on whether S3.2b/S3.3 can build on this.

## The spec (docs/s32_spec.md, r3)

# S3.2a specification: agent loop, EEM, conditions (CPU only)

_r3, 2026-11-22 (r2 + the code review and its re-review: D102-D106 and clear-cut fixes). Implements design r7 sections 3, 6 and 7 and Section III-B on top of the S3.1 suite
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
| D102 | Enum remaps are recorded in the outcome line as old -> new (e.g. `'method': "card" -> "CARD"`; both schema values), not only the value used in one task |

Docs-tool variants: W1 "If a call fails because the API may have changed, call get_api_docs with the operation name to
see its current schema." W2 "The documentation above may be out of date; get_api_docs(operation) returns the current
schema of an operation." W3 "Before retrying a failed call, check get_api_docs(operation) for the current
parameters." W4 "get_api_docs(operation) returns the current schema of an operation." W3 may shrink the room for
E2/E4 (it pushes C0 to the docs); that follows from the design rule (maximise C0) and is reported.


## Key code

### src/eem_acc/stage3/agent/memory.py

```python
"""Memory for every condition (S3.2a spec sections 6.3-6.5, 7): the EEM store (C2, C3, AN, A1, A2, C2-OU), the
reflection list (C1), the FIFO buffer (F) and no memory (C0, R)."""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..apisuite.leakcheck import item_holds, last_working_items, render_item, render_items
from ..apisuite.ops import OPS
from .backend import Embedder, Tokenizer, cosine
from .parser import KT, moved_pointer
from .prompts import A2_TEMPLATE

NOTE_CAP, KT_CAP, REFL_CAP, LINE_CAP = 120, 30, 60, 30
MOVED_TYPES = ("operation_removed", "route_not_found")


@dataclass
class ErrorCtx:
    agent: int
    task: int
    step: int
    api_name: str                 # the id the agent called
    canonical_op: str             # the suite's operation (== api_name for unknown routes)
    args: Any
    status: int
    body: Any
    raw_text: str                 # "<status> <body>" cut at 400 characters
    kt: KT
    truth: dict
    t: float = 0.0                # virtual time (note visibility, S3.2a spec 6.5)


@dataclass
class SuccessCtx:
    agent: int
    task: int
    step: int
    called_id: str
    canonical_op: str
    args: Any
    body: Any
    t: float = 0.0


@dataclass
class Note:
    id: int
    api_name: str
    canonical_op: str
    error_type: str
    kt: KT
    reflection: str | None
    trigger_args: Any
    owner: tuple
    text_raw: str | None = None          # A1
    items: list = field(default_factory=list)
    open: bool = True
    possibly_outdated: bool = False
    links: set = field(default_factory=set)
    z: tuple = ()
    truth: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {"id": self.id, "api_name": self.api_name, "canonical_op": self.canonical_op,
                "error_type": self.error_type, "kt": {"expected": self.kt.expected, "received": self.kt.received,
                                                     "error_keys": sorted(self.kt.error_keys)},
                "reflection": self.reflection, "text_raw": self.text_raw, "trigger_args": self.trigger_args,
                "owner": list(self.owner), "items": copy.deepcopy(self.items), "line": render_items(self.items),
                "open": self.open, "possibly_outdated": self.possibly_outdated, "links": sorted(self.links),
                "truth": self.truth}


class Memory:
    """Base: no memory (C0, R)."""
    reflection_mode: str | None = None   # "kt" | "raw" | "template" | None
    name = "none"

    def __init__(self, embedder: Embedder | None = None, tokenizer: Tokenizer | None = None, k: int = 3):
        self.embedder, self.tokenizer, self.k = embedder, tokenizer, k
        self.events: list[dict] = []

    def retrieve(self, query: str) -> list:
        return []

    def render(self, selected: list) -> list[str]:
        return []

    def selected_ids(self, selected: list) -> list:
        return []

    def on_error(self, ctx: ErrorCtx, reflection: str | None) -> str | None:
        return None

    def on_success(self, ctx: SuccessCtx) -> None:
        return None

    def notes_on(self, op: str) -> list[int]:
        return []

    def visible(self) -> list:
        return []

    def final_state(self) -> list[dict]:
        return []


class ReflectionList(Memory):
    """C1: reflections after non-transient errors, no merge; top-k by cosine with the query."""
    reflection_mode = "raw"
    name = "reflection_rag"

    def __init__(self, embedder, tokenizer, k=3):
        super().__init__(embedder, tokenizer, k)
        self.items: list[dict] = []

    def on_error(self, ctx, reflection):
        if not reflection:
            return None
        it = {"id": len(self.items) + 1, "text": reflection, "z": self.embedder.embed(reflection),
              "canonical_op": ctx.canonical_op, "owner": [ctx.agent, ctx.task]}
        self.items.append(it)
        self.events.append({"event": "create", "id": it["id"], "text": reflection, "truth": ctx.truth,
                            "canonical_op": ctx.canonical_op, "t": round(ctx.t, 6)})
        return "Reflection: " + self._cap(reflection)

    def _cap(self, text):
        return self.tokenizer.truncate(text, REFL_CAP)

    def retrieve(self, query):
        q = self.embedder.embed(query)
        ranked = sorted(self.items, key=lambda it: (-cosine(q, it["z"]), it["id"]))
        return ranked[:self.k]

    def render(self, selected):
        return [f"- {self._cap(it['text'])}" for it in selected]

    def selected_ids(self, selected):
        return [it["id"] for it in selected]

    def notes_on(self, op):
        return [it["id"] for it in self.items if it["canonical_op"] == op]

    def visible(self):
        return [[it["id"], it["canonical_op"]] for it in self.items]

    def final_state(self):
        return [{k: v for k, v in it.items() if k != "z"} for it in self.items]


class FifoReflections(ReflectionList):
    """F: the last k reflections, carried across tasks, oldest first (no retrieval)."""
    name = "fifo"

    def retrieve(self, query):
        return self.items[-self.k:]


class EEMStore(Memory):
    """C2 (links), C3 (flat), AN (all notes), A2 (template reflection), C2-OU (no outcome update), A1 (raw text)."""

    def __init__(self, embedder, tokenizer, k=3, k_s=2, theta_link=0.65, *, links=True, outcome=True,
                 all_notes=False, mode="eem", name="eem"):
        super().__init__(embedder, tokenizer, k)
        self.k_s, self.theta = k_s, theta_link
        self.links_on, self.outcome_on, self.all_notes, self.mode, self.name = links, outcome, all_notes, mode, name
        self.reflection_mode = {"eem": "kt", "a2": "template", "a1": None}[mode]
        self.notes: dict[int, Note] = {}
        self.index: dict[tuple, int] = {}
        self.line_keys: dict[str, set] = {}          # canonical op -> undocumented keys ever introduced by a line
        self.docs_named: dict[tuple, str] = {}       # (agent, task, op) -> id named by get_api_docs(op)
        self._next = 1

    # ------------------------------------------------------------------------------------------------- note text
    def _embed_text(self, n: Note) -> str:
        if self.mode == "a1":
            return n.text_raw or ""
        return f"{n.reflection or ''} || {n.kt.render()}"

    def note_line(self, n: Note) -> str:
        """The note as shown in prompts: at most NOTE_CAP tokens including the "- " prefix (L7). k_t <= 30, reflection
        <= 60, then the outcome line in the remaining room (<= 30), shortened at item boundaries (D106) with the
        "(possibly outdated)" mark always kept."""
        t = self.tokenizer
        cap = NOTE_CAP - t.count("-")
        if self.mode == "a1":
            return "- " + t.truncate(f"{n.api_name}: {n.text_raw}", cap)
        head = t.truncate(n.kt.render(), KT_CAP).rstrip(". ")
        parts = [head]
        if n.reflection:
            refl = t.truncate(n.reflection, REFL_CAP).rstrip(". ")
            parts.append(refl if self.mode == "a2" else "Fix: " + refl)
        body = ". ".join(parts)
        if n.items:
            room = min(LINE_CAP, cap - t.count(body + ". "))
            mark = " (possibly outdated)" if n.possibly_outdated else ""
            rendered = [render_item(it) for it in n.items]
            for keep in range(len(rendered), 0, -1):
                more = " ..." if keep < len(rendered) else ""
                cand = "Last working change: " + "; ".join(rendered[:keep]) + more + mark
                if t.count(cand) <= room:
                    body = body + ". " + cand
                    break
        return "- " + body

    # ---------------------------------------------------------------------------------------------------- store
    def _relink(self, n: Note) -> None:
        for other_id in list(n.links):
            self.notes[other_id].links.discard(n.id)
        n.links = set()
        if not self.links_on:
            return
        for o in self.notes.values():
            if o.id != n.id and cosine(n.z, o.z) > self.theta:
                n.links.add(o.id)
                o.links.add(n.id)

    def _event(self, kind: str, n: Note, **extra) -> None:
        self.events.append({"event": kind, **n.snapshot(), **extra})

    def on_error(self, ctx, reflection):
        key = (ctx.api_name, ctx.kt.error_type)
        if self.mode == "a1":
            reflection = None
        if self.mode == "a2":
            reflection = A2_TEMPLATE.format(api=ctx.api_name)
        nid = self.index.get(key)
        if nid is not None:
            n = self.notes[nid]
            n.kt, n.reflection, n.trigger_args, n.owner = ctx.kt, reflection, copy.deepcopy(ctx.args), (ctx.agent,
                                                                                                       ctx.task)
            n.text_raw = ctx.raw_text if self.mode == "a1" else None
            n.items, n.open, n.possibly_outdated, n.truth = [], True, False, ctx.truth
            kind = "merge"
        else:
            n = Note(self._next, ctx.api_name, ctx.canonical_op, ctx.kt.error_type, ctx.kt, reflection,
                     copy.deepcopy(ctx.args), (ctx.agent, ctx.task),
                     text_raw=ctx.raw_text if self.mode == "a1" else None, truth=ctx.truth)
            self._next += 1
            self.notes[n.id] = n
            self.index[key] = n.id
            kind = "create"
        n.z = self.embedder.embed(self._embed_text(n))
        self._relink(n)
        if self.outcome_on:                           # every other note on this operation: possibly outdated
            for o in self.notes.values():
                if o.id != n.id and o.canonical_op == ctx.canonical_op and o.items and not o.possibly_outdated:
                    o.possibly_outdated = True
                    self._event("mark", o, at=[ctx.agent, ctx.task, ctx.step], t=round(ctx.t, 6))
        self._event(kind, n, at=[ctx.agent, ctx.task, ctx.step], t=round(ctx.t, 6))
        return None if self.mode == "a1" else self.note_line(n)[2:]

    def on_success(self, ctx):
        if ctx.called_id == "get_api_docs":
            target = ctx.args.get("operation") if isinstance(ctx.args, dict) else None
            if isinstance(ctx.body, dict) and target and ctx.body.get("operation") not in (None, target):
                self.docs_named[(ctx.agent, ctx.task, target)] = ctx.body["operation"]
            return
        if not self.outcome_on:
            return
        op = ctx.canonical_op
        for n in sorted(self.notes.values(), key=lambda n: n.id):      # re-check marked lines (any agent)
            if n.canonical_op == op and n.possibly_outdated:
                n.items = [it for it in n.items if item_holds(it, ctx.args, ctx.called_id)]
                n.possibly_outdated = False
                self._event("recheck", n, at=[ctx.agent, ctx.task, ctx.step], t=round(ctx.t, 6),
                            success_args=copy.deepcopy(ctx.args), called_id=ctx.called_id)
        for n in sorted(self.notes.values(), key=lambda n: n.id):      # close own open notes on this operation
            if not n.open or n.owner != (ctx.agent, ctx.task) or n.canonical_op != op:
                continue
            documented = OPS[op].schema
            if n.error_type in MOVED_TYPES:
                if ctx.called_id == n.api_name:
                    n.items, n.open = [], False                      # not a move after all: no line
                    self._event("close", n, at=[ctx.agent, ctx.task, ctx.step], t=round(ctx.t, 6),
                                success_args=copy.deepcopy(ctx.args), moved_to=None, known_keys=[])
                    continue
                named = moved_pointer(n.kt) or self.docs_named.get((ctx.agent, ctx.task, n.api_name))
                if ctx.called_id != named:
                    continue
                moved_to = ctx.called_id
            elif ctx.called_id != n.api_name:
                continue
            else:
                moved_to = None
            known = sorted(self.line_keys.get(op, set()))
            n.items = last_working_items(n.trigger_args, ctx.args, documented, moved_to=moved_to,
                                         error_keys=n.kt.error_keys, known_keys=frozenset(known))
            for it in n.items:
                if it["kind"] == "rename":
                    self.line_keys.setdefault(op, set()).add(it["new"])
                elif it["kind"] == "added":
                    self.line_keys.setdefault(op, set()).add(it["key"])
            n.open = False
            self._event("close", n, at=[ctx.agent, ctx.task, ctx.step], t=round(ctx.t, 6),
                        success_args=copy.deepcopy(ctx.args), moved_to=moved_to, known_keys=known)   # M5

    # ------------------------------------------------------------------------------------------------ retrieval
    def retrieve(self, query):
        notes = sorted(self.notes.values(), key=lambda n: n.id)
        if self.all_notes:
            return notes
        q = self.embedder.embed(query)
        sim = {n.id: cosine(q, n.z) for n in notes}
        ranked = sorted(notes, key=lambda n: (-sim[n.id], n.id))
        if not self.links_on:
            return ranked[:self.k]
        chosen = ranked[:self.k_s]
        ids = {n.id for n in chosen}
        neigh = sorted({nb for n in chosen for nb in n.links} - ids, key=lambda i: (-sim[i], i))
        for i in neigh:
            if len(chosen) >= self.k:
                break
            chosen.append(self.notes[i])
            ids.add(i)
        for n in ranked:                              # fill to k (D98)
            if len(chosen) >= self.k:
                break
            if n.id not in ids:
                chosen.append(n)
                ids.add(n.id)
        return chosen[:self.k]

    def render(self, selected):
        return [self.note_line(n) for n in selected]

    def selected_ids(self, selected):
        return [n.id for n in selected]

    def notes_on(self, op):
        return [n.id for n in self.notes.values() if n.canonical_op == op]

    def visible(self):
        return [[n.id, n.canonical_op] for n in sorted(self.notes.values(), key=lambda n: n.id)]

    def final_state(self):
        return [self.notes[i].snapshot() for i in sorted(self.notes)]


CONDITIONS = ("C0", "C1", "C2", "C3", "F", "AN", "R", "A1", "A2", "C2-OU")


def make_memory(condition: str, embedder, tokenizer, k=3, k_s=2, theta_link=0.65) -> Memory:
    if condition in ("C0", "R"):
        return Memory(embedder, tokenizer, k)
    if condition == "C1":
        return ReflectionList(embedder, tokenizer, k)
    if condition == "F":
        return FifoReflections(embedder, tokenizer, k)
    common = dict(k=k, k_s=k_s, theta_link=theta_link)
    if condition == "C2":
        return EEMStore(embedder, tokenizer, name="C2", **common)
    if condition == "C3":
        return EEMStore(embedder, tokenizer, links=False, name="C3", **common)
    if condition == "AN":
        return EEMStore(embedder, tokenizer, all_notes=True, name="AN", **common)
    if condition == "A1":
        return EEMStore(embedder, tokenizer, mode="a1", outcome=False, name="A1", **common)
    if condition == "A2":
        return EEMStore(embedder, tokenizer, mode="a2", name="A2", **common)
    if condition == "C2-OU":
        return EEMStore(embedder, tokenizer, outcome=False, name="C2-OU", **common)
    raise ValueError(condition)

```

### src/eem_acc/stage3/agent/loop.py

```python
"""Agent loop (Algorithm 1) and RQ1 units (S3.2a spec sections 2-5, 9)."""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from ..apisuite import ApiSuite, World, check, min_calls
from ..apisuite.docs import documented_all, render
from ..apisuite.oracle import MAX_STEPS_EXTRA, reference_plan
from ..apisuite.ops import OP_NAMES
from ..apisuite.templates import TEMPLATES
from ..feasibility import parse_action
from . import prompts as P
from .backend import GRAMMARS, N_CTX, Gen, t_llm
from .memory import ErrorCtx, Memory, SuccessCtx, make_memory
from .parser import PARSER_VERSION, parse_error
from ..apisuite.drift import effective, get_pair
from .scrub import task_scrubber

READS = {"find_customer", "get_cart", "get_order", "search_products", "check_stock", "track_shipment",
         "list_tickets"}


@dataclass
class Outcome:
    status: int
    body: Any
    transient: bool
    truth: dict
    canonical_op: str | None


class DirectDispatcher:
    """RQ1: the call goes straight to the suite (no overload; the design's generous fixed window)."""

    def __init__(self, suite: ApiSuite):
        self.suite = suite

    async def call(self, op_id: str, args: Any, agent: int, k: int, task_index: int) -> Outcome:
        r = self.suite.execute(op_id, args, task_index, call_key=(agent, k))
        return Outcome(r.status, r.body, False, r.truth.as_json(), r.op)


@dataclass
class LoopConfig:
    condition: str = "C0"
    docs_wording: str = "W1"
    action_max_tokens: int = 320
    grammar_always: bool = False
    reflection_max_tokens: int = 60
    k: int = 3
    k_s: int = 2
    theta_link: float = 0.65


def is_success(status: int, body: Any) -> bool:
    return status == 200 and not (isinstance(body, dict) and body.get("status") == "error")


class Agent:
    def __init__(self, agent_id: int, backend, tokenizer, memory: Memory, dispatcher, cfg: LoopConfig,
                 docs_for_task):
        self.id, self.backend, self.tok, self.memory, self.disp, self.cfg = (agent_id, backend, tokenizer, memory,
                                                                             dispatcher, cfg)
        self.docs_for_task = docs_for_task       # task_index -> docs text for the system prompt
        self.k = 0                               # running call counter across the stream (D68 guard)

    async def _llm(self, messages, kind, grammar, max_tokens, rec):
        assert grammar in GRAMMARS, grammar
        g: Gen = self.backend.generate(messages, kind, grammar, max_tokens)
        dt = t_llm(g)
        rec["llm"].append({"kind": kind, "n_in": g.n_in, "n_out": g.n_out, "grammar": g.grammar,
                           "finish_reason": g.finish_reason, "t_llm": round(dt, 6)})
        await asyncio.sleep(dt)
        return g

    async def run_task(self, inst: dict) -> dict:
        cfg = self.cfg
        ti = inst["index"]
        max_steps = min_calls(inst) + MAX_STEPS_EXTRA
        system = P.system_prompt(self.docs_for_task(ti), cfg.docs_wording)
        traj: list[tuple[str, str]] = []
        scrub = task_scrubber(inst)
        steps_log: list[dict] = []
        last_action, last_keys, last_obs = None, [], ""
        reason, finished = "max_steps", False
        n_api = n_llm = tok_in = tok_out = 0
        t_total = 0.0
        echo_values: set = set()
        for step in range(max_steps):
            rec: dict[str, Any] = {"agent": self.id, "task": ti, "step": step, "llm": [],
                                   "t_start": round(asyncio.get_running_loop().time(), 6)}
            query = P.state_text(inst["instruction"], last_action, last_keys, last_obs) + "\n" + P.tools_desc()
            rec["visible"] = self.memory.visible()            # retrieval-recall denominator at retrieval time (L5)
            selected = self.memory.retrieve(query)
            lines = self.memory.render(selected)
            rec["retrieved"] = self.memory.selected_ids(selected)
            rec["memory_sha"] = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16] if lines else None
            messages = P.build_messages(system, inst["instruction"], lines, traj)
            if self.tok.count_messages(messages) + cfg.action_max_tokens > N_CTX:
                reason = "context_overflow"
                rec["event"] = "context_overflow"             # logged, not a step (L3)
                steps_log.append(rec)
                break
            g = await self._llm(messages, "action", "envelope" if cfg.grammar_always else None,
                                cfg.action_max_tokens, rec)
            text, fenced = P.strip_fence(g.text)
            parsed = parse_action(text)
            rec["fence"] = fenced
            rec["fallback"] = False
            if parsed is None and not cfg.grammar_always:
                rec["fallback"] = True
                rec["first_reply"] = g.text[:400]              # the unparsed first reply is kept in the log (L2)
                g = await self._llm(messages, "action", "envelope_cap300", cfg.action_max_tokens, rec)
                text, f2 = P.strip_fence(g.text)
                rec["fallback_fence"] = f2
                parsed = parse_action(text)
            if parsed is None:
                rec["format_error"] = True
                obs = P.FORMAT_ERROR
                traj.append((g.text[:400], obs))
                last_action, last_keys, last_obs = None, [], obs
                steps_log.append(rec)
                continue
            rec["thought_len"] = len(parsed["thought"])
            action, args = parsed["action"], parsed["args"]
            rec["action"], rec["arg_keys"] = action, sorted(map(str, args))
            if action not in ("finish", "get_api_docs") and \
                    any(isinstance(v, str) and PLACEHOLDER_RE.search(v) for v in _leaves(args)):
                rec["placeholder_args"] = True                 # D104
            if action == "finish":
                finished = True
                rec["finish"] = True
                steps_log.append(rec)
                traj.append((text, ""))
                break
            self.k += 1
            n_api += 1
            out = await self.disp.call(action, args, self.id, self.k, ti)
            rec.update({"k": self.k, "status": out.status, "truth": out.truth, "transient": out.transient,
                        "canonical_op": out.canonical_op,
                        "notes_on_op": self.memory.notes_on(out.canonical_op or action)})   # retrieval recall
            if action != "get_api_docs":                       # operation names are never scrub values (H1)
                scrub.add_tree(args)
            if out.transient:
                obs = P.TRANSIENT
            elif is_success(out.status, out.body):
                obs, trunc = P.success_observation(action, out.body)
                rec["truncated"] = trunc
                if action != "get_api_docs":
                    scrub.add_tree(out.body)
                    echo_values |= _enum_like_values(out.body)
                self.memory.on_success(SuccessCtx(self.id, ti, step, action, out.canonical_op or action, args,
                                                  out.body, asyncio.get_running_loop().time()))
            else:
                obs = P.error_observation(out.status, out.body)
                api_obs = obs
                if action != "get_api_docs":
                    shown = await self._on_error(inst, step, action, args, out, scrub, rec)
                    if shown:
                        obs += "\n" + shown
                    if _echo_trap(out.truth, args, echo_values):
                        rec["echo_trap"] = True
                traj.append((text, obs))
                last_action, last_keys, last_obs = action, sorted(map(str, args)), api_obs   # D105: no note
                steps_log.append(rec)
                continue
            traj.append((text, obs))
            last_action, last_keys, last_obs = action, sorted(map(str, args)), obs
            steps_log.append(rec)
        for rec in steps_log:
            for c in rec["llm"]:
                n_llm += 1
                tok_in += c["n_in"]
                tok_out += c["n_out"]
                t_total += c["t_llm"]
        return {"traj": traj, "steps": steps_log, "finished": finished, "reason": reason, "n_api": n_api,
                "n_llm": n_llm, "tok_in": tok_in, "tok_out": tok_out, "t_llm": round(t_total, 6),
                "max_steps": max_steps}

    async def _on_error(self, inst, step, action, args, out, scrub, rec):
        kt = parse_error(action, out.status, out.body, args)
        raw = (f"{out.status} " + P.body_text(out.body))[:P.ERROR_CAP]
        mode = self.memory.reflection_mode
        reflection = None
        if mode in ("kt", "raw"):
            err = kt.render() if mode == "kt" else raw
            msgs = P.reflection_messages(inst["instruction"], action, args, err)
            if self.tok.count_messages(msgs) + self.cfg.reflection_max_tokens > N_CTX:
                rec["reflection_skipped"] = True
            else:
                g = await self._llm(msgs, "reflection", None, self.cfg.reflection_max_tokens, rec)
                reflection, n = scrub.scrub(g.text.strip())
                rec["scrubbed"] = n
        raw_s, n2 = scrub.scrub(raw)
        if n2 and mode is None:                            # A1 stores the raw text
            rec["scrubbed"] = rec.get("scrubbed", 0) + n2
        ctx = ErrorCtx(self.id, inst["index"], step, action, out.canonical_op or action, copy.deepcopy(args),
                       out.status, out.body, raw_s, kt, out.truth, asyncio.get_running_loop().time())
        shown = self.memory.on_error(ctx, reflection)
        rec["kt"] = {"error_type": kt.error_type, "expected": kt.expected, "received": kt.received}
        if shown is None:
            return None
        return shown if shown.startswith("Reflection: ") else "Note: " + shown


PLACEHOLDER_RE = re.compile(r"<[a-z_]+>")


def _leaves(x: Any):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


def _echo_trap(truth: dict, args: Any, echo_values: set) -> bool:
    """An error from an enum drift (visible or silent) whose sent value of that field is a documented value echoed in
    an earlier success body of the task (M3)."""
    if not isinstance(args, dict):
        return False
    for pid in truth.get("pair_ids", []):
        e = effective(get_pair(pid))
        if e.type == "enum":
            field = e.path.split("[].")[-1]
            if isinstance(args.get(field), str) and args[field] in echo_values:
                return True
    return False


def _enum_like_values(body: Any) -> set:
    out: set = set()
    if isinstance(body, dict):
        for k, v in body.items():
            if isinstance(v, str) and k in ("warehouse", "status", "state", "shipping_method", "category", "method"):
                out.add(v)
            elif isinstance(v, (dict, list)):
                out |= _enum_like_values(v)
    elif isinstance(body, list):
        for v in body:
            out |= _enum_like_values(v)
    return out


def task_summary(inst: dict, res: dict, world: World, suite: ApiSuite) -> dict:
    ok, goals = check(inst, world)
    success = res["finished"] and ok
    reason = "success" if success else ("wrong_state" if res["finished"] else res["reason"])
    drift_errors = sum(1 for s in res["steps"] if s.get("truth", {}).get("kind") in ("drift", "silent_drift"))
    agent_errors = sum(1 for s in res["steps"] if s.get("truth", {}).get("kind") == "agent")
    active = {p.op for p in suite.active_pairs(inst["index"])}
    planned = sorted(TEMPLATES[inst["template"]].covers & active)
    first_call = {}
    for s in res["steps"]:
        op = s.get("canonical_op")
        if op in active and op not in first_call and s.get("action") != "get_api_docs":
            first_call[op] = "ok" if s.get("truth", {}).get("kind") == "ok" else s.get("truth", {}).get("kind")
    plan_reads = [op for op, _ in reference_plan(inst) if op in READS]
    called = [s.get("canonical_op") for s in res["steps"]]
    skipped = sum(max(0, plan_reads.count(op) - called.count(op)) for op in set(plan_reads))
    return {"task": inst["index"], "template": inst["template"], "success": success, "reason": reason,
            "state_ok_at_end": ok, "failed_subgoals": [n for n, g in goals if not g],
            "steps": sum(1 for s in res["steps"] if s.get("event") != "context_overflow"),
            "max_steps": res["max_steps"], "api_calls": res["n_api"],
            "llm_calls": res["n_llm"], "tok_in": res["tok_in"], "tok_out": res["tok_out"], "t_llm": res["t_llm"],
            "drift_errors": drift_errors, "agent_errors": agent_errors,
            "format_errors": sum(1 for s in res["steps"] if s.get("format_error")),
            "fallbacks": sum(1 for s in res["steps"] if s.get("fallback")),
            "exposures_planned": planned, "first_call": first_call, "skipped_reads": skipped,
            "echo_traps": sum(1 for s in res["steps"] if s.get("echo_trap")),
            "scrubbed": sum(s.get("scrubbed", 0) for s in res["steps"]),
            "placeholder_args": sum(1 for s in res["steps"] if s.get("placeholder_args"))}


def run_rq1_unit(spec: dict, condition: str, backend, embedder, tokenizer, cfg: LoopConfig | None = None,
                 n_tasks: int | None = None) -> dict:
    """One result unit: one condition on one stream (store reset at the start)."""
    from ...env.vtime import run_virtual
    cfg = dataclasses.replace(cfg or LoopConfig(), condition=condition)       # never mutate the caller's config
    world = World.build(spec)
    suite = ApiSuite(world, spec)
    embedder = UnitEmbedder(embedder)                                       # per-unit cache, saved (M9)
    memory = make_memory(condition, embedder, tokenizer, cfg.k, cfg.k_s, cfg.theta_link)
    documented = documented_all()

    def docs_for_task(ti: int) -> str:
        if condition == "R":
            v = suite.views(ti)
            return "\n\n".join(render(v[op]) for op in OP_NAMES)
        return documented

    agent = Agent(0, backend, tokenizer, memory, DirectDispatcher(suite), cfg, docs_for_task)
    tasks = spec["tasks"][:n_tasks] if n_tasks else spec["tasks"]

    async def main():
        out = []
        if hasattr(backend, "register_tasks"):
            backend.register_tasks(tasks)
        for inst in tasks:
            res = await agent.run_task(inst)
            out.append((inst, res))
        return out

    results = run_virtual(main())
    summaries = [task_summary(inst, res, world, suite) for inst, res in results]
    return {"stamp": {"suite_version": spec["version"], "spec_sha256": spec["sha256"], "condition": condition,
                      "params": {"k": cfg.k, "k_s": cfg.k_s, "theta_link": cfg.theta_link,
                                 "docs_wording": cfg.docs_wording, "action_max_tokens": cfg.action_max_tokens,
                                 "grammar_always": cfg.grammar_always},
                      "parser_version": PARSER_VERSION, "prompt_version": P.PROMPT_VERSION,
                      "prompt_sha256": P.prompt_hash(),
                      "backend": getattr(backend, "id", type(backend).__name__), "tokenizer": tokenizer.id,
                      "embedder": embedder.id, "code_commit": code_commit()},
            "tasks": summaries, "steps": [s for _, r in results for s in r["steps"]],
            "trajectories": [r["traj"] for _, r in results],
            "note_events": memory.events, "final_notes": memory.final_state(),
            "embedding_cache": embedder.export()}


class UnitEmbedder:
    """Per-unit view of a (possibly shared) embedder: records every vector the unit used, keyed by text sha256, so the
    unit can be replayed with its own vectors (M9; S3.3 computes cosines from these)."""

    def __init__(self, base):
        self.base, self.id, self.used = base, base.id, {}

    def embed(self, text: str) -> tuple:
        v = self.base.embed(text)
        self.used[hashlib.sha256(text.encode()).hexdigest()] = v
        return v

    def export(self) -> list:
        return [[k, list(v)] for k, v in sorted(self.used.items())]


def code_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=__file__.rsplit("/src/", 1)[0]).stdout.strip() or "unknown"
    except Exception:          # noqa: BLE001 - outside a checkout
        return "unknown"

```

### src/eem_acc/stage3/agent/parser.py

```python
"""Deterministic k_t parser (S3.2a spec 6.1, D99). Written against the S3.1 error formats (stated as a limitation:
the parser and the formats have the same author)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..apisuite.leakcheck import jtype
from ..apisuite.ops import OPS
from ..apisuite.schema import schema_paths
from ..apisuite.drift import ALL_PAIRS

PARSER_VERSION = "kt-v2"
CODE_TYPES = {
    "missing": "missing_field", "extra_forbidden": "unknown_field",
    "string_type": "type_error", "int_type": "type_error", "float_type": "type_error", "bool_type": "type_error",
    "list_type": "type_error", "dict_type": "type_error", "model_type": "type_error",
    "enum": "enum_error", "literal_error": "literal_error", "string_pattern_mismatch": "format_error",
    "string_too_short": "value_error", "too_short": "value_error", "finite_number": "value_error",
}
SILENT_RE = re.compile(r"^Validation failed: field '([^']+)': (.*)$")
MOVED_RE = re.compile(r"Use '([^']+)'")
TYPE_WORD = {"string_type": "string", "int_type": "integer", "float_type": "number", "bool_type": "boolean",
             "list_type": "list", "dict_type": "object", "model_type": "object"}


def _schema_enum_values() -> dict[tuple, set]:
    """(operation, field name) -> every documented or current (remapped) enum value of that field (D99)."""
    out: dict[tuple, set] = {}
    for d in OPS.values():
        for p, f in schema_paths(d.schema).items():
            if f.kind == "enum":
                out.setdefault((d.name, p.split("[].")[-1]), set()).update(f.values)
    for pr in ALL_PAIRS.values():
        if pr.type == "enum":
            out.setdefault((pr.op, pr.path.split("[].")[-1]), set()).update(m[1] for m in pr.mapping)
        if pr.type == "new_required" and pr.new_field.kind == "enum":
            out.setdefault((pr.op, pr.new_field.name), set()).update(pr.new_field.values)
    return out


MOVED_TO_OP = {pr.new_op: pr.op for pr in ALL_PAIRS.values() if pr.type == "moved"}


def canonical_name(api_name: str) -> str:
    return MOVED_TO_OP.get(api_name, api_name)


ENUM_BY_OP = _schema_enum_values()
ENUM_VALUES: dict[str, set] = {}
for (_op, _f), _v in ENUM_BY_OP.items():
    ENUM_VALUES.setdefault(_f, set()).update(_v)


@dataclass(frozen=True)
class KT:
    api_name: str
    expected: str
    received: str
    error_type: str
    error_keys: frozenset = frozenset()

    def render(self) -> str:
        return f"{self.api_name} ({self.error_type}): expected {self.expected}; sent {self.received}"


def _compact(code: str, field_name: str, msg: str, ctx: dict | None) -> str:
    if code == "missing":
        return f"{field_name}: missing"
    if code == "extra_forbidden":
        return f"{field_name}: not allowed"
    if code in TYPE_WORD:
        return f"{field_name}: {TYPE_WORD[code]}"
    if code == "enum":
        vals = re.findall(r"'([^']*)'", (ctx or {}).get("expected", msg))
        return f"{field_name}: one of {'|'.join(vals)}"
    if code == "literal_error":
        exp = (ctx or {}).get("expected") or msg.replace("Input should be ", "")
        return f"{field_name}: {exp.replace(chr(39), chr(34))}"
    if code == "string_pattern_mismatch":
        return f"{field_name}: pattern"
    return f"{field_name}: invalid"


def _code_from_msg(msg: str) -> str:
    """Silent informative messages carry the pydantic message, not the code."""
    table = [("Field required", "missing"), ("Extra inputs are not permitted", "extra_forbidden"),
             ("Input should be a valid string", "string_type"), ("Input should be a valid integer", "int_type"),
             ("Input should be a valid number", "float_type"), ("Input should be a valid boolean", "bool_type"),
             ("Input should be a valid list", "list_type"), ("String should match pattern", "string_pattern_mismatch"),
             ("Input should be a finite number", "finite_number")]
    for prefix, code in table:
        if msg.startswith(prefix):
            return code
    if msg.startswith("Input should be "):
        rest = msg[len("Input should be "):]
        return "enum" if " or " in rest and rest.count("'") >= 4 else "literal_error"
    return "value_error"


def _path(loc: list) -> str:
    """("body", "items", 0, "quantity") -> "items[].quantity" (M2)."""
    parts = [str(x) for x in loc[1:] if not isinstance(x, int)]
    return "[].".join(parts) if parts else str(loc[-1])


def received_text(args: Any, enum_fields: set | None = None, op: str | None = None) -> str:
    """Sent keys with JSON types; values only for enum errors and only if they are schema enum values (D99)."""
    if not isinstance(args, dict):
        return jtype(args)
    parts = []
    for k, v in args.items():
        if isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            sub = {}
            for it in v:
                for ik, iv in it.items():
                    sub.setdefault(ik, jtype(iv))
            parts += [f"{k}[].{ik}: {t}" for ik, t in sub.items()]
            continue
        t = jtype(v)
        allowed = ENUM_BY_OP.get((op, k), set()) if op else ENUM_VALUES.get(k, set())
        if enum_fields and k in enum_fields and isinstance(v, str) and v in allowed:
            parts.append(f"{k}: \"{v}\"")
        else:
            parts.append(f"{k}: {t}")
    return ", ".join(parts) if parts else "no arguments"


def parse_error(api_name: str, status: int, body: Any, args: Any) -> KT:
    if status == 200 and isinstance(body, dict) and body.get("status") == "error":
        m = SILENT_RE.match(str(body.get("message", "")))
        if m:
            fld, msg = m.group(1), m.group(2)
            code = _code_from_msg(msg)
            return KT(api_name, _compact(code, fld, msg, None),
                      received_text(args, {fld} if code == "enum" else None, canonical_name(api_name)),
                      "silent_error", frozenset({fld}))
        return KT(api_name, "unknown", received_text(args), "silent_error")
    if status == 422 and isinstance(body, dict) and isinstance(body.get("detail"), list):
        items = body["detail"]
        exp = "; ".join(_compact(i.get("type", ""), _path(i["loc"]), i.get("msg", ""), i.get("ctx"))
                        for i in items)
        keys = frozenset(_path(i["loc"]) for i in items)        # paths, like leakcheck (L5)
        enum_fields = {str(i["loc"][-1]) for i in items if i.get("type") == "enum"}
        et = CODE_TYPES.get(items[0].get("type", ""), "value_error") if items else "value_error"
        return KT(api_name, exp, received_text(args, enum_fields, canonical_name(api_name)), et, keys)
    if status == 400:
        return KT(api_name, "unknown", received_text(args), "bad_request")
    if status == 410 and isinstance(body, dict) and isinstance(body.get("detail"), str):
        m = MOVED_RE.search(body["detail"])
        return KT(api_name, f"use '{m.group(1)}'" if m else "unknown", f"operation '{api_name}'",
                  "operation_removed")
    if status == 404 and isinstance(body, str):
        return KT(api_name, "unknown", f"operation '{api_name}'", "route_not_found")
    if status == 404 and isinstance(body, dict):
        res = body.get("resource") or (body.get("detail", "").replace(" not found", "").lower() or "resource")
        return KT(api_name, f"{res} not found", received_text(args), "resource_not_found")
    if status in (409, 422) and isinstance(body, dict):
        if isinstance(body.get("detail"), str):
            return KT(api_name, body["detail"], received_text(args), "business_rule")
        return KT(api_name, "unknown", received_text(args), "business_rule")
    return KT(api_name, "unknown", "unknown", "unknown")


def moved_pointer(kt: KT) -> str | None:
    m = re.match(r"use '([^']+)'", kt.expected)
    return m.group(1) if kt.error_type == "operation_removed" and m else None

```

### src/eem_acc/stage3/agent/scrub.py

```python
"""Scrub task values from reflections and A1 raw text before they enter the shared store (S3.2a spec 6.2, D92)."""
from __future__ import annotations

import re
from typing import Any

from ..apisuite.drift import ALL_PAIRS
from ..apisuite.ops import OP_NAMES
from .parser import ENUM_VALUES

PLACEHOLDER_BY_KEY = {"customer_id": "customer_id", "email": "email", "name": "name", "sku": "sku",
                      "order_id": "order_id", "payment_id": "payment_id", "shipment_id": "shipment_id",
                      "ticket_id": "ticket_id", "reservation_id": "reservation_id", "subject": "text",
                      "text": "text", "query": "product", "product": "product", "price": "amount",
                      "total": "amount", "amount": "amount", "refunded_amount": "amount", "summary": "text"}
# schema vocabulary that is never scrubbed (enum, boolean and constant values)
SAFE = ({str(v) for vals in ENUM_VALUES.values() for v in vals} | {"IDR", "2", "30", "50", "true", "false", "True",
                                                                  "False"}
        | set(OP_NAMES) | {p.new_op for p in ALL_PAIRS.values() if p.new_op} | {"get_api_docs", "finish"}
        # response-only vocabulary of the suite (statuses, tiers, shipment states), never entities (re-review M4)
        | {"created", "paid", "shipped", "cancelled", "captured", "refunded", "in_transit", "delayed", "delivered",
           "basic", "gold", "open", "pending", "closed"})
SAFE_LOWER = {v.lower() for v in SAFE}


class Scrubber:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}             # value text -> placeholder name

    def add(self, value: Any, key: str = "value") -> None:
        if isinstance(value, bool) or value is None:
            return
        name = PLACEHOLDER_BY_KEY.get(key.split("[].")[-1], key.split("[].")[-1])
        if isinstance(value, (int, float)):
            if abs(value) < 100:                      # small counts (qty) are not entities
                return
            text = str(int(value)) if float(value).is_integer() else str(value)
            self.values.setdefault(text, name)
            if float(value).is_integer() and abs(value) >= 1000:      # D103: separator variants
                n = int(value)
                for sep in (",", ".", " "):
                    self.values.setdefault(f"{n:,}".replace(",", sep), name)
            return
        if not isinstance(value, str):
            return
        text = value.strip()
        if len(text) < 3 or text.lower() in SAFE_LOWER:           # case-insensitive, like the matching (H1)
            return
        self.values.setdefault(text, name)
        # D103: normalised forms
        m = re.fullmatch(r"[A-Z]+-(\d{4,})", text)
        if m:                                         # the number part of an id ("O-9761" -> "9761")
            self.values.setdefault(m.group(1), name)
        if re.fullmatch(r"\d{4,}", text):              # an amount sent as a string (e.g. after a type drift)
            self.add(int(text), key)
        if "@" in text and len(text.split("@")[0]) >= 3:          # the email's local part
            local = text.split("@")[0]
            self.values.setdefault(local, name)
            self.values.setdefault(local.replace(".", "_"), name)
        if key.split("[].")[-1] in ("name", "product"):          # name tokens of 4+ letters
            for tok in text.split():
                if len(tok) >= 4 and tok.lower() not in SAFE_LOWER:
                    self.values.setdefault(tok, name)
            if key.split("[].")[-1] == "name":
                self.values.setdefault(text.lower().replace(" ", "_"), name)
                self.values.setdefault(text.lower().replace(" ", "."), name)

    def add_tree(self, obj: Any, key: str = "value") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                self.add_tree(v, str(k))
        elif isinstance(obj, list):
            for v in obj:
                self.add_tree(v, key)
        else:
            self.add(obj, key)

    def scrub(self, text: str) -> tuple[str, int]:
        n = 0
        for val in sorted(self.values, key=lambda v: (-len(v), v)):
            if re.fullmatch(r"[\d.,]+", val):
                pat = re.compile(rf"(?<!\d){re.escape(val)}(?!\d)")         # digit boundaries (H2)
            else:
                pre = r"(?<!\w)" if val[0].isalnum() else ""
                post = r"(?!\w)" if val[-1].isalnum() else ""
                pat = re.compile(pre + re.escape(val) + post, re.IGNORECASE)  # word boundaries (L1)
            text, k = pat.subn(f"<{self.values[val]}>", text)
            n += k
        return text, n


def task_scrubber(inst: dict) -> Scrubber:
    """Seeded with the task's instruction values and entities (setup); the loop adds call arguments and
    observation bodies as they appear."""
    s = Scrubber()
    setup = dict(inst["setup"])
    for p in setup.pop("products", []):                     # product names get their own placeholder
        s.add(p["name"], "product")
        s.add_tree({k: v for k, v in p.items() if k != "name"})
    s.add_tree(setup)
    s.add_tree(inst["params"])
    return s

```

### src/eem_acc/stage3/agent/prompts.py

```python
"""Prompt texts and builders (S3.2a spec sections 3, 4, 6.2). Versioned: any change bumps PROMPT_VERSION."""
from __future__ import annotations

import json
import re
from typing import Any

PROMPT_VERSION = "p-v1"

DOCS_WORDING = {   # D94; the pilot keeps the variant with the highest C0 dev success
    "W1": "If a call fails because the API may have changed, call get_api_docs with the operation name to see its "
          "current schema.",
    "W2": "The documentation above may be out of date; get_api_docs(operation) returns the current schema of an "
          "operation.",
    "W3": "Before retrying a failed call, check get_api_docs(operation) for the current parameters.",
    "W4": "get_api_docs(operation) returns the current schema of an operation.",
}

RULES = """You are an agent that completes a task by calling the operations of a shop API, one call per reply.
Reply with exactly one JSON object and nothing else:
{"thought": "<one or two short sentences>", "action": "<operation>", "args": {<arguments>}}
Use the ids returned by earlier calls. After each call you receive the API response as an observation.
When the task is complete, reply with {"thought": "...", "action": "finish", "args": {"summary": "<what you did>"}}."""

EXTRA_TOOLS = """get_api_docs - Return the current documentation of an operation.
  operation         required  string
  returns: operation, docs

finish - End the task.
  summary           required  string"""

MEMORY_HEADER = "Notes from earlier API errors:"
REFLECT_INSTR = ("A call to an API operation failed. Write one imperative instruction of at most 40 words that tells "
                 "how to fix the call next time. Reply with the instruction only.")
A2_TEMPLATE = "Fix the arguments of {api} as stated above."
TRANSIENT = "Observation: the API is temporarily unavailable (the gateway retried and gave up). You may try again."
FORMAT_ERROR = "Observation: invalid action format"
SUCCESS_CAP = 1500
ERROR_CAP = 400
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


def prompt_hash() -> str:
    import hashlib
    blob = json.dumps([RULES, EXTRA_TOOLS, DOCS_WORDING, MEMORY_HEADER, REFLECT_INSTR, A2_TEMPLATE, TRANSIENT,
                       FORMAT_ERROR, SUCCESS_CAP, ERROR_CAP], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def system_prompt(docs_text: str, wording: str) -> str:
    return f"{RULES}\n\nOperations:\n\n{docs_text}\n\n{EXTRA_TOOLS}\n\n{DOCS_WORDING[wording]}"


def user_prompt(instruction: str, note_lines: list[str]) -> str:
    mem = (MEMORY_HEADER + "\n" + "\n".join(note_lines) + "\n\n") if note_lines else ""
    return f"{mem}Task: {instruction}"


def build_messages(system: str, instruction: str, note_lines: list[str],
                   trajectory: list[tuple[str, str]]) -> list[dict]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt(instruction, note_lines)}]
    for assistant, observation in trajectory:
        msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": observation})
    return msgs


def strip_fence(text: str) -> tuple[str, bool]:
    m = FENCE_RE.match(text)
    return (m.group(1).strip(), True) if m else (text.strip(), False)


def body_text(body: Any) -> str:
    return body if isinstance(body, str) else json.dumps(body)


def success_observation(op: str, body: Any) -> tuple[str, bool]:
    """Returns (text, truncated). get_api_docs bodies are shown verbatim (multi-line)."""
    if op == "get_api_docs" and isinstance(body, dict) and "docs" in body:
        return "Observation: 200\n" + body["docs"], False
    t = json.dumps(body, separators=(",", ":"))
    return "Observation: 200 " + t[:SUCCESS_CAP], len(t) > SUCCESS_CAP


def error_observation(status: int, body: Any) -> str:
    return f"Observation: {status} " + body_text(body)[:ERROR_CAP]


def reflection_messages(instruction: str, action: str, args: Any, error_text: str) -> list[dict]:
    """error_text = the uncapped k_t rendering (EEM arms) or the raw status and body (C1, F)."""
    content = (f"Task: {instruction}\nFailed call: {action} {json.dumps(args)}\nError: {error_text}\n\n"
               f"{REFLECT_INSTR}")
    return [{"role": "user", "content": content}]


def state_text(instruction: str, last_action: str | None, last_arg_keys: list[str], last_obs: str) -> str:
    """s_t for the retrieval query (D98)."""
    act = f"{last_action}({', '.join(last_arg_keys)})" if last_action else "none"
    return f"{instruction}\nLast action: {act}\nLast observation: {last_obs[:200]}"


def tools_desc() -> str:
    from ..apisuite.ops import OP_NAMES
    return " ".join(list(OP_NAMES) + ["get_api_docs", "finish"])

```

Other files (backend.py, scripted.py, the S3.1 suite incl. leakcheck.py, tests) are in https://github.com/notanut/eem-acc-skripsi (main); ask for any file you need.
