# Review request: S3.1 implementation (API suite with schema drift, task sampler, checker), 2026-11-21

You are a strict reviewer of LLM-agent benchmarks and experiment code. Be concrete; no praise.

## Context
Thesis: LLM agents (Qwen2.5-7B, local) solve multi-call tasks against a synthetic API whose schema drifts. The method
(EEM) stores parsed API errors plus an LLM reflection and a deterministic "Last working change" line, and retrieves
them in later tasks; baselines are ReAct (C0) and reflection RAG (C1). S3.1 is the CPU-only environment: the API
suite, drift injector, stream sampler and checker. The spec below (r3) was reviewed twice internally; the code was
reviewed once and the fixes (D76-D84) re-reviewed. 205 S3.1 tests pass (460 in the repo), including: an oracle and a
docs-only solver solve 200 sampled streams in the minimum number of calls; documented-schema calls fail exactly on the
drifted operations; 2,200 streams sample without failure and satisfy the exposure rule on 20/16/15-task prefixes; a
leak property test over all catalogue rows, legal pair combinations and adversarial triggers.

## Checklist
1. Validity for the C0 vs C1 vs C2 comparison: anything in the environment that favours or penalises a memory method
   artificially (error formats, docs, exposure rule, distractors, template mix, conditional templates, "exactly one"
   sub-goals, silent drift, idempotency)?
2. Realism and solvability: can a 7B agent infer every drift from what it sees (informative/terse errors, docs tool)?
   Any drift that is unfair or ambiguous?
3. Leak safety of the "Last working change" reference (leakcheck.py, D69/D76).
4. Statistical concerns: drift-set stratification, disjoint pools, the tilted template mix (0.4-1.9x uniform), and
   what Section IV must disclose.
5. Missing tests or edge cases.

## Output
Findings ranked by severity with concrete fixes; then a verdict on whether S3.2 (agent loop, EEM) can build on this.

## The spec (docs/s31_spec.md, r3)

# S3.1 specification: API suite, drift, task sampler, checker (CPU only)

_r3, 2026-11-21 (r2 + the code review and its re-review: D76-D84 and clear-cut fixes, section 12). Implements design r7 sections 3-5 (`docs/stage3_design.md`). r1 was reviewed by a fresh-context
subagent (5 High, 10 Medium, 7 Low); the user decided H1-H5, M2, M3 and Q1 (D67-D75, section 12); the rest was
clear-cut and is applied below._

## 1. Scope and interface

S3.1 delivers a pure-Python, deterministic API semantics layer with no LLM, GPU or network:

```python
spec    = sample_stream(seed, split, n_tasks=20, kind="rq1"|"sp"|"rq3")   # JSON-serialisable, hashed
world   = World.build(spec)                     # fresh per result unit (condition / cell), same spec -> same world
suite   = ApiSuite(world, spec)                 # drift schedule and error styles come from the spec
resp    = suite.execute(op_id, args, task_index, call_key)   # -> ApiResponse(status, body, truth)
text    = suite.docs(op_id, which="documented"|"current", task_index)
success, subgoals = check(spec.tasks[i], world)   # subgoals = [(name, ok), ...]
plan    = reference_plan(spec.tasks[i])          # minimal call sequence on a scratch world (tests, min_calls)
```

- `ApiResponse.status` is one of 200, 400, 404, 409, 410, 422; this set shares nothing with the gateway's
  `RETRYABLE` (tested). 408/429/5xx come only from the RQ2 `MockServer`/gateway, which call the suite only for
  admitted attempts (wired in S3.2/S3.3); the server's latency model applies to every admitted call unchanged.
- `ApiResponse.truth` is hidden from the agent: `ok`, `drift` (with the pair id), `silent_drift` (with the pair id),
  or `agent` with a reason (`unknown_operation`, `unknown_resource`, `business_rule`, `documented_schema`).
- **Idempotency (D68):** `call_key` = (agent id, logical call number) from the gateway's request id. The first
  execution's response is stored; any later attempt with the same key gets the stored response and changes nothing.
  So a call abandoned by a client timeout and retried executes exactly once, as with an Idempotency-Key header.
- **`get_api_docs` is served by the suite** (design section 4: one step, one API call through the gateway):
  `execute("get_api_docs", {"operation": X})` -> 200 with the current block of X; for a moved X the block is shown
  under the new id with the line "formerly X"; an unknown name -> resource-style 404 "Operation not found"
  (`agent/unknown_resource`). It never drifts and always uses the informative style.
- Determinism: seeded RNG substreams per component (drift, styles, templates, instances, names, distractors), so a
  pilot knob (`n_distractors`, template mix) does not reshuffle drift sets or task order. No wall clock, no global
  RNG, no reliance on set or hash order; same seed -> byte-identical JSON under different `PYTHONHASHSEED` (tested).
- Seed ranges are disjoint: dev 1000-1999, test 2000-2999, SP 3000-3099, RQ3 4000-4099.

**Module:** `src/eem_acc/stage3/apisuite/` with `schema.py` (field specs, strict validator, error items), `world.py`,
`ops.py` (16 operations: documented schema + handler + business rules), `drift.py` (catalogue, pools, schedules),
`errors.py` (two styles), `docs.py`, `templates.py` (instructions, instance sampling, reference plans, sub-goals),
`streams.py` (sampler), `suite.py`, `leakcheck.py` (the reference "Last working change" diff, shared with S3.2).
Tests in `tests/test_s31_*.py`. No new dependency (D72).

## 2. World

One world per result unit, built from the stream spec, so every condition of a stream starts from an identical world.
**Every task instance has its own entities (D67):** its customer, cart, orders, payments, shipments, tickets **and
products** (a brand+item combination no other task uses, D77; own stock per warehouse), plus `n_distractors`
look-alikes of its own (D78): products of the same brand+item in other sizes (hit by partial-name searches), tickets of
the same customer in the target's category but not open (in templates that pick a ticket by category and status), and
customers with the same name and another email. Distractors come from a separate RNG; their ids are even and primary
ids odd in the same range, and their emails use the same pattern on other domains (D82), so they look like real
entities to an agent while the knob never changes a task's primary entities, values or outcome. Nothing is shared between tasks, so a stray write in one task cannot change another (pairing holds;
concurrent RQ3 agents cannot interfere), and the checker inspects only the task's own entities.

| Entity | Fields (ids deterministic, e.g. `C-4821`) |
|---|---|
| customer | customer_id, email, name, tier (basic/gold) |
| product | sku, name (unique in the world), price (integer IDR), stock {JKT, SBY, BDG} |
| cart | customer_id -> items [{sku, qty}] |
| order | order_id, customer_id, items, shipping_method, total, status (created/paid/shipped/cancelled), payment_id, shipment_id |
| payment | payment_id, order_id, amount, method, refunded_amount, status (captured/refunded) |
| reservation | reservation_id, sku, warehouse, qty |
| shipment | shipment_id, order_id, warehouse, carrier, insured, status (created/in_transit/delayed/delivered) |
| ticket | ticket_id, customer_id, subject, category, priority, status (open/pending/closed), comments [{text, public}] |

Values are stored canonically: a type-drifted value (qty `"2"`, amount `"125000.00"`) is stored as its number; a
remapped enum value is stored by its meaning (`regular` as `standard`).

## 3. Operations (16) and documented ("old") schema

`req` = required, `opt` = optional; types: string, integer, number, boolean, enum(values), id (identifier), array.

| # | Operation | Arguments (documented) | Returns |
|---|---|---|---|
| 1 | find_customer | email: string req | customer_id, name, tier |
| 2 | get_cart | customer_id: id req | items [{sku, qty}] |
| 3 | create_order | customer_id: id req; items: array req of {sku: id req, qty: integer req}; shipping_method: enum(standard, express) req | order_id, total |
| 4 | get_order | order_id: id req | the order (customer_id, items, total, status, payment_id, shipment_id) |
| 5 | pay_order | order_id: id req; amount: number req; method: enum(card, transfer, wallet) req | payment_id |
| 6 | refund_payment | payment_id: id req; amount: number req; reason: enum(damaged, late, other) req | refund state |
| 7 | cancel_order | order_id: id req; reason: enum(customer_request, out_of_stock) req | order state |
| 8 | search_products | query: string req; limit: integer opt | [{sku, name, price}] |
| 9 | check_stock | sku: id req; warehouse: enum(JKT, SBY, BDG) req | available |
| 10 | reserve_stock | sku: id req; qty: integer req; warehouse: enum(JKT, SBY, BDG) req | reservation_id |
| 11 | create_shipment | order_id: id req; warehouse: enum(JKT, SBY, BDG) req; carrier: enum(jne, sicepat, pos) req; insured: boolean opt | shipment_id |
| 12 | track_shipment | shipment_id: id req | state, order_id |
| 13 | create_ticket | customer_id: id req; subject: string req; category: enum(billing, shipping, product) req; priority: enum(low, normal, high) opt (default normal) | ticket_id |
| 14 | update_ticket | ticket_id: id req; status: enum(open, pending, closed) req | ticket state |
| 15 | add_comment | ticket_id: id req; text: string req; public: boolean req | comment index |
| 16 | list_tickets | customer_id: id req; status: enum(open, pending, closed) opt | [{ticket_id, subject, category, status}] |

`search_products` returns every product in the world whose name contains the query (case-insensitive); names are
unique, so a full name finds one product. Success bodies never contain `"status": "error"` (tested; the agent loop
reads that as a silent error).

**Business rules** (`agent/business_rule`; messages carry no values, D73): pay amount != total -> 422; paying a
non-created order, shipping an unpaid order, a second shipment, cancelling a shipped or cancelled order, refunding
more than the remaining captured amount (a fully refunded payment has none) -> 409 or 422 as listed in `ops.py`;
reserving or shipping more than the warehouse has -> 409. Unknown id -> resource 404. Duplicate tickets and comments
are accepted by the API (as real ones would) and fail the checker's "exactly one" sub-goals.

## 4. Validation order

1. **Route.** Unknown operation id -> route 404 (`agent/unknown_operation`). A moved id -> 410 with pointer
   (informative) or plain route 404 (terse) (D74).
2. **Framework validation** against the current schema, strict (D72): unknown keys (`extra_forbidden`), missing
   required fields, type errors, enum and constant errors. Strict means: integer rejects booleans, floats and numeric
   strings; number accepts integers and floats but not booleans or strings; boolean rejects 0/1 and strings; string
   rejects numbers; non-finite numbers (NaN, infinity) are rejected. All items are collected (within an array of objects, only the first item per path and error type, a deviation from pydantic that keeps bodies short); **drift items come first**. Truth: `drift` if any item concerns a
   drifted aspect **with an error of the matching kind** (D80: rename -> missing new key / extra old key; type ->
   type error; enum -> enum error; new required -> missing or invalid new field), else `agent/documented_schema`. 422 (informative) or 400 (terse).
3. **App-level check of a silently drifted aspect.** Every matching-kind check concerning that aspect (including the extra-key
   check of a renamed field's old name, which the framework step then skips) runs here, only after step 2 passed. A
   failure returns 200 `{"status": "error", "message": ...}`, truth `silent_drift`, and changes no state.
4. **Business rules**, then execution: 200 with the result.

## 5. Error styles

Per stream: 4 of the 8 drifted and 4 of the 8 non-drifted operations are informative, the rest terse (D70, seeded).

| Case | Informative | Terse |
|---|---|---|
| validation | 422 `{"detail": [{"type": "missing", "loc": ["body", "currency"], "msg": "Field required"}, ...]}`; enum/literal items add `"ctx": {"expected": "'regular' or 'priority'"}`; no `input` echo (D72) | 400 `{"error": "Bad Request"}` |
| route 404 (unknown operation id; produced by the router, so the same in both styles) | 404 text `404 page not found` | 404 text `404 page not found` |
| moved | 410 `{"detail": "Operation 'create_ticket' was removed. Use 'open_ticket'."}` | 404 text `404 page not found` (D74) |
| resource 404 | 404 `{"detail": "Order not found"}` (no id) | 404 `{"error": "not_found", "resource": "order"}` |
| business 409/422 | `{"detail": "<one sentence, no values>"}` | `{"error": "Conflict"}` / `{"error": "Unprocessable Entity"}` |
| silent drift | 200 `{"status": "error", "message": "<template below>"}` | 200 `{"status": "error", "message": "Request failed"}` |

Silent message templates (informative): rename "Validation failed: field 'total' is required" (or "field 'amount' is
not allowed" when the new key was sent too, D83); type "Validation
failed: field 'amount' must be a string"; new required "Validation failed: field 'capture' is required (boolean)";
enum "Validation failed: value not allowed for 'method'; allowed: CARD, TRANSFER, WALLET". The worst-case
informative body stays under 400 characters with the drift items first (tested over every catalogue row and legal
combination), so the observation's truncation never hides a drift item.

## 6. Drift catalogue

One concrete change per legal (operation, type) pair. Enum changes remap **all** values of the field. New required
fields are enum, boolean or constant only; any listed enum value or boolean is accepted, a constant must match.
**Identifier fields are renamed only to `*_id` names (D69)**, so the outcome update's identifier rule covers the new
key too. Drifts on optional fields are excluded (a minimal call may not send them).

| Operation | rename | type | new required | enum (all values) | moved to |
|---|---|---|---|---|---|
| find_customer | email -> email_address | - | include_inactive: boolean | - | lookup_customer |
| get_cart | customer_id -> client_id | - | currency: const "IDR" | - | get_basket |
| create_order | items[].qty -> items[].quantity | items[].qty: integer -> string (digits) | currency: const "IDR" | shipping_method: standard/express -> regular/priority | place_order |
| get_order | order_id -> purchase_id | - | api_version: const "2" | - | fetch_order |
| pay_order | amount -> total | amount: number -> string (`^\d+(\.\d{1,2})?$`, compared numerically) | capture: boolean | method: card/transfer/wallet -> CARD/TRANSFER/WALLET | submit_payment |
| refund_payment | reason -> refund_reason | - | notify_customer: boolean | reason: damaged/late/other -> item_damaged/late_delivery/other_reason | create_refund |
| cancel_order | reason -> cancel_reason | - | confirm: const true | reason: customer_request/out_of_stock -> requested_by_customer/no_stock | void_order |
| search_products | query -> q | query: string -> array of strings (a product matches if its name contains every element) | - | - | find_products |
| check_stock | warehouse -> location | - | - | warehouse: JKT/SBY/BDG -> jakarta/surabaya/bandung | get_inventory |
| reserve_stock | qty -> quantity | qty: integer -> string (digits) | hold_minutes: const 30 | warehouse: JKT/SBY/BDG -> jakarta/surabaya/bandung | hold_stock |
| create_shipment | carrier -> courier | - | signature_required: boolean | carrier: jne/sicepat/pos -> JNE_REG/SICEPAT_REG/POS_KILAT | ship_order |
| track_shipment | shipment_id -> tracking_id | - | include_history: boolean | - | get_shipment_status |
| create_ticket | subject -> title | - | channel: enum(email, chat, phone) | category: billing/shipping/product -> payments/delivery/item | open_ticket |
| update_ticket | status -> state | - | - | status: open/pending/closed -> OPEN/ON_HOLD/RESOLVED | set_ticket_status |
| add_comment | text -> body | public: boolean -> integer (0 or 1) | - | - | post_comment |
| list_tickets | customer_id -> owner_id | - | page_size: const 50 | - | search_tickets |

Counts (asserted from the data in tests): 16 rename, 5 type, 12 new required, 9 enum, 16 moved = 58 pairs, plus
16 silent = 74.

- **Silent (D71):** each operation has one silent pair whose underlying change is one of its own type 1-4 changes,
  fixed once (seeded), and it is placed in that change's pool. A stream never takes a silent pair together with its
  underlying change.
- **Pools:** split once, seeded, into dev and test, stratified by type, with every operation in both pools; committed
  as `docs/evidence/s31_drift_pools.json` and frozen with the suite.
- **Stream drift set (D70):** 10 pairs on 8 operations (2 with two pairs) from the split's pool; each of the 6 types
  at least once where the pool allows, the rest uniform; drawn by seeded constructive search, never redrawn because a
  later step failed. Two pairs on one operation must not touch the same field, must not be rename + new required
  (D69: the outcome update could not tell them apart), and silent cannot join its underlying change. A moved
  operation with a second pair applies it at the new id. A test checks that every pool leaves enough two-pair
  combinations.
- **Staleness probe (D69):** the re-renamed field is enum-valued, so recording its value never records an entity.
  SP renames (start -> midpoint), all different from the catalogue renames so SP never reuses a dev change: check_stock
  warehouse -> depot -> site; create_shipment carrier -> shipping_partner -> shipper; refund_payment reason ->
  refund_category -> refund_cause; update_ticket status -> ticket_status -> ticket_state; cancel_order reason ->
  cancellation_reason -> cancel_cause. An SP stream takes 9 pairs from the test pool plus one forced
  SP rename on an operation not otherwise drifted. Midpoint = floor(n/2) of the length actually run; that operation
  gets at least 1 exposure before the midpoint and at least 2 after it, at lengths 20, 16 and 15.
- **Schedule:** `[(task_index, pairs)]`; RQ1 and RQ3 have one entry at index 0, SP two.
- **Intended inconsistencies (stated in Section IV):** drift is per operation, so e.g. check_stock may use `jakarta`
  while reserve_stock still uses `JKT`; success bodies echo the documented enum values (check_stock returns
  `"warehouse": "JKT"`, update_ticket and list_tickets return `open`/`pending`/`closed`, get_order returns
  `standard`/`express`) also after a remap (D80b).

## 7. Task templates (31)

Each template has an instruction pattern, an entity sampler, a per-instance reference plan, sub-goals (Section III-A:
success = all sub-goals hold) and `min_calls` = length of the instance's reference plan; MaxSteps = min_calls + 1 + 6.
Instructions give only what a user would know (emails, product names, order or shipment ids), never customer or
ticket ids; warehouses are named with city and code ("the Jakarta warehouse (JKT)"). **Conditions are randomised per
instance (D75)**; a template *covers* an operation only if every outcome's plan contains it.

| ID | Task | Reference plan | n |
|---|---|---|---|
| T01 | Order everything in the cart of <email> with <method> shipping | find_customer, get_cart, create_order | 3 |
| T02 | T01 and pay in full by <pay method> | + pay_order | 4 |
| T03 | T02 and ship from <W> with <carrier> | + create_shipment | 5 |
| T04 | Reserve <q> x <product> in <W> for <email>, then order them with <method> shipping | find_customer, search_products, reserve_stock, create_order | 4 |
| T05 | Order <q> x <product> for <email> with <method> shipping and pay by <pay method> | find_customer, search_products, create_order, pay_order | 4 |
| T06 | Reserve <q> x <product> in whichever of <W1>/<W2> has more stock | search_products, check_stock x2, reserve_stock | 4 |
| T07 | Reserve <q> x <product> in <W1> if it has at least <q>, otherwise in <W2> (random) | search_products, check_stock(W1), reserve_stock | 3 |
| T08 | Pay order <order_id> in full by <pay method> | get_order, pay_order | 2 |
| T09 | Ship paid order <order_id> from <W> with <carrier>, insured if its total exceeds Rp 1,000,000 (random) | get_order, create_shipment | 2 |
| T10 | Pay order <order_id> by <pay method> and ship from <W> with <carrier> | get_order, pay_order, create_shipment | 3 |
| T11 | Refund order <order_id> in full (<reason>) and cancel it | get_order, refund_payment, cancel_order | 3 |
| T12 | Refund order <order_id> in full (<reason>) and open a billing ticket "<subject>" for its customer | get_order, refund_payment, create_ticket | 3 |
| T13 | Cancel unpaid order <order_id> (<reason>) and open a <category> ticket "<subject>" for its customer | get_order, cancel_order, create_ticket | 3 |
| T14 | Open a <category> ticket "<subject>" with <priority> priority for <email> | find_customer, create_ticket | 2 |
| T15 | T14 and add a <public/internal> comment "<text>" | + add_comment | 3 |
| T16 | Close all open tickets of <email> (2 open; others stay as they are) | find_customer, list_tickets, update_ticket x2 | 4 |
| T17 | Put the open billing ticket of <email> on pending and add an internal comment "<text>" | find_customer, list_tickets, update_ticket, add_comment | 4 |
| T18 | Shipment <shipment_id> is delayed: open a high-priority shipping ticket "<subject>" for its customer | track_shipment, get_order, create_ticket | 3 |
| T19 | T18 and add a public comment "<text>" | + add_comment | 4 |
| T20 | T02 and open a low-priority shipping ticket "<subject>" | find_customer, get_cart, create_order, pay_order, create_ticket | 5 |
| T21 | T03 and open a shipping ticket "<subject>" | T03 + create_ticket | 6 |
| T22 | T05 and ship from <W> with <carrier> | T05 + create_shipment | 5 |
| T23 | Refund order <order_id> in full (<reason>) and add a public comment "<text>" to its customer's open billing ticket | get_order, refund_payment, list_tickets, add_comment | 4 |
| T24 | Cancel unpaid order <order_id> (<reason>), re-order the same items with <method> shipping, pay by <pay method> | get_order, cancel_order, create_order, pay_order | 4 |
| T25 | If shipment <shipment_id> is delivered, close its customer's open shipping ticket; otherwise add a public comment "<text>" to it (random) | track_shipment, get_order, list_tickets, update_ticket or add_comment | 4 |
| T26 | If <W> has at least <q> x <product>, reserve them there; otherwise open a product ticket "<subject>" for <email> (random) | search_products, check_stock, then reserve_stock or (find_customer, create_ticket) | 3/4 |
| T27 | Shipment <shipment_id> is delayed: refund its order in full (late) and add a public comment "<text>" to the customer's open shipping ticket | track_shipment, get_order, refund_payment, list_tickets, add_comment | 5 |
| T28 | <email> asks to cancel unpaid order <order_id> (customer_request) and to close their open billing ticket | find_customer, cancel_order, list_tickets, update_ticket | 4 |
| T29 | Reserve <q> x <product> in whichever of the three warehouses has the most stock | search_products, check_stock x3, reserve_stock | 5 |
| T30 | Refund order <order_id> in full (<reason>), cancel it and close its customer's open billing ticket | get_order, refund_payment, cancel_order, list_tickets, update_ticket | 5 |
| T31 | Ship paid order <order_id> from <W1> if it has the order's item in stock, otherwise from <W2>, with <carrier> (random) | get_order, check_stock(W1), create_shipment | 3 |

Covering templates per operation are counted from the data in tests (target: at least 4 for every operation;
the drafted table gives check_stock 5, reserve_stock 4, track_shipment 4, update_ticket 4 as the smallest; T25 and
T26 branches are not counted).

**Guessable reads:** T06, T29 and the conditional templates can sometimes be solved by guessing without the read
(e.g. 50% in T06). An exposure is counted from the reference plan; Section IV reports exposures *experienced* (the
operation actually called) next to exposures *planned*.

**Sub-goals** (full table generated from `templates.py` into `docs/evidence/s31_subgoals.md` and reviewed):
- created objects: **exactly one new** object of the kind among the task's entities (T24: exactly one new order
  besides the cancelled one), with the asked fields (items, method, warehouse, carrier, category, priority with
  "normal" when unspecified, public flag);
- payments and refunds: amount equal to the full total (numerically), method or reason as asked (by meaning);
- state changes: the asked status, and **negative sub-goals** where a template names exceptions (T16: the pending and
  closed tickets stay unchanged);
- free text (subject, comment text) must equal the given string after whitespace normalisation;
- never checked: values of new required fields (currency, channel, capture, ...).

## 8. Stream sampler (D70)

`sample_stream(seed, split, n_tasks, kind)`:
1. Drift set from the split's pool (section 6); SP adds its forced rename.
2. Error styles: 4/4 among drifted and 4/4 among non-drifted operations.
3. **Constructive placement:** for each drifted operation in a seeded order, count the covering tasks already placed;
   place what is missing so that the first exposure is at positions 0-6 and at least 3 exposures fall within the
   first 15 positions (so the 15- and 16-task prefixes keep the rule; positions 0-6 are the first 35% of 20, about
   the first half of 15). A placement picks, among covering templates, preferably one that also covers other
   still-uncovered drifted operations (seeded, weighted by the template mix), at a seeded free position. Remaining
   positions are filled from the template mix. If a placement fails, it is retried with the next RNG substream (at
   most 100 times, then an error); the drift set is never redrawn.
4. Instance values and entities per task (own substream).
5. RQ3: 20 tasks per seed, dealt round-robin to the 4 LLM agents (5 each); the rule applies to the pooled 20.

The realised template mix is reported per split, and a test bounds its deviation from the configured mix over 1,000
seeds (the exposure rule necessarily tilts it towards covering templates; Section IV says so). Stream specs are JSON
(seed, split, kind, drift schedule, styles, templates, instance values) with a sha256 that is stamped into every
result.

## 9. Docs rendering

Plain text per operation, marking required/optional, type, enum values, identifiers and constants:

```
create_order - Create an order for a customer.
  customer_id      required  identifier
  items            required  array of objects:
    sku            required  identifier
    qty            required  integer
  shipping_method  required  enum: standard | express
  returns: order_id, total
```

Constants render as `constant: 30`, arrays of strings as `array of strings`, the 0/1 integer as `integer (0 or 1)`.
The system prompt gets all 16 documented blocks; `get_api_docs` returns current blocks (section 1).

## 10. Leak check (shared with S3.2)

`leakcheck.last_working_change(trigger_args, success_args, documented_schema, moved_to)` is the reference
implementation of the design's diff rule (D76): the **content** is the diff between the note's trigger arguments and
the successful call (an identical call, or one differing only in values, gives no line), while the **classification**
of appeared keys (rename vs new required field) uses the documented schema: if a documented required key is absent
from the success, appeared keys are renames and carry no value; the line also names (never values) undocumented keys
the trigger sent and the success dropped, e.g. a stale staleness-probe name (D81). Only documented *required* keys
count as vanished, and an enum value is recorded only when the trigger used a documented value. D69's value rule applies: a value equal to any value in
the trigger arguments is never recorded, and identifier-like keys never carry values. Safety relies on the success
having passed the strict validator. S3.2's outcome update must match it on all test cases. Property test over every
catalogue row, every legal two-pair combination, the SP renames and adversarial triggers (documented call, keys
dropped, undocumented look-alike keys, partially translated, empty): no entity value (ids, emails, names, free text,
amounts) ever appears in a recorded line; plus business-rule, resource-404 and identical-call notes give no line.

## 11. Tests (pytest, CPU)

- Catalogue and template counts asserted from the data; every operation covered by at least 4 templates.
- **Oracle solvability:** for 50 streams per split and kind, the oracle (current schema) solves every task with
  exactly `min_calls` calls and every sub-goal holds; a **docs-only oracle** (reads only the rendered current docs
  and informative errors) also solves them.
- **Necessity:** every write in the plan is needed (removing it fails a sub-goal); every read returns a value used as
  an argument or a branch condition by a later call (data-flow check).
- **Drift observable:** a naive solver on the documented schema fails its first call to each drifted operation with
  the right truth and pair id; non-drifted operations succeed.
- **Errors:** route vs resource 404 differ in both styles; informative 422 names field and type code, drift items
  first, no `input`, under 400 characters (worst case over all rows and combinations); terse bodies carry no detail;
  moved: 410 + pointer / plain 404; silent: 200 error payload, precedence after framework validation, no state change.
- **Validation:** strictness table (section 4); unknown keys rejected; all items collected; documented-schema errors
  on non-drifted fields tagged `agent`.
- **Business rules and duplicates:** every rule in `ops.py`, including double refund, double cancel, second shipment.
- **Checker:** negatives per sub-goal kind (wrong amount, method, warehouse, duplicate order, missing comment, T16's
  untouched tickets); enum meaning and canonical values.
- **Isolation:** permuting task order, injecting stray writes, and interleaving tasks of different RQ3 agents leave
  every other task's checker result unchanged.
- **Idempotency:** the same `call_key` executes once; the retry gets the stored response.
- **Sampler:** never errors on 1,000 seeds per split and kind; exposure rule on the full stream and the 16- and
  15-task prefixes; drift-type and style stratification; pools disjoint, stratified, every operation in both;
  two-pair constraints and feasibility; SP midpoint and exposures at 20/16/15; RQ3 dealing; template-mix tolerance.
- **Determinism:** identical JSON and hash for the same seed, also under a different `PYTHONHASHSEED` (subprocess).
- **Leak check:** section 10.
- **Docs:** every operation renders; current docs reflect each drift and the SP re-rename; documented docs never
  change; success bodies never contain `"status": "error"`.
- **Status set:** disjoint from `status.RETRYABLE`.

## 12. Decisions (user, 2026-11-21)

Realised numbers (`docs/evidence/s31_template_mix.json`): template shares 0.4-1.9 x uniform (the exposure rule
favours templates covering several operations), mean min_calls about 4.0 per task.


| ID | Decision |
|---|---|
| D67 | Every task has its own entities, including products and stock (fresh per-task products with unique names and per-task distractors); one world per result unit |
| D68 | Idempotency by (agent, logical call): a retried call gets the stored response of its first execution |
| D69 | Leaks closed in both places: identifier renames only to `*_id` names, no rename + new required on one operation, SP re-renames only enum fields; plus the S3.2 rule that no value present in the trigger arguments is recorded; property-tested |
| D70 | Constructive sampler with more templates (every operation in at least 4); drift types and error styles stratified per stream; first exposure in positions 0-6, at least 3 exposures in the first 15 |
| D71 | One silent pair per operation with a fixed underlying change, in that change's pool |
| D75 | Conditional templates randomised per instance; min_calls per instance; necessity test (T07 = 3 calls) |
| D76 | Outcome line: content = diff of trigger vs success (design rule), classification of appeared keys by the documented schema (code review H1) |
| D77 | Every product has a brand+item no other task uses (code review M2) |
| D78 | The distractor knob scales look-alike products and same-category non-open tickets; separate RNG, id range and email pattern (M4, M3) |
| D79 | Instructions state "normal priority" wherever the priority is checked (T12, T13, T21, T26) |
| D80 | Drift is tagged only for error items of the matching kind; SP stage-1 names differ from catalogue renames; (b) enum echoes in success bodies listed as intended |
| D81 | The outcome line names dropped undocumented keys (names only), so the staleness probe's stale key is named |
| D82 | Distractors look like real entities: interleaved ids (odd/even, same range), same email pattern on other domains |
| D83 | Silent rename with both keys sent: the message names the old key as not allowed |
| D84 | A wrongly formatted string on a type-drifted field (e.g. "1.234" for a decimal string) counts as drift |
| D72 | Hand-written strict validator with pydantic v2 message and type codes, no `input` echo, drift items first, under 400 characters |
| D73 | Resource 404 and business messages carry no ids or values |
| D74 | A moved operation's error names the new id only in the informative style (410); terse gives a plain 404 |


## Key code

### src/eem_acc/stage3/apisuite/leakcheck.py

```python
"""Reference "Last working change" diff (design r7 section 6; S3.1 spec section 10; D69, D76). Shared with S3.2.

Given a note's trigger arguments (the failed call), the successful call's arguments and the documented schema, it
returns the deterministic outcome line. The **content** is the diff between the trigger and the success (D76: an
identical call, or one that differs only in values, gives no line); the **classification** of appeared keys uses the
documented schema, which keeps the leak protection:
* arguments are compared by JSON path; arrays of objects by item key path and type, never by item values;
* appeared = keys in the success that are neither in the trigger nor documented; vanished = documented keys of the
  trigger that are absent from the success (required keys only), plus the names of undocumented keys the trigger sent
  and the success dropped (D81; names only);
* if any documented *required* key is absent from the success, the operation has a rename (rename and new required
  field never share an operation, D69): appeared keys are recorded as renames, with their JSON type and no value, and
  the vanished keys are named;
* otherwise appeared keys are new required fields: the value is recorded only if it is a scalar, the key is not
  identifier-like (`*_id`, `id`), and the value does not occur anywhere in the trigger arguments (D69);
* a documented key present in both whose JSON type changed (and no longer matches the documented type) is recorded
  without a value;
* a documented enum key present in both, whose trigger value is a documented value and whose success value is outside
  the documented values, is recorded with that value (the remap), unless the value occurs in the trigger;
* a moved operation records the new operation id (the caller passes `moved_to` only for notes on that route).

Safety relies on the success arguments having passed the suite's strict validator: a new required field's value is
then an enum value, a boolean or a constant by construction (spec section 6).
"""
from __future__ import annotations

import json
import re
from typing import Any

from .schema import Field, schema_paths

JSON_KIND = {"string": "string", "id": "string", "integer": "integer", "number": "number", "boolean": "boolean",
             "enum": "string", "const": None, "digits": "string", "decimal": "string", "int01": "integer",
             "array_obj": "array", "array_str": "array"}


_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def jtype(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


def _flatten(args: dict) -> dict[str, list]:
    """path -> list of values (one per array item for item paths)."""
    out: dict[str, list] = {}
    for k, v in args.items():
        out.setdefault(k, []).append(v)
        if isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            for it in v:
                for ik, iv in it.items():
                    out.setdefault(f"{k}[].{ik}", []).append(iv)
    return out


def _scalars(x: Any) -> set[str]:
    if isinstance(x, dict):
        return set().union(*(_scalars(v) for v in x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*(_scalars(v) for v in x)) if x else set()
    return {json.dumps(x), str(x)}


def _idlike(path: str) -> bool:
    last = path.split("[].")[-1].lower()
    return last == "id" or last.endswith("_id")


def _type_ok(f: Field, v: Any) -> bool:
    want = JSON_KIND[f.kind]
    got = jtype(v)
    return want is None or got == want or (want == "number" and got == "integer")


def last_working_change(trigger_args: dict, success_args: dict, documented: tuple,
                        moved_to: str | None = None) -> str | None:
    doc = schema_paths(documented)
    succ = _flatten(success_args)
    trig = _flatten(trigger_args)
    trig_vals = _scalars(trigger_args)
    parts: list[str] = []
    if moved_to:
        parts.append(f"use operation '{moved_to}'")
    appeared = [p for p in succ if p not in doc and p not in trig]
    # vanished: documented required keys of the trigger, plus key *names* (never values) of undocumented keys the
    # trigger sent and the success dropped (D81: e.g. a stale staleness-probe name)
    vanished = [p for p in trig if p not in succ and
                ((p in doc and doc[p].required) or (p not in doc and _KEY_RE.match(p.split("[].")[-1])))]
    renamed_op = any(f.required and p not in succ and ("[]." not in p or p.split("[].")[0] in succ)
                     for p, f in doc.items())
    if renamed_op:
        if appeared:
            if len(vanished) == 1 and len(appeared) == 1:
                parts.append(f"'{vanished[0]}' is now '{appeared[0]}' ({jtype(succ[appeared[0]][0])})")
            else:
                parts += [f"'{m}' removed" for m in vanished]
                parts += [f"'{a}' added ({jtype(succ[a][0])})" for a in appeared]
    else:
        for a in appeared:
            v = succ[a][0]
            rec = f"'{a}' added ({jtype(v)})"
            scalar = not isinstance(v, (list, dict))
            if scalar and not _idlike(a) and not (_scalars(v) & trig_vals) and len(succ[a]) == 1:
                rec += f" = {json.dumps(v)}"
            parts.append(rec)
    for p, vals in succ.items():
        if p not in doc or p not in trig:
            continue
        f = doc[p]
        tvals = trig[p]
        if not all(_type_ok(f, v) for v in vals):
            if {jtype(v) for v in vals} != {jtype(v) for v in tvals}:
                parts.append(f"'{p}' is now {jtype(vals[0])}")
            continue
        if (f.kind == "enum" and len(vals) == 1 and vals[0] not in f.values and len(tvals) == 1
                and tvals[0] in f.values and not (_scalars(vals[0]) & trig_vals)):
            parts.append(f"'{p}' = {json.dumps(vals[0])}")
    if not parts:
        return None
    return "Last working change: " + "; ".join(parts)

```

### src/eem_acc/stage3/apisuite/suite.py

```python
"""The API suite: executes one call against the world under the stream's drift schedule (S3.1 spec sections 1, 4).

Validation order: route -> framework validation (strict; drift items first) -> app-level check of silently drifted
aspects (HTTP 200 + error payload, no state change) -> business rules -> execution. Idempotent by call_key (D68).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from . import errors
from .docs import render
from .drift import OpView, canonicalize, effective, get_pair, op_view
from .ops import OPS, OP_NAMES
from .schema import Field, schema_paths, validate_object
from .world import World

SEMANTIC_STATUSES = frozenset({200, 400, 404, 409, 410, 422})
DOCS_OP = "get_api_docs"
DOCS_SCHEMA = (Field("operation", "string"),)


@dataclass(frozen=True)
class Truth:
    kind: str                              # ok | drift | silent_drift | agent
    pair_ids: tuple = ()                   # drift pairs involved (drift / silent_drift)
    reason: str | None = None              # agent: unknown_operation | unknown_resource | business_rule | documented_schema

    def as_json(self) -> dict:
        d = {"kind": self.kind}
        if self.pair_ids:
            d["pair_ids"] = list(self.pair_ids)
        if self.reason:
            d["reason"] = self.reason
        return d


@dataclass(frozen=True)
class ApiResponse:
    status: int
    body: Any
    truth: Truth
    op: str | None = None                  # the operation the call resolved to (None: unknown route)


class ApiSuite:
    def __init__(self, world: World, spec: dict):
        self.world = world
        self.spec = spec
        self.styles: dict[str, str] = dict(spec["styles"])
        self.schedule: list[tuple[int, list[str]]] = [(int(i), list(ids)) for i, ids in spec["drift_schedule"]]
        self.schedule.sort(key=lambda e: e[0])
        self._views: dict[int, dict[str, OpView]] = {}
        self._done: dict[Any, ApiResponse] = {}
        self.executions = 0                # calls that reached the semantics (not idempotent replays)

    # ------------------------------------------------------------------------------------------------------- views
    def active_pairs(self, task_index: int) -> list:
        ids: list[str] = []
        for start, pids in self.schedule:
            if start <= task_index:
                ids = pids
        return [get_pair(i) for i in ids]

    def views(self, task_index: int) -> dict[str, OpView]:
        key = max((s for s, _ in self.schedule if s <= task_index), default=-1)
        if key not in self._views:
            pairs = self.active_pairs(task_index)
            self._views[key] = {op: op_view(op, pairs) for op in OP_NAMES}
        return self._views[key]

    def docs(self, op: str, which: str = "current", task_index: int = 0) -> str:
        if which == "documented":
            return render(op_view(op, []))
        return render(self.views(task_index)[op])

    # ----------------------------------------------------------------------------------------------------- execute
    def execute(self, op_id: str, args: Any, task_index: int, call_key: Any = None) -> ApiResponse:
        if call_key is not None and call_key in self._done:
            return copy.deepcopy(self._done[call_key])
        self.executions += 1
        resp = self._execute(op_id, args, task_index)
        if call_key is not None:
            self._done[call_key] = copy.deepcopy(resp)
        return resp

    def _execute(self, op_id: str, args: Any, task_index: int) -> ApiResponse:
        views = self.views(task_index)
        if op_id == DOCS_OP:
            return self._docs_call(args, views)
        by_route = {v.route: op for op, v in views.items()}
        if op_id in by_route:
            op = by_route[op_id]
        elif op_id in views and views[op_id].route != op_id:          # moved operation called by its old id
            v = views[op_id]
            pid = next(p.id for p in v.pairs if effective(p).type == "moved")
            st, body = errors.moved(self.styles[op_id], op_id, v.route)
            return ApiResponse(st, body, Truth("drift", (pid,)), op_id)
        else:
            return ApiResponse(404, errors.ROUTE_404_TERSE, Truth("agent", reason="unknown_operation"), None)
        v = views[op]
        style = self.styles[op]
        items = validate_object(v.schema, args)
        framework = [i for i in items if not v.is_silent(i.path, i.type)]
        silent = [i for i in items if v.is_silent(i.path, i.type)]
        if framework:
            drift_items = [i for i in framework if v.is_drift(i.path, i.type)]
            other = [i for i in framework if not v.is_drift(i.path, i.type)]
            st, body = errors.validation(style, drift_items + other)
            if drift_items:
                pids = tuple(dict.fromkeys(v.drift_paths[i.path] for i in drift_items))
                return ApiResponse(st, body, Truth("drift", pids), op)
            return ApiResponse(st, body, Truth("agent", reason="documented_schema"), op)
        if silent:
            pid = v.drift_paths[silent[0].path]
            return ApiResponse(200, errors.silent_body(self._silent_message(v, pid, style, args)),
                               Truth("silent_drift", (pid,)), op)
        res = OPS[op].handler(self.world, canonicalize(v, copy.deepcopy(args)))
        if res.kind == "ok":
            return ApiResponse(200, res.body, Truth("ok"), op)
        if res.kind == "not_found":
            st, body = errors.resource_404(style, res.reason)
            return ApiResponse(st, body, Truth("agent", reason="unknown_resource"), op)
        st, body = errors.business(style, res.status, res.reason)
        return ApiResponse(st, body, Truth("agent", reason="business_rule"), op)

    def _silent_message(self, v: OpView, pid: str, style: str, args: Any = None) -> str:
        e = effective(get_pair(pid))
        if e.type == "rename":
            cur = v.renames[e.path]
            new_key = cur.split("[].")[-1]
            if "[]." not in cur and isinstance(args, dict) and new_key in args:      # D83: old key sent too
                return errors.silent_message(style, "rename_extra", e.path)
            return errors.silent_message(style, "rename", new_key)
        if e.type == "type":
            return errors.silent_message(style, "type", e.path.split("[].")[-1], new_kind=e.new_kind)
        if e.type == "new_required":
            f = e.new_field
            desc = {"boolean": "boolean", "const": f"constant {f.const!r}".replace("'", '"'),
                    "enum": "one of " + ", ".join(f.values)}[f.kind]
            if f.kind == "const" and isinstance(f.const, bool):
                desc = "constant true" if f.const else "constant false"
            return errors.silent_message(style, "new_required", f.name, field_desc=desc)
        if e.type == "enum":
            allowed = schema_paths(v.schema)[e.path].values
            return errors.silent_message(style, "enum", e.path.split("[].")[-1], allowed=allowed)
        raise AssertionError(e.type)

    def _docs_call(self, args: Any, views: dict[str, OpView]) -> ApiResponse:
        items = validate_object(DOCS_SCHEMA, args)
        if items:
            st, body = errors.validation(errors.INFORMATIVE, items)
            return ApiResponse(st, body, Truth("agent", reason="documented_schema"), DOCS_OP)
        name = args["operation"]
        by_route = {v.route: op for op, v in views.items()}
        op = by_route.get(name) or (name if name in views else None)
        if op is None:
            st, body = errors.resource_404(errors.INFORMATIVE, "operation")
            return ApiResponse(st, body, Truth("agent", reason="unknown_resource"), DOCS_OP)
        return ApiResponse(200, {"operation": views[op].route, "docs": render(views[op])}, Truth("ok"), DOCS_OP)

```

### src/eem_acc/stage3/apisuite/drift.py

```python
"""Drift catalogue, current schemas, canonicalisation, pools and stream drift sets (S3.1 spec section 6).

A *pair* is one concrete change of one operation. Types: rename, type, new_required, enum, moved, silent (a type 1-4
change of the same operation delivered as HTTP 200 with an error payload, D71), plus the staleness-probe renames
sp1/sp2 (not in the pools). Paths: "amount" or "items[].qty".
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .ops import OPS, OP_NAMES
from .schema import Field

TYPES = ("rename", "type", "new_required", "enum", "moved", "silent")
TYPE_ERRORS = frozenset({"string_type", "int_type", "float_type", "bool_type", "list_type", "string_pattern_mismatch",
                         "literal_error", "too_short", "finite_number", "string_too_short"})
BASE_TYPES = ("rename", "type", "new_required", "enum")       # usable as silent underlying changes
POOL_SEED = 31_2026_11_21                                     # fixed: the pools are part of the frozen suite


@dataclass(frozen=True)
class Pair:
    id: str
    op: str
    type: str
    path: str | None = None            # rename/type/enum: documented path; new_required: new field name
    new_name: str | None = None        # rename / sp: new key name (last path segment)
    new_kind: str | None = None        # type
    mapping: tuple = ()                # enum: ((old, new), ...)
    new_field: Field | None = None     # new_required
    new_op: str | None = None          # moved
    underlying: str | None = None      # silent: id of the base pair it reuses
    prev_name: str | None = None       # sp2: the stage-1 name (also attributed to drift)

    def as_json(self) -> dict:
        d = {"id": self.id, "op": self.op, "type": self.type}
        for k in ("path", "new_name", "new_kind", "new_op", "underlying", "prev_name"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.mapping:
            d["mapping"] = [list(m) for m in self.mapping]
        if self.new_field is not None:
            f = self.new_field
            d["new_field"] = {"name": f.name, "kind": f.kind, "values": list(f.values), "const": f.const}
        return d


def _p(op, typ, **kw) -> Pair:
    return Pair(id=f"{op}:{typ}", op=op, type=typ, **kw)


def _const(name, value) -> Field:
    return Field(name, "const", const=value)


# One concrete change per legal (operation, type) pair (spec section 6 table).
_ROWS = [
    # find_customer
    _p("find_customer", "rename", path="email", new_name="email_address"),
    _p("find_customer", "new_required", path="include_inactive", new_field=Field("include_inactive", "boolean")),
    _p("find_customer", "moved", new_op="lookup_customer"),
    # get_cart
    _p("get_cart", "rename", path="customer_id", new_name="client_id"),
    _p("get_cart", "new_required", path="currency", new_field=_const("currency", "IDR")),
    _p("get_cart", "moved", new_op="get_basket"),
    # create_order
    _p("create_order", "rename", path="items[].qty", new_name="quantity"),
    _p("create_order", "type", path="items[].qty", new_kind="digits"),
    _p("create_order", "new_required", path="currency", new_field=_const("currency", "IDR")),
    _p("create_order", "enum", path="shipping_method", mapping=(("standard", "regular"), ("express", "priority"))),
    _p("create_order", "moved", new_op="place_order"),
    # get_order
    _p("get_order", "rename", path="order_id", new_name="purchase_id"),
    _p("get_order", "new_required", path="api_version", new_field=_const("api_version", "2")),
    _p("get_order", "moved", new_op="fetch_order"),
    # pay_order
    _p("pay_order", "rename", path="amount", new_name="total"),
    _p("pay_order", "type", path="amount", new_kind="decimal"),
    _p("pay_order", "new_required", path="capture", new_field=Field("capture", "boolean")),
    _p("pay_order", "enum", path="method",
       mapping=(("card", "CARD"), ("transfer", "TRANSFER"), ("wallet", "WALLET"))),
    _p("pay_order", "moved", new_op="submit_payment"),
    # refund_payment
    _p("refund_payment", "rename", path="reason", new_name="refund_reason"),
    _p("refund_payment", "new_required", path="notify_customer", new_field=Field("notify_customer", "boolean")),
    _p("refund_payment", "enum", path="reason",
       mapping=(("damaged", "item_damaged"), ("late", "late_delivery"), ("other", "other_reason"))),
    _p("refund_payment", "moved", new_op="create_refund"),
    # cancel_order
    _p("cancel_order", "rename", path="reason", new_name="cancel_reason"),
    _p("cancel_order", "new_required", path="confirm", new_field=_const("confirm", True)),
    _p("cancel_order", "enum", path="reason",
       mapping=(("customer_request", "requested_by_customer"), ("out_of_stock", "no_stock"))),
    _p("cancel_order", "moved", new_op="void_order"),
    # search_products
    _p("search_products", "rename", path="query", new_name="q"),
    _p("search_products", "type", path="query", new_kind="array_str"),
    _p("search_products", "moved", new_op="find_products"),
    # check_stock
    _p("check_stock", "rename", path="warehouse", new_name="location"),
    _p("check_stock", "enum", path="warehouse",
       mapping=(("JKT", "jakarta"), ("SBY", "surabaya"), ("BDG", "bandung"))),
    _p("check_stock", "moved", new_op="get_inventory"),
    # reserve_stock
    _p("reserve_stock", "rename", path="qty", new_name="quantity"),
    _p("reserve_stock", "type", path="qty", new_kind="digits"),
    _p("reserve_stock", "new_required", path="hold_minutes", new_field=_const("hold_minutes", 30)),
    _p("reserve_stock", "enum", path="warehouse",
       mapping=(("JKT", "jakarta"), ("SBY", "surabaya"), ("BDG", "bandung"))),
    _p("reserve_stock", "moved", new_op="hold_stock"),
    # create_shipment
    _p("create_shipment", "rename", path="carrier", new_name="courier"),
    _p("create_shipment", "new_required", path="signature_required",
       new_field=Field("signature_required", "boolean")),
    _p("create_shipment", "enum", path="carrier",
       mapping=(("jne", "JNE_REG"), ("sicepat", "SICEPAT_REG"), ("pos", "POS_KILAT"))),
    _p("create_shipment", "moved", new_op="ship_order"),
    # track_shipment
    _p("track_shipment", "rename", path="shipment_id", new_name="tracking_id"),
    _p("track_shipment", "new_required", path="include_history", new_field=Field("include_history", "boolean")),
    _p("track_shipment", "moved", new_op="get_shipment_status"),
    # create_ticket
    _p("create_ticket", "rename", path="subject", new_name="title"),
    _p("create_ticket", "new_required", path="channel",
       new_field=Field("channel", "enum", values=("email", "chat", "phone"))),
    _p("create_ticket", "enum", path="category",
       mapping=(("billing", "payments"), ("shipping", "delivery"), ("product", "item"))),
    _p("create_ticket", "moved", new_op="open_ticket"),
    # update_ticket
    _p("update_ticket", "rename", path="status", new_name="state"),
    _p("update_ticket", "enum", path="status",
       mapping=(("open", "OPEN"), ("pending", "ON_HOLD"), ("closed", "RESOLVED"))),
    _p("update_ticket", "moved", new_op="set_ticket_status"),
    # add_comment
    _p("add_comment", "rename", path="text", new_name="body"),
    _p("add_comment", "type", path="public", new_kind="int01"),
    _p("add_comment", "moved", new_op="post_comment"),
    # list_tickets
    _p("list_tickets", "rename", path="customer_id", new_name="owner_id"),
    _p("list_tickets", "new_required", path="page_size", new_field=_const("page_size", 50)),
    _p("list_tickets", "moved", new_op="search_tickets"),
]

CATALOGUE: dict[str, Pair] = {p.id: p for p in _ROWS}

# Staleness-probe renames (enum-valued fields only, D69): op -> (field, stage-1 name, stage-2 name)
# Stage-1 names differ from every catalogue rename (L3 of the code review), so SP test streams never reuse a dev change.
SP_RENAMES = {
    "check_stock": ("warehouse", "depot", "site"),
    "create_shipment": ("carrier", "shipping_partner", "shipper"),
    "refund_payment": ("reason", "refund_category", "refund_cause"),
    "update_ticket": ("status", "ticket_status", "ticket_state"),
    "cancel_order": ("reason", "cancellation_reason", "cancel_cause"),
}


def sp_pairs(op: str) -> tuple[Pair, Pair]:
    f, n1, n2 = SP_RENAMES[op]
    return (Pair(f"{op}:sp1", op, "rename", path=f, new_name=n1),
            Pair(f"{op}:sp2", op, "rename", path=f, new_name=n2, prev_name=n1))


def _silent_pairs() -> dict[str, Pair]:
    rng = random.Random(POOL_SEED + 1)
    out = {}
    for op in OP_NAMES:
        base = [p for p in _ROWS if p.op == op and p.type in BASE_TYPES]
        u = rng.choice(base)
        out[f"{op}:silent"] = Pair(f"{op}:silent", op, "silent", underlying=u.id)
    return out


SILENT: dict[str, Pair] = _silent_pairs()
ALL_PAIRS: dict[str, Pair] = {**CATALOGUE, **SILENT}


def get_pair(pid: str) -> Pair:
    if pid in ALL_PAIRS:
        return ALL_PAIRS[pid]
    op, _, kind = pid.partition(":")
    if kind in ("sp1", "sp2") and op in SP_RENAMES:
        return sp_pairs(op)[0 if kind == "sp1" else 1]
    raise KeyError(pid)


def effective(p: Pair) -> Pair:
    """The schema change a pair makes (a silent pair makes its underlying change)."""
    return CATALOGUE[p.underlying] if p.type == "silent" else p


def fields_touched(p: Pair) -> set[str]:
    """Documented-path-level fields a pair touches (for the two-pair same-field rule)."""
    e = effective(p)
    if e.type == "moved":
        return set()
    return {e.path}


def legal_together(p: Pair, q: Pair) -> bool:
    """Two pairs on one operation (spec section 6, D69)."""
    if p.op != q.op or p.id == q.id or p.type == q.type:
        return False
    if fields_touched(p) & fields_touched(q):
        return False
    tp, tq = effective(p).type, effective(q).type
    if {tp, tq} == {"rename", "new_required"}:
        return False
    if p.type == "silent" and p.underlying == q.id or q.type == "silent" and q.underlying == p.id:
        return False
    return True


# ------------------------------------------------------------------------------------------------- current schema per op
def _rename_in(schema: tuple, path: str, new_name: str) -> tuple:
    if "[]." in path:
        parent, child = path.split("[].")
        return tuple(f.with_(items=tuple(g.with_(name=new_name) if g.name == child else g for g in f.items))
                     if f.name == parent else f for f in schema)
    return tuple(f.with_(name=new_name) if f.name == path else f for f in schema)


def _modify(schema: tuple, path: str, **kw) -> tuple:
    if "[]." in path:
        parent, child = path.split("[].")
        return tuple(f.with_(items=tuple(g.with_(**kw) if g.name == child else g for g in f.items))
                     if f.name == parent else f for f in schema)
    return tuple(f.with_(**kw) if f.name == path else f for f in schema)


def _current_path(path: str, renames: dict[str, str]) -> str:
    """Documented path -> current path given {documented path: new last segment}."""
    if path in renames:
        if "[]." in path:
            return path.split("[].")[0] + "[]." + renames[path]
        return renames[path]
    return path


@dataclass(frozen=True)
class OpView:
    """How one operation looks under the pairs in force."""
    op: str
    route: str                         # current operation id
    schema: tuple                      # current schema
    pairs: tuple                       # the Pair objects in force on this op
    drift_paths: dict                  # current-or-old path -> pair id (paths whose items are drift)
    drift_types: dict                  # path -> error item types that count as drift there (D80)
    silent_paths: frozenset            # paths checked at the app level (silent drift)
    renames: dict                      # documented path -> current path
    kinds: dict                        # documented path -> new kind (type drift)
    enum_maps: dict                    # documented path -> {old: new}
    new_fields: tuple                  # names of new required fields

    def is_drift(self, path: str, typ: str) -> bool:
        """An error item counts as drift only on a drifted path and of the matching kind (D80)."""
        return path in self.drift_paths and typ in self.drift_types[path]

    def is_silent(self, path: str, typ: str) -> bool:
        return path in self.silent_paths and self.is_drift(path, typ)


def op_view(op: str, pairs: list[Pair]) -> OpView:
    schema = OPS[op].schema
    route = op
    drift_paths: dict[str, str] = {}
    drift_types: dict[str, frozenset] = {}
    silent: set[str] = set()
    renames_seg: dict[str, str] = {}
    kinds: dict[str, str] = {}
    enum_maps: dict[str, dict] = {}
    new_fields: list[str] = []
    order = {"moved": 0, "type": 1, "enum": 2, "new_required": 3, "rename": 4}
    mine = [p for p in pairs if p.op == op]
    for p in sorted(mine, key=lambda p: (order[effective(p).type], p.id)):
        e = effective(p)
        paths: dict[str, frozenset] = {}
        if e.type == "moved":
            route = e.new_op
        elif e.type == "type":
            schema = _modify(schema, e.path, kind=e.new_kind)
            kinds[e.path] = e.new_kind
            paths = {e.path: TYPE_ERRORS}
        elif e.type == "enum":
            m = dict(e.mapping)
            schema = _modify(schema, e.path, values=tuple(m[v] for v in _field_at(schema, e.path).values))
            enum_maps[e.path] = m
            paths = {e.path: frozenset({"enum"})}
        elif e.type == "new_required":
            schema = schema + (e.new_field,)
            new_fields.append(e.new_field.name)
            paths = {e.new_field.name: frozenset({"missing"}) | TYPE_ERRORS | {"enum"}}
        elif e.type == "rename":
            schema = _rename_in(schema, e.path, e.new_name)
            renames_seg[e.path] = e.new_name
            newp = _current_path(e.path, renames_seg)
            paths = {e.path: frozenset({"extra_forbidden"}), newp: frozenset({"missing"})}
            if e.prev_name:
                paths[_current_path(e.path, {e.path: e.prev_name})] = frozenset({"extra_forbidden"})
        for q, types in paths.items():
            drift_paths[q] = p.id
            drift_types[q] = types
        if p.type == "silent":
            silent |= set(paths)
    # a type/enum change on a renamed path is attributed under the current name too (not used by legal pairs)
    renames = {d: _current_path(d, renames_seg) for d in renames_seg}
    return OpView(op, route, schema, tuple(mine), drift_paths, drift_types, frozenset(silent), renames, kinds,
                  enum_maps, tuple(new_fields))


def _field_at(schema: tuple, path: str) -> Field:
    if "[]." in path:
        parent, child = path.split("[].")
        f = next(f for f in schema if f.name == parent)
        return next(g for g in f.items if g.name == child)
    return next(f for f in schema if f.name == path)


# ------------------------------------------------------------------------------------------------------ canonicalisation
def _to_number(s: str) -> int | float:
    d = Decimal(s)
    return int(d) if d == d.to_integral_value() else float(d)


def _canon_value(kind: str | None, emap: dict | None, v: Any) -> Any:
    if kind == "digits":
        v = int(v)
    elif kind == "decimal":
        v = _to_number(v)
    elif kind == "int01":
        v = bool(v)
    if emap:
        inv = {new: old for old, new in emap.items()}
        v = inv[v]
    return v


def canonicalize(view: OpView, args: dict) -> dict:
    """Current-schema arguments (already validated) -> documented names, types and enum values."""
    inv_top = {cur: doc for doc, cur in view.renames.items() if "[]." not in doc}
    inv_item = {}
    for doc, cur in view.renames.items():
        if "[]." in doc:
            parent, child = doc.split("[].")
            inv_item[(parent, cur.split("[].")[1])] = child
    out: dict[str, Any] = {}
    for k, v in args.items():
        if k in view.new_fields:
            continue
        doc = inv_top.get(k, k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            items = []
            for it in v:
                ni = {}
                for ik, iv in it.items():
                    dk = inv_item.get((doc, ik), ik)
                    path = f"{doc}[].{dk}"
                    ni[dk] = _canon_value(view.kinds.get(path), view.enum_maps.get(path), iv)
                items.append(ni)
            out[doc] = items
        else:
            out[doc] = _canon_value(view.kinds.get(doc), view.enum_maps.get(doc), v)
    return out


def to_current(view: OpView, doc_args: dict, new_values: dict | None = None) -> dict:
    """Documented-form arguments -> current-schema arguments (the oracle's translation). New required fields get
    their constant, the first enum value, or True (or `new_values[name]`)."""
    out: dict[str, Any] = {}

    def conv(path: str, v: Any) -> Any:
        k = view.kinds.get(path)
        m = view.enum_maps.get(path)
        if m:
            v = m[v]
        if k == "digits":
            v = str(int(v))
        elif k == "decimal":
            v = f"{Decimal(str(v)):.2f}"
        elif k == "int01":
            v = 1 if v else 0
        elif k == "array_str":
            v = [v]
        return v

    for k, v in doc_args.items():
        cur = view.renames.get(k, k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            items = []
            for it in v:
                ni = {}
                for ik, iv in it.items():
                    path = f"{k}[].{ik}"
                    ck = view.renames.get(path, path).split("[].")[1]
                    ni[ck] = conv(path, iv)
                items.append(ni)
            out[cur] = items
        else:
            out[cur] = conv(k, v)
    for f in view.schema:
        if f.name in view.new_fields and f.name not in out:
            if new_values and f.name in new_values:
                out[f.name] = new_values[f.name]
            elif f.kind == "const":
                out[f.name] = f.const
            elif f.kind == "enum":
                out[f.name] = f.values[0]
            else:
                out[f.name] = True
    return out


# ------------------------------------------------------------------------------------------------------------------ pools
def build_pools(seed: int = POOL_SEED) -> dict[str, list[str]]:
    """Split the 58 base pairs and 16 silent pairs into disjoint dev/test pools, stratified by type, with every
    operation in both pools (each op's rename and moved go to different pools)."""
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    for op in OP_NAMES:
        a, b = f"{op}:rename", f"{op}:moved"
        if rng.random() < 0.5:
            dev.append(a); test.append(b)
        else:
            dev.append(b); test.append(a)
    for typ in ("type", "new_required", "enum"):
        ids = sorted(p.id for p in _ROWS if p.type == typ)
        rng.shuffle(ids)
        n_test = (len(ids) + 1) // 2
        test += ids[:n_test]
        dev += ids[n_test:]
    for sid, s in sorted(SILENT.items()):
        (dev if s.underlying in dev else test).append(sid)
    return {"dev": sorted(dev), "test": sorted(test)}


POOLS = build_pools()


def draw_drift_set(rng: random.Random, pool: list[str], n_ops: int = 8, n_double: int = 2,
                   exclude_ops: frozenset = frozenset(), max_tries: int = 500) -> list[Pair]:
    """Stratified drift set (D70): each type at least once where the pool allows, n_ops operations, n_double of them
    with two legal pairs. Seeded constructive search; raises if no set is found."""
    pairs = [get_pair(i) for i in pool if get_pair(i).op not in exclude_ops]
    by_type = {t: [p for p in pairs if p.type == t] for t in TYPES}
    for _ in range(max_tries):
        chosen: list[Pair] = []
        used_ops: set[str] = set()
        types = [t for t in TYPES if by_type[t]]
        rng.shuffle(types)
        ok = True
        for t in types:
            cands = [p for p in by_type[t] if p.op not in used_ops]
            if not cands:
                ok = False
                break
            p = rng.choice(cands)
            chosen.append(p)
            used_ops.add(p.op)
        if not ok or len(used_ops) > n_ops:
            continue
        while len(used_ops) < n_ops:
            cands = [p for p in pairs if p.op not in used_ops]
            if not cands:
                ok = False
                break
            p = rng.choice(cands)
            chosen.append(p)
            used_ops.add(p.op)
        if not ok:
            continue
        doubles = 0
        ops_order = sorted(used_ops)
        rng.shuffle(ops_order)
        for op in ops_order:
            if doubles == n_double:
                break
            first = next(c for c in chosen if c.op == op)
            cands = [p for p in pairs if p.op == op and legal_together(first, p)]
            if cands:
                chosen.append(rng.choice(cands))
                doubles += 1
        if doubles == n_double:
            return sorted(chosen, key=lambda p: p.id)
    raise RuntimeError("no legal drift set found")

```

The rest (schema.py, ops.py, templates.py, streams.py, oracle.py, tests) is in the repository https://github.com/notanut/eem-acc-skripsi (main) if you can read it; ask for any file you need.
