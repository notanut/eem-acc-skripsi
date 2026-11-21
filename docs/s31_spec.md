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
