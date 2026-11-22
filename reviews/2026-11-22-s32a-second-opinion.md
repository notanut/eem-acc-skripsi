# S3.2a Implementation Review (second opinion, external LLM; brought by the user 2026-11-22)

Archived verbatim from the user's upload (checked at commit dd9d1f0).

I'll pull the latest code, run the S3.2a tests, and probe the note rendering, the re-check rule and the merge behaviour directly.

S3.2b and S3.3 can build on S3.2a once two EEM defects are fixed: the note cap can drop the outcome line (H1), and a merge can erase a learned lesson (H2). Both make C2 fail for reasons unrelated to the idea. Also settle M1–M2 before S3.2b fixes its interfaces. I checked commit `dd9d1f0`: all 223 S3.2a tests pass, and I ran the probes quoted below. The S3.1 fixes (D85 cases A and D, D88 `known_keys`) are in.

## High

**H1 (`memory.note_line`): the note cap squeezes out the outcome line, which is C2's most precise content**
- Each note is rendered k_t (≤30 tokens), then the reflection (≤60), and the outcome line gets whatever is left of 120.
- The labels ("- ", ". ", "Fix: ", "Last working change: ", " (possibly outdated)") are not part of the 30 + 60 + 30 budget.
- **Probe with the stub tokenizer,** k_t at its cap, two items (a rename of `items[].qty` and the constant `currency`):
  - At a 30–55-token reflection, only **1 of 2 items** is shown.
  - At a 60-token reflection, **no line at all** is shown.
- The reflection prompt allows 40 words (about 55 tokens), and Qwen tokenizes code-like text (`'items[].quantity'`, quotes, brackets) into more tokens than the stub. So this will be common in real runs.
- **Fixes:**
  - Reserve the line's room before the reflection: reflection room = cap − k_t − line − labels, measured with the tokenizer.
  - Raise the line cap so two items fit (two operations per stream have two drifts). For example, a line cap of 40 with the note cap raised to keep the reflection at about 50 or more.
  - Log `line_shown` and `items_shown / items_total` for every retrieved note.
  - Extend the planned S3.3 Qwen test from k_t to the full note, for every catalogue row and every legal two-pair combination.

**H2 (`EEMStore.on_error`, merge rule): an unrelated agent error permanently erases a learned lesson**
- The merge key is (API_Name, Error_Type).
  - Every terse validation error is `bad_request`.
  - An informative drift `missing_field` (the new key of a rename) shares its key with the agent's own missing documented field.
- A merge removes the line.
- **Probe on terse `create_order`:**
  - Task 3: the rename `qty → quantity` is learned (line `['rename']`).
  - Task 7: the agent forgets `shipping_method` (a 400, an agent error). The merge empties the line.
  - The fix in task 7 succeeds, but the new trigger already had `quantity`, so the line stays empty. The lesson is lost for the rest of the stream.
- **Decision for you (it changes the r4 rule "merge removes the line"):**
  - **(a) Recommended:** on a merge, keep the old items as a possibly-outdated line, re-check them item by item at the next success as for other notes, and add the new diff's items.
  - **(b)** Keep the current rule and report how many lines are lost to merges.
- Either way, log every line lost or kept at a merge.

## Medium

**M1 (`on_error`, marking): business-rule and resource errors mark correct lessons as possibly outdated**
- Probe: a 409 (`business_rule`) on `create_order` sets `possibly_outdated=True` on the rename lesson.
- A wrong id or a wrong amount can't indicate a schema change. Yet the mark makes the scripted agent ignore the line, and a real agent may distrust it.
- **Fix:** mark only after errors that can signal drift: 400/422 validation, silent errors, route 404 and 410. Do not mark after `business_rule` or `resource_not_found` (EEM knows the type from k_t).

**M2 (`ErrorCtx.canonical_op`, `parser`): EEM reads ground truth**
- Where it happens:
  - `canonical_op` comes from the suite's `ApiResponse.op`. It drives closing, marking, re-checking and the documented schema used for the diff.
  - The parser's `MOVED_TO_OP` and `ENUM_BY_OP` come from the drift catalogue, which includes test-pool remaps.
- In reachable cases, an alias learned from the 410 pointer or a `get_api_docs` response gives the same result. But "EEM uses only what the agent observed" is then not true by construction, and S3.2b is about to put `canonical_op` into the gateway's `Outcome`.
- **Fixes:**
  - Build an alias map from 410 pointers and `get_api_docs` responses (`operation` / "formerly X").
  - Build the enum whitelist from the documented values plus the error's own `ctx.expected`.
  - Keep the suite's op for logs only.
  - Add a test that the scripted runs produce identical results.

**M3 (logs): three gaps in the mechanism checks**
- **Retrieval recall denominator:** `notes_on_op` is computed after the dispatch. In S3.2b, other agents may have created notes during this step's LLM and API time, so recall would be underestimated. Compute recall from `visible` (taken at retrieval and already logged), filtered by operation.
- **`line_shown`:** see H1.
- **Lines lost or kept at merges:** see H2.

**M4 (fairness in interpretation): C1 and C2 get a reflection step that C0 doesn't**
- After every error, C1 and C2 get an extra LLM reflection call, and the observation shows the just-written note (D91). So C2 vs C0 (E1/E2) mixes help within the task with memory across tasks.
- The logs support splitting it: report first-exposure outcomes and repeat-exposure outcomes separately, and state this in Section IV.
- C2 vs C1 is unaffected, since both reflect.

## Low

- **Scrub k_t too:** apply the scrubber to the k_t rendering and API_Name as defence in depth, since the agent controls `action` and invented key names.
- **Echo traps** are an upper bound: an echoed value also equals the documented value the agent would send anyway. State this.
- **Replay of embeddings:** `UnitEmbedder` exports vectors, but S3.3 replay must load them instead of recomputing. Add the import path and a test.

## Your checklist

1. **Fairness:**
   - System turns are byte-identical (except R), the empty-store prompt equals C0's, and the retrieval query and reflection inputs are equivalent between C1 and C2.
   - C2's notes use more memory tokens than C1's (k_t and the line), which is part of what C2 − C1 measures. Report prompt tokens.
   - The within-task reflection is covered in M4.
2. **Faithfulness:**
   - It is faithful to III-B.
   - The failure modes unrelated to the idea are H1, H2 and M1.
   - Retrieval runs before the action is chosen, so the retrieval-recall check is essential. AN gives the ceiling for it.
3. **Leak safety:** I found no leak.
   - k_t carries keys, types, schema values and value-free sentences.
   - Reflections and A1 text are scrubbed of every task value seen, including the normalised forms.
   - Outcome lines come from the S3.1 reference.
   - Trigger arguments are stored but never rendered.
   - The S3.3 audit of real reflections stays necessary.
4. **Measurement:**
   - The note events make correctness before and after the outcome update, and precision, recomputable.
   - Planned vs experienced exposures, echo traps and skipped reads are all logged.
   - Add the M3 items.
5. **Readiness:**
   - S3.2b: the shared-store timing is consistent (single-threaded asyncio, notes visible after the reflection's sleep). M2 and the recall fix should come first.
   - S3.3: the Qwen tokenizer will make H1 worse, so fix H1 before the pilot.

## Disposition (2026-11-22)
- H1: user chose to reserve the line inside 120 tokens (D112): k_t <= 30, line <= 40 reserved first, reflection the
  rest; line_shown / items shown logged.
- H2: user chose to keep old items at a merge as a possibly-outdated part (D113).
- M1: user chose to mark only after drift-capable errors (D114).
- M2: applied (clear-cut): alias map from observations, enum whitelist from docs + error ctx, scrub safe list from
  observed vocabulary; suite op for logs only.
- M3: applied: recall from visible notes; note stats; merge logs (items_kept).
- M4: analysis obligation recorded (first vs repeat exposure).
- Low: k_t scrubbed; echo traps stated as upper bound; embedding replay preload + test.
