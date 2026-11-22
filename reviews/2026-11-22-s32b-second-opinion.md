I'll pull the latest code, run the RQ3 tests, and check a scripted RQ3 unit for the metric and shared-store behaviour.

S3.3 can build on S3.2b. I found no High issues. There are two Medium fixes for the analysis: the LLM-only metrics are diluted by semantic errors (M1), and in the shared store owner rewrites are the norm, not an edge case (M2). I checked commit `66d471a`: 69 of 69 tests in `test_s32b_rq3.py` and `test_s32_memory.py` pass. My earlier S3.2a findings are applied (D112 budget order, D113 items kept on merge, D114 marking only on drift-type errors, aliases learned from observations).

I ran scripted units on dev seed 5000. These numbers are plumbing only:

| Cell | Success | LLM attempts | Semantic 4xx | LLM overload (4xx excluded) | Merges (by another agent) |
|---|---|---|---|---|---|
| C0 + ACC | 0.95 | 171 | 36 (21%) | 0.000 | 0 |
| C2 + ACC | 0.95 | 135 | 21 (16%) | 0.000 | 20 (19) |
| C0 + none | 0.60 | 772 | 35 (5%) | 0.789 (0.826) | 0 |
| C2 + none | 0.85 | 585 | 19 (3%) | 0.781 (0.807) | 17 (13) |

## Medium

**M1 (`_llm_metrics`, `J_rq3`): semantic errors flatter the condition that makes more of them**
- Semantic 4xx attempts are fast (0.02 s), never overload and carry zero latency excess. Yet they sit in the denominators of overload rate, mean latency excess, attempts per call, and therefore J_rq3.
- So C0's extra drift errors lower its overload and excess figures and raise its J. In the table, excluding them moves overload from 0.789 to 0.826 (C0) and 0.781 to 0.807 (C2), which widens the C0–C2 gap.
- A real 7B agent will make more errors than the scripted one.
- **Fixes:**
  - Compute the LLM infrastructure metrics and J_rq3 over attempts that took the capacity path (200, 429, 503, timeout), excluding semantic 4xx. Keep the current version as secondary.
  - State the same dilution for ACC's interval statistics. The share there is tiny (LLM attempts are a few percent of all traffic), so report the measured share.

**M2 (shared store, owner rule): owner rewrites are the norm**
- In the C2 + ACC run, 19 of 20 merges came from an agent other than the note's previous owner.
- Over seeds 5000–5002, 20–40 writers per unit lost ownership before their note closed, with up to 10 distinct writers on one note.
- With the scripted backend every note still ended up with a line. With a real LLM, whether a lesson gets recorded depends on who erred last, which has nothing to do with EEM.
- **Fix (decision for you):**
  - **(a) Recommended:** keep the trigger arguments per writer (agent, task). Any writer's same-task success closes the note by diffing against that writer's own trigger, and new items are merged as with D113.
  - **(b)** Keep the rule, and log "lost closes": a writer's same-task success on X after losing ownership. Then the pilot can measure how often it happens.

## Low

- **Synchronised start:** all 4 agents start at exactly 20.0 s, and the exposure rule puts first exposures in their first one or two tasks. So the agents meet the same drifts at the same moment, before any note exists, which shrinks the shared store's room to help. Either add a seeded stagger of start times, identical across cells, or state it.
- **Environment in the stamp:** the replay of an RQ3 unit depends on event ordering, so the Python and numpy versions are part of it. Colab upgrades its Python now and then. Stamp both, and refuse to resume across a change, as D51 does for GPUs.
- **`assert fits(body)`** in `note_render` can crash a unit mid-campaign. Fall back gracefully (drop line items, then shorten the reflection) and log it.
- **Echo of earlier caveats:** the C0 vs C2 difference under ACC also includes the reflection step after errors (my earlier M4); report first-exposure and repeat-exposure outcomes separately.

## Your checklist

1. **Validity of R1 and R2:**
   - Seeds are cell-independent, and the background draws are identical across cells. The warm-up is the same for every cell, transients create no notes in either condition, T_max does not bind, and never-started tasks count as failures.
   - No bias between C0 and C2, or ACC and none, beyond M1 (secondary metrics) and M2 (shared-store attribution).
   - Under ACC the scripted LLM overload is 0, so R1 is effectively RQ1 under concurrency with a shared store. State that.
2. **Metrics:**
   - Background-only infrastructure metrics (D117) are defensible: identical draws, and LLM traffic is a small share.
   - The LLM-only metrics need the M1 fix.
3. **Semantic server:**
   - The order is right: semantics run only after 429/503 and the token, at admission.
   - Retries after a client timeout replay the stored response, so state changes once.
   - Hidden executions are detected through `suite.executed`.
   - Semantic 4xx attempts are kept off the capacity path, and keyed noise equals the plain tuple's (tested).
4. **Concurrency:**
   - The timing rules and D119 are consistent with single-threaded virtual time.
   - Owner rewrites are covered in M2.
5. **Readiness for S3.3:**
   - Deterministic replay and embedding preload are in place, and `LLMRequest` pickles.
   - An RQ3 unit is one simulation (about 200–300 LLM calls, roughly 20 minutes on a T4), so resuming by replaying the whole unit is reasonable.
   - Add the version stamp from Low.
6. **D115 and D116:**
   - No leak: aliases are operation ids learned from observed pointers and docs responses, and compact remaps are schema values.
   - Closing through an alias chain is correct.
   - Cutting the oldest items first fits D113, since the kept items come first and the new diffs last.