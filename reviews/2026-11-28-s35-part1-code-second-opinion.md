I'll review the Part 1 analysis code on the current `main`: first the spec definitions and payload fields, then the four new modules, then tests, the dry run and targeted mutations.

I reviewed d33b26d, the latest `main`, since the request didn't name a commit. There are no High findings, 4 Medium and 5 Low. With the pinned versions installed, all 94 S3.5 analysis tests pass. I re-ran the 171-unit dry run and checked each definition against the real payloads.

## Medium

**M1. The C2-OU comparison is empty after T, and shrinks if S stops early** (`descriptive.py:23`, `report.py:full_T`).
- Spec section 16 puts C2-OU in the T part of `full.json`. `full_T` has C2-OU's costs, errors and mechanism figures, but no C2-OU − C2 difference.
- That difference exists only in `full_S`'s `arms_vs_c2`. There, the streams are derived from the S arms with C2-OU excluded (`a != "C2-OU"`).
- On T units only, the C2-OU row has 0 units. If only tier 1 ran, C2-OU is cut to 3 of its 5 streams. Either way nothing warns.
- Fix:
  - add the C2-OU − C2 row (success and API calls, on C2-OU's own streams, next to C0–C3 there) to `full_T`;
  - in `arms_vs_c2`, take each arm's streams from its own units.

**M2. LLM goodput counts failed calls as good** (`secondary.py:llm_goodput`).
- It counts every call that isn't transient. Entries in `llm_calls` have no `transient` key unless the call was a 503, so 400, 404, 410 and 422 errors all count. In the dry run that is about 1,870 of 10,935 LLM calls.
- Background goodput (`metrics._completed_ok`) counts only successful calls completed in the window.
- So the two D206 versions of goodput differ, and the LLM one rises with drift errors, which favours C0 in C2 − C0.
- Fix: count `status == 200`, completed within [warmup, t_end).

**M3. The type-drift matcher misses natural phrasing** (`mechanism.py:names_truth`).
- It accepts only `field: kind`, `'field' is now kind`, `->` and `=`.

| Phrasing | Matched |
|---|---|
| `query: list` | yes |
| `query is now a list` | no |
| `query must be a list` | no |
| `query must be of type list` | no |
| `query should be list` | no |

- EEM notes still match, because their k_t expected part is structured (`items[].qty: string`).
- A1's raw text and C1/F reflections are written by the LLM, so type drifts are undercounted for exactly the arms being compared with EEM.
- Fix:
  - also match field and kind within a few words in the same clause (no `.`, `;` or newline between them);
  - report both the strict and the loose version;
  - check the loose one on S3.4 pilot notes before the freeze.

**M4. (Q4) The VB–MAP gap should be a frozen rule, not a caveat.**
- "Chapter IV quotes it only together with those two figures" leaves the choice to show the model until after the data are seen.
- Fix: freeze a rule now. Quote the mixed model only if:
  - MAP converged with no warning;
  - the largest VB–MAP gap is at most 0.5 logits;
  - no condition's success is ≥ 0.98 or ≤ 0.02.

  Otherwise report GEE alone and say why.
- The rest of D187 is sound: the priors, VB as the estimate, and MAP as a check. On the dry run, MAP failed with a gradient of about 2×10⁶, which is exactly the case this rule removes.

## Low

- **L1. (Q3) The hash depends on the scipy version.**
  - `report.primary` loads only numpy and scipy, and the hash is the same across three `PYTHONHASHSEED`s and numpy 2.0.2 or 2.4.6.
  - It changes between scipy 1.16.3 and 1.18.1 (`2f21…` vs `0dfc…`): `t.ppf(0.975, 13)` differs at about 8 parts in 10¹³, which flips one rounded digit of the lower bound of NI-R1 and S-R1's `ci95` (and their failed-as-unsolved copies).
  - Reproducing with Colab's recorded scipy works. More robust: hash only the exact parts (p-values, counts, rejections, untied n) and compare floats at a relative tolerance of 1e-9.
  - The evidence's "two runs" were in one process; add a test in a subprocess with another `PYTHONHASHSEED`.
- **L2. The mechanism checks use a different population.** `mechanism._units` takes every done unit of C0–C3, while `population()` uses only the complete streams. Use the same rule in both.
- **L3. Missing RQ3 values are averaged silently.** `rq3_secondary`'s per-cell means drop `None` metrics (e.g. an SLO rate with no judged calls) without saying so. Report n per metric.
- **L4. GEE's intervals are too narrow.** They use normal quantiles with 14 clusters and 8 coefficients. Use t with df = clusters − 1, or state it.
- **L5. (Q5) Gaps in the tests.**
  - Of 26 targeted mutations, 21 were killed. These survived:

| Mutation | Effect |
|---|---|
| `"context_overflow"` misspelled | the overflow share reads 0 |
| GEE `fit()` instead of `cov_type="bias_reduced"` | the wrong covariance goes unnoticed |
| position not centred | only the condition main effects change |
| `rq3_failed_as_unsolved` without `include_failed=True` in `report.primary` | a wrong D186 sensitivity enters the frozen, hashed `primary.json` |
| C2-OU included in the arm streams | the stream derivation is untested either way (M1) |

  - Nothing tests natural-language type notes (M3).

## Q1: definitions that check out

Against `loop.py`, `memory.py` and `rq3.py`:
- **Exposure attribution:** `first_call` is the first call to an active operation, excluding docs calls. An operation with two pairs counts under each type, and style is per operation.
- **Repeat exposure:** every RQ1 stream has one drift stage starting at task 0, so "after the first exposure" per operation is unambiguous.
- **`item_belongs` and both precisions:** last-segment paths, and kinds `rename`, `moved`, `remap`, `type` and `added`, all seen in the payloads.
- **Retrieval recall:** `notes_on_op` is recorded after the call and `visible` before it, so filtering by `visible` is correct.
- **C1 crowding:** works as defined.
- **SP:** `at` is `[agent, task, step]`, so `at[1]` is the task. The re-drift check against `sp_op` and the schedule is right.
- **Completion times:** "finished" means not censored (not necessarily solved), and the censored version leaves out never-started tasks.
- **RQ2 paths:** `throttle_rate`, `slo_violation_rate`, `goodput` and `J` under `infra.windows.llm_phase`, plus `llm_metrics`, all exist.
- **Integer TOST:** correct, and the 95% interval check is right at 0.025 per side.

## Q2: silent wrong results

Phase and tag filtering, the complete-stream population for C0–C3, keeping the `sens` units out of the primary analysis, and the per-agent flattening of the C2-private lists are all correct. The silent failures are M1 and L3.