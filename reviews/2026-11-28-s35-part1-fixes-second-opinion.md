I'll review the d33b26d → 86988ef delta, focusing on goodput, the loose matcher, the exact-part hash and the new tests.

I reviewed `86988ef` against `d33b26d`: no High, 1 Medium, 4 Low. The Medium is a real bug in the loose matcher, not wording. It's cheap to fix, and it should go in before D209 is frozen and checked on the pilot notes. After that, Part 2 can start.

I read the code, ran 107 tests (all pass), probed the matcher with adversarial strings, re-ran 8 RQ3 units with the gateway captured, and ran 25 mutants of the delta.

## Q1: goodput — yes, now the same definition as the background goodput
- `llm_calls[].status` is the gateway's `CallRecord.status` (`rq3.py:139`). `CallRecord.ok` is `status == 200` (`agent/gateway.py:60`), which is the flag `_completed_ok` uses. Semantic 4xx come back with their own status (`rq3.py:113-117`).
- The window [warmup, t_end) and the denominator are the same as `llm_phase`'s.
- In the 8 re-run RQ3 units, the new `llm_goodput` exactly equals `_completed_ok` on the gateway's LLM calls divided by (t_end − warmup). The old value was 13–35% too high.

## Q2: the loose matcher

**M1 — the "no negation" rule misses the form the reflection prompt asks for** (`mechanism.py:82-84`)
- The negation check only looks at the words between the field and the kind. But `REFLECT_INSTR` (`prompts.py:33`) asks for "one imperative instruction", where the negation comes first.
- The loose rule returns True for all of these:
  - "Do not send query as a list", "Never pass query as a list", "Avoid sending query as a list", "stop sending query as a list"
  - "tags (not query) must be a list"
  - "query cannot be a list" (the pattern needs a non-letter before `not`)
  - "query isn’t a list" (curly apostrophe)
- The gap is also greedy, so it can reach a second kind word past a negation. "Send amount as a string not decimal" returns False, although "as a string" is correct.
- **Fix:**
  - Look for negation from the start of the sub-clause (the last `, ; . : ! ? …` or newline before the field) up to the kind.
  - Add `cannot`, `avoid`, `stop` and `[A-Za-z]+n['’]t`.
  - Make the gap lazy (`{0,4}?`).
  - Test that "Don't send a string, send query as a list" stays True, because the comma resets the scan.

**L1 — natural phrasings that neither rule matches.** Add these to the D209 pilot check:
- "`query`: list", "**query**: array of strings", "query: must be a list".
  - The strict pattern (`:99`) allows no backticks or `**`, and the loose rule treats a colon right after the field as a clause end.
  - **Fix:** in the loose rule, allow `` [`'"*)]*\s*:? `` right after the field. The colon stays a clause end inside the gap.
- "Pass query as a List", "a JSON Array": the kind words are case-sensitive. **Fix:** make only the kind words case-insensitive.
- Already stated as limits: the kind before the field ("Use a list for query") and "from a string to a list" (5 words). The imperative wording makes kind-first likely, so count it in the pilot.
- From before this delta: for `add_comment:type`, "Send public as 1 or 0" is missed, because int01's kind words are only `int01` and `integer`.
- False positives from closeness stay ("query/tags are list"). That's acceptable for a loose rule; say so.

## Q3: D210
- **Nothing scipy-derived is left in `exact_part`.** scipy is used only in `t_ci` and `mdd` (`stats.py:64-82`), and their only outputs go into `ci95` and `mdd` (`endpoints.py:63-64, 147`). `mdd_alpha_one_sided` is a level, and it is correctly hashed.
- **Nothing exact is left out.** Those two keys hold only t-based values.
- **Wording:** `report.py:52` and D210 say "exact arithmetic … means of integer differences". The hashed part also holds non-integer floats: the E2/E4 differences (e.g. −3.105263158), their means and the interaction means. They are deterministic and use no scipy; call them that.
- **L2:** `RTOL = 1e-9` is the same size as the 10-digit rounding in `canonical()`.
  - If Part 2 compares two written files, both sides are rounded, and scipy-level noise across a rounding edge fails. `1.00000000049995` vs `1.00000000050005` (1e-13 apart) round to 1.0 and 1.000000001, and `_close` flags them.
  - The recorded file against the unrounded recomputation passes.
  - **Fix:** compare against the unrounded recomputation and say so in the docstring, or use `RTOL = 1e-8`. That is still about four orders above the ~1e-12 noise seen.
- **L3:** `vb_map_max_abs_diff` uses Python's `max` (`secondary.py:244`), which skips a NaN that isn't first: `max([0.1, nan, 0.2]) = 0.2`. So the NaN check in `quotable` only works when the first coefficient is NaN. **Fix:** if any difference is NaN, make the gap NaN.

## Q4: tests
I found no tautologies. The t-interval check reuses the code's own `se`, but it still pins the degrees of freedom, which is what it's for. Of the 25 mutants, 15 were killed. The survivors that matter:
- **L4:**
  - **Loose rates:** in `note_correctness` (`:201` and the final-notes line), replacing the loose rate with the strict one survives, because every test case has loose == strict. Add one note that only the loose rule matches.
  - **Hash coverage:** removing `rejected` or `mean_difference` from the hash survives; the test at `:611` changes only `p`. Add a structural test: the leaves `exact_part` removes are exactly the `ci95`/`mdd` leaves, and changing `graph.rejected` changes `primary_hash`.
  - **Tolerance:** `RTOL = 1e-7` survives, because the tests use only 1e-12 and 1e-6. Add a 1e-8 change that must be flagged.
  - **D208 boundaries:** D208's low end (≤ 0.02) and the 0.98 boundary are untested; two mutants survive.
- The other survivors are equivalent or negligible: C2-OU over T+S (C2-OU runs only in T), "either fit missing" (a missing gap fails anyway), and a case-sensitive negation inside the gap.

I couldn't run the matcher on real LLM notes: the repo evidence has none, and the S3.4 pilot notes aren't in the repo. The adversarial lists above are meant as the cases for that pilot check.