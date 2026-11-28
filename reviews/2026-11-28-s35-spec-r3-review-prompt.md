# Second-opinion review request: S3.5 spec r3 (the delta since r2)

Repository: https://github.com/notanut/eem-acc-skripsi, commit `2d3f0cd`. Contract: `docs/s35_spec.md` r3.

You reviewed r2 (`reviews/2026-11-28-s35-spec-r2-second-opinion.md`). Section 21 says how each of your findings was
applied. A fresh-context review of that delta then found 2 High, 3 Medium and 5 Low issues (section 22). The user
decided:

- **D192 (your H1):**
  - T is final at AP: a T unit is never re-run after AP, only reported as a deviation;
  - during S, a re-pin is allowed only to fix an `X` cause, and its `Z` is answered automatically;
  - a commitment to run S until the budget rule ends it.
- **D193 (your M1):** tier 1's cost is reserved in T's budget check.
- **D194:** a fix during S with no `X` behind it is allowed only when S stalls and the fix changes no behaviour,
  analysis, protocol-data or T-plan hash. Otherwise S stops, and this is reported.
- **D195:** S stays blinded (the same whitelist) until it ends.
- **D196:** one more blind budget check after the last T block, so tier 1 cannot be squeezed out. `reserved(tier 1)`
  is a fixed number in `freeze.json`.

Code changes: `endpoints.py` now reports two flags (`below_initial_level`, `unrejectable`) and requires a non-empty
`phase`. The BMBP09 worked examples are now tests (`tests/data/bmbp09_examples.json`, all 5 numerical examples). The
paper's Algorithm 2 serves as an independent adjusted-p check.

Questions:
1. After AP, is there still any decision or code path whose outcome can depend on the primary or descriptive results?
   Check sections 12, 14, 15 and 16.
2. Can any combination of `X`, halt, stall, `Z` (with or without a re-pin), epoch and `B` deadlock or contradict
   another rule? Check the section 14 table.
3. Is the hand-over check (D196) plus the reservation (D193) enough to keep tier 1 unless the user makes a blind
   `stop`?
4. Is the stated remaining leak (section 15: C2-OU and per-RQ3-seed aggregate cost from block differencing)
   acceptable? Or would you show hours only at T's end and at `B` pauses?
5. Is anything wrong in the new tests or in `tests/data/bmbp09_examples.json`? Compare them with the paper
   (Statistics in Medicine 28:586-604): Fig. 4 with section 3.1, Figs. 9, 10, 2 and 12. Note the p. 598 typo we
   believe the paper has.

Please classify each finding as High, Medium or Low, and give a concrete fix for each.
