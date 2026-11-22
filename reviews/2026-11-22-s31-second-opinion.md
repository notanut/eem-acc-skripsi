# S3.1 Implementation Review (second opinion, external LLM via Claude Docs; brought by the user 2026-11-22)

Archived summary of the user's doc (checked at commit e61fd0c; all 205 S3.1 tests passed there).

## Verdict
S3.2 can build on S3.1 once two things are fixed: the reference outcome-line diff in leakcheck.py (H1) and an
idempotency guard (M2). Drift composition (M1) needs a decision before the suite freeze, call timing (M3) before S3.3.

## High
**H1 (leakcheck.py, section 10): the outcome line misses two common paths.**
- Case D, a value fix on a key the trigger already had: get_cart trigger {"customer_id": "C-1", "currency": "USD"},
  success {"customer_id": "C-1", "currency": "IDR"} gives None. Usual path for constants under informative errors.
- Case A, a rename where the trigger already sent the new key (pay_order with both amount and total) gives None.
- Fixes proposed: record a changed value on an undocumented key present in both, under the new-required guards; in
  the rename branch name vanished documented keys when no key appeared; add both cases to the leak property test.

## Medium
**M1: drift composition concentrated and shifts between dev and test.** add_comment:type in 9 of 14 test streams,
refund_payment:new_required in 8; rename/moved split 7/9 vs 9/7; test takes the odd type and enum pairs. Fixes:
exactly 8 renames per pool; odd type pair to one pool, odd enum pair to the other; draw test (and dev) drift sets
jointly; report per-pair frequencies and a leave-one-pair-out check.

**M2 (suite.execute, D68): a reused key with a different call returns a stale response.** Fix: store (op_id, args
hash, task_index), raise on mismatch, test it.

**M3 (interface for S3.3):** run execute() at admission; count "hidden executions" per RQ3 cell.

**M4 (section 5): silent errors more informative than visible ones for constant fields** (silent message gives the
constant; the 422 says "Field required"). Fix: mirror the 422, or disclose.

## Low
- D81: name a dropped undocumented key only if an earlier outcome line on the same operation introduced it.
- get_api_docs errors: S3.2 must never create notes from them; add a test.
- D80b echo traps: tag errors caused by copying an echoed documented enum value.
- Guessable reads: report skipped reads per condition (affects E2/E4).

## Checklist answers (summary)
Validity: no condition-specific bias; strict extra_forbidden and drift from task 0 favour the outcome update (state in
Section IV). Solvability: every drift inferable via get_api_docs; hardest case terse moved (report per type x style).
Leak safety: no leak found. Statistics: see M1; state the template-mix tilt; exposures planned and experienced.

## Disposition (2026-11-22)
- H1: user chose a safer variant (D85): case A as proposed; case D records the value only if the note's own error
  named the key and the operation has no rename. The reviewer's version would have recorded an amount on rename
  operations (trigger total 99999, success total 100000). Tests added.
- M1: adopted (D86): pools 8/8 rename and moved, odd type to test, odd enum to dev, joint drawing; usage evidence.
- M2: adopted (clear-cut): reused call_key with a different call raises.
- M3: recorded for S3.3 in the spec.
- M4: adopted (D87): silent messages mirror the 422.
- Low items: user gave no preference; adopted as recommended (D88); the S3.2 and analysis parts are recorded as
  obligations in HANDOFF.
