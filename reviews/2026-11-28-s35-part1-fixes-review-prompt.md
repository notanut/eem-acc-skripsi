# Second-opinion review request: S3.5 Part 1 fixes (the delta since d33b26d)

Repository: https://github.com/notanut/eem-acc-skripsi, commit `86988ef`. See `git diff d33b26d 86988ef`.
Spec sections 27 and 2 (D208-D210) cover this round.

You reviewed d33b26d (`reviews/2026-11-28-s35-part1-code-second-opinion.md`). Every finding was applied. The user
decided:
- **D208**, a frozen rule for quoting the mixed model. On the dry-run data it refuses, for all four reasons.
- **D209**, strict and loose type-drift matching, both reported. The loose rule: field and kind within 4 words in
  one clause, with clause ends `. ; ! ? : …` and newline, optional quotes, backticks or parentheses, and no
  negation.
- **D210**, the D190 hash covers only the exact part of `primary.json`. The t intervals and MDDs are compared at a
  relative 1e-9.

A fresh-context check of the delta found 2 Medium and 6 Low issues, all applied. The Mediums:
- a NaN VB-MAP gap had passed D208;
- the hash test did not catch a whole-file hash.

Please focus on:
1. Is `llm_goodput` now the same definition as the background goodput (`metrics._completed_ok`)?
2. Is the loose matcher sound: `mechanism._loose_type`, `NEGATION` and `CLAUSE_END`? Try adversarial strings.
3. D210: is anything scipy-derived still inside `exact_part`, or anything exact left out?
4. Anything in the new tests that is wrong or tautological.

If you find only wording issues, say so. The next step is Part 2.
