# Second-opinion review request: S3.4 spec (the automatic pilot chain), r2

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`), file `docs/s34_spec.md`. Report High / Medium /
Low with the spec section and a concrete scenario; separate clear-cut fixes from decisions for the user.

Context: S3.3 is complete (evidence `docs/evidence/s33b_session2_2026-11-25.json`: ~3.1 s per fresh LLM call; C0 ~8
and C2 ~16 LLM calls per task). S3.4 implements the pre-registered dev rules of `docs/stage3_design.md` (sections 6,
8, 9, 10): docs-wording choice, C0/C2 pilot with the fallback/truncation switches and the C0 acceptance range,
recalibration (two allowed knobs), RQ3 wiring check, the 60 GPU-hour budget gate with the pre-registered cut order,
the freeze, and the EEM selection with D60. The user chose an automatic chain over many Colab free-tier sessions
(D154-D164, spec section 13). Some r2 choices were made by me under the user's delegation and are marked
[delegated].

Questions:
1. Does the spec contradict or omit any pre-registered design rule (design sections 6, 8, 9; HANDOFF D55, D57, D60,
   D65, D111)?
2. Can the automatic chain loop, deadlock, or apply a rule to the wrong data (rounds, switches, ladder, gate
   recomputation, D60)?
3. Could anything bias the EEM selection, or let dev decisions depend on test data?
4. Is the budget projection sound (wall seconds per task, conservative mapping, cut order, task count)?
5. Are the [delegated] choices reasonable?
