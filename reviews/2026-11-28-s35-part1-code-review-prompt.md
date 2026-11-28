# Second-opinion review request: S3.5 Part 1 code (the frozen analysis)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`, the commit named in the chat).
Contract: `docs/s35_spec.md`, sections 3-9 and 16; section 26 summarises this round.

**Code under review:**
- `src/eem_acc/stage3/analysis/`:
  - `data.py`, `endpoints.py`, `graphical.py`, `stats.py`, which you have reviewed before;
  - `secondary.py`, `descriptive.py`, `mechanism.py`, `report.py`, which are new.
- `tools/s35_dry_run.py`: a synthetic 171-unit campaign with the scripted agent on non-test seeds.
- `configs/s35_pins.json`.
- Tests: `tests/test_s35_secondary.py`, `test_s35_endpoints.py` and `test_s35_graphical.py`.
- Evidence: `docs/evidence/s35_dry_run_*.json`. Its numbers describe the scripted agent, not the LLM.

**Already done:**
- A fresh-context review found 2 High, 7 Medium and 8 Low issues, all applied:
  - the note-matching ground truth used full item paths;
  - the note text included the types the agent sent.
- Mutation testing against a copy: first 35 of 47 mutants survived, now 1 of 51. The survivor is equivalent on
  real payloads.
- The user decided:
  - **D204:** note precision is reported both ways, with the operation-level version as the headline;
  - **D205:** SP's "note updated after the re-drift" is reported both ways;
  - **D206:** the RQ3 metrics are reported for both the LLM and the background traffic;
  - **D207:** the dry-run SP seeds 3050-3054 are declared and retired.
- One finding from the dry run: on near-separated data, the VB and MAP fits of the mixed model disagree by up to
  2.1 logits. The model now reports its fit warnings and the largest gap.

**Questions:**
1. **Definitions.** Does each metric match its definition in spec sections 6-8 and the real payload fields?
   - Payload fields are written by `agent/loop.py` (`task_summary` and the step records), `agent/memory.py` (note
     events and final states) and `rq3.py`.
   - The definitions to check:
     - exposure attribution;
     - the repeat-exposure rule;
     - the note matcher (`truth_tokens`, `names_truth` with field-and-kind adjacency for type drifts, and
       `item_belongs`);
     - both precisions;
     - retrieval recall with the `visible` check;
     - C1 duplicate crowding;
     - SP re-drift detection via `sp_op`;
     - the completion-time estimators;
     - the RQ2 metric paths;
     - the integer TOST.
2. **Silent wrong results.** Is anything returning a plausible-looking value from an empty or mis-filtered
   selection?
   - Phase and tag filtering, the complete-stream population for C0-C3, and sensitivity units staying out of the
     primary analysis.
   - The flattened C2-private store lists.
3. **`report.primary` (the part the notebook runs on Colab).**
   - Does it use only numpy and scipy?
   - Is the canonical hash stable across machines? Floats are rounded to 10 significant digits, and the exact
     sign-flip p-values are dyadic.
4. **The mixed model.** Is the D187 implementation sound for 14 clusters? It uses VB and MAP with priors
   `fe_p = 2` and `vcp_p = 1`, and a GEE with bias-reduced covariance. Is the VB-MAP gap the right warning, or
   should Chapter IV drop the model when MAP does not converge?
5. **Tests.** What would still pass with a broken implementation?

Please classify each finding as High, Medium or Low, and give a concrete fix for each.
