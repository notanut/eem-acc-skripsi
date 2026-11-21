# Second opinion on Stage 3 design r6 (external LLM, brought by the user 2026-11-21)

_Archived summary of the user's doc "Stage 3 Design r6 Review" (claude.ai artifact 4sLG6v1cXYFF7Mry2MLmrM), reviewing
commit 462540f. Wording condensed; every finding kept. Applied in design r7 (D58, D59 decided by the user)._

**Verdict:** close to approvable; fix D57 overflow rule (H1) and the reflection cap (M1). Nothing blocks S3.1-S3.3.
Checked: the constants reproduce from the committed JSON; the 320-token action cap leaves room (natural replies at most
105 tokens; the 160s are probes); the two-process correction (no MPS, kernels take turns) is right.

- **H1:** cutting tasks per stream changes the NI test (at 16 tasks one task = 0.0625 counts against NI), shortens the
  learning curve and tightens the exposure rule, while descriptive arms survive. Fix: margin in task units ("one task
  per stream"); extend the cut list before D57: (8) drop RQ3 sensitivity, (9) drop C2+W=4, (10) drop F and A1, (11)
  EEM selection on 3 dev streams; early-stop rule (analyse only streams complete for all primary conditions; report
  planned vs achieved n).
- **M1:** reflection cap 96 with 60 kept wastes tokens (greedy: first 60 identical), only in memory arms; the 192
  switch can change nothing; C1 notes without k_t could carry more reflection. Fix (a) back to 60, report hits, drop
  the switch; or (b) keep 96 and store 90 (note 150, budget k x 150), C1 capped at 90.
- **M2:** canary may miss long-prompt and snapshot code paths; resumed units replay the cache. Fix: reference set up
  to ~5,000 tokens on the adopted path; resume canary (regenerate the last two cached calls).
- **M3:** pilot rules judged on C0/C2 at k = 3 only. Fix: apply both rules to the selected configuration's dev runs;
  report C1's rates from the test.
- **M4:** project the budget with and without the snapshot; include session overheads, discarded caches after an epoch
  change, and the grammar-always case.
- **Low:** adopt two processes only with >=20% gain; recognise returning platforms by comparing a failed canary with
  earlier epochs' references.
