I'll review the S3.5 spec r2 and the Part 1 code: first pulling `main` and reading the spec against design sections 7–11, then running the tests and probing the statistics code directly.

I've reviewed `docs/s35_spec.md` r2 and the Part 1 code at 1fafba7: 1 High, 2 Medium and 8 Low findings. The statistics code is correct, and the 41 Part 1 tests pass. The problems are in the spec's rules around blinding and the run order.

## Statistics (Q2): correct

I checked the code against implementations I wrote separately:
- **Graph updates and weights.** I wrote BMBP09 Algorithm 1 again in exact fractions. On 300 random graphs, the weights left after removing hypotheses don't depend on the removal order, and `intersection_weights` matches them.
- **Adjusted p-values.** My own closed test gives the same adjusted p and the same rejections as `reject`. That holds on those random graphs and on 400 p-value vectors for each of the RQ1 and RQ3 graphs.
- **Sign-flip test.** On 300 cases with ties and NI values on a 1/20 grid, `sign_flip_one_sided` equals exact enumeration over whole-task counts.
- **Formulas.** The weight update, the simultaneous G update, the zero rule, the closed-test p (max over intersections J of min p_j/w_j(J), capped at 1) and the MDD formula are all right.

On symmetry, the wording is right. Two refinements for the thesis text:
- Exactness needs each per-stream value to be symmetric about 0 on its own; they don't need to share a distribution.
- With bounded task counts, symmetry about −1 task is only approximate near the floor and ceiling. Under the one-sided null, the test is then conservative.

## High

**H1. D185(b) still leaves decisions after unblinding that can depend on the results** (sections 12, 14).
- **T units can be re-run.** During S, a re-pin replays every finished unit, T included. A fix to shared code (`memory.py`, `loop.py`) makes C2's T units replay differently, and on `accept` "those units are re-run at the new commit". That changes the primary data after the user and Claude have seen it.
  - This is likely: SP, A1, R and the RQ3 extras first run at test scale in S, so a fix during S is plausible.
- **The fix itself is a choice after unblinding.** Section 14 says the `Z` "waits for the user" during S, and the fix's content is chosen after seeing the primary results.
- **Stopping S early is also a choice.** With no deadline, the user can simply stop running S sessions after seeing the results.

Fix, all frozen now:
1. Make T final at AP: after AP, `adopt_repinned` never re-runs a T unit, a T unit that replays differently is reported as a deviation only, and `primary.json` from AP is final.
2. During S, allow a re-pin only to fix a failure (an X pause cause). Decide its `Z` automatically: accept if no finished S unit replays differently; otherwise accept, re-run the changed S units, and report them as changed after unblinding.
3. State a commitment to run S until the budget rule ends it.

## Medium

**M1. The "never dropped" arms can be dropped entirely** (sections 11, 13).
- Tier 1 (R, AN, A2 and SP) costs about 4 GPU-hours. If T overruns enough that tier 1 doesn't fit, S ends before it starts.
- The design says "R, AN, A2 and SP are reduced but never dropped". Its fallback is shorter streams or a user decision, and D185 forbids the user decision during S.
- Fix: pre-register that tier 1 always runs, either with a stated budget overrun or at 15-task prefixes (cut step 7) if it doesn't fit.

**M2. Progress shown at session ends still reveals per-unit costs to Claude** (sections 12, 15).
- The order within each plan is a seeded shuffle, so Claude can compute it from the code.
- The bundle gives units done and GPU-hours at each session end, and sessions end in the middle of blocks.
- That gives about 9 equations over known unit sets, enough to estimate cost per condition. Failing tasks run to MaxSteps, so cost tracks failures. This is a coarser form of the r1 H1 leak.
- Fix: bundles and printed progress carry done counts and GPU-hours only as of the last completed block, plus "block k of n in progress". The budget check still uses live numbers internally.

## Low

- **L1. The "inconclusive" flag is stricter than the design.** It uses the initial level, so a test can be flagged inconclusive and later rejected once weight is passed to it. The design's word is "unrejectable". Report two flags: below the initial level, and below α overall.
- **L2. Cut step 7 can't be honoured in a sequential stop** (D183). Stopping before tier 4 leaves no sensitivity run. The design's order would first shorten the descriptive arms to 15-task prefixes to keep sensitivity, which is impossible once tiers 1–3 have run at full length. State it in the spec.
- **L3. Two additions to the design should be stated as deviations.** The `extend` answer during T (D52 had a fixed budget), and stopping S tiers on budget (the design says no cuts "during the test phase").
- **L4. Test gaps.** I ran 17 targeted mutations; 4 survived that change behaviour:
  - S-R1's flag level set to α/4;
  - the MDD level for E2/E4 set to α/2;
  - the cap at 1 removed from the adjusted p;
  - the bootstrap seed changed.

  Add assertions for each.
- **L5. `index()` treats `phase=None` as "any phase".** If a T+S export is passed without a phase and an S unit has a primary cell's name with no tag, the call fails with a duplicate error. Make `phase` required for the primary functions.
- **L6. The paper's worked examples are still pending.** The design requires validating against BMBP09's worked examples before the freeze. My exact re-implementation lowers the risk, but get the PDF or use a published example set (e.g. the gMCP or graphicalMCP vignettes) before freezing.
- **L7. The user can still open Drive during T.** `s35/units/` and the ledger hold everything. Write the rule "don't open them during T" into the spec as part of D64.
- **L8. The D190 hash depends on numpy and scipy versions.** Pin numpy 2.4.6 (as in `t4_pins.json`) in `s35_pins.json`, have the notebook record Colab's scipy version, and use that version for the reproduction.

## Other questions

- **Q1, fidelity.** E1–E4, the one-task margin, both graphs (I checked them entry by entry), the floor rule, the completeness rule, D63 and D64 all match the design. Nothing is silently dropped; the additions are L1, L3 and L6.
- **Q3, blinding.** The whitelist, block-level progress, counts-only `X` and `Z` pauses, audit totals (they count task entities in notes, not drift learning), and the runner's output (its only per-unit lines are progress, which is replaced, and "FAILED twice") don't leak results by condition. The gaps are M2, L7 and, after unblinding, H1. B's timing reflects only the cost pooled over all conditions.
- **Q4, tiers.** Yes. I mapped each prefix of whole tiers to the configuration after the cut steps it restores (steps 10 down to 0), and every prefix gives at least the matching cut configuration. The 70 S units plus 101 T units make 171. The exceptions are M1 and L2.
- **Q5, budget.** B can't deadlock: `stop` leads to AP, and an `extend` that is still too small just raises B again. Checking only the next block is sound for T's priority, because S gets whatever is left. `setup_s` is an approximation, which is fine. The only leak is the pooled cost.
- **Q6, code.** No bugs, only L4 and L5.

## Calls you made (D182–D191)

| ID | Verdict |
|---|---|
| D182 (a) | Agree. Optionally also archive the release on Zenodo for an immutable timestamp and DOI. |
| D183 (a) | Agree, with M1 and L2. |
| D184 (a) | Agree. Show B's figures as of the last completed block (M2). |
| D185 (b) | Agree with (b), but only with H1's three rules. Without them, (a) is safer. |
| D186 (a) | Agree. |
| D187 (a) | Acceptable for a descriptive model. With 14 clusters, use GEE's `cov_type="bias_reduced"`. Check the variational fit against a MAP or MCMC fit in the dry run. |
| D188 (a) | Agree. |
| D189 (a) | Agree, plus M2's fix. |
| D190 (c) | Agree. Exact p-values are dyadic, so they hash identically; the rest needs L8. |
| D191 (a) | Agree. A shared-core refactor would change frozen S3.4 code and so trigger D188. |