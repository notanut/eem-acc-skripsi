# S3.5 spec: frozen analysis and the blinded, interleaved test phase (RQ1 and RQ3 tests, descriptive arms)

_r4, 2026-11-28: r3 plus the user's second opinion on r3 (section 23; user decisions D197-D200) and a fresh-context review of that delta (section 24; D201-D202); r4 also takes in the user's second opinion on r4 (section 25; D203). r3 was r2 plus the BMBP09 worked examples, the user's second opinion on r2 (section 21) and a fresh-context review of that delta (section 22); user decisions D192-D196. r2 was r1 plus the fresh-context review of r1 (section 19). It implements design sections 7-9 and 10
(S3.5-S3.7) of `docs/stage3_design.md`, on the S3.4 machinery in `docs/s34_spec.md` r11. User decisions are
D179-D181 (section 2). The user delegated every other choice in this round ("just do anything you recommend",
2026-11-27); those choices are marked **[delegated]** for the user to confirm or change._

## 1. Scope and cost

S3.5 has two parts, in this order:

- **Part 1: the frozen analysis (sections 3-9).** Code for every RQ1 and RQ3 test, the graphical procedure, the
  secondary and descriptive analyses and the mechanism checks. It is unit-tested and dry-run on dev and synthetic
  data. It is then committed, published as a GitHub release and frozen **before any test unit runs**.
- **Part 2: the test chain (sections 10-17).** Notebook 06 runs every test unit, primary and descriptive, as one
  automatic, blinded chain on the Colab T4, reusing the S3.4 chain machinery through subclasses.

S3.5 covers the design's milestones S3.5-S3.7 (the RQ1 and RQ3 tests, then the remaining descriptive arms). Chapter
IV comes after.

**Inputs fixed by S3.4.** The chain reads them from the S3.4 freeze and from `chain.json` on Drive, and checks them
against `docs/evidence/s34_chain_2026-11-27.json`:
- docs wording W1;
- no switch: free decoding with the grammar fallback, action cap 320;
- default knobs: 2 distractors, natural long share;
- cfg5 (k = 5, k_s = 3, theta_link = 0.65) for every memory condition;
- 20 tasks per stream, no cuts;
- platform epoch 0, one process, the fresh decoding path.

**Cost.** The last S3.4 gate (D-r2) projects 46.15 GPU-hours for this work, which is about 9 Colab sessions. With the
11.41 hours already used, that is 57.56 of the 60-hour budget.
- The 2.44 spare hours are about 9 C2-cost units or 13 C0-cost units.
- Retries and restarts come out of them.
- The run order (section 11) makes an overrun cost the least important descriptive units first.

## 2. Decisions

**By the user (2026-11-27):**

| ID | Decision | Chosen |
|----|----------|--------|
| D179 | D64 deadline | none: an early stop comes only from the budget accounting (section 13) |
| D180 | Scope | one spec, two parts; Part 1 is frozen before any test unit runs |
| D181 | Runner | reuse the S3.4 chain machinery: retries in another runtime, pauses answered from the notebook, re-pin with adoption by replay, the freeze check after a re-pin |
| D192 | (2026-11-28, second opinion H1) No decision after unblinding may depend on the results | three rules: (1) T is final at AP; a T unit is never re-run after AP; (2) during S a re-pin is allowed only to fix the cause of an `X` or a stall, under D194, D197 and D202; its `Z` is answered automatically (section 14); (3) a commitment to run S until the budget rule ends it |
| D194 | (2026-11-28, r3 review H1) A fix during S with no `X` behind it (a broken install, a cell bug, a hang) | allowed when S cannot start a unit, if it changes no behaviour module, protocol data, analysis module or the `s35-T` plan (checked by hash); if a behaviour module would change, S stops for good and this is reported |
| D195 | (2026-11-28, r3 review H1) Blinding of S | S stays blinded with the same bundle whitelist and block/tier-level progress until S ends; the descriptive results are shown only at the end |
| D196 | (2026-11-28, r3 review H2) Tier 1 at the hand-over | one more blind budget check when T is finished, after every `X` answer and retry, immediately before AP (D198): if used + reserved(tier 1) > budget, a `B` pause (`stop` or `extend`); `reserved(tier 1)` is a fixed number in `freeze.json` (per-task costs, `gap_s` per unit, one `setup_s`) |
| D197 | (2026-11-28, r3 second opinion M1) Content of a fix after AP | a re-pin after AP is accepted only if every finished unit, T and S, still replays identically. Excluded from the comparison: the units it fixes (those accepted as failed with that signature, which have no result to replay) and units already recorded as a mismatch. Otherwise it is refused, those units stay failed, and S continues after the revert re-pin (section 12) |
| D198 | (2026-11-28, r3 second opinion M2) Tier 1 after the hand-over | once the hand-over check passes, tier 1 is committed: it is exempt from the S budget stop whatever the overrun, and the actual overrun is reported; the check runs when T is finished, after every `X` answer and retry, immediately before AP |
| D199 | (2026-11-28, r3 second opinion M3) A systematic crash in S | once an error signature halts a session, later S units failing with it are accepted as failed after their first failure, without retries; if the same signature halts two sessions in a row, S ends after the current tier (D201) (reported). An adopted re-pin aimed at a signature clears its mark and its consecutive-halt count |
| D201 | (2026-11-28, r4 review M6) What "ends after the current tier" means | the rest of the current tier still runs, one attempt per unit, with no further halts ending S; then S ends. Cost is bounded by one tier (largest: tier 1, 15 units), and tier 1's commitment (D198) holds |
| D202 | (2026-11-28, r4 review M7) Fixing an `X` after AP | mandatory: one fix attempt per error signature whenever a fix that passes D197 exists, the same as for stalls (D194) |
| D200 | (2026-11-28, r3 second opinion L5) GPU-hours shown | only at T's end, at every `B` pause and at S's end; otherwise the notebook says only whether the next block or tier fits |
| D203 | (2026-11-28, r4 second opinion M1) Fixes that change the chain code after AP | refused: after AP, any change to chain code (`s35.py`, `s34.py`, `s34_platform.py`, `runner.py`) outside `MECHANICS` is refused. A refused `X` fix behaves like a D197 refusal (revert, units stay failed, S continues); a refused stall fix stops S for good (reported) |
| D204 | (2026-11-28, Part 1 code review M7) Note precision | both reported; headline `precision_op`: an item is correct if it belongs to any drift pair active on the note's operation (an operation can carry two pairs; a note's truth lists only the one that created it); also `precision_own_truth` |
| D205 | (2026-11-28, Part 1 code review M6) SP "note updated after the re-drift" | both reported: any update event (create, merge, close) on the SP operation after the re-drift, and the strict version, an update whose note names the second-stage name |
| D206 | (2026-11-28, Part 1 code review) RQ3 secondary RQ2 metrics | every metric for both traffics, labelled: LLM traffic (overload, SLO, latency excess, J_rq3, goodput) and background traffic (overload = throttle rate, SLO, goodput, J) |
| D207 | (2026-11-28, Part 1 code review) SP seeds of the dry run | the synthetic dry run used SP seeds 3050-3054 (SP has only a test-range pool; the scripted agent, never the LLM; the registered seeds 3000-3004 untouched, their drift plans unaffected). Declared here; **3050-3054 are retired**: never used for any later SP unit |
| D208 | (2026-11-28, Part 1 second opinion M4) When the mixed model may be quoted | frozen now: only if the MAP fit converged with no warning, the largest VB-MAP gap is at most 0.5 logits, and every condition's success is in (0.02, 0.98); otherwise Chapter IV reports GEE alone and says why (`secondary.quotable`) |
| D209 | (2026-11-28, Part 1 second opinion M3) Type-drift matching | both reported: strict (`field: kind`, `'field' is now kind`) and loose (field and kind within 4 words in one clause, e.g. "query must be a list"); the loose rule is checked on the S3.4 pilot notes before the freeze |
| D210 | (2026-11-28, Part 1 second opinion L1) What the D190 hash covers | the scipy-free part of `primary.json` only (exact values: p-values, counts, flags, rejections, levels, adjusted p; and deterministic numpy floats: the E2/E4 differences and means, the interaction means); the scipy-derived t intervals and MDDs are compared at a relative tolerance of 1e-8, recorded file against the unrounded recomputation (`report.primary_hash`, `report.float_mismatches`) |
| D193 | (2026-11-28, second opinion M1) Tier 1 (R, AN, A2, SP: "reduced but never dropped") | its projected cost is reserved during T: T's budget check counts it as already spent, so a shortfall pauses while still blind |

**[delegated]** (r2 applies the review's changes):

| ID | Decision | Options | Chosen |
|----|----------|---------|--------|
| D182 | How the analysis is frozen | (a) the analysis modules join the test chain's protocol files; a freeze record with their sha256 goes into every test plan's `meta`; a GitHub release `s35-freeze`, which has a server timestamp (review L9); (b) a git tag only | (a) |
| D183 | Order of the descriptive units | (a) tiers in the reverse of the pre-registered cut order, so a stop at a tier boundary leaves at least a pre-registered cut configuration; (b) arm by arm | (a) |
| D184 | Budget stop | (a) checked before each T block and each S tier against the frozen cost model; during T, a pause where the user may `stop` or `extend` (blind); during S, an automatic stop at the tier boundary; (b) a hard stop at 60 h | (a) |
| D185 | When to unblind (review H4, option b) | (a) everything at the end; (b) primary results after T, with every S rule fixed in the freeze now; (c) primary results only after S | (b), made safe by D192 and D194-D202: every S answer is automatic or mandatory (fixes under D194/D202, within D197), and S stays blinded (D195) |
| D186 | A primary unit that fails for good | (a) its stream or seed is left out for every condition (complete case), plus a sensitivity analysis counting its tasks as failures; (b) count as failures only | (a) |
| D187 | Mixed-effects logistic model (secondary) | (a) statsmodels `BinomialBayesMixedGLM.fit_vb` with crossed random intercepts for stream and template and stated priors, with GEE (`cov_type="bias_reduced"`, 14 clusters) reported alongside it; the dry run checks the VB fit against the MAP fit (`fit_map`); (b) R lme4; (c) GEE only | (a), confirmed by the second opinion with these two additions |
| D188 | Code changes between S3.4 and the test phase | (a) allowed only if every finished S3.4 unit still replays identically and no S3.4 behaviour module changes, else a pause; none is planned; (b) none at all | (a) |
| D189 | Blinding of costs and timing (review H1) | (a) no per-unit progress lines; progress and ETA only as of the last completed block or tier (section 15), with the ETA from the frozen cost model; GPU-hours only at the D200 points; (b) per-unit lines | (a) |
| D190 | Where the analysis runs (review M2, M3) | (a) all on Colab with new pins; (b) only on Claude's machine; (c) hybrid: the notebook computes only the exact primary tests (numpy and scipy, already on Colab, nothing installed) and records the hash of their canonical form in the chain log; Claude runs the full frozen analysis on the export, and its primary part must reproduce that hash | (c) |
| D191 | Code structure (review H3) | (a) a new `s35.py` with `TestChain(s34.Chain)` and `TestPlatform(s34_platform.Platform)` overriding the S3.4-specific parts (section 12); `s34.py`, `s34_platform.py` and `runner.py` stay unchanged; (b) refactor a shared core | (a) |

---

# Part 1: the frozen analysis

## 3. Data, pairing and completeness

- **Source:** the stored result payload of every test unit: per-task records (`tasks`) plus the memory logs
  (`note_events`, `final_notes`) and the step records the mechanism checks need (section 8). No cache is read.
  - The export (section 16) holds exactly these payloads.
  - The analysis rebuilds each unit's stream with `sample_stream` and the unit's stream parameters, and refuses a
    payload whose `spec_sha256` differs.
- **Pairing:** RQ1 by test stream (seeds 2000-2013); RQ3 by test seed (4000-4009); SP by SP seed (3000-3004).
- **Per unit:**
  - the success count, as an integer number of solved tasks out of the tasks run; tasks unfinished at T_max (RQ3)
    count as not solved and are reported next to every success table;
  - per task: API calls (gateway attempts including `get_api_docs`), LLM calls, prompt and output tokens, steps,
    modelled LLM time, format errors, budget exhaustion, drift and agent errors.
- **Completeness (design section 8, early stop):**
  - an RQ1 stream enters the primary tests only if C0, C1, C2 and C3 are all done;
  - an RQ3 seed enters only if all four 2x2 cells are done;
  - planned and achieved n are reported with every test.
- **A unit that failed for good (D186)**, after its retry and the automatic or chosen `accept_failed`:
  - it is left out with its stream or seed, as above, and reported by condition;
  - sensitivity analysis: the same tests with that unit's tasks counted as not solved;
  - a failure tied to one condition, for example a crash that only C2's longer prompts trigger, is a stated
    limitation.
- **Reported per condition:** the platform epoch of every unit, restarts, units marked "reproducible by replay
  only", and adoptions (design section 9). No unit mixes epochs, and the analysis does not stratify by epoch.

## 4. RQ1 primary endpoints

The endpoints are E1-E4 of design section 8. n_t is the number of tasks in the stream (20). For success, the
per-stream difference is computed from integers, d_s = (solved_C2 - solved_comp) / n_t. The NI value is
x_s = (solved_C2 - solved_comp + 1) / n_t, so a tie (C2 exactly one task worse) is exactly 0 (review M7).

| Hypothesis | Comparison | Per-stream value | H1 (one-sided) |
|------------|------------|------------------|----------------|
| NI1 | C2 vs C0, success | (solved_C2 - solved_C0 + 1) / n_t | mean > 0 |
| S1  | C2 vs C0, success | (solved_C2 - solved_C0) / n_t | mean > 0 |
| E2  | C2 vs C0, API calls | mean API calls of C0 minus C2, over the tasks of that stream solved by **both** | mean > 0 (C2 fewer) |
| NI3 | C2 vs C1, success | (solved_C2 - solved_C1 + 1) / n_t | mean > 0 |
| S3  | C2 vs C1, success | (solved_C2 - solved_C1) / n_t | mean > 0 |
| E4  | C2 vs C1, API calls | as E2, with C1 | mean > 0 |

- **Test:** the exact one-sided paired sign-flip test (`stats.sign_flip_one_sided`).
  - p = the share of all 2^n sign patterns e with mean(e * x) >= mean(x) (tolerance 1e-12; n <= 20).
  - The test decides; the interval is reported only.
  - The test is exact under exchangeable signs: under the null, each tested per-stream value must be symmetric
    about 0 on its own (the values need not share a distribution). That means symmetry about the margin for NI, and
    about 0 for superiority. Section IV states this assumption (review M7), as Stage 2 did for TOST.
  - With bounded task counts, symmetry about -1 task holds only approximately near the floor and the ceiling;
    under the one-sided null the test is then conservative (second opinion).
- **Ties:** a zero changes no pattern's mean, so the smallest attainable p is 2^-(untied n).
  - The number of untied streams is reported with every test.
  - **Two flags (second opinion L1).** Every test's p enters the graph whatever its flags.
    - `below_initial_level`: the smallest attainable p is above the level the test starts with. The S hypotheses
      start at level 0, so for them the flag uses the level they would inherit from their NI test (alpha/4 for RQ1,
      alpha/2 for RQ3; code review L8). A test with this flag can still be rejected once weight is passed to it.
    - `unrejectable` (the design's word): the smallest attainable p is above alpha, so no weight passed to the test
      can make it rejectable.
- **E2/E4:** a stream with no task solved by both conditions has no value and is left out of that test (reported).
  "Solved by both" restricts the comparison to tasks the weaker condition can solve; this is stated.
- **Reported for each test:**
  - the mean difference (C2 minus comparator, in its own units);
  - a 95% t interval;
  - the minimum detectable difference at the observed SD, (t_{1-a, n-1} + t_{0.8, n-1}) x SD / sqrt(n), with
    a = alpha/4 = 0.00625 for RQ1 and alpha/2 = 0.0125 for RQ3. The RQ3 figure is an addition to the design
    (review L2).

## 5. The graphical procedure [BMBP09]

- **Algorithm 1 of BMBP09** (`graphical.py`), with weights w as fractions of alpha (sum <= 1) and a transition
  matrix G (g_jj = 0, row sums <= 1):
  1. Find a remaining j with p_j <= w_j x alpha and w_j > 0. If there is none, stop.
  2. Reject H_j.
  3. For every remaining l: w_l <- w_l + w_j g_jl.
  4. For remaining l != k: g_lk <- (g_lk + g_lj g_jk) / (1 - g_lj g_jl) if g_lj g_jl < 1, else 0, all computed
     from the G of before this step (a simultaneous update, review L1); set g_ll = 0.
  5. Remove j and go back to step 1.

  The final rejection set does not depend on the order (BMBP09). The code takes the smallest index, and a test
  checks every order. A hypothesis with no p-value is never rejected and keeps its weight.
- **Adjusted p-values**, from the procedure's closed-testing representation.
  - For each hypothesis i, the adjusted p is the maximum, over every intersection J that contains i, of
    min_{j in J} p_j / w_j(J). Here w_j(J) is the weight left on j after every hypothesis outside J has been removed
    with the same update (the graph's intersection weights).
  - This is exact (no bisection). A test checks that it rejects the same hypotheses as the sequential procedure on
    random graphs.
  - Adjusted p-values are reported only. The decision is the rejection set at alpha = 0.025.
- **RQ1 graph** (family-wise one-sided alpha = 0.025), in the order NI1, S1, E2, NI3, S3, E4:
  - initial weights 1/4 each on NI1, E2, NI3 and E4, and 0 on S1 and S3;
  - transitions NI1 -> S1 (1); NI3 -> S3 (1); S1 -> E2, NI3, E4 (1/3 each); S3 -> E4, NI1, E2 (1/3 each);
    E2 -> NI1, NI3, E4 (1/3 each); E4 -> NI1, NI3, E2 (1/3 each).
- **RQ3 graph** (one-sided alpha = 0.025), in the order NI-R1, S-R1, R2:
  - weights 1/2, 0, 1/2;
  - transitions NI-R1 -> S-R1 (1); S-R1 -> R2 (1); R2 -> NI-R1 (1).
- **Validation before the freeze:**
  - Property tests:
    - with equal weights and full transitions, the result equals Holm's (rejections and adjusted p);
    - the fixed sequence and the fallback procedure;
    - independence from the rejection order (every permutation);
    - equivalence with the closed test;
    - invariants after every step.
  - The RQ1 and RQ3 graphs are checked on p-value vectors whose rejection sets are worked out by hand in the test.
  - **The paper's worked examples** (the PDF attached by the user, 2026-11-28) are in
    `tests/data/bmbp09_examples.json` with page numbers, and a test reproduces each one's rejections, their order
    and the local levels:
    - Figure 4: Bonferroni-Holm, including the adjusted p-values 0.04, 0.055 and 0.036 of section 3.1;
    - Figures 9 and 10: Holm as gatekeeper, with ε-edges;
    - Figure 2: the gatekeeping procedure of [7];
    - Figure 12: its improved version.
  - The paper's sentence on Figure 2 (p. 598) says "only H1, H2, and H3" are rejected. Its own walk-through two
    sentences earlier gives H1, H3 and H4, so the test encodes the walk-through and records why.
  - BMBP09 Algorithm 2 (sequential adjusted p-values) is implemented as an independent check: it equals the
    closed-test adjusted p on random graphs and on both frozen graphs.
  - The update formulas also match the restatement in arXiv:2402.13719, section 1.2.

## 6. RQ3 tests

These follow design section 8. n_r is the number of LLM tasks per run (4 agents x 5 = 20).

| Hypothesis | Comparison | Per-seed value | H1 |
|------------|------------|----------------|----|
| NI-R1 | C2 vs C0 under ACC on | (solved_{C2+acc} - solved_{C0+acc} + 1) / n_r | mean > 0 |
| S-R1  | same | (solved_{C2+acc} - solved_{C0+acc}) / n_r | mean > 0 |
| R2    | ACC on vs no window, within C2 | (solved_{C2+acc} - solved_{C2+none}) / n_r | mean > 0 |

- **Tests:** the same sign-flip test, ties rule, intervals and minimum detectable difference as RQ1, with the RQ3
  graph of section 5. NI-R1 at alpha/2 needs at least 7 untied seeds of 10.
- **Interaction:**
  - per seed, (C2 - C0 | acc) - (C2 - C0 | none), reported as a mean with a 95% t interval and no test;
  - **floor rule:** if mean success under no window is below 0.2 for both C0 and C2, the interaction is reported
    as uninformative.
- **Secondary (Q4)**, each a per-seed paired difference with a 95% t interval:
  - **Virtual task completion time** (review L5), two versions:
    - the mean over tasks that finished (an unfinished task has no completion time), with the counts of unfinished
      and never-started tasks per cell next to it;
    - a censored version, with unfinished tasks counted at their censoring time (`completion_time` as the payload
      writes it, D120), as a sensitivity check;
  - fixed W = 4 vs ACC, within C2;
  - per cell, the RQ2 metrics: overload rate, SLO violation, goodput and J (0.5, 0.5);
  - drift errors observed and notes created.

## 7. Secondary and descriptive analyses (outside the graphical procedure)

- **C1 vs C0** (cross-task memory): success, and API calls in the E2 style; a two-sided exact sign-flip p, unadjusted,
  with a 95% t interval.
- **C3 vs C2** (TOST, margin 1/n_t): `tost_sign_flip` with one-sided 0.025 per side, and the 95% t interval as the
  paired-t TOST check.
- **EEM's own cost:** LLM calls and prompt tokens per solved task, per condition. For C1, how often duplicate
  reflections fill its top-k (design section 7).
- **Ratio-of-sums version of E2/E4:** the sum of API calls over the jointly solved tasks, pooled over streams, as a
  ratio C2/C0 and C2/C1, with a stream-level percentile bootstrap 95% interval (10,000 resamples, seed 0).
- **Mixed-effects logistic model (D187):**
  - success ~ condition x position (0-19, centred), with crossed random intercepts for stream and template, on C0-C3
    over the complete streams;
  - fitted with `BinomialBayesMixedGLM.fit_vb` (priors: `fe_p` = 2, `vcp_p` = 1, statsmodels' defaults stated);
  - reported as posterior means and 95% intervals, with the caveat that variational intervals are too narrow;
  - GEE (logit, exchangeable, clustered by stream) with the same fixed effects, reported alongside;
  - it addresses the expectation that the C2 - C0 gap grows along the stream, descriptively.
- **Per drift type and per error style (review L6):** drift type and error style belong to an operation, so these
  are computed per **exposure** (a call to a drifted operation): the share of exposures whose first call succeeds,
  per condition, drift type and style.
- **Learning curves:** success by position, per condition.
- **Other design section 8 items:**
  - format errors, budget exhaustion and the drift-versus-agent error split, per condition;
  - the share of all calls that hit a drifted operation (design section 5).
- **Descriptive arms** (F, AN, R, A1, A2 and C2-OU on streams 2000-2004):
  - each arm's success and API calls, next to C0, C1, C2 and C3 on the same streams;
  - paired differences against C2, with t intervals and no test;
  - AN's prompt tokens.
- **SP** (C0 and C2 on SP seeds 3000-3004):
  - success before and after the re-drift;
  - after the re-drift, the first-call outcome at each exposure to the re-drifted operation;
  - whether C2's note on that operation was updated after the re-drift.
- **RQ3 extras:**
  - private store vs shared store (C2+acc, seeds 4000-4004);
  - sensitivity: the 2x2 with LLM time / 3 (seeds 4000-4004), with the same differences as R1 and R2, as estimates
    only.

## 8. Mechanism checks (not endpoints)

These are computed from fields the loop already writes; no loop change is planned (D188).

- **Repeat-exposure drift error rate (review L6):** among exposures to a drifted operation after its first exposure in
  the stream **where the operation was called** (a `first_call` entry exists for it in that task), the share whose
  first call failed with the drift error. Reported per condition.
- **Note correctness (review M5):**
  - each note carries `truth`: the ground-truth drift pair (operation, drift type, the old and new field, type,
    enum value or operation id). The check asks whether the note names the new field, type, enum value or operation
    id;
  - the matcher (`mechanism.note_names_truth`) uses the drift catalogue, not the scrub audit's entity list;
  - it is applied to each `note_events` entry before and after its outcome update, and to `final_notes`;
  - it covers C2, C3, C2-OU, AN and A2 (notes with k_t and items) and A1 (raw text, same matcher). C1 and F store
    reflections only, so for them the matcher is applied to the reflection text, reported separately.
- **Note precision:** the share of the items in "Last working change" lines that belong to the ground-truth drift
  (C2, C3, AN, A2; C2-OU has no outcome update and so no such lines).
- **Retrieval recall:** when an operation with an existing note (`notes_on_op`) is called, the share of calls where
  that note was among the retrieved notes (`retrieved`), with `visible` as the denominator check.
  - Computed for C1, C2, C3, A1, A2 and C2-OU.
  - F is reported as "the note was among the last k" (review L6).
  - AN shows every note.

## 9. Freeze procedure (D182, D188, D190)

1. **Modules** in `src/eem_acc/stage3/analysis/`:
   - `graphical.py`, `stats.py`, `data.py` and `endpoints.py` (written with r2);
   - `secondary.py`, `descriptive.py`, `mechanism.py` and `report.py`;
   - `report.py` writes `primary.json` and `full.json` (section 16).
   - Stage 2's `analysis.py` is reused by import (`tost_sign_flip`) and not changed.
2. **Canonical form:** every float is rounded to 10 significant digits; keys are sorted; the result is hashed with
   sha256. `primary.json` uses only numpy and scipy, and exact arithmetic except for the t intervals and MDD.
3. **Dry runs** (CPU, before the freeze; outputs go to `docs/evidence/`; no test result is involved):
   - the whole analysis on a synthetic test campaign: a ScriptedBackend run of the real plans, every condition,
     with streams shortened where needed for time;
   - the RQ1 part on the S3.4 dev data (C0 and C2 on pilot streams 1002-1006, results already known), to show the
     code reads real payloads.
   - **Done 2026-11-28 (synthetic campaign):** `tools/s35_dry_run.py` writes `docs/evidence/s35_dry_run_*.json`.
     - It ran all 171 units with the test-phase structure (plans `s35-T` and `s35-S`, every arm, tags and params)
       on seeds no test unit uses, so no held-out stream was touched by any agent: RQ1 dev 1900-1913, RQ3 dev
       5090-5099, and SP 3050-3054. SP has only a test range; the registered SP seeds are 3000-3004.
     - The scripted agent ran it in about 2 minutes on CPU. Its numbers describe the scripted agent, not the LLM.
     - The primary hash is identical across two separate runs.
     - **Found and fixed:**
       - the descriptive arms' streams were hard-coded and silently emptied the table on other seeds; they are now
         derived from the data;
       - pandas 3's string dtype broke patsy;
       - on near-separated data (the scripted C2 almost always succeeds), the VB and MAP fits of the mixed model
         disagreed by up to 2.1 on the logit scale, and MAP's Laplace step reported non-convergence. The model now
         reports its fit warnings and the largest VB-MAP gap with the estimates. It stays descriptive, and Chapter
         IV quotes it only together with those two figures.
   - **Pending:** the run on the S3.4 dev data. It needs the S3.4 unit payloads from Drive (`s34/units/`), which
     are not in the repository.
   - **Seeds:** the dry run's SP seeds 3050-3054 come from the SP test range; they are declared and retired (D207).
4. **Environment:**
   - Nothing is installed on Colab for the analysis. The primary part uses numpy and scipy, which the session
     already has; the notebook records their versions.
   - statsmodels is used only in Claude's workspace, pinned in a new `configs/s35_pins.json`, which is not an S3.4
     protocol file (review M2). The same file pins numpy 2.4.6, as in `t4_pins.json` (second opinion L8).
     Claude's workspace now has exactly these versions, and the dry-run evidence is produced with the pins checked
     strictly.
   - The notebook records Colab's scipy version with `primary.json`. Claude's reproduction (section 16) uses that
     scipy version. The exact sign-flip p-values are dyadic, so they hash identically on any version.
   - `configs/t4_pins.json` is not changed.
5. **Freeze:**
   - the analysis modules, `s35.py` and `notebooks/06_s35.ipynb` join the test chain's protocol files, which are the
     S3.4 list plus these;
   - the commit is published as GitHub release `s35-freeze`, before the first test session;
   - the test chain writes `s35/freeze.json` on Drive, holding:
     - the commit;
     - the sha256 of every protocol file;
     - the S3.4 settings read;
     - the S3.4 evidence sha256;
     - the S-phase rules (D185, D192, D194-D202) and `reserved(tier 1)` (D193, D196);
   - the freeze's sha256 goes into the `meta` of both test plans before any test unit runs.
6. **After the freeze**, a change to any protocol file, the analysis included, pauses with `Z` (section 14), as in
   S3.4.

---

# Part 2: the test chain

## 10. Units (review H2)

- Units are built in `s35.py`. The order comes from `runner.test_phase_units` with a fixed shuffle `plan_version=1`,
  which is never the plan-file version, since a re-pin bumps that.
- Each unit is then rewritten with `dataclasses.replace(u, phase=<plan phase>, params=<S3.4 params: wording W1, no
  switches, k = 5, k_s = 3, theta_link = 0.65>, stream=<S3.4 knobs>)`. The phase must equal the plan's, because the
  runner's failure, retry and halt bookkeeping is keyed by `u.phase`.
- A test asserts that every test unit carries cfg5 and the plan's phase.

| Block | Units | Kind / seeds | Conditions |
|-------|-------|--------------|------------|
| RQ1 primary | 56 | rq1 test 2000-2013 | C0, C1, C2, C3 |
| C2-OU | 5 | rq1 test 2000-2004 | C2-OU (in the primary blocks, D63) |
| RQ3 2x2 | 40 | rq3 test 4000-4009 | C0, C2 x acc, none |
| RQ1 descriptive | 25 | rq1 test 2000-2004 | F, AN, R, A1, A2 |
| SP | 10 | sp test 3000-3004 | C0, C2 |
| RQ3 W = 4 | 10 | rq3 test 4000-4009 | C2 + fixed4 |
| RQ3 private store | 5 | rq3 test 4000-4004 | C2-private + acc |
| RQ3 sensitivity | 20 | rq3 test 4000-4004 | 2x2, `llm_time_scale` = 1/3 (tag `sens`) |

These 171 units match the S3.4 gate's `workload()` with no cuts. Units with a tag (`sens`) never enter the primary
index.

## 11. Plans and run order (D63, D183)

- **Plan `s35-T` (primary).** Blocks of 2 RQ1 streams, each with C0-C3 plus C2-OU on the first 5 streams, followed by
  1 RQ3 seed with the 2x2, in a seeded shuffle within each stream or seed. `blind` is true.
  - A **block** for the budget check (section 13) is one such group of 2 streams and 1 seed.
  - With 14 streams and 10 seeds there are 7 such blocks. The **tail block** is the 3 remaining RQ3 seeds
    together.
- **Plan `s35-S` (descriptive)**, in tiers that are the reverse of the cut order. Each tier is what one more cut step
  would have removed, so a stop at a tier boundary leaves a configuration at least as large as the pre-registered
  cut configuration of that step (review L3):

  | Tier | Units | Restores cut step |
  |------|-------|-------------------|
  | 1 | R, A2, AN on 2000-2002; SP (C0, C2) on 3000-3002 | survives every cut |
  | 2 | F, A1 on 2000-2002 | 10 |
  | 3 | W = 4 on 4000-4004 | 9 |
  | 4 | sensitivity on 4000-4002 | 8 |
  | 5 | sensitivity on 4003-4004 | 6 |
  | 6 | SP on 3003-3004 | 5 |
  | 7 | F, AN on 2003-2004 | 4 |
  | 8 | A1, A2 on 2003-2004 | 3 |
  | 9 | R on 2003-2004 | 2 |
  | 10 | W = 4 on 4005-4009 | 1 |
  | 11 | private store on 4000-4004 | 0 |

  - Cut step 7 (15-task prefixes for the descriptive arms) needs no tier, because every unit runs full length.
    Cut step 11 concerns the dev selection, which is already done.
  - **Stated deviation (second opinion L2).** In a sequential stop, cut step 7 cannot be honoured. The design's cut
    order would shorten the descriptive arms to 15-task prefixes before it removes sensitivity seeds, and once tiers
    1-3 have run at full length that is no longer possible. A stop before tier 4 therefore leaves no sensitivity
    run, where the cut order would have kept 3 seeds.
  - Within a tier, units are grouped by seed in a seeded shuffle with key `s35-S|tier<k>|<seed>`.
- `s35-S` starts only after `s35-T` is finished: every unit done, or failed and accepted.

## 12. The chain (D181, D191; review H3, M6)

- **`src/eem_acc/stage3/s35.py`.** `TestChain(s34.Chain)` and `TestPlatform(s34_platform.Platform)`. The S3.4 code
  is not changed. The overrides:
  - the phase prefix `s35-`: `adopt_repinned`, `Platform.prefix`, `verify_finished` and `freeze_extra` take it as a
    class attribute;
  - the freeze path `s35/freeze.json` and the protocol snapshot `s35/protocol/<commit>.json`;
  - the protocol list: a function returning the S3.4 list plus the S3.5 files, also used by `_lists_at` (read from
    `s35.py` at that commit);
  - `FREEZE_HASHES` (S3.4's plus the analysis sources), with the gate on the freeze entry ("P" instead of "E");
  - `verify_log`: the S3.5 log has no outcome-based decisions, so its re-check compares each plan's unit list and
    `meta` (not the plan sha256, which a re-pin changes) and the freeze;
  - `adopt_repinned` keeps `blind = True` and `meta` when it writes a new plan version;
  - `_units` returns only done, failed and not-done sets, without outcome metrics;
  - the progress output (D189): the Runner's per-unit `progress_line` is replaced by a no-op; progress prints as of
    the last completed block or tier (section 15);
  - the budget check (section 13) runs inside the chain between blocks and tiers;
  - `adopt_repinned` (after AP) decides D197 over **all** plans before it writes any adoption record or plan
    version, and writes nothing if it refuses (r4 second opinion H1).
- **Runner facts that shape the implementation** (r4 second opinion M3, checked against `runner.py`):
  - `last_failure` is local to one `run_phase` call. So the D130 halt does not carry across calls, and two failures
    in a row across a block or tier boundary do not halt. `TestChain` keeps its own consecutive-failure state
    across calls, so the halt holds within a session.
  - `run_phase(retry_failed=True)` retries every failed unit of the phase. D199's per-signature rule therefore
    leaves `retry_failed` off: the chain marks the retries it wants with `Runner.mark(phase, uid, "retry")`.
  - Units halted by the "two failures in a row" rule are re-enabled, not failed (a `halted` manifest line).
    `accept_halted=True` accepts **every** halted unit of the shard, not one signature's. So after a halt in S,
    `TestChain` marks the D199 signature's units `failed` itself (`Runner.mark`), and does not rely on
    `accept_halted` per signature.
  - The "diverged twice" `RunHalted` writes no manifest line and has no error signature; `accept_halted` never
    accepts that unit. `TestChain` marks it `failed` with signature `divergence` before calling `run_phase` again,
    so the D201 loop cannot re-run it without end (r4 delta review M1).
  - D201 ("no further halts end S"): `TestChain` catches `RunHalted`, marks the halted units `failed` as above, and
    calls `run_phase` again in the same session, until the current tier's units are all done or failed.
  - `max_units` counts units attempted in plan order, and units marked for retry from earlier blocks come first. A
    block or tier call therefore loops until every unit of that block or tier is done or failed.
  - After AP no new T plan version is written, so `run_phase("s35-T")` at a later commit raises
    `TestPhaseCommitMismatch`. That enforces "T is never re-run". Progress, the export and `verify_log` read T
    results directly (`load`), never through `run_units`.
- **Steps:**
  - **P, preflight (no GPU):**
    - read and check the S3.4 inputs (section 1);
    - D188: replay every finished S3.4 unit with the current code (`verify_replay`) and compare the S3.4 behaviour
      modules and protocol data with S3.4's last accepted snapshot. **Expected result: no difference**, because S3.5
      adds files only. Any difference pauses;
    - write `s35/freeze.json`;
    - write both plans.
  - **T:** runs `s35-T` block by block.
  - **AP, primary analysis:** the notebook computes `primary.json` (section 16) and records its canonical hash in
    the chain log. It runs on whatever is complete when T finishes or is stopped.
  - **S:** runs `s35-S` tier by tier, under the fixed rules of D185, D192 and D194-D202 (section 14).
  - **G, done.**
- **Reused from S3.4:**
  - `chain.json` as a write-once log through `t4.Sections`;
  - `USER_DECISION` with `EXPECT_LOG`;
  - the retry rule (first failure -> `RETRY` in another runtime; second failure -> `X`);
  - `REPIN` with adoption by replay (D166) and the header rule (D173);
  - `MECHANICS`, the `Z` pause, the per-session backend check (D176, D178), the canary and epochs;
  - the lease: one runtime at a time; with a second account, never both at once (D164).
- **Re-pin semantics:** as in S3.4, `adopt_repinned` writes a new plan version at the new commit, keeping the units
  and `meta`, and replays every finished unit.
  - Units that replay identically are adopted.
  - **Before AP:** a finished test unit that does not replay identically is a `Z` that names how many units
    changed, never which ones or how (blinding). On `accept`, those units are re-run at the new commit; the old
    results are archived and reported as a deviation.
  - **After AP (D192):**
    - **T is final.** A T unit is never re-run, and `primary.json` from AP stays the primary result.
    - **T is never re-planned (r3 second opinion M4).** After AP, `adopt_repinned` skips `s35-T` and writes no new T
      plan version. It replays the T units to check D197: a mismatch refuses the re-pin. "Deviation" applies only to
      a T unit that mismatches without a re-pin (section 14, `Z (after AP, no re-pin)`). The "T plan" in the refusal
      rule below means its unit list plus `meta`, not the plan file.
    - **A re-pin during S is allowed only to fix the cause of an `X`** (units that failed twice, D192), or, with no
      `X`, when S cannot start a unit (D194).
      - It is refused if it changes the hash of any analysis module, any protocol data file or the T plan's unit
        list and `meta`.
      - **D197:** it is accepted only if every finished unit, T and S, still replays identically. Excluded from
        the comparison are the units it fixes, i.e. those accepted as failed with that signature (they have no
        result to replay), and units already recorded as a mismatch. Otherwise it is refused, those units stay
        failed, and S continues.
      - **D202:** for an `X` after AP, one fix attempt per error signature is mandatory whenever a fix that passes
        D197 exists.
      - **D194:** a stall fix must not change a behaviour module. One attempt per stall cause is mandatory whenever
        an environment-only fix exists (r3 second opinion L2). **If the stall fix is refused for any reason**
        (a behaviour change, D197, D203 or a hash rule), **S stops for good, and this is reported** (r4 review H1).
        - Notebook 06 is protocol data, so a stall fix in its cells is refused.
        - For a Colab image change, the escape that changes no code is Colab's "Runtime version" setting: choose
          the version the freeze recorded (r4 second opinion L2).
        - A backend change brought in by a stall re-pin must pass D197. A backend change without a re-pin is
          accepted automatically (section 14). The difference is stated (r4 second opinion L5).
      - **D203:** after AP, a change to chain code (`s35.py`, `s34.py`, `s34_platform.py`, `runner.py`) outside
        `MECHANICS` is refused, since it could change the S rules without changing any unit's replay.
    - **`REPIN_FOR` (r4 second opinion H1).** After AP, a re-pin must name what it fixes in a new notebook setting
      `REPIN_FOR`: an error signature (`X`, D202) or a stall cause (D194). "The units it fixes", "one attempt per
      signature" and the D199 reset are computed from it. A re-pin without `REPIN_FOR` after AP is refused.
    - **The refused state (r4 second opinion H1).** The re-pin is already recorded in `chain.json` before the chain
      can decide (`t4.Sections(repin=True)` replaces the stamp), and a re-pin back to an older commit is not allowed.
      So a refusal does not simply leave S running:
      - the chain records `refused` with the commit, and runs **no unit** at a commit that has not been accepted;
      - the way out is a **revert commit**: a new commit that descends from the refused one and whose protocol files
        equal those of the last accepted commit. The user re-pins to it with `REPIN_FOR` = `revert`. D197 then
        passes trivially. This is a mandatory action, not a choice;
      - **a revert is not a fix (r4 delta review H1):** it clears no D199 mark or halt count and re-runs nothing. It
        uses up D202's one attempt for that signature and counts as the `no_fix` record for the start gate. S
        continues with the cause's units failed;
      - an `X` fix refused under D197, D203 or a hash rule: revert, then S continues. A refused stall fix (D194)
        stops S for good after the revert, for the close-out (section 16).
    - **Start gate after a signature's first halt or `X` in S (r4 second opinion M2; r4 delta review H2).** Later
      halts of the same signature need no new record. The next S session starts only once one of these is
      recorded:
      - a fix re-pin for that signature;
      - Claude's written finding that no fix passes D197, D203 and the hash rules, entered as
        `USER_DECISION {"action": "no_fix", "signature": ...}`;
      - a revert (above).

      D199's consecutive-halt count includes the session that first halted, so "two in a row" means that session
      and the next one. Which units run therefore never depends on whether a session was started before or after a
      fix.
    - **"Two halts in a row" (r4 second opinion L1):** consecutive S sessions in which that signature halted; a
      session without that halt resets the count.
    - **Its `Z` is answered automatically with `accept`.** Under D197 no finished unit changes. After a re-pin that
      fixes an `X`, every S unit accepted as failed with that error signature is re-run once, automatically. Those
      re-runs happen only while S is running, before the next tier's budget check, and count with that tier's cost.
      Nothing is re-run after S has ended (r3 second opinion L3). An adopted re-pin aimed at a signature clears that
      signature's D199 mark and its consecutive-halt count.
    - **A bug in the frozen analysis** found after AP is fixed outside the chain. Both the frozen and the corrected
      numbers are reported, as a deviation (r3 second opinion L4).
    - **Commitment:** S is run until the budget rule ends it or every tier is done. Not running further S sessions
      is not an option the user takes after seeing the primary results.

## 13. Budget and early stop (D179, D184; review M1)

- **Accounting:** the ledger's GPU-hours since `benchmark_start`, as in the S3.4 gate, read live before each block or
  tier.
- **Projected cost of a block or tier:**
  - the sum over its units of (tasks x the S3.4 gate's per-task cost for the unit's cost class), plus `gap_s` per
    unit, plus `setup_s` if the block or tier would not fit in the current session;
  - the per-task costs are the frozen numbers of the last S3.4 gate, never a new measurement;
  - cost classes: C0 and R -> c0; C1, C2, C3, C2-OU, F, A1, A2 -> c2; AN -> an; SP -> sp (the mean of c0 and c2);
    every RQ3 unit -> rq3;
  - retries of earlier units count with the next block's cost.
- **Tier 1 reservation (D193):** during T, the check counts `reserved(tier 1)` (R, AN, A2 on 3 streams and SP on 3
  seeds: per-task costs, `gap_s` per unit and one `setup_s`, a fixed number in `freeze.json`) as already spent.
- **Hand-over check (D196, D198):** when T is finished, after every `X` answer and retry and immediately before
  AP, one more blind check: if used + reserved(tier 1) > budget, a `B` pause. Once it passes, tier 1 is committed:
  it is exempt from the S budget stop whatever the overrun, and the actual overrun is reported.
- **The ways tier 1 can still be lost:**
  - a blind `stop` at any `B` (the user's blind choice);
  - a stall that cannot be fixed (D194);
  - units accepted as failed.

  Each is reported against the design's "never dropped". D199 ends S only after the current tier (D201), so it
  cannot cut tier 1 short.
- **Before each T block:** if used + projected(block) + reserved(tier 1) > budget, the chain pauses with `B`:
  - `stop`: T ends, AP runs on the complete streams and seeds (section 3), and S does not run;
  - `extend` with a new budget in hours: the user buys more compute. This is blind, because no outcome has been
    shown during T. It is recorded.
- **Before each S tier:** if used + projected > budget, S ends automatically at that tier boundary (D184), except for
  tier 1 once the hand-over check has passed (D198). After AP the user has seen the primary results, so no S decision
  is left to the user.
- There are no cuts during the test phase and no deadline (D179).
- **Stated deviations from the design (second opinion L3):**
  - `extend` during T is new; D52 had a fixed budget;
  - the automatic stop of S at a tier boundary is new; the design says "no cuts during the test phase".

## 14. Pauses

| Paused at | Cause | Actions |
|-----------|-------|---------|
| P | S3.4 units no longer replay identically, or S3.4 behaviour code or data changed (D188) | `accept` (recorded in the freeze), or re-pin a fix |
| X (during T) | units failed again after their retry | `rerun` (one more retry) or `accept_failed` (D186) |
| X (during S) | the same | **automatic:** one more retry, then `accept_failed` (D185); after a halt with the same signature, no retry (D199). One fix attempt per signature is mandatory when a fix passing D197 exists (D202) |
| halt (during S) | the D130 "two failures in a row" halt | the session ends and the halt marks its error signature (D199); the next session continues; if the same signature halts two sessions in a row, S ends after the current tier (D201) (reported) |
| stall (during S) | a unit cannot start with no `X` behind it | a D194 environment fix (one attempt per cause, mandatory if one exists); if refused for any reason, S stops for good (reported) |
| Z (before AP) | a re-pin changed protocol material, analysis included, or a finished test unit replays differently | `accept`: an addendum; changed units re-run; a changed analysis is reported as a deviation from the pre-registration |
| Z (after AP) | a re-pin (D192 or D194) | **automatic `accept`**; under D197 no finished unit has changed (a re-pin that changes one is refused before this point) |
| refused (after AP) | a re-pin refused by D197, D203 or a hash rule | no unit runs; the user re-pins to a revert commit (mandatory, section 12); a refused stall or chain fix then stops S for good |
| gate (after AP) | a halt or `X` in S | the next S session waits for a fix re-pin or a recorded `no_fix` (section 12) |
| Z (after AP, no re-pin) | a `verify_finished` replay mismatch or the backend check | T unit: a reported deviation only. S unit: re-run once at the same commit, automatically, before the next tier's budget check and counted with that tier (r4 second opinion L4); a second mismatch is reported and the unit is kept as failed. A backend change: accepted automatically and recorded; output changes already go through the epoch rule (r3 second opinion L1) |
| B (hand-over, D196) | tier 1 no longer fits when T is finished, after every `X` answer and retry, before AP (D198) | `stop` or `extend` (blind) |
| B (during T) | the budget (section 13) | `stop` or `extend` |
| epoch | the canary failed and started a new epoch | `allow_epoch` (during S: automatic, recorded) |

`accept_failed` is new. S3.4 had no way to accept a unit as failed, since its decisions needed every unit.

## 15. Blinding (D64, D189)

- **During T and S (D195), the notebook and the bundle show only:**
  - progress **as of the last completed block** (units done, failed and left, per plan), plus "block k of n in
    progress" (second opinion M2). Session ends fall inside blocks, so live counts and hours at session ends would
    let Claude solve for per-unit costs from the known shuffle order;
  - **GPU-hours (D200):** only at T's end, at every `B` pause and at S's end. Otherwise the notebook says only
    whether the next block or tier fits. The budget check uses live numbers internally. S tiers are nearly
    single-arm, so per-tier hours would give per-arm cost, a proxy for failures (r3 second opinion L5);
  - an ETA from the frozen cost model;
  - epoch and canary status;
  - the counts of failed units and their error signatures (no unit ids, since ids name conditions);
  - verify and audit totals, not by condition (review L7);
  - the log with pauses and answers.
- **Never shown during T or S:**
  - success or solved counts; API, LLM or step counts; tokens; notes;
  - wall time or cost per unit or per condition;
  - per-session changes in ledger call counts. The bundle's ledger part appears only at the D200 points; otherwise it
    holds no hours (review H1, r4 review M4).
- **Enforcement:**
  - the bundle is built from a whitelist of keys, and a test fails on any other key;
  - the session log lines the runner prints are not part of the bundle;
  - Claude reads only the bundle, never the result files on Drive;
  - **the user does not open `s35/units/`, the ledger, or any result file on Drive during T or S** (second opinion
    L7, D195; part of D64). The notebook prints this rule in its setup cell;
  - no interim analysis.
- **Remaining leaks, stated:**
  - the user can see the clock while cells run, which is a rough per-session timing. It shows no per-unit or
    per-condition figure;
  - the hours shown at `B` pauses (D200) are the only intermediate figures. A `B` pause during T falls at a block
    boundary, so in the rare case of several pauses, their differences could give a rough block cost. C0-C3 always
    appear together, so the primary comparison's cost cannot be separated;
  - the log's timestamps inside a block are stripped from the bundle. An `X` pause appears in real time mid-block,
    and its retry cost is charged to the next block.
- An `X` pause names the number of failed units and their error signatures. A unit id appears only in the notebook's
  own output during the retry, which is not part of the bundle.

## 16. Analysis outputs and unblinding (D185, D190; review M3, M4)

- **AP, after T:**
  - The notebook's analysis cell checks that the analysis sources match `s35/freeze.json`. It then computes
    `primary.json`: the RQ1 and RQ3 hypotheses, both graphs, the interaction and floor rule, completeness, and the
    D186 sensitivity version.
  - It writes `primary.json` and its canonical hash (section 9.2) to the chain log.
  - It writes the T export: every T unit's payload, without caches, under `s35/export/T/`.
  - It downloads both. This is the unblinding of the primary results.
- **Claude, from the T export:**
  - runs `report.py` at the freeze commit and must reproduce the recorded hash of the exact part of `primary.json`
    (D210), with the t intervals and MDDs equal to a relative 1e-8. A mismatch is reported and investigated before
    anything is written up;
  - then writes `full.json`, which holds the section 6 secondaries, the section 7 items that use only T units (C1 vs
    C0, C3 vs C2, the cost measures, the ratio-of-sums, the mixed model, per drift type and style, learning curves,
    C2-OU) and the section 8 mechanism checks on T units.
- **Close-out when S stops for good (r4 second opinion L3):** notebook 06 has a close-out cell that needs no GPU and
  installs nothing. It writes the S export and the hours at S's end. If even that cannot run, the user zips
  `s35/units/s35-S` and the ledger from Drive by hand.
- **After S:**
  - the S export (`s35/export/S/`);
  - Claude extends `full.json` with the descriptive arms, SP and the RQ3 extras.
- The descriptive units run under the frozen analysis and the fixed S rules (D192, D194-D202). Any S re-pin is
  limited, automatic in its answers and reported, so seeing the primary results cannot steer them.
- **Commits:** `full.json`, for T units and later for S units, is computed at the freeze commit only, so the
  analysis is the frozen one whatever code S ran. If a bug in the frozen analysis is found, the corrected numbers
  are computed outside the chain and reported next to the frozen ones, as a deviation (section 12).

## 17. Notebook 06 (`tools/make_nb06.py` -> `notebooks/06_s35.ipynb`)

- **Cells:**
  - notebook 05's settings cell (`USER_DECISION`, `EXPECT_LOG`, `REPIN`, `RELEASE_RUNTIME`);
  - its setup, with the cell check (D177);
  - the wheel, model and load cells, the canary and the epoch;
  - the session stop;
  - the chain cell (`TestPlatform.make_chain`), with block- and tier-level progress;
  - the analysis cell, which runs only when AP is due;
  - the bundle cell: copy, then flush and unmount, then download and release, as in notebook 05.
- `verify_finished` and the scrub audit run each session on new units, as in S3.4. Their totals go to the bundle.

## 18. Tests (pytest, before any Colab run)

- **Part 1:**
  - sign-flip by hand, with ties, integer NI values and n limits;
  - MDD and the t interval against scipy;
  - a seeded, repeatable bootstrap;
  - the graphical procedure (section 5);
  - RQ1 and RQ3 assembly on synthetic units with known answers: completeness, E2 without jointly solved tasks, D186,
    both flags (`below_initial_level`, `unrejectable`), the floor rule;
  - the canonical hash, stable across float noise below 10 significant digits;
  - mechanism checks on hand-built records, including the note matcher on every drift type;
  - the mixed model recovering a planted position effect on simulated data.
- **Part 2:**
  - unit contents: cfg5, phase equals plan, tags;
  - D63 order and the D183 tiers: tier contents, the cut-configuration property, the 171 total;
  - preflight (D188) on a fake platform;
  - the budget rule (blocks and tiers; `B` stop and extend; automatic S stop);
  - `accept_failed`; the automatic S answers;
  - the bundle whitelist and the absence of per-unit progress;
  - AP's timing (only after T is finished or stopped) and the hash record;
  - D192-D196:
    - a re-pin in S refused with no `X` and no stall;
    - a re-pin after AP refused if it changes an analysis module, protocol data or the T plan's unit list and `meta`;
    - `adopt_repinned` skipping `s35-T` after AP; the automatic backend `Z`; re-runs before the next tier check;
    - a refused D194 re-pin stopping S; the D199 reset after an adopted fix; D201 finishing the current tier; D202;
    - D197 decided over all plans before any write; the refused state, no unit run at an unaccepted commit, and the
      revert-commit way out; `REPIN_FOR` required after AP; D203 refusals;
    - the start gate after a halt or `X`; the consecutive-halt definition;
    - the runner facts: the chain's own consecutive-failure state across calls; retries marked per signature with
      `retry_failed` off; `accept_halted`; `RunHalted` caught for D201; block loops; T read without `run_units`;
    - the close-out cell;
    - the automatic `Z` after AP; T units never re-run; the automatic re-run of accepted-failed S units after a
      fix;
    - the tier-1 reservation in the T check and the hand-over `B`;
    - S bundles blinded; hours shown only at the D200 points;
    - D197 (a re-pin that changes any other finished unit is refused); D198 (tier 1 exempt after the hand-over);
      D199 (signature halts: no retries, two in a row end S at the tier boundary);
  - the freeze sha in both plans' `meta`;
  - `Z` on an analysis change; re-pin keeps `blind` and `meta`;
  - an end-to-end CPU run with ScriptedBackend through P, a shortened T, AP and the export;
  - notebook 06 static checks.
- Colab-only behaviour (the real T4 run) is never claimed as tested.

## 19. Fresh-context review of r1 (2026-11-27): 4 High, 7 Medium, 10 Low, all applied

- **H1:** the per-unit progress lines and ledger counts leaked unit timing, and so outcomes. Progress is now printed
  only at block and tier ends, and the bundle carries GPU-hours only (D189).
- **H2:** `test_phase_units` hardcodes phase "test" and no params, so units would run with k = 3. Units are now
  rewritten with cfg5 and the plan's phase, under a fixed shuffle version (section 10).
- **H3:** "reused unchanged" did not hold. The overrides are listed, and `Platform` is subclassed too (section 12,
  D191).
- **H4:** unblinding the primary results early made the later S answers non-blind. Every S rule is now fixed in the
  freeze (D185, option b).
- **M1:** the budget is checked per block and per tier, inside the chain, with a cost-class map, `gap_s` and
  `setup_s`, and a defined `stop` path (section 13).
- **M2:** nothing is installed on Colab; statsmodels is pinned in a new `configs/s35_pins.json`; no loop change.
- **M3:** a canonical rounded hash, and only the primary part must match (D190 hybrid).
- **M4:** the split between `primary.json` and `full.json` is explicit (section 16).
- **M5:** the note matcher now uses the drift truth (section 8).
- **M6:** the re-pin semantics are spelled out, and the re-check uses unit lists and `meta` (section 12).
- **M7:** NI values are computed from integers, and the symmetry assumption is stated (section 4).
- **L1:** the G update is simultaneous, and adjusted p-values come from the exact closed test.
- **L2:** the RQ3 MDD uses alpha/2.
- **L3:** tier seeds are named, the claim is restated as "at least the cut configuration", and the dead sentence is
  removed.
- **L4:** the dropped design items were added (sections 3 and 7).
- **L5:** the completion-time estimator is defined (section 6).
- **L6:** per-exposure attribution, the repeat-exposure denominator, and F's recall (sections 7 and 8).
- **L7:** audit totals only.
- **L8:** priors are stated, and GEE is reported alongside the mixed model.
- **L9:** a GitHub release instead of a tag.
- **L10:** the spare budget is restated.

## 20. Fresh-context review of the first Part 1 code (2026-11-27): 2 High, 5 Medium, 7 Low, applied

- **No correctness bug** was found in the graphical procedure (Algorithm 1, the closed-test adjusted p) or in the
  sign-flip test. 22 mutations were run: 14 survived, 4 of them equivalent.
- **H1:** the endpoint fixtures could not catch:
  - C1 swapped for C0;
  - S tested on shifted values;
  - E2 not restricted to jointly solved tasks;
  - floor-rule `or` or `<=`;
  - the interaction sign.

  The fixtures were rewritten: C0, C1 and C2 differ; there are C2 == comparator streams; solved sets differ, with
  distinct API calls; the interaction is nonzero; the floor edge cases are parametrised.
- **H2:** the RQ1 transitions are now checked entry by entry against the design table, and C1-side mirror cases were
  added. The RQ3 graph is also checked entry by entry.
- **M3:** the epoch is read from `stamp.llm`, not `run`. The test fixture now uses the real payload layout.
- **M4:** failed and not-done units: a failure becomes a D186 placeholder; not-done is dropped.
- **M5:** the `spec_sha256` check against the rebuilt stream is implemented.
- **M6:** a real D186 sensitivity test.
- **M7:** per-cell unfinished and never-started counts; dropped seeds are reported with the cell and status.
- **Low items applied:**
  - **L8:** the S-level flag is stated (section 4).
  - **L9, L10:** random graphs with zero weights, short rows and None or 0 p-values.
  - **L11:** position comes from `task`.
  - **L12:** the phase and n_tasks filter.
  - **L13:** sign patterns are generated in chunks.
  - **L14:** the bootstrap guard.

## 21. The user's second opinion on r2 (2026-11-28): 1 High, 2 Medium, 8 Low; all applied

- **Statistics:** confirmed correct against the reviewer's own exact re-implementations (BMBP09 in exact fractions,
  the closed test, the sign-flip test by exact enumeration). The symmetry wording was refined (section 4).
- **H1:** after AP, a re-pin could re-run T units, and a fix or a decision to stop S would be chosen after seeing
  results. Fixed by the three rules of D192, chosen by the user (sections 12 and 14).
- **M1:** tier 1 could be dropped entirely. Its cost is now reserved during T (D193, chosen by the user).
- **M2:** session-end progress let Claude solve for per-unit costs. Progress is now shown as of the last completed
  block (section 15).
- **Low items:**
  - **L1:** two flags, `below_initial_level` and `unrejectable` (section 4).
  - **L2, L3:** stated deviations (sections 11 and 13).
  - **L4:** the four surviving mutants (S-R1's flag level, the MDD level for E2/E4, the cap at 1, the bootstrap
    seed) now have tests. The bootstrap test first used 6 streams, whose quantiles do not change with the seed;
    that was our wrong test, fixed with 14 varied streams.
  - **L5:** `phase` is required in the primary functions.
  - **L6:** the paper's worked examples were added (section 5).
  - **L7:** the "don't open Drive during T" rule (section 15).
  - **L8:** numpy is pinned and scipy's version recorded (section 9).
- **On D182-D191:** the reviewer agreed with all ten, with D185 only under H1's rules (now D192). Optional suggestion
  for D182, a Zenodo archive of the release for a DOI: deferred, not needed for the freeze.

## 22. Fresh-context review of the r3 delta (2026-11-28): 2 High, 3 Medium, 5 Low; applied

- **H1:**
  - clear-cut fixes applied (section 12): an after-AP re-pin is refused if it changes an analysis module, protocol
    data or the T plan; `full.json` is computed at the freeze commit only; accepted-failed S units are re-run
    automatically after a fix; new section 14 rows for a `Z` with no re-pin and for the D130 halt during S;
  - user decisions D194 (environment-only fixes when S stalls) and D195 (S stays blinded).
- **H2:** the hand-over blind check (D196, user) and a fixed `reserved(tier 1)` in the freeze.
- **M1:** stale wording fixed (D185, D189, sections 12, 16 and 9.5).
- **M2:** tests for D192-D196 are listed (section 18).
- **M3:** the remaining C2-OU and RQ3 aggregate leak is stated; the tail block is defined; timestamps inside blocks
  are stripped.
- **L1:** a falsy `phase` raises, and the fixtures use an explicit phase.
- **L2:** the "inconclusive" leftovers are renamed.
- **L3:** an RQ1 7-untied-stream flag test.
- **L4:** the bootstrap pin's coupling is noted in its docstring.
- **L5:** the automatic `Z` rule is one sentence.

## 23. The user's second opinion on r3 (2026-11-28): 0 High, 4 Medium, 7 Low; all applied

- The statistics and the new tests were confirmed against the reviewer's own exact implementations (4,000 p-vectors
  for Algorithm 2).
- **User decisions:** D197 (M1: a fix after AP must leave every other finished unit replay-identical), D198 (M2:
  tier 1 committed after the hand-over check), D199 (M3: crash-signature rule in S), D200 (L5: hours only at T's
  end, `B` pauses and S's end).
- **M4:** T is never re-planned after AP, and the refusal hash is the T unit list plus `meta` (section 12).
- **Low items:**
  - **L1:** a backend `Z` after AP is accepted automatically.
  - **L2:** a stall fix is mandatory when an environment-only one exists.
  - **L3:** re-runs after a fix are scheduled within S.
  - **L4:** the path for an analysis bug.
  - **L6:** the n = 0 flags test.
  - **L7:** `eval` without builtins, and an assertion that eps is far above `TOL`.
- **The five figure encodings** in `tests/data/bmbp09_examples.json` were checked by Claude against the PDF's rendered
  figures (Figure 2 on p. 588, Figures 9 and 10 on pp. 596-597, Figures 11 and 12 on p. 598), and Figure 4 against
  the text on pp. 589-590 and 595.
  - Every edge and weight matches, including where the ε edges sit (Fig. 9: H2 -> H3; Fig. 10: H2 -> H3 and H4,
    r1 ε and r2 ε; Fig. 12: H3 -> H1 and H4 -> H2).
  - The reviewer independently showed that H2 cannot be rejected in Fig. 2 (its weight never exceeds 1/2, a level
    of 0.025 < 0.04), which supports the p. 598 typo reading.

## 24. Fresh-context review of the r4 delta (2026-11-28): 1 High, 6 Medium, 6 Low; applied

- **H1:** a refused stall fix stopped nothing; now any refusal stops S, with one attempt per stall cause.
- **M1:** the T-final wording now agrees with D197.
- **M2:** an adopted fix resets D199.
- **M3:** D197 leaves out units already recorded as a mismatch.
- **M4:** the bundle carries hours only at the D200 points.
- **M5:** the freeze lists D194-D202.
- **M6:** D201, the user's decision: the current tier finishes, then S ends.
- **M7:** D202, the user's decision: one `X` fix attempt per signature is mandatory.
- **Lows:**
  - the S stop cites D184 and exempts committed tier 1;
  - hand-over timing and the D198 overrun wording;
  - stale D185 and D192 rows;
  - the analysis-bug path in section 16;
  - the "units it fixes" definition;
  - the section 18 tests.

## 25. The user's second opinion on r4 (2026-11-28): 1 High, 3 Medium, 5 Low; applied

- The reviewer judged the rules implementable on `s34.Chain` and `Platform` once these are fixed. Its four runner
  claims were checked against `runner.py` and `t4.py`: correct.
- **H1:** a D197 refusal could not leave S running, because the re-pin is recorded before the check. Fixes:
  `adopt_repinned` decides over all plans before writing; a `refused` state; the revert-commit way out;
  `REPIN_FOR`.
- **M1:** D203, the user's decision: chain-code changes after AP are refused.
- **M2:** the start gate after a halt or `X`, with a recorded `no_fix`.
- **M3:** the runner facts are written down (section 12) and listed as tests.
- **Lows:**
  - **L1:** the consecutive-halt definition.
  - **L2:** the Colab "Runtime version" escape, and notebook 06 being protocol data.
  - **L3:** the close-out cell.
  - **L4:** scheduling of mismatch re-runs.
  - **L5:** the two backend-change paths, stated.
- **Next:** implement. Further corner cases are left to the Part 2 tests (section 18) rather than more spec-only
  rounds; the reviewer agreed this is reasonable once these are fixed.
- **Fresh-context check of this delta:** 2 High, 2 Medium, 4 Low, all applied:
  - **H1:** a revert is recorded as `revert`, not a fix;
  - **H2:** the gate opens on a signature's first halt, and the count includes that session;
  - **M1:** the "diverged twice" halt has no signature and is marked `divergence`;
  - **M2:** an `X` fix refused under D203 behaves like a D197 refusal;
  - **Lows:** D203 is in the stall-refusal list; "S continues after the revert"; `no_fix` covers D203 and the hash
    rules; `accept_halted` acts on every halted unit, so the chain marks the signature's units itself.

## 26. Part 1 code: the rest of the analysis (2026-11-28)

- **New modules:**
  - `secondary.py`: C1 vs C0; the integer TOST for C3 vs C2; cost per solved task; ratio-of-sums; error splits;
    exposures by drift type and style; the share of calls that hit a drifted operation; learning curves; the mixed
    model; the RQ3 secondaries;
  - `descriptive.py`: arms vs C2, SP and the RQ3 extras;
  - `mechanism.py`: repeat exposure, note correctness, both precisions, retrieval recall and C1 duplicate crowding;
  - `report.py`: `primary.json`, `full_T` and `full_S`, the canonical hash, pins and epochs;
  - `data.py`: raw payload parts and the stream.
- **Populations (code review M5):** C0-C3 are compared over the complete streams only, and every descriptive arm
  next to C2 on its own streams.
- **Fresh-context code review:** 2 High, 7 Medium, 8 Low, applied.
  - **H1:** item paths are compared by their last segment, and a new required field may come out as `value`.
  - **H2:** a note's text no longer includes what the agent sent. A type pair needs the field and the kind
    together. "list" is an alias for array kinds.
  - Mutation testing first: 35 of 47 mutants survived. After the fixes and new tests, re-run against a copy of the
    code: 1 of 51 survives. It is equivalent on real payloads: a never-started task is always censored.
  - Our own test mistakes, fixed and stated:
    - an "11th digit" pair that crossed a rounding boundary;
    - RQ3 fixtures with 4 tasks where units are indexed at 20;
    - the completion expectations, where "finished" means not censored, not solved;
    - a rename item without its `old` field.
  - One code bug found by the new tests: a variable in the SP loop shadowed the "success after" list.
- **Decisions:** D204-D207 (user).

## 27. The user's second opinion on the Part 1 code (2026-11-28): 0 High, 4 Medium, 5 Low; applied

- The reviewer checked every definition against the real payloads: exposure attribution, repeat exposure,
  `item_belongs` and both precisions, retrieval recall, C1 crowding, SP, the completion times, the RQ2 paths and the
  integer TOST. All correct.
- **M1:** `full_T` now has the C2-OU vs C2 row on C2-OU's own streams, and every arm uses its own streams.
- **M2:** LLM goodput counts only status-200 calls completed in [warmup, t_end), as the background goodput does.
  4xx answers had been counted as good, about 1,870 of 10,935 calls in the dry run.
- **M3:** D209, strict and loose type matching, both reported.
- **M4:** D208, the frozen rule for quoting the mixed model.
- **Low items:**
  - **L1:** D210, the hash covers the exact part only. A subprocess test checks that the hash does not change with
    `PYTHONHASHSEED`.
  - **L2:** the mechanism checks use the same complete-stream population.
  - **L3:** `n_values` per RQ3 metric.
  - **L4:** GEE intervals use t with df = clusters - 1.
  - **L5:** the five surviving mutants are now killed.
- **Mutation testing, re-run on this code:** 4 of 67 mutants survived. Two were killed by new tests (position
  centring, counting missing metrics). The two left are equivalent on real payloads: the never-started filter
  (never-started tasks are always censored), and iterating every stream instead of the arms' streams (a stream
  without the arm is skipped either way).
- One of our new assertions was tautological; it is replaced by an exact check of the t interval.
- **Fresh-context check of this delta:** 2 Medium, 6 Low, applied.
  - **M:** a NaN VB-MAP gap had passed D208, since `NaN > 0.5` is false; it now fails.
  - **M:** the hash test did not catch a whole-file hash; it does now.
  - **The loose matcher (D209) also:**
    - treats `!`, `?`, `:` and `…` as clause ends;
    - accepts backticks, quotes and parentheses around the field and the kind;
    - rejects a negated statement ("must not be", "no longer", "isn't").
  - **Known limits, stated:** the field must come before the kind ("a list is expected for query" is not matched),
    and the GEE t interval uses df = clusters - 1; t(K - p) is a common alternative with bias-reduced errors.
  - **Lows:** `float_mismatches` flags a missing subtree; `write()` is documented as not the D210 record; the
    fewer-than-2-streams case gets a `quotable` entry; a tautological assertion was replaced.

## 28. The user's second opinion on the Part 1 fixes (2026-11-28): 0 High, 1 Medium, 4 Low; applied

- **Confirmed:**
  - LLM goodput now equals `_completed_ok` on the gateway's LLM calls. The reviewer re-ran 8 RQ3 units; the old
    value was 13-35% too high.
  - Nothing scipy-derived is left in the D210 hash, and nothing exact is left out.
  - No tautological tests.
- **M1, the loose matcher (D209) and negation.** The negation check had looked only between the field and the kind,
  but the reflection prompt asks for an imperative, where the negation comes first ("Do not send query as a list").
  - Negation is now looked for from the start of the field's sub-clause (after the last `, ; . : ! ? …` or newline)
    up to the kind.
  - The negation words now include cannot, avoid, stop, nor and contractions with either apostrophe.
  - The gap is lazy, so "Send amount as a string not decimal" counts the nearer "string".
  - "Don't send a string, send query as a list" stays a match: the comma starts a new sub-clause.
- **L1, the loose matcher's wording coverage:**
  - backticks, `**` and a colon are allowed right after the field ("`query`: list", "**query**: array");
  - kind words are case-insensitive ("a JSON Array");
  - "1 or 0" and "0 or 1" are words for the 0/1-integer kind.
  - **Stated limits:** the kind before the field ("Use a list for query") and "from a string to a list" are not
    matched; the pilot check counts them. Closeness can give false positives ("query/tags are list"), which is
    accepted for a loose rule.
- **L2:** the tolerance is 1e-8, and the recorded file is compared with the unrounded recomputation.
- **L3:** a NaN VB-MAP gap anywhere makes the gap NaN, so D208 refuses (`max_gap`).
- **L4:** tests were added for:
  - a note only the loose rule matches;
  - the exact-part leaves (only `ci95` and `mdd` are removed; changing `rejected`, a mean difference, the
    interaction or n changes the hash);
  - the 1e-8 tolerance;
  - the D208 boundaries at 0.02 and 0.98.
- **D210 wording:** the hashed part is "scipy-free", namely exact values plus deterministic numpy floats.
- **Mutation testing of the new code:** 0 of 14 survive.
- **Fresh-context check of the matcher rewrite:** 2 Medium, 5 Low; applied, and two stated as limits.
  - **M1:** two patterns backtracked quadratically: 20,000 spaces after a field took 17 s. Both are fixed; it now
    takes milliseconds, and a test enforces it.
  - **M2:** the field's first letter now matches in either case ("Query must be a list").
  - **L3:** "0/1" and "1/0" are 0/1-integer words; "1 or 0.5" is not.
  - **L4:** a field is bounded by Unicode word characters and hyphens ("sub-query" is not "query").
  - **L5:** "no-" is not a negation.
  - **Stated limits (L1, L2):** a comma opens a new sub-clause, which both helps and hurts ("Do not, under any
    circumstances, send query as a list" matches); a field that is an ordinary word ("public") can match ordinary
    prose.
- **Pilot check, prepared:** `tools/s35_dev_check.py` runs the dev-data dry run and the D209 pilot check on the
  S3.4 pilot payloads.
  - It reads the 10 C0/C2 files of `s34-B-r1`. It was tested on scripted files in the same layout and naming.
  - The frozen code compares C0-C3 over complete streams, so it adds stand-in C1/C3 units (copies of C0 and C2).
    Nothing computed for them is reported.

## References

- [BMBP09] F. Bretz, W. Maurer, W. Brannath and M. Posch, "A graphical approach to sequentially rejective multiple
  test procedures," *Statistics in Medicine*, vol. 28, no. 4, pp. 586-604, 2009.
- arXiv:2402.13719, "Informative Simultaneous Confidence Intervals for Graphical Test Procedures", section 1.2
  (restates the BMBP09 update rules; checked 2026-11-27).
