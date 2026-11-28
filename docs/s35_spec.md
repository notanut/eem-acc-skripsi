# S3.5 spec: frozen analysis and the blinded, interleaved test phase (RQ1 and RQ3 tests, descriptive arms)

_r2 (+ BMBP09 examples 2026-11-28), 2026-11-27: r1 plus the fresh-context review of r1 (section 19). It implements design sections 7-9 and 10
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

**[delegated]** (r2 applies the review's changes):

| ID | Decision | Options | Chosen |
|----|----------|---------|--------|
| D182 | How the analysis is frozen | (a) the analysis modules join the test chain's protocol files; a freeze record with their sha256 goes into every test plan's `meta`; a GitHub release `s35-freeze`, which has a server timestamp (review L9); (b) a git tag only | (a) |
| D183 | Order of the descriptive units | (a) tiers in the reverse of the pre-registered cut order, so a stop at a tier boundary leaves at least a pre-registered cut configuration; (b) arm by arm | (a) |
| D184 | Budget stop | (a) checked before each T block and each S tier against the frozen cost model; during T, a pause where the user may `stop` or `extend` (blind); during S, an automatic stop at the tier boundary; (b) a hard stop at 60 h | (a) |
| D185 | When to unblind (review H4, option b) | (a) everything at the end; (b) primary results after T, with every S rule fixed in the freeze now, so no S answer depends on the user; (c) primary results only after S | (b) |
| D186 | A primary unit that fails for good | (a) its stream or seed is left out for every condition (complete case), plus a sensitivity analysis counting its tasks as failures; (b) count as failures only | (a) |
| D187 | Mixed-effects logistic model (secondary) | (a) statsmodels `BinomialBayesMixedGLM.fit_vb` with crossed random intercepts for stream and template and stated priors, with GEE reported alongside it; (b) R lme4; (c) GEE only | (a) |
| D188 | Code changes between S3.4 and the test phase | (a) allowed only if every finished S3.4 unit still replays identically and no S3.4 behaviour module changes, else a pause; none is planned; (b) none at all | (a) |
| D189 | Blinding of costs and timing (review H1) | (a) no per-unit progress lines; progress and ETA only at block, tier and session ends, with the ETA from the frozen cost model; bundles carry total GPU-hours only; (b) per-unit lines | (a) |
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
  - The test is exact under exchangeable signs, i.e. symmetry of the per-stream values about 0. That means symmetry
    about the margin for NI, and about 0 for superiority. Section IV states this assumption (review M7), as Stage 2
    did for TOST.
- **Ties:** a zero changes no pattern's mean, so the smallest attainable p is 2^-(untied n).
  - The number of untied streams is reported with every test.
  - A test whose smallest attainable p exceeds its initial level is flagged inconclusive. Its p still enters the
    graph, because a later level can be larger. The S hypotheses start at level 0, so for them the flag uses the
    level they would inherit from their NI test (alpha/4 for RQ1, alpha/2 for RQ3; code review L8).
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
4. **Environment:**
   - Nothing is installed on Colab for the analysis. The primary part uses numpy and scipy, which the session
     already has; the notebook records their versions.
   - statsmodels is used only in Claude's workspace, pinned in a new `configs/s35_pins.json`, which is not an S3.4
     protocol file (review M2).
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
     - the S-phase rules (D185);
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
  1 RQ3 seed with the 2x2, in a seeded shuffle within each stream or seed. `blind` is true. A **block** for the
  budget check (section 13) is one such group: 2 streams and 1 seed, or the last seeds alone.
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
  - the progress output (D189): the Runner's per-unit `progress_line` is replaced by a no-op; progress prints at
    block, tier and session ends;
  - the budget check (section 13) runs inside the chain between blocks and tiers, around `run_phase(max_units=...)`.
    The runner's D130 state (`last_failure`) persists across those calls, so the "two failures in a row" halt still
    works.
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
  - **S:** runs `s35-S` tier by tier, under the fixed rules of D185 (section 14).
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
  - A finished test unit that does not replay identically is a `Z` that names how many units changed, never which
    ones or how (blinding).
  - On `accept`, those units are re-run at the new commit; the old results are archived and reported as a
    deviation.

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
- **Before each T block:** if used + projected > budget, the chain pauses with `B`:
  - `stop`: T ends, AP runs on the complete streams and seeds (section 3), and S does not run;
  - `extend` with a new budget in hours: the user buys more compute. This is blind, because no outcome has been
    shown during T. It is recorded.
- **Before each S tier:** if used + projected > budget, S ends automatically at that tier boundary (D185). After AP
  the user has seen the primary results, so no S decision is left to the user.
- There are no cuts during the test phase and no deadline (D179).

## 14. Pauses

| Paused at | Cause | Actions |
|-----------|-------|---------|
| P | S3.4 units no longer replay identically, or S3.4 behaviour code or data changed (D188) | `accept` (recorded in the freeze), or re-pin a fix |
| X (during T) | units failed again after their retry | `rerun` (one more retry) or `accept_failed` (D186) |
| X (during S) | the same | **automatic:** one more retry, then `accept_failed` (D185) |
| Z | a re-pin changed protocol material, analysis included, or a finished test unit replays differently | `accept`: an addendum; changed units re-run; a changed analysis is reported as a deviation from the pre-registration. During S the Z still waits for the user, since a code fix is not an outcome decision; it is reported as made after unblinding |
| B (during T) | the budget (section 13) | `stop` or `extend` |
| epoch | the canary failed and started a new epoch | `allow_epoch` (during S: automatic, recorded) |

`accept_failed` is new. S3.4 had no way to accept a unit as failed, since its decisions needed every unit.

## 15. Blinding (D64, D189)

- **During T, the notebook and the bundle show only:**
  - progress at block and session ends (units done, failed and left, per plan);
  - total GPU-hours used vs the budget;
  - an ETA from the frozen cost model;
  - epoch and canary status;
  - the counts of failed units and their error signatures (no unit ids, since ids name conditions);
  - verify and audit totals, not by condition (review L7);
  - the log with pauses and answers.
- **Never shown during T:**
  - success or solved counts; API, LLM or step counts; tokens; notes;
  - wall time or cost per unit or per condition;
  - per-session changes in ledger call counts. The bundle's ledger part is the total GPU-hours only (review H1).
- **Enforcement:**
  - the bundle is built from a whitelist of keys, and a test fails on any other key;
  - the session log lines the runner prints are not part of the bundle;
  - Claude reads only the bundle, never the result files on Drive;
  - no interim analysis.
- **Remaining leak, stated:** the user can see the clock while cells run, which is a rough per-session timing. It
  shows no per-unit or per-condition figure.
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
  - runs `report.py` at the freeze commit and must reproduce the recorded hash of `primary.json`. A mismatch is
    reported and investigated before anything is written up;
  - then writes `full.json`, which holds the section 6 secondaries, the section 7 items that use only T units (C1 vs
    C0, C3 vs C2, the cost measures, the ratio-of-sums, the mixed model, per drift type and style, learning curves,
    C2-OU) and the section 8 mechanism checks on T units.
- **After S:**
  - the S export (`s35/export/S/`);
  - Claude extends `full.json` with the descriptive arms, SP and the RQ3 extras.
- The descriptive units run under the frozen code and the fixed S rules, so seeing the primary results cannot change
  them.

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
    inconclusive tests, the floor rule;
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

## References

- [BMBP09] F. Bretz, W. Maurer, W. Brannath and M. Posch, "A graphical approach to sequentially rejective multiple
  test procedures," *Statistics in Medicine*, vol. 28, no. 4, pp. 586-604, 2009.
- arXiv:2402.13719, "Informative Simultaneous Confidence Intervals for Graphical Test Procedures", section 1.2
  (restates the BMBP09 update rules; checked 2026-11-27).
