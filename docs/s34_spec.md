# S3.4 spec: the automatic pilot chain (docs wording, pilot, RQ3 wiring check, budget gate, freeze, EEM selection)

_r10, 2026-11-27 (r9 + the seventh second opinion, section 25). Implements design section 8 (dev rules), section 9 (budget gate) and section 10 (S3.4) of
`docs/stage3_design.md`, on the S3.3 machinery (`docs/s33_spec.md`, `docs/s33b_spec.md`). User decisions D154-D164
(section 13). r2 applies the fresh-context review of r1 (section 14); where it needed a judgment, the choice was made
by Claude under the user's delegation of 2026-11-25 ("do as you please" for this round) and is marked **[delegated]**
for the user to confirm or change._

## 1. Scope and cost

S3.4 runs as **one automatic chain** (D154) on the Colab T4, over as many Colab sessions as it takes:

| Step | What runs | Rule applied when it finishes |
|------|-----------|-------------------------------|
| A. Docs wording | C0 x W1-W3 x dev streams 1000-1001 | highest C0 success (section 4) |
| B. Pilot | C0 and C2 (default EEM) x dev streams 1002-1006 | switches, acceptance, ladder (section 5) |
| C. RQ3 wiring | C2+ACC and C0+none x RQ3 dev seed 5001 | wiring sane, nothing unfinished at T_max (section 6) |
| D. Budget gate | no units | projection, pre-registered cuts, task count (section 7) |
| E. Freeze | no units | write the freeze record (section 8) |
| F. EEM selection | 6 configs x dev streams 1007-1010 (3 if cut step 11) | selection rule, D60 (section 9) |
| G. Report | no units | the S3.4 bundle (section 10) |

The chain stops for the user only at the pauses of section 11. Out of scope: the test phase (S3.5-S3.7), freezing the
analysis code and the graphical procedure (prepared before S3.5), any change to the design's rules.

**Cost** (session 2's measurements; the gate replaces these with the pilot's): the base chain is about 10 GPU-hours
(A 0.8, B 2.1, C 0.4, F 6.8, plus session overheads). Each extra round of A + B (a switch or a ladder step) adds about
3 hours, a D60 re-run about 2. The worst case (two switch rounds and three ladder steps) is about 25-27 GPU-hours;
the gate (section 7) counts whatever was actually used.

Fixed inputs: platform epoch 0 (canary fingerprint c0e366fa, S3.3b session 2); the pinned embedder libraries and
MiniLM files (`configs/t4_pins.json`); the fresh decoding path (D151); one process (D152); the default EEM
configuration k = 3, k_s = 2, theta_link = 0.65; 20 tasks per stream; T_max = 3,600 s; the free tier (D153), possibly
from a second Google account (D164).

## 2. Code version, plans, state and resume (D158)

- **One commit per chain version.** The first session records the chain's commit and notebook fingerprint in
  `stage3/s34/chain.json` (an existing file refuses another notebook copy, `t4.Sections`). Every session checks out
  that commit; every plan carries it as `frozen_commit`, so no unit runs at another commit.
- **A code fix: adopt by replay (D166, replaces r3's new chain version).** The chat names the fixed commit; the
  notebook's `REPIN` setting checks it out (it must descend from the chain's commit) and `t4.Sections(repin=True)`
  replaces the chain's stamp, keeping the old one under `repins`. At the start of the next `Chain.run`, every plan of
  the chain at another commit gets a new plan version at the current commit; each finished unit is replayed with the
  fixed code (`verify_replay`, no GPU) and, if identical **and** its cache header equals the current session's on
  every field that decides fresh generation and embedding (the backend stamp: model, wheel, llama.cpp version,
  template, load parameters, sampling, decoding path; the embedder id, torch, sentence-transformers; the GPU),
  **adopted**. Fields that replay re-executes or that change without a fix (prompt, parser, suite, RQ3 and simulator
  versions; python, numpy; the platform epoch and fingerprint) do not block adoption (D173); the versions are
  still enforced through the payload stamp that `verify_replay` recomputes (e.g. after an RQ3 version bump, RQ1
  units are adopted and RQ3 units re-run, with the reason `False`) (`adoptions/<phase>.jsonl`: unit, from-commit,
  to-commit, payload sha256; the runner counts an adopted result as made at the frozen commit). Units that do not
  replay identically re-run. Every recorded decision is then recomputed; if one differs, the chain halts ("the code
  fix changed a decided result"): the user decides. A pause that needed the fix (step `X`, below) is cleared by the
  re-pin and its step runs again. The freeze is never rewritten (a freeze written just before a re-pin, whose log
  entry was lost, is kept as it is). **Limit:** `verify_replay` replays cached records; it never re-runs the code
  that computed what the cache stores (the backend's `generate`: overflow and control-token checks, retokenisation;
  how records such as `finish_reason` and `n_out` are written; the embedder's computation). The header condition
  catches changes to the load cell, the embedder's identity and the platform; for a fix inside that code the chat
  says so and the affected units are re-run instead. After a re-pin, the freeze's `plans_sha256` and
  `payload_sha256` name the old plan versions: S3.5 reads the code commit from the chain's stamp and `repins`.
- **A re-pin past the freeze** (2nd opinion #2, M2): the frozen material (suite version, template set, system
  prompt) is recomputed at the new commit. Unchanged: a freeze addendum names the commit (`freeze_addenda`).
  Changed: the chain pauses (step `Z`); the user accepts it as an addendum, or makes another fix and re-pins. The
  frozen material is: the suite version, template set, system prompt, prompt hash (reflection instruction, memory
  header, templates), parser, RQ3 and simulator versions, the EEM configurations, the loop's stopping rules
  (`LoopConfig`, `MAX_STEPS_EXTRA`), and the code of every behaviour module (`PROTOCOL_FILES`: `s34.py`, the agent
  loop, backend constants, memory, parser, prompts, scrub and action parser, the API suite, RQ3, the scenarios,
  status codes, keyed draws, the simulator, ACC, the gateway, the mock server, the metrics, the notebook's chain
  platform, the unit runner, the llama.cpp backend and tokenizer, the embedder and the scrub audit) plus RQ3's ACC
  parameters (`configs/rq2_frozen.json`, which must exist). A test walks the import closure of the chain, the agent loop and RQ3 and
  fails on any module that is neither frozen nor listed as infrastructure. The freeze keeps a snapshot of these
  sources on Drive (`s34/protocol/<commit>.json`); a later check parses the snapshot and the current sources with the
  same interpreter, per function, class body, method, sorted imports and other module statements (comments,
  docstrings and import order ignored), so a Python update cannot cause a false pause. The snapshot is checked against its recorded sha256;
  a missing one is rebuilt from git at its commit; one that differs from the sources halts. The `Z` pause names the
  code that changed. Changes only to code that cannot change a result or a rule (`MECHANICS`: the runner's
  lease, claims, heartbeat, file paths, progress text, exception classes and status marks; `write_once`,
  `save_once`) are recorded in the addendum without a
  pause (D174, narrowed by D175; the platform's constructor and verification, `Runner.mark` and the step choice after a
  fix pause are not mechanics, since they feed what replay trusts or which units run); every other change pauses. The frozen material also includes the backend and load
  settings that fresh generation and embedding use (`backend_sha256`, from the session header without the GPU name),
  compared in every session after the freeze, not only after a re-pin: a rebuilt llama.cpp wheel pauses with
  "wheel rebuilt, canary epoch unchanged" (D176). Any adoption reason except an I/O error counts
  as a finished unit replaying differently. A re-pin past the freeze is checked before a user decision is applied and
  while the chain is paused; a `Z` that interrupts a pause re-opens that pause once accepted. A behaviour check is added: finished units of the chain that no longer
  replay identically under the new code also count as changed. The check also covers the natural long share, runs after the decisions are recomputed and again before step F, and
  compares with the latest accepted addendum (an accepted change does not pause again). A retry that the re-pin
  granted is kept through the `Z` pause. The chain-version mechanism stays in the code
  (`version`, `chain-v<k>.json`, `s34v<k>-` phases) but the notebook fixes `CHAIN_VERSION = 1`.
- **Plans:** one per step and round, `plans/s34-<step>-r<n>.json`, written once; a reload must name the same units
  (and the same `meta`).
- **Decisions are write-once.** When a step's units are all done, its rule is computed from their results (pure,
  unit-tested functions) and written to `chain.json` (`decisions.<step>-r<n>`), with the inputs it used. A later
  session recomputes every recorded decision and must get the same answer, else the chain halts.
- **Sessions:** each session runs the notebook top to bottom and continues from the recorded state. No kills, no
  benchmarks. One runtime at a time. With a second account (D164) the lease depends on Drive propagating `lease.json`
  between two accounts' mounts, which can take longer than the lease's staleness window: never run both accounts at
  once.

## 3. Units and parameters

- `Unit(phase="s34-<step>-r<n>", rq, "dev", kind, seed, condition, window, params, tag, n_tasks, stream)`.
- **New: stream parameters (`stream`).** The two recalibration knobs, `n_distractors` (default 2) and `long_share`
  (default None = the natural share, uniform template weights), are `Unit.stream` overrides. They are part of the unit
  id when set (a unit without overrides keeps its id), and they are passed to **every** `sample_stream` call for that
  unit: the runner, `verify_replay`, the scrub audit (`audit.py`) and any analysis that rebuilds the stream (review
  M4). **[delegated]** The final knob values apply to every stream kind from then on: RQ1 dev and test, SP and RQ3
  (so the RQ3 wiring check and the RQ3 test run tasks of the calibrated difficulty).
- Loop parameters (`params`): `docs_wording`, `grammar_always`, `action_max_tokens`; in step F also `k`, `k_s`,
  `theta_link`.
- Tags: `W1`-`W3` in step A; `cfg1`-`cfg6` in step F. Within a stream the conditions, variants or configs run in a
  seeded shuffle (`runner.shuffled`).

## 4. Step A: docs wording (D156)

- Units: C0 x {W1, W2, W3} x streams 1000, 1001 (6 units, 120 tasks), under the current switches and knobs.
- **Rule:** the variant with the highest C0 success over its 40 tasks. Ties: fewer mean API calls over the tasks
  (stream, position) solved by **every** tied variant (pooled over tasks, review L2); if no task is solved by all of
  them, or still tied: W1.
- [delegated, code review L6] "then W1" is read as "W1 if it is among the tied, else the lowest-numbered tied
  variant": a variant with lower success is never chosen by a tie-break.
- Step A is re-run as a new round after every pilot round that changes a setting (a switch or a ladder step), before
  the next pilot round (D55 and design: "after any change the docs-wording choice is re-run").

## 5. Step B: pilot, switches, acceptance, ladder (D157, D162)

- Units per round: C0 and C2 (default EEM) x streams 1002-1006 (10 units, 200 tasks) under the current settings and
  the wording chosen by the preceding step A round.
- **Measured** per condition: success; fallback rate = fallback calls / first-attempt action calls; truncation rate
  of action replies = action replies with finish_reason = length / all action replies, fallback replies included
  (review L3); format errors; context overflows; share of reflections reaching 60 tokens; prompt and output tokens;
  LLM and API calls per task; unit wall seconds per task.
- **Switches, evaluated after every pilot round** (review H1; each can only switch on, so at most two rounds of them):
  1. "grammar always" (no thought cap, D65) if the fallback rate pooled over C0 and C2 exceeds 5%, or the two
     conditions' rates differ by more than 3 percentage points;
  2. action cap 640 if the truncation rate of action replies exceeds 1% in either condition.
- **Order within a round [delegated]:** if a switch fires, the round's acceptance is not judged; step A is re-run
  under the new setting, then a new pilot round (C0 **and** C2, D162). Acceptance and the ladder are judged only on a
  round in which no switch fired.
- **Acceptance:** C0 success over the 100 pilot tasks between 0.2 and 0.8 inclusive.
- **Ladder (D157)** when acceptance fails, one cumulative step per round, at most 3:
  - too hard (< 0.2): step 1 long share -0.1 (from the natural share); step 2 distractors 1; step 3 distractors 0;
  - too easy (> 0.8): step 1 long share +0.1; step 2 distractors 3; step 3 distractors 4.

  After each step: step A, then a new pilot round (C0 and C2, D162). A step that would take the long share outside
  [0, 1] is skipped. **Pause** if acceptance still fails after step 3, or if it fails in the opposite direction from
  the previous judged round (an overshoot). The direction is taken only from judged rounds (review L4).
- The only knobs used are the two the design allows; the docs wording, drift share, error styles and MaxSteps are
  never used to move C0 (design section 8).

## 6. Step C: RQ3 wiring check (D161)

- Units: C2 + ACC and C0 + none on RQ3 dev seed 5001 (5000 was S3.3b's smoke run), under the final settings.
- **Wiring sane:** both units done (not failed or halted), `verify_replay` passes, the scrub audit has no hit.
- **Pause** (D161, D111) if an LLM task is unfinished at T_max, or the wiring is not sane.
- Re-run (same rules) after a D60 switch (section 9, review M2) **[delegated]**: "grammar always" lengthens virtual
  LLM time and can leave tasks unfinished at T_max, which this check must see before any RQ3 test seed.

## 7. Step D: budget gate (D159, D160)

- **Used:** the ledger's GPU-hours since `benchmark_start` (S3.3b session 2), runtime wall clock as Colab charges
  it, including discarded calls (epoch changes, restarts) (review L5).
- **Cost per unit = tasks x wall seconds per task** for its condition, from the unit wall seconds of the final
  judged pilot round, plus the measured mean gap between units (session wall time not inside a unit, per unit), plus
  per-session overhead (measured mean of setup, model load and canary) x the projected number of sessions (projected
  hours / mean unit hours per session so far, at least 1). Using wall seconds, not LLM seconds per call, includes the
  simulator, embeddings, Drive appends and `verify_replay` (review H3).
- **Conservative mapping (D159):** C0 and C2 measured; C1, C3, F, AN, A1, A2, C2-OU at C2's cost; R at C0's; SP at the
  mean of C0 and C2; **every** RQ3 cell at the larger of the two measured wiring cells' per-task costs (review L1).
  "Grammar always", if on, is already in the measured numbers. The second projection "without the static-prefix
  snapshot" of the design no longer applies (D151).
- **Projected remainder:** step F (24 units, or 18 after cut 11), a D60 reserve while any switch is still off (the
  selected configuration's selection streams, the C0 pilot streams and the RQ3 wiring check, once), and the test
  phase and descriptive arms at the design's sizes.
- **Fits** if used + projected <= 60 GPU-hours. Otherwise the pre-registered cut steps 0-11 are applied in order,
  re-projecting after each; a cut that saves nothing (its work is already done) is skipped (review H2).
- **Task count (D160):** if it still does not fit after step 11, the largest n in {19, 18, 17, 16} that fits (RQ3
  keeps 5 tasks per LLM agent, D62). It applies to work not yet run: step F, D60 re-runs of the selection streams,
  and the test phase and descriptive arms (prefixes of the full streams). Steps A and B stay at 20 tasks, and so do
  the D60 C0 acceptance re-runs, since acceptance is defined on the 20-task pilot (review M5). The S3.1 exposure rule
  is tested on the 17-19 prefixes; an n for which it does not hold is skipped. If 16 does not fit: **pause** (D57).
- The gate is recomputed once after the selection, after its D60 rounds if any (so it sees their cost and the
  re-measured per-task costs of the new setting, code review M4); each computation is a D entry of the chain log,
  cuts are only ever added, and the last one before the test phase is what S3.5 uses.

## 8. Step E: freeze

- `stage3/s34/freeze.json`, **immutable once written** (review H2): the chain's commit; suite version; template set
  sha256; system prompt sha256 for the chosen wording; the switches; the knobs; the task count; the cuts; the gate's
  projection; the sha256 of every step A-C result.
- Later facts are recorded **after** the freeze: `selection` (a write-once key, step F), each D60 round (an H entry
  of the log: switch, re-run results, C0 acceptance outcome) and each gate (a D entry). Gates and D60 rounds are read
  from the log, never kept as separate lists, so a resumed step cannot duplicate them (code review L2). The test phase reads the freeze plus these
  keys; the effective settings are the freeze's plus any D60 switch plus the last gate's cuts and task count.
- The freeze holds the payload sha256 of every step A-C unit (code review M2). Its sha256 is written into
  `chain.json` and into the step F plan before any F unit runs (review M6), so the record shows the freeze came first. After S3.4 the freeze and `chain.json` are committed to the repo
  (`configs/s34_freeze.json`, `docs/evidence/`).

## 9. Step F: EEM selection (D163) and D60

- cfg1-cfg6 = k in {3, 5} (k_s = 2 for k = 3, k_s = 3 for k = 5) x theta_link in {0.5, 0.65, 0.8}; C2 under the
  frozen settings on dev streams 1007-1010 (1007-1009 if cut step 11 applies): 24 (18) units. Seed 1010 continues the
  dev seeds' joint drift draw (balanced against all earlier dev seeds).
- **Rule:** the highest mean success over the selection streams (pooled over tasks). Configurations within one task
  per stream (1/n at n tasks) of the best go to fewer mean API calls over the tasks solved by all of them, then to
  fewer mean prompt tokens per task; then (D163) the default configuration if still among them, else the smaller k,
  then theta closest to 0.65. If no task is solved by all of them, the API-call step is skipped. The whole sweep is
  reported (RQ4).
- **D60 (review M1):** the switch rules are applied to the selected configuration's selection runs, with "overall" =
  the selected configuration's runs pooled with the final pilot round's C0, and "differ" = the selected
  configuration's rate vs that C0's. If a switch fires, the selection is kept; under the new setting the selected
  configuration's selection streams, the C0 pilot streams (acceptance) and the RQ3 wiring check (section 6) are
  re-run, and the gate is recomputed, all before any test stream. The switch rules are then applied once more to the
  re-run (so the other switch can still fire; at most two D60 rounds). A failed C0 acceptance after D60 is reported,
  and the campaign proceeds (D65).

## 10. Step G: report bundle

At the end of every session (also after a pause or an error: the chain cell catches every exception, so the
heartbeat stops and the bundle is written, code review M5), the notebook runs `verify_replay` and the scrub audit on
every finished unit not yet checked (kept in `chain.json` under `verified`), then writes and downloads a JSON bundle:
the chain file (state, the log with every decision and its inputs, selection, verified units), the freeze, the
ledger summary, the units that did not verify and the audit hits. I add it
to `docs/evidence/` and write the S3.4 results note.

## 11. Pauses (the chain stops and asks)

1. A code fix is needed: units that failed again after their retry (below), or the replay check before the freeze or
   before the selection (D168). Recorded as a step `X` entry (`before` = the step to run next, `fix` = true);
   cleared by a re-pin (section 2), or by `USER_DECISION` "rerun" (a transient cause). A recomputed decision that
   differs halts.
2. Acceptance still fails after ladder step 3, or a ladder step overshoots.
3. An RQ3 task is unfinished at T_max, or the wiring check is not sane (D161), including its D60 re-run.
4. The budget does not fit at 16 tasks per stream (D57, D160).
5. The canary fails and starts a new epoch: the user decides whether S3.4 continues or restarts in the new epoch.

Not pauses: a switch, a ladder step, cuts, a task-count reduction, a D60 re-run, a failed C0 check after D60, and
units that fail for the first time (second opinion M3): the chain stops with `RETRY` (`retries.<phase>` records each
failed unit set and its runtime); the next session, which must run in another runtime, retries them once
(`run_phase(retry_failed=True)`; halted units are re-enabled by the runner itself), and the retry is recorded only
after it ran. Only units that fail again after their own retry pause; units failing for the first time get their own
retry. This also covers a recorded phase whose units must run again at the start of a session (a repaired result, a
unit not adopted after a re-pin): every recorded phase is completed through this rule before the decisions are
re-checked.

**Answering a pause (D165).** The bundle gives the pause's log number `n` and the log fingerprint
(`<entries>:<sha256 prefix>`). The chat gives `USER_DECISION = {"answers": n, "action": ...}` and `EXPECT_LOG` = that
fingerprint; the notebook applies it before the chain runs (a 'U' log entry; refused if the fingerprint, `n` or the
action does not match; a copy left in the notebook after it was applied is a no-op). `EXPECT_LOG` is also checked
when the setup cell opens the chain file, before anything is written to it: a log on Drive that is shorter than the
bundle's (another account's session has not propagated) stops the session with nothing written; one that does not
start with the bundle's entries stops it too. A view that is stale but already longer than the last bundle cannot be
detected. Cell 8 ends with `drive.flush_and_unmount()`, so a session's writes are on Drive before the runtime ends;
if a refusal still persists (a write lost for good), the chat clears `EXPECT_LOG` and the chain recomputes the lost
steps (deterministic, except that the gate sees the used hours as recorded).

| Paused at | Actions |
|-----------|---------|
| B (ladder exhausted or overshoot) | `accept` (the paused round is the judged pilot; C next) or `settings` with `{"settings": {...}}` (only `n_distractors` 0-4 and `long_share` 0-1: the switches and the wording follow their rules; A and B re-run; the ladder ends: a miss under the user's settings pauses again) |
| C (wiring) | `accept` (D next; only if the wiring is sane, i.e. only unfinished tasks at T_max, D111) or `rerun` (a new C round) |
| D (D57) | `gate` with `cuts` (0-11) and `n_tasks`: must keep every cut of the last applied gate and not raise its task count; the projection is recomputed for exactly that choice; recorded as a D entry with `user = true`. At the first gate there is no applied gate, so the user may choose fewer cuts than the paused projection, or none, and up to 20 tasks: that is the decision to buy compute beyond the budget, not a hole |
| H (wiring after D60) | `accept` (the chain moves on; only if the wiring is sane) or `rerun` (a new H round) |
| Z (a re-pin changed frozen material) | `accept` (recorded as a freeze addendum) |
| X (failed units or the replay check) | `rerun` (one more retry of the units, or the replay check again) |
| epoch (section 11.5) | `{"action": "allow_epoch", "epoch": k}`: recorded under `allowed_epochs` |

## 12. Infrastructure changes

- **D148:** the Drive wheel cache holds wheels for all nine embedder libraries at the pinned versions plus whatever
  the install changed, so an offline install depends on Colab's image only for their dependencies.
- **D149:** a CUDA-library change is recorded (`cuda_checks`, `ok = false`) and does not stop; the canary decides.
- Setup installs the committed pins, so the first S3.4 session runs `network_exact` and builds the cache.
- Notebook `notebooks/05_s34.ipynb`, generated by `tools/make_nb05.py`: notebook 04's setup, wheel, model and load
  cells (fingerprint guard, embedder setup, kernel probe, CUDA check), the canary and epoch, then the chain driver
  (`eem_acc.stage3.s34`) with a progress line per unit (step, round, done/total, ETA, GPU-hours vs 60), and the bundle.
- New module `src/eem_acc/stage3/s34.py`: plans per step and round, the rule functions, the gate, the freeze record and
  the chain state machine over `chain.json`.

## 13. Decisions (user, 2026-11-25; D165-D169 2026-11-26)

| ID | Decision |
|----|----------|
| D154 | S3.4 runs as one automatic chain; it stops only at the pauses of section 11 |
| D155 | Disjoint dev streams: wording 1000-1001, pilot 1002-1006, selection 1007-1010 |
| D156 | Wording tie: fewer API calls on tasks solved by all tied variants, then W1 |
| D157 | Recalibration ladder, at most 3 cumulative steps; then the user decides |
| D158 | One commit for all of S3.4; a needed fix stops the chain and re-runs what it touches |
| D159 | Budget projection: conservative mapping of unmeasured arms |
| D160 | Task count after all cuts: the largest n in {19, 18, 17, 16} that fits; else the user decides |
| D161 | RQ3 wiring check on dev seed 5001; unfinished tasks at T_max or bad wiring pause the chain |
| D162 | After a switch or a ladder step, C2 is re-run on the pilot streams too |
| D163 | Selection's last tie-break: the default configuration, then the smaller k, then theta closest to 0.65 |
| D164 | Runs may use a second Google account with the `Skripsi` folder shared (the user's decision; Colab's FAQ restricts using several accounts to work around usage limits; noted) |
| D165 | Pauses are answered in the notebook (`USER_DECISION` + `EXPECT_LOG`), recorded in the log (section 11) |
| D166 | A code fix is adopted by replay after a re-pin (`REPIN`); units that do not replay identically re-run (section 2) |
| D167 | Gate costs: after the selection, C2-mapped work is priced at the selected configuration's measured cost; AN at 1.2 x C2 (section 7) |
| D168 | Replay check (verify_replay + scrub audit) of every A/B unit before the freeze and of every F unit before the selection; a failure pauses for a fix |
| D170 | `RELEASE_RUNTIME` on by default: cell 8 saves the bundle on Drive, downloads a local copy, flushes Drive, then ends the runtime |
| D171 | Without a D60 round, the final gate scales the RQ3 cells by the selected configuration's C2 cost over the pilot's |
| D173 | Adoption by replay does not require the header fields that replay re-executes or that change without a fix (prompt, parser, suite, RQ3, simulator versions; python, numpy; platform epoch and fingerprint) |
| D174 | After the freeze, a change only to the chain's mechanics (`MECHANICS`) is recorded in the freeze addendum without a pause; rule and behaviour code still pauses |
| D175 | `MECHANICS` keeps only code that cannot change a result or a rule; a fix anywhere else after the freeze pauses at Z |
| D176 | After the freeze, a different llama.cpp wheel sha256 (e.g. a rebuilt wheel) pauses at Z; the backend is compared every session |
| D172 | No extra rule for an ERROR repeated across sessions (2nd opinion #2, Low 6): ERROR already stops each session for the chat |
| D169 | The r2 delegated choices 1, 3, 4, 5 confirmed; #6 (selection tie: the lowest-numbered of the still-tied configurations after D163) and #7 (the realised long share reported once per pilot round, at the end in the bundle) adopted on Claude's recommendation at the user's request |

**[delegated] choices in r2** (Claude, under the user's delegation, to confirm): switch-before-acceptance order
(section 5); a code fix before the freeze restarts the chain from A, after it pauses (section 2); the knobs apply to
every stream kind (section 3); the RQ3 wiring check is re-run after a D60 switch (sections 6, 9); every RQ3 cell is
projected at the larger measured cell's cost (section 7).

## 14. Review of r1 (fresh context, 2026-11-25): 3 High, 7 Medium, 6 Low, all applied

H1 switches after every round (section 5); H2 immutable freeze plus write-once keys, no-saving cuts skipped (sections
7, 8); H3 wall seconds per task instead of LLM seconds per call (section 7); M1 D60 pooling and repetition (section
9); M2 RQ3 re-check after D60 (section 6); M3 the code-fix rule (section 2); M4 knobs on every `sample_stream` call
and every stream kind (section 3); M5 which runs the task count applies to, prefix tests for 17-19 (section 7); M6 the
freeze's sha before step F (section 8); M7 the cost range (section 1); L1 RQ3 mapping; L2 tie means and empty joint
sets; L3 denominators; L4 ladder direction from judged rounds; L5 used hours include discarded work, no snapshot
projection; L6 the two-account lease caution and seed 1010 (sections 2, 9).

## 15. Tests (pytest, before any Colab run)

- Every rule function on constructed results: ties (including no jointly solved task), the boundary values (0.2, 0.8,
  5%, 3 pp, 1%), switch-before-acceptance, both switches firing in successive rounds, the ladder (direction, bounds,
  overshoot pause, 3-step pause), the wiring check, the selection tie chain, D60 (both switches, two rounds).
- The gate: wall-second costs, the conservative mapping, cuts in order with re-projection and skipping of no-saving
  cuts, the task-count step with the 17-19 prefix exposure check, the D57 pause, recomputation appending.
- Unit ids with stream overrides (unchanged without them); knobs reach the runner, `verify_replay` and the audit;
  plans per round with the chain's commit; the freeze's immutability and its sha in the F plan.
- The chain end to end on CPU with `ScriptedBackend`: all steps; a switch round; a ladder step; a pause; resume after
  an interrupted session; a recomputed decision that differs halts; a new chain version after a code fix.
- D148 cache with all nine wheels; D149 recording without a stop; the generated notebook equals the committed one.
- r4: the chain on the real unit runner with the scripted backend (`tests/test_s34_platform.py`: ladder, pause, user
  accept, C, gate, freeze with the replay check, selection, final gate; resume recomputes everything; a re-pin adopts
  every unit with nothing re-run; a unit that no longer replays re-runs); retries; every USER_DECISION action and its
  guards; D167 pricing; missing costs; verify_log of D, H, F and the freeze; the runner's adoption rule. The scripted
  backend keeps a registry of task instructions, so the test gives each phase a fresh one (a ladder step re-samples a
  seed's stream); llama.cpp has no such registry.

## 16. Code review (fresh context, 2026-11-25): 1 High, 6 Medium, 9 Low, applied

- H1: a code fix could not start a new chain version (the old chain file pinned the old commit; plans were reused).
  Chain versions now have their own chain file, phase names and freeze file (section 2); `run_units` turns a commit
  mismatch into a pause instead of a crash. Tested.
- M1: the 3-pp rule fired at exactly 3 pp through float error (0.07 - 0.04 > 0.03); switches and acceptance use exact
  fractions. M2: payload hashes in the freeze. M3: per-unit `verify_replay` and audit in the bundle. M4: the gate
  after D60 uses the re-measured costs. M5: no exception skips the bundle or leaves the heartbeat running; D146 and
  the epoch check stop the chain through a flag. M6: tests for `unit_metrics`, both switches in successive rounds,
  two D60 rounds, an H pause, failed checks, a crash after the freeze write, a new chain version.
- L1: a crash between the freeze write and its log entry no longer skips E. L2: gates and D60 rounds are read from
  the log; `selection` and `freeze_sha` are write-once. L3: `verify_log` also recomputes H and reuses the recorded
  replay/audit outcome. L4: `unit_metrics` under "grammar always" (no fallbacks) and for fully replayed units (no
  cost). L5: the exposure check also tests the SP re-drift after the midpoint and uses the S3.1 constants. L6: the
  wording tie (section 4). L7: the gate timing (section 7). L8: the selection plan's `meta` is checked on reload. L9:
  the notebook refuses to run outside Colab.

## 17. User's second opinion on the code (2026-11-26): no High, 6 Medium, 7 Low; decisions D165-D169

- M1: `verify_log` for H compares only acceptance, wiring, switches, fired and again (timings may differ after a
  repaired unit). M2: a missing cost (a unit rebuilt only from its cache) is never passed on: the larger measured RQ3
  cell, else C2's cost; the pilot's costs must exist (else a halt); the D60 re-measure replaces only measured values.
  M3: first failures retry once in a fresh runtime (section 11). M4: pauses are answered in the notebook (D165).
  M5: the second gate prices C2-mapped work at the selected configuration's cost (D167; AN at 1.2 x C2, from the
  answer on the budget projection). M6: `EXPECT_LOG` from the last bundle: the setup cell refuses a chain file on
  Drive whose log is shorter (wait for Drive) or does not start with the bundle's entries, before writing anything
  (section 11); the bundle reports the log fingerprint and the chain file's sha256.
- Restarting from A after a code fix is replaced by adoption by replay (D166), as the second opinion proposed.
- L1: the D60 round's RQ3 C2+ACC unit runs the selected configuration. L2: a failure during `T4Session` start-up
  stops the heartbeat and sets `STOP` in cell 6 (the bundle still runs); an interrupt in cell 7 still writes the
  bundle. L3: `verify_log` also recomputes each gate from its recorded inputs and F's D60 switches, and checks the
  freeze file's sha256. L4: `CHAIN_VERSION` is fixed at 1 (section 2). L5: `RELEASE_RUNTIME` stays off by default
  for now: the bundle is also on Drive, so turning it on loses nothing; the idle tail after a session is not counted
  by the ledger (stated, not projected). L6: the replay check before E and F (D168). L7: the realised long share of
  each pilot round is recorded with its decision.
- Stated omissions: caches discarded after an epoch change are not projected by the gate; the 16-19 task exposure
  check always holds by construction (the rules concern positions below 15) and reads only the planned streams'
  structure, never outcomes.
- The notebook's chain callbacks moved to `src/eem_acc/stage3/s34_platform.py`, so the tests run the same code.
- A D60 round is counted only when a switch fired: a user `rerun` of an H round does not use up the second round
  (`again` = a switch fired; each switch fires once, so there are still at most two such rounds).

## 18. Fresh-context review of the r4 code (2026-11-26): 2 High, 5 Medium, 6 Low, all applied

- H1: `EXPECT_LOG` was checked in cell 7, after cells 1-6 had already rewritten `chain.json` from a stale view; it
  is now checked in `t4.Sections` before any write. H2: re-running a recorded phase's unit (verify, replay check,
  F's pilot reload) bypassed the retry rule and could halt the chain for good; every recorded phase now goes through
  it (`Chain._load`).
- M1: retries per unit set, recorded after they ran. M2: the platform loads a result only if it was made or adopted
  at the plan's commit; a commit mismatch halts. M3: an H rerun does not use up the D60 rounds. M4: a replay-check
  pause accepts "rerun". M5: a user gate keeps earlier cuts and task counts and gets its own projection.
- L1: cell 8 always stops the heartbeat and writes the bundle. L2: adoptions are recorded per phase. L3: the freeze
  is kept across a re-pin. L4: the adoption limit (section 2). L5: a user's settings are validated. L6: the
  undetectable longer stale view (section 11).
- Tests: every finding has a test (`tests/test_s34.py`, `tests/test_s34_platform.py`, `tests/test_notebook_static.py`).

## 19. Second opinion on the r4 code (2026-11-26): no High, 4 Medium, 7 Low

- M1: adoption also requires the unit's cache header to match the session's (section 2). M2: a re-pin past the freeze
  recomputes the frozen material (section 2). M3: `accept` after a C or H wiring pause only when the wiring is sane.
  M4: cell 8 flushes Drive before the runtime ends; the way out of a persistent `EXPECT_LOG` refusal (section 11).
- L1: a user's settings are limited to the two knobs. L2: units a halt left unrun are not counted as retried; only a
  unit that failed again after its own retry pauses. L3: while paused after a re-pin, if every unit was adopted, the
  decisions are recomputed at once (a fix that changed a decided result halts before the user answers the pause);
  if some units must re-run, nothing runs and the recompute waits until the pause is answered. L4: RQ3 scaled to the selected configuration (D171). L5: verify
  outcomes kept per payload and commit (`verified`), so the replay checks and the bundle do not replay a unit twice.
  L6: not adopted (D172). L7: plan hashes after a re-pin (section 2).
- `RELEASE_RUNTIME` on (D170).

## 20. Fresh-context review of the r5 code (2026-11-26): 1 High, 1 Medium, 7 Low, applied

- H1: a verify outcome without an audit (a unit replayed for adoption but not finished) crashed the bundle cell; only
  complete outcomes are persisted and the bundle reads them defensively. M1: a retry granted by a re-pin was lost
  when the re-pin also changed frozen material; it is carried through the `Z` pause.
- L3: the freeze check also runs before step F (a freeze kept across a re-pin). L4: nothing runs while paused. L5:
  units marked `halted` count as failed for the retry rule. L6: the freeze check runs after the recompute and against
  the latest addendum. L7: the paused recompute is decided from the pause's commit, not from in-memory state. L8:
  `adoptions` records why each unit re-runs. L9: cell 8 copies the bundle, flushes Drive, then downloads; cells 6-8
  refuse to run with Drive unmounted. Nits: a missing freeze file halts clearly; the natural long share is checked.

## 21. Third second opinion (on c33fbae, 2026-11-26): no High, 2 Medium, 5 Low

- M1: the frozen material is widened (section 2). M2: adoption's header condition relaxed (D173, section 2).
- L1: while paused, recorded phases are only loaded (`run_units(load_only=True)`): nothing runs. L2: the verify memo
  is keyed by the payload sha256 and the content hash of the cache file itself. L3: `verify_finished` tolerates a unit without a memo
  entry. L4: if `drive.flush_and_unmount()` fails, cell 8 still downloads the bundle but keeps the runtime (so uploads
  can finish) and says so. L5: the first gate's user decision (section 11 table).
- Fresh-context review of that round (no High, 2 Medium, 3 Low), applied: the code hash is now independent of the
  Python version (checked on 3.11, 3.12 and 3.13; `ast.dump` was not), it covers whole behaviour modules plus the
  replay behaviour check, the `Z` pause names changed functions, the memo reads the cache file itself, and the D173
  note on payload stamps.

## 22. Fourth second opinion (on 24e85d7, 2026-11-26): no High, 2 Medium, 4 Low

- M1: behaviour modules that were not frozen (action parser, context-overflow constants, RQ3 scenarios and latency
  target, status codes, keyed draws, RQ3's ACC parameters) are added, with the import-closure test (section 2).
- M2: methods are hashed separately; mechanics changes do not pause (D174); import order does not count.
- L1: only `False` or `ReplayDivergence` counts as a finished unit replaying differently; an I/O error makes the unit
  re-run without a pause. L2 and L3: the snapshot on Drive replaces the stored hashes (no Python-version false
  pauses; the bundle no longer carries the per-function hashes). L4: the Python version is in the E log entry, not in
  the write-once freeze.
- Fresh-context review of this round (no High, 3 Medium, 5 Low), applied: the platform, runner, llama.cpp backend
  and tokenizer, embedder and audit are frozen (with mechanics lists; the import-closure test now starts from them
  and allows only cache, fsio, ledger, canary, t4, unit worker and embedder setup as infrastructure); the rule guards
  and the checker itself are not mechanics; divergence counts every reason except I/O; snapshot sha checks, rebuild
  from git, the data file must exist, same-name definitions keep separate keys, and the Z check runs before a
  decision.

## 23. Fifth second opinion (on 38d1fbf, 2026-11-26): no High, 4 Medium, 2 Low

- M1: `MECHANICS` narrowed (D175). M2: a `Z` that interrupted a pause, answered by another re-pin, re-opens the pause
  in one record (no step-less `U`; a `U` without a next step halts clearly). M3: a re-pin and a user decision in the
  same session: the decision first adopts and clears, so divergence is still seen. M4: the backend and load settings
  are frozen. L1: an `X` pause answered by both a re-pin and `rerun`: the re-pin's answer stands. L2: a git rebuild
  that cannot reach a commit halts with what to restore. Nit: UTF-8 reads; untracked files in frozen folders would
  make a git rebuild differ from the Drive snapshot (the Colab checkout is clean).
- Fresh-context review of this round (no High, 1 Medium, 2 Low), applied: the platform's constructor and
  end-of-session verification are not mechanics (they feed the remembered replay passes), nor are `Runner.mark` and
  `_next_or_none`; a pause re-opened twice keeps a clean copy pointing at the original; a decision that a `Z`
  pre-empted is a no-op afterwards (the pause re-opens after the `Z`).

## 24. Sixth second opinion (on b4ad2ad, 2026-11-27): no High, 2 Medium, 4 Low

- M1: the unit cache (the replay check itself, the embedding cache), the ledger (the gate's used hours and costs)
  and the canary (the epoch rule) are frozen; only their pure I/O helpers are mechanics. M2: an open fix pause is
  answered before the freeze check (the `Z` follows once the step re-ran). L1: the GPU name is not part of the frozen
  backend. L2: the backend is compared every session (D176). L3: a decision needs `answers`; a decision for a pause
  that re-opened names its new number. L4: `write_once` and `save_once` are not mechanics.
- The first S3.4 session was running at 6ad55ac when these were made; the next session re-pins (`REPIN`). A
  fresh-context review ran the old code's chain to each plausible end state and re-pinned it to the new code: every
  finished unit is adopted; before E there is no pause; after E there is exactly one `Z` (backend: the GPU name, which
  the old freeze included; newly frozen: cache.py, ledger.py, canary.py; and the changed s34 and platform functions),
  to accept once. Its Lows are applied: an accepted divergence is not listed again, the pause text says "this session
  differs" when there was no re-pin, a pause re-opened twice names the last open number, the per-session backend
  check reads no Drive files (`Platform.backend_extra`), and newly frozen modules are grouped in the `Z` text.

## 25. Seventh second opinion (on 7a31978, 2026-11-27): no High, 1 Medium, 2 Low

- The 6ad55ac -> 7a31978 re-pin was checked again on 13 end states: before E no pause; after E exactly one `Z`.
- M1: the session rules (D146 kernel probe, the epoch rule) and the chain's inputs (used GPU-hours since the
  benchmark start, the cost profile of this chain's units only, metrics, checks, freeze extras, adoption, backend)
  moved from the notebook into frozen code (`s34_platform.session_stop`, `Platform.make_chain`); the notebook itself
  is also part of the frozen material (`PROTOCOL_DATA`).
- L1: a `Z` raised only by the backend clears itself when the backend equals the last accepted one again; accepting a
  `Z` records the backend in use at the accept. L2: the GPU query is retried; a GPU runtime that still returns
  nothing stops the session instead of recording "none".
- Fresh-context review of this round (no High, 1 Medium, 3 Low), applied: accepting a `Z` records the session's
  backend only if the user saw it (the pause's, or the accepted one restored); otherwise the next check shows the
  new backend in its own `Z`. A lost snapshot is rebuilt with the file lists of its own commit's `s34.py`. The GPU
  query accepts only a name and a driver version. The notebook is hashed per code cell (markdown and the notebook
  fingerprint left out), so only code edits pause.

