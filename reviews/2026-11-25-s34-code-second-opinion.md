Now the code.

**Verdict:** no High. The code implements the rules of design sections 6, 8 and 9 exactly, including every boundary and tie-break I checked. The problems are in recovery: a session lost at the wrong moment, or an ordinary transient failure, can stop the chain falsely. Before the freeze, any stop that needs a code change costs a new chain version, about 10 GPU-hours. After the freeze it means reopening the freeze.

**What I checked:**
- **Tests:** the 45 S3.4 tests pass.
- **Real-runner run:** I executed notebook 05's cell-7 callbacks (`run_units`, `metrics`, `freeze_extra`, the audit) against the real `Runner` with `ScriptedBackend` on CPU.
  - A→G completes, and a second session's `verify_log` passes.
  - I had to force acceptance, because the scripted agent is too easy: 0.84–0.88, so the ladder runs out and pauses, which is itself correct behaviour.
- **APIs:** every `t4`, `runner`, `ledger` and `embed` call in cells 6–8 matches, and every name they use is defined in cells 1–5.

## Medium: clear-cut bugs

**M1. A resumed D60 round halts the chain falsely after the freeze.**
- **Where:** `verify_log` compares the whole recomputed H decision (`s34.py:749`), which includes the timing fields `c0/c2/rq3_seconds_per_task` (`:699–701`).
- **Scenario:** an H unit is repaired, for example its cache tail never reached Drive.
  1. The runner re-runs it from the cache, which rewrites `run.wall_seconds` and the call counts.
  2. Its cost changes, so the recomputed decision differs.
  3. `ChainHalt` follows, and the same happens in every later session.
- **Probe:** a 1.3× cost change on an H unit raised `ChainHalt`; the same change on a B unit did not, because B compares only switches and acceptance.
- **Fix:** compare only acceptance, wiring, switches, `fired` and `again`.

**M2. One wiring unit without a measurable cost crashes the gate permanently.**
- **Where:** `unit_metrics` returns `cost_seconds=None` for a unit rebuilt entirely from its cache (`:144–149`). `_costs` then computes `max()` over the two RQ3 cells (`:617`, and `:701` in H).
- **Scenario:** a session dies right after a C unit's result was written; Drive keeps the cache but not the result. The next session repairs the unit by full replay.
- **Probe:** `TypeError` at D in two sessions running; the chain never gets past C.
- **Fix:** take the wall time of the unit's original run from the ledger's `unit` event, or use the other cell's cost. Never pass `None` into `budget_gate`.

**M3. Any failed unit records a permanent "code fix needed" pause, including transient platform failures.**
- **Where:** `run_units` turns a `RunHalted`, or a unit that failed twice, into `failed`/`not_done` (`make_nb05.py:123–133`), and `_phase` records a pause (`s34.py:519–522`).
- **Scenario:** the CUDA context dies (an illegal memory access or an Xid error on a Colab T4).
  - Every later call in that process fails, so two units fail with the same error and D130 halts.
  - The chain pauses for good, although a fresh runtime would succeed.
- **Fix:**
  1. On the first occurrence, stop without a pause entry (an ERROR state).
  2. Let the next session run `run_phase(retry_failed=True)` once in its fresh runtime.
  3. Pause only if the same units fail again.

**M4. A pause is terminal: there is no way to record the user's decision and continue** (`s34.py:480–482`).
- The overshoot, ladder-exhausted, D57, wiring and epoch pauses all need either a new commit (so a new chain version, re-running from A) or hand-editing `chain.json`.
- **Proposal:**
  - A user-decision file on Drive, or a notebook setting such as `USER_DECISION = {...}`, appends a `U` log entry (the pause it answers, the decision, any settings) and clears the pause.
  - `verify_log` replays it.
- This needs your decision, but without it every pause costs about 10 GPU-hours.

## Medium: decisions for you

**M5. The second gate prices C2-mapped work at the pilot's default configuration.**
- **Where:** `_costs` uses the final pilot round's C2, the default k = 3 (`:612–622`; spec section 7).
- **Problem:** step F has just measured the selected configuration directly.
- **Size:** if k = 5 is selected:
  - That is 2 more notes of up to 120 tokens each per call, about +0.2 s per call (b_in 0.99 ms), roughly +6%.
  - It applies to about two-thirds of the remaining work: C1–C3, C2-OU, A1, A2, F, AN and half of SP.
  - Total: about 1 hour of 60, enough to flip a cut decision near the line.
- **Recommendation:** in the second gate, use the selected configuration's F cost per task for every arm priced at C2's cost.

**M6. With two accounts (D164), a session can start from a stale Drive view.**
- **Problem:** `chain.json` and the ledger are read once in the setup cell. If the other account's last session hasn't propagated yet:
  - the session redoes steps on an older log;
  - the gate sees fewer used hours;
  - and `write_once` can collide with a different freeze.
- "Never run both at once" isn't enough: switching accounts one after the other also needs the files to propagate first.
- **Fix:**
  - The bundle reports the log length and the `chain.json` sha256.
  - The notebook gets `EXPECT_LOG = n` from the last bundle and refuses to start while the log on Drive is shorter.

## Low

1. **H's RQ3 re-run uses the default configuration, not the selected one.**
   - `_h_units` passes `rq3_units(phase, s)` with no k (`:685–686`).
   - With k = 5, the "unfinished at T_max" re-check after a D60 switch is too lenient. Pass `CONFIGS[selected]` to the C2+ACC unit.
2. **Cell-6 exceptions skip the bundle and can leave the heartbeat running.**
   - `T4Session` starts the heartbeat before the canary (`t4.py:707–716`). A canary repeat mismatch raises with the heartbeat still on, so the ledger keeps counting hours while the VM idles.
   - "Run all" also stops there, so there's no bundle.
   - **Fix:** `T4Session` should close itself on any exception during start-up, and cell 6 should catch it and set `STOP`.
   - The same applies to an interrupt during cell 7.
3. **`verify_log` gaps.**
   - It doesn't recompute F's D60 switch decision (`:744–746`).
   - D could be recomputed from its recorded inputs (`used`, `costs`).
   - It doesn't check that the freeze file's sha256 still equals `freeze_sha`.
4. **`CHAIN_VERSION` is hand-edited and defaults to 1** (`make_nb05.py:32`).
   - At a fix commit, forgetting to change it pins the old commit, and the refusal message sends the user to the old link.
   - Generate the value into the notebook instead.
5. **`RELEASE_RUNTIME` defaults to `False`.**
   - Colab keeps charging the idle runtime after the bundle downloads, but the ledger stops counting.
   - Spec section 7 counts used hours "as Colab charges it".
6. **A failed `verify_replay` on an A, B or F unit only appears in the bundle.**
   - The chain goes on to the freeze and the selection anyway.
   - Pausing before E and F is cheap (`checks()` already exists). This is your call.
7. **`long_share` is a sampling-weight target, not the realised share.**
   - On dev streams 1000–1010 the realised long share is:

     | `long_share` setting | Realised long share |
     |---|---|
     | natural (`None`) | 0.26 |
     | natural − 0.1 | 0.195 |
     | natural + 0.1 | 0.355 |

   - Report the realised share for each pilot round.

## Answers to the code-review questions

1. **Exact rules:** yes.
   - Exact fractions are used at 5%, 3 percentage points, 1%, 0.2 and 0.8.
   - The wording and selection tie chains match D156 and D163, including skipping the API-call step when no task is jointly solved.
   - The ladder (direction from judged rounds only, bounds, overshoot) matches.
   - The cut order matches, a cut that saves nothing is skipped, and the D160 task-count step and D60 pooling match.
2. **Lost sessions:** every crash window is idempotent. That includes a crash between writing the freeze and recording it, and one between saving the selection and recording it. The exceptions are M1 (a false halt), M2 (a crash loop), M3 (a false permanent pause) and M6.
3. **Will the cells work on Colab?** Yes; my real-runner run covered cells 7–8. The two remaining gaps are L2 and L4.
4. **Is the budget projection sound?** Yes, apart from M5 and these:
   - AN isn't conservatively priced. Its prompt carries every note: about 8–10 notes against the top 3, so +500–700 tokens per call, roughly +20% for AN, about 0.3 h. This is your call under D159.
   - Caches discarded after an epoch change aren't projected; the design lists them. State the omission.
   - The idle tail isn't counted (Low 5).
   - The exposure check for 16–19 tasks always holds by construction, because the rules only concern positions below 15.
5. **Test fidelity:**
   - The chain tests inject metrics through a fake. It never produces `None` costs, repaired units, a real `not_done`, or a real halt. That is why M1–M3 went unseen.
   - The spec's "chain end to end on CPU with ScriptedBackend" test doesn't exist yet. My run shows it's feasible; add it as a test.
   - One catch: a single `ScriptedBackend` asserts unique task instructions across units, so a ladder step trips its assertion ("task instructions must be unique"). The test must use a fresh scripted backend per phase. This affects only the test harness, not Colab runs.

## Answers to the spec (r2) questions

1. **Does the spec contradict or omit a design rule?** No contradiction. The deviations are justified:
   - The D60 C0 re-run stays at 20 tasks.
   - The projection uses wall seconds per task.
   - There is no snapshot version of the projection (D151).

   Omission: discarded caches after an epoch change.
2. **Loops and deadlocks:** everything is bounded. At most 2 switch rounds, 3 ladder steps and 2 D60 rounds, plus `max_actions`. The real issue is that pauses never resume (M4).
3. **Bias:** none found.
   - Selection runs on the disjoint streams 1007–1010 under frozen settings, with a seeded order and a separate cache per unit.
   - `prefix_exposure_ok` reads the *structure* of the test streams, never their outcomes. Worth saying so in the spec.
4. **Budget:** see code-review answer 4.
5. **[delegated] choices:**
   - I agree with switch-before-acceptance, the knobs applying to every stream kind, re-running the RQ3 check after a D60 switch, and pricing every RQ3 cell at the larger cell.
   - On "a code fix restarts the chain from A", consider adopting the old results by replay instead:
     - `verify_replay` at the new commit ignores `code_commit` and needs no GPU.
     - So a fix that doesn't change behaviour (in `s34.py`, the notebook or the gate arithmetic) would re-run nothing.
     - That matches D158's "re-runs what it touches" without guessing which steps a fix touches.