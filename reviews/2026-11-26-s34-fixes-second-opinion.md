Now the notebook cells.

**Verdict:** no High. The retry rule, user decisions and re-pin hold up against a runtime dying at any line. Every window I traced redoes the step with the same result, apart from a harmless duplicate adoption record. Four Mediums remain:
- Adoption trusts the unit's old platform identity.
- A re-pin after the freeze can change frozen material without stopping the chain.
- `accept` is allowed for a wiring failure that is a real bug or a leak.
- A lost Drive write turns `EXPECT_LOG` into a permanent refusal.

**What I checked:**
- **Tests:** after pulling `1d15135`, 138 tests pass (`test_s34`, `test_s34_platform`, the notebook static tests, `test_s33b_t4`).
- **Probes:**
  - A wiring pause caused by audit hits can be accepted.
  - Every `long_share` value the user is allowed to set (0 to 1) can be sampled on all 31 planned streams.

## Medium: clear-cut

**M1. Adoption checks behaviour only, so a fix to the platform setup is adopted silently** (`s34_platform.py:91–97`).
- **What adoption uses:** `verify_replay` builds its key source from the unit's *old* cache header (`llama_backend.py:211–223`) and copies the stored `stamp.llm` (`runner.py:830`).
- **What it misses:**
  - **The load cell.** If the fix is in the notebook's load cell (e.g. `n_batch`, `n_ubatch`, flash attention), every replay still matches, so nothing re-runs. Fresh runs would differ.
  - **The embedder.** Embeddings are never recomputed, so a fix to `MiniLMEmbedder`, `CPU_PATHS_ENV` or the rounding is adopted the same way.
  - **Cache records.** Anything that *wrote* the cached records is also trusted, e.g. how `finish_reason` or `n_out` were recorded. The switch metrics come from those records.
- **Why this matters:** adoption is the only path that skips the S3.3a header compatibility check.
- **Fix:** adopt only if the unit's cache header matches the current session's `SES.header` on every compatibility field except `code_commit`/`code_dirty`.
- **Spec section 2 limit:** widen it to "code that computed what the cache stores (backend `generate`, record writing, the embedder)". Note that `key_source_from_header` already catches changes to `SAMPLING`, the llama.cpp version, the template and the grammar.

**M2. A re-pin after the freeze can change frozen material without stopping the chain.**
- **Where:** `_do_E` keeps the old freeze (`s34.py:894–899`), and `verify_log` checks only the freeze file's sha256 (`:1022–1025`).
- **Scenario:**
  1. After E, a fix changes the system prompt, the docs or the templates.
  2. Replay diverges, so the affected units re-run.
  3. If the recomputed decisions happen to match, the chain carries on.
  4. The freeze still records the old `system_prompt_sha256` and template set, while the test phase runs the new ones.
- **Fix:** after a re-pin past E, recompute `freeze_extra` at the new commit. Stop for your decision if any frozen hash, or the settings, differs. Otherwise record a freeze addendum naming the new commit; today only `repins` and `adoptions` document it.

**M3. `accept` is allowed after a C or H wiring pause even when the wiring isn't sane** (`s34.py:55`, `:650–651`, `:663–664`).
- "Not sane" means `verify_replay` failed or the scrub audit found leaks. D111 lets T_max change after the wiring check; anything else needs a code fix.
- **Probe:** `checks` returned 3 audit hits, the chain paused, `accept` was recorded, and the chain ran on to G.
- The replay check before E doesn't cover the C units, so nothing else catches it.
- **Fix:** allow `accept` only when the wiring is sane but tasks were unfinished at T_max. Otherwise allow only `rerun` or a re-pin.

**M4. A Drive write that is lost for good makes `EXPECT_LOG` refuse every later session.**
- **Scenario:**
  1. Cell 8 writes `chain.json` and the bundle, and the browser downloads the bundle.
  2. The VM dies (a disconnect, or `RELEASE_RUNTIME`) before drivefs uploads `chain.json`.
  3. Drive keeps the shorter log, but the chat's `EXPECT_LOG` comes from the bundle.
  4. `check_expected_log` raises `StaleView` ("wait a few minutes") in every session; the missing data never arrives.
- **Fix:**
  - End cell 8 with `drive.flush_and_unmount()` (it blocks until uploads finish).
  - Document the way out: if the refusal persists, the chat clears `EXPECT_LOG`, and the chain recomputes the lost steps. They are deterministic, except that D sees a different number of used hours.

## Low

1. **A user's `settings` can turn the D55 switches on or off** (`:641–647`, `_check_settings :1031–1039`).
   - At a ladder pause, only the knobs are in question.
   - Turning a switch off is self-correcting (the next B round fires it again), but turning one on skips its rule.
   - Restrict `settings` to `n_distractors` and `long_share`. `wording` is overwritten by A anyway.
2. **Units a D130 halt never ran count as already retried** (`:585–597`).
   - The halt leaves the rest of the phase `not_done`, and all of them go into the attempt.
   - If one fails for a transient reason in the retry session, the chain pauses without that unit's own retry, against the spec's "units failing for the first time get their own retry".
3. **While paused, the recomputation after a re-pin is deferred** (`:514–518`).
   - `run()` raises the pause before `verify_log`. So the bundle the user answers from doesn't show whether the fix changed a decided result, and the halt comes only after their answer is in the log.
   - Run `verify_log` before raising when this session adopted anything.
4. **Without D60, RQ3 is still priced at the default configuration.**
   - The C wiring units ran k = 3. D167 reprices C2-mapped work, but not the RQ3 C2 cells (`:847–849`).
   - Scale RQ3 by the ratio of the selected configuration's C2 cost to the pilot's C2 cost.
5. **Verifications run twice.**
   - The replay check before E and F verifies units, then `verify_finished` verifies the same units again at the end of the session (CPU minutes on every check).
   - Keep the outcomes keyed by payload sha256 in `verified`.
6. **A deterministic exception outside the unit runner repeats in every session.**
   - Examples: a corrupt cache read in `metrics`, or a plan mismatch.
   - It shows as ERROR every session with no pause. Consider pausing after two ERROR sessions at the same log fingerprint.
7. **Stale plan hashes after a re-pin.** After a re-pin, the freeze's `plans_sha256` and `payload_sha256` refer to the old plan versions. S3.5 should read the commit from the chain's stamp and `repins`, not from the freeze.

## Answers to your questions

1. **Crash windows:** idempotent everywhere I traced.
   - **`_units`:**
     - A crash before saving the retry record costs one extra session.
     - A crash after a retry but before `retried_by` is saved grants one harmless extra retry.
   - **`adopt_repinned`:** a crash before `write_plan` redoes the phase, leaving duplicate adoption lines, which are harmless.
   - **`_clear_fixed_pause` and `apply_user_decision`:** each writes a single log entry, and a re-applied decision is a no-op.
   - **`_do_E`:**
     - The redone freeze content is identical.
     - The re-pin case keeps the old freeze.
     - An "immutable and differs" halt is possible only if a repaired result changed (nondeterminism).
   - **Recomputing decisions on resume:** they don't change, since D is recomputed from its recorded inputs and F and H from their stored results.
   - **Stuck states:** M4 and Low 6.
2. **Can a user decision or a re-pin skip a rule or freeze inconsistent settings?**
   - **Skipping rules:**
     - `accept` after a wiring failure (M3).
     - A user's settings turning switches on (Low 1).
   - **Inconsistent freeze:** a re-pin after E (M2).
   - **Where the guards hold:**
     - A user gate keeps all earlier cuts and never raises the task count.
     - After a user's settings, a miss pauses again.
     - A fix pause accepts only `rerun` or a re-pin.
3. **Is adoption sound?**
   - Mostly. A result that would not reproduce can still be counted as done when the fix touches the platform setup or the embedder rather than the code `verify_replay` re-runs (M1).
   - So the spec's limit is not the only one. The embedder and record writing are two more, and the load cell is the third gap. The header-compatibility condition closes all three mechanically, except the backend's `generate`.
4. **Is the retry rule right?**
   - Yes: first failures always get one retry in another runtime. A kernel restart keeps the runtime id and is correctly refused.
   - Persistent failures pause rather than loop.
   - Exception: halt-skipped units (Low 2).
   - Exceptions raised outside the runner are retried by simply running again, with no counting (Low 6).
5. **Will the notebook cells work on Colab?** Yes.
   - **Settings (0):** `NB_VERSION`, `USER_DECISION`, `EXPECT_LOG` and `REPIN` are all read where expected.
   - **REPIN:**
     - It is checked as a descendant after `git fetch origin`.
     - It is applied before the notebook-fingerprint check, so the user must open the fixed commit's notebook, which is consistent.
     - Leaving it set afterwards is a no-op.
   - **Setup cell:** `Sections` checks `EXPECT_LOG` before `ensure_embed` writes anything.
   - **Cell 6:** a start-up failure closes the heartbeat and sets `STOP`.
   - **Cell 7:** applies the decision before `run()`. Nothing earlier in the session touches the log, so the fingerprint still matches exactly.
   - **Cell 8:** always closes the heartbeat and writes the bundle. Add the flush (M4).
6. **Should `RELEASE_RUNTIME` default to on?** Yes, on the condition below.
   - **Benefits:**
     - Free-tier quota.
     - Colab's idle tail stops being uncounted.
     - The next session is always a fresh runtime, which the retry rule needs.
   - **Condition:** call `drive.flush_and_unmount()` before `runtime.unassign()`. Otherwise killing the VM 20 s after the last writes can lose them (M4).
   - **Two smaller costs:**
     - `files.download` is asynchronous and the browser may block a second download, so treat the Drive copy as canonical.
     - The kernel is gone, so a failed cell 8 can't be re-run (its output stays visible).