**Verdict:** no High. The round-2 fixes hold, and the new freeze check, `Z` pause and retry rule add no crash window, stuck state or loop. Two Mediums:
- The freeze check compares too little, so a fix that changes frozen behaviour gets recorded as "unchanged".
- The header condition for adoption is stricter than it needs to be. One RQ3-only fix would force every RQ1 unit to re-run.

**What I checked:**
- **Tests:** after pulling `c33fbae`, the 116 tests in `test_s34`, `test_s34_platform` and the notebook static tests pass.
- **Probe:** I changed frozen material after a freeze and compared the frozen-material fields (`FREEZE_HASHES`) before and after.

## Medium

**M1 (clear-cut). The freeze check misses most of the frozen behaviour** (`s34.py` `FREEZE_HASHES`, `_check_freeze_after_repin`).

- **What it compares:** the suite version, the template set, the system prompt of the frozen wording, and the natural long share.
- **Probe:** each of these changes left the compared fields identical, so the check would record an addendum saying "frozen material unchanged":
  - the reflection instruction;
  - the memory header;
  - the A2 template;
  - `MAX_STEPS_EXTRA`, a stopping rule.
  - `P.prompt_hash()` did change for the first three.
- **Scenario:**
  1. After E, a re-pin edits the reflection instruction.
  2. The memory-condition units re-run.
  3. If the recomputed decisions happen to match, the chain goes on and the test phase runs a different protocol from the one frozen.
- **Not covered either:** `PARSER_VERSION`, `RQ3_VERSION`, `SIM_VERSION`, the `LoopConfig` defaults, `CONFIGS`, and the pre-registered rule code itself (thresholds, margins, tie-breaks, cut order, `AN_FACTOR`). A re-pin before F could change the selection rule and nothing would notice.
- **Fix:** nothing is frozen yet, so the freeze format can still grow. Add these to `freeze_extra` and `FREEZE_HASHES`:
  - `prompt_hash`;
  - the parser, RQ3 and simulator versions;
  - a sha256 of `dataclasses.asdict(LoopConfig())` plus `MAX_STEPS_EXTRA`;
  - a sha256 of `CONFIGS`;
  - a "rules" sha256 over the s34 constants and the source of the rule functions.

**M2 (your decision). The header condition is complete, but too strict** (`s34_platform.py` `header_diff`, `ADOPT_FREE`).
- **Complete:** it covers everything replay cannot see:
  - the backend stamp (model, wheel, llama.cpp version, template, load parameters, sampling, decoding path);
  - the embedder id, torch and sentence-transformers;
  - the GPU.

  That closes the gap from the last round.
- **Too strict:** it also requires fields that replay already re-executes, or that change without any re-pin:
  - **Versions and hashes:** `prompt_version`, `prompt_sha256`, `parser_version`, `suite_version`, `rq3_version`, `sim_version`. These are session-wide.
    - An RQ3-only fix that bumps `RQ3_VERSION` refuses adoption of *every RQ1 unit*, although their replays are identical.
    - That is up to the whole chain's GPU time spent re-running results that already verify.
  - **`python` and `numpy`:** replay runs the unit's Python again, so a Colab patch update is already tested by replay.
  - **`platform_epoch` and `platform_fingerprint`:** once you allow epoch 1, done epoch-0 units stay done without a re-pin, but any re-pin forces them all to re-run.
- **Safe relaxation:** add those eight fields to `ADOPT_FREE` and keep the rest. The condition only has to cover what decides fresh generation and embedding.

## Low

1. **"Nothing runs while paused" isn't strictly true.**
   - The paused recompute loads phases through `run_units` → `run_phase`, so a unit whose result was lost since the re-pin would be *run* on the GPU before `_Deferred` is raised.
   - Pass a load-only flag (e.g. `max_units=0`) during `_paused_check`.
2. **The verify memo is keyed by payload sha256 and commit only.**
   - `verify_replay` also checks the cache's content hash and record count. A cache that changes later without the result changing (a repair, damage) would keep its memoised pass.
   - Add `run["cache_content_hash"]` to the key.
3. **`verify_finished` can raise `KeyError`.**
   - If `_payload_sha` returns `None` (an unreadable result), `self.verified[u.unit_id]` was never set.
   - Cell 8 catches the error, but the remaining units go unverified that session.
4. **If `drive.flush_and_unmount()` raises (it can time out with large pending uploads), cell 8 skips both the download and the release.**
   - Keeping the VM alive is the right call, but say so. Wrap the flush, download anyway, and skip `unassign()` with a message.
5. **At the first gate's D57 pause, a user gate may pick fewer cuts than the paused projection, even none, and up to 20 tasks.**
   - That's the "buy compute" decision, so it's not a bug. State it in spec section 11 so it isn't read as a hole.

## Answers to your questions

1. **Crash windows, stuck states, loops:** none found.
   - **Freeze check:** the addendum write is idempotent (the next session finds the addendum for the current commit).
   - **`Z` accept:** writes the addendum and then the `U` entry. A crash between them leaves a duplicate addendum when the decision is re-applied, which is harmless.
   - **Another re-pin during an open `Z`:** `_clear_fixed_pause` clears it. The next check compares against the latest *accepted* state, so it either records an addendum or pauses again.
   - **Retry rule:**
     - It stays bounded: a unit pauses once it's in `tried_failed`.
     - Units a halt left unrun get their own retry.
     - Units "not done" with no failure pause after one retry.
     - Older attempts without `failed` read correctly.
   - **Remaining caveat:** the paused recompute can run a unit (Low 1).
2. **Header condition:** complete for what replay can't see, but too strict (M2).
3. **What a user decision or a re-pin can still skip:**
   - **User decisions:** nothing. `settings` is limited to the knobs, `accept` requires a sane wiring, a gate keeps earlier cuts, and `Z` only takes `accept`.
   - **Re-pin:** it can change frozen behaviour outside the compared hashes, or future rules, with the chain still recording "unchanged" (M1).
4. **Will cell 8's new order work on Colab?** Yes.
   - The model, MiniLM and wheels are local copies, and the heartbeat is closed in `finally`. Nothing should hold the mount, so `flush_and_unmount()` can unmount cleanly.
   - Downloading the local copy after the unmount is correct.
   - **Caveats:**
     - From the second session in the same tab, Chrome may ask to "allow multiple downloads". If nobody clicks within 20 s, `unassign()` kills the download. The Drive copy is the canonical one, as the intro says, so nothing is lost.
     - A flush failure needs explicit handling (Low 4).