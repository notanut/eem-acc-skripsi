# Second-opinion review request: S3.4 fixes after your code review (decisions D165-D169)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`, the commit named in the chat). Report High /
Medium / Low with file:line and a concrete scenario; separate clear-cut bugs from decisions for the user. The spec
`docs/s34_spec.md` (r4) is the contract; sections 2, 11, 13, 17 and 18 describe what changed.

What changed since your review of e3742ae (your findings M1-M6, Low 1-7, and the user's decisions):
- **Retries before pauses (your M3):** `Chain._units` in `src/eem_acc/stage3/s34.py`. First failures stop with
  `ChainRetry`; the next session, in another runtime, retries them once; only units failing again after their own
  retry pause (a step `X` log entry). Every recorded phase is loaded through the same rule (`Chain._load`), also
  inside `verify_log`, the replay check and F's pilot reload.
- **User decisions (D165, your M4):** `Chain.apply_user_decision` (actions per pause in spec section 11), notebook
  settings `USER_DECISION` and `EXPECT_LOG`, `allowed_epochs` for the epoch pause.
- **Adopt by replay (D166, replaces the new chain version):** notebook `REPIN` (setup cell checks it descends from
  the chain's commit), `t4.Sections(repin=True)`, `Chain.adopt_repinned`, `runner.record_adoption` and
  `Runner.stored_matches`; a fix pause is cleared by the re-pin (`_clear_fixed_pause`).
- **Gate pricing (D167, your M5):** `Chain._costs` (selected configuration's measured C2 cost after F; AN = 1.2 x C2,
  `Costs.an`, `workload`).
- **Replay check (D168, your Low 6):** `Chain._replay_gate` before E and inside F, before the selection is decided.
- **Stale Drive view (your M6):** `EXPECT_LOG` is checked in `t4.Sections` (`llm/fsio.check_expected_log`) before
  anything writes the chain file; the bundle carries the log fingerprint and the chain file's sha256.
- Your M1 (H compares non-timing fields), M2 (None-safe costs), Low 1 (H's RQ3 unit at the selected config), Low 2
  (T4Session closes on a start-up failure; cell 6 sets STOP; cell 7 catches an interrupt), Low 3 (`verify_log`
  recomputes D and F's switches, checks the freeze sha), Low 4 (`CHAIN_VERSION` fixed at 1), Low 5 (`RELEASE_RUNTIME`
  still off: see the question below), Low 7 (realised long share per pilot round).
- **The notebook's chain callbacks moved** to `src/eem_acc/stage3/s34_platform.py` (`Platform`), so tests run them.
- A fresh-context review of these changes (2 High, 5 Medium, 6 Low) is applied; spec section 18 lists it.

Tests: `tests/test_s34.py` (fake platform), `tests/test_s34_platform.py` (the chain end to end on the real runner
with the scripted backend: ladder, pause, user accept, C, gate, freeze with replay check, selection, final gate;
resume; a re-pin adopting every unit; a unit that no longer replays re-runs), `tests/test_notebook_static.py`,
`tests/test_s33_runner.py::test_adopted_result_counts_at_the_new_frozen_commit`, `tests/test_s33b_t4.py` (re-pin,
start-up failure).

Questions:
1. Is every crash window idempotent now (a Colab runtime can die at any line, including inside `_units`,
   `adopt_repinned`, `_clear_fixed_pause`, `apply_user_decision`, `_do_E`)? Can the chain loop, get stuck, or decide
   differently on resume?
2. Can a user decision or a re-pin let the chain skip a pre-registered rule, or freeze inconsistent settings?
3. Is adoption by replay sound: can a result that would not reproduce under the fixed code be counted as done? Is
   the limit stated in spec section 2 (backend generation code is not replayed) the only one?
4. Is the retry rule right: are transient failures always retried once in a fresh runtime, and do persistent ones
   pause instead of looping?
5. Will the changed notebook cells (0, 1's REPIN and Sections lines, 6, 7, 8) work on Colab as written?
6. Decision for the user (Low 5): `RELEASE_RUNTIME` defaults to off. The bundle is also saved on Drive, so ending the
   runtime after the download loses nothing; is there a reason not to default it to on?
