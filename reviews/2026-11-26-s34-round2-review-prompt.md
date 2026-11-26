# Second-opinion review request: S3.4 round 2 (your second opinion on 1d15135, decisions D170-D172)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`, the commit named in the chat). Report High /
Medium / Low with file:line and a concrete scenario; separate clear-cut bugs from decisions for the user. Contract:
`docs/s34_spec.md` r5; sections 2, 11, 13, 19 and 20 describe the changes.

What changed (your M1-M4 and Lows; user decisions D170 RELEASE_RUNTIME on, D171 RQ3 scaled to the selected
configuration, D172 no rule for repeated ERRORs):
- M1: `Platform.header_diff` / `Platform.adopt` in `src/eem_acc/stage3/s34_platform.py` (cache header must equal the
  session's on every compat field except the code commit).
- M2: `Chain._check_freeze_after_repin` in `src/eem_acc/stage3/s34.py` (addendum or a `Z` pause; baseline = latest
  addendum; also before F; carries a re-pin's retry).
- M3: `Chain.apply_user_decision` (no `accept` for a wiring that is not sane).
- M4 + D170: `tools/make_nb05.py` cell 8 (copy, `drive.flush_and_unmount()`, download, release); cells 6-8 check
  the mount.
- Lows: knobs-only user settings; the retry rule in `Chain._units` (failed vs not_done; halted counts as failed);
  paused recompute in `Chain.run` (only when every unit was adopted; nothing runs); RQ3 scaling in `Chain._costs`;
  verify memo (`Platform.verify`, `verify_finished`); adoption reasons; plan hashes after a re-pin (spec section 2).
- A fresh-context review of this round (1 High, 1 Medium, 7 Low) is applied (spec section 20).

Tests: `tests/test_s34.py`, `tests/test_s34_platform.py`, `tests/test_notebook_static.py`.

Questions:
1. Any crash window, stuck state or loop introduced by the freeze check, the `Z` pause, or the new retry rule?
2. Is the header condition for adoption complete and not too strict for sessions in the same platform epoch?
3. Any rule a user decision or a re-pin can still skip?
4. Will cell 8's new order work on Colab (flush, then download of the local copy, then `runtime.unassign()`)?
