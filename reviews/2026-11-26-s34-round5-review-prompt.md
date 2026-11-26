# Second-opinion review request: S3.4 round 5 (your fifth second opinion on 38d1fbf, decision D175)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`, the commit named in the chat). Report High /
Medium / Low with file:line and a concrete scenario; separate clear-cut bugs from decisions for the user. Contract:
`docs/s34_spec.md` r8, sections 2 and 23.

What changed in `src/eem_acc/stage3/s34.py` / `s34_platform.py`:
- D175 (your M1): `MECHANICS` keeps only the runner's lease/claims/heartbeat/paths/progress text/exception classes,
  `write_once` and `save_once`; a fresh-context review also took out the platform's constructor and
  `verify_finished`, `Runner.mark` and `_next_or_none`.
- Your M2: `Chain._reopen` re-opens a pause interrupted by a `Z` in one record (from the `Z` accept and from
  `_clear_fixed_pause`); `next_step` halts on a `U` without a next step.
- Your M3/L1: `apply_user_decision` adopts and clears a re-pin first; a decision a re-pin or a `Z` already answered or
  pre-empted is a no-op.
- Your M4: `backend` / `backend_sha256` in the freeze (`Platform.freeze_extra`).
- Your L2: `sources_at` halts clearly on git errors; UTF-8 reads.

Questions:
1. Is anything left in `MECHANICS` that can change a result or a rule?
2. Any crash window, stuck state or wrong no-op in the new `apply_user_decision` / `_reopen` paths?
3. Can `backend_sha256` change between Colab sessions of the same epoch without a real change?
