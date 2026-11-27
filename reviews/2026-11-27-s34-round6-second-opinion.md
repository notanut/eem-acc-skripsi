Reviewing round 6: pulling `main`, reading spec r9 sections 2 and 24, and diffing against b4ad2ad, with the 6ad55ac → new-commit re-pin as the main focus.

I've reviewed 7a31978. No High findings: 1 Medium and 2 Low. All 157 tests in the three S3.4 test files pass. My round-5 probes confirm the fixes: an X pause answered with `rerun` re-runs the units and then raises the `Z`, and a decision without `answers` is refused.

## Q1: Is the 6ad55ac → 7a31978 re-pin safe in every end state?

Yes, with one caveat (L2). I checked out 6ad55ac, used its own test chain to reach 13 end states, then re-pinned each one with 7a31978's code:

- **Before E: no pause; the chain continues as before.** States covered:
  - interrupted during A, B or C, or before the freeze was written;
  - a pending retry in B, and an X pause in B (the re-pin answers it);
  - the B ladder pause and the first-gate D57 pause, both still open with no `Z`.
- **After E: exactly one `Z`, accepted once.** States covered:
  - the freeze file written but E not yet recorded;
  - E done with F interrupted;
  - a pending retry in F, and an X pause in F;
  - the final D57 pause, which re-opens after the accept;
  - the chain already finished.

  The `Z` lists cache.py, canary.py and ledger.py as newly frozen, plus the s34 and platform edits. On the real platform it will also say "backend: gpu", because the old freeze included the GPU name.
- **Adoption should pass on the real platform.** Between the two commits only `s34.py` and `s34_platform.py` changed, and only in chain and freeze code, so replay of finished units is unaffected. No unit was re-run in any state.
- **Operational:** open the 7a31978 notebook. The pinned-notebook check stops an old copy.

## Q2: Anything left that affects results or rules but is neither frozen nor recorded?

**M1. Rule code still lives in the notebook. `tools/make_nb05.py:98`, `:123`, `:146-147`.**
Cell 6 holds two session rules:
- the epoch rule: stop unless the epoch is 0 or allowed (`SES.epoch != 0 and ... not in allowed_epochs`);
- the D146 strict kernel-probe stop, whose comparison lives in `t4.kernel_probe_check`, which is not frozen.

Cell 7 wires the chain's inputs: `summarise(ROOT)["gpu_hours"]` (with its default `budget_from`), `profile(ROOT, "s34")`, and which `metrics` and `checks` functions are passed. The notebook is pinned to the commit, but its code is not in the protocol snapshot and isn't recorded in the addendum.

Scenario: a re-pin after the freeze changes `profile(ROOT, "s34")` to `profile(ROOT)`. S3.3b's benchmark units then enter the cost profile, the final gate's projection changes, and so do the cuts. The addendum records `changed: {}` and nothing pauses. A cell-6 edit that lets epoch 1 through would also pass silently.

I missed this in rounds 4 and 5: I only checked the notebook's model-load line.

Fix, either:
- move these into frozen code (e.g. `Platform.open_chain(...)` and `s34_platform.session_stop(SES, S, KP, decision)`), so the cells only call them; or
- add the notebook's code cells to `PROTOCOL_DATA`. Then any edit to the notebook text pauses.

I prefer the first option.

Everything else in the exempt infrastructure list is fine:
- `utils/*` isn't imported by Stage 3.
- `agent/scripted.py` is used only by tests.
- The backend that `t4` builds is now frozen through `backend_sha256`.

## Q3: New crash window or stuck state?

No crash windows and nothing stuck:
- The per-session backend check reads no Drive files when the backend is unchanged. When it differs, it falls into the existing check, which makes one write per path.
- The loop that finds the last open pause always moves forward, so it can't run forever.

Two Lows:

- **L1. A backend `Z` can't be cleared by restoring the backend, and accepting it records the wrong backend. `s34.py:838`, `:902`.** In my test chain:
  1. Session 2 has a rebuilt wheel (B), so a `Z` is raised.
  2. Session 3 restores the original wheel (A). The `Z` stays open because fix pauses aren't re-checked.
  3. Accepting it records backend **B** in the addendum, which is no longer in use.
  4. Session 4 (A vs B) pauses with another `Z`.

  The way out is to accept twice, leaving a misleading record. Fix: on accept, record the session's current `backend_extra()`. Also clear a backend-only `Z` automatically when the backend matches the last accepted one again.
- **L2. `gpu` still blocks adoption and resume. `runner.py:549-555`, `s34_platform.py:123-137`, `canary.py:218-223`.** The GPU name left the frozen backend, but `Runner.compat` and `header_diff` still compare it. If `session_env`'s `nvidia-smi` query fails, that session's units carry `gpu: "none"`: later re-pins re-run them instead of adopting them, and their partial caches never resume. This is the Q1 caveat: if session 1 hit it, the next session re-runs its units. Fix it at the source: retry the query, and raise on a GPU runtime that still returns nothing. To check session 1 before re-pinning, look at the `gpu` field in the header of any one of its unit caches.