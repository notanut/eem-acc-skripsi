Reviewing round 7: pulling `main`, reading spec r10 section 25, diffing against 7a31978, then re-running my 6ad55ac re-pin harness against the new code.

I've reviewed 7788996. There is no High or Medium finding this round, only 3 Low, and the re-pin from 6ad55ac to this commit still behaves as intended. All 164 tests in the three S3.4 test files pass.

Last round's three findings are fixed:
- **M1 (rule code in the notebook):** `session_stop` and `make_chain` keep the old behaviour; the cost profile still uses the `s34` prefix and the GPU-hours still come from the ledger. Hashing the notebook per code cell works: between 6ad55ac and this commit only the two cells that really changed differ, so markdown and `NB_VERSION` edits don't pause.
- **L1 (backend `Z` after a restore):** my round-6 test now clears the `Z` when the original wheel is restored, and the D57 pause re-opens.
- **L2 (GPU name recorded as "none"):** `session_env` retries the query and stops the runtime with a clear message instead of writing "none".

## Q1: Is anything that decides results or rules still outside the frozen material?

Only small things.

- **L1. Two config files are read at run time but not frozen.**
  - `configs/canary_prompts.json` (`canary.py:17`, `:80-83`) decides which prompts the canary runs, and so the epoch. If it changes, `EpochStore.resolve` registers a new epoch and the session stops with "the canary put this runtime in epoch 1". That reads like a platform change, so the user could allow it without realising the prompts changed.
  - `configs/t4_pins.json` (`make_nb04.py:91-92`) sets the library versions. Most of them show up in the embedder id and so in the frozen backend, but numpy does not.

  Fix: add both files to `PROTOCOL_DATA`.
- **L2. One determinism check still depends on code that isn't frozen.** `t4.T4Session._rerun` and `_gen` (`t4.py:751-756`) produce the canary fingerprint that `Runner._session_mismatch` (frozen) compares when a resume mismatches (D61). If `_rerun` were edited to return the stored fingerprint, nondeterminism within a session would go unnoticed. Fix: move these two methods into `canary.py`.
- **L3. The notebook actually running in Colab is matched only by `NB_VERSION` (`make_nb04.py:78`).** If someone edits a code cell in their Colab copy (say, following a chat suggestion), `NB_VERSION` still matches. The session then runs code that is neither the frozen notebook nor recorded anywhere.
  - Option 1: in the setup cell, compare the running notebook's code cells with the checked-out notebook, using the same per-cell hashes. Colab can read the running notebook with `google.colab._message.blocking_request("get_ipynb")`.
  - Option 2: treat "edit only through commits and re-pins" as a working rule. This one is your call.

There is also one decision for you. Python and numpy versions are recorded (in the headers and the E entry) but never compared after the freeze. The risk is small: the simulator's seeds come from numpy's `SeedSequence`, which is stable across versions. Adding `numpy` and Python's major.minor version to the per-session backend check would be cheap if you want it.

## Q2: Is the 6ad55ac → 7788996 re-pin still "no pause before E, exactly one `Z` after"?

Yes. I checked out 6ad55ac again, built the same 14 end states with its code, and re-pinned each one with 7788996:

- **Before E, no pause.** States covered:
  - interrupted during A, B or C, or before the freeze file was written;
  - a pending retry in B, and an X pause in B;
  - the ladder pause and the first budget-gate pause (D57), both of which stay open.
- **After E, exactly one `Z`.** States covered:
  - the freeze file written but E not recorded;
  - F interrupted;
  - a pending retry in F, and an X pause in F;
  - the final D57 pause, which re-opens after the accept;
  - the chain already finished.

I then repeated the test with a frozen backend that includes the GPU name, as the real 6ad55ac freeze does:
- The single `Z` lists both the backend change (the GPU name) and the code changes, including the notebook as newly frozen.
- After accepting it, two more sessions at the same commit run without another pause.

No finished unit was re-run in any state.

The caveat from last round still applies: session 1 runs 6ad55ac's version of `session_env`. If its GPU query failed, its unit headers say `gpu: "none"`, and the re-pin will re-run those units instead of adopting them. Before you re-pin, check the `gpu` field in one of session 1's unit cache headers.