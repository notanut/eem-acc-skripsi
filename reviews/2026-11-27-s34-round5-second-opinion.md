Reviewing round 5: pulling `main`, then reading spec r8 sections 2 and 23 and the diff since 38d1fbf.

I've reviewed b4ad2ad (`HEAD` 6ad55ac only adds the handoff note). No High findings: 2 Medium and 4 Low. Each finding comes from a probe I ran.

All 152 tests in the three S3.4 test files pass. I re-ran my round-4 probes on this code: round 4's M2, M3 and L1 are fixed.
- The D57 pause re-opens after a second re-pin, and after a re-pin that reverts the change.
- A re-pin plus a decision in the same session now pauses with "10 finished units replay differently".
- An X pause answered by both a re-pin and `rerun` now gives a `Z` instead of a halt.

## Medium

**M1. Code that changes results or rules sits in the exempt list (`INFRA`), outside `MECHANICS`.**
These modules are neither frozen nor recorded in the addendum, yet they change results or rules:
- **`llm/cache.py`.** `UnitCache.call:153-154` (`rec["key"] != key` → `ReplayDivergence`) is the replay check itself, and its output is what the agent reads on resume. `CachedEmbedder` decides which embeddings are served.
- **`llm/ledger.py`.** `summarise` and `profile` supply the budget gate's used GPU-hours and session costs (`make_nb05.py:146-147`), so they decide the cuts and the task count.
- **`llm/canary.py`.** It holds the epoch rule, and `session_env` supplies `gpu` to `backend_sha256`.

Scenario: after the freeze, an I/O fix in `UnitCache.call` drops `or rec["key"] != key`. A unit whose prompts changed then replays as "identical", gets adopted, shows no divergence, and the chain never pauses and records nothing.

I suggested these three as infrastructure in round 3; on a closer look they are on the result path.

Fix: add `llm/cache.py`, `llm/ledger.py` and `llm/canary.py` to `PROTOCOL_FILES`. Put their pure I/O helpers (`append_line`, `archive`, `zpack`, path helpers) into `MECHANICS`. Keep `fsio`, `t4` (its backend is now frozen through `backend_sha256`), `embed_setup`, `unit_worker` (S3.4 doesn't use it) and `utils` as infrastructure.

**M2. An X pause recorded before the freeze check can't be answered with `rerun` (clear-cut). `s34.py:806` → `:922` → `:705`.**
Sequence:
1. A re-pin past the freeze changes frozen code.
2. `verify_log` re-runs units that weren't adopted, and they fail twice (e.g. a Drive outage).
3. The X pause is recorded at the new commit before `run()` reaches the freeze check, so there is no addendum yet.
4. `rerun` → `apply_user_decision` runs `_check_freeze_after_repin`. The code changed, so it builds the `Z` entry, which calls `next_step()` on an open X → `ChainHalt: a pause for a code fix is still open`.

Probe:
- This repeats every session while `USER_DECISION` is set.
- Without the decision, the X stays open.
- The only way out is an extra empty re-pin, and the message doesn't say so.

Fix: skip the freeze check in `apply_user_decision` when the open pause is a fix pause, as `run()`'s paused branch already does. After `rerun`, `run()` re-runs the units and raises the `Z` after `verify_log`.

## Low

- **L1. `backend_sha256` changes when the GPU query fails, though nothing changed (Q3). `canary.py:218-223`.** `session_env` makes its own `nvidia-smi` call with a 10 s timeout, and any timeout or error records `gpu: "none"`. The setup's T4 check is a separate call, so the session still runs.
  - If that happens in the session that writes the freeze, every later re-pin sees "Tesla T4" ≠ "none" and pauses with `Z`.
  - If it happens in a later re-pin session, that re-pin pauses once.

  Fix: leave `gpu` out of `backend` (the setup already requires a T4), or retry the query and halt if it still fails.
- **L2. The backend is compared only at re-pins (Q3).** `_check_freeze_after_repin` returns early when there is no new commit, so a change between sessions at the same commit is never compared. It is caught at the next re-pin and blamed on that re-pin's commit, after units have already run under the new backend.
  - The realistic case is a rebuilt wheel. It happens when the Drive cache is lost, or when the user follows the wheel cell's own hint to delete the cached wheel. pip wheel builds aren't byte-reproducible, so `wheel_sha256` changes while the canary, and so the epoch, stays the same.
  - Fix: compare `backend_sha256` every session after the freeze.
  - **Your decision:** keep `wheel_sha256` in the frozen backend (my recommendation: it is a real binary change; the `Z` text can say "wheel rebuilt, canary epoch unchanged"), or drop it and rely on `llama_cpp_version` plus the epoch.
- **L3. Two wrong no-ops in `apply_user_decision` (Q2). `s34.py:792`.**
  - A decision with no `"answers"` key matches any `Z` whose `resume_pause` is `None` and is silently skipped. The probe ends still paused at the `Z`, with no error. Fix: require `answers` before the no-op checks.
  - If the user answers the original pause number after the `Z` was already accepted, the log still says "waits: the Z came first". It should name the re-opened pause's number.
- **L4. `write_once` and `Chain.save_once` should not be mechanics (Q1).** They are the immutability guards for the freeze record and the F selection (`s34.py:1202-1203`, `:1225`). Weakened, they would let a re-run F overwrite `selection` after a crash between saving it and recording F. They never need to change, so making them pause costs nothing.

## Answers

**Q1: Is anything left in `MECHANICS` that can change a result or a rule?**
Only `write_once` and `save_once` (L4). Lease, claims and heartbeat, `error_signature`, paths, progress text and the exception classes can only stop or serialise runs; they can't change a result. The bigger gap is the exempt infrastructure list (M1).

**Q2: Any crash window, stuck state or wrong no-op in `apply_user_decision` / `_reopen`?**
- **No crash windows.** `_reopen` writes a single record. A crash after the addendum but before `_reopen` only leads to a duplicate addendum when the accept is re-sent. Crashes between adoption, clearing and recording all resume cleanly.
- **Re-opening works.** A pause re-opened twice points back to the original.
- **Stuck state:** M2.
- **Wrong no-ops:** L3.

**Q3: Can `backend_sha256` change between Colab sessions of the same epoch without a real change?**
Yes, in two ways:
- **`gpu` falls back to `"none"`** when the GPU query fails (L1).
- **A rebuilt wheel changes bytes but not outputs** (L2).

Everything else is stable:
- Load parameters read back as plain ints and bools, identical across processes (checked with the tiny model on the CPU build).
- The model digest is enforced against S3.0.
- The embedder id is built from pinned versions plus the CPU-paths tag.
- The driver, CUDA and CPU model fields are already excluded.