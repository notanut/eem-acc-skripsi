**Verdict:** no High. The recovery logic in `ensure_embed` holds up against a kill or an interrupt at every step. One Medium: the outdated-notebook guard covers only a session's first runtime, not later ones. The rest is Low or a judgment call.

**What I checked:**
- **Tests:** after pulling `5a4222f`, the 63 tests in `test_s33b_embed_setup` and `test_s33b_t4` pass.
- **CUDA paths:** stripping the addresses turns session 1's 36 `ldd` lines into 3 paths, and the aborted runtime's paths equal session 1's.
- **Probes:** I ran the guard and the recovery paths through the repo's own fake Colab VM.

## Medium (clear-cut)

**M1. An outdated notebook copy is still accepted in every runtime after the first.**
- **Where:** `t4.Sections` refuses only a *new* session file without `nb_version` (`t4.py:40`). The `NB_VERSION` check lives in the new setup cell (`make_nb04.py:77–80`), which an old copy doesn't have.
- **What old copies do:** every copy since `c800cac` checks out the session's pinned commit and passes `stamp={"commit": …}` with no `nb_version`.
- **Scenario:**
  1. After a kill, the user reopens the notebook from Colab's Recent list.
  2. With commit-pinned links, every hand-off appears there as another "04_stage3.ipynb".
  3. If they pick the `acdc83e` copy, it runs against `5a4222f`. `ensure_embed` is compatible with it, so nothing errors, and that runtime skips the D146 and CUDA checks without saying so.
  4. Older copies fail loudly instead, the way the aborted attempt did.
- **Probe:** an existing session file opened with a stamp that has no `nb_version` was accepted.
- **Fix:** in `Sections.__init__`, when the file's stamp has an `nb_version`, require the caller's `nb_version` to be equal. The legacy `session1.json` has no `nb_version`, so it is unaffected.

## Low (clear-cut)

**L1. The CUDA reference is whichever entry was saved first, even if it failed.**
- **Where:** `cuda_checks` is appended before the stop (`make_nb04.py:250–256`), and `equal_first` compares against `prev[0]`.
- **Problem:** `cuda_fingerprint` saves a per-file read error as `[None, None, "OSError"]` (`t4.py:120–131`). "A failed read is never saved" holds only when the whole `cuda_libraries()` call fails.
- **Consequence:** a bad entry in runtime 1 becomes the reference that every later runtime fails against.
- **Practical risk:** small, since llama.cpp wouldn't have loaded with a missing library.
- **Fix:** save only entries with no read errors, and use the first passing entry as the reference.

**L2. An offline install that passes the dry run but fails for real would repeat on every new VM.**
- **What happens:**
  1. The failure comes after `pip_started`, so the runtime is refused.
  2. The next VM sees the same Drive state and passes the same pre-flight and dry run.
  3. It fails the same way. Only a human uploading the attempts file breaks the loop.
- **Likelihood:** low. Real `--dry-run` resolves but doesn't run the install phase, so file conflicts or a bad RECORD would get through.
- **Fix:** if any earlier attempt failed after `pip_started` in offline mode for this cache, set the cache aside and use `network_exact`.

**L3. Pre-flight runs outside the `try` and before the `start` event** (`embed_setup.py:383–398`).
- A Drive I/O error (`OSError`, not `SetupError`) during the wheel copy or `set_aside` comes out as a raw exception: no hint and no attempt record. Re-running is still allowed, so the only cost is the message.

**L4. The NB check calls a newer copy "OUTDATED"** (`make_nb04.py:78–80`).
- If a mid-session push changes the notebook and the user opens the new link, the refusal is correct, because the session is pinned. But the message should say "differs from this session's commit `<target>`", not "outdated".

**L5. The cell-8 guards are untested.**
- The D146 comparison, the CUDA reference choice and the save-before-stop ordering are notebook strings, so the static tests only check that the text is there.
- Moving them into small `t4` helpers would let L1 and M1 have unit tests.

## Judgment calls

**J1. D146 could stop a session whose production embeddings are consistent.**
- What reproducibility needs is the same in-process embeddings across runtimes. A systematic, harmless kernel-vs-fresh difference (say, a floating-point mode set by a library in the llama.cpp process) would fail D146 in every runtime.
- I expect it to pass, since both sides use one thread, the same local snapshot and the same environment.
- **Option:** if it ever fails, stop only when kernel probes differ across runtimes on the same CPU model.

**J2. The CUDA stop is stricter than the S3.3a design.**
- In S3.3a, driver and CUDA are recorded but don't block resuming (`NOT_COMPAT`), and the canary decides epochs.
- Stopping is reasonable for session 2's evidence. For the campaign, consider recording the change and letting the canary decide.

**J3. The benchmark-C workers are the only in-process embedders without a probe.**
- Add `probe_vectors` to the worker report and compare it with the kernel's `kernel_probes`.

## Answers to your questions

1. **Loops, locks, half-installed VMs, wrong caches:**
   - No install onto a half-installed VM: an unfinished pip is refused before `in_place` or `resumed` is considered.
   - No lock across VMs: Drive caches heal themselves before any pip command runs.
   - No wrong cache:
     - The cache key covers the nine versions.
     - A cache built on a different image is caught by `image_changed` (for the nine libraries) and by the dry run (for dependencies).
     - A resumed run builds the cache from Colab's original `pip list`.
   - The remaining loop is L2.
2. **Kill or interrupt at each step.** A kill always means a fresh VM with a new runtime id, so only interrupts matter for the attempts file:

   | Step | What a re-run does | Right? |
   |---|---|---|
   | Before pip | Allowed | Yes |
   | During pip | Refused | Yes |
   | After `pip_done` | Resumed | Yes |
   | During the probe | Resumed | Yes |
   | During the cache publish | Resumed; stale temp folders removed, wheels downloaded again | Yes |

   - An interrupt between the `pip_done` write and its rename is refused. That's conservative and acceptable.
   - Trivial: an interrupt between `save("libraries")` and `ok` appends `libraries_runtimes` twice for that runtime.
3. **Do the notebook guards cover the actual failure?**
   - The aborted case (a new session file opened from an old copy) is covered.
   - The same failure in a later runtime is not (M1).
   - They can refuse a legitimate run only when the link's notebook differs from the session's pinned commit. That refusal is correct; only the wording is off (L4).
4. **Can D146 or the CUDA checks stop a healthy session?**
   - D146: unlikely, see J1.
   - CUDA paths and fingerprint: only on a different Colab image (J2), or through a failed reference entry (L1).
5. **Where the fakes are more permissive than reality:**
   - **Dry run:** the fake's dry run executes the real install logic. So no test can show "dry run passes, install fails" (L2).
   - **Downloads:** `pip download` never fails unless configured to.
   - **Drive:** the fake is a local disk.
     - Renames are atomic and visible at once.
     - No upload is ever partial or eventually consistent.
     - Two folders can never share a name (a late upload from a dead VM could do that on real Drive).
   - **Cell 8:** its logic is never executed in tests (L5).
   - **Interrupts:** these are modelled by writing event sequences, which matches real behaviour, since a `KeyboardInterrupt` writes no `failed` event.