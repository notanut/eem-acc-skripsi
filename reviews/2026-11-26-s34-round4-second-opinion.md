Reviewing round 4: pulling `main`, then reading the spec r7 sections 2 and 22 and the diff since 24e85d7.

I've reviewed 38d1fbf. No High findings: 4 Medium and 2 Low, all clear-cut except M1's scope, which is your call. All 142 tests in the three S3.4 test files pass, and every finding below comes from a probe I ran.

## Medium

**M1. `MECHANICS` still contains code that changes results or rules. `s34.py:70-77`**
I applied 13 single-line edits to frozen sources and checked each with `protocol_detail` and `is_mechanics`. None of them pauses:

| Edit | What it does |
|---|---|
| `Platform._memo` no longer checks the commit | After a re-pin, an old replay pass is reused, so units are adopted without being replayed and divergence can't show |
| `Platform.verify` treats an error as a pass | Adoption and the replay gate pass on broken replays |
| `Platform.load` drops `plan=plan` | Results made at the old commit are used |
| `Platform.run_units` no longer counts halted units as failed | Changes which units get a retry or a pause |
| `Chain._units` treats not-done units as done | Phases complete with missing units, so F can select on a subset |
| `Chain._clear_fixed_pause` clears any pause | A re-pin skips the D57 gate decision |
| `Chain.adopt_repinned` records every reason as "not finished" | Turns off the divergence check |
| `apply_user_decision`: three edits | Allows `n_tasks` 12; H accept never repeats the D60 round; B settings may change any knob |
| `code_identity`: dirty is never set | Uncommitted code can run |
| `Runner._session_mismatch` swallows the error | A canary that differs within the session no longer halts |

The edit to `Chain.next_step` (my control) does pause.

- **Clear-cut:** these methods are rules or the replay check, not plumbing.
- **Your decision:** either shrink `MECHANICS` or move the rule lines out into pausing helpers, as was done for the four guards this round.
  - Shrinking keeps only code that cannot change a result: lease, claims, heartbeat, paths, `progress_line`, exception classes, `error_signature`, `mark`, `write_once`, `save_once`, `_next_or_none`, `Platform.__init__` and `verify_finished`. `record` is borderline, because it stamps the commit that `_clear_fixed_pause` relies on.
  - Moving the rule lines out is a large job for `apply_user_decision`.
  - I recommend shrinking the list: a fix to the methods above should pause.

**M2. A `Z` that interrupted a pause, followed by another fix and re-pin, leaves the chain stuck. `s34.py:943-950`, `:702`**
The `Z` pause text offers "or another fix and re-pin". If the user takes that route:
1. `_clear_fixed_pause` treats the `Z` as a fix pause and records `U` with `next = p["before"]`. For an interrupting `Z`, `before` is `None`.
2. `next_step` then returns `None`, and the chain fails with `AttributeError: 'Chain' object has no attribute '_do_None'` every session.

Probe results:
- If the second fix changes code again, a second `Z` appears; accepting it gives the same `_do_None` error.
- If the second fix reverts the code, the error comes immediately.
- In both cases the D57 pause is never re-opened, and recovery means editing chain.json by hand.

A crash between the `U` record and the re-opened-pause record (`:806-809`, two separate writes) gives the same stuck state, and re-sending the accept does nothing because that decision is already recorded.

Fix:
- Use one helper to re-open the interrupted pause from both the `Z` accept and `_clear_fixed_pause`.
- Write the re-opened pause in a single record, with no separate `U`.
- Make a `U` whose `next` is `None` raise a clear `ChainHalt`.

**M3. A re-pin and a user decision in the same session hide replay divergence. `s34.py:789-790`**
Cell 7 calls `apply_user_decision` before `run()`, so the freeze check runs before `adopt_repinned`. At that point there are no adoptions for the new commit, so `replay_diverged` is empty.

Probe: the chain is paused at the D57 gate; the user sets `REPIN` and `USER_DECISION` (gate) together; 10 B units replay `False`.
- An addendum is recorded with `changed: {}`.
- The gate is applied and `run()` returns `done`.
- No `Z` is raised, even though 10 diverging units are sitting in `adoptions`.

Fix: call `self.adopt_repinned()` at the start of `apply_user_decision` (it skips plans already at the current commit, so calling it twice is safe), or have cell 7 refuse a `USER_DECISION` in a re-pin session. L1 below has the same cause.

**M4. Q1 completeness: the backend's load parameters are not frozen. `s34.py:79`, `tools/make_nb04.py:151`**
The `LOAD` line (`n_ctx=8192, n_batch=512, seed=0, n_gpu_layers=-1`) lives in the notebook generator, not in the snapshot. A change to it reaches the frozen header fields (load parameters, sampling, decoding path), which show up only as "header differs", and that reason is in `IO_REASONS`.

Probe: after a re-pin where every unit is "header differs: ['load_params']", the units re-run, `run()` returns `done`, and the addendum records `changed: {}`. If the canary's outputs don't change, the epoch doesn't change either, so the decoding setup changes after the freeze with no record.

Fix: add the fields `header_diff` compares (`Runner.compat` minus `ADOPT_FREE`) to `freeze_extra`, with a hash in `FREEZE_HASHES`.

## Low

- **L1. An X pause answered with both a re-pin and `rerun` in one session halts. `s34.py:790`.** When the user sets both, the freeze check runs before `_clear_fixed_pause`, so `next_step` raises `ChainHalt: a pause for a code fix is still open` every session while `USER_DECISION` stays set. Clearing it recovers, but the message points the wrong way. The M3 fix covers this too.
- **L2. `sources_at` fails with a raw git error. `s34.py:1329`.** If a commit is unreachable (see Q3), `CalledProcessError` surfaces as `ERROR: ... exit status 128` every session. Fix: wrap it in a `ChainHalt` that says to restore `s34/protocol/<commit>.json` or push the commit.
- **Nit:** pass `encoding="utf-8"` to `read_text` and to `git show`. The sources are all ASCII today, so nothing breaks yet. Also, the setup's dirty check ignores untracked files, so an untracked `.py` file in a frozen folder would make the Drive snapshot differ from a git rebuild.

## Answers

**Q1: Is the frozen material complete, and is `MECHANICS` free of anything that affects results or rules?**
- The code is complete per the import closure: the new test passes. The remaining gap is the notebook's load cell and backend stamp (M4).
- `MECHANICS` is not free of results or rules (M1).

**Q2: Any crash window, loop or stuck state?**
- **Snapshot handling is sound.** The snapshot is written before the record that names it, rewriting it is safe, and the sha256 is checked. A crash leaves at most a stray file.
- **The stuck states are in the resume-pause path:** M2, plus the crash between the two writes.
- **The duplicated pause entry is harmless:**
  - It makes the next H round skip a number, but the round number is never used in a decision.
  - In `verify_log`, the copied D57 gate is recomputed starting from the paused gate's cuts, and that gives the same result.

**Q3: Will `sources_at` work on Colab?**
Yes:
- The setup clones the full history and runs `git fetch origin` every runtime.
- `__file__` resolves inside the checkout, so `repo` is `/content/eem-acc-skripsi`.
- Locally, `sources_at("HEAD")` exactly matches `protocol_sources()` (40 files, 499 KB).

The condition is that the freeze and addendum commits stay reachable on GitHub, so no force-push or rebase of `main` after the freeze. It's also only needed when the Drive copy is missing.