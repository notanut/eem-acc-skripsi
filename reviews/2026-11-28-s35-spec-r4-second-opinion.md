I'll review r4 at c9f9f3e: reading sections 12–15 as a state machine, checking D197–D202 against each other, and checking each rule against what `s34.Chain`, `Platform` and the runner can actually do.

I've reviewed r4 at c9f9f3e (55 Part 1 tests pass). There is more than wording to fix: 1 High and 3 Medium findings remain. None needs a new kind of rule, and once they are fixed, implementing and letting the Part 2 tests carry the rest is reasonable.

## High

**H1. D197's "refused, … S continues" can't happen as written: S would run on the refused code** (section 12).
- **The re-pin is already recorded before the chain can refuse it.** The setup cell creates `t4.Sections(..., repin=True)`, which replaces `data["stamp"]` with the new commit (`t4.py:44-49`). That happens before `Chain.run` reaches any D197 check.
- **Going back is blocked.** After a refusal, every later session checks out the refused commit. Re-pinning to the previous commit fails the setup's rule that a re-pin must descend from the chain's commit, because the previous commit is an ancestor.
- **So "S continues" runs new S units at the refused commit**, which is exactly what D197 forbids.
- **Adoption also writes too early.** S3.4's `adopt_repinned` records each adoption and writes each new plan version plan by plan (`s34.py:1096-1106`), before any all-units decision.

Fix:
1. In `TestChain.adopt_repinned`, decide D197 over all plans before writing any adoption or plan version.
2. Add a **refused** state: the chain records the refusal and runs no unit at a commit that hasn't been accepted.
3. The way out is a revert commit that descends from the refused one and whose protocol files equal the last accepted commit's. The user re-pins to it; D197 then passes trivially, and S continues. This is a mandatory action, not a decision.
4. Have `REPIN` after AP name the signature or stall cause it targets (e.g. `REPIN_FOR`). Without that, "one attempt per signature", "the units it fixes" and "aimed at a signature" can't be computed.

## Medium

**M1. After AP, a fix can still change the S rules themselves.**
- The refusal set covers analysis modules, protocol data and the T plan, plus behaviour modules for stall fixes.
- An `X` fix may touch `s35.py`, `s34.py` or `s34_platform.py`. Changes to tier order, the budget stop or D199's counts don't change any unit's replay, so they pass D197.
- S's chain code first runs at scale after AP, so a fix there is plausible.
- Fix, two options:
  - refuse after-AP changes to any chain function outside `MECHANICS`;
  - or, to keep S alive through a chain bug, accept such a change only if the frozen S-rule tests (section 18, run from the freeze commit's test files) still pass against the new code.

**M2. D202 and D199 race, and the timing is a human choice after AP.**
- A mandatory fix for a signature takes Claude time. Meanwhile, the commitment says to keep running sessions.
- If that signature halts the next session, it is "two in a row", and S ends after the tier (D201) before the mandated fix can land. Its re-runs then never happen.
- So which units run depends on whether the user starts a session before or after the fix.
- Fix: after a halt or `X` in S, the next S session starts only once one of these is recorded:
  - a fix re-pin for that signature;
  - or Claude's written "no fix passes D197", entered as e.g. `USER_DECISION {"action": "no_fix", "signature": …}`.

  D199's consecutive-halt count then counts only sessions after that record.

**M3. Four runner facts the spec gets wrong or leaves open** (`runner.py`). Each changes how D199/D201 behave.

- **The halt state doesn't persist.** `last_failure` is a local variable of `run_phase` (line 574), so it resets on every call. Section 12's "persists across those calls" is wrong: two failures in a row across a block or tier boundary don't halt.
- **Retries can't be selected by signature.** `run_phase(retry_failed=True)` retries every failed unit of the phase (lines 567-569). D199's per-signature no-retry needs the chain to mark retries itself with `Runner.mark(phase, uid, "retry")` and leave `retry_failed` off.
- **Halted units come back.** They are re-enabled, not failed (line 533). Unless S runs with `accept_halted=True` after a halt, the two halted units run first next session and halt again at once.
- **Making D201 hold needs a catch.** D201's "no further halts ending S" means `TestChain` must catch `RunHalted` and call `run_phase(accept_halted=True)` again in the same session.

Two more implementation points:
- **`max_units` doesn't bound a block.** It counts units attempted in plan order. Units marked for retry from earlier blocks are taken first, so a block call must loop until every unit of the block is done or failed.
- **The commit guard locks T after AP.** Without a new plan version, `run_phase("s35-T")` at a later commit raises `TestPhaseCommitMismatch` (line 561). That usefully enforces "T is never re-run", but progress, the export and `verify_log` must read T results directly, never through `run_units`.

## Low

- **L1. "Two halts in a row" is undefined in one case.** Say whether a session without that halt resets the count; I'd count consecutive sessions in which the signature halted.
- **L2. Environment fixes in notebook 06 would stop S.** The notebook is protocol data, so a stall fix in its setup cell is refused, which stops S for good. For a Colab image change, the escape that changes no code is choosing the old version in Colab's "Runtime Version" setting; state it in the D194 text.
- **L3. No close-out path if the notebook can't run.** A refused stall fix stops S, but the S export and the hours at S's end come from the notebook. Define a cell that runs without the GPU and without installing anything, or a manual zip of `s35/units/s35-S`.
- **L4. One re-run has no schedule.** The automatic re-run of an S unit after a replay mismatch with no re-pin needs the same scheduling as L3 of r3: before the next tier's check, counted with that tier.
- **L5. Two backend-change paths are treated differently.** A change with no re-pin is accepted automatically, while the same change brought in by a stall re-pin must pass D197. State it.

## Answers

1. **State machine.** Every state has a next step except two: after a D197 refusal (H1), and the close-out when the notebook can't run (L3). Human action after AP is limited to mandated fixes, which are fine once M2's record exists. Without M1, a fix can still change the rules themselves.
2. **Contradictions.** D198 and D201 agree: tier 1 finishes, and it is bounded. D200 and `B` agree; the hours shown at `B` are the stated leak. D197 with D194 is consistent but strict (L2). D197 with D202 conflicts only through timing (M2) and through the missing refused state (H1).
3. **Implementability.** Everything is buildable on `s34.Chain` and `Platform` once these are fixed: the commit-already-recorded problem (H1), the four runner semantics (M3), and adoption deciding before it writes. The rest is ordinary `TestChain` override work.