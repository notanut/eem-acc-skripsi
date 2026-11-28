I'll review the r3 delta: pulling 2d3f0cd, reading sections 12–16, 21 and 22, then checking the new tests and the BMBP09 worked-example data against the paper.

The r3 delta at 2d3f0cd has no High findings, but 4 Medium and 7 Low, and tier 1 can still be lost without a blind `stop` (Q3). All 54 Part 1 tests pass, and the statistics still match my own exact implementations.

The four mutants that survived last time are now killed, along with new mutants on the two flags, Algorithm 2's running maximum and the `phase` guard. The new `adjusted_p_sequential` (Algorithm 2) equals my closed test on 4,000 p-vectors for the two frozen graphs, including dyadic and missing p-values.

## Medium

**M1. A fix after AP can still change finished results, and nothing checks it** (sections 12 and 14, D192).
- A re-pin that fixes an `X` cause may change behaviour modules: only the analysis, protocol-data and T-plan hashes are checked.
- Its `Z` is accepted automatically, and changed S units are re-run.
- Claude writes that fix having seen the primary results, and the T part of `full.json` (C2-OU, mechanism checks), since that is computed during S.

Scenario: an `X` in an SP unit is fixed by an edit to `memory.py` that also changes retrieval. Every finished AN and A2 unit re-runs under it automatically.

Fix: after AP, an `X`-fix re-pin must leave every finished unit replay-identical, T and S, apart from the units that failed. If any other unit replays differently, refuse the re-pin, keep the failed units as failed, and let S continue.

**M2. Tier 1 can still be dropped without a blind `stop`** (sections 13 and 14, D193 and D196).
- The hand-over check passes when used + reserved(tier 1) ≤ budget, which can be by a few minutes.
- After it, AP and the export run on the GPU runtime, a new session adds its setup, and any T retries after the last block add cost. None of these is in the reservation.
- Then the ordinary S check for tier 1 ("used + projected > budget → S ends") can fail, and tier 1 is gone with nobody deciding.

Fix:
- Exempt tier 1 from the S budget check once the hand-over check has passed. The reservation is a commitment.
- Run the hand-over check when T is finished, after every `X` answer and retry, immediately before AP.

**M3. A systematic crash in S becomes an endless grind** (section 14, the halt, `X` during S and the commitment).
- Suppose every SP unit crashes the same way. Each session halts after two failures in a row (D130), and each failed unit gets two more automatic attempts.
- Ten SP units × 3 attempts at 2 per session is about 15 sessions, each paying setup. With no fix, the commitment obliges the user to keep starting them.

Fix: during S, a halt marks its error signature.
- Until a re-pin fixes it, a unit that fails with that signature is accepted as failed after its first failure, without retries.
- If the same signature halts two sessions in a row, S ends at the current tier boundary, and this is reported.

**M4. The "T-plan hash" check contradicts the re-pin machinery** (sections 12 and 14, D192 and D194).
- `adopt_repinned` writes a new plan version at every re-pin. If "changes the hash of the `s35-T` plan" means the plan file, every re-pin after AP is refused, so D192's `X` fixes are impossible.
- Even with a laxer hash, a new T plan version marks T units that don't replay identically as not done. S3.4's machinery then re-runs not-adopted units when the phase is loaded, which breaks "T is final".

Fix:
- After AP, `adopt_repinned` skips `s35-T`: it replays T units only to report deviations, and writes no new plan version.
- Define the refusal hash as the T plan's unit list plus `meta`.

## Low

- **L1. A backend change after AP has no automatic answer.** The "Z after AP, no re-pin" row lists only unit-level actions, so a rebuilt wheel would wait for the user. Fix: accept it automatically and record it; output changes already go through the epoch rule.
- **L2. Whether to attempt a stall fix is still a choice made after unblinding** (D194). Fix: make the attempt mandatory whenever a fix that changes only the environment exists; S stops for good only when the fix would need a behaviour change.
- **L3. Re-runs of accepted-failed S units after a fix aren't scheduled.** Fix: they run only while S is running, before the next tier's budget check, and count with that tier's cost. After S has ended, nothing is re-run.
- **L4. Bugs in the frozen analysis have no stated path.** Analysis changes are refused after AP, so a bug found in `report.py` has no route. Fix: say it is fixed outside the chain and both the frozen and the corrected numbers are reported as a deviation.
- **L5. (Q4) Yes, show hours only at T's end, at every `B` pause and at S's end.**
  - The stated T leak is small: the C2-OU cost is noisy across 5 units.
  - But in S, tier composition is nearly single-arm (tier 3 is W = 4 only, tier 9 R only, tier 11 the private store only). Per-tier hours therefore give per-arm costs, a proxy for failures, so D195 as written doesn't hold.
  - The user and Claude make no decision that needs per-block hours: the check runs inside the chain, and "the next block fits" is enough to show.
- **L6. One test gap.** The mutant that sets `unrejectable` to False for an empty test (n = 0) survives. Add an assertion.
- **L7. (Q5) Points in `bmbp09_examples.json` and its tests.**
  - `_num` calls `eval` with builtins available; pass `{"__builtins__": {}}`, or encode ε edges as (constant, ε coefficient) pairs.
  - Assert that `eps` is well above `graphical.TOL`. If eps fell below 1e-12, `_remove`'s zero rule would treat 1 − (1 − ε) as 0.
  - These p-values can't pin where the ε edges sit. In Figs. 9 and 12, the mirror placement gives the same rejections, so the tests validate the algorithm on the encoded graphs, not that the encoding matches each figure. Have someone check the five encodings against the PDF once.

## Q5: comparison with the paper

I couldn't open the paper: the only copy I found redirects to a host this workspace can't fetch. So I checked the examples for internal consistency:
- I re-derived every example in exact arithmetic at ε = 10⁻³ and 10⁻¹². The rejection order is the same at both, and the levels match the JSON at the small ε.
- Fig. 4: the adjusted p-values equal Holm's: 3 × 0.012 = 0.036, max(0.036, 2 × 0.02) = 0.04, and 0.055.
- Fig. 2: the most weight H2 can ever reach is ½, a level of 0.025, below p₂ = 0.04. So H2 can't be rejected, and "only H1, H2, and H3" can't hold for these p-values. That supports your typo reading, provided the encoded p-values match the paper.

If you attach the PDF here, I'll check the five encodings page by page.

## Other questions

- **Q1, decisions after AP.** Three remain: the content of an `X` fix (M1), whether to attempt a stall fix (L2), and how an analysis bug is handled (L4). Two code paths still depend on rules that aren't written down: re-planning T at a re-pin (M4) and the backend `Z` (L1).
- **Q2, pause combinations.** Nothing deadlocks outright. The problems are a grind (M3), a contradiction (M4), an undefined answer (L1), unscheduled re-runs after S ends (L3), and the hand-over running before T's last retries (M2).
- **Q3, tier 1.** Not quite: see M2. With M2's two changes, only a blind `stop` can drop it.