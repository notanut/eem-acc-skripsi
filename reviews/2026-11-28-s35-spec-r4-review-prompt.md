# Second-opinion review request: S3.5 spec r4 (the delta since r3)

Repository: https://github.com/notanut/eem-acc-skripsi, commit `c9f9f3e`. Contract: `docs/s35_spec.md` r4.

You reviewed r3 (`reviews/2026-11-28-s35-spec-r3-second-opinion.md`). Section 23 says how each finding was applied.
A fresh-context review of that delta found 1 High, 6 Medium and 6 Low issues (section 24). The user decided
D197-D202:

- **D197:** a re-pin after AP is accepted only if every finished unit, T and S, still replays identically. The
  comparison leaves out the units it fixes and units already recorded as a mismatch.
- **D198:** tier 1 is committed once the hand-over check passes, whatever the overrun. The check runs after T's last
  retries.
- **D199:** after a halt with a given error signature, units failing with it get no retries. Two halts in a row end
  S after the current tier. An adopted fix resets this.
- **D200:** GPU-hours are shown only at T's end, at `B` pauses and at S's end.
- **D201:** when D199 ends S, the rest of the current tier still runs, one attempt per unit, then S ends.
- **D202:** after AP, one fix attempt per error signature is mandatory when a fix that passes D197 exists. The same
  holds for stalls under D194; a refused stall fix stops S.

Please focus on these questions:
1. Read sections 12-15 as a state machine. Is every state's next step defined, with no decision left to the user
   after AP?
2. Do any two rules contradict? Consider D197 with D202 and D194, D198 with D201, and D200 with B.
3. Is anything in the rules not implementable as written on top of `s34.Chain` and `Platform`? Check `s34.py`,
   `s34_platform.py` and `llm/runner.py`.

The S-phase rules have been refined over three rounds. If you find only wording issues, say so: the next step would
be to implement them and let the Part 2 tests (section 18) carry the rest.
