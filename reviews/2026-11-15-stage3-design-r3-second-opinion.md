# Second opinion on Stage 3 design r3 (external LLM, brought by the user 2026-11-15)

_Archived summary of the user's doc "Stage 3 Design Review (r3)" (claude.ai artifact FYsjsTzhnkX3ZiKFceD8Fs), reviewing
`docs/stage3_design.md` at commit 964b432. Wording condensed; every finding kept._

**Verdict:** close to approvable after four fixes. Fix before S3.2: H1. Fix before the analysis is frozen: M1-M3. Can
start now: S3.0 and S3.1.

**H1 (section 6), the diffing rule fails in two directions.**
1. Terse enum drift is never recorded: values only for fields named in k_t or the error payload; a terse 400 names
   none; enum change keeps key and type.
2. Entity ids leak into the shared store: a wrong-id 404 (logic error) creates a note; a later success records
   "order_id: O-17 -> O-23".
3. Spurious key changes: adding/dropping a documented optional key is recorded as the change that worked.

Fix: key the rule on the documented (old) schema: record appeared keys not in the documented schema (with JSON type);
documented keys that disappeared only with an undocumented key appearing (rename) or when the error names them; type
changes of documented keys; values only for enum/boolean fields and newly appeared keys; never values of identifier
fields (`*_id`); new operation id for moved operations; empty diff gives no line; nested arguments by JSON path, arrays
as whole values. Add **note precision** as a mechanism check.

**M1 (section 7), 400-token budget contradicts two arms.** One note is ~100 tokens, so 400 tokens hold ~4 notes. AN
would show 4 of 10-14 notes: exempt AN and report its tokens. The k sweep: k > 4 is silently cut; set the budget to
k_max x maximum note length or report the effective k.

**M2 (section 6, staleness probe).** Merge clears the outcome line only for the same (API_Name, Error_Type) key; a
second drift of another type leaves a wrong line in the old note. Fix: any new non-transient error on X reopens every
note keyed on X and removes their lines. State the probe's second drift type.

**M3 (section 8), stream-count rule rests on a noisy estimate.** SD from 4 dev streams (3 df): 90% interval 0.62x-2.92x.
Fix: fix 14 test streams now (+320 tasks), drop the C0/C1 dev runs (-160 tasks); report the minimum detectable
difference.

**Low.**
- Holm over six with gating is valid but conservative (first threshold 0.025/6). Graphical procedure (Bretz et al.,
  Stat. Med. 28:586-604, 2009): weights alpha/4 on NI1, E2, NI3, E4, 0 on S1, S3; NI1 -> S1, NI3 -> S3; a rejected
  superiority hypothesis splits its weight over the rest. The power simulation must use the chosen thresholds.
- Drift type 6: say what the correct fix is and whether error-style randomisation applies to its payload.
- Pre-register the 6 dev-selection configurations and the pilot's recalibration knobs.
- Cut order: the RQ3 private-store run is missing; put it first. Keep C2-OU out of step (6).
- [MIR]: cite the MirrorAPI model card (MIT licence, fine-tune of Qwen2.5-7B-Instruct).

**Q1-Q4:** (a) on all four. Q1: "C1 + outcome update" is not well defined (C1 has no k_t); A2 covers the update
without reflection. Q2: same-task keeps entities and agent identical; (b) brings back H1 leakage. Q3: without R2 the
ACC half has no confirmatory test; expect a modest effect. Q4: with T4 LLM times, LLM time dominates completion time;
the 3x-faster sensitivity is where completion time is informative.

**S1-S12:** agree with all open recommendations; conditions: S3 with the M2 reopen rule; S8 with the cut-order
changes; S12 must include the corrected diff rule. S5 and S11 already decided.
