# Stage 3 Design r7 Review (second opinion, external LLM; brought by the user 2026-11-21)

Archived from the user's Claude Doc (transcribed; wording kept).

## Verdict
r7 is close to approvable. D58 and D59 introduce no bias between conditions, and stating the margin in tasks keeps
the NI tests, TOST and the selection tie rule consistent if streams shrink.
- Fix before approval: two gaps in how the test phase runs (H1, H2), D60's justification and a missing rule there
  (M1), and a missing stop condition in D61 (M2).
- Clarifications: M3 and M4.
- Not blocked: S3.1-S3.3, but M3 belongs in the S3.1 sampler tests.

## High
**H1 (§10, §8 early stop): the run order puts "never cut" evidence at risk.** Milestones run RQ1 (S3.5), then RQ3
(S3.6), then the descriptive arms (S3.7). An early stop during RQ1 leaves RQ3 with zero seeds; RQ3 is the
contribution the Section II gap statement claims. C2-OU is "never cut" and needed to interpret E3, but runs last.
Fix: pre-register an interleaved order: blocks of 2 RQ1 streams (C0-C3, plus C2-OU on its 5 streams, order
randomised within each stream), then 1 RQ3 seed (all four 2x2 cells), repeat; remaining descriptive arms after.

**H2 (§9-10): nothing keeps test outcomes hidden while the test runs.** The RQ2 live view showed a partial ranking
during runs. Fix: during S3.5-S3.7 the live view shows only progress, budget, epoch and canary status; no interim
analysis; an early stop is triggered only by the budget gate's accounting or a calendar date fixed now.

## Medium
**M1 (§6, D60): one false claim and one missing rule.** "The grammar changes only replies that would not parse" does
not hold for grammar-always: the grammar caps the thought at 300 characters, so a valid free reply with a longer
thought would also be cut (D50 leaves the thought unbounded; memory arms may think longer). Fix: in grammar-always
mode, drop the thought cap from the grammar and rely on max_tokens. The truncation switch (320 -> 640) is fine.
Missing rule: after the freeze, D60 re-runs the C0 acceptance check but says nothing about a failure. Fix:
pre-register "proceed and report".

**M2 (§9, D61): no stop condition for nondeterminism on long prompts within one platform.** Fix: after 2
resume-canary mismatches in a session while the session canary passes, classify as long-prompt nondeterminism; from
then on resume from cache, flag the unit "reproducible by replay only", stop restarting; run the canary's long
prompts twice; archive discarded caches; report restarts per condition.

**M3 (§9, cut step 7 and D62): shortened streams change what gets compared.** Fixes: define every shortened stream as
a prefix of the sampled full-length stream; compare a descriptive arm with the primary arms' prefix of the same
length; run the sampler for the final length and test in S3.1 that the exposure rule is satisfiable at 15 and 16
tasks; define SP's midpoint re-drift relative to the actual stream length.

**M4 (cut step 0): dropping the private-store run removes the only evidence on the shared store.** Add: "If dropped,
the paper makes no claim that sharing the store across agents helps; III-B's shared store is a design choice without
evidence."

## Low
- §9: "R, AN, A2, C2-OU and SP are reduced but never dropped" includes C2-OU, which no step reduces. Remove it.
- "Units never share caches, so a result does not depend on run order" holds only within an epoch.

## Checklist answers
1. D58, D59, D62 introduce no bias; D60 sound once M1 is fixed; the only route to an outcome-informed decision is the
   unblinded live view (H2).
2. Cut list coherent apart from M4 and M3.
3. D61 has no terminal state (M2); the early stop depends on the run order (H1).
4. NI, TOST and the tie rule consistent with the margin in tasks; the graphical procedure depends on the number of
   streams; D62 leaves R1's margin unchanged; remaining issue M3.

## Disposition (2026-11-21)
User decided H1 = interleave (D63), H2 = blind (D64), M1 = adopt both (D65), M2 = adopt (D66). M3, M4 and the Low
items applied as clear-cut. One addition of mine in M1: code-fenced replies also differ under the grammar; their
share is reported.
