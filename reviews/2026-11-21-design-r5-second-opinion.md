# Second opinion on Stage 3 design r5 (external LLM, brought by the user 2026-11-21)

_Archived summary of the user's doc "Stage 3 Design r5 Review" (claude.ai artifact HffkLevjLove5czvrMC7Pw), reviewing
commit e606a61. Wording condensed; every finding kept. Applied in design r6 except the two user decisions noted._

**Verdict:** D46-D51 sound in direction; before the pilot fix output-token limits (H1) and the budget plan (H2) and add
pre-registered decision rules. Does not block S3.1/S3.2. Facts: the S3.0 grammar capped the thought (maxLength 300);
`docs/data/` was gitignored, so the S3.0 JSON was never committed.

- **H1:** action max_tokens 160 can truncate (thought unbounded, long args), and the fallback is cut at the same 160,
  so the step is lost; may differ by condition. Reflections (40 words) are cut by a 60-token cap. Fix: action 320,
  reflection 96, keep maxLength 300 in the fallback grammar, report truncation per condition, pre-register: >1%
  truncation in any pilot condition -> raise caps before freeze.
- **H2:** 49 GPU-hours understated (10-12 calls per task, 2,500-3,000 input tokens -> 60-80 h); never-cut parts alone
  about 42-51 h; cut order saves only 12-14 h. Fix: set a GPU-hour budget from the deadline and gate after the pilot;
  test two throughput levers in the pilot under the determinism canary: (a) a fixed static-prefix KV snapshot per
  session (history-independent, about 25% saving), (b) two llama.cpp processes on one T4 (bandwidth headroom; host RAM
  risk). [User decisions.]
- **M1:** cross-session determinism untested; drivers/CUDA can change. Fix: canary of 8 reference prompts at every
  session start (mismatch -> refuse resume), interrupted test resumed in a new session, randomised condition order
  within each stream.
- **M2:** D47 is also a validity requirement in RQ3 (trajectory reuse would make output depend on the schedule); only
  history-independent reuse may be considered.
- **M3:** 0/20 invalid allows up to ~14%; identical-to-grammar is by construction; free decoding never multi-step.
  Pre-register: fallback >5% overall or C0-C2 difference >3 points -> grammar always. Render notes as plain text;
  optionally strip one code fence before the strict parse and log it.
- **Other confounds:** fallback and longer prompts change only t_llm/LLM calls, not steps; in RQ3 they shift completion
  time (T_max failures reported next to success).
- **Low:** commit the S3.0 JSON and re-fit the constants with committed code; state the t_llm fit range in Section IV;
  assert prompt + max_tokens <= n_ctx with a defined overflow outcome; replace the illustrative section 9 table with
  the pilot projection.
