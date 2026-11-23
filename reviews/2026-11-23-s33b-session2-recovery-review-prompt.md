# Second-opinion review request: S3.3b session-2 recovery paths and notebook guards (spec r6, section 13)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`). Report findings as High / Medium / Low with
file:line and a concrete failure scenario; separate clear-cut bugs from judgment calls. Read the code.

## Context
This follows your review `reviews/2026-11-23-s33b-embed-setup-second-opinion.md`. Since then:
- The first session-2 attempt was aborted: an outdated copy of the notebook (session-1 cells) ran against the newer
  code (evidence `docs/evidence/s33b_session2_aborted_2026-11-23.json`). Two guards were added: the `NB_VERSION` cell
  fingerprint (checked against the checked-out commit's notebook) and `t4.Sections` refusing to start a new session
  file without it. Hand-offs now use commit-pinned Colab links.
- Your M1, L1, L2, J2, J4 were applied; the user decided J1 -> D146 (kernel embedder must equal the setup check),
  J3 -> D147 (`resumed` re-runs after pip finished), M1 point 4 -> D148 (cache all nine libraries: deferred).
- An internal review then found 0 High / 3 Medium / 6 Low, applied (spec section 13, end).

## Where to look
- `src/eem_acc/stage3/llm/embed_setup.py`: `preflight_wheels`, `preflight_minilm`, `ensure_embed` (attempt events
  start / pip_started / pip_done / ok / failed; modes resolve / offline / network_exact / in_place / resumed; the
  offline dry run).
- `src/eem_acc/stage3/llm/t4.py`: `Sections` guard, `linked_paths`, `cuda_fingerprint`, `s30_equality(limit=...)`.
- `tools/make_nb04.py`: settings cell (`NB_VERSION`), setup cell, cell `# 8) SESSION 2: open` (kernel probe, S3.0
  check, CUDA checks every runtime).
- Tests: `tests/test_s33b_embed_setup.py`, `tests/test_notebook_static.py`, `tests/test_s33b_t4.py`.

## Questions
1. Can any recovery path loop, lock the session, install onto a half-installed VM, or build a wrong cache?
2. Are the refusal / resume rules right for a kill or an interrupt at each step (before pip, during pip, after
   `pip_done`, during the probe, during the cache publish)?
3. Do the notebook guards cover the failure that actually happened, and can they refuse a legitimate run?
4. Could the D146 kernel-vs-check comparison or the CUDA checks stop a healthy session?
5. Test fidelity: where is the fake pip / Drive more permissive than reality?
