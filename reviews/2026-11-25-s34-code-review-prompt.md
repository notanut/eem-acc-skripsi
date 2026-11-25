# Second-opinion review request: S3.4 implementation (the automatic pilot chain)

Repository: https://github.com/notanut/eem-acc-skripsi (branch `main`). Report High / Medium / Low with file:line and
a concrete scenario; separate clear-cut bugs from decisions for the user. Read the code; the spec
`docs/s34_spec.md` (r3) is the contract, including the user's decisions D154-D164 and the [delegated] choices.

Where to look:
- `src/eem_acc/stage3/s34.py`: the rule functions (wording, switches, acceptance, ladder, wiring, selection), the
  budget gate (`workload`, `projection`, `budget_gate`), the freeze, the `Chain` state machine and `verify_log`.
- `tools/make_nb05.py` -> `notebooks/05_s34.ipynb`: cells 1-5 reused from notebook 04; cell 6 (open the runtime),
  cell 7 (the chain callbacks: `run_units`, `metrics`, `checks`, `freeze_extra`), cell 8 (per-unit checks, bundle).
- `src/eem_acc/stage3/llm/runner.py` (`Unit.stream`, `Unit.spec()`, `write_plan(meta=...)`), `llm/ledger.py`
  (`profile`), `llm/embed_setup.py` (D148: wheels of all nine libraries), `llm/audit.py`.
- Tests: `tests/test_s34.py` (a scripted fake platform), `tests/test_notebook_static.py`.

Questions:
1. Does the code implement every rule of the spec and the pre-registered design (sections 6, 8, 9 of
   `docs/stage3_design.md`) exactly, including boundaries and tie-breaks?
2. Can the chain loop, halt falsely, or apply a rule to the wrong data when a Colab session is lost at any point?
3. Will the notebook cells work on Colab as written (APIs of `t4`, `runner`, `ledger`, `embed`)?
4. Is the budget projection sound (wall seconds per task, conservative mapping, cut order, task count)?
5. Test fidelity: where is the fake more permissive than the real runner?

Internal reviews so far: spec r1 (3H/7M/6L) and code (1H/6M/9L), all applied (spec sections 14 and 16).
