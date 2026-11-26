I've reviewed 24e85d7. No High findings: 2 Medium and 4 Low. All 130 tests in `test_s34`, `test_s34_platform` and `test_notebook_static` pass, and every finding below was reproduced with a probe.

## Medium

**M1. The frozen material still misses some behaviour code (clear-cut). `s34.py:61` (`PROTOCOL_FILES`)**
I walked everything `stage3/agent/loop.py` and `stage3/rq3.py` import from the package. These behaviour modules are not hashed:
- `stage3/feasibility.py`: `parse_action`, the agent's action parser (`loop.py:122,131`), and `THOUGHT_MAX_CHARS`.
- `stage3/agent/backend.py`: `N_CTX` and `CTX_MARGIN` (the context-overflow rule, `loop.py:114,210`), `T_LLM`/`t_llm` (simulated LLM time, `loop.py:81`), and `GRAMMARS`.
- `scenarios.py`: `L_TARGET` (RQ3's latency target, `rq3.py:186`), `SUCCESS_GRACE`, `RETRY`, `REGIMES` and `build`.
- `status.py` (`RETRYABLE`, `OVERLOAD`) and `keyed.py` (the simulator's random draws).
- `configs/rq2_frozen.json`: the ACC parameters of RQ3's `acc` window (`rq3.py:36,172`).

Scenario: I changed `N_CTX` 8192→4096, `CTX_MARGIN`, `THOUGHT_MAX_CHARS`, `L_TARGET` 0.45→0.9 and `RETRYABLE`. `rules_sha256` stayed the same each time. The replay check only catches a change that a finished unit actually exercises. For example, with `N_CTX` = 4096, pilot prompts under 4096 tokens replay identically, so there is no `Z`, but later F/H units with long prompts overflow differently.

Fix:
- Add these files, plus the sha256 of `rq2_frozen.json`.
- Add a test that walks this import closure and fails on any module that is neither in `PROTOCOL_FILES` nor in an explicit INFRA list. INFRA would be `llm/cache`, `fsio`, `ledger`, `runner`, `canary`, `stage3/agent/scripted.py`, and `llama_backend`; `llama_backend` is covered by its stamp and by the cache key, which includes the grammar GBNF sha256.
- Do this before the freeze. Adding files afterwards is itself a `Z`, because every function in a new file shows as changed.

**M2. `Chain` is hashed as one block, so log or retry fixes cause `Z` pauses. `s34.py:487`**
Each top-level class gets one hash. `Chain` (`s34.py:527` on) is about 680 lines and 40 methods, mostly mechanics.

Scenario: I edited the log text in `Chain.adopt_repinned`, and separately an error message in `Chain.run`. Each gave `changed_code = ['stage3/s34.py:Chain']`, which means a `Z` pause.

Why it matters:
- A re-pin past the freeze is always a code fix, and every step that runs after the freeze (D, F, H, retries, user decisions) lives in `Chain`. Almost every such re-pin will pause.
- The pause text cannot tell a rule change from a log message, so the user will learn to accept `Z` without reading it, which defeats the check.

What to do:
- **Clear-cut:** hash class methods separately (`s34.py:Chain._do_F`).
- **Your decision:** should mechanics methods pause at all? These are `run`, `_units`, `_load`, `record`, `save_once`, `apply_user_decision`, `adopt_repinned`, `_clear_fixed_pause`, `_check_freeze_after_repin` and `_next_or_none`. My suggestion is to record their changes in the addendum without pausing. The rule methods would keep pausing: `next_step`, `_do_*`, `_ladder_state`, `_switches_on`, `_apply_switches`, `_fired`, `_gate_for`, `_costs`, `_h_decision` and `_final_pilot*`.
- **Smaller:** `<module>` includes import statements, so reordering imports also pauses.

## Low

- **L1. A Drive read error is counted as "replays differently" (clear-cut). `s34.py:819-821`, `s34_platform.py:99`.** Any error text from verify, such as `CacheError: unreadable cache: OSError…`, counts as divergence. In a probe, one unit hit an I/O error with no code change and the chain paused with "1 finished units replay differently". Accepting it writes a false `replay_diverged` into the addendum. Fix: count only `"False"` and `ReplayDivergence…`; for any other error, verify once more or just re-run the unit.
- **L2. The canonical AST form breaks on Python 3.15** (details under Q2). It causes a false `Z` listing 44 functions.
- **L3. The bundle grows with each re-pin past the freeze.** `protocol_detail` is about 33 KB, and a full copy is stored in the freeze record, in every addendum (including unchanged ones), and in each `Z` entry. After the freeze and one accepted `Z`, the bundle carries about 100 KB of hex, roughly 50k tokens the chat has to read. Fix: write it once to `s34/protocol/<commit>.json` on Drive and keep only its sha256 and `changed_code` in the records.
- **L4. A rare crash leaves the chain halted. `s34.py:507`, `s34.py:1073`.** The new `python` field is inside the write-once freeze record. If the runtime dies after the freeze file is written but before E is recorded, and Colab's Python changes before the next session, that session gets `ChainHalt: freeze.json is immutable and differs` (probe confirmed). A re-pin to any new commit gets past it through the "kept" branch at `:1069`, but the message doesn't say so. Fix: move `python` to the E log entry.

## Answers

**Q1: Is the frozen material complete, and does anything cause needless `Z` pauses?**
- Not complete yet: see M1.
- Needless pauses: M2 and L1. Stage 1–2 files in `agent/*.py` (e.g. `agent/scripted.py`) and unused functions in `metrics.py` can also pause, but those files rarely change.
- D173 holds:
  - The replayed payload recomputes the suite version (`Unit.spec()` rebuilds the stream with current code), plus the parser version, prompt version and hash, and the RQ3 and simulator versions.
  - `ENV_STAMP_FIELDS` copies only the commit, Python and numpy.
- The backend (model, wheel, sampling) is not in the freeze, but a change there is caught by the canary's epoch pause and blocks adoption.

**Q2: Is the canonical AST form stable across Colab Python updates?**
Stable enough for the thesis window, with the 3.15 issue in L2:
- I ran `protocol_detail` standalone on Python 3.11.15, 3.12.13, 3.13.18 and 3.14.6. All 319 entries were identical.
- A set of f-string edge cases (`=` specifier, nested format specs, empty parts, implicit concatenation) was also identical on 3.11 through 3.15.
- On 3.15.0b4 it breaks. PEP 810 adds `is_lazy=0` to every `Import`/`ImportFrom`, so 44 of 319 entries change: every `<module>`, `s34.py:Chain`, `run_rq1_unit`, `prompt_hash`, and others.
- Colab's 2026.01 runtime is on Python 3.12.12, and you can use a past runtime version by selecting it from the Runtime -> Change runtime type dialog, with the "Runtime Version" dropdown.

Cheap hardening: drop `is_lazy` when it is 0, store `python` in each addendum and `Z` entry, and say "Python changed X→Y" in the pause text. The robust alternative is to save the protocol sources at the freeze and hash both sides with the current interpreter; that pairs naturally with L3's side file.

**Q3: Any crash window, loop or stuck state?**
- No loops.
- Accepting a `Z` writes two records, the addendum then the decision. A crash between them only leads to a duplicate addendum when the decision is re-sent, which is harmless.
- The paused, load-only recompute runs nothing.
- Cell 8 behaves correctly: `FLUSHED` short-circuits outside Colab, and if the download fails the runtime is kept.
- The only stuck state is L4.