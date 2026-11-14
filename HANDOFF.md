# HANDOFF

_Last updated: 2026-11-14 (RQ2 design v2 IMPLEMENTED and pushed; 5 decisions pending (spec section 15) before the dev run; no full run yet)._ Update after every milestone and before ending a session._

## 1. Project goal
Thesis / IEEE paper: **"Learning from API Execution Experience: Self-Reflective Agentic AI with Adaptive Concurrency Control"** (Bina Nusantara University). Sections I-III (Intro, Related Works, Methodology) are written; the Abstract, Chapter IV (Result) and Chapter V (Conclusion) are empty and must be produced from experiments that are **actually run** (never fabricate numbers).

The system has two parts. **EEM** (Execution Experience Memory) parses failed API calls into a tuple `k_t`, has an LLM write a short fix instruction, stores it in a Zettelkasten-style vector store and injects the top-k=3 entries into later prompts. **ACC** (Adaptive Concurrency Controller) runs an AIMD window `W` driven by HTTP 429/503 and latency vs `L_target`. Algorithm 1 combines them in a ReAct loop. Evaluation must show (RQ1) cognitive recovery, (RQ2) infrastructure protection, (RQ3) their combination, (RQ4) hyperparameter sensitivity.

## 2. Working rules (from the user, mandatory)
1. Ask before deciding anything (design, datasets, LLM, baselines, metrics, hyperparameters, repo layout, libraries, any Methodology change): give 2-3 options + recommendation, wait for confirmation.
2. Never fabricate results; report weak/contradicting results honestly; do not claim a paper did something unless verified.
3. Code runs in Google Colab (free T4); first cell sets up env, mounts Drive, clones/pulls repo, seeds.
4. Commit small and often to https://github.com/notanut/eem-acc-skripsi.git; never commit secrets or large files.
5. Heavy outputs (checkpoints, logs, results, figures, datasets) go to Drive, not the repo.
6. Long jobs need checkpoint/resume, per-unit result saving, atomic writes, progress + ETA.
7. Keep this file updated (copy on Drive).
8. Communicate in English, concisely. Work in stages; report briefly and ask confirmation after each stage.
9. (2026-11-14) **Independent review of every spec and every piece of code** by a fresh-context subagent; triage, fix clear-cut findings, real decisions go to the user. ALSO write a self-contained review prompt for another LLM under `reviews/<date>-<topic>-review-prompt.md` and show it in chat. When the user brings that LLM's feedback, evaluate each point critically and ask before non-trivial changes.
10. (2026-11-14) **Own pytest suite, run BEFORE every commit** (unit, edge/failure paths, determinism, checkpoint/resume equality, tiny e2e smoke). Never weaken/skip/delete a test to pass it (fix a wrong test and say so). Test summary in commit message/report. GPU/LLM/Drive/Colab/notebook tests are marked and excluded from the fast suite; never claim a Colab-only test passed.
11. (2026-11-14) **Order:** code+tests -> run tests -> review subagent, fix, re-run -> external review prompt -> commit+push -> update HANDOFF (incl. review prompt location + test status) -> short report, ask before next stage. Specs without code skip the test steps.

## 3. Decisions confirmed by the user (2026-11-01)
User replied "agree with all recommendations" and asked to keep everything **as affordable as possible**.

| ID | Decision | Reasoning |
|----|----------|-----------|
| D1 | Hybrid environment: build a **custom mock API server (FastAPI)** first (controllable rate limiter, latency model, capacity changes, schema-drift injection); later add a **StableToolBench subset** for external validity of RQ1 | No existing benchmark covers both problems; 429/503 ground truth must be controllable |
| D2 | **Free local open LLM on Colab T4** (e.g. Qwen2.5-7B-Instruct, 4-bit) for main runs; RQ2 uses **scripted non-LLM agents** | Zero cost, reproducible; scripted agents isolate ACC and allow many seeds cheaply. Paid API only if user later supplies credits |
| D3 | Baselines: ReAct (no EEM); ReAct + Reflexion-style last-failure reflection; ReAct + EEM; EEM with flat store (no Zettelkasten links). Concurrency: fixed W (low/mid/high), token bucket + exponential backoff, AIMD on 429 only, latency-only controller. 2x2 factorial {EEM, ACC} x {off, on} | Isolates each contribution |
| D4 | Metrics: task success rate, steps to success, invalid-call rate, repeated-error rate, recovery rate after first failure, tokens/task; 429 and 503 counts, throttle rate, p50/p95/p99 latency, SLO-violation rate, goodput, makespan, window trace; combined objective J from Eq. (1). At least 5 seeds (10 where cheap), mean +- 95% CI, paired bootstrap | |
| D5 | Scenarios: steady load, bursts, mid-run capacity drop, M in {4,16,64} agents; schema drift (renamed param, type change, new required field) at fault rates p in {0.1, 0.3} (trim to Colab budget) | |
| D6 | Ablations: EEM (raw traceback instead of `k_t`; `k_t` without reflection; flat store; k in {1,3,5}); ACC (no latency branch, no 429 branch, additive-only; alpha, beta, epsilon sweeps) | |
| D7 | Hyperparameters: heuristic start values + small grid on a **dev split with separate seeds**, then **frozen** before test runs | Avoids tuning on test data |
| D8 | Methodology gaps: **user approved all of P1-P12 (2026-11-01)** as proposed in `docs/section3_patch.md`. Revised text is in `docs/section3_revised.md` (awaiting user's review before it is pasted into the paper) | Keeps paper and code consistent |
| D9 | Libraries/model (confirmed 2026-11-01): **llama-cpp-python** with GGUF **Qwen2.5-7B-Instruct (4-bit)**; **all-MiniLM-L6-v2** embeddings; **FastAPI + uvicorn** mock API server | Cheapest and simplest on a free T4 |
| D10 | Scenarios (approved 2026-11-05): CAP-steady, CAP-burst, CAP-drop (capacity x0.5 at 20 s, back to x1.0 at 40 s), RATE-steady; M in {4,16,64}. CAP: capacity 8, hard cap 16, rate limit 200/s (non-binding). RATE: capacity 32, hard cap 64, rate limit 40/s burst 20 | Calibrated so that CAP exercises latency/503 and RATE exercises 429 |
| D11 | Workload: mean think time 0.1 s, 150 calls/agent, start jitter 1 s, burst = think 0 during [10,15) s; retries <= 6, backoff 0.1 s capped at 4 s | |
| D12 | L_target = 0.45 s = 1.5 x unloaded server p95 (0.303 s, measured) | |
| D13 | ACC: W0=8, W_min=1, W_max=64 fixed; dev grid alpha,beta in {.25,.5,.75}, epsilon in {.05,.1,.2}, n_c in {8,16,32} | |
| D14 | Baselines: fixed W {4,16,64} + best global fixed W (dev, from {1,2,4,8,16,32,64}); token bucket + backoff (rate tuned on dev from {10,20,40,80}/s); ablations no-429, no-latency, additive-only | Avoid strawman baselines |
| D15 | Protocol: dev seeds 0-4, test seeds 100-109, one global frozen ACC config committed before test; selection = maximise goodput s.t. mean SLO-violation <= 5% and mean call success >= 99%; J reported (lambda=gamma=0.5 primary, 1.0 sensitivity), not tuned on | J has no throughput reward, tuning on it alone would collapse the window |
| D16 | (2026-11-14) H1: scenarios get DIFFERENT throughput-optimal static windows (e.g. RATE ~16; longer CAP-drop) + cross-regime robustness test (best fixed W tuned on one regime, evaluated on the other) | Fixed W=8 was an oracle everywhere |
| D17 | (2026-11-14) H2: SLO <= 5% and success >= 99% constraints applied per scenario over LOADED cells; Pareto front (goodput vs SLO) also reported | Averaging hid ~10% per-cell violation |
| D18 | (2026-11-14) H3: open-loop burst (extra arrivals/s) instead of think=0; M levels {8, 32, 96} replace {4, 16, 64} | M=4 and closed-loop burst did not discriminate |
| D19 | (2026-11-14) Also: event-window metrics + recovery time; token-bucket grid ~12 rates 20-45/s; normalise goodput by best FEASIBLE method per cell; per-attempt client timeouts retried like 429/503 | Review M1, M2, M3, timeouts |
| D20 | (2026-11-14) RQ2 v2 feasibility = in EVERY loaded cell: SLO <= 5%, success >= 99%, throttle rate (429/503/504) <= 10%; loaded cells fixed a priori from offered load (8 of 12) | Replaces D17's per-scenario mean (review showed it hid violations) |
| D21 | (2026-11-14) Fixed horizon T = 120 s (unfinished calls = not successful, backlog reported); client timeout 1.0 s, counted in F, in f_j of Eq. 1 and as SLO violation; token bucket 15 rates 10-130/s | Replaces 150 calls/agent and D19's 20-45/s grid |
| D22 | (2026-11-14) Controller target L_ctrl = kappa * L_SLO, kappa in {0.8, 0.9, 1.0} tuned on dev (243 ACC configs); kappa = 1 always reported | Without a margin no ACC config met the 5% SLO in capacity-bound cells (calibration) |
| D23 | (2026-11-14) Leave-one-scenario-out analysis included. NOT included (user's choice): LAT-shift scenario, tuned 429-only AIMD baseline, capacity sweep | Fairness safeguards; v1 null calibration result disclosed in Chapter IV |

Repo/Drive layout approved as proposed (below).

## 4. Current status
**Done (Stage 1, local):**
- Repo cloned (was empty); `.gitignore`, `README.md`, `pyproject.toml`, `requirements.txt` (minimal: numpy, pyyaml, pytest).
- `src/eem_acc/utils/`: `seed.py`, `atomic.py` (temp file + `os.replace`), `checkpoint.py` (checksummed checkpoints, rotation, fallback to previous valid one, RNG state), `results.py` (per-unit JSON), `progress.py` (ETA), `runner.py` (`run_resumable`: stateless/stateful resume, config-mismatch guard), `paths.py` (Drive layout).
- `tests/test_resilience.py`: 12 tests pass locally (also in a clean venv after `pip install -e .`).
- `notebooks/00_setup.ipynb`: Colab setup (mount Drive, clone/pull, install, seed, create Drive folders, run tests, optional code sync). **Syntax-checked only; not yet executed in Colab.**

**Resolved - GitHub push:** the Claude GitHub App was installed by the user; all Stage-1 commits are on `origin/main` (verified: remote HEAD == local HEAD at commit `c0062ad`). Pushing works from the assistant's shell.

**Done (Stage 2, pushed):** deterministic virtual-time simulation for RQ2 (scripted agents), 44 tests pass.
- `env/vtime.py`: virtual-time asyncio loop (clock jumps to the next timer; same seed -> identical results; hours of API time run in ms, so many seeds are cheap).
- `env/server.py`: `MockServer` with token-bucket rate limit (429 + Retry-After), load-dependent latency, hard overload cap (503), optional capacity events (mid-run drop). Latency is fixed at arrival from the load at that moment (simplification).
- `acc/controller.py`: `ACCController` implementing revised Section III-C exactly; flags for ablations/baselines (`use_throttle_branch`, `use_latency_branch`, `multiplicative_decrease=False` = additive-only, `frozen=True` = fixed window); `state_dict()/load_state_dict()` for checkpointing.
- `agent/gateway.py`: `SlotPool` (adjustable semaphore, FIFO), `Gateway` (global window, retry with truncated exponential backoff + jitter honouring Retry-After, slot released while backing off, every attempt logged and fed to the controller). `agent/scripted.py`: scripted agents with time-phased think time (steady / burst patterns).
- `metrics.py`: n_429/n_503, throttle rate, server latency p50/p95/p99 (successful attempts), SLO-violation rate, mean latency excess, end-to-end latency, goodput, makespan, window stats. `sim.py`: `Scenario` dataclass tree + `run_scenario` (config is JSON-able via `dataclasses.asdict`, for `run_resumable`).
- `experiments/smoke_stage2.py`: smoke test on arbitrary PLACEHOLDER parameters; **not a result, do not cite**.
- All hyperparameters/scenario parameters are required constructor fields with NO defaults (experimental decisions still pending).

**Done (Stage 2b, pushed):** RQ2 design wired in code and tested (60 tests pass). `agent/gateway.py`: `TokenBucketPacer`/`PacerConfig` (client-side baseline; paces retries too). `scenarios.py`: approved scenarios, grids, baselines, protocol constants, `build()`, unit ids (dev = 5,520 runs = (81 ACC + 7 fixed + 4 token-bucket) x 4 scenarios x 3 M x 5 seeds; test = 1,080 runs). `analysis.py`: result loading and dev selection (criterion D15-b; never relaxes constraints silently, reports infeasible families instead). `experiments/run_rq2.py`: resumable CLI (`--phase dev|select|test`, `--root`, `--limit`); `notebooks/01_rq2.ipynb`: Colab driver. Timing: single runs take 0.02-1 s, so dev is about half an hour of CPU. No full run executed yet.

**Done (live view, pushed):** `notebooks/02_live.ipynb` (open in Colab: https://colab.research.google.com/github/notanut/eem-acc-skripsi/blob/main/notebooks/02_live.ipynb). Part A: interactive ACC-vs-fixed-window illustration (`viz.demo_figure`, uses ACC start values, labelled "not a result"). Parts B-D: run the real RQ2 dev / select / test phases in a background thread with a live progress view (`monitor.snapshot`, `viz.monitor_figure`: progress bar, tiles per scenario x M, partial ranking). ETA is weighted by M because run time grows with the number of agents (sample of 60 dev runs: 0.02 s at M=4, 0.07 s at M=16, 0.35 s at M=64; full dev estimated ~15 min on the sandbox, expect 15-40 min on Colab CPU). `sim.simulate()` returns the finished Gateway (attempt/call logs). Notebook verified by executing all cells headlessly here with LIMIT=40 (not in Colab itself). Note: `viz.py`, `monitor.py` and `tests/test_viz_monitor.py` were found uncommitted in the working tree (not written in this turn); reviewed, tests pass, committed as found, then the ETA was changed to the M-weighted version (and its test updated).

**Done (review round, 2026-11-14, pushed: fd50dfc, f4a2ecb, afb2784, 8e4c5c7):** retroactive reviews of the whole RQ2 stack (reviewer A: code; reviewer B: design with 432 sim runs; reviewer C: the fixes). Code fixes applied:
- Gateway: new window applied BEFORE the finished attempt's slot is released (a shrink is no longer undercut by an immediate re-grant); window/in_use logged at grant time; slot released in `finally` (server or controller exception); a cancelled waiter that was already granted returns its slot.
- Common random numbers: server noise keyed by logical call (agent, k), not attempt index; backoff jitter keyed by (agent, k, retry); per-agent cached keyed streams (`src/eem_acc/keyed.py`, same speed as sequential RNG); agents always draw one think-time variate per call.
- Controller `floor()` +1e-9 (12*(1-0.5*25/30) = 6.999...). ResultStore: collision-free filenames; unit_id and file name verified.
- Metrics: `mean_latency_excess_all_attempts` (= Eq. 1's term), `wasted_attempts_per_ok_call`, `objective_J(m, lam, gam)`. Note for the paper: Eq. 1 mixes a per-call success rate with per-attempt latency/throttle terms; say so explicitly and report both excess variants.
- Result integrity: `SIM_VERSION = "3"` in `sim.py` (**bump whenever simulator/gateway/controller/metrics semantics change**); every result stores spec_hash, sim_version, numpy version, source hash. `run_rq2`: resume/select/test refuse results from another config/version or foreign leftover files; `configs/rq2_frozen.json` records sim_version + dev digest; select refuses after test results exist and refuses to overwrite a different frozen file without `--force`; test phase requires repo and Drive copies identical and the dev digest of THIS root.
- Tests: 123 passed, 1 skipped, 1 deselected (fast suite); 2 notebook tests pass in a venv with nbclient (`pytest -m notebook`). `tools/mutation_check.py <scratch>` reverts each of 14 fixes; all are caught. `tests/test_pipeline.py` (12 tests) was found untracked in the working tree (written by another actor on Nov 5), reviewed, kept, committed.
- External review prompt: **`reviews/2026-11-14-rq2-sim-review-prompt.md`** (self-contained, embeds the code). Waiting for the user to bring back another LLM's feedback.
- Practical note: never run two sessions/notebooks on the same Drive results folder at once (temp files of one could be cleaned by the other); use a fresh `--root` per experiment version.

**Done (RQ2 design v2 implemented, 2026-11-14, pushed 7185a71..e14b980):** SIM_VERSION 4. Simulator core (client timeout 504 with server work continuing, kappa, horizon runs that stop at T, open-loop burst, noise keyed by admission), v2 metrics (success over calls requested before T - 20 s, 504 in throttle and SLO, censored e2e, queue, event windows with completion-time goodput, recovery time), v2 scenarios (`src/eem_acc/scenarios.py`, 16,080 dev runs) and selection/statistics (`src/eem_acc/analysis.py`), runner/viz/notebook updated. The v1 scenario/selection code is replaced (still in git history). Implementation notes I1-I10 are in spec section 14.
- Tests: 178 passed, 1 skipped, 1 deselected (fast suite, ~50 s); notebook tests 2 passed (~5 min, venv with nbclient); `tools/mutation_check.py <scratch> "v2:"` catches all 16 v2 reversions. Not yet run inside Colab.
- Independent review of the implementation: H1 (recovery missed post-event damage) and H2 (window goodput by request time) fixed; M5/L1-L4 fixed; H3 (fixed-W grid gap 16-24), M4 (equivalence test), deviation 3 (best kappa=1 config) and recovery client-side condition went to the user as Q-GRID, Q-EQ, Q-K1, Q-REC (spec section 15), plus Q-I1 (success grace).
- External review prompt: `reviews/2026-11-14-rq2-v2-implementation-review-prompt.md`. Section III text draft: `docs/section3_patch_v2.md` (P13-P15: timeouts in f_j and F, L_ctrl = kappa * L_target).

**Done 2026-11-14 (later):** user answered spec section 15 -> D24 (fixed W 12 and 20 added; dev = 16,200 runs), D25 (TOST +-5% for the 'about equal' endpoint, plus paired-t check), D26 (recovery also needs e2e p95 <= 2 x pre), D27 (best kappa=1 ACC config also tested), D28 (success grace 20 s confirmed). Implemented, reviewed (fixes applied), merged (d47b7d5). Tests 185 passed / notebook 2 passed / all v2 mutations caught. Review prompt `reviews/2026-11-14-rq2-v2-decisions-review-prompt.md`.
- Section III patch v2: the user asked for a longer rewrite (keep notation and tone, use widely known references). Revision 3 in `docs/section3_patch_v2.md` (P13-P16: timeouts in f_j and F with RFC 5681 Sec. 3.1 analogy; L_ctrl = kappa * L_target with SRE book ch. 4 'safety margin' analogy; window definition and Algorithm 1 lines 6-8 made per-attempt; retry policy). Both references checked against their sources. Reviewed by a subagent (3 major issues fixed). Review prompt `reviews/2026-11-14-section3-patch-v2-review-prompt.md`.

**In progress:** waiting for the user's approval of Section III patch v2 r3 (then apply it to docs/section3_revised.md and the .docx). **Then:** short Colab trial with RUN_NAME "v2-trial", then the full dev run (RUN_NAME "v2"; ~2 h sandbox / 4-6 h Colab), select, commit frozen config, test run, analysis.

**Implementation assumptions made in Stage 2 (user to confirm):**
1. The p95 latency used for dL is computed over *successful* attempts only (429/503 carry no service time); an interval with no success has dL = 0 (the throttle branch handles it). Section III text says "p95 server latency of the attempts"; suggest adding "successful".
2. Check order in the server: rate limit (429) first, then overload (503).
3. Scripted agents are sequential loops; concurrency comes from M agents; load pattern via time-phased think times.
4. Virtual-time simulation is used for RQ2 instead of a real FastAPI server (cheaper, deterministic). The FastAPI wrapper (needed so LLM agents call a real HTTP API with schema-drift endpoints) is deferred to Stage 3 (RQ1).
5. Not implemented yet: the client-side *token bucket + exponential backoff* baseline (needs a rate pacer in the gateway); the other concurrency baselines/ablations are covered by the ACCConfig flags.

**Drafted, awaiting user review:** `docs/section3_revised.md` and `docs/Section_III_Methodology_revised.docx` (approved patch P1-P12 applied to Section III text; EEM store shared across agents; paper itself not yet edited). The .docx has native Word equations; it was not visually verified (no Math renderer in the sandbox), only checked by round-trip; 4 cosmetic schema notes remain in pandoc's default styles. **Not started (RQ2 runs):** full dev/test runs; analysis and figures. **Not started (other):** FastAPI wrapper; EEM implementation; LLM agent loop; experiments; Chapter IV/V; Abstract; reference verification.

## 5. Repo structure (current -> planned)
```
HANDOFF.md  README.md  pyproject.toml  requirements.txt  .gitignore
notebooks/00_setup.ipynb
src/eem_acc/utils/        # done (see above)
src/eem_acc/env/          # done: vtime.py, server.py
src/eem_acc/acc/          # done: controller.py
src/eem_acc/agent/        # gateway.py, scripted.py done; LLM agent loop planned
src/eem_acc/{metrics,sim}.py  # done
src/eem_acc/eem/          # planned (Stage 3)
experiments/smoke_stage2.py   # placeholder smoke test only
experiments/calibrate_server_profile.py   # design aid (candidate profiles)
experiments/run_rq2.py    # resumable RQ2 runner (dev/select/test)
src/eem_acc/{scenarios,analysis}.py   # approved design constants; dev selection
notebooks/01_rq2.ipynb    # Colab driver for run_rq2 (plain, no live view)
notebooks/02_live.ipynb   # Colab live view + runner (recommended)
src/eem_acc/{viz,monitor}.py   # figures and progress snapshot for the live notebook
docs/section3_patch.md  docs/section3_revised.md  docs/Section_III_Methodology_revised.docx
tests/test_resilience.py  tests/test_acc.py  tests/test_sim.py  tests/test_rq2.py  tests/test_viz_monitor.py   # 66 tests
```
**Drive folder (`Skripsi`, mounted at `/content/drive/MyDrive/Skripsi` via a My Drive shortcut):** `/checkpoints /results /logs /figures /datasets /handoff /code`. Each experiment writes to its own run folder with `units/` (per-unit JSON) and `checkpoints/`.

## 6. How to run
- Colab: open `notebooks/00_setup.ipynb`, edit `DRIVE_ROOT` in cell 1, run all cells. For a private repo add a Colab secret `GITHUB_TOKEN` (read access).
- Local tests: `pip install -r requirements.txt -e . && pytest -q`.
- Live (Colab): open `notebooks/02_live.ipynb` and run the cells top to bottom (Part A illustration, B dev, C select, D test). Results go to `Drive/Skripsi/rq2_runs/<RUN_NAME>/` (set `RUN_NAME` in Part B; default `trial`; use a new name per experiment version). Set `LIMIT = 60` for a quick trial. Stopping a cell stops only the display; restart the runtime to stop a run. (2026-11-14: user is about to try it in Colab; review prompt `reviews/2026-11-14-colab-notebook-review-prompt.md`; tests 124 passed / notebook 2 passed outside Colab; not yet verified inside Colab.) Colab fix 2026-11-14: user's first cell failed with `ModuleNotFoundError: eem_acc` (editable install not visible until runtime restart); all notebooks now add `src/` to sys.path after installing; 02_live warns to restart when a pull brings new code. Review prompt `reviews/2026-11-14-colab-import-fix-review-prompt.md`; tests 128 passed. Then Part A failed with `Cannot run the event loop while another loop is running` (Colab kernels run an asyncio loop): `run_virtual` now runs the simulation in a worker thread when a loop is already running (identical results; cancelled if the cell is stopped). Review prompt `reviews/2026-11-14-colab-event-loop-fix-review-prompt.md`; tests 133 passed. **User confirmed 2026-11-14: 02_live runs fine in Colab (setup, Part A, trial Part B-D with LIMIT).**
- RQ2 (Colab: `notebooks/01_rq2.ipynb`; or shell, repo root): `PYTHONPATH=src python experiments/run_rq2.py --phase dev|select|test --root <ROOT>`; outputs under `<ROOT>/results/rq2/<phase>/units/*.json` (+ checkpoints, log in `<ROOT>/logs/`). Resumable; `select` refuses if dev is incomplete; `test` refuses without a frozen config.
- Resumable experiment pattern: `run_resumable(unit_ids, run_unit, run_dir, config=cfg, stateful=..., get_state=..., set_state=...)`. Re-running the same call resumes automatically. Use `stateful=True` when memory/window state carries across units (EEM store, ACC window), with `checkpoint_every=1`.

## 7. Configuration / hyperparameters in use
Frozen in `src/eem_acc/scenarios.py` exactly as approved (D10-D15 above). ACC grid start values: alpha=0.5, beta=0.5, epsilon=0.1, n_c=16 (centre of the grid; the *selected* values come from the dev phase -> `configs/rq2_frozen.json`, not yet produced). Still pending: lambda/gamma (proposed 0.5/0.5 primary, 1.0/1.0 sensitivity), EEM hyperparameters (k=3, k_s, theta_link), LLM-agent suite and fault rates, MaxSteps (Stage 3).

Implementation assumptions made (user please note): token-bucket baseline uses burst = 1 (pure pacing at `rate`); per-cell goodput is normalised by the best seed-averaged goodput of any dev method in that cell, a method's score is the mean over cells, and a candidate must have results in every dev cell; the SLO/success constraints are applied to cell-and-seed means averaged over cells; a missing SLO value counts as 1.0 (worst).

## 8. Interim results
None. No experiments have been run. (`experiments/smoke_stage2.py` prints numbers from placeholder parameters, only to sanity-check the mechanics; they are not results.)

## 9. Open questions (waiting on the user)
1. ~~GitHub push access~~ **Resolved** (see section 4).
2. **Drive connector (non-blocking).** User confirmed the Google account is the same one used for Colab/Drive. Even so, the connector returns "not found" for the project folder id (`1guM1H9HZEe94s0B8Xmn4GFW-qjoAgvED`) and returns empty result pages for any search, including a broad `modifiedTime > 2020` query, so it appears to see no files (possibly limited scope). Decision: **do not rely on the connector**; all Drive I/O goes through Colab's Drive mount (folder `Skripsi` as a My Drive shortcut), and the Drive copy of this file is made by setup notebook cell 6. Revisit only if the user wants the assistant to write to Drive directly (would need the connector re-authorized with full Drive access).
3. Review `docs/section3_revised.md` / the .docx (done; user approved). Interpretations I made while writing it (flag if you disagree): F and the p95 latency are computed over the current control interval only (n = n_f = n_c, non-overlapping) to avoid double-counting; exhausted retries count as one agent step with no reflection; k_s (seed notes) is a new hyperparameter.
4. ~~EEM memory scope~~ **Decided: shared** (private-per-agent only as a small ablation). User also approved the three interpretations listed in item 3.

5. ~~RQ2 design proposal~~ **Approved 2026-11-05 ("approve all").**
6. **Where to run the full dev/test phases (pending):** (a) here in the assistant sandbox (recommended: ~30 min CPU for dev; I can read the results to select the config and write Chapter IV; results bundle sent to the user to archive on Drive), or (b) in Colab via `notebooks/01_rq2.ipynb` (reproducible, but the assistant cannot read Drive, so the user must upload result files back to the chat). The notebook exists either way.

7. **RQ2 design v2 APPROVED 2026-11-14: `docs/rq2_design_v2.md` (D16-D23). Next: implement it (SIM_VERSION 4), tests, review, Colab trial, full run with RUN_NAME `v2`.** External review prompt: `reviews/2026-11-14-rq2-design-v2-review-prompt.md`. Calibration prototype (scratch, not in repo): horizon/burst/timeout prototype on dev seeds 0-4 only. Original findings for the record:
   - H1: both regimes have optimal static W ~= 8 (= W0); fixed W=8 beats ACC almost everywhere, so scenarios cannot show adaptation value. Options: make the optimal W differ across scenarios / longer drop; cross-regime robustness test (tune W on one regime, test on the other); keep as is and report honestly.
   - H2: averaged SLO constraint hides ~10% violation per loaded cell; eps>0 makes >5% violation structural. Options: per-cell (worst-cell) constraints, constraints over loaded cells only, or Pareto front instead of thresholds.
   - H3: M=4 and CAP-burst do not discriminate. Options: open-loop/arrival-rate burst, higher M levels, drop non-discriminating cells.
   - M1 absolute-time events vs method-dependent run length (report event-window transient metrics + recovery time); M2 coarse token-bucket grid (finer 20-45/s grid, optional in-flight cap); M3 normalisation by possibly infeasible best method (use best feasible / capacity / ranks); no timeouts in the simulator although the paper mentions them; tie tolerance and statistical plan (CI, paired bootstrap) to fix before running.

## 10. Next steps
0. User answers the design questions in open question 7 (and optionally returns external-LLM feedback on the review prompt).
1. User decides where to run the full RQ2 phases (open question 6).
2. Run dev (5,520 runs) -> `select` -> commit `configs/rq2_frozen.json` -> run test (1,080 runs). No peeking at test seeds before freezing.
3. Analysis + figures for Chapter IV (RQ2/RQ4 sweeps, 95% CI, paired bootstrap), honest reporting of whatever comes out.
4. Stage 3: FastAPI wrapper with schema-drift endpoints, EEM, LLM agent loop (llama-cpp-python, Qwen2.5-7B-Instruct 4-bit), baselines, then RQ1/RQ3 experiments with checkpoint/resume on Colab T4.
5. Chapter IV, V, Abstract; verify every cited reference.
