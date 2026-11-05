# HANDOFF

_Last updated: 2026-11-05 (RQ2 design implemented; awaiting decision on where to run)._ Update after every milestone and before ending a session._

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

**In progress:** none.

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
notebooks/01_rq2.ipynb    # Colab driver for run_rq2
docs/section3_patch.md  docs/section3_revised.md  docs/Section_III_Methodology_revised.docx
tests/test_resilience.py  tests/test_acc.py  tests/test_sim.py  tests/test_rq2.py   # 60 tests
```
**Drive folder (`Skripsi`, mounted at `/content/drive/MyDrive/Skripsi` via a My Drive shortcut):** `/checkpoints /results /logs /figures /datasets /handoff /code`. Each experiment writes to its own run folder with `units/` (per-unit JSON) and `checkpoints/`.

## 6. How to run
- Colab: open `notebooks/00_setup.ipynb`, edit `DRIVE_ROOT` in cell 1, run all cells. For a private repo add a Colab secret `GITHUB_TOKEN` (read access).
- Local tests: `pip install -r requirements.txt -e . && pytest -q`.
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

## 10. Next steps
1. User decides where to run the full RQ2 phases (open question 6).
2. Run dev (5,520 runs) -> `select` -> commit `configs/rq2_frozen.json` -> run test (1,080 runs). No peeking at test seeds before freezing.
3. Analysis + figures for Chapter IV (RQ2/RQ4 sweeps, 95% CI, paired bootstrap), honest reporting of whatever comes out.
4. Stage 3: FastAPI wrapper with schema-drift endpoints, EEM, LLM agent loop (llama-cpp-python, Qwen2.5-7B-Instruct 4-bit), baselines, then RQ1/RQ3 experiments with checkpoint/resume on Colab T4.
5. Chapter IV, V, Abstract; verify every cited reference.
