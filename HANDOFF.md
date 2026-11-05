# HANDOFF

_Last updated: 2026-11-01 (Stage 1 pushed to GitHub). Update after every milestone and before ending a session._

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

Repo/Drive layout approved as proposed (below).

## 4. Current status
**Done (Stage 1, local):**
- Repo cloned (was empty); `.gitignore`, `README.md`, `pyproject.toml`, `requirements.txt` (minimal: numpy, pyyaml, pytest).
- `src/eem_acc/utils/`: `seed.py`, `atomic.py` (temp file + `os.replace`), `checkpoint.py` (checksummed checkpoints, rotation, fallback to previous valid one, RNG state), `results.py` (per-unit JSON), `progress.py` (ETA), `runner.py` (`run_resumable`: stateless/stateful resume, config-mismatch guard), `paths.py` (Drive layout).
- `tests/test_resilience.py`: 12 tests pass locally (also in a clean venv after `pip install -e .`).
- `notebooks/00_setup.ipynb`: Colab setup (mount Drive, clone/pull, install, seed, create Drive folders, run tests, optional code sync). **Syntax-checked only; not yet executed in Colab.**

**Resolved - GitHub push:** the Claude GitHub App was installed by the user; all Stage-1 commits are on `origin/main` (verified: remote HEAD == local HEAD at commit `c0062ad`). Pushing works from the assistant's shell.

**In progress:** none.

**Drafted, awaiting user review:** `docs/section3_revised.md` and `docs/Section_III_Methodology_revised.docx` (approved patch P1-P12 applied to Section III text; EEM store shared across agents; paper itself not yet edited). The .docx has native Word equations; it was not visually verified (no Math renderer in the sandbox), only checked by round-trip; 4 cosmetic schema notes remain in pandoc's default styles. **Not started:** mock API server; scripted agents; ACC implementation; EEM implementation; LLM agent loop; experiments; Chapter IV/V; Abstract; reference verification.

## 5. Repo structure (current -> planned)
```
HANDOFF.md  README.md  pyproject.toml  requirements.txt  .gitignore
notebooks/00_setup.ipynb
src/eem_acc/utils/        # done (see above)
src/eem_acc/{eem,acc,agent,env}/   # planned, not created yet
experiments/  configs/  docs/      # planned (empty dirs are not tracked by git)
tests/test_resilience.py
```
**Drive folder (`Skripsi`, mounted at `/content/drive/MyDrive/Skripsi` via a My Drive shortcut):** `/checkpoints /results /logs /figures /datasets /handoff /code`. Each experiment writes to its own run folder with `units/` (per-unit JSON) and `checkpoints/`.

## 6. How to run
- Colab: open `notebooks/00_setup.ipynb`, edit `DRIVE_ROOT` in cell 1, run all cells. For a private repo add a Colab secret `GITHUB_TOKEN` (read access).
- Local tests: `pip install -r requirements.txt -e . && pytest -q`.
- Resumable experiment pattern: `run_resumable(unit_ids, run_unit, run_dir, config=cfg, stateful=..., get_state=..., set_state=...)`. Re-running the same call resumes automatically. Use `stateful=True` when memory/window state carries across units (EEM store, ACC window), with `checkpoint_every=1`.

## 7. Configuration / hyperparameters in use
None chosen yet. alpha, beta, lambda, gamma, W0, W_max, W_min, epsilon, tau, L_target, MaxSteps are all **pending** (D7 + Section III patch). Seed convention proposed: dev seeds and test seeds are separate sets (values not yet fixed).

Interpretations are proposed in `docs/section3_patch.md` (**NOT confirmed**). Nothing in Section III has been changed.

## 8. Interim results
None. No experiments have been run.

## 9. Open questions (waiting on the user)
1. ~~GitHub push access~~ **Resolved** (see section 4).
2. **Drive connector (non-blocking).** User confirmed the Google account is the same one used for Colab/Drive. Even so, the connector returns "not found" for the project folder id (`1guM1H9HZEe94s0B8Xmn4GFW-qjoAgvED`) and returns empty result pages for any search, including a broad `modifiedTime > 2020` query, so it appears to see no files (possibly limited scope). Decision: **do not rely on the connector**; all Drive I/O goes through Colab's Drive mount (folder `Skripsi` as a My Drive shortcut), and the Drive copy of this file is made by setup notebook cell 6. Revisit only if the user wants the assistant to write to Drive directly (would need the connector re-authorized with full Drive access).
3. Review `docs/section3_revised.md` / the .docx (user wants .docx: done). Interpretations I made while writing it (flag if you disagree): F and the p95 latency are computed over the current control interval only (n = n_f = n_c, non-overlapping) to avoid double-counting; exhausted retries count as one agent step with no reflection; k_s (seed notes) is a new hyperparameter.
4. ~~EEM memory scope~~ **Decided: shared** (private-per-agent only as a small ablation). User also approved the three interpretations listed in item 3.

## 10. Next steps
1. Fix Drive visibility (open question 2).
2. User reviews `docs/section3_revised.md`; settle EEM memory scope; then propose hyperparameter starting values (D7) once the mock-server latency model is sketched.
3. Confirm libraries/model (Q3), then Stage 2: mock API server + ACC + scripted agents (RQ2), with tests.
4. Stage 3: EEM + LLM agent loop (RQ1). Stage 4: experiments with checkpoint/resume on Colab. Stage 5: Chapter IV, V, Abstract; verify every cited reference.
