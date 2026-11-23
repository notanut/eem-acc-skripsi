# S3.3b specification: notebook `04_stage3.ipynb`, T4 verification, epoch 0, smoke run, D53 benchmark

_r3, 2026-11-23 (r2 + the code review, section 10, D141). Implements D128 (benchmark section of the S3.3b notebook) and the verification list of
docs/s33_spec.md section 15. r1 was reviewed by a fresh-context subagent (3 High, 6 Medium, Lows); the user decided
D139-D140; the clear-cut findings are applied. User decisions D135-D140 (section 9). Nothing here is a thesis result._

## 1. Shape (two T4 sessions)

The real chat template must be committed, and the canary prompts recounted with it, **before** epoch 0 is registered
(the canary file's sha enters the epoch; a dirty checkout never resumes). Hence:

- **Session 1, verification (about 20-25 min; before the budget counter starts):** setup, load, template capture,
  equality with S3.0, D47 on the GPU, load parameters and the libraries llama.cpp links, real counts of the current
  canary prompts, the MiniLM revision and library versions to pin, Colab-marked tests, a timing check. The bundle is
  downloaded and uploaded (D135); I commit the real template (replacing the reconstruction), the pins
  (`configs/t4_pins.json`), regenerate `configs/canary_prompts.json`, update the tests, and push to `main`.
- **Session 2, epoch 0 + smoke + kills + benchmark (about 2 h, three runtimes):** setup and load; canary -> epoch 0;
  smoke units with two forced kills (D140) and resumes in new runtimes; CPU `verify_replay` and the audit; the D53
  benchmark (the GPU-hour counter starts at `benchmark_start`). The bundle is downloaded and uploaded.

**Code pinning:** the first runtime of a session checks out `origin/main` and records the commit in the session file
on Drive; every later runtime of that session checks out the recorded commit (detached HEAD, never `git pull`) and
asserts a clean tree, so a commit pushed meanwhile cannot void the resume tests. [r3: tags were planned, but pushing
tags is refused by the session's git policy (HTTP 403); a recorded commit pins the code equally.] **Drive root:** the campaign root `Skripsi/stage3` (runner, `canary/epochs.json`,
ledger, lease, claims), so epoch 0 is the campaign's. Section results go to `Skripsi/stage3/s33b/<session>.json`.
**Per-runtime sections** (setup, load, canary, lease, claims) always run; measurement sections are skipped when saved.
The notebook is thin: logic lives in `src/eem_acc/stage3/llm/t4.py` and `unit_worker.py`, CPU-tested with the tiny
qwen2 GGUF.

## 2. Setup (every runtime)

Record VM boot time (`/proc/stat` btime), the first cell's `time.time()` and the Drive-mount time (s33 15.10). Drive
mount; checkout of the session tag, commit and clean-tree assertion; S3.0's wheel cache (wheel sha recorded); model
cache (part sha256s) and a copy to `/content/models` **verified against S3.0's part sha256s** (dfce12e3...,
539cf93f...). Session 1 records the preinstalled sentence-transformers, transformers and torch versions and resolves
the MiniLM commit sha (40 characters); session 2 installs the pins from `configs/t4_pins.json` **before anything imports
transformers** (else restart). Load: `Llama(n_gpu_layers=-1, n_ctx=8192, n_batch=512, seed=0, verbose=False)` as in
S3.0 (A's thread settings recorded). GPU check: no T4 -> stop (on reconnect without a T4: wait and retry; never run
on CPU). At most **two** `Llama` instances exist at any time (about 5.2 GB VRAM each).

## 3. Session 1: verification

1. **Template:** sha256 of `tokenizer.chat_template` must equal S3.0's `d5495a1e...`; the text goes into the bundle.
2. **Equality with S3.0:** S3.0's 20 free and 20 grammar timing records are regenerated with `LlamaCppBackend` in
   S3.0's exact order (the warm-up call, then free/grammar alternating by `i % 2`), grammar = `envelope_cap300`,
   `max_tokens` 160, prompts `feasibility.sample_prompts(20, seed=0)`; `n_in`, `n_out`, `finish_reason` and the text's
   first 500 characters (all stored texts are shorter) must be equal. **Interpretation rule** for a mismatch: the
   mismatching prompts are regenerated with S3.0's own path (`feasibility.make_generate`, `create_chat_completion`):
   backend differs from that path -> a code bug; neither reproduces S3.0 -> the platform changed. Either stops the
   plan (no epoch). Coverage limits stated: n_in <= 2,627, n_out <= 105, no output stopped at max_tokens.
3. **D47 on the GPU:** on the two longest canary prompts, through `LlamaCppBackend.generate` on both sides: (a) after an
   unrelated call at least as long as the target + 320 tokens (about 6,400 tokens) and (b) after a grammar call, the
   reset path's full `Gen` equals the first output of a freshly constructed `Llama` (loaded from local disk). The fresh
   model is closed and collected before step 6.
4. **Load parameters and libraries:** `load_params_of(llm)` (with `flash_attn`/`flash_attn_type`, `n_ubatch`,
   `type_k/type_v`); equal constructor arguments and wheel to S3.0 plus step 2's bitwise equality are the evidence; the
   CUDA runtime and cuBLAS that llama.cpp links are recorded (`/usr/local/cuda` version, `ldd` of the library).
5. **Canary prompts under the real template:** real token counts of the 8 committed prompts (to check the sandbox
   recount).
6. **Colab-marked tests:** `pytest -m colab tests/test_s33_colab.py` (`EEM_GGUF`, `EEM_MINILM_REV`); counts and
   failures into the bundle.
7. **Timing check:** seconds per call of the 40 calls vs `t_llm`; flagged if the median ratio leaves 1 +/- 0.15 (S3.0:
   median 0.98, SD 0.08). The constants are frozen (D49): a deviation changes only the budget projection.

## 4. Session 2: epoch 0, smoke run, forced kills

1. **Canary:** the committed `n_tokens` are asserted against real counts; `run_canary` (fresh path) -> `EpochStore`
   epoch 0 "new" in runtime 1; the fingerprint goes into the header and the `Session`.
2. **Smoke plan (D136)**, phase `smoke`, in this order: RQ1 dev 1000 **C2** (8-task prefix; the kill-(a) target),
   RQ1 dev 1000 C0 (8 tasks; the kill-(b) target), RQ3 dev 5000 C2 + ACC; phase `smoke-ref`: a copy of RQ1 C2 run
   uninterrupted **first** (the reference). Runner with claims, the parent's lease and the session heartbeat.
3. **Kill (a), mid-unit (D140):** armed only if the target has no cache on Drive. After the N-th record (N = 12) is
   appended, the runtime is killed from code with no pause (`google.colab.runtime.unassign()`). Runtime 2 (new VM):
   setup, load, canary (expected "pass" for epoch 0; a new epoch makes the test inconclusive, reported), claims of other
   runtimes are broken once this runtime holds the lease; **before** the cache is opened, its Drive state is read
   without writing: records present (vs N), torn last line, bytes. Then the unit resumes; it must actually run in
   runtime 2 (asserted). Verdict vs `smoke-ref`: the LLM record sequences (key, text, n_in, n_out, finish_reason) must be
   equal; payload equality (without `unit`) is required only when all embedding vectors are equal; otherwise the first
   differing vector and whether retrieval changed are reported.
4. **Kill (b), result before cache (D140):** armed only if the C0 unit has no result on Drive. Right after its result
   file is written, the runtime is killed from code. Runtime 3: read without writing: result present, its
   `cache_bytes` vs the cache's size, records present, `is_done`; then `run_phase` must either count it done or repair
   it (resume and identical payload).
5. **Smoke checks:** `verify_replay` of every smoke unit on CPU (`key_source_from_header`, vocab-only `Llama` on the
   local GGUF); `audit_phase` with note events: zero hits; per condition: fallback rate, action truncation, format
   errors, prompt sizes, LLM calls per task, `retok_diff` rate, control-token firings (0), transient notices (RQ3),
   drift errors; Drive append + fsync latency per record, cache bytes per unit. These and the smoke results go into the
   bundle (inputs of the post-pilot budget projection).

## 5. Session 2: D53 benchmark

- **Counter:** a `benchmark_start` ledger event; GPU-hours count each runtime from max(its start, the first
  `benchmark_start`) (`ledger.summarise`), so verification and smoke sessions are outside the 60 h.
- **A vs B (snapshot), token replay (D137):** up to 60 distinct recorded LLM calls from the smoke caches, deterministic,
  stratified by (kind, grammar) (empty strata allowed), de-duplicated. Each runs from its rendered prompt's tokens
  (`len(tokens)` must equal the recorded n_in) with the backend's sampling, grammar, stop and stopping criteria; A's
  output must equal the recorded output (the replay is the production computation). B goes through
  `backend._load_prefix_state` (prefix = tokens up to the system turn's end; **reflection records use the fresh path**,
  as in `LlamaCppBackend`). One warm-up call; A/B order alternates per prompt; B is repeated for exact repeatability;
  GPU clocks, temperature and throttle reasons are logged. Snapshot RAM: host RSS with the LRU filled by three distinct
  system prefixes (the docs-wording variants), and the first eval at the prefix length (eval spy). B's canary runs
  through `LlamaCppBackend(snapshot=True)` on the messages (its fingerprint recorded; a separate epoch series if
  adopted).
- **C (two processes), real units (D139):** before C the kernel's model is closed, snapshot states dropped and the
  freed VRAM asserted (nvidia-smi). Phase `bench2` = the two RQ1 smoke units again, run by two worker processes started
  as fresh interpreters (never `fork`), one runner shard each (`unit_worker`), each with its own model,
  `n_threads = n_threads_batch = 1`, claims on, the parent holding the lease and heartbeat; each worker runs the canary
  while the other generates (fingerprint vs epoch 0). Measured: wall time from a synchronised start (after both loads)
  vs the single-process wall times of the same units in the smoke run; peak VRAM; payloads (without `unit`) must equal
  the smoke results.
- **Adoption:** the design's rules (B: canary repeatable, memory holds; C: >= 20% gain and canary conditions); the
  numbers go to the user, who decides before the pilot.

## 6. End and bundles

A final cell stops the heartbeat, releases the lease and (optionally) the runtime. `s33b_session1.json` and
`s33b_session2.json`: stamps (env incl. llama.cpp's CUDA/cuBLAS, load params, model digest, wheel sha, template sha,
code commit), every section's result, the template text (session 1), the smoke results (session 2). They go to
`docs/evidence/` after review.

## 7. Library changes in this milestone

`ledger.summarise(budget_from="benchmark_start")`; `Claims.break_other_runtimes` (only while holding the lease);
`Runner.after_result` hook (kill (b)); a kill hook after the N-th cache append (kill (a)); `unit_worker` CLI.

## 8. Tests (CPU, tiny GGUF)

S3.0-equality logic in S3.0's order and the interpretation rule (fabricated records); D47 logic; smoke plan order and
arming rules; kill hooks with an injected kill function; read-only Drive inspection; LLM vs embedding comparison;
benchmark workload selection (stratified, deduplicated, empty strata) and A/B with A equal to the recorded outputs and
reflections on the fresh path; `unit_worker` with two subprocesses on the tiny model (shards, claims, payload equality);
ledger budget counting from `benchmark_start`; breaking foreign claims; the notebook's structure (cell order, tag
checkout, no writes into the checkout) when nbformat is available.

## 9. Decisions (user, 2026-11-22)

| ID | Decision |
|---|---|
| D135 | T4 results return as a downloaded bundle uploaded to the chat; I commit; no GitHub token in Colab |
| D136 | Smoke run: RQ1 dev stream 1000 C0 and C2 with 8-task prefixes, one RQ3 dev unit (C2 + ACC) |
| D137 | D53 benchmark A vs B: up to 60 prompts recorded in the smoke caches, replayed from tokens |
| D138 | The interruption-across-runtimes and forced-recycle tests run in session 2, guided by the notebook |
| D139 | Benchmark C (two processes) is measured on real units: the two RQ1 smoke units re-run with two runner shards vs their single-process wall times; payloads must equal |
| D140 | Two forced kills from code (no pause): (a) after the N-th cache record of RQ1 C2, (b) right after the C0 result is written |
| D141 | Kill arming is recorded in the session file (with the runtime) before the run; a kill armed earlier whose target file is missing is recorded as lost and never re-armed |

## 10. Code review (2026-11-23): changes

3 High, 8 Medium, 9 Low; all applied. S3.0's records all say `"prompt": 0` (S3.0 benchmarked one prompt per call):
records are paired by list position (20 per kind asserted). `unassign()` returns before the VM dies: `colab_kill`
blocks (no further writes) and raises `KillFailed` (a BaseException, never retried) if the VM is still alive after 300 s;
`KillAfter` raises it too; a `<kill>_requested` record is saved first. Benchmark B's timing includes the reset or the
snapshot-state load (as production pays), and state creations are counted separately. Each kill's arming and each Drive
read record the runtime; the Drive read after kill (a) is keyed on the arming record (the "requested" record may itself
be lost) and must happen in another runtime; an armed kill that never fired is recorded; a kill-(b) verdict (counted
done without rerun, payload identical) is saved; D141. Benchmark C: a fresh phase per attempt (`bench2-a<k>`), guarded
model close, worker output to log files, an overall timeout, two outputs with ran 1 / failed 0 required (else
`bench_c_failed`), late starts recorded, the single-process baseline from each unit's first uninterrupted run in the
ledger, VRAM polled every 5 s. Order guards: the smoke checks need the smoke run complete, the benchmark needs the
checks (the budget counter cannot start early), C needs A/B. Stamps: wheel sha (vs S3.0's `ab93c390...`), part sha256s,
model digest, template sha, load parameters, environment and llama.cpp's CUDA libraries in both sessions. Smoke
statistics add format errors, drift errors, transient notices, control-token firings and the run metadata (cache bytes,
wall, restarts); `compare_units` reports whether retrieval changed; `smoke_runs` are appended, never overwritten. The
RAM probe starts with an empty LRU; re-running cell 8 closes the old session; `break_other_runtimes` checks that this
runtime holds the lease; the workload keeps at least one call per stratum and rejects an empty workload; the worker's
header uses the backend's stamp in every mode. The intro warns never to use "Restart session" in session 2. An
optional runtime release ends the notebook.
