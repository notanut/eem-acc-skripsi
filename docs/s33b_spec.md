# S3.3b specification: notebook `04_stage3.ipynb`, T4 verification, epoch 0, smoke run, D53 benchmark

_r1, 2026-11-22. Implements D128 (benchmark section of the S3.3b notebook) and the verification list of
docs/s33_spec.md section 15. User decisions D135-D138 (section 8). Nothing here is a thesis result._

## 1. Shape (two T4 sessions)

The real chat template must be committed, and the canary prompts recounted with it, **before** epoch 0 is registered
(the canary file's sha enters the epoch; a dirty Colab checkout never resumes). Hence:

- **Session 1, verification (about 15 min, before the budget counter starts):** setup, load, template capture,
  equality with S3.0, D47 on the GPU, load parameters, Colab-marked tests, a timing spot check. A results bundle is
  downloaded and uploaded to the chat (D135); I commit the real template (replacing the reconstruction), regenerate
  `configs/canary_prompts.json`, update the tests and push.
- **Session 2, epoch 0 + smoke + benchmark (about 1.6 h):** setup, load, session canary -> epoch 0 (fresh path), the
  smoke units with a guided interruption across runtimes, CPU `verify_replay` and the audit, then the D53 benchmark (the
  GPU-hour counter of D52 starts at the benchmark). A second bundle is downloaded and uploaded.

Every section saves into `Skripsi/stage3/s33b/<session>.json` on Drive and is skipped when already done (rerun-safe).
The notebook is thin: logic lives in `src/eem_acc/stage3/llm/t4.py` (CPU-tested with the tiny GGUF) and the S3.3a
modules.

## 2. Setup (both sessions)

Drive mount; repo clone or `git pull --ff-only` and a **clean-tree check** (the notebook never writes into the
checkout); the S3.0 wheel cache (`Skripsi/wheels`, wheel sha recorded) and model cache (`Skripsi/models`, part
sha256s); the GGUF parts copied to `/content/models` (local disk, shared page cache for two processes);
`pip install sentence-transformers==<pinned>`; MiniLM at a pinned revision. Load: `Llama(n_gpu_layers=-1, n_ctx=8192,
n_batch=512, seed=0, verbose=False)` as in S3.0. Model digest = sha256 over the sorted part sha256s.
`session_env()` with GPU, driver, CUDA, Python, numpy, torch, sentence-transformers, CPU model. Runtime id =
`ledger.runtime_id()`; the session heartbeat (ledger, lease, claims) starts right after setup.

## 3. Session 1: verification

1. **Template:** the GGUF's `tokenizer.chat_template` sha256 must equal S3.0's `d5495a1e...`; its text goes into the
   bundle.
2. **Equality with S3.0:** S3.0's 20 free and 20 grammar timing records (fresh path via `create_chat_completion`,
   `max_tokens` 160, prompts `feasibility.sample_prompts(20, seed=0)`) are regenerated with `LlamaCppBackend` (same
   grammar schema); `n_in`, `n_out`, `finish_reason` and the first 500 characters of the text (what S3.0 stored) must be
   equal for all 40. Any difference stops the S3.3b plan (report; no epoch is registered).
3. **D47 on the GPU:** on the two longest canary prompts: (a) after a longer unrelated call and (b) after a grammar
   call, the reset path's output equals the output of a freshly constructed `Llama` (the model reloaded from local
   disk) for the same prompt, bitwise (full `Gen`).
4. **Load parameters:** `load_params_of(llm)` recorded (including `flash_attn`/`flash_attn_type`, `n_ubatch`,
   `type_k`/`type_v`); compared with S3.0 where S3.0 recorded them (n_ctx, n_batch); the rest are recorded as the
   reference for later sessions.
5. **Colab-marked tests:** `pytest -m colab tests/test_s33_colab.py` with `EEM_GGUF` and `EEM_MINILM_REV` set
   (counts and failures into the bundle). The snapshot test checks repeatability only.
6. **Timing spot check:** seconds per call of the 40 calls vs `t_llm`'s prediction (median ratio and spread), to see
   that the S3.0 constants still describe this platform (recorded, not a gate).

## 4. Session 2: epoch 0, smoke run, guided interruption

1. **Session canary** (fresh path): `run_canary` on the committed reference prompts; `EpochStore.resolve` must give
   epoch 0 "new" the first time; the fingerprint goes into the header and the `Session`.
2. **Smoke plan (D136)**, phase `smoke`, plan version 1: RQ1 dev stream 1000 C0 and C2 with 8-task prefixes, RQ3 dev
   seed 5000 C2 + ACC; plus `smoke-ref`: a second copy of the RQ1 C2 unit run uninterrupted (reference for the
   interruption test). Runner with lease, claims and the session heartbeat; the scripted policy is never used here.
3. **Guided interruption across runtimes (D138):** the first smoke unit runs with a stop hook that halts the process
   after N fresh LLM calls (N = 12) and prints the records written. The notebook then tells the user to use **Runtime ->
   Disconnect and delete runtime**, reconnect, and re-run the setup cells (a new runtime: new VM, new runtime id). On
   reconnect the notebook records what reached Drive (cache records and whether the last line is torn, vs the count
   the user copies from the earlier output), the canary status in the new runtime (expected "pass" or a returned
   epoch), and resumes the unit. After `smoke-ref` finishes, the interrupted unit's payload must equal the reference's,
   with LLM records compared separately from embedding records (the CPU model may differ).
4. **Smoke checks:** `verify_replay` of every smoke unit on CPU, using `key_source_from_header` with a vocab-only
   `Llama` on the local GGUF; `audit_phase` (including note events) must give zero hits; per condition: fallback rate,
   action truncation rate, format errors, prompt-size distribution, LLM calls per task, `retok_diff` rate, control-token
   check firings (expected 0), transient notices (RQ3), drift errors; Drive latency of append + fsync per record and
   cache bytes per unit. These are the inputs of the post-pilot budget projection (design section 9).

## 5. Session 2: D53 throughput benchmark (D137)

- **Counter:** a `benchmark_start` ledger event starts the D52 GPU-hour count.
- **Workload:** the rendered prompts recorded in the smoke caches (`prompt_z`), up to 60 LLM calls drawn
  deterministically (seeded, stratified by kind and grammar), each with its recorded grammar and `max_tokens`; run
  directly from tokens (`llm.tokenize(prompt, add_bos=False, special=True)`), measurement only.
- **Configurations:** (A) fresh path, one process; (B) static-prefix snapshot, one process (prefix = tokens up to and
  including the first `<|im_end|>`; a prompt without a system turn uses the fresh path); (C) fresh path, two worker
  processes (each loads the model from local disk, `n_threads` 1 each, runs half of the workload; wall time of both).
  A and B are interleaved per prompt to share thermal conditions; C runs after A and B.
- **Measured:** seconds per call and total wall per configuration; for B: exact repeatability (each prompt twice),
  host RSS per saved state and with the LRU full, the first eval starting at the prefix length (eval spy), and its
  canary outputs (a separate epoch series if adopted); for C: peak VRAM, both workers' canary outputs while the other
  generates (equal to epoch 0 or a separate epoch path), and the throughput gain over A.
- **Adoption** follows the design's rules (B: canary repeatable and memory holds; C: at least 20% gain and canary
  conditions), but the numbers are reported to the user, who decides adoption before the pilot (standing rule).

## 6. Bundles

`s33b_session1.json` and `s33b_session2.json`: stamps (env, load params, model digest, wheel sha, template sha, code
commit), every section's result, and for session 1 the template text. Downloaded at the end and uploaded to the chat;
they go to `docs/evidence/` after review.

## 7. Tests (CPU, tiny GGUF)

`t4.py` functions on the tiny model: S3.0-record equality logic (with fabricated records from the tiny model), D47
comparison logic, load-parameter recording, smoke plan (units, ids, plan version), the stop hook and the
interruption-report logic, the benchmark workload selection (deterministic, stratified) and configurations A/B (and C
with two subprocesses on the tiny model), bundle assembly; the notebook's structure (cells in order, no writes into the
checkout) when nbformat is available.

## 8. Decisions (user, 2026-11-22)

| ID | Decision |
|---|---|
| D135 | T4 results return as a downloaded bundle uploaded to the chat; I commit; no GitHub token in Colab |
| D136 | Smoke run: RQ1 dev stream 1000 C0 and C2 with 8-task prefixes, one RQ3 dev unit (C2 + ACC) |
| D137 | D53 benchmark workload: up to 60 prompts recorded in the smoke caches, replayed through each configuration |
| D138 | The interruption-across-runtimes and forced-recycle tests run in session 2, guided by the notebook |
