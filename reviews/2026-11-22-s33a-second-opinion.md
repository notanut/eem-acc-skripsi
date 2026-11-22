Now probes for the suspected runner issues.

**Verdict: S3.3b can build on S3.3a.** Nothing I found breaks replay: every result can still be rebuilt from its cache. The problems are in the bookkeeping around it. A unit's recorded stamp can disagree with its actual outputs; some units become stuck so they never run again; deterministic errors waste GPU time; and the scrub audit doesn't see every note. Fix H1, M4 and M5 before the smoke run, because the smoke run registers epoch 0, uses the lease and runs the audit. Fix M1–M3 before the pilot.

All 83 S3.3a tests pass on a CPU build of llama-cpp-python 0.3.36 compiled here (64 without the wheel, 19 with it), and the full suite gives 838 passed. I checked the claims against the installed 0.3.36 source and with small probe scripts.

**What holds:**
- **Rendered path:** it matches the chat path. The handler adds the formatter's stop and stopping criteria the same way, and the parameters the backend doesn't pass get the same defaults.
- **Fresh KV (D47):** it is real. `reset()` sets `n_tokens = 0`, so the prefix match is skipped, and the first `eval` clears every KV cell. On CPU, a call made after a longer, unrelated call produced exactly the same output as a brand-new context (565-token prompt).
- **Snapshot:** after `load_state`, the prefix match reuses exactly the prefix length, because `_input_ids` holds only the prefix.

## High

**H1. A canary that fails mid-session registers a new epoch and keeps running.** The spec says: if the re-run canary fails, the failed-canary rule applies (stop, alert the user). The code raises `PlatformChanged`, the runner switches the header to the new epoch, and units keep running.
- **Probe:** two interrupted units, then a session whose outputs differ from the cache.
  - Unit A restarted under epoch 0. Its outputs came from the platform that the canary then said had changed.
  - Unit B and every later unit got epoch 1, which was written into `epochs.json` mid-session.
  - Every further mismatch re-runs the canary and can add yet another epoch.
- **Why it matters:** Colab's GPU, driver and wheel cannot change inside a running process. A mid-session difference means the platform is nondeterministic, not that it changed. The bad epoch stays in the shared registry for good, and later sessions can match it.
- **Fix:**
  - Re-run the canary against this session's own fingerprint, never through `EpochStore.resolve`.
  - If it differs, raise a halting error (e.g. `SessionNondeterministic`), log it in the ledger and run no more units in this session.
  - Only a pass enters the long-prompt nondeterminism mode (D66).

## Medium

**M1. `is_done` never checks that the stored result matches the planned unit.**
- **Cause:** `params` is not part of `unit_id`. `is_done` checks only the payload hash and the cache size.
- **Probe:** plan v1 with `k=3`, then plan v2 with `k=4` and the same id. The unit counted as done, `ran = 0`, and the stored result still says `k=3`.
- **Same gap elsewhere:**
  - A frozen-commit change in a new test-phase plan version keeps old-commit results.
  - Dev units re-run under a new commit only when Drive happens to lose their cache tail, so commits mix arbitrarily.
- **Fix:**
  - In `is_done`, require `payload["unit"] == u.to_json()`. When the plan has a frozen commit, also require `stamp.llm.code_commit == frozen_commit`.
  - Add `p<sha8(params)>` to `unit_id` whenever params are set.
  - On a mismatch, archive the old result and cache, then re-run.

**M2. Failed units are skipped forever, including the ones the D130 halt marked.**
- **Probe:** two units hit the same CUDA error, the run halted, and both were written as `failed` in the manifest. After the fix, `run_phase` ran only the third unit and skips the first two for good. Units that hit the attempt cap are stuck the same way.
- **Fix:**
  - When D130 halts, write `retry` lines for the units that triggered it, and add a `--retry-failed` option.
  - `failed_units` should use the last status line per unit.
  - `run_phase` and the progress line should report the failed count.

**M3. A deterministic exception costs a full regeneration of the unit.**
- **Cause:** a generic exception archives the cache and restarts from scratch.
- **Probe:** a failure at call 30 threw away 59 records (29 of them LLM calls). The retry regenerated every LLM call: 79 calls for a 50-call unit.
- **Why it's unnecessary:** an exception never leaves a half-written record, so the cache is still valid.
- **Fix:**
  - Retry by resuming, without archiving. If the same error comes back during replay, mark the unit failed at no GPU cost; if not, it was transient and no work is lost.
  - Keep archiving only for corrupt caches, divergence, stamp mismatch and resume mismatch.

**M4. `atomic_write` uses one shared temp name, and the heartbeat thread dies silently.**
- **Cause:** both workers refresh `lease.json` through the same `lease.json.tmp`, and their 60 s timers line up if they start together.
- **Interleaved-write probe:**
  - The lease file came out as `{"runtime": "rt", "t": 1732300000.2}3456}`.
  - The next `Lease.acquire` raised `JSONDecodeError`, so nothing can start until someone deletes the file by hand.
  - The second writer's `os.replace` raised `FileNotFoundError`.
- **Heartbeat probe:** one `OSError` in `beat()` killed the thread (`is_alive() == False`). After that, GPU-hours are undercounted and the lease goes stale while the units are still running.
- **Also:** `refresh()` overwrites without reading, so two runtimes would overwrite each other forever, and the lease is never released at the end of a run.
- **Fix:**
  - Use unique temp names (`.{name}.{pid}.{uuid8}.tmp`).
  - Let only the parent refresh the lease.
  - Wrap `beat()` in try/except and log failures to the ledger.
  - Make `refresh()` read the file first and raise `LeaseLost` if another runtime owns it.
  - Release the lease on exit.

**M5. The scrub audit misses intermediate note texts.**
- **Cause:** it scans `final_notes` and step k_t only. Merged, overwritten or evicted note versions exist only in `note_events`, but the model still saw them.
- **Probe:** scripted C2 on dev stream 1000: 4 of 22 distinct note texts never reach `final_notes`. Real runs will have more.
- **Fix:** also scan `note_events` (`reflection`, `text_raw`, `line`, `text`), and report hits per condition.

## Low

1. **`verify_replay` key source.**
   - A vocab-only `Llama` reports `n_ctx = 512` (8,192 at run time).
   - The CPU wheel's sha also differs from the T4 wheel's.
   - Either way the keys differ, so verification fails loudly (it raises; it doesn't pass silently).
   - Add a `key_source_from_header(hdr, vocab_llm)` helper that passes the header's digest, wheel sha, load parameters and decoding path, and asserts that the stamps are equal. With the header's load parameters, keys matched in my probe.
2. **From-scratch repair overwrites the old result without archiving it**, although the spec says old files are archived.
   - A from-scratch repair whose payload differs in the same epoch is evidence of nondeterminism.
   - Log it in the ledger and count it like a resume mismatch.
3. **Concurrency guards are per runtime and per generation, not per unit.**
   - Same-runtime cases are not caught:
     - a duplicate `--shard` spec;
     - a change of K while an old worker is still running;
     - worker subprocesses that survive an interrupted cell and get launched again.
   - Add a per-unit `O_EXCL` claim file with runtime, pid and a heartbeat mtime.
4. **GPU-hours accounting.**
   - `/proc/stat` btime can be earlier than the runtime assignment on pre-warmed VMs, which overcounts.
   - Use runtime id = hostname + btime, so kernel restarts aren't double-counted.
   - Start the heartbeat at session setup, not inside `run_phase` (canary, benchmark and tail time are missed).
   - `seconds_per_fresh_call` divides LLM seconds by LLM plus embedding calls, so it comes out too low.
5. **D130 error signatures include unit-specific numbers** (e.g. context-overflow sizes), so "same error" never matches. Strip digits before comparing.
6. **The canary prompts' sha is not in the unit header.**
   - If `canary_prompts.json` changes after epoch 0, an identical platform gets a new epoch and every interrupted unit restarts.
   - Recount with the real template, freeze the file, then register epoch 0.
7. **Embeddings across CPU types.**
   - Record the CPU model and torch's CPU capability in the header (recorded, not compatibility-checked) and in each `resume` line.
   - Flag units that resumed across CPU types: their embeddings mix two CPUs, and the resume check skips embeddings.
8. **Spec drift:**
   - The progress line has no GPU-hours vs the 60 h gate.
   - Its ETA comes from the session's pace, not the ledger.

## What S3.3b must verify before the pilot

1. **Template.**
   - The real GGUF template's sha equals S3.0's `d5495a1e…`.
   - Commit the real text through the repo, not by writing into the Colab checkout. A dirty tree never resumes, and the test phase refuses it.
   - Recount and freeze the canary prompts.
2. **Equality with S3.0.** The rendered path reproduces S3.0's stored outputs token for token (free and grammar), with equal `n_in`/`n_out`, and the t_llm constants are unchanged.
3. **D47 on the GPU.** After a longer unrelated call and after a grammar call, the reset path equals a brand-new `Llama` (bitwise) on the longest canary prompts.
4. **Load parameters.** Record `flash_attn_type`, `n_ubatch` and `type_k`/`type_v` on the T4 and confirm they match S3.0. Flash attention on "auto" vs "off" changes outputs.
5. **Canary.** The within-session repeat matches, and a second fresh runtime gives "pass".
6. **Interrupted = uninterrupted across runtimes** (kill mid-unit, resume in a new runtime). Compare LLM records separately from embeddings in case the CPU model differs.
7. **Two processes.**
   - VRAM fits.
   - Canary outputs in 2-process mode equal 1-process mode; if not, give it its own epoch path.
   - The lease and heartbeat behave with two workers.
8. **Snapshot path.**
   - It repeats exactly.
   - Host RAM with an LRU of 3: about 0.43 GB per state (311 MB of copied scores plus KV).
   - Measure the speedup.
   - On the GPU, the call's first eval starts at the prefix length.
9. **Drive.**
   - Append + fsync latency per record and cache size per unit.
   - A forced runtime recycle, to see what actually reaches Drive: tail loss, and the result arriving before the cache.
10. **Budget.** Compare btime with the first cell's time.
11. **Smoke units.**
    - `verify_replay` on CPU with the header-built key source.
    - The audit, including `note_events`, gives zero hits.
    - Record the `retok_diff` rate.
    - The control-token check never fires on real content.