# S3.3a specification: llama.cpp backend, replay cache, canaries, unit runner (CPU-testable code)

_r3, 2026-11-22 (r2 + changes found while building, marked [r3]). Implements design r7 sections 3 and 9 (LLM backend, embeddings, resume by replay, determinism and
resume canaries, platform epochs, GPU rule, integrity stamps, run order, budget ledger) and D53's two throughput levers
behind flags. r1 was reviewed by a fresh-context subagent (3 High, 9 Medium, 8 Low); the user decided D129; the
clear-cut findings are applied below. User decisions D125-D129 (section 12). The notebook and T4 smoke run are S3.3b._

## 1. Scope and testing (D125)

- **S3.3a (this spec, CPU):** `LlamaCppBackend`, `LlamaTokenizer`, `MiniLMEmbedder`, `CachedBackend` (strict replay
  of LLM and embedding calls), determinism canary with platform epochs, resume check, unit runner with resume and
  completeness checks, `verify_replay`, stamps, budget ledger, scrub audit, the D53 levers behind flags (off).
- **CPU tests use a CPU build of the pinned llama-cpp-python 0.3.36 sdist** (same Python-level code as the CUDA
  build: formatter, handler, `generate`, `reset`, `load_state`, tokenizer). Model files: the Qwen2 vocabulary GGUF
  shipped in the sdist (`vendor/llama.cpp/models/ggml-vocab-qwen2.gguf`) for tokenizer tests, and a tiny random-weight
  qwen2 GGUF written at test time with the sdist's bundled `gguf-py` (never committed) for generation tests. The
  Qwen2.5 chat template is a committed fixture; [r3] it is a **reconstruction** of the official template whose sha256
  does **not** equal S3.0's recorded `d5495a1e...` (the real text is not reachable from the sandbox), so CPU tests use
  it only to exercise the code paths; a Colab-marked test checks the real GGUF's template sha against S3.0's and the
  S3.3b notebook saves the real text to replace the fixture. eos = 151645 (`<|im_end|>`). The vocabulary GGUF and
  the tiny GGUF are not committed (`tools/fetch_test_models.py` builds them from the PyPI sdist). A fake `Llama` is used only for fault injection (crashes, zombie writers, mismatches).
  Tests that need the wheel are skipped with a reason when it is not installed; CI-style runs here have it.
- **S3.3b (next):** notebook `04_stage3.ipynb`: setup (S3.0's cached wheel and GGUF), T4 smoke run, Colab-marked
  tests, canary references (epoch 0 per path), scrub audit of real reflections, D53 benchmark section (D128), and
  the "interrupted = uninterrupted, resumed in a new Colab session" T4 test named in design section 9.

## 2. Prompt path (D126)

- The formatter is llama-cpp-python's own `Jinja2ChatFormatter`, built as `Llama.__init__` builds it from the GGUF
  (`tokenizer.chat_template`, eos/bos token texts from the model, `stop_token_ids=[eos]`).
- Per call, the chat handler's exact steps: render -> `ChatFormatterResponse`; tokens =
  `llm.tokenize(prompt.encode("utf-8"), add_bos=not added_special, special=True)`; stop = the formatter's stop;
  `stopping_criteria` = the formatter's; then `llm.create_completion(prompt=tokens, ...)`. (Review: 12/12 equal to
  `create_chat_completion` on a CPU build with a qwen2 GGUF, free and grammar.)
- **Sampling (greedy):** temperature 0.0, top_k 1, top_p 0.95, min_p 0.05, repeat_penalty 1.0, seed 0 (S3.0's
  values; at temperature 0 only greedy selection after the grammar matters). max_tokens per kind (action 320,
  reflection 60).
- **Grammars:** `envelope_cap300` = S3.0's envelope schema (thought `maxLength` 300); `envelope` = without the cap
  (D65). Built once per session with `LlamaGrammar.from_json_schema(json.dumps(schema))` (the handler's call; a
  failure raises, unlike the handler's silent fallback) and cached by id; the cache key uses the sha256 of the
  generated GBNF text.
- **Fresh KV cache (D47):** `llm.reset()` before every call (0.3.36: `n_tokens = 0` skips prefix matching and the
  first `eval` clears the whole KV cache, equal to a new context). Tests assert the call structure: the first `eval`
  of every call starts at position 0 with the full prompt (fresh) or at the snapshot length (section 9).
- **Context guard:** `len(tokens) + max_tokens + 4 <= n_ctx` (8,192): `n_out` can exceed max_tokens by up to 3 in
  0.3.36 (the multibyte check skips the length check). The same 4-token margin is applied in the S3.2 loop's guard.
- **Control tokens:** message contents must not contain the model's special-token strings (e.g. `<|im_end|>`,
  `<tool_call>`); with `special=True` they would become control tokens and could inject a turn boundary. The prompt
  builders neutralise them (`<|` -> `< |`, and the angle-bracket tool tags) in every content they assemble (prompt
  version bumped to p-v2), and the backend asserts that the number of control tokens in the prompt equals the
  number the template adds for that message list.
- **Result:** `Gen(text, n_in=len(tokens), n_out=usage.completion_tokens, finish_reason, grammar, flags)`; `n_in`
  equals `usage.prompt_tokens` (asserted). [r3] `flags` records `retok_diff:<d>` when the output text re-tokenizes to
  a different count than `n_out` (0.3.36's detokenize drops tokens over 32 bytes, which then shows as a deficit; the
  dropped token itself cannot be seen in the text). Both paths are affected equally, so equality with S3.0 holds.
  Wall seconds go to the run metadata only.
- **Concurrency guard:** one `Llama` behind a lock and a re-entry guard. The runner refuses to start a unit while the
  lock is held (a generation left running in a daemon thread by an interrupted cell), and every cache append carries
  the current writer token (section 4), so a zombie generation cannot write into a cache being replayed.

## 3. Tokenizer and embedder

- **`LlamaTokenizer`:** `count(text)` = tokens of `text` with `add_bos=False, special=False`; `count_messages` = the
  rendered prompt's token count (identical to the call); `truncate(text, n)` = the **longest character prefix** of
  `text` with `count <= n` (binary search; not detokenize, which drops tokens over 32 bytes). Review (real Qwen2
  vocabulary): a note line counted alone is equal to or one more than its share inside the prompt, so the 120-token
  caps are conservative and the context guard is exact. `id` = `qwen-gguf-<model digest[:12]>-llcpp0.3.36`.
- **`MiniLMEmbedder` (D127, D129):** sentence-transformers (pinned), `all-MiniLM-L6-v2` at a pinned revision,
  `device="cpu"`, `torch.set_num_threads(1)`, one text per call, `normalize_embeddings=True`; vectors rounded to
  1e-8. `id` = `minilm-l6-v2-<revision[:8]>-st<v>-torch<v>-tf<v>`.
- **Embeddings are per unit (D129):** every embedding is a record in the unit's own replay cache (kind `embed`,
  replayed strictly like an LLM call; section 4). No shared embedding cache, so a unit never depends on another
  unit or session (torch CPU output is not bitwise stable across versions, thread counts or CPU types). The unit's
  exported vectors (S3.2 preload) stay in the result.

## 4. Replay cache (`CachedBackend`)

- **One file per result unit** (design section 9): `cache/<unit_id>.jsonl` on Drive, append-only.
- **Header line** (line 0): the full unit stamp: code commit and dirty flag, spec sha, prompt/parser/sim/rq3/suite
  versions, LoopConfig parameters, condition and unit parameters, model digest (sha256 over the sorted sha256s of
  the GGUF parts), wheel sha256, llama-cpp-python version, template sha, load parameters (n_ctx, n_batch, n_ubatch,
  n_gpu_layers, flash_attn, type_k/type_v, offload_kqv), GPU name and driver, CUDA, Python, numpy, torch,
  sentence-transformers, embedder id, decoding path (fresh or snapshot) and snapshot procedure version, platform
  epoch, and the writer token of the run that created it.
- **Records:** `{"i", "kind" ("action"|"reflection"|"embed"), "key", "grammar", "max_tokens", "text", "n_in",
  "n_out", "finish_reason", "flags", "vector" (embed only), "prompt_z" (zlib + base64 of the rendered prompt or the
  embedded text), "writer", "seconds", "crc"}`. `crc` = first 16 hex of sha256 over the record without `crc`,
  serialised with `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`. `i` must equal the line
  index; `writer` must equal the header's or a later resume's registered token (resumes append a `resume` line with
  their new token).
- **Key** = sha256 of a canonical JSON of the parts of the header that affect outputs plus the call: kind, grammar
  GBNF sha, sampling parameters, max_tokens, stop strings, snapshot id, and the rendered prompt (or, for `embed`, the
  text). The scripted backend uses its own id and the canonical JSON of messages, kind, grammar and max_tokens, so
  the replay machinery is exercised end to end on CPU. `CachedBackend` forwards `register_tasks` only when the inner
  backend has it and composes its `id` from the inner one.
- **Strict replay:** call i must find record i with an equal key; a mismatch while records remain raises
  `ReplayDivergence`. Past the last record, calls go to the inner backend and are appended (write + flush + fsync,
  one line). At the end of a unit every record must have been consumed (leftovers = an undetected divergence).
- **Resume check (replaces r1's resume canary, design D61):** when a unit resumes from an interrupted cache, the
  replay of its last two records also calls the inner backend and compares the full `Gen` (text, n_in, n_out,
  finish_reason) or the vector; a difference raises `ResumeMismatch`.
- **Corruption:** a corrupt or torn **last** line is dropped (interrupted write); a corrupt line before it, an `i`
  gap or a foreign writer token is an error (archive, restart).
- **Archive, never delete:** `cache/archive/<unit_id>.<utc>.<reason>.jsonl`, listed in `cache/archive/index.jsonl`
  with unit, condition, reason and record count (reported per condition, design section 9).
- **Content hash** of a cache = sha256 over its records' deterministic fields (everything except `seconds`,
  `writer` and `crc`); stamped in the result.

## 5. Determinism canary and platform epochs (design section 9, D51, D56, D61, D66)

- **Reference prompts:** 8 fixed message lists built from dev stream 1000 with the real prompt builders: free action
  calls, grammar calls (`envelope_cap300`, and `envelope` once), reflection calls; [r3] about 2,200 to 6,100 tokens
  for the action calls (the longest recorded action prompt is about 2,200; longer ones extend its trajectory with the
  stream's own steps under the scripted policy, never filler) and about 150 for the reflections; counted with the
  Qwen2 vocabulary and the reconstructed template, recounted on the real GGUF in S3.3b. Committed as
  `configs/canary_prompts.json` (sha stamped).
- **Epochs are per decoding path** (fresh, snapshot): adopting the snapshot path after the benchmark starts its own
  epoch series and is not mistaken for a platform change.
- **Session canary:** the 8 prompts are generated on the adopted path, the two longest twice. Full `Gen`s are
  compared. If the within-session repeat differs, **no unit runs** and the user is alerted (a nondeterministic
  platform would otherwise start a new epoch every session). Else compared with every stored epoch of that path:
  equal to the current epoch -> pass; equal to an earlier epoch -> it is reused; no match -> a new epoch with these
  outputs as references. Interrupted units of another epoch **restart** with their cache archived (design). Epochs
  are registered only by the parent process under an exclusive lock file (`O_EXCL`), with GPU, driver, CUDA,
  llama-cpp version, wheel sha, Python and numpy.
- **Resume mismatch (D61):** the unit's cache is archived and the unit restarts. A second `ResumeMismatch` in the same
  session re-runs the session canary: if it fails, the failed-canary rule applies; if it passes, the session enters
  **long-prompt nondeterminism** mode (D66): the unit that triggered it restarts, later interrupted units in the
  session resume without the check and are flagged `replay_only`.
- **Two processes:** each worker runs the session canary in its process while the other process is generating
  (design), and reports to the parent; only the parent writes epochs.

## 6. Unit runner and resume

- **Unit ids** are deterministic, readable and include every parameter that distinguishes units: phase, RQ, stream
  kind and seed, condition, window, and dev parameters (EEM config id, docs-wording variant), e.g.
  `test-rq1-s2003-C2`, `dev-rq1-s1001-C2-cfg3`, `test-rq3-s4001-C2-acc`.
- **Plan** (`plans/<phase>.json`, written once by the parent before any worker starts): plan version, frozen code
  commit (test phase), the unit list, its sha256 (stamped in results). Order: within each stream or seed a seeded
  shuffle of the conditions (seed = sha256 of phase, stream and plan version); test phase (D63): blocks of 2 RQ1 test
  streams (C0-C3, plus C2-OU on the first 5 streams) then 1 RQ3 seed (the four 2x2 cells), repeated until RQ1 has 14
  streams and RQ3 10 seeds, remaining RQ3 seeds alone, then the descriptive arms and RQ3 extras. A plan changes only
  through a new plan version (budget cuts), never by re-sorting.
- **Run a unit:** refuse if the backend lock is held; check stamp compatibility (below); open the cache (resume check
  if it has records); run `run_rq1_unit` / `run_rq3_unit` with `CachedBackend` around the LLM backend and the
  embedder; separate the result into the **deterministic payload** and `result["run"]` (wall seconds, session ids,
  writer tokens, `replay_only`, restarts, cache content hash, ledger refs); write atomically (tmp + fsync + rename) to
  `units/<phase>/<unit_id>.json` with the payload's sha256; then append to the process's manifest
  `manifest/<phase>/<shard>.jsonl`.
- **Done** = result present, its payload hash valid, its recorded number of cache records equal to the cache's
  record count, and the cache's content hash equal to the stamped one. A result whose cache is short or missing (Drive
  durability: `fsync` reaches only the VM's local cache, and files reach the cloud in no fixed order) is repaired by
  resuming the unit from what is left in the same epoch and requiring a byte-identical payload; otherwise the unit
  re-runs from scratch (old files archived).
- **Stamp compatibility:** a unit resumes only if every header field marked output-affecting (section 4) equals the
  current session's, including the code commit; otherwise the cache is archived and the unit restarts. In the test
  phase no unit runs when the code commit differs from the plan's frozen commit.
- **Failures:** a `ReplayDivergence` or exception archives the cache with the reason and restarts the unit once; a
  second divergence of the same unit halts the run with diagnostics; two identical exceptions mark the unit `failed`
  (reported, never silently skipped).
- **`verify_replay(unit)`** (CPU): replays a finished unit with inner backends that raise (no generation, no
  embedding), using a vocab-only `Llama` on the GGUF for the tokenizer and formatter; the payload must be
  byte-identical. Required for every unit before analysis.
- **Lease:** a lease file with a heartbeat (`lease.json`, refreshed every 60 s); a second Colab runtime on the same
  Drive folder is refused while the lease is fresh (Drive allows duplicate file names, so two runtimes would write
  silent duplicates).
- **Progress and blinding (D64):** a progress line per unit (done/total, ETA from the ledger's seconds per fresh LLM
  call, GPU-hours used vs the 60 h gate); in the test phase it shows progress, budget, epoch and canary status only.
- **Sharding:** `--shard k/K` runs the plan's units with index mod K == k (whole units per process); per-process
  manifest and ledger files are merged on read.

## 7. Budget ledger (D52)

Per process `ledger/<session>-<shard>.jsonl`: session start (runtime id, model load seconds), heartbeats every 60 s,
canary seconds, per unit wall seconds and calls fresh vs replayed, and discarded calls by reason (epoch change,
resume mismatch, divergence, failure). **GPU-hours** = sum over runtimes of (last heartbeat - session start), counted
from the benchmark onwards (Colab charges runtime wall clock: setup, installs and idle time included; two processes
in one runtime are counted once).

## 8. Scrub audit

`audit_scrub(results)` scans every stored note, reflection, k_t and A1 text of a phase's results for the unit's task
entities and instruction values (the S3.2 extended entity set) and for placeholders left in arguments; report per
condition with every hit (unit, step). Run on real reflections in the S3.3b smoke run; zero hits expected.

## 9. Throughput levers (D53, flags, off by default; benchmark in S3.3b, D128)

- **Static-prefix snapshot:** per session and prefix, the prefix tokens = the rendered prompt up to and including the
  system turn's end token; the KV state is computed once (`reset`, evaluate the prefix) and saved; before **every**
  call it is loaded, and the call asserts its tokens start with the prefix tokens. Prefixes: the agent system prompt
  per docs variant; R gets one per drift state. The reflection prompt has no system turn of its own (the template adds
  Qwen's default), so it uses the fresh path. The snapshot store is bounded (LRU, default 3 states; `save_state`
  copies about 0.3 GB of logits plus the KV cache). Snapshot id = sha256 of the prefix tokens + procedure version; it
  enters the key and the stamps. t_llm still charges the full n_in.
- **Two processes:** sharding (section 6); the benchmark launches two workers from the notebook.

## 10. Tests (pytest)

- **CPU with the wheel:** rendered path == `create_chat_completion` on the tiny GGUF (free and grammar); call
  structure via a wrapped `Llama.eval` (first eval at position 0 with the full prompt; at the snapshot length on the
  snapshot path); exact `create_completion` arguments; grammar cache and GBNF sha; context guard with the 4-token
  margin; control-token assertion and neutralisation; re-entry guard and runner refusal while locked; `n_in ==
  prompt_tokens`; tokenizer `count`/`truncate` (longest prefix, a >32-byte token case) on the real Qwen2 vocabulary;
  template fixture sha.
- **CPU without the wheel:** cache format, crc canonical form, `i` and writer checks, torn last line, mid-file
  corruption, strict divergence, leftover records, resume check pass/mismatch, archive index; canary logic (pass,
  new epoch, return to an earlier epoch, repeat mismatch -> halt, per-path epochs, lock); D66 trigger unit restarts;
  runner: crash at every point (mid unit, after result before manifest, cache tail lost after result), done criteria,
  stamp mismatch (including code commit) -> restart, test-phase commit refusal, divergence/exception policy, lease,
  plan order and D63 interleaving, unit-id uniqueness; a scripted RQ1 and RQ3 unit through `CachedBackend` (with BoW
  embeddings as `embed` records) interrupted and resumed gives a byte-identical payload to an uninterrupted run, also
  in a subprocess with another `PYTHONHASHSEED`; `verify_replay`; ledger GPU-hours from heartbeats; scrub audit on
  a planted leak.
- **Colab-marked (S3.3b):** the real GGUF: formatter, `count_messages == n_in`, the 1,200-note fuzz of `note_render`
  caps with the real tokenizer, rendered path == `create_chat_completion` on S3.0-style prompts and S3.0's stored
  outputs; MiniLM repeatability (same text twice, two processes); snapshot path.

## 11. Out of scope

Running the pilot, freezing, dev selection, analysis (S3.4+).

## 12. Decisions (user, 2026-11-22)

| ID | Decision |
|---|---|
| D125 | S3.3 split: S3.3a CPU code (this spec), then S3.3b notebook with a T4 smoke run the user executes |
| D126 | Prompts rendered with the GGUF chat template through llama-cpp-python's own formatter, tokenized with `llm.tokenize`, sent to `create_completion`; exact counts and cache key; equality with the chat path checked |
| D127 | all-MiniLM-L6-v2 through sentence-transformers on CPU (pinned version and revision) |
| D128 | The D53 throughput benchmark is a section of the S3.3b notebook; the levers are built in S3.3a behind flags, off by default |
| D129 | Embeddings are recorded per unit in the unit's replay cache and replayed strictly; no shared embedding cache; single thread, one text per call |
