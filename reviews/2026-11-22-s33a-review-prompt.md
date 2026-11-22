# Review request: S3.3a (llama.cpp backend, per-unit replay cache, canaries, unit runner), 2026-11-22

You are a strict reviewer of reproducible ML experiment infrastructure. Be concrete; no praise.

## Context
Thesis: LLM agents (Qwen2.5-7B-Instruct q4_k_m GGUF via llama-cpp-python 0.3.36, greedy, on a free Colab T4, ~60
GPU-hour budget) solve multi-call API tasks under schema drift; conditions compare a reflection memory (EEM) and an
adaptive concurrency window (RQ1, RQ3). Agent time is virtual (LLM time modelled from token counts). Every "result
unit" (one condition on one task stream) must be exactly reproducible by replaying a per-unit cache of LLM outputs
and embeddings; Colab sessions are interrupted often and Google Drive (FUSE) is not durable.

S3.3a builds the T4 machinery as CPU-testable code. The spec (below, r4) was reviewed by a fresh-context reviewer (3
High, 9 Medium, 8 Low at r1), the code by a second (3 High, 6 Medium, Lows), all applied with the user's decisions
D125-D130. 83 S3.3a tests pass (838 in the repo); the llama tests run on the real 0.3.36 CPU wheel with a tiny
random-weight qwen2 GGUF (real Qwen2 vocabulary); 5 Colab-marked tests wait for the T4 (S3.3b notebook). Hugging Face
is unreachable from the build sandbox, so the Qwen2.5 chat template used in CPU tests is a reconstruction (its sha
does not match the one recorded in S3.0; a Colab test checks the real one).

## Checklist
1. Reproducibility: can any path produce a result that is not reproducible by replay, or block a resume forever?
   (strict replay, resume check, stamp compatibility, platform epochs + fingerprint, repair of lost cache tails, D66)
2. Does the rendered-prompt path really equal `create_chat_completion` token for token (formatter, tokenize, stop,
   stopping criteria, sampling), and is the fresh-KV guarantee (D47) real? The snapshot lever?
3. Drive durability and concurrency (two processes, lease, per-process manifests/ledgers, epochs lock).
4. Failure policy (divergence, exceptions, D130, busy backend) and budget accounting (GPU-hours from VM boot).
5. Embeddings per unit (D129) and their exclusion from the resume check.
6. What S3.3b (the notebook and T4 smoke run) must verify before the pilot.

## Output
Findings ranked by severity with concrete fixes; then a verdict on whether S3.3b can build on this.

## The spec (docs/s33_spec.md, r4)

# S3.3a specification: llama.cpp backend, replay cache, canaries, unit runner (CPU-testable code)

_r4, 2026-11-22 (r3 + the code review, section 13, and D130; r3 = changes found while building, marked [r3]). Implements design r7 sections 3 and 9 (LLM backend, embeddings, resume by replay, determinism and
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
| D130 | Two units in a row failing with the same error signature halt the run (something systemic); at most 4 attempts per unit and session |

## 13. Code review (2026-11-22): clear-cut changes

3 High, 6 Medium, Lows; all applied except as noted. A busy backend halts `run_phase` before any unit is touched
(it was failing every remaining unit). The resume check regenerates the last two **LLM** records; embedding records
are replayed and never compared (CPU torch is not bitwise stable). Files are split on `\n` only (`splitlines` broke
records containing U+2028/U+2029/U+0085). A session "returned" to its own epoch counts as a canary pass for D66.
The control-token check counts control-token ids (`LLAMA_TOKEN_ATTR_CONTROL`) in the rendered tokens against the
bare template's, not a text pattern. A repair must reproduce the stored payload only when it resumed in the same
stamp; a from-scratch repair is accepted and noted; the plan sha moved to the run metadata. A heartbeat thread
writes the ledger heartbeat and refreshes the lease every 60 s while a phase runs; GPU-hours start at the VM boot
time. The header carries a platform fingerprint (sha256 of the canonical canary outputs) next to the epoch number.
Code identity fails closed (git errors -> dirty), a dirty tree never resumes and the test phase refuses it. Load
parameters are read from the `Llama` object (full offload normalised to "all"). The done check compares the cache
size at session start and the full content hash in `verify_replay`, which also checks the cache's content hash.
Writers must carry the most recent registered token. Stamp archives are ledger "discarded" events. `truncate` is
documented as a prefix within the cap (BPE counts are not monotonic, so not always the longest). The Colab snapshot
test checks repeatability, not equality with the fresh path. The test sdist is pinned by sha256. Not changed (stated):
canary long prompts append the stream's steps from its start, so they are real, deterministic text but not one
coherent trajectory; `retok_diff` also fires on non-canonical tokenisation (a flag only); the package must run from a
git checkout (`code_identity`, `configs/`); the discarded repair run leaves a "done" line in the manifest, which is
not authoritative (`is_done` is).


## Code: src/eem_acc/stage3/llm/cache.py

```python
"""Per-unit replay cache of LLM and embedding calls, strict replay, resume check, archive (S3.3a spec section 4)."""
from __future__ import annotations

import base64
import hashlib
import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..agent.backend import Gen
from .fsio import append_line, atomic_write, canon

CACHE_FORMAT = "s33-cache-v1"
DETERMINISTIC = ("i", "kind", "key", "grammar", "max_tokens", "text", "n_in", "n_out", "finish_reason", "flags",
                 "vector", "prompt_z")
COMPARED = ("text", "n_in", "n_out", "finish_reason")   # resume check: the full Gen of LLM records only; embeddings
#                                                        are replayed, never compared (CPU torch is not bitwise stable)


class CacheError(Exception):
    """A cache that cannot be trusted (mid-file corruption, index gap, foreign writer): archive and restart."""


class ReplayDivergence(Exception):
    def __init__(self, i: int, expected: dict, got_kind: str, got_key: str):
        super().__init__(f"replay diverged at call {i}: cached {expected.get('kind')} {expected.get('key', '')[:12]}, "
                         f"now {got_kind} {got_key[:12]}")
        self.i = i


class ResumeMismatch(Exception):
    def __init__(self, i: int, field: str):
        super().__init__(f"resume check failed at call {i} ({field} differs)")
        self.i, self.field = i, field


class LeftoverRecords(Exception):
    """The unit ended before consuming every cached record: an undetected divergence."""


def crc(rec: dict) -> str:
    return hashlib.sha256(canon({k: v for k, v in rec.items() if k != "crc"}).encode()).hexdigest()[:16]


def zpack(text: str) -> str:
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 6)).decode()


def zunpack(blob: str) -> str:
    return zlib.decompress(base64.b64decode(blob)).decode("utf-8")


def key_of(parts: dict) -> str:
    return hashlib.sha256(canon(parts).encode()).hexdigest()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class UnitCache:
    """One append-only JSONL per unit: a header line, records `{"i", ...}`, and `resume` lines registering the writer
    token of every run that appended to it."""

    def __init__(self, path: Path, header: dict, writer: str, resume_check: bool = True, readonly: bool = False):
        """`readonly`: replay only (verify_replay): no resume line, no appends."""
        self.path, self.writer, self.resume_check, self.readonly = Path(path), writer, resume_check, readonly
        self.records: list[dict] = []
        self.writers: set[str] = set()
        self.pos = 0
        if readonly:
            self._load(rewrite=False)
            self.resuming = False
            return
        if self.path.exists():
            self._load()
            self.resuming = bool(self.records)
            llm_idx = [r["i"] for r in self.records if r["kind"] != "embed"]
            self.check_idx = set(llm_idx[-2:])          # the last two LLM calls are regenerated on resume (D61)
            append_line(self.path, canon({"type": "resume", "writer": writer, "at": utc_stamp()}))
            self.writers.add(writer)
        else:
            self.header = {"type": "header", "format": CACHE_FORMAT, **header, "writer": writer}
            append_line(self.path, canon(self.header))
            self.writers.add(writer)
            self.resuming = False
        self.check_idx = getattr(self, "check_idx", set())

    # ------------------------------------------------------------------------------------------------------- load
    def _load(self, rewrite: bool = True) -> None:
        raw = self.path.read_text(encoding="utf-8")
        lines = raw.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        parsed: list[dict | None] = []
        for ln in lines:
            try:
                parsed.append(json.loads(ln))
            except json.JSONDecodeError:
                parsed.append(None)
        torn = bool(parsed) and (parsed[-1] is None or not raw.endswith("\n")
                                 or (parsed[-1].get("type") is None and parsed[-1].get("crc") != crc(parsed[-1])))
        if torn:                                         # an interrupted last write: drop it (rewrite without it)
            parsed.pop()
            lines.pop()
            if rewrite:
                atomic_write(self.path, "".join(ln + "\n" for ln in lines))
        if not parsed or not isinstance(parsed[0], dict) or parsed[0].get("type") != "header":
            raise CacheError("missing header")
        self.header = parsed[0]
        if self.header.get("format") != CACHE_FORMAT:
            raise CacheError(f"cache format {self.header.get('format')}")
        self.writers = {self.header["writer"]}
        current = self.header["writer"]                  # a record must carry the most recent registered token
        for j, obj in enumerate(parsed[1:], start=1):
            if obj is None:
                raise CacheError(f"corrupt line {j}")
            if obj.get("type") == "resume":
                self.writers.add(obj["writer"])
                current = obj["writer"]
                continue
            if obj.get("crc") != crc(obj):
                raise CacheError(f"crc mismatch at line {j}")
            if obj["i"] != len(self.records):
                raise CacheError(f"index gap at line {j}: i={obj['i']}, expected {len(self.records)}")
            if obj.get("writer") != current:
                raise CacheError(f"foreign writer at line {j}")
            self.records.append(obj)

    # -------------------------------------------------------------------------------------------------- replay
    def call(self, kind: str, key: str, compute: Callable[[], dict], extra: dict) -> dict:
        """Record i if cached (strict), else compute, append and return. `compute` returns the result fields
        (text/n_in/n_out/finish_reason/flags or vector); `extra` = grammar, max_tokens, prompt_z."""
        i = self.pos
        if i < len(self.records):
            rec = self.records[i]
            if rec["kind"] != kind or rec["key"] != key:
                raise ReplayDivergence(i, rec, kind, key)
            if self.resuming and self.resume_check and i in self.check_idx:
                fresh = compute()
                for f in COMPARED:
                    if f in fresh and fresh[f] != rec.get(f):
                        raise ResumeMismatch(i, f)
            self.pos += 1
            return rec
        if self.readonly:
            raise ReplayDivergence(i, {"kind": "end of cache", "key": ""}, kind, key)
        t0 = time.perf_counter()
        out = compute()
        rec = {"i": i, "kind": kind, "key": key, "grammar": None, "max_tokens": None, "text": None, "n_in": None,
               "n_out": None, "finish_reason": None, "flags": [], "vector": None, "prompt_z": None, **extra, **out,
               "writer": self.writer, "seconds": round(time.perf_counter() - t0, 4)}
        rec["crc"] = crc(rec)
        append_line(self.path, canon(rec))
        self.records.append(rec)
        self.pos += 1
        return rec

    def finish(self) -> None:
        if self.pos != len(self.records):
            raise LeftoverRecords(f"{len(self.records) - self.pos} cached records were not replayed")

    def n_records(self) -> int:
        return len(self.records)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for r in self.records:
            h.update(canon({k: r.get(k) for k in DETERMINISTIC}).encode())
        return h.hexdigest()

    def fresh_calls(self) -> int:
        return sum(1 for r in self.records if r["writer"] == self.writer)


def read_header(path: Path) -> dict | None:
    """The header of a cache file (None if missing or unreadable), without registering a writer."""
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.loads(fh.readline())
    except (OSError, json.JSONDecodeError):
        return None
    return obj if obj.get("type") == "header" else None


def content_hash_of(path: Path) -> tuple[int, str]:
    """(records, content hash) of a cache file without opening it for writing."""
    recs = []
    for ln in Path(path).read_text(encoding="utf-8").split("\n"):    # not splitlines(): U+2028 etc. are content
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            break
        if obj.get("type") is None and obj.get("crc") == crc(obj):
            recs.append(obj)
    h = hashlib.sha256()
    for r in recs:
        h.update(canon({k: r.get(k) for k in DETERMINISTIC}).encode())
    return len(recs), h.hexdigest()


def archive(path: Path, reason: str, meta: dict) -> Path | None:
    """Rename a cache into cache/archive/ (never delete) and index it."""
    path = Path(path)
    if not path.exists():
        return None
    adir = path.parent / "archive"
    adir.mkdir(parents=True, exist_ok=True)
    dest = adir / f"{path.stem}.{utc_stamp()}.{reason}.jsonl"
    n, _ = content_hash_of(path)
    path.replace(dest)
    append_line(adir / "index.jsonl", canon({"unit": path.stem, "reason": reason, "records": n, "file": dest.name,
                                             "at": utc_stamp(), **meta}))
    return dest


# -------------------------------------------------------------------------------------------------------- wrappers
def default_key_parts(backend: Any, messages: list[dict], kind: str, grammar: str | None, max_tokens: int):
    """Key parts and the prompt text for backends without `cache_key` (scripted): canonical JSON of the call."""
    prompt = canon(messages)
    return {"backend": getattr(backend, "id", type(backend).__name__), "kind": kind, "grammar": grammar,
            "max_tokens": max_tokens, "prompt": prompt}, prompt


class CachedBackend:
    """LLM backend wrapper: every call goes through the unit's cache (strict replay)."""

    def __init__(self, inner: Any, cache: UnitCache):
        self.inner, self.cache = inner, cache
        self.id = f"cached({getattr(inner, 'id', type(inner).__name__)})"
        if hasattr(inner, "register_tasks"):
            self.register_tasks = inner.register_tasks

    def generate(self, messages: list[dict], kind: str, grammar: str | None, max_tokens: int) -> Gen:
        if hasattr(self.inner, "cache_key"):
            parts, prompt = self.inner.cache_key(messages, kind, grammar, max_tokens)
        else:
            parts, prompt = default_key_parts(self.inner, messages, kind, grammar, max_tokens)
        key = key_of(parts)

        def compute() -> dict:
            g = self.inner.generate(messages, kind, grammar, max_tokens)
            return {"text": g.text, "n_in": g.n_in, "n_out": g.n_out, "finish_reason": g.finish_reason,
                    "flags": list(getattr(g, "flags", ()) or ())}
        rec = self.cache.call(kind, key, compute, {"grammar": grammar, "max_tokens": max_tokens,
                                                   "prompt_z": zpack(prompt)})
        return Gen(rec["text"], rec["n_in"], rec["n_out"], rec["finish_reason"], rec["grammar"],
                   tuple(rec.get("flags") or ()))


class CachedEmbedder:
    """Embedder wrapper (D129): the first embedding of each text in a unit is a cache record; repeats in the same unit
    are served from memory (deterministic: the call sequence is)."""

    def __init__(self, inner: Any, cache: UnitCache):
        self.inner, self.cache, self.id = inner, cache, inner.id
        self.memo: dict[str, tuple] = {}

    def embed(self, text: str) -> tuple:
        if text in self.memo:
            return self.memo[text]
        key = key_of({"kind": "embed", "embedder": self.id, "text": text})

        def compute() -> dict:
            return {"vector": [round(float(x), 8) for x in self.inner.embed(text)]}
        rec = self.cache.call("embed", key, compute, {"prompt_z": zpack(text)})
        v = tuple(rec["vector"])
        self.memo[text] = v
        return v
```

## Code: src/eem_acc/stage3/llm/llama_backend.py

```python
"""llama.cpp backend on the T4: rendered GGUF chat template -> tokens -> create_completion, fresh KV cache per call,
grammar cache, guards; the Qwen tokenizer; the static-prefix snapshot lever (S3.3a spec sections 2, 3, 9).

`llm` is a `llama_cpp.Llama` (llama-cpp-python 0.3.36). Nothing here imports llama_cpp at module import time except
inside functions, so CPU code that only needs the cache or runner can import this module without the wheel."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from ..agent.backend import CTX_MARGIN, Gen
from ..feasibility import ENVELOPE_SCHEMA
from .fsio import canon

LLAMA_CPP_VERSION = "0.3.36"
SAMPLING = {"temperature": 0.0, "top_k": 1, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.0, "seed": 0}
SCHEMAS = {
    "envelope_cap300": ENVELOPE_SCHEMA,
    "envelope": {**ENVELOPE_SCHEMA, "properties": {**ENVELOPE_SCHEMA["properties"], "thought": {"type": "string"}}},
}
SNAPSHOT_PROCEDURE = "snap-v1"
TOKEN_ATTR_CONTROL = 1 << 3          # llama.h LLAMA_TOKEN_ATTR_CONTROL: only control tokens can open a turn boundary


class BackendBusy(RuntimeError):
    """A generation is already running (e.g. left in a daemon thread by an interrupted Colab cell)."""


def make_formatter(llm: Any):
    """The formatter exactly as `Llama.__init__` builds it from the GGUF metadata (llama-cpp-python 0.3.36)."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    template = llm.metadata["tokenizer.chat_template"]
    eos_id, bos_id = llm.token_eos(), llm.token_bos()
    eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
    return Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos, stop_token_ids=[eos_id]), template


class LlamaCppBackend:
    def __init__(self, llm: Any, model_digest: str, wheel_sha256: str, load_params: dict | None = None,
                 snapshot: bool = False, snapshot_lru: int = 3):
        self.llm = llm
        self.formatter, template = make_formatter(llm)
        self.template_sha256 = hashlib.sha256(template.encode()).hexdigest()
        self.model_digest, self.wheel_sha256 = model_digest, wheel_sha256
        self.load_params = {**load_params_of(llm), **(load_params or {})}
        vocab = llm._model
        self.control_ids = frozenset(t for t in range(vocab.n_vocab()) if vocab.token_get_attr(t) & TOKEN_ATTR_CONTROL)
        self.n_ctx = llm.n_ctx()
        self.snapshot = snapshot
        self.snapshot_lru = snapshot_lru
        self._states: OrderedDict[str, tuple[list[int], Any]] = OrderedDict()
        self._grammars: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.id = f"llama-{model_digest[:12]}-{'snap' if snapshot else 'fresh'}"

    # ---------------------------------------------------------------------------------------------- identity
    def stamp(self) -> dict:
        """Output-affecting identity of this backend (cache header and result stamps)."""
        return {"model_digest": self.model_digest, "wheel_sha256": self.wheel_sha256,
                "llama_cpp_version": LLAMA_CPP_VERSION, "template_sha256": self.template_sha256,
                "load_params": self.load_params, "sampling": SAMPLING,
                "decoding_path": "snapshot" if self.snapshot else "fresh",
                "snapshot_procedure": SNAPSHOT_PROCEDURE if self.snapshot else None}

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # ----------------------------------------------------------------------------------------------- render
    def render(self, messages: list[dict]):
        """The chat handler's steps: (prompt text, tokens, stop, stopping criteria)."""
        r = self.formatter(messages=messages)
        tokens = self.llm.tokenize(r.prompt.encode("utf-8"), add_bos=not r.added_special, special=True)
        stop = [] if r.stop is None else (r.stop if isinstance(r.stop, list) else [r.stop])
        return r.prompt, tokens, stop, r.stopping_criteria

    def grammar(self, gid: str | None):
        if gid is None:
            return None, None
        if gid not in self._grammars:
            from llama_cpp.llama_grammar import LlamaGrammar
            g = LlamaGrammar.from_json_schema(json.dumps(SCHEMAS[gid]), verbose=False)   # raises on failure
            self._grammars[gid] = (g, hashlib.sha256(g._grammar.encode()).hexdigest())
        return self._grammars[gid]

    def cache_key(self, messages: list[dict], kind: str, grammar: str | None, max_tokens: int):
        prompt, _, stop, _ = self.render(messages)
        _, gsha = self.grammar(grammar)
        snap = self._snapshot_id(messages) if self.snapshot else None
        parts = {**self.stamp(), "kind": kind, "grammar": grammar, "grammar_gbnf_sha256": gsha,
                 "max_tokens": max_tokens, "stop": stop, "snapshot_id": snap, "prompt": prompt}
        return parts, prompt

    def check_control_tokens(self, messages: list[dict], tokens: list[int]) -> None:
        """Content must not add control tokens: the rendered prompt has exactly as many control-token ids as the
        template renders for the same message list with empty contents."""
        bare = self.render([{"role": m["role"], "content": ""} for m in messages])[1]
        n = sum(t in self.control_ids for t in tokens)
        n_bare = sum(t in self.control_ids for t in bare)
        if n != n_bare:
            raise ValueError(f"message content adds {n - n_bare} special (control) tokens")

    # ------------------------------------------------------------------------------------------- snapshot lever
    def _prefix(self, messages: list[dict], tokens: list[int] | None = None) -> list[int] | None:
        """Tokens up to and including the end token of the system turn (None: no own system turn -> fresh path)."""
        if not messages or messages[0]["role"] != "system":
            return None
        if tokens is None:
            tokens = self.render(messages)[1]
        eos = self.llm.token_eos()
        return tokens[:tokens.index(eos) + 1] if eos in tokens else None

    def _snapshot_id(self, messages: list[dict]) -> str | None:
        p = self._prefix(messages)
        return None if p is None else hashlib.sha256(canon([SNAPSHOT_PROCEDURE, p]).encode()).hexdigest()

    def _load_prefix_state(self, prefix: list[int]) -> None:
        sid = hashlib.sha256(canon([SNAPSHOT_PROCEDURE, prefix]).encode()).hexdigest()
        if sid not in self._states:
            self.llm.reset()
            self.llm.eval(prefix)
            self._states[sid] = (list(prefix), self.llm.save_state())
            while len(self._states) > self.snapshot_lru:
                self._states.popitem(last=False)
        self._states.move_to_end(sid)
        self.llm.load_state(self._states[sid][1])          # before EVERY call (otherwise prefix reuse of the last call)

    # ------------------------------------------------------------------------------------------------ generate
    def generate(self, messages: list[dict], kind: str, grammar: str | None, max_tokens: int) -> Gen:
        if not self._lock.acquire(blocking=False):
            raise BackendBusy("a generation is already running")
        try:
            prompt, tokens, stop, criteria = self.render(messages)
            self.check_control_tokens(messages, tokens)
            if len(tokens) + max_tokens + CTX_MARGIN > self.n_ctx:
                raise ValueError(f"context overflow: {len(tokens)} + {max_tokens} + {CTX_MARGIN} > {self.n_ctx}")
            g, _ = self.grammar(grammar)
            prefix = self._prefix(messages, tokens) if self.snapshot else None
            if prefix is not None:
                self._load_prefix_state(prefix)              # state keyed by the prefix tokens themselves
            else:
                self.llm.reset()                            # D47: fresh KV cache, no prefix reuse
            out = self.llm.create_completion(prompt=tokens, max_tokens=max_tokens, stop=stop,
                                             stopping_criteria=criteria, grammar=g, **SAMPLING)
            usage, choice = out["usage"], out["choices"][0]
            if usage["prompt_tokens"] != len(tokens):
                raise AssertionError(f"n_in {len(tokens)} != usage.prompt_tokens {usage['prompt_tokens']}")
            text = choice["text"]
            n_out = int(usage["completion_tokens"])
            diff = n_out - len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
            flags = (f"retok_diff:{diff}",) if diff else ()
            return Gen(text, len(tokens), n_out, choice.get("finish_reason") or "stop", grammar, flags)
        finally:
            self._lock.release()


def load_params_of(llm: Any) -> dict:
    """Output-affecting load parameters read from the Llama object itself (not from what the caller says); full GPU
    offload is normalised to "all" (n_gpu_layers -1 and 99 are the same offload)."""
    cp, mp = llm.context_params, llm.model_params
    out = {}
    for name in ("n_ctx", "n_batch", "n_ubatch", "type_k", "type_v", "offload_kqv", "flash_attn", "flash_attn_type",
                 "n_seq_max", "swa_full"):
        if hasattr(cp, name):
            v = getattr(cp, name)
            out[name] = v if isinstance(v, (int, float, bool, str)) else str(v)
    try:
        n_layer = int(llm.metadata.get(f"{llm.metadata.get('general.architecture', '')}.block_count", 0))
    except (TypeError, ValueError):
        n_layer = 0
    ngl = int(mp.n_gpu_layers)
    out["n_gpu_layers"] = "all" if (ngl < 0 or (n_layer and ngl >= n_layer)) else ngl
    return out


class LlamaTokenizer:
    """Counts with the model's own tokenizer; `count_messages` is the rendered prompt's length (S3.3a spec section 3)."""

    def __init__(self, backend: LlamaCppBackend):
        self.backend, self.llm = backend, backend.llm
        self.id = f"qwen-gguf-{backend.model_digest[:12]}-llcpp{LLAMA_CPP_VERSION}"

    def count(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)) if text else 0

    def count_messages(self, messages: list[dict]) -> int:
        return len(self.backend.render(messages)[1])

    def truncate(self, text: str, n: int) -> str:
        """A character prefix with count <= n found by binary search (not detokenize: 0.3.36 drops tokens over 32
        bytes). BPE counts are not monotonic in the prefix length, so it can be a few characters short of the longest
        such prefix; it never exceeds n."""
        if n <= 0:
            return ""
        if self.count(text) <= n:
            return text
        lo, hi = 0, len(text)                              # count(text[:lo]) <= n < count(text[:hi])
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.count(text[:mid]) <= n:
                lo = mid
            else:
                hi = mid
        return text[:lo].rstrip()
```

## Code: src/eem_acc/stage3/llm/canary.py

```python
"""Determinism canary and platform epochs (S3.3a spec section 5; design section 9: D51, D56, D61, D66).

The 8 reference prompts are real prompts of dev stream 1000 (scripted policy), committed in configs/canary_prompts.json.
Epochs are per decoding path (fresh, snapshot); only the parent process registers epochs, under an O_EXCL lock."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .fsio import atomic_write, canon

REPEATS_OF_LONGEST = 2
PROMPTS_FILE = Path(__file__).resolve().parents[4] / "configs" / "canary_prompts.json"


class CanaryRepeatMismatch(RuntimeError):
    """The same prompt gave different outputs within one session: no unit may run (alert the user)."""


class LockTimeout(RuntimeError):
    pass


def gen_record(g: Any) -> dict:
    return {"text": g.text, "n_in": g.n_in, "n_out": g.n_out, "finish_reason": g.finish_reason}


def fingerprint(outputs: list[dict]) -> str:
    """sha256 of the canonical reference outputs: identifies a platform even if epochs.json was lost on Drive (the
    epoch counter alone could be re-used by another platform)."""
    import hashlib
    return hashlib.sha256(canon(outputs).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------- reference prompts
def build_reference_prompts(tokenizer: Any, targets=(1500, 2500, 3500, 4500, 5200, 6000)) -> list[dict]:
    """8 calls from dev stream 1000 under the scripted policy: free action calls at increasing lengths, one
    `envelope_cap300` and one `envelope` grammar call, and two reflection calls. Long prompts extend a recorded action
    prompt's trajectory with the stream's own later steps (never filler)."""
    from ..agent import BagOfWordsEmbedder, ScriptedBackend, run_rq1_unit
    from ..apisuite import sample_stream
    calls: list[dict] = []

    class Rec(ScriptedBackend):
        def generate(self, messages, kind, grammar, max_tokens):
            g = super().generate(messages, kind, grammar, max_tokens)
            calls.append({"messages": messages, "kind": kind, "max_tokens": max_tokens})
            return g
    spec = sample_stream(1000, "dev")
    run_rq1_unit(spec, "C2", Rec(tokenizer, "memory"), BagOfWordsEmbedder(), tokenizer)
    actions = [c for c in calls if c["kind"] == "action"]
    refl = [c for c in calls if c["kind"] == "reflection"]
    pairs = []                                  # (assistant, observation) steps of the whole stream, in order
    for c in actions:
        m = c["messages"]
        if len(m) >= 4 and m[-2]["role"] == "assistant":
            pairs.append((m[-2], m[-1]))
    base = max(actions, key=lambda c: tokenizer.count_messages(c["messages"]))
    out = []
    for tgt in targets:
        msgs = [dict(x) for x in base["messages"]]
        j = 0
        while tokenizer.count_messages(msgs) < tgt and j < len(pairs):
            msgs.extend([dict(pairs[j][0]), dict(pairs[j][1])])
            j += 1
        out.append({"messages": msgs, "kind": "action", "grammar": None, "max_tokens": 320})
    out[1]["grammar"] = "envelope_cap300"
    out[3]["grammar"] = "envelope"
    out += [{"messages": refl[0]["messages"], "kind": "reflection", "grammar": None, "max_tokens": 60},
            {"messages": refl[-1]["messages"], "kind": "reflection", "grammar": None, "max_tokens": 60}]
    for p in out:
        p["n_tokens"] = tokenizer.count_messages(p["messages"])
    return out


def load_reference_prompts(path: Path = PROMPTS_FILE) -> tuple[list[dict], str]:
    import hashlib
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)["prompts"], hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------- lock
class FileLock:
    """O_EXCL lock file; a lock older than `stale` seconds is broken (a crashed holder)."""

    def __init__(self, path: Path, timeout: float = 400.0, stale: float = 300.0):
        self.path, self.timeout, self.stale = Path(path), timeout, stale

    def __enter__(self):
        t0 = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > self.timeout:
                    raise LockTimeout(str(self.path))
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# -------------------------------------------------------------------------------------------------------- epochs
class EpochStore:
    """canary/epochs.json: {"<path>": [{"epoch": n, "outputs": [...], "env": {...}, "prompts_sha256": ...}]}."""

    def __init__(self, root: Path):
        self.file = Path(root) / "canary" / "epochs.json"
        self.lock = FileLock(self.file.with_suffix(".lock"))

    def read(self) -> dict:
        return json.loads(self.file.read_text()) if self.file.exists() else {}

    def resolve(self, path: str, outputs: list[dict], env: dict, prompts_sha: str) -> tuple[int, str]:
        """(epoch, status) with status in pass | returned | new. Registers a new epoch when nothing matches."""
        with self.lock:
            data = self.read()
            series = data.setdefault(path, [])
            same = [e for e in series if e["prompts_sha256"] == prompts_sha and e["outputs"] == outputs]
            if same:
                ep = same[0]["epoch"]
                return ep, ("pass" if ep == series[-1]["epoch"] else "returned")
            ep = (series[-1]["epoch"] + 1) if series else 0
            series.append({"epoch": ep, "outputs": outputs, "fingerprint": fingerprint(outputs), "env": env,
                           "prompts_sha256": prompts_sha,
                           "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            atomic_write(self.file, json.dumps(data, indent=1, sort_keys=True))
            return ep, "new"


def run_canary(generate: Callable[[dict], Any], prompts: list[dict]) -> list[dict]:
    """Generate every reference prompt once and the longest REPEATS_OF_LONGEST prompts twice; a within-session repeat
    mismatch raises CanaryRepeatMismatch."""
    outs = [gen_record(generate(p)) for p in prompts]
    longest = sorted(range(len(prompts)), key=lambda i: -prompts[i].get("n_tokens", 0))[:REPEATS_OF_LONGEST]
    for i in longest:
        again = gen_record(generate(prompts[i]))
        if again != outs[i]:
            raise CanaryRepeatMismatch(f"reference prompt {i} differs within the session")
    return outs


# ---------------------------------------------------------------------------------------------- session state
@dataclass
class Session:
    """Per-session canary state: epoch on the adopted path, resume mismatches, D66 long-prompt nondeterminism mode."""
    path: str
    epoch: int
    status: str
    fingerprint: str = ""
    mismatches: int = 0
    replay_only_mode: bool = False
    events: list = field(default_factory=list)

    def on_resume_mismatch(self, rerun_canary: Callable[[], tuple[int, str, str]]) -> str:
        """D61/D66. Returns "restart" (the unit restarts; the trigger unit always does) or raises if the re-run canary
        fails (a platform change: the caller stops resuming)."""
        self.mismatches += 1
        self.events.append({"event": "resume_mismatch", "n": self.mismatches})
        if self.mismatches >= 2 and not self.replay_only_mode:
            epoch, status, fp = rerun_canary()
            if epoch != self.epoch:                      # "returned" to the session's own epoch is a pass (review M1)
                self.epoch, self.status, self.fingerprint = epoch, status, fp
                self.events.append({"event": "canary_failed_after_mismatch", "epoch": epoch, "status": status})
                raise PlatformChanged(epoch, status, fp)
            self.replay_only_mode = True
            self.events.append({"event": "long_prompt_nondeterminism"})
        return "restart"


class PlatformChanged(RuntimeError):
    def __init__(self, epoch: int, status: str, fingerprint: str = ""):
        super().__init__(f"canary changed epoch to {epoch} ({status})")
        self.epoch, self.status, self.fingerprint = epoch, status, fingerprint


def session_env() -> dict:
    """Environment recorded with an epoch (GPU via nvidia-smi when present)."""
    import platform
    import subprocess

    import numpy
    env = {"python": platform.python_version(), "numpy": numpy.__version__}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        env["gpu"], env["driver"] = [x.strip() for x in q.split(",")[:2]] if q else ("none", "none")
    except (OSError, subprocess.SubprocessError):
        env["gpu"], env["driver"] = "none", "none"
    for mod, key in (("llama_cpp", "llama_cpp"), ("torch", "torch"), ("sentence_transformers",
                                                                         "sentence_transformers")):
        try:
            env[key] = __import__(mod).__version__
        except Exception:      # noqa: BLE001
            env[key] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
    except Exception:          # noqa: BLE001
        env["cuda"] = None
    return env

```

## Code: src/eem_acc/stage3/llm/runner.py

```python
"""Unit runner with resume, completeness checks, failure policy, plans, lease and verify_replay (S3.3a spec section 6).

Drive layout under `root` (e.g. MyDrive/Skripsi/stage3):
  plans/<phase>.json            the phase's unit list (written once by the parent)
  cache/<unit_id>.jsonl         per-unit replay cache (cache.UnitCache); cache/archive/ for discarded caches
  units/<phase>/<unit_id>.json  results (deterministic payload + "run" metadata)
  manifest/<phase>/<shard>.jsonl, failures/<phase>.jsonl, ledger/, canary/, lease.json
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cache import (CacheError, CachedBackend, CachedEmbedder, LeftoverRecords, ReplayDivergence, ResumeMismatch,
                    UnitCache, archive, content_hash_of, read_header)
from .canary import PlatformChanged, Session
from .fsio import append_line, atomic_write, canon, read_jsonl
from .ledger import Ledger
from .llama_backend import BackendBusy

PLAN_FORMAT = "s33-plan-v1"
NOT_COMPAT = ("type", "format", "writer", "gpu_driver", "cuda", "created")   # recorded, but a resume may differ
MAX_ATTEMPTS = 4                     # per unit and session (D130)


class RunHalted(RuntimeError):
    """Stop the campaign with diagnostics: a unit diverged twice, or two units in a row failed with the same error
    signature (D130: something systemic, e.g. a broken CUDA context)."""


class RepairMismatch(Exception):
    """A resumed repair of a unit whose cache tail was lost gave a different payload than its stored result."""


class TestPhaseCommitMismatch(RuntimeError):
    pass


class LeaseHeld(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------ units
@dataclass(frozen=True)
class Unit:
    phase: str                  # pilot | dev | test | bench ...
    rq: str                     # rq1 | rq3
    split: str                  # dev | test
    kind: str                   # rq1 | sp | rq3
    seed: int
    condition: str
    window: str | None = None   # RQ3 only
    params: tuple = ()          # LoopConfig overrides as sorted (name, value) pairs
    tag: str = ""               # dev parameter label (EEM config id, docs-wording variant)
    n_tasks: int = 20

    @property
    def unit_id(self) -> str:
        parts = [self.phase, self.rq, f"{self.kind if self.kind != self.rq else ''}s{self.seed}", self.condition]
        if self.window:
            parts.append(self.window)
        if self.tag:
            parts.append(self.tag)
        if self.n_tasks != 20:
            parts.append(f"n{self.n_tasks}")
        return "-".join(parts)

    def to_json(self) -> dict:
        return {**dataclasses.asdict(self), "params": [list(p) for p in self.params]}

    @staticmethod
    def from_json(d: dict) -> "Unit":
        return Unit(**{**d, "params": tuple(tuple(p) for p in d.get("params", ()))})


def shuffled(conditions: list[str], phase: str, stream_key: str, plan_version: int) -> list[str]:
    """Seeded shuffle of the conditions within one stream or seed (seed = sha256 of phase, stream, plan version)."""
    seed = int(hashlib.sha256(f"{phase}|{stream_key}|{plan_version}".encode()).hexdigest()[:16], 16)
    out = list(conditions)
    random.Random(seed).shuffle(out)
    return out


def test_phase_units(rq1_seeds: list[int], rq3_seeds: list[int], plan_version: int,
                     rq1_conditions=("C0", "C1", "C2", "C3"), c2ou_streams: int = 5,
                     rq3_cells=(("C0", "acc"), ("C2", "acc"), ("C0", "none"), ("C2", "none")),
                     descriptive: list[Unit] = (), rq3_extra: list[Unit] = ()) -> list[Unit]:
    """D63: blocks of 2 RQ1 test streams (C0-C3, plus C2-OU on the first `c2ou_streams` streams) then 1 RQ3 seed (the
    2x2 cells), repeated; remaining RQ3 seeds alone; then the descriptive arms and the RQ3 extras."""
    out: list[Unit] = []
    r1, r3 = list(rq1_seeds), list(rq3_seeds)
    n_streams = 0
    while r1 or r3:
        for _ in range(2):
            if r1:
                s = r1.pop(0)
                conds = list(rq1_conditions) + (["C2-OU"] if n_streams < c2ou_streams else [])
                n_streams += 1
                for c in shuffled(conds, "test", f"rq1-{s}", plan_version):
                    out.append(Unit("test", "rq1", "test", "rq1", s, c))
        if r3:
            s = r3.pop(0)
            keys = [f"{c}-{w}" for c, w in rq3_cells]
            for k in shuffled(keys, "test", f"rq3-{s}", plan_version):
                c, w = k.rsplit("-", 1)
                out.append(Unit("test", "rq3", "test", "rq3", s, c, w))
    out += list(descriptive) + list(rq3_extra)
    return out


def write_plan(root: Path, phase: str, units: list[Unit], version: int, frozen_commit: str | None = None,
               blind: bool = False) -> dict:
    ids = [u.unit_id for u in units]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate unit ids in the plan")
    body = {"format": PLAN_FORMAT, "phase": phase, "version": version, "frozen_commit": frozen_commit,
            "blind": blind, "units": [u.to_json() for u in units]}
    body["sha256"] = hashlib.sha256(canon(body).encode()).hexdigest()
    path = Path(root) / "plans" / f"{phase}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["version"] == version and old["sha256"] != body["sha256"]:
            raise ValueError("a plan changes only through a new plan version")
        if old["version"] > version:
            raise ValueError("plan version must not go back")
    atomic_write(path, json.dumps(body, indent=1))
    return body


def read_plan(root: Path, phase: str) -> dict:
    return json.loads((Path(root) / "plans" / f"{phase}.json").read_text())


# ------------------------------------------------------------------------------------------------------------ lease
class Lease:
    """lease.json with a heartbeat: a second runtime on the same Drive folder is refused while the lease is fresh."""

    def __init__(self, root: Path, runtime: str, stale: float = 180.0, clock=time.time):
        self.file, self.runtime, self.stale, self.clock = Path(root) / "lease.json", runtime, stale, clock

    def acquire(self) -> None:
        if self.file.exists():
            cur = json.loads(self.file.read_text())
            if cur["runtime"] != self.runtime and self.clock() - cur["t"] < self.stale:
                raise LeaseHeld(f"runtime {cur['runtime']} holds the lease (heartbeat {self.clock() - cur['t']:.0f}s ago)")
        self.refresh()

    def refresh(self) -> None:
        atomic_write(self.file, json.dumps({"runtime": self.runtime, "t": self.clock()}))

    def release(self) -> None:
        if self.file.exists() and json.loads(self.file.read_text())["runtime"] == self.runtime:
            self.file.unlink()


class Heartbeat:
    """Daemon thread: ledger heartbeat and lease refresh every `seconds` while a phase runs (review M4)."""

    def __init__(self, ledger: Ledger, lease: Lease | None, seconds: float = 60.0):
        import threading
        self.ledger, self.lease, self.seconds = ledger, lease, seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.seconds):
            self.beat()

    def beat(self) -> None:
        self.ledger.heartbeat()
        if self.lease is not None:
            self.lease.refresh()

    def __enter__(self):
        if self.lease is not None:
            self.lease.acquire()
        self.beat()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self.beat()


# ---------------------------------------------------------------------------------------------------------- results
ENV_STAMP_FIELDS = ("code_commit", "python_version", "numpy_version")   # recorded; verify_replay may run elsewhere


def payload_sha(payload: dict) -> str:
    return hashlib.sha256(canon(payload).encode()).hexdigest()


def normalise(obj: Any) -> Any:
    """The JSON form of a result (tuples -> lists), so hashes are over exactly what is stored."""
    return json.loads(json.dumps(obj, default=str))


def code_identity() -> dict:
    import subprocess
    repo = Path(__file__).resolve().parents[4]
    try:
        a = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10)
        b = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True,
                           text=True, timeout=10)
        if a.returncode or b.returncode or not a.stdout.strip():
            raise OSError(a.stderr or b.stderr)
        commit, dirty = a.stdout.strip(), bool(b.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", True                    # fail closed: a dirty/unknown tree never resumes
    return {"code_commit": commit, "code_dirty": dirty}


def make_header(backend_stamp: dict, embedder_id: str, env: dict, epoch: int | None, fingerprint: str = "") -> dict:
    """The session's output-affecting header fields (the unit adds its own); see spec section 4."""
    from ..agent import prompts as P
    from ..agent.parser import PARSER_VERSION
    from ..apisuite import SUITE_VERSION
    from ..rq3 import RQ3_VERSION
    from ...sim import SIM_VERSION
    return {**code_identity(), **backend_stamp, "embedder": embedder_id, "prompt_version": P.PROMPT_VERSION,
            "prompt_sha256": P.prompt_hash(), "parser_version": PARSER_VERSION, "suite_version": SUITE_VERSION,
            "rq3_version": RQ3_VERSION, "sim_version": SIM_VERSION, "gpu": env.get("gpu"), "gpu_driver": env.get("driver"),
            "cuda": env.get("cuda"), "python": env.get("python"), "numpy": env.get("numpy"), "torch": env.get("torch"),
            "sentence_transformers": env.get("sentence_transformers"), "platform_epoch": epoch,
            "platform_fingerprint": fingerprint}


def split_result(result: dict) -> tuple[dict, dict]:
    payload = {k: v for k, v in result.items() if k != "run"}
    return payload, result.get("run", {})


@dataclass
class RunContext:
    """What a session provides: the LLM backend (or a scripted stand-in), the embedder, the tokenizer, the output-
    affecting header fields of this session (backend stamp, environment, code identity, epoch)."""
    backend: Any
    embedder: Any
    tokenizer: Any
    header: dict
    session: Session | None = None
    rerun_canary: Callable[[], tuple[int, str]] | None = None


@dataclass
class Runner:
    root: Path
    ctx: RunContext
    ledger: Ledger
    shard: tuple[int, int] = (0, 1)
    log: Callable[[str], None] = print
    halted_units: set = field(default_factory=set)
    lease: "Lease | None" = None
    heartbeat_seconds: float = 60.0
    last_failure_signature: str | None = None

    # --------------------------------------------------------------------------------------------------- paths
    def cache_path(self, u: Unit) -> Path:
        return Path(self.root) / "cache" / f"{u.unit_id}.jsonl"

    def result_path(self, u: Unit) -> Path:
        return Path(self.root) / "units" / u.phase / f"{u.unit_id}.json"

    def manifest_path(self, phase: str) -> Path:
        return Path(self.root) / "manifest" / phase / f"{self.shard[0]}.jsonl"

    # ---------------------------------------------------------------------------------------------- done check
    def is_done(self, u: Unit, full: bool = False) -> bool:
        """Result present with a valid payload hash and its cache complete: the cache size recorded at run time (a lost
        tail shows as a smaller file), and with `full` the record count and content hash (verify_replay uses full)."""
        rp, cp = self.result_path(u), self.cache_path(u)
        if not rp.exists() or not cp.exists():
            return False
        try:
            res = json.loads(rp.read_text())
        except json.JSONDecodeError:
            return False
        payload, run = split_result(res)
        if run.get("payload_sha256") != payload_sha(payload):
            return False
        if cp.stat().st_size < run.get("cache_bytes", 1 << 62):
            return False
        if not full:
            return True
        n, h = content_hash_of(cp)
        return n == run.get("cache_records") and h == run.get("cache_content_hash")

    def failed_units(self, phase: str) -> set[str]:
        out = set()
        for f in (Path(self.root) / "manifest" / phase).glob("*.jsonl"):
            out |= {e["unit"] for e in read_jsonl(f) if e.get("status") == "failed"}
        return out

    # ------------------------------------------------------------------------------------------------- header
    def unit_header(self, u: Unit) -> dict:
        return {**self.ctx.header, "unit": u.to_json(), "unit_id": u.unit_id}

    @staticmethod
    def compat(a: dict, b: dict) -> list[str]:
        """Output-affecting header fields that differ; a dirty or unknown code tree never resumes (review M6)."""
        keys = (set(a) | set(b)) - set(NOT_COMPAT)
        diff = sorted(k for k in keys if a.get(k) != b.get(k))
        if (a.get("code_dirty") or b.get("code_dirty")) and "code_dirty" not in diff:
            diff.append("code_dirty")
        return diff

    # ----------------------------------------------------------------------------------------------- run plan
    def run_phase(self, phase: str, max_units: int | None = None) -> dict:
        plan = read_plan(self.root, phase)
        if plan.get("frozen_commit") and (self.ctx.header.get("code_commit") != plan["frozen_commit"]
                                          or self.ctx.header.get("code_dirty", True)):
            raise TestPhaseCommitMismatch(f"code {self.ctx.header.get('code_commit')} "
                                          f"(dirty={self.ctx.header.get('code_dirty')}) != frozen {plan['frozen_commit']}")
        units = [Unit.from_json(d) for d in plan["units"]]
        mine = [u for i, u in enumerate(units) if i % self.shard[1] == self.shard[0]]
        failed = self.failed_units(phase)
        done_n, ran, last_failure = 0, 0, None
        t0 = time.time()
        with Heartbeat(self.ledger, self.lease, self.heartbeat_seconds):
            for u in mine:
                if u.unit_id in failed:
                    continue
                if self.is_done(u):
                    done_n += 1
                    continue
                if max_units is not None and ran >= max_units:
                    break
                if getattr(self.ctx.backend, "busy", False):          # never fail units for a busy backend (H1)
                    raise BackendBusy("the backend lock is held (a generation still running): restart the runtime")
                res = self.run_with_policy(u, plan)
                ran += 1
                if res is None:                                       # failed twice with the same signature
                    sig = self.last_failure_signature
                    if sig is not None and sig == last_failure:
                        raise RunHalted(f"two units in a row failed with the same error: {sig}")   # D130
                    last_failure = sig
                    continue
                last_failure = None
                done_n += 1
                el = time.time() - t0
                left = len(mine) - done_n
                self.log(f"[{phase}] {done_n}/{len(mine)} units done; ETA {el / ran * left / 3600:.1f} h "
                         f"(this session's pace)")
        return {"done": done_n, "total": len(mine), "ran": ran}

    # ------------------------------------------------------------------------------------------- failure policy
    def run_with_policy(self, u: Unit, plan: dict) -> dict | None:
        prev = None
        if self.result_path(u).exists():                      # done-check failed: repair needs an identical payload
            try:
                prev = split_result(json.loads(self.result_path(u).read_text()))[0]
            except json.JSONDecodeError:
                prev = None
        divergences, errors, restarts, attempts = 0, [], [], 0
        self.last_failure_signature = None
        while True:
            attempts += 1
            if attempts > MAX_ATTEMPTS:                       # D130: bounded retries per unit
                sig = errors[-1] if errors else "attempt cap"
                append_line(self.manifest_path(u.phase), canon({"unit": u.unit_id, "status": "failed",
                                                                "error": f"attempt cap: {sig}"}))
                self.last_failure_signature = sig
                return None
            try:
                return self.run_unit(u, plan, restarts, prev_payload=prev)
            except BackendBusy:
                raise
            except RepairMismatch as e:
                self._discard(u, "repair_mismatch")
                restarts.append(f"repair_mismatch: {e}")
                prev = None
            except (ReplayDivergence, LeftoverRecords) as e:
                divergences += 1
                self._discard(u, "divergence")
                restarts.append(f"divergence: {e}")
                if divergences >= 2:
                    self.halted_units.add(u.unit_id)
                    raise RunHalted(f"{u.unit_id} diverged twice: {e}") from e
            except ResumeMismatch as e:
                self._discard(u, "resume_mismatch")
                restarts.append(f"resume_mismatch: {e}")
                if self.ctx.session is not None and self.ctx.rerun_canary is not None:
                    try:
                        self.ctx.session.on_resume_mismatch(self.ctx.rerun_canary)
                    except PlatformChanged as pc:
                        self.ctx.header = {**self.ctx.header, "platform_epoch": pc.epoch,
                                           "platform_fingerprint": pc.fingerprint}
                        restarts.append(f"platform_changed: epoch {pc.epoch}")
            except CacheError as e:
                self._discard(u, "corrupt")
                restarts.append(f"corrupt: {e}")
            except Exception as e:        # noqa: BLE001 - recorded; two identical exceptions mark the unit failed
                sig = f"{type(e).__name__}: {e}"
                append_line(Path(self.root) / "failures" / f"{u.phase}.jsonl",
                            canon({"unit": u.unit_id, "error": sig, "at": time.time()}))
                if sig in errors:
                    append_line(self.manifest_path(u.phase), canon({"unit": u.unit_id, "status": "failed",
                                                                    "error": sig}))
                    self.log(f"[{u.phase}] {u.unit_id} FAILED twice: {sig}")
                    self.last_failure_signature = sig
                    return None
                errors.append(sig)
                self._discard(u, "exception")
                restarts.append(f"exception: {sig}")

    def _discard(self, u: Unit, reason: str) -> None:
        cp = self.cache_path(u)
        n = content_hash_of(cp)[0] if cp.exists() else 0
        archive(cp, reason, {"phase": u.phase, "condition": u.condition, "unit_id": u.unit_id})
        if self.result_path(u).exists() and reason == "repair_mismatch":
            self.result_path(u).replace(self.result_path(u).with_suffix(f".{int(time.time())}.discarded.json"))
        self.ledger.event("discarded", unit=u.unit_id, reason=reason, calls=n)

    # ------------------------------------------------------------------------------------------------ one unit
    def run_unit(self, u: Unit, plan: dict, restarts: list | None = None, prev_payload: dict | None = None) -> dict:
        from ..agent import run_rq1_unit
        from ..agent.loop import LoopConfig
        from ..apisuite import sample_stream
        from ..rq3 import run_rq3_unit
        backend = self.ctx.backend
        if getattr(backend, "busy", False):
            from .llama_backend import BackendBusy
            raise BackendBusy("the backend lock is held (a generation still running): restart the runtime")
        header = self.unit_header(u)
        cp = self.cache_path(u)
        if cp.exists():
            old = read_header(cp)
            diff = ["header"] if old is None else self.compat(old, header)
            if diff:
                n_old = content_hash_of(cp)[0]
                archive(cp, "stamp", {"phase": u.phase, "condition": u.condition, "fields": diff})
                self.ledger.event("discarded", unit=u.unit_id, reason="stamp", calls=n_old, fields=diff)
                (restarts if restarts is not None else []).append(f"stamp: {diff}")
        session = self.ctx.session
        replay_only = bool(session and session.replay_only_mode)
        writer = uuid.uuid4().hex
        cache = UnitCache(cp, header, writer, resume_check=not replay_only)
        resumed = cache.resuming
        cb = CachedBackend(backend, cache)
        ce = CachedEmbedder(self.ctx.embedder, cache)
        cfg = LoopConfig(**dict(u.params)) if u.params else None
        t0 = time.time()
        if u.rq == "rq1":
            spec = sample_stream(u.seed, u.split, kind=u.kind)          # full stream; shorter units run a prefix
            res = run_rq1_unit(spec, u.condition, cb, ce, self.ctx.tokenizer, cfg,
                               n_tasks=u.n_tasks if u.n_tasks != 20 else None)
        else:
            spec = sample_stream(u.seed, u.split, kind="rq3")
            res = run_rq3_unit(spec, u.condition, u.window, cb, ce, self.ctx.tokenizer, cfg)
        cache.finish()
        wall = time.time() - t0
        res["unit"] = u.to_json()
        res["stamp"]["llm"] = {k: v for k, v in header.items() if k not in NOT_COMPAT and k not in ("unit", "unit_id")}
        payload = normalise(res)
        if prev_payload is not None:
            same = payload_sha(payload) == payload_sha(prev_payload)
            if resumed and not same:                          # resumed in the same stamp must reproduce the result
                raise RepairMismatch(f"{u.unit_id}: resumed payload differs from the stored result")
            (restarts if restarts is not None else []).append(
                f"repair: {'resumed' if resumed else 'from scratch'}, payload {'equal' if same else 'differs'}")
        n, h = cache.n_records(), cache.content_hash()
        run = {"payload_sha256": payload_sha(payload), "cache_records": n, "cache_content_hash": h,
               "wall_seconds": round(wall, 2), "session": self.ledger.session, "writer": writer,
               "resumed": resumed, "replay_only": bool(resumed and replay_only), "restarts": list(restarts or []),
               "calls_fresh": cache.fresh_calls(), "calls_replayed": n - cache.fresh_calls(),
               "gpu_driver": self.ctx.header.get("gpu_driver"), "cuda": self.ctx.header.get("cuda"),
               "plan_sha256": plan.get("sha256"), "cache_bytes": cp.stat().st_size}
        atomic_write(self.result_path(u), json.dumps({**payload, "run": run}))
        append_line(self.manifest_path(u.phase), canon({"unit": u.unit_id, "status": "done",
                                                        "payload_sha256": run["payload_sha256"]}))
        llm_seconds = sum(r.get("seconds", 0.0) for r in cache.records if r["writer"] == writer and r["kind"] != "embed")
        self.ledger.event("unit", unit=u.unit_id, wall_seconds=round(wall, 2), calls_fresh=run["calls_fresh"],
                          calls_replayed=run["calls_replayed"], llm_seconds=round(llm_seconds, 3))
        return {**payload, "run": run}


# ----------------------------------------------------------------------------------------------------- verify
class _Raising:
    def __init__(self, key_source: Any = None, emb_id: str | None = None):
        self.key_source = key_source
        self.id = getattr(key_source, "id", None) or emb_id or "raising"
        if key_source is not None and hasattr(key_source, "cache_key"):
            self.cache_key = key_source.cache_key
        if key_source is not None and hasattr(key_source, "register_tasks"):
            self.register_tasks = key_source.register_tasks

    def generate(self, *a, **k):
        raise ReplayDivergence(-1, {"kind": "verify", "key": ""}, "generate", "")

    def embed(self, text):
        raise ReplayDivergence(-1, {"kind": "verify", "key": ""}, "embed", "")


def verify_replay(root: Path, phase: str, unit_id: str, key_source: Any, tokenizer: Any) -> bool:
    """Replay a finished unit from its cache with backends that raise (no generation, no embedding); the payload must
    be byte-identical. `key_source` provides `cache_key` (a vocab-only LlamaCppBackend, or the scripted backend)."""
    from ..agent import run_rq1_unit
    from ..agent.loop import LoopConfig
    from ..apisuite import sample_stream
    from ..rq3 import run_rq3_unit
    rp = Path(root) / "units" / phase / f"{unit_id}.json"
    cp = Path(root) / "cache" / f"{unit_id}.jsonl"
    stored = json.loads(rp.read_text())
    payload, run = split_result(stored)
    u = Unit.from_json(payload["unit"])
    cache = UnitCache(cp, {}, "verify", readonly=True)
    cb = CachedBackend(_Raising(key_source), cache)
    ce = CachedEmbedder(_Raising(emb_id=payload["stamp"]["embedder"]), cache)
    cfg = LoopConfig(**dict(u.params)) if u.params else None
    if u.rq == "rq1":
        res = run_rq1_unit(sample_stream(u.seed, u.split, kind=u.kind), u.condition, cb, ce, tokenizer, cfg,
                           n_tasks=u.n_tasks if u.n_tasks != 20 else None)
    else:
        res = run_rq3_unit(sample_stream(u.seed, u.split, kind="rq3"), u.condition, u.window, cb, ce, tokenizer, cfg)
    cache.finish()
    res["unit"] = u.to_json()
    res["stamp"]["llm"] = payload["stamp"]["llm"]
    res = normalise(res)
    for k in ENV_STAMP_FIELDS:                                # verify may run at another commit or on another machine
        if k in payload["stamp"]:
            res["stamp"][k] = payload["stamp"][k]
    # the replay must reproduce the stored payload exactly, that payload must be the one recorded at run time, and the
    # cache must be the one the run recorded
    return (payload_sha(res) == payload_sha(payload) == run["payload_sha256"]
            and cache.content_hash() == run["cache_content_hash"] and cache.n_records() == run["cache_records"])
```
