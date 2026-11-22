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
from .fsio import append_line, atomic_write, canon, read_lines

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

    def __init__(self, path: Path, header: dict, writer: str, resume_check: bool = True, readonly: bool = False,
                 resume_meta: dict | None = None):
        """`readonly`: replay only (verify_replay): no resume line, no appends. `resume_meta`: recorded in the resume
        line (e.g. the CPU model: embeddings of a unit resumed across CPU types mix two CPUs)."""
        self.path, self.writer, self.resume_check, self.readonly = Path(path), writer, resume_check, readonly
        self.records: list[dict] = []
        self.writers: set[str] = set()
        self.resumes: list[dict] = []
        self.pos = 0
        if readonly:
            self._safe_load(rewrite=False)
            self.resuming = False
            return
        if self.path.exists():
            self._safe_load()
            self.resuming = bool(self.records)
            llm_idx = [r["i"] for r in self.records if r["kind"] != "embed"]
            self.check_idx = set(llm_idx[-2:])          # the last two LLM calls are regenerated on resume (D61)
            line = {"type": "resume", "writer": writer, "at": utc_stamp(), **(resume_meta or {})}
            append_line(self.path, canon(line))
            self.resumes.append(line)
            self.writers.add(writer)
        else:
            self.header = {"type": "header", "format": CACHE_FORMAT, **header, "writer": writer}
            append_line(self.path, canon(self.header))
            self.writers.add(writer)
            self.resuming = False
        self.check_idx = getattr(self, "check_idx", set())

    # ------------------------------------------------------------------------------------------------------- load
    def _safe_load(self, rewrite: bool = True) -> None:
        """Any failure while loading is a CacheError (archive and restart), never an error that is retried."""
        try:
            self._load(rewrite)
        except CacheError:
            raise
        except Exception as e:     # noqa: BLE001 - e.g. a bad key in a crc-valid line, OSError
            raise CacheError(f"unreadable cache: {type(e).__name__}: {e}") from e

    def _load(self, rewrite: bool = True) -> None:
        ends_nl = self.path.read_bytes().endswith(b"\n")
        lines = read_lines(self.path)                    # undecodable line (torn inside a character) -> None
        parsed: list[dict | None] = []
        for ln in lines:
            try:
                parsed.append(None if ln is None else json.loads(ln))
            except json.JSONDecodeError:
                parsed.append(None)
        torn = bool(parsed) and (parsed[-1] is None or not ends_nl
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
                self.resumes.append(obj)
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
        with open(path, "rb") as fh:                     # bytes: a bad byte further down must not break the header
            obj = json.loads(fh.readline().decode("utf-8"))
    except (OSError, ValueError):                         # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return None
    return obj if obj.get("type") == "header" else None


def content_hash_of(path: Path) -> tuple[int, str]:
    """(records, content hash) of a cache file without opening it for writing."""
    recs = []
    for ln in read_lines(path):                          # bytes split on \n; undecodable -> None
        if ln is None:
            break
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
