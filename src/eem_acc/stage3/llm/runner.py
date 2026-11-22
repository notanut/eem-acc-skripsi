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
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cache import (CacheError, CachedBackend, CachedEmbedder, LeftoverRecords, ReplayDivergence, ResumeMismatch,
                    UnitCache, archive, content_hash_of, read_header)
from .canary import Session, SessionNondeterministic
from .fsio import append_line, atomic_write, canon, read_jsonl
from .ledger import Ledger
from .llama_backend import BackendBusy

PLAN_FORMAT = "s33-plan-v1"
NOT_COMPAT = ("type", "format", "writer", "gpu_driver", "cuda", "created", "cpu_model")   # recorded; may differ
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


class LeaseLost(RuntimeError):
    """Another runtime took the lease while this one was running: stop."""


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
        if self.params:                                   # params change outputs: they are part of the identity (M1)
            parts.append("p" + hashlib.sha256(canon([list(p) for p in self.params]).encode()).hexdigest()[:8])
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

    def _read(self) -> dict | None:
        try:
            return json.loads(self.file.read_text())
        except (OSError, json.JSONDecodeError):
            return None                                   # missing or unreadable: treated as free

    def acquire(self) -> None:
        cur = self._read()
        if cur and cur.get("runtime") != self.runtime and self.clock() - cur.get("t", 0) < self.stale:
            raise LeaseHeld(f"runtime {cur['runtime']} holds the lease (heartbeat {self.clock() - cur['t']:.0f}s ago)")
        self._write()

    def refresh(self) -> None:
        """Read first: if another runtime owns a fresh lease, this runtime lost it (LeaseLost)."""
        cur = self._read()
        if cur and cur.get("runtime") != self.runtime and self.clock() - cur.get("t", 0) < self.stale:
            raise LeaseLost(f"runtime {cur['runtime']} took the lease")
        self._write()

    def _write(self) -> None:
        atomic_write(self.file, json.dumps({"runtime": self.runtime, "t": self.clock()}))

    def release(self) -> None:
        cur = self._read()
        if cur and cur.get("runtime") == self.runtime:
            try:
                self.file.unlink()
            except FileNotFoundError:
                pass


class Claims:
    """Per-unit claim files (D133): `claims/<unit_id>.claim` created with O_EXCL (runtime, pid); the heartbeat keeps
    the mtime fresh; a fresh claim of another worker is skipped, a stale one (crashed worker) is taken over."""

    def __init__(self, root: Path, owner: str, stale: float = 180.0, clock=time.time):
        self.dir, self.owner, self.stale, self.clock = Path(root) / "claims", owner, stale, clock
        self.held: set[str] = set()

    def path(self, unit_id: str) -> Path:
        return self.dir / f"{unit_id}.claim"

    def claim(self, unit_id: str) -> bool:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.path(unit_id)
        for _ in range(2):
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"owner": self.owner, "pid": os.getpid()}).encode())
                os.close(fd)
                self.held.add(unit_id)
                return True
            except FileExistsError:
                try:
                    owner = json.loads(p.read_text()).get("owner")
                    age = self.clock() - p.stat().st_mtime
                except (OSError, json.JSONDecodeError):
                    owner, age = None, self.stale + 1
                if owner == self.owner:
                    self.held.add(unit_id)
                    return True
                if age < self.stale:
                    return False
                try:
                    p.unlink()                            # stale: take over
                except FileNotFoundError:
                    pass
        return False

    def touch(self) -> None:
        for u in list(self.held):
            try:
                os.utime(self.path(u))
            except FileNotFoundError:
                self.held.discard(u)

    def release(self, unit_id: str) -> None:
        self.held.discard(unit_id)
        try:
            self.path(unit_id).unlink()
        except FileNotFoundError:
            pass


class Heartbeat:
    """Daemon thread: ledger heartbeat and lease refresh every `seconds` while a phase runs (review M4)."""

    def __init__(self, ledger: Ledger, lease: Lease | None, seconds: float = 60.0, claims: "Claims | None" = None):
        import threading
        self.ledger, self.lease, self.seconds, self.claims = ledger, lease, seconds, claims
        self.lost: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.seconds):
            self.beat()

    def beat(self) -> None:
        """Never dies: an error is logged to the ledger (if possible) and the next beat tries again."""
        try:
            self.ledger.heartbeat()
            if self.claims is not None:
                self.claims.touch()
            if self.lease is not None:
                self.lease.refresh()
        except LeaseLost as e:
            self.lost = e
        except Exception as e:     # noqa: BLE001
            try:
                self.ledger.event("heartbeat_error", error=f"{type(e).__name__}: {e}")
            except Exception:      # noqa: BLE001
                pass

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

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
        if self.lease is not None:
            self.lease.release()


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


def error_signature(e: BaseException) -> str:
    """Type and message with numbers stripped (unit-specific sizes, addresses), for "the same error" (D130)."""
    return re.sub(r"\d+", "#", f"{type(e).__name__}: {e}")


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
            "platform_fingerprint": fingerprint, "cpu_model": env.get("cpu_model")}


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
    rerun_canary: Callable[[], str] | None = None          # re-runs the canary, returns its fingerprint (D131)


@dataclass
class Runner:
    root: Path
    ctx: RunContext
    ledger: Ledger
    shard: tuple[int, int] = (0, 1)
    log: Callable[[str], None] = print
    halted_units: set = field(default_factory=set)
    lease: "Lease | None" = None                           # the parent process only (second opinion M4)
    heartbeat_seconds: float = 60.0
    last_failure_signature: str | None = None
    claims: "Claims | None" = None
    session_heartbeat: "Heartbeat | None" = None           # started at session setup by the notebook (Low 4)

    # --------------------------------------------------------------------------------------------------- paths
    def cache_path(self, u: Unit) -> Path:
        return Path(self.root) / "cache" / f"{u.unit_id}.jsonl"

    def result_path(self, u: Unit) -> Path:
        return Path(self.root) / "units" / u.phase / f"{u.unit_id}.json"

    def manifest_path(self, phase: str) -> Path:
        return Path(self.root) / "manifest" / phase / f"{self.shard[0]}.jsonl"

    # ---------------------------------------------------------------------------------------------- done check
    def stored_matches(self, u: Unit, plan: dict | None) -> bool | None:
        """None: no readable result; else whether the stored result is this planned unit (and, in a frozen plan, was
        made at the frozen commit) (second opinion M1)."""
        try:
            payload = split_result(json.loads(self.result_path(u).read_text()))[0]
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("unit") != u.to_json():
            return False
        if plan and plan.get("frozen_commit"):
            return payload.get("stamp", {}).get("llm", {}).get("code_commit") == plan["frozen_commit"]
        return True

    def is_done(self, u: Unit, full: bool = False, plan: dict | None = None) -> bool:
        """Result present with a valid payload hash, made for this planned unit, and its cache complete: the cache size
        recorded at run time (a lost tail shows as a smaller file), and with `full` the record count and content hash
        (verify_replay uses full)."""
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
        if self.stored_matches(u, plan) is False:
            return False
        if cp.stat().st_size < run.get("cache_bytes", 1 << 62):
            return False
        if not full:
            return True
        n, h = content_hash_of(cp)
        return n == run.get("cache_records") and h == run.get("cache_content_hash")

    def failed_units(self, phase: str) -> set[str]:
        """Units whose LAST status line (over all shards' manifests) is `failed` (a `retry` line re-enables them)."""
        last: dict[str, tuple[float, str]] = {}
        for f in (Path(self.root) / "manifest" / phase).glob("*.jsonl"):
            for e in read_jsonl(f):
                t = e.get("t", 0.0)
                if e["unit"] not in last or t >= last[e["unit"]][0]:
                    last[e["unit"]] = (t, e.get("status"))
        return {u for u, (_, st) in last.items() if st == "failed"}

    def mark(self, phase: str, unit_id: str, status: str, **fields) -> None:
        append_line(self.manifest_path(phase), canon({"unit": unit_id, "status": status, "t": time.time(), **fields}))

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
    def run_phase(self, phase: str, max_units: int | None = None, retry_failed: bool = False) -> dict:
        plan = read_plan(self.root, phase)
        if plan.get("frozen_commit") and (self.ctx.header.get("code_commit") != plan["frozen_commit"]
                                          or self.ctx.header.get("code_dirty", True)):
            raise TestPhaseCommitMismatch(f"code {self.ctx.header.get('code_commit')} "
                                          f"(dirty={self.ctx.header.get('code_dirty')}) != frozen {plan['frozen_commit']}")
        units = [Unit.from_json(d) for d in plan["units"]]
        mine = [u for i, u in enumerate(units) if i % self.shard[1] == self.shard[0]]
        if retry_failed:
            for uid in self.failed_units(phase) & {u.unit_id for u in mine}:
                self.mark(phase, uid, "retry")
        failed = self.failed_units(phase)
        done_n, ran, skipped, last_failure = 0, 0, 0, None
        n_failed = sum(u.unit_id in failed for u in mine)
        hb = None if self.session_heartbeat is not None else Heartbeat(self.ledger, self.lease, self.heartbeat_seconds,
                                                                        self.claims)
        beat = self.session_heartbeat or hb
        if hb is not None:
            hb.__enter__()
        try:
            for u in mine:
                if beat.lost is not None:
                    raise beat.lost
                if u.unit_id in failed:
                    continue
                if self.result_path(u).exists() and self.stored_matches(u, plan) is False:
                    self._discard(u, "unit_mismatch", result_too=True)     # a stale result of another plan (M1)
                if self.is_done(u, plan=plan):
                    done_n += 1
                    continue
                if max_units is not None and ran >= max_units:
                    break
                if getattr(self.ctx.backend, "busy", False):          # never fail units for a busy backend
                    raise BackendBusy("the backend lock is held (a generation still running): restart the runtime")
                if self.claims is not None and not self.claims.claim(u.unit_id):
                    skipped += 1                                      # another worker runs it (D133)
                    continue
                try:
                    res = self.run_with_policy(u, plan)
                finally:
                    if self.claims is not None:
                        self.claims.release(u.unit_id)
                ran += 1
                if res is None:                                       # failed (same signature twice, or cap)
                    n_failed += 1
                    sig = self.last_failure_signature
                    if sig is not None and last_failure is not None and sig == last_failure[0]:
                        for uid in (last_failure[1], u.unit_id):      # D130: halt; they are retried after a fix
                            self.mark(phase, uid, "retry", reason="halted run")
                        raise RunHalted(f"two units in a row failed with the same error: {sig}")
                    last_failure = (sig, u.unit_id)
                    continue
                last_failure = None
                done_n += 1
                self.log(self.progress_line(phase, done_n, len(mine), n_failed))
        finally:
            if hb is not None:
                hb.__exit__(None, None, None)
        return {"done": done_n, "total": len(mine), "ran": ran, "failed": n_failed, "skipped_claimed": skipped}

    def progress_line(self, phase: str, done: int, total: int, failed: int) -> str:
        """Progress, ETA from the ledger's mean wall seconds per unit, GPU-hours vs the 60 h gate (D52). Never shows
        outcomes (D64)."""
        from .ledger import summarise
        s = summarise(self.root)
        walls = s.get("unit_wall_seconds") or []
        eta = (sum(walls) / len(walls)) * (total - done - failed) / 3600 if walls else float("nan")
        return (f"[{phase}] {done}/{total} done, {failed} failed; ETA {eta:.1f} h; "
                f"GPU-hours {s['gpu_hours']:.1f} / 60")

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
                self.mark(u.phase, u.unit_id, "failed", error=f"attempt cap: {sig}")
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
                self._session_mismatch()
            except CacheError as e:
                self._discard(u, "corrupt")
                restarts.append(f"corrupt: {e}")
            except SessionNondeterministic:
                raise
            except Exception as e:        # noqa: BLE001 - recorded; two identical exceptions mark the unit failed
                sig = error_signature(e)
                append_line(Path(self.root) / "failures" / f"{u.phase}.jsonl",
                            canon({"unit": u.unit_id, "error": f"{type(e).__name__}: {e}", "signature": sig,
                                   "at": time.time()}))
                if sig in errors:
                    self.mark(u.phase, u.unit_id, "failed", error=sig)
                    self.log(f"[{u.phase}] {u.unit_id} FAILED twice: {sig}")
                    self.last_failure_signature = sig
                    return None
                errors.append(sig)
                restarts.append(f"exception: {sig}")      # D132: retry by resuming (the cache is intact; no archive)

    def _session_mismatch(self) -> None:
        """D61/D66/D131: a resume mismatch (or a same-platform repair difference); a canary that differs within the
        session halts it (SessionNondeterministic propagates)."""
        if self.ctx.session is None or self.ctx.rerun_canary is None:
            return
        try:
            self.ctx.session.on_resume_mismatch(self.ctx.rerun_canary)
        except SessionNondeterministic as e:
            self.ledger.event("session_nondeterministic", error=str(e))
            raise

    def _archive_result(self, u: Unit, reason: str) -> None:
        rp = self.result_path(u)
        if rp.exists():
            rp.replace(rp.with_name(f"{rp.stem}.{int(time.time() * 1000)}.{reason}.discarded.json"))

    def _discard(self, u: Unit, reason: str, result_too: bool = False) -> None:
        cp = self.cache_path(u)
        n = content_hash_of(cp)[0] if cp.exists() else 0
        archive(cp, reason, {"phase": u.phase, "condition": u.condition, "unit_id": u.unit_id})
        if result_too or reason == "repair_mismatch":
            self._archive_result(u, reason)
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
        cpu = self.ctx.header.get("cpu_model")
        cache = UnitCache(cp, header, writer, resume_check=not replay_only, resume_meta={"cpu_model": cpu})
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
            if not same:
                self._archive_result(u, "repair_superseded")  # the old result is kept, never overwritten
                prev_fp = prev_payload.get("stamp", {}).get("llm", {}).get("platform_fingerprint")
                if not resumed and prev_fp and prev_fp == self.ctx.header.get("platform_fingerprint"):
                    self.ledger.event("repair_nondeterminism", unit=u.unit_id)    # same platform, other payload
                    self._session_mismatch()
        n, h = cache.n_records(), cache.content_hash()
        run = {"payload_sha256": payload_sha(payload), "cache_records": n, "cache_content_hash": h,
               "wall_seconds": round(wall, 2), "session": self.ledger.session, "writer": writer,
               "resumed": resumed, "replay_only": bool(resumed and replay_only), "restarts": list(restarts or []),
               "calls_fresh": cache.fresh_calls(), "calls_replayed": n - cache.fresh_calls(),
               "gpu_driver": self.ctx.header.get("gpu_driver"), "cuda": self.ctx.header.get("cuda"),
               "plan_sha256": plan.get("sha256"), "cache_bytes": cp.stat().st_size,
               "cpu_model": cpu, "cross_cpu_resume": len({cache.header.get("cpu_model"), cpu,
                                                          *(r.get("cpu_model") for r in cache.resumes)}) > 1}
        atomic_write(self.result_path(u), json.dumps({**payload, "run": run}))
        self.mark(u.phase, u.unit_id, "done", payload_sha256=run["payload_sha256"])
        fresh_llm = [r for r in cache.records if r["writer"] == writer and r["kind"] != "embed"]
        self.ledger.event("unit", unit=u.unit_id, wall_seconds=round(wall, 2), calls_fresh=run["calls_fresh"],
                          calls_fresh_llm=len(fresh_llm), calls_replayed=run["calls_replayed"],
                          llm_seconds=round(sum(r.get("seconds", 0.0) for r in fresh_llm), 3))
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
