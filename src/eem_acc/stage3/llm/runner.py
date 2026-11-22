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

PLAN_FORMAT = "s33-plan-v1"
NOT_COMPAT = ("type", "format", "writer", "gpu_driver", "cuda", "created")   # recorded, but a resume may differ


class RunHalted(RuntimeError):
    """A unit diverged twice: stop the campaign with diagnostics."""


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
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                                timeout=10).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", True
    return {"code_commit": commit, "code_dirty": dirty}


def make_header(backend_stamp: dict, embedder_id: str, env: dict, epoch: int | None) -> dict:
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
            "sentence_transformers": env.get("sentence_transformers"), "platform_epoch": epoch}


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

    # --------------------------------------------------------------------------------------------------- paths
    def cache_path(self, u: Unit) -> Path:
        return Path(self.root) / "cache" / f"{u.unit_id}.jsonl"

    def result_path(self, u: Unit) -> Path:
        return Path(self.root) / "units" / u.phase / f"{u.unit_id}.json"

    def manifest_path(self, phase: str) -> Path:
        return Path(self.root) / "manifest" / phase / f"{self.shard[0]}.jsonl"

    # ---------------------------------------------------------------------------------------------- done check
    def is_done(self, u: Unit) -> bool:
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
        keys = (set(a) | set(b)) - set(NOT_COMPAT)
        return sorted(k for k in keys if a.get(k) != b.get(k))

    # ----------------------------------------------------------------------------------------------- run plan
    def run_phase(self, phase: str, max_units: int | None = None) -> dict:
        plan = read_plan(self.root, phase)
        if plan.get("frozen_commit") and self.ctx.header.get("code_commit") != plan["frozen_commit"]:
            raise TestPhaseCommitMismatch(f"code {self.ctx.header.get('code_commit')} != frozen "
                                          f"{plan['frozen_commit']}")
        units = [Unit.from_json(d) for d in plan["units"]]
        mine = [u for i, u in enumerate(units) if i % self.shard[1] == self.shard[0]]
        failed = self.failed_units(phase)
        done_n, ran = 0, 0
        t0 = time.time()
        for u in mine:
            if u.unit_id in failed:
                continue
            if self.is_done(u):
                done_n += 1
                continue
            if max_units is not None and ran >= max_units:
                break
            self.run_with_policy(u, plan)
            ran += 1
            done_n += 1
            self.ledger.heartbeat()
            el = time.time() - t0
            left = len(mine) - done_n
            self.log(f"[{phase}] {done_n}/{len(mine)} units done; ETA {el / ran * left / 3600:.1f} h "
                     f"(this session's pace)" if ran else "")
        return {"done": done_n, "total": len(mine), "ran": ran}

    # ------------------------------------------------------------------------------------------- failure policy
    def run_with_policy(self, u: Unit, plan: dict) -> dict | None:
        prev = None
        if self.result_path(u).exists():                      # done-check failed: repair needs an identical payload
            try:
                prev = split_result(json.loads(self.result_path(u).read_text()))[0]
            except json.JSONDecodeError:
                prev = None
        divergences, errors, restarts = 0, [], []
        while True:
            try:
                res = self.run_unit(u, plan, restarts)
                if prev is not None and payload_sha(split_result(res)[0]) != payload_sha(prev):
                    self._discard(u, "repair_mismatch")
                    restarts.append("repair_mismatch")
                    prev = None
                    continue
                return res
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
                if self.ctx.session is not None:
                    try:
                        self.ctx.session.on_resume_mismatch(self.ctx.rerun_canary)
                    except PlatformChanged as pc:
                        self.ctx.header = {**self.ctx.header, "platform_epoch": pc.epoch}
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
    def run_unit(self, u: Unit, plan: dict, restarts: list | None = None) -> dict:
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
                archive(cp, "stamp", {"phase": u.phase, "condition": u.condition, "fields": diff})
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
        res["stamp"]["plan_sha256"] = plan.get("sha256")
        payload = normalise(res)
        n, h = cache.n_records(), cache.content_hash()
        run = {"payload_sha256": payload_sha(payload), "cache_records": n, "cache_content_hash": h,
               "wall_seconds": round(wall, 2), "session": self.ledger.session, "writer": writer,
               "resumed": resumed, "replay_only": bool(resumed and replay_only), "restarts": list(restarts or []),
               "calls_fresh": cache.fresh_calls(), "calls_replayed": n - cache.fresh_calls(),
               "gpu_driver": self.ctx.header.get("gpu_driver"), "cuda": self.ctx.header.get("cuda")}
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
    res["stamp"]["plan_sha256"] = payload["stamp"]["plan_sha256"]
    res = normalise(res)
    for k in ENV_STAMP_FIELDS:                                # verify may run at another commit or on another machine
        if k in payload["stamp"]:
            res["stamp"][k] = payload["stamp"][k]
    # the replay must reproduce the stored payload exactly, and that payload must be the one recorded at run time
    return payload_sha(res) == payload_sha(payload) == run["payload_sha256"]
