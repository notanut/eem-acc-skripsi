"""S3.3b helpers for notebook 04_stage3 on the T4 (docs/s33b_spec.md r2). Thin notebook, logic here; CPU-tested with
the tiny qwen2 GGUF. Nothing here is a thesis result: verification, smoke run, forced kills, throughput benchmark."""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ..agent.backend import Gen, t_llm
from .cache import UnitCache, crc, zunpack
from .fsio import atomic_write, read_lines
from .llama_backend import SAMPLING, LlamaCppBackend, load_params_of

S30_TEMPLATE_SHA = "d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4"
S30_PART_SHA256 = {"qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf":
                   "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
                   "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf":
                   "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a"}
S30_WHEEL_SHA256 = "ab93c390ddbbff08a721f39db45b09fce1653b7c9cf1a8e8b5675b4e8516ef99"
S30_TEXT_CHARS = 500           # S3.0 stored the first 500 characters of each output
S30_MAX_TOKENS = 160
TIMING_TOLERANCE = 0.15        # flag if the median measured/t_llm ratio leaves 1 +/- 0.15 (S3.0: 0.98, SD 0.08)
KILL_A_AFTER = 12              # kill (a): after this many fresh LLM calls of the target unit


# ------------------------------------------------------------------------------------------------------- sections
class Sections:
    """`Skripsi/stage3/s33b/<session>.json`: one key per finished measurement section (skipped on a re-run). Per-runtime
    sections (setup, load, canary, lease, claims) are not stored here and always run."""

    def __init__(self, path: Path, stamp: dict | None = None):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        if stamp and "stamp" not in self.data and "nb_version" not in stamp:
            # a new session file is started only by a notebook that carries its cell fingerprint: an outdated copy
            # of the notebook (no NB_VERSION) running against newer code is refused before it pins the commit
            raise RuntimeError("This notebook is an OUTDATED copy: close it and open the Colab link from the chat "
                               "(it is pinned to the latest commit), then run from the top.")
        if stamp:
            self.data.setdefault("stamp", stamp)

    def done(self, name: str) -> bool:
        return name in self.data

    def save(self, name: str, value: Any) -> Any:
        self.data[name] = value
        atomic_write(self.path, json.dumps(self.data, indent=1, default=str))
        return value

    def append(self, name: str, value: Any) -> None:
        self.data.setdefault(name, []).append(value)
        atomic_write(self.path, json.dumps(self.data, indent=1, default=str))


def model_digest(part_sha256s: dict[str, str]) -> str:
    """sha256 over the sorted part sha256s of a split GGUF (S3.3a spec section 4)."""
    return hashlib.sha256("".join(sorted(part_sha256s.values())).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_parts(paths: list[Path], expected: dict[str, str] = S30_PART_SHA256) -> dict:
    got = {Path(p).name: sha256_file(p) for p in paths}
    return {"sha256": got, "equal_s30": set(got) == set(expected) and all(expected[k] == v for k, v in got.items())}


def timeline() -> dict:
    """VM boot time and now (s33 15.10)."""
    from .ledger import vm_boot_time
    return {"vm_boot": vm_boot_time(), "now": time.time()}


def gpu_telemetry() -> dict:
    """GPU clocks, temperature, memory and throttle reasons (nvidia-smi); empty without a GPU."""
    q = "clocks.sm,clocks.mem,temperature.gpu,memory.used,memory.total,clocks_throttle_reasons.active,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    if not out:
        return {}
    return dict(zip(q.split(","), [v.strip() for v in out.splitlines()[0].split(",")]))


def cuda_libraries() -> dict:
    """The CUDA runtime and cuBLAS that llama.cpp links (not torch's CUDA)."""
    info: dict[str, Any] = {}
    try:
        info["cuda_version_file"] = Path("/usr/local/cuda/version.json").read_text()[:400]
    except OSError:
        info["cuda_version_file"] = None
    try:
        import llama_cpp
        libdir = Path(llama_cpp.__file__).parent / "lib"
        libs = sorted(str(p) for p in libdir.glob("*.so*")) if libdir.exists() else []
        info["llama_libs"] = libs
        linked = []
        for lib in libs:
            out = subprocess.run(["ldd", lib], capture_output=True, text=True, timeout=10).stdout
            linked += [ln.strip() for ln in out.splitlines() if "cublas" in ln or "cudart" in ln]
        info["linked"] = sorted(set(linked))
    except Exception as e:     # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def cuda_fingerprint(cuda_libs: dict) -> dict:
    """{library path: [realpath, size, sha256]} of the CUDA libraries llama.cpp links (review M3: a cuBLAS update keeps
    the path; this does not). Empty when `cuda_libraries()` failed."""
    out = {}
    for ln in linked_paths(cuda_libs):
        path = ln.split("=>", 1)[-1].strip()
        try:
            real = os.path.realpath(path)
            out[path] = [real, os.path.getsize(real), sha256_file(Path(real))]
        except OSError as e:
            out[path] = [None, None, f"{type(e).__name__}"]
    return out


def linked_paths(cuda_libs: dict) -> list[str]:
    """`cuda_libraries()["linked"]` without the load addresses `(0x...)`, which differ in every process (review L2)."""
    import re
    return sorted({re.sub(r"\s*\(0x[0-9a-fA-F]+\)\s*$", "", ln).strip() for ln in cuda_libs.get("linked", [])})


# ------------------------------------------------------------------------------------------- session 1: verify
def template_check(llm: Any) -> dict:
    tpl = llm.metadata["tokenizer.chat_template"]
    sha = hashlib.sha256(tpl.encode()).hexdigest()
    return {"sha256": sha, "equals_s30": sha == S30_TEMPLATE_SHA, "text": tpl}


def _s30_view(d: dict) -> dict:
    return {"n_in": d["n_in"], "n_out": d["n_out"], "finish_reason": d["finish_reason"],
            "text": d["text"][:S30_TEXT_CHARS]}


def s30_equality(backend: LlamaCppBackend, s30_records: dict, prompts: list[list[dict]],
                 chat_path: Callable[[list[dict], bool], dict] | None = None, limit: int | None = None) -> dict:
    """`limit`: only the first `limit` prompts (both kinds each; D145 re-checks 2 prompts = 2 free + 2 grammar records
    with torch loaded). Regenerate S3.0's fresh-path timing records with the rendered path, in S3.0's order (warm-up calls, then free and
    grammar alternating by i % 2), and compare what S3.0 stored. On a mismatch the prompt is also regenerated with
    S3.0's own chat path (`chat_path(messages, grammar) -> {text, n_in, n_out, finish_reason}`): backend != chat path ->
    code bug; neither equals S3.0 -> platform change."""
    # S3.0 benchmarked one prompt per call, so every record says "prompt": 0; the list POSITION is the prompt index
    by = {k: dict(enumerate(s30_records[k])) for k in ("free", "grammar")}
    for k in by:
        if len(by[k]) != len(prompts):
            raise ValueError(f"S3.0 has {len(by[k])} {k} records for {len(prompts)} prompts")
    gid = {"free": None, "grammar": "envelope_cap300"}
    backend.generate(prompts[0], "action", None, S30_MAX_TOKENS)                 # S3.0's warm-up calls
    backend.generate(prompts[0], "action", "envelope_cap300", S30_MAX_TOKENS)
    out: dict[str, Any] = {"compared": 0, "equal": 0, "differences": [], "records": []}
    for i in range(len(prompts) if limit is None else min(limit, len(prompts))):
        for kind in (("free", "grammar") if i % 2 == 0 else ("grammar", "free")):
            rec = by[kind].get(i)
            if rec is None:
                continue
            t0 = time.perf_counter()
            g = backend.generate(prompts[i], "action", gid[kind], S30_MAX_TOKENS)
            sec = time.perf_counter() - t0
            got = _s30_view(gen_dict(g))
            out["records"].append({"kind": kind, "prompt": i, "n_in": g.n_in, "n_out": g.n_out, "seconds": sec,
                                   "grammar": gid[kind]})
            want = _s30_view(rec)
            out["compared"] += 1
            if got == want:
                out["equal"] += 1
                continue
            diff = {"kind": kind, "prompt": i, "fields": sorted(k for k in got if got[k] != want[k])}
            if chat_path is not None:
                cv = _s30_view(chat_path(prompts[i], kind == "grammar"))
                diff["chat_path_equals_s30"] = cv == want
                diff["classification"] = ("code bug (backend differs from the chat path)" if cv != got else
                                          "platform change (the chat path does not reproduce S3.0 either)")
            out["differences"].append(diff)
    out["all_equal"] = out["equal"] == out["compared"] > 0
    out["coverage"] = {"n_in_max": max(r["n_in"] for k in by for r in by[k].values()),
                       "n_out_max": max(r["n_out"] for k in by for r in by[k].values()),
                       "any_length_stop": any(r["finish_reason"] == "length" for k in by for r in by[k].values())}
    return out


def gen_dict(g: Gen) -> dict:
    return {"text": g.text, "n_in": g.n_in, "n_out": g.n_out, "finish_reason": g.finish_reason}


def unrelated_long(tokenizer: Any, base: dict, target_tokens: int, n_ctx: int = 8192) -> dict:
    """A call at least target + 320 tokens long (D47 check): `base`'s messages extended with copies of its last turn pair
    (any content: this is not a canary prompt), within the context."""
    msgs = [dict(m) for m in base["messages"]]
    tail = [dict(m) for m in base["messages"][-2:]]
    while tokenizer.count_messages(msgs) < target_tokens + 320:
        msgs += [dict(m) for m in tail]
    while tokenizer.count_messages(msgs) + 64 + 4 > n_ctx and len(msgs) > 2:
        msgs = msgs[:-2]
    return {"messages": msgs, "kind": "action", "grammar": None, "max_tokens": 64}


def d47_check(backend: LlamaCppBackend, fresh_backend: Callable[[], LlamaCppBackend], prompts: list[dict],
              tokenizer: Any) -> dict:
    """For each prompt, through LlamaCppBackend.generate on both sides: the reset path's output after a longer unrelated
    call and after a grammar call vs the first output of a freshly constructed model (D47 on the GPU). Each fresh model
    is closed after use, so at most two models exist."""
    res, after = [], []
    others = sorted(prompts, key=lambda p: -p.get("n_tokens", 0))
    for p in prompts:
        base = next((q for q in others if q is not p), p)
        u = unrelated_long(tokenizer, base, tokenizer.count_messages(p["messages"]))
        backend.generate(u["messages"], u["kind"], u["grammar"], u["max_tokens"])
        a = gen_dict(backend.generate(p["messages"], p["kind"], p["grammar"], p["max_tokens"]))
        backend.generate(p["messages"], "action", "envelope_cap300", 64)
        b = gen_dict(backend.generate(p["messages"], p["kind"], p["grammar"], p["max_tokens"]))
        after.append((a, b, tokenizer.count_messages(u["messages"])))
    for p, (a, b, n_unrel) in zip(prompts, after):
        fb = fresh_backend()
        try:
            f = gen_dict(fb.generate(p["messages"], p["kind"], p["grammar"], p["max_tokens"]))
        finally:
            close = getattr(fb.llm, "close", None)
            if close:
                close()
            del fb
            import gc
            gc.collect()
        res.append({"n_tokens": p.get("n_tokens"), "unrelated_tokens": n_unrel,
                    "after_unrelated_equal": a == f, "after_grammar_equal": b == f})
    return {"prompts": res, "all_equal": all(r["after_unrelated_equal"] and r["after_grammar_equal"] for r in res)}


def timing_check(records: list[dict]) -> dict:
    """Measured seconds vs t_llm's prediction (records: n_in, n_out, grammar, seconds); flagged outside 1 +/- 0.15."""
    ratios = [r["seconds"] / max(1e-9, t_llm(Gen("", r["n_in"], r["n_out"], "stop", r["grammar"]))) for r in records]
    med = statistics.median(ratios) if ratios else None
    return {"n": len(ratios), "median_ratio": med, "sd_ratio": statistics.pstdev(ratios) if len(ratios) > 1 else None,
            "min_ratio": min(ratios) if ratios else None, "max_ratio": max(ratios) if ratios else None,
            "flag": med is not None and abs(med - 1) > TIMING_TOLERANCE}


def load_params_report(llm: Any, s30_stamp: dict) -> dict:
    lp = load_params_of(llm)
    return {"load_params": lp, "n_ctx_equals_s30": lp.get("n_ctx") == s30_stamp.get("n_ctx"),
            "n_batch_equals_s30": lp.get("n_batch") == s30_stamp.get("n_batch"),
            "threads": {"n_threads": getattr(llm.context_params, "n_threads", None),
                        "n_threads_batch": getattr(llm.context_params, "n_threads_batch", None)}}


def canary_counts(tokenizer: Any, prompts: list[dict]) -> dict:
    real = [tokenizer.count_messages(p["messages"]) for p in prompts]
    stored = [p.get("n_tokens") for p in prompts]
    return {"real": real, "stored": stored, "equal": real == stored}


def library_versions() -> dict:
    out = {}
    for m in ("sentence_transformers", "transformers", "torch", "huggingface_hub", "numpy"):
        try:
            out[m] = __import__(m).__version__
        except Exception:      # noqa: BLE001
            out[m] = None
    return out


def minilm_revision() -> str | None:
    """The current commit sha of all-MiniLM-L6-v2 on the Hub (session 1; pinned afterwards)."""
    try:
        from huggingface_hub import model_info
        return model_info("sentence-transformers/all-MiniLM-L6-v2").sha
    except Exception:          # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------- session 2: smoke
def smoke_plan():
    """Spec 4.2: the reference first, then RQ1 C2 (kill-(a) target), RQ1 C0 (kill-(b) target), RQ3."""
    from .runner import Unit
    ref = [Unit("smoke-ref", "rq1", "dev", "rq1", 1000, "C2", n_tasks=8)]
    units = [Unit("smoke", "rq1", "dev", "rq1", 1000, "C2", n_tasks=8),
             Unit("smoke", "rq1", "dev", "rq1", 1000, "C0", n_tasks=8),
             Unit("smoke", "rq3", "dev", "rq3", 5000, "C2", "acc")]
    return ref, units


class KillFailed(BaseException):
    """The runtime was not killed (unassign returned and the VM is still alive after the wait): the user must use
    Runtime -> Disconnect and delete runtime. A BaseException, so the runner's failure policy never retries past it."""


def colab_kill(wait: float = 300.0) -> None:
    """Kill the runtime from code (D140). `unassign()` returns before the VM dies, so this process then BLOCKS (no
    further writes) until the VM is gone; if it is still alive after `wait`, KillFailed tells the user what to do."""
    from google.colab import runtime          # type: ignore
    runtime.unassign()
    t0 = time.time()
    while time.time() - t0 < wait:
        time.sleep(1.0)
    raise KillFailed("the runtime is still alive: use Runtime -> Disconnect and delete runtime, reconnect, run again")


def make_kill(sections: "Sections", name: str, runtime: str, kill: Callable[[], None] = colab_kill) -> Callable[[], None]:
    """A kill that first saves `<name>_requested` (runtime, time) to the session file."""
    def k():
        sections.save(f"{name}_requested", {"runtime": runtime, "t": time.time()})
        kill()
    return k


class KillAfter:
    """Wraps the LLM backend: at the start of fresh call N+1 (so N fresh LLM records are appended) it calls `kill`
    (`colab_kill` on the T4; tests inject an exception)."""

    def __init__(self, inner: Any, n: int, kill: Callable[[], None]):
        self.inner, self.n, self.k, self.kill = inner, n, 0, kill
        self.id = getattr(inner, "id", type(inner).__name__)
        for name in ("cache_key", "register_tasks", "stamp"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    @property
    def busy(self) -> bool:
        return bool(getattr(self.inner, "busy", False))

    def generate(self, *a, **k):
        if self.k >= self.n:
            self.kill()
            raise KillFailed("kill() returned")         # a BaseException: never retried by the runner
        self.k += 1
        return self.inner.generate(*a, **k)


def arm_kill_a(runner: Any, unit: Any, sections: "Sections | None" = None, runtime: str = "") -> bool:
    """Kill (a): armed once (D141). The arming is saved (with the runtime) before the run; if it was armed earlier and
    the target's cache is now missing, the loss is recorded and the kill is not armed again."""
    if sections is not None and sections.done("kill_a_armed"):
        if not runner.cache_path(unit).exists() and not sections.done("kill_a_lost"):
            sections.save("kill_a_lost", {"runtime": runtime, "what": "the whole cache file is missing on Drive"})
        return False
    if runner.cache_path(unit).exists():
        return False
    if sections is not None:
        sections.save("kill_a_armed", {"runtime": runtime, "t": time.time()})
    return True


def arm_kill_b(runner: Any, unit: Any, kill: Callable[[], None], sections: "Sections | None" = None,
               runtime: str = "") -> bool:
    """Kill (b): armed once (D141), only if the unit has no result yet; kills right after its result file is written.
    A result missing after an earlier arming is recorded as lost and not re-armed."""
    if sections is not None and sections.done("kill_b_armed"):
        if not runner.result_path(unit).exists() and not sections.done("kill_b_lost"):
            sections.save("kill_b_lost", {"runtime": runtime, "what": "the result file is missing on Drive"})
        runner.after_result = None
        return False
    if runner.result_path(unit).exists():
        return False
    if sections is not None:
        sections.save("kill_b_armed", {"runtime": runtime, "t": time.time()})

    def hook(u):
        if u.unit_id == unit.unit_id:
            kill()
    runner.after_result = hook
    return True


def drive_state(cache_path: Path, result_path: Path | None = None) -> dict:
    """What reached Drive, read WITHOUT writing (UnitCache would rewrite a torn tail on open): bytes, records with a
    valid crc, whether the last line is torn, and for a result: present, its recorded cache_bytes vs the file size."""
    cp = Path(cache_path)
    raw = cp.read_bytes() if cp.exists() else b""
    n, torn = 0, bool(raw) and not raw.endswith(b"\n")
    for ln in (read_lines(cp) if raw else []):
        if ln is None:
            torn = True
            break
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            torn = True
            break
        if obj.get("type") is None:
            if obj.get("crc") != crc(obj):
                torn = True
                break
            n += 1
    out: dict[str, Any] = {"cache_bytes": len(raw), "records_present": n, "last_line_torn": torn}
    if result_path is not None:
        rp = Path(result_path)
        out["result_present"] = rp.exists()
        if rp.exists():
            try:
                run = json.loads(rp.read_text()).get("run", {})
                out["result_cache_bytes"] = run.get("cache_bytes")
                out["result_cache_records"] = run.get("cache_records")
                out["cache_short"] = len(raw) < (run.get("cache_bytes") or 0)
            except json.JSONDecodeError:
                out["result_unreadable"] = True
    return out


def records_at_kill(ref_cache: Path, n_llm: int) -> int:
    """Records the target had appended when kill (a) fired: everything before its (n_llm+1)-th LLM record, read from the
    deterministic reference unit's cache."""
    seen = 0
    recs = UnitCache(ref_cache, {}, "r", readonly=True).records
    for i, r in enumerate(recs):
        if r["kind"] != "embed":
            if seen == n_llm:
                return i
            seen += 1
    return len(recs)


def compare_units(cache_a: Path, cache_b: Path, payload_a: dict, payload_b: dict) -> dict:
    """Interrupted vs reference (spec 4.3): LLM record sequences must be equal; payload equality (without `unit`) is
    required only when all embedding vectors are equal; otherwise the first differing vector is reported."""
    ra = UnitCache(cache_a, {}, "a", readonly=True).records
    rb = UnitCache(cache_b, {}, "b", readonly=True).records
    keys = ("key", "text", "n_in", "n_out", "finish_reason")
    la = [{k: r[k] for k in keys} for r in ra if r["kind"] != "embed"]
    lb = [{k: r[k] for k in keys} for r in rb if r["kind"] != "embed"]
    ea = [(r["key"], r["vector"]) for r in ra if r["kind"] == "embed"]
    eb = [(r["key"], r["vector"]) for r in rb if r["kind"] == "embed"]
    first_vec = next((i for i, (x, y) in enumerate(zip(ea, eb)) if x != y), None)
    retr = lambda p: [st.get("retrieved") for st in p.get("steps", [])]
    strip = lambda p: {k: v for k, v in p.items() if k != "unit"}
    payload_equal = strip(payload_a) == strip(payload_b)
    emb_equal = ea == eb
    return {"llm_equal": la == lb, "llm_records": [len(la), len(lb)], "embeddings_equal": emb_equal,
            "first_differing_vector": first_vec, "payload_equal": payload_equal,
            "retrieval_equal": retr(payload_a) == retr(payload_b),
            "verdict": "pass" if la == lb and (payload_equal or not emb_equal) else "fail"}


def _quantiles(xs: list[int]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * (len(s) - 1)))]
    return {"min": s[0], "p50": q(0.5), "p90": q(0.9), "max": s[-1]}


def smoke_stats(results: list[dict], cache_of: Callable[[dict], Path], failures: list[dict] | None = None) -> dict:
    """Per unit: inputs of the post-pilot budget projection and the smoke checks (spec 4.5)."""
    out = {}
    fails = failures or []
    for r in results:
        u = r["unit"]
        key = f"{u['phase']}-{u['rq']}-{u['condition']}" + (f"-{u['window']}" if u.get("window") else "")
        recs = UnitCache(cache_of(u), {}, "s", readonly=True).records
        llm = [x for x in recs if x["kind"] != "embed"]
        actions = [x for x in llm if x["kind"] == "action"]
        tasks = r.get("tasks", [])
        out[key] = {"llm_calls": len(llm), "tasks": len(tasks), "calls_per_task": len(llm) / max(1, len(tasks)),
                    "fallback_calls": sum(1 for x in actions if x["grammar"]),
                    "fallback_rate": sum(1 for x in actions if x["grammar"]) / max(1, len(actions)),
                    "truncated_actions": sum(1 for x in actions if x["finish_reason"] == "length"),
                    "n_in": _quantiles([x["n_in"] for x in llm]), "n_out": _quantiles([x["n_out"] for x in llm]),
                    "retok_diff_calls": sum(1 for x in llm if any(str(f).startswith("retok_diff")
                                                                  for f in x.get("flags") or [])),
                    "embed_records": len(recs) - len(llm), "seconds_llm": round(sum(x.get("seconds", 0) for x in llm), 1),
                    "format_errors": sum(t.get("format_errors", 0) for t in tasks),
                    "drift_errors": r.get("drift_errors", sum(t.get("drift_errors", 0) for t in tasks)),
                    "transient_notices": sum(1 for c in r.get("llm_calls", []) if c.get("transient")),
                    "control_token_firings": sum(1 for f in fails if "control" in f.get("error", "")
                                                 and f.get("unit") == r.get("unit_id", _uid(u))),
                    "run": r.get("run")}
    return out


def _uid(u: dict) -> str:
    from .runner import Unit
    return Unit.from_json(u).unit_id


def first_full_wall(root: Path, unit_id: str) -> tuple[float | None, str]:
    """The wall seconds of the unit's first run with no replayed calls (a repair's wall is mostly replay)."""
    from .fsio import read_jsonl
    for f in sorted((Path(root) / "ledger").glob("*.jsonl")):
        for e in read_jsonl(f):
            if e.get("event") == "unit" and e.get("unit") == unit_id and e.get("calls_replayed", 1) == 0:
                return e["wall_seconds"], f"ledger {f.name}"
    return None, "no uninterrupted run in the ledger"


def append_latency(path: Path, n: int = 20) -> dict:
    """Append + fsync latency per record on Drive (a scratch file next to the caches)."""
    from .fsio import append_line
    secs = []
    for i in range(n):
        t0 = time.perf_counter()
        append_line(path, json.dumps({"i": i, "pad": "x" * 2000}))
        secs.append(time.perf_counter() - t0)
    try:
        os.remove(path)
    except OSError:
        pass
    return {"n": n, "median_s": statistics.median(secs), "max_s": max(secs)}


# ------------------------------------------------------------------------------------------- D53 benchmark
def benchmark_workload(caches: list[Path], n: int = 60, seed: int = 0) -> list[dict]:
    """Up to n distinct recorded LLM calls (by key) from the smoke caches, deterministic, stratified by (kind, grammar);
    empty strata are simply absent."""
    pool: dict[tuple, dict[str, dict]] = {}
    for cp in sorted(caches, key=str):
        for r in UnitCache(cp, {}, "w", readonly=True).records:
            if r["kind"] == "embed" or not r.get("prompt_z"):
                continue
            pool.setdefault((r["kind"], str(r["grammar"])), {}).setdefault(r["key"], {
                "prompt": zunpack(r["prompt_z"]), "kind": r["kind"], "grammar": r["grammar"],
                "max_tokens": r["max_tokens"], "n_in": r["n_in"], "text": r["text"], "n_out": r["n_out"]})
    rng = random.Random(seed)
    total = sum(len(v) for v in pool.values())
    if not total:
        raise ValueError("no recorded LLM calls in the given caches")
    keys = sorted(pool)
    alloc = {k: min(len(pool[k]), max(1, round(n * len(pool[k]) / total))) for k in keys}
    while sum(alloc.values()) > n:                       # trim the largest strata, never below one per stratum
        big = max((k for k in keys if alloc[k] > 1), key=lambda k: alloc[k], default=None)
        if big is None:
            break
        alloc[big] -= 1
    out: list[dict] = []
    for key in keys:
        items = [pool[key][k] for k in sorted(pool[key])]
        out += rng.sample(items, alloc[key])
    rng.shuffle(out)
    return out


class TokenRunner:
    """Runs a recorded rendered prompt from its tokens through the backend's production pieces (measurement only):
    fresh path (`reset`) or the snapshot path (`backend._load_prefix_state`); reflection records always use the fresh
    path (as `LlamaCppBackend` does); stop and stopping criteria are the formatter's."""

    def __init__(self, backend: LlamaCppBackend):
        self.b, self.llm = backend, backend.llm
        self.eos = self.llm.token_eos()
        r = backend.formatter(messages=[{"role": "user", "content": "x"}])
        self.stop = [] if r.stop is None else (r.stop if isinstance(r.stop, list) else [r.stop])
        self.criteria = r.stopping_criteria

    def tokens(self, prompt: str) -> list[int]:
        return self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    def run(self, item: dict, snapshot: bool) -> dict:
        toks = self.tokens(item["prompt"])
        if "n_in" in item and len(toks) != item["n_in"]:
            raise AssertionError(f"recorded n_in {item['n_in']} != {len(toks)} tokens")
        g, _ = self.b.grammar(item["grammar"])
        prefix = None
        if snapshot and item["kind"] != "reflection" and self.eos in toks:
            prefix = toks[:toks.index(self.eos) + 1]
        n_states = len(self.b._states)
        t0 = time.perf_counter()                        # includes reset / state load, as production generate() pays
        if prefix is not None:
            self.b._load_prefix_state(prefix)
        else:
            self.llm.reset()
        out = self.llm.create_completion(prompt=toks, max_tokens=item["max_tokens"], stop=self.stop,
                                         stopping_criteria=self.criteria, grammar=g, **SAMPLING)
        sec = time.perf_counter() - t0
        return {"seconds": sec, "text": out["choices"][0]["text"], "n_in": len(toks),
                "n_out": out["usage"]["completion_tokens"], "snapshot": prefix is not None,
                "created_state": prefix is not None and len(self.b._states) > n_states}


def benchmark_ab(backend: LlamaCppBackend, workload: list[dict], telemetry: Callable[[], dict] = gpu_telemetry) -> dict:
    """A (fresh) and B (snapshot): one warm-up, A/B order alternating per prompt, B repeated for repeatability; A must
    reproduce the recorded smoke output; GPU telemetry every 10 prompts."""
    tr = TokenRunner(backend)
    tr.run(workload[0], False)                          # warm-up (not recorded)
    a, b, tele = [], [], []
    repeat_equal = a_equals_recorded = 0
    for i, item in enumerate(workload):
        if i % 10 == 0:
            tele.append({"i": i, **telemetry()})
        res = {}
        for snap in ((False, True) if i % 2 == 0 else (True, False)):
            res[snap] = tr.run(item, snap)
        a.append(res[False])
        b.append(res[True])
        again = tr.run(item, True)
        repeat_equal += (again["text"], again["n_out"]) == (res[True]["text"], res[True]["n_out"])
        a_equals_recorded += (res[False]["text"], res[False]["n_out"]) == (item["text"], item["n_out"])
    sa, sb = sum(x["seconds"] for x in a), sum(x["seconds"] for x in b)
    return {"n": len(workload), "seconds_A": round(sa, 2), "seconds_B": round(sb, 2),
            "saving_B": round(1 - sb / sa, 4) if sa else None, "B_repeatable": repeat_equal == len(workload),
            "B_used_snapshot": sum(x["snapshot"] for x in b), "B_state_creations": sum(x["created_state"] for x in b),
            "A_equals_recorded": a_equals_recorded, "A_all_equal_recorded": a_equals_recorded == len(workload),
            "A_equals_B_outputs": sum(x["text"] == y["text"] for x, y in zip(a, b)),
            "per_call_A": [round(x["seconds"], 3) for x in a], "per_call_B": [round(x["seconds"], 3) for x in b],
            "telemetry": tele}


def rss_bytes() -> int | None:
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return None


def snapshot_ram_probe(backend: LlamaCppBackend, system_prompts: list[str], user: str = "x") -> dict:
    """Host RSS with the snapshot LRU filled by distinct system prefixes, and the first eval at the prefix length."""
    drop_snapshot_states(backend)
    rss0, steps, first = rss_bytes(), [], []
    for sp in system_prompts:
        backend.generate([{"role": "system", "content": sp}, {"role": "user", "content": user}], "action", None, 4)
        steps.append(rss_bytes())
    orig = backend.llm.eval

    def spy(tokens):
        first.append((backend.llm.n_tokens, len(tokens)))
        return orig(tokens)
    msgs = [{"role": "system", "content": system_prompts[-1]}, {"role": "user", "content": user}]
    prefix = backend._prefix(msgs)
    backend.llm.eval = spy
    try:
        backend.generate(msgs, "action", None, 4)
    finally:
        backend.llm.eval = orig
    return {"rss_before": rss0, "rss_after_each_state": steps, "states": len(backend._states),
            "first_eval_at_prefix": bool(first) and first[0][0] == len(prefix), "prefix_tokens": len(prefix)}


def drop_snapshot_states(backend: LlamaCppBackend) -> None:
    backend._states.clear()
    import gc
    gc.collect()


# ------------------------------------------------------------------------------------------- session wiring
class T4Session:
    """Everything a session-2 runtime needs, built in one place: backend, tokenizer, embedder, canary + epoch, header,
    ledger, lease (foreign claims broken once held), claims, the session heartbeat and the runner."""

    def __init__(self, root: Path, llm: Any, part_sha256s: dict[str, str], wheel_sha256: str, embedder: Any,
                 session_id: str, load_seconds: float | None = None, snapshot: bool = False, register: bool = True,
                 heartbeat_seconds: float = 60.0, log: Callable[[str], None] = print, check_counts: bool = True):
        from .canary import EpochStore, Session, fingerprint, load_reference_prompts, run_canary, session_env
        from .ledger import Ledger, runtime_id
        from .llama_backend import LlamaTokenizer
        from .runner import Claims, Heartbeat, Lease, RunContext, Runner, make_header
        self.root = Path(root)
        self.env = session_env()
        self.backend = LlamaCppBackend(llm, model_digest(part_sha256s), wheel_sha256, snapshot=snapshot)
        self.tokenizer = LlamaTokenizer(self.backend)
        self.embedder = embedder
        self.runtime = runtime_id()
        self.ledger = Ledger(self.root, session_id, self.runtime)
        self.ledger.start(load_seconds)
        self.lease = Lease(self.root, self.runtime)
        self.claims = Claims(self.root, self.runtime)
        self.heartbeat = Heartbeat(self.ledger, self.lease, heartbeat_seconds, self.claims).__enter__()
        self.broken_claims = self.claims.break_other_runtimes(self.lease)   # we hold the lease: others are dead
        self.prompts, self.prompts_sha = load_reference_prompts()
        self.counts = canary_counts(self.tokenizer, self.prompts)
        if check_counts and not self.counts["equal"]:
            self.close()
            raise AssertionError(f"committed canary n_tokens differ from real counts: {self.counts}")
        self.path = "snapshot" if snapshot else "fresh"
        t0 = time.time()
        self.canary_outputs = run_canary(self._gen, self.prompts)
        self.canary_seconds = time.time() - t0
        self.fingerprint = fingerprint(self.canary_outputs)
        if register:
            self.epoch, self.status = EpochStore(self.root).resolve(self.path, self.canary_outputs, self.env,
                                                                    self.prompts_sha)
        else:
            self.epoch, self.status = None, "not registered"
        self.ledger.event("canary", seconds=round(self.canary_seconds, 1), epoch=self.epoch, status=self.status,
                          fingerprint=self.fingerprint)
        self.header = make_header(self.backend.stamp(), embedder.id, self.env, self.epoch, self.fingerprint)
        self.session = Session(self.path, self.epoch if self.epoch is not None else -1, self.status, self.fingerprint)
        self.runner = Runner(self.root, RunContext(self.backend, embedder, self.tokenizer, self.header, self.session,
                                                   self._rerun), self.ledger, claims=self.claims,
                             session_heartbeat=self.heartbeat, log=log)

    def _gen(self, p: dict) -> Gen:
        return self.backend.generate(p["messages"], p["kind"], p["grammar"], p["max_tokens"])

    def _rerun(self) -> str:
        from .canary import fingerprint, run_canary
        return fingerprint(run_canary(self._gen, self.prompts))

    def close(self) -> None:
        self.heartbeat.__exit__(None, None, None)


def wording_system_prompts(prompts: list[dict]) -> list[str]:
    """Three distinct real-size system prefixes for the snapshot RAM probe: a canary prompt's system message with each
    docs-wording variant (W1-W3)."""
    from ..agent.prompts import DOCS_WORDING
    base = next(p["messages"][0]["content"] for p in prompts if p["messages"][0]["role"] == "system")
    cur = next(w for w, txt in DOCS_WORDING.items() if txt in base)
    return [base.replace(DOCS_WORDING[cur], DOCS_WORDING[w]) for w in ("W1", "W2", "W3")]
