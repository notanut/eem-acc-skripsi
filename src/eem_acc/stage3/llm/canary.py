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


def canary_generate(backend: Any, p: dict) -> Any:
    """One canary prompt on the backend (fresh generation)."""
    return backend.generate(p["messages"], p["kind"], p["grammar"], p["max_tokens"])


def rerun_fingerprint(backend: Any, prompts: list[dict]) -> str:
    """The canary re-run that a resume mismatch compares (D61): always freshly generated."""
    return fingerprint(run_canary(lambda p: canary_generate(backend, p), prompts))


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

    def on_resume_mismatch(self, rerun_canary: Callable[[], str]) -> str:
        """D61/D66/D131. Returns "restart" (the unit restarts; the trigger unit always does). On the second mismatch the
        canary is re-run and compared with THIS session's fingerprint (never registered as an epoch: GPU, driver and
        wheel cannot change inside a running process); a difference means a nondeterministic platform and halts."""
        if not self.fingerprint:
            raise ValueError("Session.fingerprint must be set from the session canary")
        self.mismatches += 1
        self.events.append({"event": "resume_mismatch", "n": self.mismatches})
        if self.mismatches >= 2 and not self.replay_only_mode:
            try:
                fp = rerun_canary()
            except Exception as e:     # noqa: BLE001 - e.g. CanaryRepeatMismatch: no unit may run (fix review M2)
                self.events.append({"event": "canary_rerun_failed", "error": f"{type(e).__name__}: {e}"})
                raise SessionNondeterministic(f"the canary re-run failed: {type(e).__name__}: {e}") from e
            if fp != self.fingerprint:
                self.events.append({"event": "canary_differs_within_session", "fingerprint": fp})
                raise SessionNondeterministic(f"the canary differs from this session's start ({fp[:12]} vs "
                                              f"{self.fingerprint[:12]})")
            self.replay_only_mode = True
            self.events.append({"event": "long_prompt_nondeterminism"})
        return "restart"


class SessionNondeterministic(RuntimeError):
    """The canary re-run differs within one session (D131): no further unit runs in this session; alert the user."""


def cpu_model() -> str | None:
    """Recorded (not compatibility-checked): embeddings of a unit resumed on another CPU type mix two CPUs."""
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or None


def session_env() -> dict:
    """Environment recorded with an epoch (GPU via nvidia-smi when present)."""
    import platform
    import subprocess

    import numpy
    import time
    from pathlib import Path
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "cpu_model": cpu_model()}
    env["gpu"], env["driver"] = "none", "none"
    for attempt in range(3):             # S3.4 7th opinion L2: a failed query must not record "none" on a GPU runtime
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
            fields = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")] \
                if r.returncode == 0 and r.stdout.strip() else []
        except (OSError, subprocess.SubprocessError):
            fields = []
        import re
        if len(fields) == 2 and fields[0] and re.fullmatch(r"\d+(\.\d+)+", fields[1]):   # name, driver version:
            #                                    an error text is never taken for a GPU name (review r7 L3)
            env["gpu"], env["driver"] = fields
            break
        if not Path("/dev/nvidia0").exists():
            break                                                       # no GPU runtime: "none" is the truth
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError("nvidia-smi returned nothing three times on a GPU runtime: restart the runtime")
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

