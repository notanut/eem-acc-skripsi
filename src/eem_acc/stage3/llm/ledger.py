"""Budget ledger (S3.3a spec section 7; D52): per process `ledger/<session>-<shard>.jsonl`. GPU-hours = the sum over
runtimes of (last heartbeat - session start): Colab charges runtime wall clock, setup and idle time included; two
processes in one runtime are counted once."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from .fsio import append_line, canon, read_jsonl


class Ledger:
    def __init__(self, root: Path, session: str, runtime: str, shard: str = "0", clock=time.time):
        self.file = Path(root) / "ledger" / f"{session}-{shard}.jsonl"
        self.session, self.runtime, self.shard, self.clock = session, runtime, shard, clock

    def event(self, kind: str, **fields) -> None:
        append_line(self.file, canon({"event": kind, "t": round(self.clock(), 3), "session": self.session,
                                      "runtime": self.runtime, "shard": self.shard, **fields}))

    def start(self, model_load_seconds: float | None = None, vm_boot: float | None = None) -> None:
        """`vm_boot`: the VM's boot time (`/proc/stat` btime on Colab), so setup before the start counts too."""
        if vm_boot is None:
            vm_boot = vm_boot_time()
        self.event("session_start", model_load_seconds=model_load_seconds, vm_boot=vm_boot)

    def heartbeat(self) -> None:
        self.event("heartbeat")


def runtime_id() -> str:
    """hostname + VM boot time: one id per Colab VM, so kernel restarts on the same VM are not double-counted. The boot
    time can precede the runtime assignment on pre-warmed VMs, which over-counts (conservative for the budget gate)."""
    import socket
    return f"{socket.gethostname()}-{int(vm_boot_time() or 0)}"


def vm_boot_time() -> float | None:
    try:
        for ln in Path("/proc/stat").read_text().splitlines():
            if ln.startswith("btime "):
                return float(ln.split()[1])
    except OSError:
        pass
    return None


def summarise(root: Path) -> dict:
    """GPU-hours per runtime and in total; calls fresh vs replayed; discarded calls by reason; seconds per fresh call."""
    starts: dict[str, float] = {}
    last: dict[str, float] = defaultdict(float)
    fresh = replayed = fresh_llm = 0
    seconds = 0.0
    walls: list[float] = []
    discarded: dict[str, int] = defaultdict(int)
    for f in sorted((Path(root) / "ledger").glob("*.jsonl")):
        for e in read_jsonl(f):
            rt = e["runtime"]
            if e["event"] == "session_start":
                t0 = e["t"] if e.get("vm_boot") is None else min(e["t"], e["vm_boot"])
                starts[rt] = min(starts.get(rt, t0), t0)
            last[rt] = max(last[rt], e["t"])
            if e["event"] == "unit":
                fresh += e.get("calls_fresh", 0)
                fresh_llm += e.get("calls_fresh_llm", e.get("calls_fresh", 0))
                replayed += e.get("calls_replayed", 0)
                seconds += e.get("llm_seconds", 0.0)
                walls.append(e.get("wall_seconds", 0.0))
            if e["event"] == "discarded":
                discarded[e["reason"]] += e.get("calls", 0)
    hours = {rt: max(0.0, last[rt] - t0) / 3600 for rt, t0 in starts.items()}
    return {"gpu_hours": round(sum(hours.values()), 4), "per_runtime": hours, "calls_fresh": fresh,
            "calls_replayed": replayed, "calls_fresh_llm": fresh_llm,
            "seconds_per_fresh_call": (seconds / fresh_llm) if fresh_llm else None,
            "discarded_calls": dict(discarded), "unit_wall_seconds": walls}
