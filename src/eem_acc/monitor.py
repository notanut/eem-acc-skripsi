"""Progress snapshot of a running RQ2 phase (reads the per-unit result files).

Pure Python, no plotting: used by the notebook's live monitor and tested locally.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from . import scenarios as S
from .utils.results import _fname


def phase_unit_list(phase: str, frozen: dict | None = None):
    if phase == "dev":
        return S.units(S.dev_methods(), S.DEV_SEEDS)
    if phase == "test":
        if frozen is None:
            raise ValueError("test phase needs the frozen configuration")
        return S.units(S.test_methods(frozen), S.TEST_SEEDS)
    raise ValueError(phase)


def snapshot(root: str | Path, phase: str, frozen: dict | None = None, now: float | None = None,
             rate_window: int = 300) -> dict[str, Any]:
    """Done/total overall and per (scenario, M) cell, recent throughput and ETA."""
    units = phase_unit_list(phase, frozen)
    total = len(units)
    udir = Path(root) / "results" / "rq2" / phase / "units"
    stamps: dict[str, float] = {}
    if udir.exists():
        with os.scandir(udir) as it:
            for e in it:
                if e.name.endswith(".json"):
                    stamps[e.name[:-5]] = e.stat().st_mtime
    cell_total: dict[tuple[str, int], int] = {}
    cell_done: dict[tuple[str, int], int] = {}
    weight_of: dict[str, int] = {}
    done = 0
    done_w = total_w = 0
    for uid, scn, m, _meth, _seed in units:
        key = (scn, m)
        fn = _fname(uid)[:-5]
        weight_of[fn] = m
        total_w += m
        cell_total[key] = cell_total.get(key, 0) + 1
        if fn in stamps:
            cell_done[key] = cell_done.get(key, 0) + 1
            done += 1
            done_w += m
    now = time.time() if now is None else now
    recent_items = sorted(stamps.items(), key=lambda kv: kv[1])[-rate_window:]
    recent = [t for _, t in recent_items]
    rate = None
    rate_w = None
    if len(recent) >= 2 and recent[-1] > recent[0]:
        span = recent[-1] - recent[0]
        rate = (len(recent) - 1) / span                               # runs per second
        # Run time grows roughly with the number of agents M (calls = 150 x M), and the units are
        # ordered light cells first, so a plain runs/s extrapolation is far too optimistic early on.
        # ETA therefore uses M-weighted work: remaining weight / recent weighted throughput.
        rate_w = sum(weight_of.get(name, 0) for name, _ in recent_items[1:]) / span
    idle = (now - recent[-1]) if recent else None
    running = bool(recent) and idle is not None and idle < 60 and done < total
    remaining_w = total_w - done_w
    eta = remaining_w / rate_w if rate_w and running else None
    return {"phase": phase, "done": done, "total": total, "cell_done": cell_done, "cell_total": cell_total,
            "rate_per_s": rate, "eta_s": eta, "seconds_since_last_result": idle, "running": running}


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def summary_line(snap: dict[str, Any]) -> str:
    pct = 100.0 * snap["done"] / max(snap["total"], 1)
    state = "running" if snap["running"] else ("finished" if snap["done"] >= snap["total"] else "not running / idle")
    rate = f"{snap['rate_per_s']:.1f} runs/s" if snap["rate_per_s"] else "-"
    return (f"[{snap['phase']}] {snap['done']}/{snap['total']} runs ({pct:.1f}%) | {state} | {rate} | "
            f"ETA ~{fmt_duration(snap['eta_s'])} (rough)")
