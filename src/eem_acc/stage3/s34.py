"""S3.4: the automatic pilot chain (docs/s34_spec.md r2; D154-D164).

Steps A (docs wording) -> B (pilot: switches, acceptance, ladder) -> C (RQ3 wiring) -> D (budget gate) -> E (freeze)
-> F (EEM selection, D60 rounds H) -> D again (gate after the selection) -> G (done).

Everything here is deterministic and free of Colab: the rule functions are pure (they take per-unit metrics), and the
`Chain` drives the steps through two callbacks supplied by the notebook (`run_units`: run a phase's plan and return
its results; `metrics`: a result's per-unit metrics). The chain's state is an append-only log in a `t4.Sections` file
(`chain.json`); every recorded decision is recomputed from the stored results when a session resumes and must agree.
"""
from __future__ import annotations

import dataclasses
import hashlib
from fractions import Fraction
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from .llm.fsio import canon
from .llm.runner import Unit, shuffled

WORDING_SEEDS = (1000, 1001)
PILOT_SEEDS = (1002, 1003, 1004, 1005, 1006)
SELECTION_SEEDS = (1007, 1008, 1009, 1010)
RQ3_SEED = 5001
WORDINGS = ("W1", "W2", "W3")
CONFIGS = {"cfg1": {"k": 3, "k_s": 2, "theta_link": 0.5}, "cfg2": {"k": 3, "k_s": 2, "theta_link": 0.65},
           "cfg3": {"k": 3, "k_s": 2, "theta_link": 0.8}, "cfg4": {"k": 5, "k_s": 3, "theta_link": 0.5},
           "cfg5": {"k": 5, "k_s": 3, "theta_link": 0.65}, "cfg6": {"k": 5, "k_s": 3, "theta_link": 0.8}}
DEFAULT_CFG = "cfg2"
BUDGET_HOURS = 60.0
TASK_COUNTS = (19, 18, 17, 16)            # D160
LADDER = {"hard": (("long_share", -0.1), ("n_distractors", 1), ("n_distractors", 0)),
          "easy": (("long_share", +0.1), ("n_distractors", 3), ("n_distractors", 4))}      # D157
MAX_D60_ROUNDS = 2


class ChainPause(Exception):
    """The chain stops and asks the user (spec section 11)."""


class ChainHalt(RuntimeError):
    """A recomputed decision differs from the recorded one: a bug (spec section 2)."""


# ------------------------------------------------------------------------------------------------- settings
@dataclasses.dataclass(frozen=True)
class Settings:
    wording: str = "W1"
    grammar_always: bool = False
    action_max_tokens: int = 320
    n_distractors: int = 2
    long_share: float | None = None           # None = the natural share (uniform template weights)

    def params(self, extra: dict | None = None) -> tuple:
        p = {"docs_wording": self.wording}
        if self.grammar_always:
            p["grammar_always"] = True
        if self.action_max_tokens != 320:
            p["action_max_tokens"] = self.action_max_tokens
        p.update(extra or {})
        return tuple(sorted(p.items()))

    def stream(self) -> tuple:
        s = {}
        if self.n_distractors != 2:
            s["n_distractors"] = self.n_distractors
        if self.long_share is not None:
            s["long_share"] = self.long_share
        return tuple(sorted(s.items()))

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def natural_long_share() -> float:
    from .apisuite.templates import TEMPLATES
    return round(sum(1 for t in TEMPLATES.values() if t.long) / len(TEMPLATES), 6)


# ---------------------------------------------------------------------------------------------------- units
def phase_name(step: str, rnd: int, version: int = 1) -> str:
    """Chain version 1: s34-<step>-r<n>; a new chain version after a code fix (spec section 2): s34v<k>-<step>-r<n>, so
    its plans, units and caches never collide with the old version's (kept as evidence)."""
    return f"s34-{step}-r{rnd}" if version == 1 else f"s34v{version}-{step}-r{rnd}"


def chain_file(version: int = 1) -> str:
    return "chain.json" if version == 1 else f"chain-v{version}.json"


def freeze_file(version: int = 1) -> str:
    return "freeze.json" if version == 1 else f"freeze-v{version}.json"


def wording_units(rnd: int, s: Settings, version: int = 1) -> list[Unit]:
    out = []
    phase = phase_name("A", rnd, version)
    for seed in WORDING_SEEDS:
        for w in shuffled(list(WORDINGS), phase, f"rq1-{seed}", 1):
            out.append(Unit(phase, "rq1", "dev", "rq1", seed, "C0",
                            params=dataclasses.replace(s, wording=w).params(), tag=w, stream=s.stream()))
    return out


def pilot_units(phase: str, s: Settings, conditions=("C0", "C2")) -> list[Unit]:
    out = []
    for seed in PILOT_SEEDS:
        for c in shuffled(list(conditions), phase, f"rq1-{seed}", 1):
            out.append(Unit(phase, "rq1", "dev", "rq1", seed, c, params=s.params(), stream=s.stream()))
    return out


def rq3_units(phase: str, s: Settings) -> list[Unit]:
    return [Unit(phase, "rq3", "dev", "rq3", RQ3_SEED, c, w, params=s.params(), stream=s.stream())
            for c, w in (("C2", "acc"), ("C0", "none"))]


def selection_units(phase: str, s: Settings, seeds: tuple, n_tasks: int, cfgs=tuple(CONFIGS)) -> list[Unit]:
    out = []
    for seed in seeds:
        for cfg in shuffled(list(cfgs), phase, f"rq1-{seed}", 1):
            out.append(Unit(phase, "rq1", "dev", "rq1", seed, "C2", params=s.params(CONFIGS[cfg]), tag=cfg,
                            n_tasks=n_tasks, stream=s.stream()))
    return out


# --------------------------------------------------------------------------------------------------- metrics
def unit_metrics(result: dict, records: list[dict]) -> dict:
    """Per-unit inputs of every rule (spec sections 5-9). `records`: the unit's cache records."""
    u = result["unit"]
    tasks = result.get("tasks", [])
    llm = [r for r in records if r.get("kind") != "embed"]
    actions = [r for r in llm if r.get("kind") == "action"]
    grammar_always = bool(dict(tuple(p) for p in u.get("params", ())).get("grammar_always"))
    first = actions if grammar_always else [r for r in actions if not r.get("grammar")]   # no fallback in that mode
    refl = [r for r in llm if r.get("kind") == "reflection"]
    run = result.get("run", {})
    wall = float(run.get("wall_seconds") or 0.0)
    fresh, rep = run.get("calls_fresh") or 0, run.get("calls_replayed") or 0
    if fresh and rep:
        cost = wall * (fresh + rep) / fresh                 # a resumed run replays instantly: scale to all calls
    elif fresh or not rep:
        cost = wall
    else:
        cost = None                                         # fully replayed (a repair): its wall says nothing
    return {"unit_id": Unit.from_json(u).unit_id, "rq": u["rq"], "condition": u["condition"], "window": u.get("window"),
            "tag": u.get("tag", ""), "seed": u["seed"],
            "tasks": [{"pos": t.get("task", i), "success": bool(t.get("success")), "api_calls": t.get("api_calls", 0),
                       "tok_in": t.get("tok_in", 0), "unfinished": bool(t.get("unfinished_at_t_max"))}
                      for i, t in enumerate(tasks)],
            "first_actions": len(first), "fallbacks": len(actions) - len(first), "action_replies": len(actions),
            "truncated_actions": sum(1 for r in actions if r.get("finish_reason") == "length"),
            "reflections": len(refl), "reflections_at_cap": sum(1 for r in refl if (r.get("n_out") or 0) >= 60),
            "format_errors": sum(t.get("format_errors", 0) for t in tasks),
            "overflows": sum(1 for t in tasks if t.get("reason") == "context_overflow"),
            "llm_calls": len(llm), "cost_seconds": None if cost is None else round(cost, 3),
            "status": result.get("status", "done")}


def _sel(ms: list[dict], **kw) -> list[dict]:
    return [m for m in ms if all(m.get(k) == v for k, v in kw.items())]


def _succ(ms: list[dict]) -> tuple[int, int]:
    return sum(t["success"] for m in ms for t in m["tasks"]), sum(len(m["tasks"]) for m in ms)


def _solved(ms: list[dict]) -> set:
    return {(m["seed"], t["pos"]) for m in ms for t in m["tasks"] if t["success"]}


def _mean_api(ms: list[dict], keys: set) -> float:
    xs = [t["api_calls"] for m in ms for t in m["tasks"] if (m["seed"], t["pos"]) in keys]
    return round(sum(xs) / len(xs), 9) if xs else math.inf


def condition_summary(ms: list[dict]) -> dict:
    s, n = _succ(ms)
    fa, fb = sum(m["first_actions"] for m in ms), sum(m["fallbacks"] for m in ms)
    ar, tr = sum(m["action_replies"] for m in ms), sum(m["truncated_actions"] for m in ms)
    rf, rc = sum(m["reflections"] for m in ms), sum(m["reflections_at_cap"] for m in ms)
    return {"units": len(ms), "tasks": n, "solved": s, "success": s / n if n else None,
            "first_actions": fa, "fallbacks": fb, "fallback_rate": fb / fa if fa else 0.0,
            "action_replies": ar, "truncated": tr, "truncation_rate": tr / ar if ar else 0.0,
            "reflections_at_cap_share": rc / rf if rf else 0.0,
            "format_errors": sum(m["format_errors"] for m in ms), "overflows": sum(m["overflows"] for m in ms),
            "llm_calls_per_task": sum(m["llm_calls"] for m in ms) / n if n else None,
            "api_calls_per_task": sum(t["api_calls"] for m in ms for t in m["tasks"]) / n if n else None,
            "seconds_per_task": _per_task_cost(ms)}


def _per_task_cost(ms: list[dict]) -> float | None:
    timed = [m for m in ms if m.get("cost_seconds") is not None]
    n = sum(len(m["tasks"]) for m in timed)
    return sum(m["cost_seconds"] for m in timed) / n if n else None


# ----------------------------------------------------------------------------------------------------- rules
def choose_wording(ms: list[dict]) -> dict:
    """Spec section 4 (D156): highest C0 success; ties -> fewer mean API calls on tasks solved by every tied variant;
    still tied (or nothing jointly solved) -> W1 if tied, else the lowest-numbered tied variant."""
    by = {w: _sel(ms, tag=w) for w in WORDINGS}
    solved = {w: _succ(by[w])[0] for w in WORDINGS}
    best = max(solved.values())
    tied = [w for w in WORDINGS if solved[w] == best]
    step = "success"
    if len(tied) > 1:
        joint = set.intersection(*[_solved(by[w]) for w in tied])
        if joint:
            api = {w: _mean_api(by[w], joint) for w in tied}
            lo = min(api.values())
            tied = [w for w in tied if api[w] == lo]
            step = "api_calls"
    choice = tied[0] if len(tied) == 1 else ("W1" if "W1" in tied else sorted(tied)[0])
    if len(tied) > 1:
        step = "default"
    return {"wording": choice, "solved": solved, "decided_by": step}


def switch_check(c0: list[dict], c2: list[dict]) -> dict:
    """Design section 6 / spec section 5: grammar always if fallback > 5% pooled or |C0 - C2| > 3 pp; cap 640 if
    truncation of action replies > 1% in either condition."""
    a, b = condition_summary(c0), condition_summary(c2)
    fr = lambda x, y: Fraction(x, y) if y else Fraction(0)            # exact: no float edge at 5%, 3 pp, 1% (M1)
    fa = a["first_actions"] + b["first_actions"]
    pooled = fr(a["fallbacks"] + b["fallbacks"], fa)
    f0, f2 = fr(a["fallbacks"], a["first_actions"]), fr(b["fallbacks"], b["first_actions"])
    t0, t2 = fr(a["truncated"], a["action_replies"]), fr(b["truncated"], b["action_replies"])
    grammar = pooled > Fraction(5, 100) or abs(f0 - f2) > Fraction(3, 100)
    cap = t0 > Fraction(1, 100) or t2 > Fraction(1, 100)
    return {"grammar_always": grammar, "cap640": cap, "fallback_pooled": float(pooled),
            "fallback_diff": float(abs(f0 - f2)), "fallback_c0": float(f0), "fallback_other": float(f2),
            "truncation_c0": float(t0), "truncation_other": float(t2)}


def acceptance(c0: list[dict]) -> dict:
    s, n = _succ(c0)
    r = Fraction(s, n) if n else Fraction(0)
    v = "ok" if Fraction(1, 5) <= r <= Fraction(4, 5) else ("hard" if r < Fraction(1, 5) else "easy")
    return {"c0_success": float(r), "verdict": v}


def ladder_step(s: Settings, direction: str, slot: int, natural: float) -> tuple[Settings | None, int, list[int]]:
    """Apply ladder slot `slot` (0-based) in `direction`; a long-share step outside [0, 1] is skipped (its slot is
    used up). Returns (new settings or None if the ladder is exhausted, next slot, skipped slots)."""
    skipped = []
    while slot < 3:
        knob, v = LADDER[direction][slot]
        if knob == "long_share":
            cur = natural if s.long_share is None else s.long_share
            new = round(cur + v, 6)
            if 0.0 <= new <= 1.0:
                return dataclasses.replace(s, long_share=new), slot + 1, skipped
            skipped.append(slot)
            slot += 1
            continue
        return dataclasses.replace(s, n_distractors=v), slot + 1, skipped
    return None, slot, skipped


def wiring_check(ms: list[dict], verify_ok: bool, audit_hits: int) -> dict:
    unfinished = sum(t["unfinished"] for m in ms for t in m["tasks"])
    bad = [m["unit_id"] for m in ms if m.get("status", "done") != "done"]
    sane = not bad and verify_ok and audit_hits == 0 and len(ms) == 2
    return {"unfinished_at_t_max": unfinished, "sane": sane, "bad_units": bad, "verify_ok": verify_ok,
            "audit_hits": audit_hits, "ok": sane and unfinished == 0}


def select_config(ms: list[dict], n_streams: int) -> dict:
    """Spec section 9 (D163): highest success; within one task per stream of the best -> fewer mean API calls on tasks
    solved by all of them -> fewer mean prompt tokens per task -> the default config -> smaller k -> theta closest to
    0.65 -> lower config index."""
    by = {c: _sel(ms, tag=c) for c in CONFIGS if _sel(ms, tag=c)}
    solved = {c: _succ(v)[0] for c, v in by.items()}
    best = max(solved.values())
    cand = [c for c in by if best - solved[c] <= n_streams]
    steps = ["success"]
    joint = set.intersection(*[_solved(by[c]) for c in cand]) if cand else set()
    if len(cand) > 1 and joint:
        api = {c: _mean_api(by[c], joint) for c in cand}
        cand = [c for c in cand if api[c] == min(api.values())]
        steps.append("api_calls")
    if len(cand) > 1:
        tok = {c: round(sum(t["tok_in"] for m in by[c] for t in m["tasks"]) / max(1, _succ(by[c])[1]), 9)
               for c in cand}
        cand = [c for c in cand if tok[c] == min(tok.values())]
        steps.append("prompt_tokens")
    if len(cand) > 1:
        steps.append("default")
        cand = sorted(cand, key=lambda c: (c != DEFAULT_CFG, CONFIGS[c]["k"],
                                           abs(CONFIGS[c]["theta_link"] - 0.65), c))
    return {"config": cand[0], "solved": solved, "within_margin": [c for c in by if best - solved[c] <= n_streams],
            "decided_by": steps[-1]}


# ------------------------------------------------------------------------------------------------ the gate
@dataclasses.dataclass(frozen=True)
class Costs:
    c0: float                     # seconds per task, C0 (final judged pilot round)
    c2: float                     # seconds per task, C2
    rq3: float                    # seconds per LLM task, the larger RQ3 wiring cell (review L1)
    setup_s: float                # per session: setup, model load, canary
    gap_s: float                  # per unit: session time outside units
    unit_hours_per_session: float


def workload(n_tasks: int, cuts: frozenset, selection_left: int, d60_reserve: bool, selected_seeds: int) -> list:
    """(name, units, tasks per unit, cost key) for the remaining work, design sections 7 and 9."""
    desc_t = min(15, n_tasks) if 7 in cuts else n_tasks
    items = []
    if selection_left:
        items.append(("selection", selection_left, n_tasks, "c2"))
    if d60_reserve:
        items += [("d60_selection", selected_seeds, n_tasks, "c2"), ("d60_c0", len(PILOT_SEEDS), 20, "c0"),
                  ("d60_rq3", 2, 20, "rq3")]
    items += [("rq1_C0", 14, n_tasks, "c0"), ("rq1_C1", 14, n_tasks, "c2"), ("rq1_C2", 14, n_tasks, "c2"),
              ("rq1_C3", 14, n_tasks, "c2"), ("C2-OU", 5, n_tasks, "c2"),
              ("R", 3 if 2 in cuts else 5, desc_t, "c0"),
              ("A1", 0 if 10 in cuts else (3 if 3 in cuts else 5), desc_t, "c2"),
              ("A2", 3 if 3 in cuts else 5, desc_t, "c2"),
              ("F", 0 if 10 in cuts else (3 if 4 in cuts else 5), desc_t, "c2"),
              ("AN", 3 if 4 in cuts else 5, desc_t, "c2"),
              ("SP", 2 * (3 if 5 in cuts else 5), desc_t, "sp"),
              ("rq3_2x2", 4 * 10, 20, "rq3"),
              ("rq3_W4", 0 if 9 in cuts else (5 if 1 in cuts else 10), 20, "rq3"),
              ("rq3_private", 0 if 0 in cuts else 5, 20, "rq3"),
              ("rq3_sensitivity", 0 if 8 in cuts else 4 * (3 if 6 in cuts else 5), 20, "rq3")]
    return [i for i in items if i[1] > 0]


def projection(costs: Costs, items: list) -> dict:
    per = {"c0": costs.c0, "c2": costs.c2, "rq3": costs.rq3, "sp": (costs.c0 + costs.c2) / 2}
    unit_h = sum(u * t * per[k] for _, u, t, k in items) / 3600
    units = sum(u for _, u, _, _ in items)
    gap_h = units * costs.gap_s / 3600
    sessions = max(1, math.ceil(unit_h / costs.unit_hours_per_session)) if costs.unit_hours_per_session > 0 else 1
    over_h = sessions * costs.setup_s / 3600
    return {"unit_hours": round(unit_h, 3), "gap_hours": round(gap_h, 3), "sessions": sessions,
            "overhead_hours": round(over_h, 3), "total": round(unit_h + gap_h + over_h, 3),
            "items": [list(i) for i in items]}


def budget_gate(used: float, costs: Costs, cuts: tuple = (), n_tasks: int = 20, selection_done: bool = False,
                d60_reserve: bool = True, selected_seeds: int = 4,
                exposure_ok: Callable[[int], bool] = lambda n: True, budget: float = BUDGET_HOURS) -> dict:
    """Spec section 7: cuts 0-11 in order (only ever added; a cut that saves nothing is skipped), then the task count
    (D160), else a pause (D57)."""
    cuts_set = set(cuts)

    def proj(cs, n):
        sel_seeds = 3 if 11 in cs else 4
        left = 0 if selection_done else len(CONFIGS) * sel_seeds
        return projection(costs, workload(n, frozenset(cs), left, d60_reserve,
                                          min(selected_seeds, sel_seeds)))
    p = proj(cuts_set, n_tasks)
    skipped = []
    for step in range(12):
        if used + p["total"] <= budget:
            break
        if step in cuts_set:
            continue
        q = proj(cuts_set | {step}, n_tasks)
        if q["total"] >= p["total"]:
            skipped.append(step)
            continue
        cuts_set.add(step)
        p = q
    n = n_tasks
    if used + p["total"] > budget:
        for cand in TASK_COUNTS:
            if cand >= n or not exposure_ok(cand):
                continue
            q = proj(cuts_set, cand)
            n, p = cand, q
            if used + q["total"] <= budget:
                break
    fits = used + p["total"] <= budget
    return {"fits": fits, "used": round(used, 3), "projected": p, "costs": dataclasses.asdict(costs),
            "cuts": sorted(cuts_set), "skipped_cuts": skipped,
            "n_tasks": n, "pause": None if fits else "the budget does not fit at 16 tasks per stream (D57, D160)"}


def prefix_exposure_ok(n: int, s: Settings) -> bool:
    """The S3.1 exposure rule on n-task prefixes of the planned test and SP streams, under the knobs (spec 7)."""
    from .apisuite.streams import (COUNT_WINDOW, FIRST_WINDOW, MIN_EXPOSURES, exposures, prefix, sample_stream)
    for kind, split, seeds in (("rq1", "test", range(2000, 2014)), ("sp", "test", range(3000, 3005))):
        for seed in seeds:
            spec = prefix(sample_stream(seed, split, kind=kind, **dict(s.stream())), n)
            for op in spec["drifted_ops"]:
                pos = exposures(spec, op)
                if not (pos and pos[0] < FIRST_WINDOW and sum(p < COUNT_WINDOW for p in pos) >= MIN_EXPOSURES):
                    return False
            if kind == "sp":                    # the re-drifted op needs >= 2 exposures after the midpoint n // 2
                if sum(p >= n // 2 for p in exposures(spec, spec["sp_op"])) < 2:
                    return False
    return True


# ------------------------------------------------------------------------------------------------- the chain
def write_once(path: Path, obj: dict) -> str:
    """Write a JSON record once (atomic); an existing file must hold the same content. Returns its sha256."""
    path = Path(path)
    text = json.dumps(obj, indent=1, sort_keys=True)
    sha = hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        old = path.read_text()
        if hashlib.sha256(old.encode()).hexdigest() != sha:
            raise ChainHalt(f"{path.name} is immutable and differs from what would be written")
        return sha
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
    return sha


class Chain:
    """The S3.4 state machine. `sections`: the chain file (t4.Sections, stamped with commit + nb_version);
    `run_units(phase, units, meta) -> list[result]` runs (or loads) a phase's plan; results of units that are not done
    carry `status`; `metrics(result) -> unit metrics`; `checks(results) -> (verify_ok, audit_hits)`; `used_hours()`;
    `costs_extra()` -> dict(setup_s, gap_s, unit_hours_per_session) from the ledger; `freeze_extra(settings)` -> dict
    of hashes for the freeze record; `version`: the chain version (a new one after a code fix, spec section 2)."""

    def __init__(self, sections, root: Path, run_units: Callable, metrics: Callable, checks: Callable,
                 used_hours: Callable[[], float], costs_extra: Callable[[], dict],
                 freeze_extra: Callable[[Settings], dict] = lambda s: {},
                 exposure_ok: Callable[[int, Settings], bool] = prefix_exposure_ok, log: Callable = print,
                 version: int = 1):
        self.S, self.root = sections, Path(root)
        self.run_units, self.metrics, self.checks = run_units, metrics, checks
        self.used_hours, self.costs_extra, self.freeze_extra = used_hours, costs_extra, freeze_extra
        self.exposure_ok, self.log, self.version = exposure_ok, log, version
        self.natural = natural_long_share()

    # ---------------------------------------------------------------- state
    @property
    def entries(self) -> list[dict]:
        return self.S.data.get("log", [])

    def record(self, entry: dict) -> None:
        self.S.append("log", {"n": len(self.entries), **entry})

    def save_once(self, key: str, value) -> None:
        """A write-once key: a resumed step may write it again only with the same content (review L2)."""
        old = self.S.data.get(key)
        if old is not None and json.dumps(old, sort_keys=True, default=str) != json.dumps(value, sort_keys=True,
                                                                                            default=str):
            raise ChainHalt(f"{key} is write-once and would change")
        if old is None:
            self.S.save(key, value)

    def settings(self) -> Settings:
        s = Settings()
        for e in self.entries:
            if "settings" in e:
                s = Settings(**e["settings"])
        return s

    def _round(self, step: str) -> int:
        return 1 + sum(1 for e in self.entries if e["step"] == step)

    def _phase_name(self, step: str, rnd: int) -> str:
        return phase_name(step, rnd, self.version)

    def gates(self) -> list[dict]:
        return [e["decision"] for e in self.entries if e["step"] == "D"]

    def d60(self) -> list[dict]:
        return [e["decision"] for e in self.entries if e["step"] == "H"]

    # ---------------------------------------------------------------- driver
    def run(self, max_actions: int = 1000) -> str:
        """Advance until done or a pause. Returns 'done' or raises ChainPause."""
        self.verify_log()
        for _ in range(max_actions):
            last = self.entries[-1] if self.entries else None
            if last and last.get("pause"):
                raise ChainPause(last["pause"])
            nxt = self.next_step()
            if nxt == "done":
                return "done"
            getattr(self, f"_do_{nxt}")()
        raise RuntimeError("too many chain actions")

    def next_step(self) -> str:
        if not self.entries:
            return "A"
        last = self.entries[-1]
        st = last["step"]
        if st == "A":
            return "B"
        if st == "B":
            return "A" if last["next"] == "A" else "C"
        if st == "C":
            return "D"
        if st == "D":                    # E is recorded (not just written): a crash in between redoes E (review L1)
            if not any(e["step"] == "E" for e in self.entries):
                return "E"
            return "G" if last.get("final") else "F"
        if st == "E":
            return "F"
        if st == "F":
            return "H" if last["d60"] else "D"
        if st == "H":
            return "H" if last.get("again") else "D"
        if st == "G":
            return "done"
        raise ChainHalt(f"unknown step {st}")

    # ---------------------------------------------------------------- steps
    def _phase(self, step: str, units: list[Unit], meta: dict | None = None) -> tuple[list[dict], list[dict]]:
        phase = units[0].phase
        results = self.run_units(phase, units, meta)
        ms = [self.metrics(r) for r in results]
        bad = [m["unit_id"] for m in ms if m.get("status", "done") != "done"]
        if len(ms) != len(units) or bad:
            self.record({"step": step, "phase": phase, "pause": f"a code fix is needed: units not done {bad}"})
            raise ChainPause(f"units not done in {phase}: {bad}")
        return results, ms

    def _do_A(self) -> None:
        s, rnd = self.settings(), self._round("A")
        units = wording_units(rnd, s, self.version)
        _, ms = self._phase("A", units)
        d = choose_wording(ms)
        self.log(f"[s34] A-r{rnd}: wording {d['wording']} ({d['decided_by']}; solved {d['solved']})")
        self.record({"step": "A", "round": rnd, "phase": units[0].phase, "decision": d,
                     "settings": dataclasses.replace(s, wording=d["wording"]).to_json()})

    def _ladder_state(self) -> tuple[str | None, int]:
        direction, slot = None, 0
        for e in self.entries:
            if e["step"] == "B" and e.get("decision", {}).get("ladder"):
                direction, slot = e["decision"]["ladder"]["direction"], e["decision"]["ladder"]["next_slot"]
        return direction, slot

    @staticmethod
    def _switches_on(s: Settings) -> tuple[bool, bool]:
        return s.grammar_always, s.action_max_tokens == 640

    @staticmethod
    def _apply_switches(s: Settings, fired: list) -> Settings:
        return dataclasses.replace(s, grammar_always=s.grammar_always or "grammar_always" in fired,
                                   action_max_tokens=640 if "cap640" in fired else s.action_max_tokens)

    def _fired(self, s: Settings, sw: dict) -> list:
        g_on, c_on = self._switches_on(s)
        return sorted(k for k, on in (("grammar_always", sw["grammar_always"] and not g_on),
                                      ("cap640", sw["cap640"] and not c_on)) if on)

    def _do_B(self) -> None:
        s, rnd = self.settings(), self._round("B")
        units = pilot_units(self._phase_name("B", rnd), s)
        _, ms = self._phase("B", units)
        c0, c2 = _sel(ms, condition="C0"), _sel(ms, condition="C2")
        sw = switch_check(c0, c2)
        fired = self._fired(s, sw)
        d: dict[str, Any] = {"switches": sw, "fired": fired, "summary": {
            "C0": condition_summary(c0), "C2": condition_summary(c2)}}
        entry = {"step": "B", "round": rnd, "phase": units[0].phase, "decision": d}
        if fired:                                 # the round is not judged (spec section 5)
            entry.update(settings=self._apply_switches(s, fired).to_json(), next="A")
            self.log(f"[s34] B-r{rnd}: switch(es) {fired} -> re-run A and B")
            self.record(entry)
            return
        acc = acceptance(c0)
        d["acceptance"] = acc
        if acc["verdict"] == "ok":
            entry["next"] = "C"
            self.log(f"[s34] B-r{rnd}: C0 success {acc['c0_success']:.3f} accepted")
            self.record(entry)
            return
        direction, slot = self._ladder_state()
        if direction and direction != acc["verdict"]:
            entry["pause"] = (f"the ladder overshot: C0 was too {direction}, now too {acc['verdict']} "
                              f"({acc['c0_success']:.3f}); the user decides (D157)")
        else:
            new, nxt, skipped = ladder_step(s, acc["verdict"], slot if direction else 0, self.natural)
            if new is None:
                entry["pause"] = (f"C0 success {acc['c0_success']:.3f} after 3 ladder steps is still outside "
                                  "0.2-0.8; the user decides (D157)")
            else:
                d["ladder"] = {"direction": acc["verdict"], "next_slot": nxt, "skipped": skipped,
                               "natural_long_share": self.natural}
                entry.update(settings=new.to_json(), next="A")
        self.log(f"[s34] B-r{rnd}: C0 success {acc['c0_success']:.3f} ({acc['verdict']}) -> "
                 f"{entry.get('pause') or 'ladder step'}")
        self.record(entry)

    def _final_pilot(self) -> dict:
        return [e for e in self.entries if e["step"] == "B" and e.get("next") == "C"][-1]

    def _do_C(self) -> None:
        s = self.settings()
        rnd = self._round("C")
        units = rq3_units(self._phase_name("C", rnd), s)
        results, ms = self._phase("C", units)
        v_ok, hits = self.checks(results)
        w = wiring_check(ms, v_ok, hits)
        entry = {"step": "C", "round": rnd, "phase": units[0].phase, "decision": {
            **w, "summary": {m["unit_id"]: condition_summary([m]) for m in ms}}}
        if not w["ok"]:
            entry["pause"] = ("RQ3 wiring check: " + ("tasks unfinished at T_max" if w["sane"] else "wiring not sane")
                              + "; the user decides (D161, D111)")
        self.log(f"[s34] C-r{rnd}: ok {w['ok']} (unfinished {w['unfinished_at_t_max']}, verify {v_ok}, audit {hits})")
        self.record(entry)

    def _costs(self) -> Costs:
        """Seconds per task: the final judged pilot round, replaced by the latest D60 round's re-measurement under the
        new setting (review M4); RQ3 at the larger cell of the latest wiring check."""
        b = self._final_pilot()["decision"]["summary"]
        c0, c2 = b["C0"]["seconds_per_task"], b["C2"]["seconds_per_task"]
        rq3 = max(v["seconds_per_task"] for v in
                  [e for e in self.entries if e["step"] == "C"][-1]["decision"]["summary"].values())
        for h in self.d60():
            c0, c2, rq3 = h["c0_seconds_per_task"], h["c2_seconds_per_task"], h["rq3_seconds_per_task"]
        x = self.costs_extra()
        return Costs(c0, c2, rq3, x["setup_s"], x["gap_s"], x["unit_hours_per_session"])

    def _do_D(self) -> None:
        s = self.settings()
        prev = self.gates()
        cuts = tuple(prev[-1]["cuts"]) if prev else ()
        n0 = prev[-1]["n_tasks"] if prev else 20
        selection_done = any(e["step"] == "F" for e in self.entries)
        g_on, c_on = self._switches_on(s)
        g = budget_gate(self.used_hours(), self._costs(), cuts, n0, selection_done=selection_done,
                        d60_reserve=not selection_done and not (g_on and c_on),
                        selected_seeds=3 if 11 in cuts else 4, exposure_ok=lambda n: self.exposure_ok(n, s))
        entry = {"step": "D", "round": len(prev) + 1, "decision": g, "final": selection_done}
        if g["pause"]:
            entry["pause"] = g["pause"]
        self.log(f"[s34] D-r{len(prev) + 1}: used {g['used']} + projected {g['projected']['total']} h; cuts {g['cuts']}; "
                 f"n_tasks {g['n_tasks']}{'; PAUSE' if g['pause'] else ''}")
        self.record(entry)

    def _do_E(self) -> None:
        s = self.settings()
        g = self.gates()[-1]
        rec = {"commit": self.S.data.get("stamp", {}).get("commit"), "chain_version": self.version,
               "settings": s.to_json(),
               "switches": {"grammar_always": s.grammar_always, "action_max_tokens": s.action_max_tokens},
               "knobs": {"n_distractors": s.n_distractors, "long_share": s.long_share,
                         "natural_long_share": self.natural},
               "n_tasks": g["n_tasks"], "cuts": g["cuts"], "gate": g,
               "decisions": [e for e in self.entries if e["step"] in ("A", "B", "C", "D")], **self.freeze_extra(s)}
        sha = write_once(self.root / "s34" / freeze_file(self.version), rec)
        self.save_once("freeze_sha", sha)
        self.log(f"[s34] E: freeze written ({sha[:12]})")
        self.record({"step": "E", "freeze_sha": sha})

    def _do_F(self) -> None:
        s, g = self.settings(), self.gates()[-1]
        seeds = SELECTION_SEEDS[:3] if 11 in g["cuts"] else SELECTION_SEEDS
        units = selection_units(self._phase_name("F", 1), s, seeds, g["n_tasks"])
        _, ms = self._phase("F", units, {"freeze_sha": self.S.data["freeze_sha"]})
        sel = select_config(ms, len(seeds))
        best = _sel(ms, tag=sel["config"])
        c0 = _sel(self._unit_metrics_of(self._final_pilot()["phase"]), condition="C0")
        sw = switch_check(c0, best)
        fired = self._fired(s, sw)
        entry = {"step": "F", "phase": units[0].phase, "d60": bool(fired),
                 "decision": {**sel, "d60_switches": sw, "fired": fired,
                              "sweep": {c: condition_summary(_sel(ms, tag=c)) for c in CONFIGS if _sel(ms, tag=c)}}}
        if fired:
            entry["settings"] = self._apply_switches(s, fired).to_json()
        self.save_once("selection", {"config": sel["config"], "params": CONFIGS[sel["config"]], "seeds": list(seeds),
                                     "n_tasks": g["n_tasks"], "freeze_sha": self.S.data["freeze_sha"]})
        self.log(f"[s34] F: selected {sel['config']} ({sel['decided_by']}); D60 {fired or 'none'}")
        self.record(entry)

    def _unit_metrics_of(self, phase: str) -> list[dict]:
        units = [Unit.from_json(u) for u in _read_plan_units(self.root, phase)]
        return [self.metrics(r) for r in self.run_units(phase, units, None)]

    def _h_units(self, rnd: int, s: Settings) -> list[Unit]:
        g = self.gates()[-1]
        cfg = self.S.data["selection"]["config"]
        seeds = tuple(self.S.data["selection"]["seeds"])
        phase = self._phase_name("H", rnd)
        return (selection_units(phase, s, seeds, g["n_tasks"], cfgs=(cfg,)) + pilot_units(phase, s, ("C0",))
                + rq3_units(phase, s))

    def _h_decision(self, ms: list[dict], s: Settings, rnd: int, v_ok: bool, hits: int) -> dict:
        cfg = self.S.data["selection"]["config"]
        rq1 = [m for m in ms if m["rq"] == "rq1"]
        sel_ms, c0 = _sel(rq1, tag=cfg), _sel(rq1, condition="C0", tag="")
        rq3 = [m for m in ms if m["rq"] == "rq3"]
        w = wiring_check(rq3, v_ok, hits)
        acc = acceptance(c0)
        sw = switch_check(c0, sel_ms)
        fired = self._fired(s, sw)
        return {"acceptance": acc, "acceptance_failed_reported": acc["verdict"] != "ok", "wiring": w,
                "switches": sw, "fired": fired, "again": bool(fired) and rnd < MAX_D60_ROUNDS,
                "c0_seconds_per_task": condition_summary(c0)["seconds_per_task"],
                "c2_seconds_per_task": condition_summary(sel_ms)["seconds_per_task"],
                "rq3_seconds_per_task": max(condition_summary([m])["seconds_per_task"] for m in rq3)}

    def _do_H(self) -> None:
        """D60 round (spec section 9): selected config's selection streams, C0 pilot streams, RQ3 wiring."""
        s = self.settings()
        rnd = self._round("H")
        units = self._h_units(rnd, s)
        results, ms = self._phase("H", units)
        v_ok, hits = self.checks([r for r in results if r["unit"]["rq"] == "rq3"])
        d = self._h_decision(ms, s, rnd, v_ok, hits)
        entry = {"step": "H", "round": rnd, "phase": units[0].phase, "again": d["again"], "decision": d}
        if d["fired"]:
            entry["settings"] = self._apply_switches(s, d["fired"]).to_json()
        if not d["wiring"]["ok"]:
            entry["pause"] = "RQ3 wiring check after D60: " + (
                "tasks unfinished at T_max" if d["wiring"]["sane"] else "wiring not sane") + "; the user decides (D161)"
        self.log(f"[s34] H-r{rnd}: C0 {d['acceptance']['c0_success']:.3f} ({d['acceptance']['verdict']}), wiring ok "
                 f"{d['wiring']['ok']}, switches {d['fired'] or 'none'}")
        self.record(entry)

    def _do_G(self) -> None:
        self.record({"step": "G"})
        self.log("[s34] G: done")

    # ---------------------------------------------------------------- resume check
    def verify_log(self) -> None:
        """Recompute every recorded A/B/C/F/H decision from the stored results (spec section 2). The C and H checks
        reuse the recorded verify_replay/audit outcome (a transient failure must not halt the chain, review L3)."""
        s = Settings()
        for i, e in enumerate(self.entries):
            st = e["step"]
            if st in ("A", "B", "C", "F", "H") and "phase" in e and not e.get("pause"):
                units = [Unit.from_json(u) for u in _read_plan_units(self.root, e["phase"])]
                ms = [self.metrics(r) for r in self.run_units(e["phase"], units, None)]
                if st == "A":
                    same = choose_wording(ms) == e["decision"]
                elif st == "B":
                    c0, c2 = _sel(ms, condition="C0"), _sel(ms, condition="C2")
                    same = switch_check(c0, c2) == e["decision"]["switches"] and (
                        "acceptance" not in e["decision"] or acceptance(c0) == e["decision"]["acceptance"])
                elif st == "C":
                    d = e["decision"]
                    same = all(d[k] == v for k, v in wiring_check(ms, d["verify_ok"], d["audit_hits"]).items())
                elif st == "F":
                    sel = select_config(ms, len({m["seed"] for m in ms}))
                    same = all(e["decision"][k] == sel[k] for k in sel)
                else:
                    w = e["decision"]["wiring"]
                    same = self._h_decision(ms, s, e["round"], w["verify_ok"], w["audit_hits"]) == e["decision"]
                if not same:
                    raise ChainHalt(f"recomputed decision for {e['phase']} differs from the recorded one")
            if "settings" in e:
                s = Settings(**e["settings"])


def _read_plan_units(root: Path, phase: str) -> list[dict]:
    from .llm.runner import read_plan
    return read_plan(root, phase)["units"]


def template_set_sha() -> str:
    from .apisuite.templates import TEMPLATES
    return hashlib.sha256(canon({t: {"covers": sorted(v.covers), "long": v.long} for t, v in TEMPLATES.items()})
                          .encode()).hexdigest()
