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

from .llm.fsio import canon, log_fingerprint  # noqa: F401 - log_fingerprint is part of this module's API
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
MAX_D60_ROUNDS = 2       # documentation: each switch fires once, so at most two D60 rounds re-measure (review M3)


class ChainPause(Exception):
    """The chain stops and asks the user (spec section 11)."""


class ChainHalt(RuntimeError):
    """A recomputed decision differs from the recorded one: a bug (spec section 2)."""


class ChainRetry(RuntimeError):
    """Units failed for the first time: stop without a pause; the next session retries them once in a fresh runtime
    (second opinion M3, spec section 11)."""


AN_FACTOR = 1.2                           # AN carries every note in its prompt: C2's cost +20% (D167)
USER_ACTIONS = {"B": ("settings", "accept"), "C": ("accept", "rerun"), "D": ("gate",), "H": ("accept", "rerun")}


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


def realised_long_share(units: list[Unit]) -> float:
    """The share of long-template tasks over the distinct streams of `units` (second opinion Low 7: the ladder moves the
    sampling share; this is what the pilot actually ran)."""
    from .apisuite.streams import sample_stream
    from .apisuite.templates import TEMPLATES
    seen, n, k = set(), 0, 0
    for u in units:
        key = (u.seed, u.split, u.kind, u.stream)
        if key in seen:
            continue
        seen.add(key)
        tasks = sample_stream(u.seed, u.split, kind=u.kind, **dict(u.stream))["tasks"][:u.n_tasks]
        n += len(tasks)
        k += sum(TEMPLATES[t["template"]].long for t in tasks)
    return round(k / n, 6) if n else 0.0


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


def rq3_units(phase: str, s: Settings, c2_extra: dict | None = None) -> list[Unit]:
    """`c2_extra`: the selected EEM configuration for the C2 cell (the D60 re-check, second opinion Low 1)."""
    return [Unit(phase, "rq3", "dev", "rq3", RQ3_SEED, c, w,
                 params=s.params(c2_extra if c == "C2" else None), stream=s.stream())
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
    an: float                     # seconds per task, AN (D167: C2's cost x AN_FACTOR)
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
              ("AN", 3 if 4 in cuts else 5, desc_t, "an"),
              ("SP", 2 * (3 if 5 in cuts else 5), desc_t, "sp"),
              ("rq3_2x2", 4 * 10, 20, "rq3"),
              ("rq3_W4", 0 if 9 in cuts else (5 if 1 in cuts else 10), 20, "rq3"),
              ("rq3_private", 0 if 0 in cuts else 5, 20, "rq3"),
              ("rq3_sensitivity", 0 if 8 in cuts else 4 * (3 if 6 in cuts else 5), 20, "rq3")]
    return [i for i in items if i[1] > 0]


def projection(costs: Costs, items: list) -> dict:
    per = {"c0": costs.c0, "c2": costs.c2, "rq3": costs.rq3, "an": costs.an, "sp": (costs.c0 + costs.c2) / 2}
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
    return {"fits": fits, "used": used, "projected": p, "costs": dataclasses.asdict(costs),
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
                 version: int = 1, runtime: str = "local", adopt: Callable | None = None):
        """`runtime`: this runtime's id (a retry must run in another one); `adopt(phase, unit) -> (from_commit,
        payload_sha256, verify)`: replay a unit made at an earlier commit with the current code (D166)."""
        self.S, self.root = sections, Path(root)
        self.runtime, self.adopt = runtime, adopt
        self.run_units, self.metrics, self.checks = run_units, metrics, checks
        self.used_hours, self.costs_extra, self.freeze_extra = used_hours, costs_extra, freeze_extra
        self.exposure_ok, self.log, self.version = exposure_ok, log, version
        self.natural = natural_long_share()
        self._loaded: dict[str, tuple[list[dict], list[dict]]] = {}     # recorded phases loaded this session

    # ---------------------------------------------------------------- state
    @property
    def entries(self) -> list[dict]:
        return self.S.data.get("log", [])

    def record(self, entry: dict) -> None:
        if entry.get("pause"):                   # a pause that needs a code fix is cleared by a re-pin (D166)
            entry = {**entry, "commit": self.S.data.get("stamp", {}).get("commit")}
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
        """A pause for a code fix is a step 'X' entry: the retry after it runs the same round and phase (M3)."""
        return 1 + sum(1 for e in self.entries if e["step"] == step)

    def _phase_name(self, step: str, rnd: int) -> str:
        return phase_name(step, rnd, self.version)

    def gates(self) -> list[dict]:
        return [e["decision"] for e in self.entries if e["step"] == "D" and "decision" in e]

    def d60(self) -> list[dict]:
        return [e["decision"] for e in self.entries if e["step"] == "H" and "decision" in e]

    # ---------------------------------------------------------------- driver
    def run(self, max_actions: int = 1000) -> str:
        """Advance until done or a pause. Returns 'done' or raises ChainPause (ChainRetry: stop, retry next session)."""
        self.adopt_repinned()
        self._clear_fixed_pause()
        last = self.entries[-1] if self.entries else None
        if last and last.get("pause"):           # answered first (D165); the checks below run once it is answered
            raise ChainPause(last["pause"])
        self.verify_log()                        # loads every recorded phase through the retry rule (review H2)
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
        if st == "U":
            return last["next"]
        if st == "X":
            raise ChainHalt("a pause for a code fix is still open")
        raise ChainHalt(f"unknown step {st}")

    # ---------------------------------------------------------------- steps
    def _phase(self, step: str, units: list[Unit], meta: dict | None = None) -> tuple[list[dict], list[dict]]:
        return self._units(units[0].phase, units, meta, step)

    def _units(self, phase: str, units: list[Unit], meta: dict | None, before: str | None):
        """Run (or load) a phase's units through the retry rule (second opinion M3, review M1): units that are not
        done for the first time stop the chain without a pause (ChainRetry, recorded per unit set under
        `retries.<phase>`); the next session, in another runtime, retries them once; only units that are not done
        again after their retry pause the chain (step 'X', `before` = the step to run once it is answered). The retry
        is recorded as made only after it ran."""
        retries = dict(self.S.data.get("retries", {}))
        attempts = [dict(a) for a in retries.get(phase, [])]
        last = self.entries[-1] if self.entries else {}
        user_retry = last.get("step") == "U" and last.get("retry_phase") == phase
        pending = [a for a in attempts if not a.get("retried_by")]
        if pending and not user_retry and all(a["runtime"] == self.runtime for a in pending):
            raise ChainRetry(f"{phase}: the units failed in this runtime; Runtime -> Disconnect and delete runtime, "
                             "reconnect and run again (the retry must run in a fresh runtime)")
        results = self.run_units(phase, units, meta, retry=user_retry or bool(pending))
        if pending:
            for a in attempts:
                if not a.get("retried_by") and a["runtime"] != self.runtime:
                    a["retried_by"] = self.runtime
            retries[phase] = attempts
            self.S.save("retries", retries)
        ms = [self.metrics(r) for r in results]
        bad = [m["unit_id"] for m in ms if m.get("status", "done") != "done"]
        if len(ms) == len(units) and not bad:
            return results, ms
        tried = {u for a in attempts if a.get("retried_by") for u in a["units"]}
        if user_retry or set(bad) <= tried:
            nxt = before or self._next_or_none()
            if nxt is None:
                raise ChainHalt(f"units of the recorded phase {phase} are not done: {bad}")
            self.record({"step": "X", "before": nxt, "phase": phase, "failed": True, "fix": True, "units": bad,
                         "pause": f"a code fix is needed: units not done {bad} (they failed again after a retry); "
                                  "after the fix, re-pin (REPIN), or answer with USER_DECISION action 'rerun'"})
            raise ChainPause(f"units not done in {phase} after a retry: {bad}")
        attempts.append({"units": bad, "runtime": self.runtime})
        retries[phase] = attempts
        self.S.save("retries", retries)
        raise ChainRetry(f"{phase}: units not done {bad}; the next session retries them once in a fresh runtime")

    def _next_or_none(self) -> str | None:
        try:
            return self.next_step()
        except (KeyError, ChainHalt):
            return None

    def _load(self, phase: str, before: str | None = None) -> tuple[list[dict], list[dict]]:
        """A recorded phase's results and metrics, through the retry rule (review H2: a re-run of a unit that was not
        adopted, or the repair of a lost result, may run and fail like any other unit)."""
        if phase not in self._loaded:
            units = [Unit.from_json(u) for u in _read_plan_units(self.root, phase)]
            self._loaded[phase] = self._units(phase, units, None, before)
        return self._loaded[phase]

    # ---------------------------------------------------------------- user decisions (D165)
    def apply_user_decision(self, d: dict, expect: str | None = None) -> dict:
        """Record the user's answer to the current pause as a 'U' entry and clear the pause (spec section 11).
        `expect`: the log fingerprint the chat saw (EXPECT_LOG); required unless the answer was recorded already."""
        done = [e for e in self.entries if e.get("user_decision") == d]
        if done:                                 # the notebook still carries an answer that was recorded already
            return done[-1]
        if str(expect) != log_fingerprint(self.entries):
            raise ValueError(f"EXPECT_LOG {expect!r} is not this chain's state {log_fingerprint(self.entries)!r}: the "
                             "decision was written for another chain state (another Drive folder or a stale bundle)")
        if not self.entries or not self.entries[-1].get("pause"):
            raise ValueError("the chain is not paused: there is nothing to decide")
        p = self.entries[-1]
        if d.get("answers") != p["n"]:
            raise ValueError(f"USER_DECISION answers pause n={d.get('answers')}, but the current pause is n={p['n']}")
        act, st = d.get("action"), p.get("before", p["step"])
        entry: dict[str, Any] = {"step": "U", "answers": p["n"], "decision": d, "user_decision": d}
        if p.get("fix"):                         # failed units or the replay check (review M4)
            if act != "rerun":
                raise ValueError("after a pause for a code fix the only action is 'rerun' (or a fix and a re-pin)")
            entry["next"] = st
            if p.get("failed"):
                entry["retry_phase"] = p["phase"]
        elif act not in USER_ACTIONS.get(st, ()):
            raise ValueError(f"action {act!r} is not allowed after a {st} pause; allowed: {USER_ACTIONS.get(st)}")
        elif st == "B" and act == "settings":
            fields = {f.name for f in dataclasses.fields(Settings)}
            if not d.get("settings") or set(d["settings"]) - fields:
                raise ValueError(f"settings must be a non-empty subset of {sorted(fields)}")
            new = dataclasses.replace(self.settings(), **d["settings"])
            _check_settings(new)
            entry.update(settings=new.to_json(), next="A")
        elif st == "B":
            entry.update(next="C", final_pilot=p["n"])
        elif st == "C":
            entry["next"] = "D" if act == "accept" else "C"
        elif st == "D":
            if (not isinstance(d.get("cuts"), list) or not set(d["cuts"]) <= set(range(12))
                    or d.get("n_tasks") not in TASK_COUNTS + (20,)):
                raise ValueError(f"a gate decision needs 'cuts' (a list of cut numbers 0-11) and 'n_tasks' "
                                 f"(one of {sorted(set(TASK_COUNTS + (20,)))})")
            applied = [g for g in self.gates() if not g.get("pause")]
            if applied and (not set(applied[-1]["cuts"]) <= set(d["cuts"]) or d["n_tasks"] > applied[-1]["n_tasks"]):
                raise ValueError(f"cuts are only ever added and the task count never grows (spec section 7): the last "
                                 f"gate has cuts {applied[-1]['cuts']} and {applied[-1]['n_tasks']} tasks")
            g = self._gate_for(p["decision"], sorted(set(d["cuts"])), d["n_tasks"])
            entry.update(step="D", decision=g, final=p.get("final", False), user=True)
        elif st == "H":
            entry["next"] = ("H" if p.get("again") else "D") if act == "accept" else "H"
        self.record(entry)
        self.log(f"[s34] user decision recorded for pause n={p['n']}: {act}")
        return self.entries[-1]

    def _gate_for(self, paused: dict, cuts: list, n_tasks: int) -> dict:
        """The user's gate (D165, review M5): the projection recomputed for exactly the chosen cuts and task count."""
        s = self.settings()
        g = budget_gate(paused["used"], Costs(**paused["costs"]), tuple(cuts), n_tasks,
                        selection_done=any(e["step"] == "F" for e in self.entries),
                        d60_reserve=_d60_reserve(self.entries, s), selected_seeds=3 if 11 in cuts else 4,
                        exposure_ok=lambda n: True, budget=math.inf)
        return {**g, "fits": paused["used"] + g["projected"]["total"] <= BUDGET_HOURS, "pause": None, "user": True}

    def _clear_fixed_pause(self) -> None:
        """A pause that needs a code fix, recorded at an older commit than the chain's current pin: the re-pin is the
        fix, so the paused step runs again (a failed-units pause retries its units once more)."""
        p = self.entries[-1] if self.entries else {}
        cur = self.S.data.get("stamp", {}).get("commit")
        if p.get("pause") and p.get("fix") and p.get("commit") not in (None, cur):
            entry: dict[str, Any] = {"step": "U", "answers": p["n"], "decision": {"action": "repin", "to": cur},
                                     "next": p["before"]}
            if p.get("failed"):
                entry["retry_phase"] = p["phase"]
            self.record(entry)
            self.log(f"[s34] re-pinned to {cur[:7]}: the paused step {p['before']} runs again")

    # ---------------------------------------------------------------- re-pin and adoption (D166)
    def adopt_repinned(self) -> list[dict]:
        """After a re-pin to a fixed commit: every plan of this chain at another commit gets a new version at the
        current commit, and every finished unit whose replay with the current code is identical is adopted (no GPU);
        the others will re-run. Recorded under `adoptions`."""
        from .llm.runner import read_plan, write_plan
        cur = self.S.data.get("stamp", {}).get("commit")
        if not cur or self.adopt is None:
            return []
        prefix = "s34-" if self.version == 1 else f"s34v{self.version}-"
        done = []
        for pf in sorted((self.root / "plans").glob(f"{prefix}*.json")):
            plan = read_plan(self.root, pf.stem)
            if plan.get("frozen_commit") in (None, cur):
                continue
            units = [Unit.from_json(u) for u in plan["units"]]
            adopted, rerun = [], []
            for u in units:
                frm, sha, verify = self.adopt(pf.stem, u)
                if frm is not None and verify is True:
                    from .llm.runner import record_adoption
                    record_adoption(self.root, pf.stem, u.unit_id, frm, cur, sha, verify)
                    adopted.append(u.unit_id)
                else:
                    rerun.append(u.unit_id)
            write_plan(self.root, pf.stem, units, plan["version"] + 1, frozen_commit=cur, meta=plan.get("meta"))
            rec = {"phase": pf.stem, "from": plan["frozen_commit"], "to": cur, "adopted": adopted, "rerun": rerun}
            self.S.append("adoptions", rec)                   # per phase (review L2)
            done.append(rec)
            self.log(f"[s34] re-pin: {pf.stem}: {len(adopted)} adopted, {len(rerun)} to re-run")
        return done

    def _replay_gate(self, step: str, phases: list[str]) -> None:
        """D168: every unit of `phases` must replay identically (and pass the scrub audit) before a frozen decision."""
        results = []
        for ph in phases:
            results += self._load(ph, step)[0]
        ok, hits = self.checks(results)
        if not ok or hits:
            self.record({"step": "X", "before": step, "fix": True,
                         "pause": f"replay check before {step}: verify_replay ok {ok}, audit hits {hits}; a code fix "
                                  "and a re-pin are needed, or USER_DECISION 'rerun' for a transient error (D168)"})
            raise ChainPause(f"replay check before {step} failed")

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
            elif e["step"] == "U" and "settings" in e:
                direction, slot = "user", 0      # the user's settings end the ladder: a miss pauses again (D165)
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
            "C0": condition_summary(c0), "C2": condition_summary(c2)},
            "realised_long_share": realised_long_share(units)}
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
        if direction == "user":
            entry["pause"] = (f"C0 success {acc['c0_success']:.3f} under the user's settings is outside 0.2-0.8; the "
                              "user decides (D157, D165)")
        elif direction and direction != acc["verdict"]:
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
        return self._final_pilot_before(len(self.entries))

    def _final_pilot_before(self, i: int) -> dict:
        """The judged pilot round: a B entry that moved on to C, or a paused B round the user accepted (D165)."""
        out = None
        for e in self.entries[:i]:
            if e["step"] == "B" and e.get("next") == "C":
                out = e
            elif e["step"] == "U" and "final_pilot" in e:
                out = self.entries[e["final_pilot"]]
        if out is None:
            raise ChainHalt("no judged pilot round")
        return out

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
        """Seconds per task: the final judged pilot round; after the selection, C2-mapped work at the selected
        configuration's measured cost (D167); the latest D60 round's re-measurement under the new setting (review M4);
        RQ3 at the larger measured wiring cell; AN at C2 x AN_FACTOR (D167). A missing cost (a unit rebuilt only from
        its cache) is never passed on (second opinion M2)."""
        b = self._final_pilot()["decision"]["summary"]
        c0, c2 = b["C0"]["seconds_per_task"], b["C2"]["seconds_per_task"]
        rq3_cells = [v["seconds_per_task"] for v in
                     [e for e in self.entries if e["step"] == "C"][-1]["decision"]["summary"].values()]
        rq3 = max([x for x in rq3_cells if x is not None], default=None)
        sel = [e for e in self.entries if e["step"] == "F"]
        if sel:
            sweep = sel[-1]["decision"]["sweep"][sel[-1]["decision"]["config"]]
            c2 = sweep["seconds_per_task"] if sweep["seconds_per_task"] is not None else c2
        for h in self.d60():
            c0 = h["c0_seconds_per_task"] if h["c0_seconds_per_task"] is not None else c0
            c2 = h["c2_seconds_per_task"] if h["c2_seconds_per_task"] is not None else c2
            rq3 = h["rq3_seconds_per_task"] if h["rq3_seconds_per_task"] is not None else rq3
        if c0 is None or c2 is None:
            raise ChainHalt("the pilot has no measurable cost per task (every unit rebuilt from its cache)")
        if rq3 is None:
            rq3 = c2                                  # no measured RQ3 cell: C2's cost per task (second opinion M2)
            self.log("[s34] no measured RQ3 cost: using C2's cost per task")
        x = self.costs_extra()
        return Costs(c0, c2, rq3, c2 * AN_FACTOR, x["setup_s"], x["gap_s"], x["unit_hours_per_session"])

    def _do_D(self) -> None:
        s = self.settings()
        prev = self.gates()
        cuts = tuple(prev[-1]["cuts"]) if prev else ()
        n0 = prev[-1]["n_tasks"] if prev else 20
        selection_done = any(e["step"] == "F" for e in self.entries)
        g = budget_gate(self.used_hours(), self._costs(), cuts, n0, selection_done=selection_done,
                        d60_reserve=_d60_reserve(self.entries, s),
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
        self._replay_gate("E", sorted({e["phase"] for e in self.entries if e["step"] in ("A", "B") and "decision" in e}))
        rec = {"commit": self.S.data.get("stamp", {}).get("commit"), "chain_version": self.version,
               "settings": s.to_json(),
               "switches": {"grammar_always": s.grammar_always, "action_max_tokens": s.action_max_tokens},
               "knobs": {"n_distractors": s.n_distractors, "long_share": s.long_share,
                         "natural_long_share": self.natural},
               "n_tasks": g["n_tasks"], "cuts": g["cuts"], "gate": g,
               "decisions": [e for e in self.entries if e["step"] in ("A", "B", "C", "D")], **self.freeze_extra(s)}
        ff = self.root / "s34" / freeze_file(self.version)
        old = json.loads(ff.read_text()) if ff.exists() else None
        if old is not None and old.get("decisions") == rec["decisions"] and old.get("commit") != rec["commit"]:
            sha = hashlib.sha256(ff.read_bytes()).hexdigest()     # review L3: written before a re-pin; kept as is
            self.log(f"[s34] E: the freeze written at {old.get('commit', '?')[:7]} before the re-pin is kept")
        else:
            sha = write_once(ff, rec)
        self.save_once("freeze_sha", sha)
        self.log(f"[s34] E: freeze written ({sha[:12]})")
        self.record({"step": "E", "freeze_sha": sha})

    def _do_F(self) -> None:
        s, g = self.settings(), self.gates()[-1]
        seeds = SELECTION_SEEDS[:3] if 11 in g["cuts"] else SELECTION_SEEDS
        units = selection_units(self._phase_name("F", 1), s, seeds, g["n_tasks"])
        results, ms = self._phase("F", units, {"freeze_sha": self.S.data["freeze_sha"]})
        self._replay_gate("F", [units[0].phase])                     # D168
        sel = select_config(ms, len(seeds))
        best = _sel(ms, tag=sel["config"])
        c0 = _sel(self._unit_metrics_of(self._final_pilot()["phase"], "F"), condition="C0")
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

    def _unit_metrics_of(self, phase: str, before: str | None = None) -> list[dict]:
        return self._load(phase, before)[1]

    def _h_units(self, rnd: int, s: Settings) -> list[Unit]:
        g = self.gates()[-1]
        cfg = self.S.data["selection"]["config"]
        seeds = tuple(self.S.data["selection"]["seeds"])
        phase = self._phase_name("H", rnd)
        return (selection_units(phase, s, seeds, g["n_tasks"], cfgs=(cfg,)) + pilot_units(phase, s, ("C0",))
                + rq3_units(phase, s, CONFIGS[cfg]))

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
                "switches": sw, "fired": fired, "again": bool(fired),     # review M3: a user rerun is no D60 round
                "c0_seconds_per_task": condition_summary(c0)["seconds_per_task"],
                "c2_seconds_per_task": condition_summary(sel_ms)["seconds_per_task"],
                "rq3_seconds_per_task": max([x for x in (condition_summary([m])["seconds_per_task"] for m in rq3)
                                             if x is not None], default=None)}

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
        """Recompute every recorded A/B/C/F/H decision from the stored results (spec section 2), each phase loaded
        through the retry rule (`_load`, review H2), and every gate from its recorded inputs. The C and H checks reuse
        the recorded verify_replay/audit outcome (a transient failure must not halt the chain, review L3)."""
        s = Settings()
        for i, e in enumerate(self.entries):
            st = e["step"]
            if st in ("A", "B", "C", "F", "H") and "phase" in e and "decision" in e:
                ms = self._load(e["phase"])[1]
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
                    if same:                                          # second opinion Low 3: D60 switches too
                        c0 = _sel(self._unit_metrics_of(self._final_pilot_before(i)["phase"]), condition="C0")
                        same = switch_check(c0, _sel(ms, tag=sel["config"])) == e["decision"]["d60_switches"]
                else:                                                 # second opinion M1: no timing fields
                    w = e["decision"]["wiring"]
                    new = self._h_decision(ms, s, e["round"], w["verify_ok"], w["audit_hits"])
                    same = all(new[k] == e["decision"][k] for k in ("acceptance", "wiring", "switches", "fired",
                                                                     "again"))
                if not same:
                    raise ChainHalt(f"recomputed decision for {e['phase']} differs from the recorded one"
                                    + (" - after the re-pin: the code fix changed a decided result; ask in the chat"
                                       if self.S.data.get("adoptions") else ""))
            if st == "D" and not e.get("user"):                       # second opinion Low 3: from its inputs
                g = e["decision"]
                prev = [x["decision"] for x in self.entries[:i] if x["step"] == "D" and "decision" in x]
                c = g.get("costs")
                if c:
                    again = budget_gate(g["used"], Costs(**c), tuple(prev[-1]["cuts"]) if prev else (),
                                        prev[-1]["n_tasks"] if prev else 20,
                                        selection_done=any(x["step"] == "F" for x in self.entries[:i]),
                                        d60_reserve=_d60_reserve(self.entries[:i], s),
                                        selected_seeds=3 if 11 in (prev[-1]["cuts"] if prev else ()) else 4,
                                        exposure_ok=lambda n: self.exposure_ok(n, s))
                    if {k: again[k] for k in ("cuts", "n_tasks", "fits")} != {k: g[k] for k in ("cuts", "n_tasks",
                                                                                                   "fits")}:
                        raise ChainHalt(f"recomputed gate D-r{e['round']} differs from the recorded one")
            if st == "E":
                ff = self.root / "s34" / freeze_file(self.version)
                if not ff.exists() or hashlib.sha256(ff.read_bytes()).hexdigest() != e["freeze_sha"]:
                    raise ChainHalt("the freeze file is missing or differs from its recorded sha256")
            if "settings" in e:
                s = Settings(**e["settings"])



def _check_settings(s: Settings) -> None:
    """A user's settings (D165, review L5) must be values the chain itself could produce."""
    from .agent.prompts import DOCS_WORDING
    ok = (s.wording in DOCS_WORDING and isinstance(s.grammar_always, bool) and s.action_max_tokens in (320, 640)
          and isinstance(s.n_distractors, int) and not isinstance(s.n_distractors, bool) and 0 <= s.n_distractors <= 4
          and (s.long_share is None or (isinstance(s.long_share, (int, float)) and 0 <= s.long_share <= 1)))
    if not ok:
        raise ValueError(f"invalid settings {s.to_json()}: wording one of {sorted(DOCS_WORDING)}, grammar_always a "
                         "bool, action_max_tokens 320 or 640, n_distractors 0-4 (the ladder's range), long_share null or 0-1")


def _d60_reserve(entries: list[dict], s: Settings) -> bool:
    return not any(e["step"] == "F" for e in entries) and not (s.grammar_always and s.action_max_tokens == 640)


def _read_plan_units(root: Path, phase: str) -> list[dict]:
    from .llm.runner import read_plan
    return read_plan(root, phase)["units"]


def template_set_sha() -> str:
    from .apisuite.templates import TEMPLATES
    return hashlib.sha256(canon({t: {"covers": sorted(v.covers), "long": v.long} for t, v in TEMPLATES.items()})
                          .encode()).hexdigest()
