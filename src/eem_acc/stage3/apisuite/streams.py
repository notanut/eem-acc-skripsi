"""Stream sampler (S3.1 spec section 8, D70): drift set, error styles, constructive task placement, instances.

All randomness comes from named substreams `random.Random(f"{seed}:{component}")` (string seeding is sha512-based,
independent of PYTHONHASHSEED), so a pilot knob (`n_distractors`, `long_share`) never reshuffles drift sets or styles.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

from .drift import POOLS, SP_RENAMES, draw_drift_set, sp_pairs
from .ops import OP_NAMES
from .templates import TEMPLATES, Gen, sample_instance

SUITE_VERSION = "s31-r2"
SEED_RANGES = {("rq1", "dev"): range(1000, 2000), ("rq1", "test"): range(2000, 3000),
               ("sp", "test"): range(3000, 3100), ("rq3", "test"): range(4000, 4100)}
FIRST_WINDOW = 7          # first exposure within positions 0-6
COUNT_WINDOW = 15         # at least MIN_EXPOSURES within the first 15 positions
MIN_EXPOSURES = 3
SP_AFTER = range(10, 15)  # SP op: >= 2 exposures here (after the midpoint at lengths 15, 16 and 20)
N_AGENTS_RQ3 = 4
DEFAULT_DISTRACTORS = 2


def _rng(seed: int, component: str) -> random.Random:
    return random.Random(f"{seed}:{component}")


def covering(op: str) -> list[str]:
    return sorted(t for t, tpl in TEMPLATES.items() if op in tpl.covers)


def mix_weights(long_share: float | None) -> dict[str, float]:
    if long_share is None:
        return {t: 1.0 for t in TEMPLATES}
    n_long = sum(1 for t in TEMPLATES.values() if t.long)
    n_short = len(TEMPLATES) - n_long
    wl, ws = long_share / n_long, (1 - long_share) / n_short
    return {t: (wl if tpl.long else ws) for t, tpl in TEMPLATES.items()}


def _weighted(rng: random.Random, items: list[str], w: dict[str, float]) -> str:
    return rng.choices(items, weights=[w[i] for i in items], k=1)[0]


def place_templates(seed: int, n_tasks: int, drifted: list[str], sp_op: str | None,
                    weights: dict[str, float], max_attempts: int = 100) -> list[str]:
    """Constructive placement of templates satisfying the exposure rule (D70)."""
    if n_tasks < COUNT_WINDOW:
        raise ValueError("streams are sampled at full length (>= 15); shorter runs use prefixes")
    for attempt in range(max_attempts):
        rng = _rng(seed, f"place:{attempt}")
        slots: list[str | None] = [None] * n_tasks

        def cov(i: int, op: str) -> bool:
            return slots[i] is not None and op in TEMPLATES[slots[i]].covers

        def count(op: str, lo: int, hi: int) -> int:
            return sum(cov(i, op) for i in range(lo, hi))

        def put(op: str, positions: range, deficits: dict[str, int]) -> bool:
            free = [i for i in positions if slots[i] is None]
            if not free:
                return False
            cands = covering(op)
            score = {t: sum(1 for o, d in deficits.items() if d > 0 and o != op and o in TEMPLATES[t].covers)
                     for t in cands}
            w = {t: weights[t] * (1 + score[t]) for t in cands}     # mild preference for multi-covering templates
            slots[rng.choice(free)] = _weighted(rng, cands, w)
            return True

        order = list(drifted)
        rng.shuffle(order)
        ok = True
        # 1. first exposures in 0-6
        for op in order:
            if count(op, 0, FIRST_WINDOW) == 0:
                deficits = {o: int(count(o, 0, FIRST_WINDOW) == 0) for o in order}
                if not put(op, range(FIRST_WINDOW), deficits):
                    ok = False
                    break
        # 2. SP op: >= 2 exposures in 10-14
        if ok and sp_op is not None:
            while count(sp_op, SP_AFTER.start, SP_AFTER.stop) < 2:
                if not put(sp_op, SP_AFTER, {}):
                    ok = False
                    break
        # 3. >= 3 exposures in the first 15
        if ok:
            for op in order:
                while count(op, 0, COUNT_WINDOW) < MIN_EXPOSURES:
                    deficits = {o: MIN_EXPOSURES - count(o, 0, COUNT_WINDOW) for o in order}
                    if not put(op, range(COUNT_WINDOW), deficits):
                        ok = False
                        break
                if not ok:
                    break
        if not ok:
            continue
        all_t = sorted(TEMPLATES)
        for i in range(n_tasks):
            if slots[i] is None:
                slots[i] = _weighted(rng, all_t, weights)
        return [s for s in slots if s is not None]
    raise RuntimeError(f"exposure placement failed for seed {seed}")


def _styles(seed: int, drifted: list[str]) -> dict[str, str]:
    rng = _rng(seed, "styles")
    out = {}
    for group in (sorted(drifted), sorted(o for o in OP_NAMES if o not in drifted)):
        g = list(group)
        rng.shuffle(g)
        half = len(g) // 2
        for i, op in enumerate(g):
            out[op] = "informative" if i < half else "terse"
    return {op: out[op] for op in OP_NAMES}


def spec_hash(spec: dict) -> str:
    body = {k: v for k, v in spec.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def sample_stream(seed: int, split: str, n_tasks: int = 20, kind: str = "rq1",
                  n_distractors: int = DEFAULT_DISTRACTORS, long_share: float | None = None) -> dict:
    if (kind, split) not in SEED_RANGES or seed not in SEED_RANGES[(kind, split)]:
        raise ValueError(f"seed {seed} outside the range for kind={kind}, split={split}")
    pool = POOLS[split]
    sp_op = None
    if kind == "sp":
        sp_op = _rng(seed, "sp_op").choice(sorted(SP_RENAMES))
        pairs = draw_drift_set(_rng(seed, "drift"), pool, n_ops=7, n_double=2, exclude_ops=frozenset({sp_op}))
        p1, _ = sp_pairs(sp_op)
        start_ids = sorted([p.id for p in pairs] + [p1.id])
    else:
        pairs = draw_drift_set(_rng(seed, "drift"), pool)
        start_ids = [p.id for p in pairs]
    drifted = sorted({pid.split(":")[0] for pid in start_ids})
    templates = place_templates(seed, n_tasks, drifted, sp_op, mix_weights(long_share))
    gen = Gen(_rng(seed, "instances"), n_distractors)
    tasks = [sample_instance(t, gen, i) for i, t in enumerate(templates)]
    spec: dict[str, Any] = {
        "version": SUITE_VERSION, "seed": seed, "split": split, "kind": kind, "n_tasks": n_tasks,
        "n_distractors": n_distractors, "long_share": long_share, "drifted_ops": drifted,
        "styles": _styles(seed, drifted), "drift_schedule": [[0, start_ids]], "tasks": tasks,
    }
    if sp_op is not None:
        spec["sp_op"] = sp_op
        spec["drift_schedule"] = _sp_schedule(start_ids, sp_op, n_tasks)
    if kind == "rq3":
        spec["agents"] = [[i for i in range(n_tasks) if i % N_AGENTS_RQ3 == a] for a in range(N_AGENTS_RQ3)]
    spec["sha256"] = spec_hash(spec)
    return spec


def _sp_schedule(start_ids: list[str], sp_op: str, n: int) -> list:
    p1, p2 = sp_pairs(sp_op)
    mid_ids = sorted([i for i in start_ids if i != p1.id] + [p2.id])
    return [[0, start_ids], [n // 2, mid_ids]]


def prefix(spec: dict, n: int) -> dict:
    """The first n tasks of a stream (shortened streams are prefixes, design r7); SP's re-drift moves to n // 2."""
    if n > spec["n_tasks"]:
        raise ValueError("prefix longer than the stream")
    out = copy.deepcopy(spec)
    out["tasks"] = out["tasks"][:n]
    out["n_tasks"] = n
    out["prefix_of"] = spec["sha256"]
    if "sp_op" in out:
        out["drift_schedule"] = _sp_schedule(out["drift_schedule"][0][1], out["sp_op"], n)
    if "agents" in out:
        out["agents"] = [[i for i in a if i < n] for a in out["agents"]]
    out["sha256"] = spec_hash(out)
    return out


def exposures(spec: dict, op: str) -> list[int]:
    """Positions whose template covers op (planned exposures)."""
    return [t["index"] for t in spec["tasks"] if op in TEMPLATES[t["template"]].covers]
