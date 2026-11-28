"""S3.5 spec sections 4-6: the RQ1 and RQ3 primary hypotheses and the graphical procedure."""
from __future__ import annotations

import math

import numpy as np

from . import graphical as gr
from . import stats as st
from .data import PRIMARY_RQ1, PRIMARY_RQ3, UnitData, complete, index

ALPHA = 0.025
N_PLANNED = {"rq1": 14, "rq3": 10}


def _solved_pos(u: UnitData) -> dict:
    return {t["pos"]: t for t in u.tasks if t["success"]}


def _solved_n(u: UnitData) -> int:
    return sum(bool(t["success"]) for t in u.tasks)


def _n(a: UnitData, b: UnitData) -> int:
    if len(a.tasks) != len(b.tasks) or not a.tasks:
        raise ValueError(f"paired units differ in task count ({len(a.tasks)} vs {len(b.tasks)})")
    return len(a.tasks)


def success_diff(a: UnitData, b: UnitData) -> float:
    """(solved_a - solved_b) / n, from integers (review M7)."""
    return (_solved_n(a) - _solved_n(b)) / _n(a, b)


def ni_value(a: UnitData, b: UnitData) -> float:
    """(solved_a - solved_b + 1) / n: exactly 0 when a is one task worse (the NI tie, D59)."""
    return (_solved_n(a) - _solved_n(b) + 1) / _n(a, b)


def api_diff(a: UnitData, b: UnitData) -> float | None:
    """Mean API calls of a minus of b over the tasks of that stream solved by both; None if there are none."""
    sa, sb = _solved_pos(a), _solved_pos(b)
    joint = sorted(set(sa) & set(sb))
    if not joint:
        return None
    return float(np.mean([sa[p]["api_calls"] for p in joint]) - np.mean([sb[p]["api_calls"] for p in joint]))


def hypothesis(name: str, seeds: list[int], diffs: list[float | None], h1: str, level0: float,
               tested: list[float | None] | None = None) -> dict:
    """One tested hypothesis. `diffs`: per-seed differences C2 - comparator (None = no difference for that seed,
    left out and reported). `tested`: the values the sign-flip test uses (H1: mean > 0); default: diffs for
    'greater', -diffs for 'less'. NI hypotheses pass their integer-based values (review M7)."""
    tested = tested if tested is not None else [None if d is None else (d if h1 == "greater" else -d) for d in diffs]
    pairs = [(s, d, x) for s, d, x in zip(seeds, diffs, tested) if d is not None]
    left_out = [s for s, d in zip(seeds, diffs) if d is None]
    d = np.array([p[1] for p in pairs], dtype=float)
    x = np.array([p[2] for p in pairs], dtype=float)
    n = len(d)
    out = {"name": name, "h1": h1, "n": n, "seeds": [p[0] for p in pairs], "left_out_no_difference": left_out,
           "tested_values": x.tolist(), "differences": d.tolist(),
           "mean_difference": float(d.mean()) if n else None,
           "ci95": list(st.t_ci(d)) if n >= 2 else [None, None],
           "mdd": st.mdd(d, level0) if n >= 2 else None, "mdd_alpha_one_sided": level0}
    if n == 0:
        out.update({"p": None, "untied": 0, "min_attainable_p": None, "below_initial_level": True,
                    "unrejectable": True})
        return out
    out["untied"] = st.untied(x)
    out["min_attainable_p"] = 2.0 ** -out["untied"]
    out["p"] = st.sign_flip_one_sided(x)
    # second-opinion L1: two flags. below_initial_level: cannot reach the level it starts with (the S hypotheses:
    # the level they inherit from their NI test); unrejectable (the design's word): cannot reach alpha even with
    # every bit of weight passed to it.
    out["below_initial_level"] = out["min_attainable_p"] > level0
    out["unrejectable"] = out["min_attainable_p"] > ALPHA
    return out


def _fmt(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def run_graph(graph: gr.Graph, hyps: dict[str, dict]) -> dict:
    p = {n: hyps[n]["p"] for n in graph.names}
    res = gr.reject(graph, p, ALPHA)
    adj = gr.adjusted_p(graph, p)
    return {"alpha": ALPHA, "rejected": res.rejected, "steps": res.steps, "adjusted_p": adj,
            "initial_weights": dict(zip(graph.names, graph.weights))}


def rq1_primary(units: list[UnitData], phase: str, include_failed: bool = False) -> dict:
    """`phase` is required and non-empty (second-opinion L5, r3 review L1): only units of that plan phase are
    primary."""
    if not phase:
        raise ValueError("the primary analysis needs the plan phase (e.g. 's35-T')")
    by = index(units, "rq1", "rq1", phase)
    seeds, dropped = complete(by, PRIMARY_RQ1, include_failed)
    hyps = {}
    for comp, ni, sup, cost in (("C0", "NI1", "S1", "E2"), ("C1", "NI3", "S3", "E4")):
        d = [success_diff(by[s]["C2"], by[s][comp]) for s in seeds]
        hyps[ni] = hypothesis(ni, seeds, d, "greater", ALPHA / 4,
                              tested=[ni_value(by[s]["C2"], by[s][comp]) for s in seeds])
        hyps[sup] = hypothesis(sup, seeds, d, "greater", ALPHA / 4)
        hyps[cost] = hypothesis(cost, seeds, [None if by[s]["C2"].status != "done" or by[s][comp].status != "done"
                                             else api_diff(by[s]["C2"], by[s][comp]) for s in seeds],
                                "less", ALPHA / 4)
    return {"n_planned": N_PLANNED["rq1"], "n_achieved": len(seeds), "seeds": seeds, "dropped_incomplete": dropped,
            "include_failed": include_failed, "hypotheses": hyps, "graph": run_graph(gr.rq1_graph(), hyps),
            "cells": cell_counts(by, seeds, PRIMARY_RQ1)}


def cell_counts(by: dict, seeds: list[int], cells: tuple[str, ...]) -> dict:
    """Per cell over the analysed seeds: tasks, solved, unfinished at T_max, never started (spec section 3)."""
    out = {}
    for c in cells:
        ts = [t for s in seeds for t in by[s][c].tasks]
        out[c] = {"tasks": len(ts), "solved": sum(bool(t["success"]) for t in ts),
                  "unfinished_at_t_max": sum(bool(t.get("unfinished_at_t_max")) for t in ts),
                  "never_started": sum(bool(t.get("never_started")) for t in ts)}
    return out


def rq3_primary(units: list[UnitData], phase: str, include_failed: bool = False) -> dict:
    """`phase` is required and non-empty (second-opinion L5, r3 review L1): only units of that plan phase are
    primary."""
    if not phase:
        raise ValueError("the primary analysis needs the plan phase (e.g. 's35-T')")
    by = index(units, "rq3", "rq3", phase)
    cells = tuple(f"{c}+{w}" for c, w in PRIMARY_RQ3)
    seeds, dropped = complete(by, cells, include_failed)
    r1 = [success_diff(by[s]["C2+acc"], by[s]["C0+acc"]) for s in seeds]
    r2 = [success_diff(by[s]["C2+acc"], by[s]["C2+none"]) for s in seeds]
    hyps = {"NI-R1": hypothesis("NI-R1", seeds, r1, "greater", ALPHA / 2,
                                tested=[ni_value(by[s]["C2+acc"], by[s]["C0+acc"]) for s in seeds]),
            "S-R1": hypothesis("S-R1", seeds, r1, "greater", ALPHA / 2),
            "R2": hypothesis("R2", seeds, r2, "greater", ALPHA / 2)}
    inter = [success_diff(by[s]["C2+acc"], by[s]["C0+acc"]) - success_diff(by[s]["C2+none"], by[s]["C0+none"])
             for s in seeds]
    none_c0 = [by[s]["C0+none"].success_rate() for s in seeds]
    none_c2 = [by[s]["C2+none"].success_rate() for s in seeds]
    floor = bool(seeds) and np.mean(none_c0) < 0.2 and np.mean(none_c2) < 0.2
    return {"n_planned": N_PLANNED["rq3"], "n_achieved": len(seeds), "seeds": seeds, "dropped_incomplete": dropped,
            "include_failed": include_failed, "hypotheses": hyps, "graph": run_graph(gr.rq3_graph(), hyps),
            "cells": cell_counts(by, seeds, cells),
            "interaction": {"differences": inter, "mean": float(np.mean(inter)) if inter else None,
                            "ci95": list(st.t_ci(inter)) if len(inter) >= 2 else [None, None],
                            "uninformative_floor": bool(floor),
                            "mean_success_none": {"C0": _fmt(float(np.mean(none_c0))) if seeds else None,
                                                  "C2": _fmt(float(np.mean(none_c2))) if seeds else None}}}
