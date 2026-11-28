"""S3.5 spec section 7 (descriptive arms, SP, RQ3 extras): estimates and t intervals only, no tests."""
from __future__ import annotations

import numpy as np

from .data import PRIMARY_RQ1, UnitData, complete, index
from .mechanism import names_truth, note_text
from .endpoints import api_diff, success_diff
from .secondary import _mean, completion, summary

ARMS = ("F", "AN", "R", "A1", "A2", "C2-OU")
DESC_STREAMS = tuple(range(2000, 2005))
UPDATE_EVENTS = ("create", "merge", "close")


def arms_vs_c2(units: list[UnitData], phases=("s35-T", "s35-S"), arms=ARMS) -> dict:
    """Each arm on streams 2000-2004 next to C0-C3 there: means, and paired differences against C2 (success, API
    calls over jointly solved tasks) with t intervals; AN's prompt tokens per task."""
    by = index(units, "rq1", "rq1", phases)
    # the streams where the descriptive arms ran (registered: 2000-2004); derived from the data, not hard-coded
    # (the dry run found a hard-coded list silently empties the table on other seeds)
    streams = sorted(s for s in by if any(a in by[s] for a in arms))
    full = set(complete({s: by[s] for s in streams}, PRIMARY_RQ1)[0])
    out = {"streams": streams, "registered_streams": list(DESC_STREAMS), "arms": {}}

    def row_for(arm, seeds):
        us = [by[s][arm] for s in seeds if arm in by[s] and by[s][arm].status == "done"]
        return {"streams": [u.seed for u in us], "units": len(us), "success": _mean([u.success_rate() for u in us]),
                "api_calls_per_task": _mean([t.get("api_calls") for u in us for t in u.tasks]),
                "tok_in_per_task": _mean([t.get("tok_in") for u in us for t in u.tasks])}

    for arm in PRIMARY_RQ1:                             # the primary arms on the complete streams only (review M5)
        out["arms"][arm] = row_for(arm, sorted(full))
    for arm in arms:                       # each arm on its own streams (Part 1 second opinion M1)
        seeds = [s for s in sorted(by) if arm in by[s] and "C2" in by[s]
                 and by[s][arm].status == "done" and by[s]["C2"].status == "done"]
        row = row_for(arm, seeds)
        row["C2_on_these_streams"] = row_for("C2", seeds)          # C2's level on this arm's own streams (M5)
        row["minus_C2_success"] = summary(seeds, [success_diff(by[s][arm], by[s]["C2"]) for s in seeds], False)
        row["minus_C2_api_calls"] = summary(seeds, [api_diff(by[s][arm], by[s]["C2"]) for s in seeds], False)
        out["arms"][arm] = row
    return out


def _redrift(spec: dict) -> tuple[int, str] | None:
    """SP: the task index where the second drift stage starts and the SP operation (`spec["sp_op"]`, checked against
    the schedule: its pair changes sp1 -> sp2 there; code review L6)."""
    sched = sorted(spec["drift_schedule"], key=lambda e: e[0])
    if len(sched) < 2:
        return None
    before, after = set(sched[0][1]), set(sched[1][1])
    changed = sorted({p.partition(":")[0] for p in after - before})
    op = spec.get("sp_op") or (changed[0] if changed else None)
    if op is None or changed != [op] or f"{op}:sp2" not in after:
        raise ValueError(f"SP schedule does not re-drift its sp_op ({op}; changed {changed})")
    return sched[1][0], op


def sp(units: list[UnitData], phases=("s35-S",)) -> dict:
    """SP (C0, C2 on seeds 3000-3004): success before/after the re-drift; first-call outcome at the 1st, 2nd, ...
    exposure to the re-drifted operation after it; whether C2's note on that operation was updated after it."""
    by = index(units, "rq1", "sp", phases)
    out = {"seeds": sorted(by), "conditions": {}}
    for c in ("C0", "C2"):
        before, after, expo, updated, named = [], [], {}, [], []
        for s in sorted(by):
            u = by[s].get(c)
            if u is None or u.status != "done" or not u.spec:
                continue
            rd = _redrift(u.spec)
            if rd is None:
                continue
            start, op = rd
            before.append(np.mean([t["success"] for t in u.tasks if t["pos"] < start]))
            after.append(np.mean([t["success"] for t in u.tasks if t["pos"] >= start]))
            k = 0
            for t in sorted(u.tasks, key=lambda t: t["pos"]):
                fc = (t.get("first_call") or {}).get(op)
                if t["pos"] >= start and fc is not None:
                    k += 1
                    expo.setdefault(k, []).append(fc == "ok")
            if c == "C2":
                upd = [e for e in u.raw.get("note_events", []) if e.get("event") in UPDATE_EVENTS
                       and e.get("suite_op") == op and (e.get("at") or [0, -1])[1] >= start]
                updated.append(bool(upd))
                # D205 strict: an update after the re-drift whose note names the second-stage name
                named.append(any(names_truth(note_text(e), f"{op}:sp2", e.get("items")) for e in upd))
        out["conditions"][c] = {"units": len(before), "success_before": _mean(before), "success_after": _mean(after),
                                "first_call_ok_by_exposure": {str(k): {"n": len(v), "ok": float(np.mean(v))}
                                                              for k, v in sorted(expo.items())}}
        if c == "C2":
            out["conditions"][c]["note_updated_after_redrift"] = {"units": len(updated), "updated": sum(updated),
                                                                  "updated_naming_new_name": sum(named)}
    return out


def rq3_extras(units: list[UnitData], phases=("s35-T", "s35-S")) -> dict:
    """Private vs shared store (C2-private+acc vs C2+acc, seeds 4000-4004); the sensitivity 2x2 (tag 'sens', LLM time
    / 3) with the R1 and R2 differences as estimates."""
    by = index(units, "rq3", "rq3", phases)
    seeds = [s for s in sorted(by) if "C2-private+acc" in by[s] and "C2+acc" in by[s]
             and by[s]["C2-private+acc"].status == "done" and by[s]["C2+acc"].status == "done"]
    out = {"private_minus_shared": {
        "success": summary(seeds, [success_diff(by[s]["C2-private+acc"], by[s]["C2+acc"]) for s in seeds], False),
        "completion_finished": summary(seeds, [_cd(by[s]["C2-private+acc"], by[s]["C2+acc"]) for s in seeds], False)}}
    sens = index(units, "rq3", "rq3", phases, tag="sens")
    cells = ("C2+acc", "C0+acc", "C2+none", "C0+none")
    ss = [s for s in sorted(sens) if all(c in sens[s] and sens[s][c].status == "done" for c in cells)]
    out["sensitivity"] = {
        "seeds": ss,
        "R1_C2_minus_C0_acc": summary(ss, [success_diff(sens[s]["C2+acc"], sens[s]["C0+acc"]) for s in ss], False),
        "R2_acc_minus_none_C2": summary(ss, [success_diff(sens[s]["C2+acc"], sens[s]["C2+none"]) for s in ss], False),
        "cells_success": {c: _mean([sens[s][c].success_rate() for s in ss]) for c in cells}}
    return out


def _cd(a: UnitData, b: UnitData):
    x, y = completion(a)["finished_mean"], completion(b)["finished_mean"]
    return None if x is None or y is None else x - y
