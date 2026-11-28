"""S3.5 spec section 9.3 (pending items before the freeze), on the S3.4 pilot payloads (dev data, results already
known; C0 and C2 on pilot streams 1002-1006, phase s34-B-r1):

1. the dev-data dry run: the RQ1 primary analysis on real LLM payloads (read as if C0/C2 were a T-phase campaign;
   C1 and C3 are absent, so the complete-stream rule leaves 0 streams for the four-condition tests; the C2 - C0
   comparison is run directly), and every secondary/mechanism function that applies to C0/C2;
2. the D209 pilot check: every C2 note (events and final notes) through the strict and loose type matchers, with
   the counts of the stated limits (kind before field, five-word gaps) for a person to read.

    python tools/s35_dev_check.py <dir with s34-B-r1-rq1-s100X-{C0,C2}-*.json> [out.json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eem_acc.stage3.analysis import data as D, endpoints as E, mechanism as M, secondary as S   # noqa: E402
from eem_acc.stage3.llm import runner as R                                                       # noqa: E402

PHASE = "s34-B-r1"


def load(folder: Path) -> list[D.UnitData]:
    out = []
    for f in sorted(folder.glob(f"{PHASE}-rq1-s*-C[02]-*.json")):
        payload, meta = R.split_result(json.loads(f.read_text()))
        out.append(D.from_result({**payload, "run": meta}))
    return out


def with_standins(units: list[D.UnitData]) -> list[D.UnitData]:
    """The frozen analysis compares C0-C3 over complete streams only; the pilot has C0 and C2. Stand-in C1 (a copy
    of C0) and C3 (a copy of C2) complete each stream so the population rule keeps the real units. Nothing computed
    for C1 or C3 is reported."""
    import dataclasses
    extra = [dataclasses.replace(u, condition={"C0": "C1", "C2": "C3"}[u.condition]) for u in units]
    return units + extra


def c2_minus_c0(units):
    by = D.index(units, "rq1", "rq1", PHASE)
    seeds = [s for s in sorted(by) if {"C0", "C2"} <= set(by[s])]
    d = [E.success_diff(by[s]["C2"], by[s]["C0"]) for s in seeds]
    api = [E.api_diff(by[s]["C2"], by[s]["C0"]) for s in seeds]
    return {"seeds": seeds, "success": E.hypothesis("S1-dev", seeds, d, "greater", E.ALPHA / 4),
            "ni": E.hypothesis("NI1-dev", seeds, d, "greater", E.ALPHA / 4,
                               tested=[E.ni_value(by[s]["C2"], by[s]["C0"]) for s in seeds]),
            "api": E.hypothesis("E2-dev", seeds, api, "less", E.ALPHA / 4)}


def pilot_check(units) -> dict:
    """D209: strict vs loose on every type-drift note of C2, plus the stated limits, with the texts for reading."""
    rows, counts = [], {"type_notes": 0, "strict": 0, "loose": 0, "kind_before_field": 0}
    for u in units:
        if u.condition != "C2":                       # the real C2 units only (stand-ins are C3)
            continue
        for e in u.raw.get("note_events", []) + u.raw.get("final_notes", []):
            tr = e.get("truth") or {}
            for pid in tr.get("pair_ids", []):
                if M._pair(pid).type != "type":
                    continue
                text = M.note_text(e)
                p = M._pair(pid)
                field, words = M._last(p.path), M.kind_words(p.new_kind)
                strict = M.names_truth(text, pid, e.get("items"))
                loose = M.names_truth(text, pid, e.get("items"), loose=True)
                kfirst = bool(re.search("(?i:" + "|".join(map(re.escape, words)) + r")\W+(?:\w+\W+){0,6}?"
                                        + re.escape(field), text))
                counts["type_notes"] += 1
                counts["strict"] += strict
                counts["loose"] += loose
                counts["kind_before_field"] += kfirst and not loose
                rows.append({"unit": f"{u.seed}-{u.condition}", "event": e.get("event", "final"), "pair": pid,
                             "strict": strict, "loose": loose, "kind_before_field": kfirst, "text": text[:400]})
    return {"counts": counts, "rows": rows}


def main(folder: Path) -> dict:
    real = load(folder)
    if not real:
        raise SystemExit(f"no {PHASE} C0/C2 unit files in {folder}")
    us = with_standins(real)
    return {"what": "S3.5 dev-data checks on the S3.4 pilot (dev data, results already known)",
            "units": [f"{u.seed}-{u.condition}" for u in real],
            "standins": "C1 = copy of C0, C3 = copy of C2 (population rule only; never reported)",
            "c2_minus_c0": c2_minus_c0(us),
            "cost_per_solved": S.cost_per_solved(us, ("C0", "C2"), PHASE),
            "error_splits": S.error_splits(us, ("C0", "C2"), PHASE),
            "exposures": S.exposures(us, ("C0", "C2"), PHASE),
            "drifted_call_share": S.drifted_call_share(us, ("C0", "C2"), PHASE),
            "mechanism": {"repeat_exposure": M.repeat_exposure(us, ("C0", "C2"), (PHASE,)),
                          "note_correctness": M.note_correctness(us, ("C2",), (PHASE,)),
                          "note_precision": M.note_precision(us, ("C2",), (PHASE,)),
                          "retrieval_recall": M.retrieval_recall(us, ("C2",), (PHASE,))},
            "d209_pilot_check": pilot_check(us)}


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    text = json.dumps(out, indent=1, default=str)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    print(json.dumps({k: out[k] for k in ("units",)}, indent=1), json.dumps(out["d209_pilot_check"]["counts"]))
