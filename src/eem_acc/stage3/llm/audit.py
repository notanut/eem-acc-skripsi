"""Scrub audit (S3.3a spec section 8): every stored note, reflection, k_t and A1 text of a phase's results is searched
for the unit's task entities and instruction values; placeholders left in call arguments are counted."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ENTITY_KEYS = ("customer_id", "email", "name", "sku", "order_id", "payment_id", "shipment_id", "ticket_id", "subject")
AMOUNT_KEYS = ("price", "total", "amount")


def stream_entities(spec: dict) -> set[str]:
    """The S3.2 extended entity set: entity ids and names, large amounts, product names, ticket subjects, and the
    instruction's free-text parameters."""
    out: set[str] = set()
    for t in spec["tasks"]:
        for ents in t["setup"].values():
            for e in ents:
                for k, v in e.items():
                    if k in ENTITY_KEYS and isinstance(v, str):
                        out.add(v)
                    if k in AMOUNT_KEYS and isinstance(v, int) and v >= 1000:
                        out.add(str(v))
        for p in t["setup"].get("products", []):
            out.add(p["name"])
        for tk in t["setup"].get("tickets", []):
            out.add(tk["subject"])
        for k in ("subject", "text"):
            if k in t["params"]:
                out.add(t["params"][k])
    return out


def _notes(result: dict) -> list[dict]:
    fn = result.get("final_notes") or []
    if fn and isinstance(fn[0], list):                    # C2-private: one list per agent
        fn = [n for lst in fn for n in lst]
    return fn


def audit_result(result: dict) -> list[dict]:
    from ..apisuite import sample_stream
    u = result["unit"]
    spec = sample_stream(u["seed"], u["split"], kind=u["kind"])
    spec = {**spec, "tasks": spec["tasks"][:u.get("n_tasks", 20)]}
    ents = stream_entities(spec)
    hits = []
    for n in _notes(result):
        shown = json.dumps({k: n.get(k) for k in ("kt", "reflection", "text_raw", "line", "line_display")})
        for e in ents:
            if e in shown:
                hits.append({"unit": result["unit"], "where": "note", "note": n.get("id"), "value": e})
    for st in result.get("steps", []):
        if "kt" in st:
            blob = json.dumps(st["kt"])
            for e in ents:
                if e in blob:
                    hits.append({"unit": result["unit"], "where": "step_kt", "step": st.get("step"), "value": e})
    return hits


def audit_scrub(results: list[dict]) -> dict:
    by_cond: dict[str, int] = defaultdict(int)
    placeholders: dict[str, int] = defaultdict(int)
    hits = []
    for r in results:
        h = audit_result(r)
        hits += h
        by_cond[r["unit"]["condition"]] += len(h)
        placeholders[r["unit"]["condition"]] += sum(t.get("placeholder_args", 0) for t in r.get("tasks", []))
    return {"hits": hits, "hits_by_condition": dict(by_cond), "placeholder_args_by_condition": dict(placeholders),
            "units": len(results)}


def audit_phase(root: Path, phase: str) -> dict:
    results = [json.loads(p.read_text()) for p in sorted((Path(root) / "units" / phase).glob("*.json"))
               if not p.name.endswith(".discarded.json")]
    return audit_scrub(results)
