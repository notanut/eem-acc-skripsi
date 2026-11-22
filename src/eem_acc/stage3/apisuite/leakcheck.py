"""Reference "Last working change" diff (design r7 section 6; S3.1 spec section 10; D69, D76). Shared with S3.2.

Given a note's trigger arguments (the failed call), the successful call's arguments and the documented schema, it
returns the deterministic outcome line. The **content** is the diff between the trigger and the success (D76: an
identical call, or one that differs only in values, gives no line); the **classification** of appeared keys uses the
documented schema, which keeps the leak protection:
* arguments are compared by JSON path; arrays of objects by item key path and type, never by item values;
* appeared = keys in the success that are neither in the trigger nor documented; vanished = documented keys of the
  trigger that are absent from the success (required keys only), plus the names of undocumented keys the trigger sent
  and the success dropped if an earlier outcome line on the same operation introduced them (`known_keys`; D81, D88;
  names only);
* if any documented *required* key is absent from the success, the operation has a rename (rename and new required
  field never share an operation, D69): appeared keys are recorded as renames, with their JSON type and no value, and
  the vanished keys are named (also when no key appeared because the trigger already sent the new key, D85);
* otherwise appeared keys are new required fields: the value is recorded only if it is a scalar, the key is not
  identifier-like (`*_id`, `id`), and the value does not occur anywhere in the trigger arguments (D69); likewise the
  changed value of an undocumented key present in both calls, if the note's own error named that key (D85);
* a documented key present in both whose JSON type changed (and no longer matches the documented type) is recorded
  without a value;
* a documented enum key present in both, whose trigger value is a documented value and whose success value is outside
  the documented values, is recorded as the remap "old -> new" (D102; both are schema values), unless the new value
  occurs in the trigger;
* a moved operation records the new operation id (the caller passes `moved_to` only for notes on that route).

Safety relies on the success arguments having passed the suite's strict validator: a new required field's value is
then an enum value, a boolean or a constant by construction (spec section 6).
"""
from __future__ import annotations

import json
from typing import Any

from .schema import Field, schema_paths

JSON_KIND = {"string": "string", "id": "string", "integer": "integer", "number": "number", "boolean": "boolean",
             "enum": "string", "const": None, "digits": "string", "decimal": "string", "int01": "integer",
             "array_obj": "array", "array_str": "array"}


def jtype(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


def _flatten(args: dict) -> dict[str, list]:
    """path -> list of values (one per array item for item paths)."""
    out: dict[str, list] = {}
    for k, v in args.items():
        out.setdefault(k, []).append(v)
        if isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            for it in v:
                for ik, iv in it.items():
                    out.setdefault(f"{k}[].{ik}", []).append(iv)
    return out


def _scalars(x: Any) -> set[str]:
    if isinstance(x, dict):
        return set().union(*(_scalars(v) for v in x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*(_scalars(v) for v in x)) if x else set()
    return {json.dumps(x), str(x)}


def _idlike(path: str) -> bool:
    last = path.split("[].")[-1].lower()
    return last == "id" or last.endswith("_id")


def _type_ok(f: Field, v: Any) -> bool:
    want = JSON_KIND[f.kind]
    got = jtype(v)
    return want is None or got == want or (want == "number" and got == "integer")


def last_working_items(trigger_args: dict, success_args: dict, documented: tuple,
                       moved_to: str | None = None, error_keys: frozenset = frozenset(),
                       known_keys: frozenset = frozenset()) -> list[dict]:
    """The structured items of the outcome line (S3.2a spec 6.4). `error_keys`: keys the note's own parsed error named
    (k_t); `known_keys`: undocumented keys introduced by earlier outcome lines on the same operation (D88).
    Item kinds: moved {op}; rename {old, new, type}; removed {key}; added {key, type[, value]}; type {key, type};
    value {key, value}; remap {key, from, value} (D102)."""
    doc = schema_paths(documented)
    succ = _flatten(success_args)
    trig = _flatten(trigger_args)
    trig_vals = _scalars(trigger_args)
    items: list[dict] = []
    if moved_to:
        items.append({"kind": "moved", "op": moved_to})
    appeared = [p for p in succ if p not in doc and p not in trig]
    # vanished: documented required keys of the trigger, plus the names (never values) of undocumented keys the
    # trigger sent and the success dropped, if an earlier outcome line on this operation introduced them (D81, D88)
    vanished = [p for p in trig if p not in succ and
                ((p in doc and doc[p].required) or (p not in doc and p in known_keys))]
    renamed_op = any(f.required and p not in succ and ("[]." not in p or p.split("[].")[0] in succ)
                     for p, f in doc.items())
    if renamed_op:
        if appeared:
            if len(vanished) == 1 and len(appeared) == 1:
                items.append({"kind": "rename", "old": vanished[0], "new": appeared[0],
                              "type": jtype(succ[appeared[0]][0])})
            else:
                items += [{"kind": "removed", "key": m} for m in vanished]
                items += [{"kind": "added", "key": a, "type": jtype(succ[a][0])} for a in appeared]
        else:                                   # D85 case A: the trigger already sent the new key as well
            items += [{"kind": "removed", "key": m} for m in vanished]
    else:
        for a in appeared:
            v = succ[a][0]
            it = {"kind": "added", "key": a, "type": jtype(v)}
            scalar = not isinstance(v, (list, dict))
            if scalar and not _idlike(a) and not (_scalars(v) & trig_vals) and len(succ[a]) == 1:
                it["value"] = v
            items.append(it)
        # D85 case D: a new required field fixed after an error that named it (e.g. a constant after a literal error)
        for p, vals in succ.items():
            if p in doc or p not in trig or p not in error_keys or len(vals) != 1 or len(trig[p]) != 1:
                continue
            v = vals[0]
            if (v != trig[p][0] and not isinstance(v, (list, dict)) and not _idlike(p)
                    and not (_scalars(v) & trig_vals)):
                items.append({"kind": "value", "key": p, "value": v})
    for p, vals in succ.items():
        if p not in doc or p not in trig:
            continue
        f = doc[p]
        tvals = trig[p]
        if not all(_type_ok(f, v) for v in vals):
            if {jtype(v) for v in vals} != {jtype(v) for v in tvals}:
                items.append({"kind": "type", "key": p, "type": jtype(vals[0])})
            continue
        if (f.kind == "enum" and len(vals) == 1 and vals[0] not in f.values and len(tvals) == 1
                and tvals[0] in f.values and not (_scalars(vals[0]) & trig_vals)):
            # D102: the remap is recorded as old -> new (both are schema values), not only the value of one task
            items.append({"kind": "remap", "key": p, "from": tvals[0], "value": vals[0]})
    return items


def render_item(it: dict) -> str:
    k = it["kind"]
    if k == "moved":
        return f"use operation '{it['op']}'"
    if k == "rename":
        return f"'{it['old']}' is now '{it['new']}' ({it['type']})"
    if k == "removed":
        return f"'{it['key']}' removed"
    if k == "added":
        return f"'{it['key']}' added ({it['type']})" + (f" = {json.dumps(it['value'])}" if "value" in it else "")
    if k == "type":
        return f"'{it['key']}' is now {it['type']}"
    if k == "value":
        return f"'{it['key']}' = {json.dumps(it['value'])}"
    if k == "remap":
        return f"'{it['key']}': {json.dumps(it['from'])} -> {json.dumps(it['value'])}"
    raise ValueError(k)


def render_items(items: list[dict]) -> str | None:
    return ("Last working change: " + "; ".join(render_item(i) for i in items)) if items else None


def item_holds(it: dict, success_args: dict, called_op: str) -> bool:
    """Item-wise re-check of a possibly-outdated line against a later successful call (S3.2a spec 6.4)."""
    succ = _flatten(success_args)
    k = it["kind"]
    if k == "moved":
        return called_op == it["op"]
    if k == "rename":
        return it["new"] in succ and jtype(succ[it["new"]][0]) == it["type"] and it["old"] not in succ
    if k == "removed":
        return it["key"] not in succ
    if k == "added":
        return (it["key"] in succ and jtype(succ[it["key"]][0]) == it["type"]
                and ("value" not in it or succ[it["key"]] == [it["value"]]))
    if k == "type":
        return it["key"] in succ and jtype(succ[it["key"]][0]) == it["type"]
    if k == "value":
        return it["key"] in succ and succ[it["key"]] == [it["value"]]
    if k == "remap":                       # the field still takes remapped (non-documented) values
        return it["key"] in succ and succ[it["key"]] == [it["value"]]
    raise ValueError(k)


def last_working_change(trigger_args: dict, success_args: dict, documented: tuple,
                        moved_to: str | None = None, error_keys: frozenset = frozenset(),
                        known_keys: frozenset = frozenset()) -> str | None:
    """The rendered outcome line (see `last_working_items`)."""
    return render_items(last_working_items(trigger_args, success_args, documented, moved_to, error_keys, known_keys))
