"""Reference "Last working change" diff (design r7 section 6; S3.1 spec section 10; D69, D76). Shared with S3.2.

Given a note's trigger arguments (the failed call), the successful call's arguments and the documented schema, it
returns the deterministic outcome line. The **content** is the diff between the trigger and the success (D76: an
identical call, or one that differs only in values, gives no line); the **classification** of appeared keys uses the
documented schema, which keeps the leak protection:
* arguments are compared by JSON path; arrays of objects by item key path and type, never by item values;
* appeared = keys in the success that are neither in the trigger nor documented; vanished = documented keys of the
  trigger that are absent from the success (required keys only), plus the names of undocumented keys the trigger sent
  and the success dropped (D81; names only);
* if any documented *required* key is absent from the success, the operation has a rename (rename and new required
  field never share an operation, D69): appeared keys are recorded as renames, with their JSON type and no value, and
  the vanished keys are named (also when no key appeared because the trigger already sent the new key, D85);
* otherwise appeared keys are new required fields: the value is recorded only if it is a scalar, the key is not
  identifier-like (`*_id`, `id`), and the value does not occur anywhere in the trigger arguments (D69); likewise the
  changed value of an undocumented key present in both calls, if the note's own error named that key (D85);
* a documented key present in both whose JSON type changed (and no longer matches the documented type) is recorded
  without a value;
* a documented enum key present in both, whose trigger value is a documented value and whose success value is outside
  the documented values, is recorded with that value (the remap), unless the value occurs in the trigger;
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


def last_working_change(trigger_args: dict, success_args: dict, documented: tuple,
                        moved_to: str | None = None, error_keys: frozenset = frozenset(),
                        known_keys: frozenset = frozenset()) -> str | None:
    """`error_keys`: keys the note's own parsed error named (k_t); `known_keys`: undocumented keys introduced by
    earlier outcome lines on the same operation (D88)."""
    doc = schema_paths(documented)
    succ = _flatten(success_args)
    trig = _flatten(trigger_args)
    trig_vals = _scalars(trigger_args)
    parts: list[str] = []
    if moved_to:
        parts.append(f"use operation '{moved_to}'")
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
                parts.append(f"'{vanished[0]}' is now '{appeared[0]}' ({jtype(succ[appeared[0]][0])})")
            else:
                parts += [f"'{m}' removed" for m in vanished]
                parts += [f"'{a}' added ({jtype(succ[a][0])})" for a in appeared]
        else:                                   # D85 case A: the trigger already sent the new key as well
            parts += [f"'{m}' removed" for m in vanished]
    else:
        for a in appeared:
            v = succ[a][0]
            rec = f"'{a}' added ({jtype(v)})"
            scalar = not isinstance(v, (list, dict))
            if scalar and not _idlike(a) and not (_scalars(v) & trig_vals) and len(succ[a]) == 1:
                rec += f" = {json.dumps(v)}"
            parts.append(rec)
        # D85 case D: a new required field fixed after an error that named it (e.g. a constant after a literal error)
        for p, vals in succ.items():
            if p in doc or p not in trig or p not in error_keys or len(vals) != 1 or len(trig[p]) != 1:
                continue
            v = vals[0]
            if (v != trig[p][0] and not isinstance(v, (list, dict)) and not _idlike(p)
                    and not (_scalars(v) & trig_vals)):
                parts.append(f"'{p}' = {json.dumps(v)}")
    for p, vals in succ.items():
        if p not in doc or p not in trig:
            continue
        f = doc[p]
        tvals = trig[p]
        if not all(_type_ok(f, v) for v in vals):
            if {jtype(v) for v in vals} != {jtype(v) for v in tvals}:
                parts.append(f"'{p}' is now {jtype(vals[0])}")
            continue
        if (f.kind == "enum" and len(vals) == 1 and vals[0] not in f.values and len(tvals) == 1
                and tvals[0] in f.values and not (_scalars(vals[0]) & trig_vals)):
            parts.append(f"'{p}' = {json.dumps(vals[0])}")
    if not parts:
        return None
    return "Last working change: " + "; ".join(parts)
