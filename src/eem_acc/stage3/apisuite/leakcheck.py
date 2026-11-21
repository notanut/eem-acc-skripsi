"""Reference "Last working change" diff (design r7 section 6; S3.1 spec section 10; D69). Shared with S3.2.

Given a note's trigger arguments (the failed call), the successful call's arguments and the documented schema, it
returns the deterministic outcome line. Rules:
* arguments are compared by JSON path; arrays of objects by item key path and type, never by item values;
* keys that appeared and are not documented are recorded with their JSON type;
* if any documented *required* key is absent from the successful call, the appeared keys are renames (rename and new
  required field never share an operation, D69): no value is recorded for them, and the vanished keys are named;
* otherwise appeared keys are new required fields: their value is recorded only if it is a scalar, the key is not
  identifier-like (`*_id`, `id`), and the value does not occur anywhere in the trigger arguments (D69);
* JSON type changes of documented keys are recorded without values;
* for documented enum fields, a value outside the documented values is recorded (the remapped value), unless it
  occurs in the trigger arguments;
* a moved operation records the new operation id.
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
                        moved_to: str | None = None) -> str | None:
    doc = schema_paths(documented)
    succ = _flatten(success_args)
    trig_vals = _scalars(trigger_args)
    parts: list[str] = []
    if moved_to:
        parts.append(f"use operation '{moved_to}'")
    appeared = [p for p in succ if p not in doc]
    missing = []
    for p, f in doc.items():
        if not f.required or p in succ:
            continue
        if "[]." in p and p.split("[].")[0] not in succ:
            continue
        missing.append(p)
    if missing:
        if len(missing) == 1 and len(appeared) == 1:
            parts.append(f"'{missing[0]}' is now '{appeared[0]}' ({jtype(succ[appeared[0]][0])})")
        else:
            parts += [f"'{m}' removed" for m in missing]
            parts += [f"'{a}' added ({jtype(succ[a][0])})" for a in appeared]
    else:
        for a in appeared:
            v = succ[a][0]
            rec = f"'{a}' added ({jtype(v)})"
            scalar = not isinstance(v, (list, dict))
            if scalar and not _idlike(a) and not (_scalars(v) & trig_vals) and len(succ[a]) == 1:
                rec += f" = {json.dumps(v)}"
            parts.append(rec)
    for p, vals in succ.items():
        if p not in doc:
            continue
        f = doc[p]
        if not all(_type_ok(f, v) for v in vals):
            parts.append(f"'{p}' is now {jtype(vals[0])}")
            continue
        if f.kind == "enum" and len(vals) == 1 and vals[0] not in f.values and not (_scalars(vals[0]) & trig_vals):
            parts.append(f"'{p}' = {json.dumps(vals[0])}")
    if not parts:
        return None
    return "Last working change: " + "; ".join(parts)
