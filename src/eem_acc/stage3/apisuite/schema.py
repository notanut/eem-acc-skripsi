"""Field specifications and the strict request validator (S3.1 spec sections 3-4, D72).

The validator imitates pydantic v2's error item format and `type` codes, in strict mode, without echoing the input
(`input` is never included, D72), so messages are fixed and never carry entity values. All items of a call are
collected; the caller orders drift items first.

Deviation from pydantic (documented in the spec): within an array of objects only the first item per (path, type)
is reported, so bodies stay short.

A *path* names a field: a top-level key ``"amount"`` or a key inside the objects of an array ``"items[].qty"``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

# Field kinds. "id" is a non-empty string documented as an identifier.
KINDS = ("string", "integer", "number", "boolean", "enum", "const", "id",
         "digits", "decimal", "int01", "array_obj", "array_str")

DIGITS_RE = re.compile(r"^\d+$")
DECIMAL_RE = re.compile(r"^\d+(\.\d{1,2})?$")


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    required: bool = True
    values: tuple = ()                 # enum values (in display order)
    const: Any = None                  # const value
    items: tuple = ()                  # array_obj: tuple of Field
    default: Any = None                # documented default of an optional field (display only)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind}")
        if self.kind == "enum" and not self.values:
            raise ValueError(f"enum field {self.name} without values")
        if self.kind == "array_obj" and not self.items:
            raise ValueError(f"array_obj field {self.name} without item fields")

    def with_(self, **kw) -> "Field":
        return replace(self, **kw)


Schema = tuple  # tuple[Field, ...] in display order


def schema_paths(schema: Schema) -> dict[str, Field]:
    """All paths of a schema -> Field (array item fields as ``parent[].child``)."""
    out: dict[str, Field] = {}
    for f in schema:
        out[f.name] = f
        if f.kind == "array_obj":
            for g in f.items:
                out[f"{f.name}[].{g.name}"] = g
    return out


def _quote(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


def expected_text(values: tuple) -> str:
    """pydantic v2 style: "'a', 'b' or 'c'"."""
    q = [_quote(v) for v in values]
    return q[0] if len(q) == 1 else ", ".join(q[:-1]) + " or " + q[-1]


@dataclass(frozen=True)
class ErrorItem:
    type: str
    loc: tuple                 # ("body", key) or ("body", key, index, subkey)
    msg: str
    path: str                  # schema path the item concerns ("items[].qty", or the extra key's path)
    ctx: dict | None = None

    def as_json(self) -> dict:
        d = {"type": self.type, "loc": list(self.loc), "msg": self.msg}
        if self.ctx:
            d["ctx"] = dict(self.ctx)
        return d


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_number(v: Any) -> bool:
    return (type(v) is int) or (type(v) is float)


def check_value(f: Field, v: Any, loc: tuple, path: str) -> list[ErrorItem]:
    """Strict type/enum/const check of one present value."""
    k = f.kind
    if k in ("string", "id"):
        if not isinstance(v, str):
            return [ErrorItem("string_type", loc, "Input should be a valid string", path)]
        if k == "id" and not v:
            return [ErrorItem("string_too_short", loc, "String should have at least 1 character", path)]
        return []
    if k == "integer":
        return [] if _is_int(v) else [ErrorItem("int_type", loc, "Input should be a valid integer", path)]
    if k == "number":
        if not _is_number(v):
            return [ErrorItem("float_type", loc, "Input should be a valid number", path)]
        if isinstance(v, float) and not math.isfinite(v):
            return [ErrorItem("finite_number", loc, "Input should be a finite number", path)]
        return []
    if k == "boolean":
        return [] if type(v) is bool else [ErrorItem("bool_type", loc, "Input should be a valid boolean", path)]
    if k == "enum":
        if isinstance(v, str) and v in f.values:
            return []
        exp = expected_text(f.values)
        return [ErrorItem("enum", loc, f"Input should be {exp}", path, {"expected": exp})]
    if k == "const":
        if type(v) is type(f.const) and v == f.const:
            return []
        exp = _quote(f.const)
        return [ErrorItem("literal_error", loc, f"Input should be {exp}", path, {"expected": exp})]
    if k in ("digits", "decimal"):
        if not isinstance(v, str):
            return [ErrorItem("string_type", loc, "Input should be a valid string", path)]
        rx = DIGITS_RE if k == "digits" else DECIMAL_RE
        if not rx.match(v):
            return [ErrorItem("string_pattern_mismatch", loc, f"String should match pattern '{rx.pattern}'", path,
                              {"pattern": rx.pattern})]
        return []
    if k == "int01":
        if not _is_int(v):
            return [ErrorItem("int_type", loc, "Input should be a valid integer", path)]
        if v not in (0, 1):
            return [ErrorItem("literal_error", loc, "Input should be 0 or 1", path, {"expected": "0 or 1"})]
        return []
    if k == "array_str":
        if not isinstance(v, list):
            return [ErrorItem("list_type", loc, "Input should be a valid list", path)]
        out = []
        for i, x in enumerate(v):
            if not isinstance(x, str):
                out.append(ErrorItem("string_type", loc + (i,), "Input should be a valid string", path))
        if not v:
            out.append(ErrorItem("too_short", loc, "List should have at least 1 item after validation, not 0", path))
        return out
    if k == "array_obj":
        if not isinstance(v, list):
            return [ErrorItem("list_type", loc, "Input should be a valid list", path)]
        if not v:
            return [ErrorItem("too_short", loc, "List should have at least 1 item after validation, not 0", path)]
        out = []
        seen: set = set()
        for i, item in enumerate(v):            # one item per (path, type) across array elements (keeps bodies short)
            for e in validate_object(f.items, item, loc + (i,), prefix=f"{f.name}[]."):
                if (e.path, e.type) not in seen:
                    seen.add((e.path, e.type))
                    out.append(e)
        return out
    raise AssertionError(k)


def validate_object(schema: Schema, obj: Any, loc: tuple = ("body",), prefix: str = "",
                    skip_paths: frozenset = frozenset()) -> list[ErrorItem]:
    """Validate one object against a schema (strict, extra keys forbidden). `skip_paths` are paths whose checks are
    deferred to the app level (silent drift): no missing/type/extra item is produced for them here."""
    if not isinstance(obj, dict):
        return [ErrorItem("dict_type" if prefix else "model_type", loc,
                          "Input should be a valid dictionary" if prefix else
                          "Input should be a valid dictionary or instance of the model", prefix.rstrip(".") or "")]
    out: list[ErrorItem] = []
    known = {f.name for f in schema}
    for f in schema:
        path = prefix + f.name
        if path in skip_paths:
            continue
        if f.name not in obj:
            if f.required:
                out.append(ErrorItem("missing", loc + (f.name,), "Field required", path))
            continue
        out.extend(check_value(f, obj[f.name], loc + (f.name,), path))
    for key in obj:                      # dict order = the agent's order; deterministic
        path = prefix + str(key)
        if key not in known and path not in skip_paths:
            out.append(ErrorItem("extra_forbidden", loc + (key,), "Extra inputs are not permitted", path))
    return out
