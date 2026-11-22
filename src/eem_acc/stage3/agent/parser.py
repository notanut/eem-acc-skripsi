"""Deterministic k_t parser (S3.2a spec 6.1, D99). Written against the S3.1 error formats (stated as a limitation:
the parser and the formats have the same author)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..apisuite.leakcheck import jtype
from ..apisuite.ops import OPS
from ..apisuite.schema import schema_paths

PARSER_VERSION = "kt-v3"
CODE_TYPES = {
    "missing": "missing_field", "extra_forbidden": "unknown_field",
    "string_type": "type_error", "int_type": "type_error", "float_type": "type_error", "bool_type": "type_error",
    "list_type": "type_error", "dict_type": "type_error", "model_type": "type_error",
    "enum": "enum_error", "literal_error": "literal_error", "string_pattern_mismatch": "format_error",
    "string_too_short": "value_error", "too_short": "value_error", "finite_number": "value_error",
}
SILENT_RE = re.compile(r"^Validation failed: field '([^']+)': (.*)$")
MOVED_RE = re.compile(r"Use '([^']+)'")
TYPE_WORD = {"string_type": "string", "int_type": "integer", "float_type": "number", "bool_type": "boolean",
             "list_type": "list", "dict_type": "object", "model_type": "object"}


def _documented_enum_values() -> dict[str, set]:
    """field name -> its documented enum values (the agent's docs). No drift catalogue is read (second opinion M2)."""
    out: dict[str, set] = {}
    for d in OPS.values():
        for p, f in schema_paths(d.schema).items():
            if f.kind == "enum":
                out.setdefault(p.split("[].")[-1], set()).update(f.values)
    return out


DOC_ENUM = _documented_enum_values()


@dataclass(frozen=True)
class KT:
    api_name: str
    expected: str
    received: str
    error_type: str
    error_keys: frozenset = frozenset()
    stated_values: frozenset = frozenset()      # schema values the error itself states (enum/literal expected)

    def render(self) -> str:
        return f"{self.api_name} ({self.error_type}): expected {self.expected}; sent {self.received}"


def _compact(code: str, field_name: str, msg: str, ctx: dict | None) -> str:
    if code == "missing":
        return f"{field_name}: missing"
    if code == "extra_forbidden":
        return f"{field_name}: not allowed"
    if code in TYPE_WORD:
        return f"{field_name}: {TYPE_WORD[code]}"
    if code == "enum":
        vals = re.findall(r"'([^']*)'", (ctx or {}).get("expected", msg))
        return f"{field_name}: one of {'|'.join(vals)}"
    if code == "literal_error":
        exp = (ctx or {}).get("expected") or msg.replace("Input should be ", "")
        return f"{field_name}: {exp.replace(chr(39), chr(34))}"
    if code == "string_pattern_mismatch":
        return f"{field_name}: pattern"
    return f"{field_name}: invalid"


def _code_from_msg(msg: str) -> str:
    """Silent informative messages carry the pydantic message, not the code."""
    table = [("Field required", "missing"), ("Extra inputs are not permitted", "extra_forbidden"),
             ("Input should be a valid string", "string_type"), ("Input should be a valid integer", "int_type"),
             ("Input should be a valid number", "float_type"), ("Input should be a valid boolean", "bool_type"),
             ("Input should be a valid list", "list_type"), ("String should match pattern", "string_pattern_mismatch"),
             ("Input should be a finite number", "finite_number")]
    for prefix, code in table:
        if msg.startswith(prefix):
            return code
    if msg.startswith("Input should be "):
        rest = msg[len("Input should be "):]
        return "enum" if " or " in rest and rest.count("'") >= 4 else "literal_error"
    return "value_error"


def _path(loc: list) -> str:
    """("body", "items", 0, "quantity") -> "items[].quantity" (M2)."""
    parts = [str(x) for x in loc[1:] if not isinstance(x, int)]
    return "[].".join(parts) if parts else str(loc[-1])


def received_text(args: Any, enum_fields: set | None = None, stated: frozenset = frozenset()) -> str:
    """Sent keys with JSON types; values only for enum errors and only if they are schema enum values (D99)."""
    if not isinstance(args, dict):
        return jtype(args)
    parts = []
    for k, v in args.items():
        if isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            sub = {}
            for it in v:
                for ik, iv in it.items():
                    sub.setdefault(ik, jtype(iv))
            parts += [f"{k}[].{ik}: {t}" for ik, t in sub.items()]
            continue
        t = jtype(v)
        allowed = DOC_ENUM.get(k, set()) | set(stated)          # observed values only (D99, M2)
        if enum_fields and k in enum_fields and isinstance(v, str) and v in allowed:
            parts.append(f"{k}: \"{v}\"")
        else:
            parts.append(f"{k}: {t}")
    return ", ".join(parts) if parts else "no arguments"


def _stated(text: str) -> frozenset:
    return frozenset(re.findall(r"'([^']*)'", text)) | frozenset(re.findall(r'"([^"]*)"', text))


def parse_error(api_name: str, status: int, body: Any, args: Any) -> KT:
    if status == 200 and isinstance(body, dict) and body.get("status") == "error":
        m = SILENT_RE.match(str(body.get("message", "")))
        if m:
            fld, msg = m.group(1), m.group(2)
            code = _code_from_msg(msg)
            st = _stated(msg) if code in ("enum", "literal_error") else frozenset()
            return KT(api_name, _compact(code, fld, msg, None),
                      received_text(args, {fld} if code == "enum" else None, st),
                      "silent_error", frozenset({fld}), st)
        return KT(api_name, "unknown", received_text(args), "silent_error")
    if status == 422 and isinstance(body, dict) and isinstance(body.get("detail"), list):
        items = body["detail"]
        exp = "; ".join(_compact(i.get("type", ""), _path(i["loc"]), i.get("msg", ""), i.get("ctx"))
                        for i in items)
        keys = frozenset(_path(i["loc"]) for i in items)        # paths, like leakcheck (L5)
        enum_fields = {str(i["loc"][-1]) for i in items if i.get("type") == "enum"}
        et = CODE_TYPES.get(items[0].get("type", ""), "value_error") if items else "value_error"
        st = frozenset().union(*[_stated((i.get("ctx") or {}).get("expected", "")) for i in items
                                 if i.get("type") in ("enum", "literal_error")]) if items else frozenset()
        return KT(api_name, exp, received_text(args, enum_fields, st), et, keys, st)
    if status == 400:
        return KT(api_name, "unknown", received_text(args), "bad_request")
    if status == 410 and isinstance(body, dict) and isinstance(body.get("detail"), str):
        m = MOVED_RE.search(body["detail"])
        return KT(api_name, f"use '{m.group(1)}'" if m else "unknown", f"operation '{api_name}'",
                  "operation_removed", frozenset(), frozenset({m.group(1)}) if m else frozenset())
    if status == 404 and isinstance(body, str):
        return KT(api_name, "unknown", f"operation '{api_name}'", "route_not_found")
    if status == 404 and isinstance(body, dict):
        res = body.get("resource") or (body.get("detail", "").replace(" not found", "").lower() or "resource")
        return KT(api_name, f"{res} not found", received_text(args), "resource_not_found")
    if status in (409, 422) and isinstance(body, dict):
        if isinstance(body.get("detail"), str):
            return KT(api_name, body["detail"], received_text(args), "business_rule")
        return KT(api_name, "unknown", received_text(args), "business_rule")
    return KT(api_name, "unknown", "unknown", "unknown")


def moved_pointer(kt: KT) -> str | None:
    m = re.match(r"use '([^']+)'", kt.expected)
    return m.group(1) if kt.error_type == "operation_removed" and m else None
