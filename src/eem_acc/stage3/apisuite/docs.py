"""Plain-text API docs (S3.1 spec section 9). Marks required/optional, type, enum values, identifiers, constants."""
from __future__ import annotations

from .drift import OpView, op_view
from .ops import OPS, OP_NAMES
from .schema import Field

KIND_WORDS = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean", "id": "identifier",
              "digits": "string of digits", "decimal": "decimal string (e.g. \"1250.00\")",
              "int01": "integer (0 or 1)", "array_str": "array of strings"}


def _value_text(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _line(f: Field, indent: int) -> list[str]:
    req = "required" if f.required else "optional"
    if f.kind == "enum":
        t = "enum: " + " | ".join(f.values)
    elif f.kind == "const":
        t = "constant: " + _value_text(f.const)
    elif f.kind == "array_obj":
        t = "array of objects:"
    else:
        t = KIND_WORDS[f.kind]
    if f.default is not None:
        t += f" (default {_value_text(f.default)})"
    pad = " " * indent
    out = [f"{pad}{f.name:<{18 - indent}} {req:<9} {t}"]
    if f.kind == "array_obj":
        for g in f.items:
            out += _line(g, indent + 2)
    return out


def render(view: OpView) -> str:
    d = OPS[view.op]
    head = f"{view.route} - {d.description}"
    lines = [head]
    if view.route != view.op:
        lines.append(f"  (formerly {view.op})")
    for f in view.schema:
        lines += _line(f, 2)
    lines.append(f"  returns: {d.returns}")
    return "\n".join(lines)


def documented(op: str) -> str:
    return render(op_view(op, []))


def documented_all() -> str:
    return "\n\n".join(documented(op) for op in OP_NAMES)
