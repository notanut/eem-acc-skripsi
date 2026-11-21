"""Error bodies in the two styles (S3.1 spec section 5, D72-D74). No body ever carries an entity value."""
from __future__ import annotations

from typing import Any

from .schema import ErrorItem

INFORMATIVE, TERSE = "informative", "terse"
ROUTE_404_TERSE = "404 page not found"          # plain-text body (like a router's default 404)
RESOURCE_NAMES = {"customer": "Customer", "product": "Product", "order": "Order", "payment": "Payment",
                  "shipment": "Shipment", "ticket": "Ticket", "operation": "Operation"}


def validation(style: str, items: list[ErrorItem]) -> tuple[int, Any]:
    if style == INFORMATIVE:
        return 422, {"detail": [i.as_json() for i in items]}
    return 400, {"error": "Bad Request"}


def moved(style: str, old: str, new: str) -> tuple[int, Any]:
    if style == INFORMATIVE:
        return 410, {"detail": f"Operation '{old}' was removed. Use '{new}'."}
    return 404, ROUTE_404_TERSE


def resource_404(style: str, resource: str) -> tuple[int, Any]:
    if style == INFORMATIVE:
        return 404, {"detail": f"{RESOURCE_NAMES[resource]} not found"}
    return 404, {"error": "not_found", "resource": resource}


def business(style: str, status: int, msg: str) -> tuple[int, Any]:
    if style == INFORMATIVE:
        return status, {"detail": msg}
    return status, {"error": "Conflict" if status == 409 else "Unprocessable Entity"}


TYPE_WORDS = {"digits": "a string of digits", "decimal": "a decimal string", "array_str": "a list of strings",
              "int01": "the integer 0 or 1"}


def silent_message(style: str, effective_type: str, field_name: str, *, new_kind: str | None = None,
                   field_desc: str | None = None, allowed: tuple = ()) -> str:
    """The message of a silent-drift 200 response (spec section 5); built from the change, never from the input."""
    if style != INFORMATIVE:
        return "Request failed"
    if effective_type == "rename":
        return f"Validation failed: field '{field_name}' is required"
    if effective_type == "rename_extra":
        return f"Validation failed: field '{field_name}' is not allowed"
    if effective_type == "type":
        return f"Validation failed: field '{field_name}' must be {TYPE_WORDS[new_kind]}"
    if effective_type == "new_required":
        return f"Validation failed: field '{field_name}' is required ({field_desc})"
    if effective_type == "enum":
        return f"Validation failed: value not allowed for '{field_name}'; allowed: {', '.join(allowed)}"
    raise AssertionError(effective_type)


def silent_body(message: str) -> dict:
    return {"status": "error", "message": message}
