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


def silent_body(message: str) -> dict:
    return {"status": "error", "message": message}
