"""The 16 operations: documented ("old") schema, description, and handler with business rules (spec section 3).

Handlers receive *canonical* arguments: the call's arguments translated back to the documented names, types and
enum values (drift.canonicalize), so business logic never depends on drift. A handler returns an `OpResult`.
Business-rule messages carry no ids or values (D73).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from .schema import Field
from .world import WAREHOUSES, World


@dataclass(frozen=True)
class OpResult:
    status: int
    body: Any = None                  # success body (dict)
    reason: str | None = None         # business message, or the resource name for a resource 404
    kind: str = "ok"                  # ok | not_found | business


def ok(body: dict) -> OpResult:
    return OpResult(200, body)


def not_found(resource: str) -> OpResult:
    return OpResult(404, reason=resource, kind="not_found")


def business(status: int, msg: str) -> OpResult:
    return OpResult(status, reason=msg, kind="business")


WH = Field("warehouse", "enum", values=WAREHOUSES)


@dataclass(frozen=True)
class OpDef:
    name: str
    description: str
    schema: tuple
    returns: str
    handler: Callable[[World, dict], OpResult]


def _num(v: Any) -> Decimal:
    return Decimal(str(v))


def _plain(d: Decimal) -> int | float:
    return int(d) if d == d.to_integral_value() else float(d)


# ---------------------------------------------------------------------------------------------------------------- handlers
def h_find_customer(w: World, a: dict) -> OpResult:
    email = a["email"].strip().lower()
    for c in w.customers.values():
        if c["email"].lower() == email:
            return ok({"customer_id": c["customer_id"], "name": c["name"], "tier": c["tier"]})
    return not_found("customer")


def h_get_cart(w: World, a: dict) -> OpResult:
    if a["customer_id"] not in w.customers:
        return not_found("customer")
    return ok({"customer_id": a["customer_id"], "items": [dict(i) for i in w.carts.get(a["customer_id"], [])]})


def h_create_order(w: World, a: dict) -> OpResult:
    if a["customer_id"] not in w.customers:
        return not_found("customer")
    total = 0
    items = []
    for it in a["items"]:
        p = w.products.get(it["sku"])
        if p is None:
            return not_found("product")
        if it["qty"] < 1:
            return business(422, "Quantity must be at least 1.")
        total += p["price"] * it["qty"]
        items.append({"sku": it["sku"], "qty": it["qty"]})
    o = w.create("orders", {"customer_id": a["customer_id"], "items": items, "shipping_method": a["shipping_method"],
                            "total": total, "status": "created", "payment_id": None, "shipment_id": None})
    return ok({"order_id": o["order_id"], "total": total})


def h_get_order(w: World, a: dict) -> OpResult:
    o = w.orders.get(a["order_id"])
    if o is None:
        return not_found("order")
    keys = ("order_id", "customer_id", "items", "shipping_method", "total", "status", "payment_id", "shipment_id")
    return ok({k: (list(map(dict, o[k])) if k == "items" else o[k]) for k in keys})


def h_pay_order(w: World, a: dict) -> OpResult:
    o = w.orders.get(a["order_id"])
    if o is None:
        return not_found("order")
    if o["status"] != "created":
        return business(409, "The order cannot be paid in its current state.")
    if _num(a["amount"]) != _num(o["total"]):
        return business(422, "The amount does not match the order total.")
    p = w.create("payments", {"order_id": o["order_id"], "amount": o["total"], "method": a["method"],
                              "refunded_amount": 0, "status": "captured"})
    o["status"] = "paid"
    o["payment_id"] = p["payment_id"]
    return ok({"payment_id": p["payment_id"], "status": "captured"})


def h_refund_payment(w: World, a: dict) -> OpResult:
    p = w.payments.get(a["payment_id"])
    if p is None:
        return not_found("payment")
    amt = _num(a["amount"])
    remaining = _num(p["amount"]) - _num(p["refunded_amount"])
    if amt <= 0 or amt > remaining:
        return business(422, "The refund amount exceeds the refundable amount.")
    p["refunded_amount"] = _plain(_num(p["refunded_amount"]) + amt)
    p.setdefault("refunds", []).append({"amount": _plain(amt), "reason": a["reason"]})
    if _num(p["refunded_amount"]) == _num(p["amount"]):
        p["status"] = "refunded"
    return ok({"payment_id": p["payment_id"], "status": p["status"]})


def h_cancel_order(w: World, a: dict) -> OpResult:
    o = w.orders.get(a["order_id"])
    if o is None:
        return not_found("order")
    if o["status"] in ("shipped", "cancelled"):
        return business(409, "The order cannot be cancelled in its current state.")
    if o["status"] == "paid":
        p = w.payments.get(o["payment_id"])
        if p is None or p["status"] != "refunded":
            return business(409, "Refund the payment in full before cancelling a paid order.")
    o["status"] = "cancelled"
    o["cancel_reason"] = a["reason"]
    return ok({"order_id": o["order_id"], "status": "cancelled"})


def _matches(name: str, query: Any) -> bool:
    terms = query if isinstance(query, list) else [query]
    n = name.lower()
    return all(t.strip().lower() in n for t in terms)


def h_search_products(w: World, a: dict) -> OpResult:
    limit = a.get("limit", 10)
    hits = sorted((p for p in w.products.values() if _matches(p["name"], a["query"])), key=lambda p: p["name"])
    return ok({"products": [{"sku": p["sku"], "name": p["name"], "price": p["price"]} for p in hits[:max(0, limit)]]})


def h_check_stock(w: World, a: dict) -> OpResult:
    p = w.products.get(a["sku"])
    if p is None:
        return not_found("product")
    return ok({"sku": p["sku"], "warehouse": a["warehouse"], "available": p["stock"][a["warehouse"]]})


def h_reserve_stock(w: World, a: dict) -> OpResult:
    p = w.products.get(a["sku"])
    if p is None:
        return not_found("product")
    if a["qty"] < 1:
        return business(422, "Quantity must be at least 1.")
    if p["stock"][a["warehouse"]] < a["qty"]:
        return business(409, "Not enough stock in this warehouse.")
    p["stock"][a["warehouse"]] -= a["qty"]
    r = w.create("reservations", {"sku": p["sku"], "warehouse": a["warehouse"], "qty": a["qty"]})
    return ok({"reservation_id": r["reservation_id"]})


def h_create_shipment(w: World, a: dict) -> OpResult:
    o = w.orders.get(a["order_id"])
    if o is None:
        return not_found("order")
    if o["shipment_id"] is not None:
        return business(409, "The order already has a shipment.")
    if o["status"] != "paid":
        return business(409, "The order must be paid before it can be shipped.")
    wh = a["warehouse"]
    for it in o["items"]:
        if w.products[it["sku"]]["stock"][wh] < it["qty"]:
            return business(409, "Not enough stock in this warehouse.")
    for it in o["items"]:
        w.products[it["sku"]]["stock"][wh] -= it["qty"]
    s = w.create("shipments", {"order_id": o["order_id"], "warehouse": wh, "carrier": a["carrier"],
                               "insured": bool(a.get("insured", False)), "status": "created"})
    o["shipment_id"] = s["shipment_id"]
    o["status"] = "shipped"
    return ok({"shipment_id": s["shipment_id"]})


def h_track_shipment(w: World, a: dict) -> OpResult:
    s = w.shipments.get(a["shipment_id"])
    if s is None:
        return not_found("shipment")
    return ok({"shipment_id": s["shipment_id"], "order_id": s["order_id"], "state": s["status"]})


def h_create_ticket(w: World, a: dict) -> OpResult:
    if a["customer_id"] not in w.customers:
        return not_found("customer")
    t = w.create("tickets", {"customer_id": a["customer_id"], "subject": a["subject"], "category": a["category"],
                             "priority": a.get("priority", "normal"), "status": "open", "comments": []})
    return ok({"ticket_id": t["ticket_id"]})


def h_update_ticket(w: World, a: dict) -> OpResult:
    t = w.tickets.get(a["ticket_id"])
    if t is None:
        return not_found("ticket")
    t["status"] = a["status"]
    return ok({"ticket_id": t["ticket_id"], "status": t["status"]})


def h_add_comment(w: World, a: dict) -> OpResult:
    t = w.tickets.get(a["ticket_id"])
    if t is None:
        return not_found("ticket")
    t["comments"].append({"text": a["text"], "public": bool(a["public"])})
    return ok({"ticket_id": t["ticket_id"], "comment_index": len(t["comments"]) - 1})


def h_list_tickets(w: World, a: dict) -> OpResult:
    if a["customer_id"] not in w.customers:
        return not_found("customer")
    ts = [t for t in w.tickets.values() if t["customer_id"] == a["customer_id"]
          and ("status" not in a or t["status"] == a["status"])]
    ts.sort(key=lambda t: t["ticket_id"])
    return ok({"tickets": [{"ticket_id": t["ticket_id"], "subject": t["subject"], "category": t["category"],
                            "status": t["status"]} for t in ts]})


# ------------------------------------------------------------------------------------------------------------- definitions
OPS: dict[str, OpDef] = {d.name: d for d in [
    OpDef("find_customer", "Find a customer by email address.",
          (Field("email", "string"),), "customer_id, name, tier", h_find_customer),
    OpDef("get_cart", "Get the items in a customer's cart.",
          (Field("customer_id", "id"),), "customer_id, items [{sku, qty}]", h_get_cart),
    OpDef("create_order", "Create an order for a customer.",
          (Field("customer_id", "id"),
           Field("items", "array_obj", items=(Field("sku", "id"), Field("qty", "integer"))),
           Field("shipping_method", "enum", values=("standard", "express"))),
          "order_id, total", h_create_order),
    OpDef("get_order", "Get an order.",
          (Field("order_id", "id"),),
          "order_id, customer_id, items, shipping_method, total, status, payment_id, shipment_id", h_get_order),
    OpDef("pay_order", "Pay an order in full.",
          (Field("order_id", "id"), Field("amount", "number"),
           Field("method", "enum", values=("card", "transfer", "wallet"))),
          "payment_id, status", h_pay_order),
    OpDef("refund_payment", "Refund (part of) a payment.",
          (Field("payment_id", "id"), Field("amount", "number"),
           Field("reason", "enum", values=("damaged", "late", "other"))),
          "payment_id, status", h_refund_payment),
    OpDef("cancel_order", "Cancel an order (a paid order must be refunded in full first).",
          (Field("order_id", "id"), Field("reason", "enum", values=("customer_request", "out_of_stock"))),
          "order_id, status", h_cancel_order),
    OpDef("search_products", "Search products by name.",
          (Field("query", "string"), Field("limit", "integer", required=False, default=10)),
          "products [{sku, name, price}]", h_search_products),
    OpDef("check_stock", "Check the available stock of a product in a warehouse.",
          (Field("sku", "id"), WH), "sku, warehouse, available", h_check_stock),
    OpDef("reserve_stock", "Reserve stock of a product in a warehouse.",
          (Field("sku", "id"), Field("qty", "integer"), WH), "reservation_id", h_reserve_stock),
    OpDef("create_shipment", "Ship a paid order from a warehouse.",
          (Field("order_id", "id"), WH, Field("carrier", "enum", values=("jne", "sicepat", "pos")),
           Field("insured", "boolean", required=False, default=False)),
          "shipment_id", h_create_shipment),
    OpDef("track_shipment", "Get the state of a shipment.",
          (Field("shipment_id", "id"),), "shipment_id, order_id, state", h_track_shipment),
    OpDef("create_ticket", "Open a support ticket for a customer.",
          (Field("customer_id", "id"), Field("subject", "string"),
           Field("category", "enum", values=("billing", "shipping", "product")),
           Field("priority", "enum", required=False, values=("low", "normal", "high"), default="normal")),
          "ticket_id", h_create_ticket),
    OpDef("update_ticket", "Change the status of a ticket.",
          (Field("ticket_id", "id"), Field("status", "enum", values=("open", "pending", "closed"))),
          "ticket_id, status", h_update_ticket),
    OpDef("add_comment", "Add a comment to a ticket.",
          (Field("ticket_id", "id"), Field("text", "string"), Field("public", "boolean")),
          "ticket_id, comment_index", h_add_comment),
    OpDef("list_tickets", "List a customer's tickets.",
          (Field("customer_id", "id"),
           Field("status", "enum", required=False, values=("open", "pending", "closed"))),
          "tickets [{ticket_id, subject, category, status}]", h_list_tickets),
]}

OP_NAMES = tuple(OPS)
assert len(OP_NAMES) == 16
