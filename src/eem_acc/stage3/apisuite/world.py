"""World state (S3.1 spec section 2, D67).

The world is built from a stream spec: every task's entities are listed concretely (with ids) in the spec, so the
same spec always gives the same world, and one world is built per result unit. Objects created by the agent get ids
from per-kind counters in a range the setup never uses, so they are deterministic given the call sequence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

WAREHOUSES = ("JKT", "SBY", "BDG")
WAREHOUSE_CITY = {"JKT": "Jakarta", "SBY": "Surabaya", "BDG": "Bandung"}
CREATED_ID_START = 50001          # setup ids are 4 digits (1000-9999); created ids start here

KINDS = ("customers", "products", "carts", "orders", "payments", "reservations", "shipments", "tickets")
ID_FIELD = {"customers": "customer_id", "products": "sku", "orders": "order_id", "payments": "payment_id",
            "reservations": "reservation_id", "shipments": "shipment_id", "tickets": "ticket_id"}
ID_PREFIX = {"orders": "O", "payments": "P", "reservations": "R", "shipments": "S", "tickets": "T"}


@dataclass
class World:
    customers: dict = field(default_factory=dict)      # customer_id -> dict
    products: dict = field(default_factory=dict)       # sku -> dict (stock: {wh: int})
    carts: dict = field(default_factory=dict)          # customer_id -> [ {sku, qty} ]
    orders: dict = field(default_factory=dict)
    payments: dict = field(default_factory=dict)
    reservations: dict = field(default_factory=dict)
    shipments: dict = field(default_factory=dict)
    tickets: dict = field(default_factory=dict)
    created: list = field(default_factory=list)        # (kind, id) in creation order
    _next: dict = field(default_factory=dict)

    @classmethod
    def build(cls, spec: dict) -> "World":
        w = cls()
        for task in spec["tasks"]:
            w.add_setup(task["setup"])
        return w

    def add_setup(self, setup: dict) -> None:
        for kind in KINDS:
            for ent in setup.get(kind, []):
                ent = copy.deepcopy(ent)
                if kind == "carts":
                    key = ent["customer_id"]
                    if key in self.carts:
                        raise ValueError(f"duplicate cart {key}")
                    self.carts[key] = ent["items"]
                    continue
                key = ent[ID_FIELD[kind]]
                store = getattr(self, kind)
                if key in store:
                    raise ValueError(f"duplicate id {key} in {kind}")
                store[key] = ent

    def new_id(self, kind: str) -> str:
        n = self._next.get(kind, CREATED_ID_START)
        self._next[kind] = n + 1
        return f"{ID_PREFIX[kind]}-{n}"

    def create(self, kind: str, ent: dict) -> dict:
        key = self.new_id(kind)
        ent = dict(ent)
        ent[ID_FIELD[kind]] = key
        getattr(self, kind)[key] = ent
        self.created.append((kind, key))
        return ent

    def snapshot(self) -> dict:
        """A JSON-serialisable deep copy (for tests and determinism checks)."""
        return copy.deepcopy({k: getattr(self, k) for k in KINDS} | {"created": self.created})
