"""Scrub task values from reflections and A1 raw text before they enter the shared store (S3.2a spec 6.2, D92)."""
from __future__ import annotations

import re
from typing import Any

from .parser import ENUM_VALUES

PLACEHOLDER_BY_KEY = {"customer_id": "customer_id", "email": "email", "name": "name", "sku": "sku",
                      "order_id": "order_id", "payment_id": "payment_id", "shipment_id": "shipment_id",
                      "ticket_id": "ticket_id", "reservation_id": "reservation_id", "subject": "text",
                      "text": "text", "query": "product", "product": "product", "price": "amount",
                      "total": "amount", "amount": "amount", "refunded_amount": "amount", "summary": "text"}
# schema vocabulary that is never scrubbed (enum, boolean and constant values)
SAFE = {str(v) for vals in ENUM_VALUES.values() for v in vals} | {"IDR", "2", "30", "50", "true", "false", "True",
                                                                 "False"}


class Scrubber:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}             # value text -> placeholder name

    def add(self, value: Any, key: str = "value") -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            if abs(value) < 100:                      # small counts (qty) are not entities
                return
            text = str(int(value)) if float(value).is_integer() else str(value)
        elif isinstance(value, str):
            text = value.strip()
            if len(text) < 3 or text in SAFE:
                return
        else:
            return
        name = PLACEHOLDER_BY_KEY.get(key.split("[].")[-1], key.split("[].")[-1])
        self.values.setdefault(text, name)

    def add_tree(self, obj: Any, key: str = "value") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                self.add_tree(v, str(k))
        elif isinstance(obj, list):
            for v in obj:
                self.add_tree(v, key)
        else:
            self.add(obj, key)

    def scrub(self, text: str) -> tuple[str, int]:
        n = 0
        for val in sorted(self.values, key=lambda v: (-len(v), v)):
            if val.isdigit() or re.fullmatch(r"\d+(\.\d+)?", val):
                pat = re.compile(rf"(?<![\w.]){re.escape(val)}(?![\w])")
            else:
                pat = re.compile(re.escape(val), re.IGNORECASE)
            text, k = pat.subn(f"<{self.values[val]}>", text)
            n += k
        return text, n


def task_scrubber(inst: dict) -> Scrubber:
    """Seeded with the task's instruction values and entities (setup); the loop adds call arguments and
    observation bodies as they appear."""
    s = Scrubber()
    s.add_tree(inst["setup"])
    s.add_tree(inst["params"])
    return s
