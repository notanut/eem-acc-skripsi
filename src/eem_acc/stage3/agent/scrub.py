"""Scrub task values from reflections and A1 raw text before they enter the shared store (S3.2a spec 6.2, D92)."""
from __future__ import annotations

import re
from typing import Any

from ..apisuite.drift import ALL_PAIRS
from ..apisuite.ops import OP_NAMES
from .parser import ENUM_VALUES

PLACEHOLDER_BY_KEY = {"customer_id": "customer_id", "email": "email", "name": "name", "sku": "sku",
                      "order_id": "order_id", "payment_id": "payment_id", "shipment_id": "shipment_id",
                      "ticket_id": "ticket_id", "reservation_id": "reservation_id", "subject": "text",
                      "text": "text", "query": "product", "product": "product", "price": "amount",
                      "total": "amount", "amount": "amount", "refunded_amount": "amount", "summary": "text"}
# schema vocabulary that is never scrubbed (enum, boolean and constant values)
SAFE = ({str(v) for vals in ENUM_VALUES.values() for v in vals} | {"IDR", "2", "30", "50", "true", "false", "True",
                                                                  "False"}
        | set(OP_NAMES) | {p.new_op for p in ALL_PAIRS.values() if p.new_op} | {"get_api_docs", "finish"})


class Scrubber:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}             # value text -> placeholder name

    def add(self, value: Any, key: str = "value") -> None:
        if isinstance(value, bool) or value is None:
            return
        name = PLACEHOLDER_BY_KEY.get(key.split("[].")[-1], key.split("[].")[-1])
        if isinstance(value, (int, float)):
            if abs(value) < 100:                      # small counts (qty) are not entities
                return
            text = str(int(value)) if float(value).is_integer() else str(value)
            self.values.setdefault(text, name)
            if float(value).is_integer() and abs(value) >= 1000:      # D103: separator variants
                n = int(value)
                self.values.setdefault(f"{n:,}", name)
                self.values.setdefault(f"{n:,}".replace(",", "."), name)
            return
        if not isinstance(value, str):
            return
        text = value.strip()
        if len(text) < 3 or text in SAFE:
            return
        self.values.setdefault(text, name)
        # D103: normalised forms
        m = re.fullmatch(r"[A-Z]+-(\d{4,})", text)
        if m:                                         # the number part of an id ("O-9761" -> "9761")
            self.values.setdefault(m.group(1), name)
        if "@" in text:                               # the email's local part
            self.values.setdefault(text.split("@")[0], name)
        if key.split("[].")[-1] == "name":            # name tokens of 4+ letters
            for tok in text.split():
                if len(tok) >= 4 and tok not in SAFE:
                    self.values.setdefault(tok, name)

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
            if re.fullmatch(r"[\d.,]+", val):
                pat = re.compile(rf"(?<!\d){re.escape(val)}(?!\d)")         # digit boundaries (H2)
            else:
                pre = r"(?<!\w)" if val[0].isalnum() else ""
                post = r"(?!\w)" if val[-1].isalnum() else ""
                pat = re.compile(pre + re.escape(val) + post, re.IGNORECASE)  # word boundaries (L1)
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
