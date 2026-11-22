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
        | set(OP_NAMES) | {p.new_op for p in ALL_PAIRS.values() if p.new_op} | {"get_api_docs", "finish"}
        # response-only vocabulary of the suite (statuses, tiers, shipment states), never entities (re-review M4)
        | {"created", "paid", "shipped", "cancelled", "captured", "refunded", "in_transit", "delayed", "delivered",
           "basic", "gold", "open", "pending", "closed"})
SAFE_LOWER = {v.lower() for v in SAFE}


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
                for sep in (",", ".", " "):
                    self.values.setdefault(f"{n:,}".replace(",", sep), name)
            return
        if not isinstance(value, str):
            return
        text = value.strip()
        if len(text) < 3 or text.lower() in SAFE_LOWER:           # case-insensitive, like the matching (H1)
            return
        self.values.setdefault(text, name)
        # D103: normalised forms
        m = re.fullmatch(r"[A-Z]+-(\d{4,})", text)
        if m:                                         # the number part of an id ("O-9761" -> "9761")
            self.values.setdefault(m.group(1), name)
        if re.fullmatch(r"\d{4,}", text):              # an amount sent as a string (e.g. after a type drift)
            self.add(int(text), key)
        if "@" in text and len(text.split("@")[0]) >= 3:          # the email's local part
            local = text.split("@")[0]
            self.values.setdefault(local, name)
            self.values.setdefault(local.replace(".", "_"), name)
        if key.split("[].")[-1] in ("name", "product"):          # name tokens of 4+ letters
            for tok in text.split():
                if len(tok) >= 4 and tok.lower() not in SAFE_LOWER:
                    self.values.setdefault(tok, name)
            if key.split("[].")[-1] == "name":
                self.values.setdefault(text.lower().replace(" ", "_"), name)
                self.values.setdefault(text.lower().replace(" ", "."), name)

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
    setup = dict(inst["setup"])
    for p in setup.pop("products", []):                     # product names get their own placeholder
        s.add(p["name"], "product")
        s.add_tree({k: v for k, v in p.items() if k != "name"})
    s.add_tree(setup)
    s.add_tree(inst["params"])
    return s
