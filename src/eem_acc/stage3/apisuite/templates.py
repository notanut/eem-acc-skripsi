"""Task templates (S3.1 spec section 7): instance sampling, reference solution, sub-goals.

An instance is plain JSON: template id, instruction, params, the task's own entities (`setup`, D67), and `refs`
(ids the checker needs). `solve(api, inst)` is the reference policy written against the *documented* schema; an
oracle wrapper translates its calls to the current schema, and the recorded trace is the instance's reference plan
(`min_calls` = its length, D75). `goals(inst, world)` returns the sub-goals (Section III-A: success = all hold).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

from .world import WAREHOUSE_CITY, WAREHOUSES, World

FIRST = ["Budi", "Siti", "Agus", "Dewi", "Rina", "Andi", "Putri", "Joko", "Wulan", "Hendra", "Maya", "Rizky", "Fitri",
         "Bayu", "Indah", "Yusuf", "Sari", "Dimas", "Lestari", "Fajar"]
LAST = ["Santoso", "Wijaya", "Pratama", "Hidayat", "Kusuma", "Saputra", "Nugroho", "Halim", "Setiawan", "Gunawan",
        "Rahman", "Siregar", "Susanto", "Purnama", "Wibowo"]
BRANDS = ["Nusantara", "Sari Rasa", "Bumi Hijau", "Kencana", "Merapi", "Laut Biru", "Cahaya", "Pelangi"]
ITEMS = ["Kopi Arabica", "Teh Melati", "Gula Aren", "Sambal Bawang", "Keripik Singkong", "Madu Hutan", "Beras Pandan",
         "Minyak Kelapa", "Kecap Manis", "Rendang Kering", "Abon Sapi", "Kacang Mete", "Saus Tiram", "Bumbu Soto",
         "Kerupuk Udang", "Susu Kedelai"]
SIZES = ["100 g", "250 g", "500 g", "1 kg", "2 kg", "5 kg"]
SUBJECTS = ["Package arrived damaged", "Request for an invoice copy", "Wrong item received", "Question about delivery",
            "Payment charged twice", "Item missing from package", "Change of delivery address", "Product quality issue",
            "Refund not received yet", "Courier did not arrive"]
COMMENTS = ["We are checking this with the warehouse.", "The refund has been issued.", "Customer called again today.",
            "Escalated to the logistics team.", "Replacement will be sent this week.",
            "Please keep the original packaging.", "The courier confirmed the delay.", "Awaiting the customer's photo."]
SHIP = ("standard", "express")
PAY = ("card", "transfer", "wallet")
CARRIERS = ("jne", "sicepat", "pos")
CARRIER_NAMES = {"jne": "JNE (jne)", "sicepat": "SiCepat (sicepat)", "pos": "Pos Indonesia (pos)"}
REFUND = ("damaged", "late", "other")
CANCEL = ("customer_request", "out_of_stock")
CATEGORIES = ("billing", "shipping", "product")
PRIORITIES = ("low", "normal", "high")
INSURE_LIMIT = 1_000_000


def wh_text(w: str) -> str:
    return f"the {WAREHOUSE_CITY[w]} warehouse ({w})"


def rp(x: int) -> str:
    return f"Rp {x:,}"


# ------------------------------------------------------------------------------------------------------------- generator
class Gen:
    """Unique ids, names and emails within one stream.

    Each task draws its primary entities and values from its own RNG (`rng`) and its look-alike distractors from a
    separate RNG (`drng`), and distractors use their own id range and email pattern, so changing the distractor knob
    never changes a task's primary entities, values or outcome (spec sections 1 and 8). Every product gets a
    brand+item combination that no other task uses (D77); its look-alikes are the other sizes of that combination.
    """

    def __init__(self, rng: random.Random, n_distractors: int, drng: random.Random | None = None):
        self.n_distractors = n_distractors
        self._ids: dict[str, set] = {}
        self._emails: set = set()
        self._combos: set = set()
        self.start_task(rng, drng or random.Random(rng.random()))

    def start_task(self, rng: random.Random, drng: random.Random) -> None:
        self.rng, self.drng = rng, drng

    def id(self, prefix: str, distract: bool = False) -> str:
        used = self._ids.setdefault(prefix, set())
        r, lo, hi = (self.drng, 7000, 10000) if distract else (self.rng, 1000, 7000)
        for _ in range(100000):
            n = r.randrange(lo, hi)
            if n not in used:
                used.add(n)
                return f"{prefix}-{n}"
        raise RuntimeError("id space exhausted")

    def _email(self, first: str, last: str, distract: bool = False) -> str:
        r, sep = (self.drng, "_") if distract else (self.rng, ".")
        for _ in range(1000):
            dom = r.choice(["mail.example", "post.example", "inbox.example"])
            e = f"{first.lower()}{sep}{last.lower()}{r.randrange(10, 100)}@{dom}"
            if e not in self._emails:
                self._emails.add(e)
                return e
        raise RuntimeError("email space exhausted")

    def customer(self, setup: dict, distract: bool = False) -> dict:
        first, last = self.rng.choice(FIRST), self.rng.choice(LAST)
        c = {"customer_id": self.id("C"), "email": self._email(first, last), "name": f"{first} {last}",
             "tier": self.rng.choice(["basic", "gold"])}
        setup.setdefault("customers", []).append(c)
        if distract:                                    # same name, other email (look-alikes)
            for _ in range(self.n_distractors):
                setup["customers"].append({"customer_id": self.id("C", True), "email": self._email(first, last, True),
                                           "name": c["name"], "tier": self.drng.choice(["basic", "gold"])})
        return c

    def product(self, setup: dict, stock: dict | None = None, price: int | None = None,
                distract: bool = False) -> dict:
        free = [(b, i) for b in BRANDS for i in ITEMS if (b, i) not in self._combos]
        if not free:
            raise RuntimeError("product name space exhausted")
        brand, item = self.rng.choice(free)
        self._combos.add((brand, item))
        size = self.rng.choice(SIZES)
        p = {"sku": self.id("SKU"), "name": f"{brand} {item} {size}",
             "price": price or self.rng.randrange(15, 450) * 1000,
             "stock": dict(stock or {w: self.rng.randrange(5, 40) for w in WAREHOUSES})}
        setup.setdefault("products", []).append(p)
        if distract:
            others = [z for z in SIZES if z != size]
            for z in self.drng.sample(others, min(self.n_distractors, len(others))):
                setup["products"].append({"sku": self.id("SKU", True), "name": f"{brand} {item} {z}",
                                          "price": self.drng.randrange(15, 450) * 1000,
                                          "stock": {w: self.drng.randrange(5, 40) for w in WAREHOUSES}})
        return p

    def items(self, setup: dict, n: int, qty_max: int = 3, stock_min: int = 10) -> list[dict]:
        out = []
        for _ in range(n):
            p = self.product(setup, stock={w: self.rng.randrange(stock_min, 40) for w in WAREHOUSES})
            out.append({"sku": p["sku"], "qty": self.rng.randrange(1, qty_max + 1)})
        return out

    def order(self, setup: dict, customer: dict, items: list[dict], status: str, method: str | None = None) -> dict:
        prices = {p["sku"]: p["price"] for p in setup["products"]}
        total = sum(prices[i["sku"]] * i["qty"] for i in items)
        o = {"order_id": self.id("O"), "customer_id": customer["customer_id"], "items": items,
             "shipping_method": method or self.rng.choice(SHIP), "total": total, "status": status,
             "payment_id": None, "shipment_id": None}
        setup.setdefault("orders", []).append(o)
        if status in ("paid", "shipped"):
            p = {"payment_id": self.id("P"), "order_id": o["order_id"], "amount": total,
                 "method": self.rng.choice(PAY), "refunded_amount": 0, "status": "captured"}
            setup.setdefault("payments", []).append(p)
            o["payment_id"] = p["payment_id"]
        return o

    def shipment(self, setup: dict, order: dict, status: str) -> dict:
        s = {"shipment_id": self.id("S"), "order_id": order["order_id"], "warehouse": self.rng.choice(WAREHOUSES),
             "carrier": self.rng.choice(CARRIERS), "insured": False, "status": status}
        setup.setdefault("shipments", []).append(s)
        order["shipment_id"] = s["shipment_id"]
        order["status"] = "shipped"
        return s

    def ticket(self, setup: dict, customer: dict, category: str, status: str, distract: bool = False) -> dict:
        r = self.drng if distract else self.rng
        t = {"ticket_id": self.id("T", distract), "customer_id": customer["customer_id"],
             "subject": r.choice(SUBJECTS), "category": category, "priority": r.choice(PRIORITIES),
             "status": status, "comments": []}
        setup.setdefault("tickets", []).append(t)
        return t

    def ticket_distractors(self, setup: dict, customer: dict, category: str) -> None:
        """n_distractors more tickets of the customer in the target's category, not open (D78)."""
        for _ in range(self.n_distractors):
            self.ticket(setup, customer, category, self.drng.choice(["pending", "closed"]), distract=True)


# --------------------------------------------------------------------------------------------------------- goal helpers
def _created(w: World, kind: str, pred: Callable[[dict], bool]) -> list[dict]:
    store = getattr(w, kind)
    return [store[i] for k, i in w.created if k == kind and pred(store[i])]


def _items_eq(a: list, b: list) -> bool:
    return sorted((i["sku"], i["qty"]) for i in a) == sorted((i["sku"], i["qty"]) for i in b)


def _norm(s: Any) -> str:
    return " ".join(str(s).split())


def g_one_new_order(w, cid, items, method) -> list:
    os_ = _created(w, "orders", lambda o: o["customer_id"] == cid)
    one = len(os_) == 1
    o = os_[0] if one else None
    return [("exactly one new order", one),
            ("order items", bool(o) and _items_eq(o["items"], items)),
            ("shipping method", bool(o) and o["shipping_method"] == method)], o


def g_paid(w, order: dict | None, method: str) -> list:
    if order is None:
        return [("exactly one payment", False), ("full amount", False), ("payment method", False)]
    ps = _created(w, "payments", lambda p: p["order_id"] == order["order_id"])
    one = len(ps) == 1
    return [("exactly one payment", one),
            ("full amount", one and ps[0]["amount"] == order["total"] and order["status"] in ("paid", "shipped")),
            ("payment method", one and ps[0]["method"] == method)]


def g_shipped(w, order: dict | None, wh: str, carrier: str, insured: bool | None = None) -> list:
    if order is None:
        return [("exactly one shipment", False), ("warehouse", False), ("carrier", False)]
    ss = _created(w, "shipments", lambda s: s["order_id"] == order["order_id"])
    one = len(ss) == 1
    out = [("exactly one shipment", one), ("warehouse", one and ss[0]["warehouse"] == wh),
           ("carrier", one and ss[0]["carrier"] == carrier)]
    if insured is not None:
        out.append(("insured flag", one and ss[0]["insured"] == insured))
    return out


def g_refunded(w, payment_id: str, reason: str) -> list:
    p = w.payments[payment_id]
    refs = p.get("refunds", [])
    return [("refunded in full", p["refunded_amount"] == p["amount"] and len(refs) == 1),
            ("refund reason", len(refs) == 1 and refs[0]["reason"] == reason)]


def g_cancelled(w, order_id: str, reason: str) -> list:
    o = w.orders[order_id]
    return [("order cancelled", o["status"] == "cancelled"), ("cancel reason", o.get("cancel_reason") == reason)]


def g_one_new_ticket(w, cid, category, subject, priority="normal") -> tuple[list, dict | None]:
    ts = _created(w, "tickets", lambda t: t["customer_id"] == cid)
    one = len(ts) == 1
    t = ts[0] if one else None
    return [("exactly one new ticket", one), ("ticket category", bool(t) and t["category"] == category),
            ("ticket subject", bool(t) and _norm(t["subject"]) == _norm(subject)),
            ("ticket priority", bool(t) and t["priority"] == priority)], t


def g_no_new_ticket(w, cid) -> list:
    return [("no new ticket", not _created(w, "tickets", lambda t: t["customer_id"] == cid))]


def g_comment(w, ticket_id: str | None, before: list, text: str, public: bool) -> list:
    if ticket_id is None:
        return [("exactly one new comment", False), ("comment text", False), ("comment visibility", False)]
    cs = w.tickets[ticket_id]["comments"]
    new = cs[len(before):]
    one = len(new) == 1 and cs[:len(before)] == before
    return [("exactly one new comment", one), ("comment text", one and _norm(new[0]["text"]) == _norm(text)),
            ("comment visibility", one and new[0]["public"] == public)]


def g_no_new_comment(w, ticket_id: str, before: list) -> list:
    return [("no new comment", w.tickets[ticket_id]["comments"] == before)]


def g_ticket_status(w, ticket_id: str, status: str, label: str = "ticket status") -> list:
    return [(label, w.tickets[ticket_id]["status"] == status)]


def g_unchanged_ticket(w, t0: dict) -> list:
    t = w.tickets[t0["ticket_id"]]
    return [(f"other ticket unchanged", t["status"] == t0["status"] and t["comments"] == t0["comments"])]


def g_one_reservation(w, sku: str, wh: str, qty: int) -> list:
    rs = _created(w, "reservations", lambda r: r["sku"] == sku)
    one = len(rs) == 1
    return [("exactly one reservation", one), ("reservation warehouse", one and rs[0]["warehouse"] == wh),
            ("reservation quantity", one and rs[0]["qty"] == qty)]


def g_no_reservation(w, sku: str) -> list:
    return [("no reservation", not _created(w, "reservations", lambda r: r["sku"] == sku))]


def _setup_ticket(inst: dict, tid: str) -> dict:
    return next(t for t in inst["setup"]["tickets"] if t["ticket_id"] == tid)


# ------------------------------------------------------------------------------------------------------------- templates
@dataclass(frozen=True)
class Template:
    id: str
    covers: frozenset                     # operations in the plan of every outcome
    sample: Callable[[Gen, dict], dict]   # (gen, setup) -> {"instruction", "params", "refs"}
    solve: Callable[[Any, dict], None]
    goals: Callable[[dict, World], list]
    long: bool = False                    # a 5-6-call template (template-mix knob)


def _cart_base(g: Gen, s: dict) -> tuple[dict, list]:
    c = g.customer(s, distract=True)
    items = g.items(s, g.rng.randrange(1, 4))
    s.setdefault("carts", []).append({"customer_id": c["customer_id"], "items": items})
    return c, items


# T01-T03, T20, T21: cart order (+ pay, + ship, + ticket)
def _mk_cart(pay: bool, ship: bool, ticket: str | None):
    def sample(g: Gen, s: dict) -> dict:
        c, items = _cart_base(g, s)
        p = {"email": c["email"], "method": g.rng.choice(SHIP)}
        txt = f"Customer {c['email']} wants to order everything in their cart with {p['method']} shipping"
        if pay:
            p["pay"] = g.rng.choice(PAY)
            txt += f" and pay the order in full by {p['pay']}"
        if ship:
            p["wh"], p["carrier"] = g.rng.choice(WAREHOUSES), g.rng.choice(CARRIERS)
            txt += f", then ship it from {wh_text(p['wh'])} with {CARRIER_NAMES[p['carrier']]}"
        if ticket:
            p["subject"] = g.rng.choice(SUBJECTS)
            prio = "low" if ticket == "low" else "normal"
            txt += f". Then open a shipping ticket with {prio} priority for them with the subject \"{p['subject']}\""
        return {"instruction": txt + ".", "params": p, "refs": {"customer_id": c["customer_id"], "items": items}}

    def solve(api, inst) -> None:
        p = inst["params"]
        cid = api.call("find_customer", {"email": p["email"]})["customer_id"]
        items = api.call("get_cart", {"customer_id": cid})["items"]
        o = api.call("create_order", {"customer_id": cid, "items": items, "shipping_method": p["method"]})
        if pay:
            api.call("pay_order", {"order_id": o["order_id"], "amount": o["total"], "method": p["pay"]})
        if ship:
            api.call("create_shipment", {"order_id": o["order_id"], "warehouse": p["wh"], "carrier": p["carrier"],
                                         "insured": False})
        if ticket:
            api.call("create_ticket", {"customer_id": cid, "subject": p["subject"], "category": "shipping",
                                       "priority": "low" if ticket == "low" else "normal"})

    def goals(inst, w) -> list:
        p, r = inst["params"], inst["refs"]
        out, o = g_one_new_order(w, r["customer_id"], r["items"], p["method"])
        if pay:
            out += g_paid(w, o, p["pay"])
        if ship:
            out += g_shipped(w, o, p["wh"], p["carrier"])
        if ticket:
            out += g_one_new_ticket(w, r["customer_id"], "shipping", p["subject"],
                                    "low" if ticket == "low" else "normal")[0]
        return out

    covers = {"find_customer", "get_cart", "create_order"} | ({"pay_order"} if pay else set()) | \
             ({"create_shipment"} if ship else set()) | ({"create_ticket"} if ticket else set())
    return sample, solve, goals, frozenset(covers)


# T04 reserve + order; T05 order + pay; T22 order + pay + ship
def _mk_product_order(reserve: bool, pay: bool, ship: bool):
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s, distract=True)
        q = g.rng.randrange(1, 4)
        wh = g.rng.choice(WAREHOUSES)
        stock = {w: g.rng.randrange(q + 2, 40) for w in WAREHOUSES}
        prod = g.product(s, stock=stock, distract=True)
        p = {"email": c["email"], "product": prod["name"], "q": q, "method": g.rng.choice(SHIP)}
        if reserve:
            p["wh"] = wh
            txt = (f"Reserve {q} x \"{prod['name']}\" in {wh_text(wh)} for customer {c['email']}, then create an "
                   f"order for them for those {q} unit(s) with {p['method']} shipping.")
        else:
            txt = f"Create an order for customer {c['email']} for {q} x \"{prod['name']}\" with {p['method']} shipping"
            if pay:
                p["pay"] = g.rng.choice(PAY)
                txt += f" and pay it in full by {p['pay']}"
            if ship:
                p["wh"], p["carrier"] = wh, g.rng.choice(CARRIERS)
                txt += f", then ship it from {wh_text(wh)} with {CARRIER_NAMES[p['carrier']]}"
            txt += "."
        return {"instruction": txt, "params": p, "refs": {"customer_id": c["customer_id"], "sku": prod["sku"]}}

    def solve(api, inst) -> None:
        p = inst["params"]
        cid = api.call("find_customer", {"email": p["email"]})["customer_id"]
        sku = api.call("search_products", {"query": p["product"]})["products"][0]["sku"]
        if reserve:
            api.call("reserve_stock", {"sku": sku, "qty": p["q"], "warehouse": p["wh"]})
        o = api.call("create_order", {"customer_id": cid, "items": [{"sku": sku, "qty": p["q"]}],
                                      "shipping_method": p["method"]})
        if pay:
            api.call("pay_order", {"order_id": o["order_id"], "amount": o["total"], "method": p["pay"]})
        if ship:
            api.call("create_shipment", {"order_id": o["order_id"], "warehouse": p["wh"], "carrier": p["carrier"],
                                         "insured": False})

    def goals(inst, w) -> list:
        p, r = inst["params"], inst["refs"]
        out, o = g_one_new_order(w, r["customer_id"], [{"sku": r["sku"], "qty": p["q"]}], p["method"])
        if reserve:
            out += g_one_reservation(w, r["sku"], p["wh"], p["q"])
        if pay:
            out += g_paid(w, o, p["pay"])
        if ship:
            out += g_shipped(w, o, p["wh"], p["carrier"])
        return out

    covers = {"find_customer", "search_products", "create_order"} | ({"reserve_stock"} if reserve else set()) | \
             ({"pay_order"} if pay else set()) | ({"create_shipment"} if ship else set())
    return sample, solve, goals, frozenset(covers)


# T06 (2 warehouses, more stock), T29 (3 warehouses, most stock)
def _mk_most_stock(n_wh: int):
    def sample(g: Gen, s: dict) -> dict:
        q = g.rng.randrange(1, 4)
        whs = g.rng.sample(list(WAREHOUSES), n_wh)
        levels = g.rng.sample(range(q + 1, 40), n_wh)
        stock = {w: g.rng.randrange(q + 1, 40) for w in WAREHOUSES}
        stock.update(dict(zip(whs, levels)))
        prod = g.product(s, stock=stock, distract=True)
        best = max(whs, key=lambda w: stock[w])
        if n_wh == 2:
            where = f"whichever of {wh_text(whs[0])} and {wh_text(whs[1])} has more stock of it"
        else:
            where = "whichever of the three warehouses (JKT, SBY, BDG) has the most stock of it"
        return {"instruction": f"Reserve {q} x \"{prod['name']}\" in {where}.",
                "params": {"product": prod["name"], "q": q, "whs": whs},
                "refs": {"sku": prod["sku"], "best": best}}

    def solve(api, inst) -> None:
        p = inst["params"]
        sku = api.call("search_products", {"query": p["product"]})["products"][0]["sku"]
        avail = {w: api.call("check_stock", {"sku": sku, "warehouse": w})["available"] for w in p["whs"]}
        best = max(p["whs"], key=lambda w: avail[w])
        api.call("reserve_stock", {"sku": sku, "qty": p["q"], "warehouse": best})

    def goals(inst, w) -> list:
        return g_one_reservation(w, inst["refs"]["sku"], inst["refs"]["best"], inst["params"]["q"])

    return sample, solve, goals, frozenset({"search_products", "check_stock", "reserve_stock"})


# T07: W1 if enough else W2 (random)
def t07():
    def sample(g: Gen, s: dict) -> dict:
        q = g.rng.randrange(2, 5)
        w1, w2 = g.rng.sample(list(WAREHOUSES), 2)
        enough = g.rng.random() < 0.5
        stock = {w: g.rng.randrange(q, 40) for w in WAREHOUSES}
        stock[w1] = g.rng.randrange(q, 40) if enough else g.rng.randrange(0, q)
        stock[w2] = g.rng.randrange(q, 40)
        prod = g.product(s, stock=stock, distract=True)
        return {"instruction": f"Reserve {q} x \"{prod['name']}\" in {wh_text(w1)} if it has at least {q} available; "
                               f"otherwise reserve them in {wh_text(w2)}.",
                "params": {"product": prod["name"], "q": q, "w1": w1, "w2": w2},
                "refs": {"sku": prod["sku"], "target": w1 if enough else w2}, "outcome": "w1" if enough else "w2"}

    def solve(api, inst) -> None:
        p = inst["params"]
        sku = api.call("search_products", {"query": p["product"]})["products"][0]["sku"]
        a = api.call("check_stock", {"sku": sku, "warehouse": p["w1"]})["available"]
        api.call("reserve_stock", {"sku": sku, "qty": p["q"], "warehouse": p["w1"] if a >= p["q"] else p["w2"]})

    def goals(inst, w) -> list:
        return g_one_reservation(w, inst["refs"]["sku"], inst["refs"]["target"], inst["params"]["q"])

    return sample, solve, goals, frozenset({"search_products", "check_stock", "reserve_stock"})


def _existing_order(g: Gen, s: dict, status: str, n_items: int | None = None, stock_min: int = 10):
    c = g.customer(s)
    items = g.items(s, n_items or g.rng.randrange(1, 3), stock_min=stock_min)
    o = g.order(s, c, items, status)
    return c, o


# T08 pay; T10 pay + ship
def _mk_pay_existing(ship: bool):
    def sample(g: Gen, s: dict) -> dict:
        c, o = _existing_order(g, s, "created")
        p = {"order_id": o["order_id"], "pay": g.rng.choice(PAY)}
        txt = f"Pay order {o['order_id']} in full by {p['pay']}"
        if ship:
            p["wh"], p["carrier"] = g.rng.choice(WAREHOUSES), g.rng.choice(CARRIERS)
            txt += f" and ship it from {wh_text(p['wh'])} with {CARRIER_NAMES[p['carrier']]}"
        return {"instruction": txt + ".", "params": p, "refs": {"order_id": o["order_id"]}}

    def solve(api, inst) -> None:
        p = inst["params"]
        o = api.call("get_order", {"order_id": p["order_id"]})
        api.call("pay_order", {"order_id": p["order_id"], "amount": o["total"], "method": p["pay"]})
        if ship:
            api.call("create_shipment", {"order_id": p["order_id"], "warehouse": p["wh"], "carrier": p["carrier"],
                                         "insured": False})

    def goals(inst, w) -> list:
        p = inst["params"]
        o = w.orders[p["order_id"]]
        out = g_paid(w, o, p["pay"])
        if ship:
            out += g_shipped(w, o, p["wh"], p["carrier"])
        return out

    return sample, solve, goals, frozenset({"get_order", "pay_order"} | ({"create_shipment"} if ship else set()))


# T09: ship a paid order, insured if total > limit (random)
def t09():
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s)
        insured = g.rng.random() < 0.5
        price = g.rng.randrange(550, 900) * 1000 if insured else g.rng.randrange(20, 300) * 1000
        qty = g.rng.randrange(2, 4)
        prod = g.product(s, price=price, stock={w: g.rng.randrange(qty + 1, 40) for w in WAREHOUSES})
        o = g.order(s, c, [{"sku": prod["sku"], "qty": qty}], "paid")
        assert (o["total"] > INSURE_LIMIT) == insured
        p = {"order_id": o["order_id"], "wh": g.rng.choice(WAREHOUSES), "carrier": g.rng.choice(CARRIERS)}
        return {"instruction": f"Ship paid order {o['order_id']} from {wh_text(p['wh'])} with "
                               f"{CARRIER_NAMES[p['carrier']]}. Insure the shipment if the order total exceeds "
                               f"{rp(INSURE_LIMIT)}; otherwise do not insure it.",
                "params": p, "refs": {"insured": insured}, "outcome": "insured" if insured else "uninsured"}

    def solve(api, inst) -> None:
        p = inst["params"]
        o = api.call("get_order", {"order_id": p["order_id"]})
        api.call("create_shipment", {"order_id": p["order_id"], "warehouse": p["wh"], "carrier": p["carrier"],
                                     "insured": o["total"] > INSURE_LIMIT})

    def goals(inst, w) -> list:
        p = inst["params"]
        return g_shipped(w, w.orders[p["order_id"]], p["wh"], p["carrier"], inst["refs"]["insured"])

    return sample, solve, goals, frozenset({"get_order", "create_shipment"})


# T11 refund+cancel; T12 refund+ticket; T23 refund+comment; T30 refund+cancel+close
def _mk_refund(cancel: bool, ticket: bool, comment: bool, close: bool):
    def sample(g: Gen, s: dict) -> dict:
        c, o = _existing_order(g, s, "paid")
        p = {"order_id": o["order_id"], "reason": g.rng.choice(REFUND)}
        refs = {"order_id": o["order_id"], "payment_id": o["payment_id"], "customer_id": c["customer_id"]}
        txt = f"Refund the payment of order {o['order_id']} in full (reason: {p['reason']})"
        if cancel:
            p["cancel_reason"] = g.rng.choice(CANCEL)
            txt += f" and then cancel the order (reason: {p['cancel_reason']})"
        if ticket:
            p["subject"] = g.rng.choice(SUBJECTS)
            txt += (f", then open a billing ticket with normal priority for the order's customer with the subject "
                    f"\"{p['subject']}\"")
        if comment or close:
            bt = g.ticket(s, c, "billing", "open")
            g.ticket(s, c, "billing", "closed")
            g.ticket(s, c, "shipping", "open")
            g.ticket_distractors(s, c, "billing")
            refs["ticket_id"] = bt["ticket_id"]
        if comment:
            p["text"] = g.rng.choice(COMMENTS)
            txt += f", then add a public comment \"{p['text']}\" to the customer's open billing ticket"
        if close:
            txt += ", then close the customer's open billing ticket"
        return {"instruction": txt + ".", "params": p, "refs": refs}

    def solve(api, inst) -> None:
        p = inst["params"]
        o = api.call("get_order", {"order_id": p["order_id"]})
        api.call("refund_payment", {"payment_id": o["payment_id"], "amount": o["total"], "reason": p["reason"]})
        if cancel:
            api.call("cancel_order", {"order_id": p["order_id"], "reason": p["cancel_reason"]})
        if ticket:
            api.call("create_ticket", {"customer_id": o["customer_id"], "subject": p["subject"],
                                       "category": "billing", "priority": "normal"})
        if comment or close:
            ts = api.call("list_tickets", {"customer_id": o["customer_id"]})["tickets"]
            tid = next(t["ticket_id"] for t in ts if t["category"] == "billing" and t["status"] == "open")
            if comment:
                api.call("add_comment", {"ticket_id": tid, "text": p["text"], "public": True})
            if close:
                api.call("update_ticket", {"ticket_id": tid, "status": "closed"})

    def goals(inst, w) -> list:
        p, r = inst["params"], inst["refs"]
        out = g_refunded(w, r["payment_id"], p["reason"])
        if cancel:
            out += g_cancelled(w, r["order_id"], p["cancel_reason"])
        if ticket:
            out += g_one_new_ticket(w, r["customer_id"], "billing", p["subject"])[0]
        if comment:
            out += g_comment(w, r["ticket_id"], _setup_ticket(inst, r["ticket_id"])["comments"], p["text"], True)
        if close:
            out += g_ticket_status(w, r["ticket_id"], "closed")
        if comment or close:
            for t in inst["setup"]["tickets"]:
                if t["ticket_id"] != r["ticket_id"]:
                    out += g_unchanged_ticket(w, t)
        return out

    covers = {"get_order", "refund_payment"} | ({"cancel_order"} if cancel else set()) | \
             ({"create_ticket"} if ticket else set()) | ({"list_tickets"} if comment or close else set()) | \
             ({"add_comment"} if comment else set()) | ({"update_ticket"} if close else set())
    return sample, solve, goals, frozenset(covers)


# T13 cancel + ticket; T24 cancel + reorder + pay
def _mk_cancel(ticket: bool, reorder: bool):
    def sample(g: Gen, s: dict) -> dict:
        c, o = _existing_order(g, s, "created")
        p = {"order_id": o["order_id"], "cancel_reason": g.rng.choice(CANCEL)}
        refs = {"order_id": o["order_id"], "customer_id": c["customer_id"], "items": o["items"]}
        txt = f"Cancel unpaid order {o['order_id']} (reason: {p['cancel_reason']})"
        if ticket:
            p["category"], p["subject"] = g.rng.choice(CATEGORIES), g.rng.choice(SUBJECTS)
            txt += (f" and open a {p['category']} ticket with normal priority for the order's customer with the "
                    f"subject \"{p['subject']}\"")
        if reorder:
            p["method"], p["pay"] = g.rng.choice(SHIP), g.rng.choice(PAY)
            txt += (f", then create a new order for the same customer with the same items and {p['method']} "
                    f"shipping, and pay it in full by {p['pay']}")
        return {"instruction": txt + ".", "params": p, "refs": refs}

    def solve(api, inst) -> None:
        p = inst["params"]
        o = api.call("get_order", {"order_id": p["order_id"]})
        api.call("cancel_order", {"order_id": p["order_id"], "reason": p["cancel_reason"]})
        if ticket:
            api.call("create_ticket", {"customer_id": o["customer_id"], "subject": p["subject"],
                                       "category": p["category"], "priority": "normal"})
        if reorder:
            n = api.call("create_order", {"customer_id": o["customer_id"], "items": o["items"],
                                          "shipping_method": p["method"]})
            api.call("pay_order", {"order_id": n["order_id"], "amount": n["total"], "method": p["pay"]})

    def goals(inst, w) -> list:
        p, r = inst["params"], inst["refs"]
        out = g_cancelled(w, r["order_id"], p["cancel_reason"])
        if ticket:
            out += g_one_new_ticket(w, r["customer_id"], p["category"], p["subject"])[0]
        if reorder:
            o_goals, o = g_one_new_order(w, r["customer_id"], r["items"], p["method"])
            out += o_goals + g_paid(w, o, p["pay"])
        return out

    covers = {"get_order", "cancel_order"} | ({"create_ticket"} if ticket else set()) | \
             ({"create_order", "pay_order"} if reorder else set())
    return sample, solve, goals, frozenset(covers)


# T14 ticket; T15 ticket + comment
def _mk_ticket(comment: bool):
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s, distract=True)
        p = {"email": c["email"], "category": g.rng.choice(CATEGORIES), "subject": g.rng.choice(SUBJECTS),
             "priority": g.rng.choice(PRIORITIES)}
        txt = (f"Open a {p['category']} ticket with {p['priority']} priority for customer {c['email']} with the "
               f"subject \"{p['subject']}\"")
        if comment:
            p["text"], p["public"] = g.rng.choice(COMMENTS), g.rng.random() < 0.5
            vis = "a public" if p["public"] else "an internal (not public)"
            txt += f", then add {vis} comment \"{p['text']}\" to it"
        return {"instruction": txt + ".", "params": p, "refs": {"customer_id": c["customer_id"]}}

    def solve(api, inst) -> None:
        p = inst["params"]
        cid = api.call("find_customer", {"email": p["email"]})["customer_id"]
        t = api.call("create_ticket", {"customer_id": cid, "subject": p["subject"], "category": p["category"],
                                       "priority": p["priority"]})
        if comment:
            api.call("add_comment", {"ticket_id": t["ticket_id"], "text": p["text"], "public": p["public"]})

    def goals(inst, w) -> list:
        p, r = inst["params"], inst["refs"]
        out, t = g_one_new_ticket(w, r["customer_id"], p["category"], p["subject"], p["priority"])
        if comment:
            out += g_comment(w, t["ticket_id"] if t else None, [], p["text"], p["public"])
        return out

    return sample, solve, goals, frozenset({"find_customer", "create_ticket"} | ({"add_comment"} if comment else set()))


# T16: close all open tickets
def t16():
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s, distract=True)
        opens = [g.ticket(s, c, g.rng.choice(CATEGORIES), "open") for _ in range(2)]
        g.ticket(s, c, g.rng.choice(CATEGORIES), "pending")
        g.ticket(s, c, g.rng.choice(CATEGORIES), "closed")
        g.ticket_distractors(s, c, g.drng.choice(CATEGORIES))
        return {"instruction": f"Close every ticket of customer {c['email']} whose status is \"open\". Leave their "
                               f"other tickets (pending or closed) as they are.",
                "params": {"email": c["email"]}, "refs": {"open": [t["ticket_id"] for t in opens]}}

    def solve(api, inst) -> None:
        cid = api.call("find_customer", {"email": inst["params"]["email"]})["customer_id"]
        for t in api.call("list_tickets", {"customer_id": cid})["tickets"]:
            if t["status"] == "open":
                api.call("update_ticket", {"ticket_id": t["ticket_id"], "status": "closed"})

    def goals(inst, w) -> list:
        out = []
        for tid in inst["refs"]["open"]:
            out += g_ticket_status(w, tid, "closed", f"open ticket closed")
        for t in inst["setup"]["tickets"]:
            if t["ticket_id"] not in inst["refs"]["open"]:
                out += g_unchanged_ticket(w, t)
        return out

    return sample, solve, goals, frozenset({"find_customer", "list_tickets", "update_ticket"})


# T17: open billing ticket -> pending + internal comment
def t17():
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s, distract=True)
        bt = g.ticket(s, c, "billing", "open")
        g.ticket(s, c, "shipping", "open")
        g.ticket(s, c, "billing", "pending")
        g.ticket_distractors(s, c, "billing")
        text = g.rng.choice(COMMENTS)
        return {"instruction": f"Put the open billing ticket of customer {c['email']} on pending and add an internal "
                               f"(not public) comment \"{text}\" to it.",
                "params": {"email": c["email"], "text": text}, "refs": {"ticket_id": bt["ticket_id"]}}

    def solve(api, inst) -> None:
        p = inst["params"]
        cid = api.call("find_customer", {"email": p["email"]})["customer_id"]
        ts = api.call("list_tickets", {"customer_id": cid})["tickets"]
        tid = next(t["ticket_id"] for t in ts if t["category"] == "billing" and t["status"] == "open")
        api.call("update_ticket", {"ticket_id": tid, "status": "pending"})
        api.call("add_comment", {"ticket_id": tid, "text": p["text"], "public": False})

    def goals(inst, w) -> list:
        tid = inst["refs"]["ticket_id"]
        out = g_ticket_status(w, tid, "pending") + g_comment(w, tid, [], inst["params"]["text"], False)
        for t in inst["setup"]["tickets"]:
            if t["ticket_id"] != tid:
                out += g_unchanged_ticket(w, t)
        return out

    return sample, solve, goals, frozenset({"find_customer", "list_tickets", "update_ticket", "add_comment"})


# T18 delayed -> high ticket; T19 + public comment
def _mk_delayed(comment: bool):
    def sample(g: Gen, s: dict) -> dict:
        c, o = _existing_order(g, s, "paid")
        sh = g.shipment(s, o, "delayed")
        p = {"shipment_id": sh["shipment_id"], "subject": g.rng.choice(SUBJECTS)}
        txt = (f"Shipment {sh['shipment_id']} is delayed. Open a high-priority shipping ticket for the customer of its "
               f"order with the subject \"{p['subject']}\"")
        if comment:
            p["text"] = g.rng.choice(COMMENTS)
            txt += f", then add a public comment \"{p['text']}\" to that ticket"
        return {"instruction": txt + ".", "params": p, "refs": {"customer_id": c["customer_id"]}}

    def solve(api, inst) -> None:
        p = inst["params"]
        oid = api.call("track_shipment", {"shipment_id": p["shipment_id"]})["order_id"]
        cid = api.call("get_order", {"order_id": oid})["customer_id"]
        t = api.call("create_ticket", {"customer_id": cid, "subject": p["subject"], "category": "shipping",
                                       "priority": "high"})
        if comment:
            api.call("add_comment", {"ticket_id": t["ticket_id"], "text": p["text"], "public": True})

    def goals(inst, w) -> list:
        p, r = inst["params"], inst["refs"]
        out, t = g_one_new_ticket(w, r["customer_id"], "shipping", p["subject"], "high")
        if comment:
            out += g_comment(w, t["ticket_id"] if t else None, [], p["text"], True)
        return out

    return sample, solve, goals, frozenset({"track_shipment", "get_order", "create_ticket"} |
                                           ({"add_comment"} if comment else set()))


# T25: delivered -> close shipping ticket, else comment (random)
def t25():
    def sample(g: Gen, s: dict) -> dict:
        c, o = _existing_order(g, s, "paid")
        delivered = g.rng.random() < 0.5
        sh = g.shipment(s, o, "delivered" if delivered else "in_transit")
        st = g.ticket(s, c, "shipping", "open")
        g.ticket(s, c, "billing", "open")
        g.ticket_distractors(s, c, "shipping")
        text = g.rng.choice(COMMENTS)
        return {"instruction": f"Check shipment {sh['shipment_id']}. If it has been delivered, close the open shipping "
                               f"ticket of the customer of its order; otherwise add a public comment \"{text}\" to that "
                               f"ticket and leave it open.",
                "params": {"shipment_id": sh["shipment_id"], "text": text},
                "refs": {"ticket_id": st["ticket_id"], "delivered": delivered},
                "outcome": "delivered" if delivered else "in_transit"}

    def solve(api, inst) -> None:
        p = inst["params"]
        tr = api.call("track_shipment", {"shipment_id": p["shipment_id"]})
        cid = api.call("get_order", {"order_id": tr["order_id"]})["customer_id"]
        ts = api.call("list_tickets", {"customer_id": cid})["tickets"]
        tid = next(t["ticket_id"] for t in ts if t["category"] == "shipping" and t["status"] == "open")
        if tr["state"] == "delivered":
            api.call("update_ticket", {"ticket_id": tid, "status": "closed"})
        else:
            api.call("add_comment", {"ticket_id": tid, "text": p["text"], "public": True})

    def goals(inst, w) -> list:
        r = inst["refs"]
        tid = r["ticket_id"]
        if r["delivered"]:
            out = g_ticket_status(w, tid, "closed") + g_no_new_comment(w, tid, [])
        else:
            out = g_ticket_status(w, tid, "open") + g_comment(w, tid, [], inst["params"]["text"], True)
        for t in inst["setup"]["tickets"]:
            if t["ticket_id"] != tid:
                out += g_unchanged_ticket(w, t)
        return out

    return sample, solve, goals, frozenset({"track_shipment", "get_order", "list_tickets"})


# T26: reserve if enough, else product ticket (random)
def t26():
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s, distract=True)
        q = g.rng.randrange(2, 5)
        wh = g.rng.choice(WAREHOUSES)
        enough = g.rng.random() < 0.5
        stock = {w: g.rng.randrange(0, 40) for w in WAREHOUSES}
        stock[wh] = g.rng.randrange(q, 40) if enough else g.rng.randrange(0, q)
        prod = g.product(s, stock=stock, distract=True)
        subject = g.rng.choice(SUBJECTS)
        return {"instruction": f"If {wh_text(wh)} has at least {q} x \"{prod['name']}\" available, reserve {q} of them "
                               f"there. Otherwise open a product ticket with normal priority for customer {c['email']} with the subject "
                               f"\"{subject}\".",
                "params": {"product": prod["name"], "q": q, "wh": wh, "email": c["email"], "subject": subject},
                "refs": {"sku": prod["sku"], "customer_id": c["customer_id"], "enough": enough},
                "outcome": "reserve" if enough else "ticket"}

    def solve(api, inst) -> None:
        p = inst["params"]
        sku = api.call("search_products", {"query": p["product"]})["products"][0]["sku"]
        a = api.call("check_stock", {"sku": sku, "warehouse": p["wh"]})["available"]
        if a >= p["q"]:
            api.call("reserve_stock", {"sku": sku, "qty": p["q"], "warehouse": p["wh"]})
        else:
            cid = api.call("find_customer", {"email": p["email"]})["customer_id"]
            api.call("create_ticket", {"customer_id": cid, "subject": p["subject"], "category": "product",
                                       "priority": "normal"})

    def goals(inst, w) -> list:
        p, r = inst["params"], inst["refs"]
        if r["enough"]:
            return g_one_reservation(w, r["sku"], p["wh"], p["q"]) + g_no_new_ticket(w, r["customer_id"])
        return g_one_new_ticket(w, r["customer_id"], "product", p["subject"])[0] + g_no_reservation(w, r["sku"])

    return sample, solve, goals, frozenset({"search_products", "check_stock"})


# T27: delayed -> refund (late) + public comment on shipping ticket
def t27():
    def sample(g: Gen, s: dict) -> dict:
        c, o = _existing_order(g, s, "paid")
        sh = g.shipment(s, o, "delayed")
        st = g.ticket(s, c, "shipping", "open")
        g.ticket(s, c, "shipping", "closed")
        g.ticket_distractors(s, c, "shipping")
        text = g.rng.choice(COMMENTS)
        return {"instruction": f"Shipment {sh['shipment_id']} is delayed. Refund the payment of its order in full "
                               f"(reason: late) and add a public comment \"{text}\" to the customer's open shipping "
                               f"ticket.",
                "params": {"shipment_id": sh["shipment_id"], "text": text},
                "refs": {"payment_id": o["payment_id"], "ticket_id": st["ticket_id"]}}

    def solve(api, inst) -> None:
        p = inst["params"]
        oid = api.call("track_shipment", {"shipment_id": p["shipment_id"]})["order_id"]
        o = api.call("get_order", {"order_id": oid})
        api.call("refund_payment", {"payment_id": o["payment_id"], "amount": o["total"], "reason": "late"})
        ts = api.call("list_tickets", {"customer_id": o["customer_id"]})["tickets"]
        tid = next(t["ticket_id"] for t in ts if t["category"] == "shipping" and t["status"] == "open")
        api.call("add_comment", {"ticket_id": tid, "text": p["text"], "public": True})

    def goals(inst, w) -> list:
        r = inst["refs"]
        out = g_refunded(w, r["payment_id"], "late") + g_comment(w, r["ticket_id"], [], inst["params"]["text"], True)
        for t in inst["setup"]["tickets"]:
            if t["ticket_id"] != r["ticket_id"]:
                out += g_unchanged_ticket(w, t)
        return out

    return sample, solve, goals, frozenset({"track_shipment", "get_order", "refund_payment", "list_tickets",
                                            "add_comment"})


# T28: cancel unpaid order + close open billing ticket
def t28():
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s, distract=True)
        o = g.order(s, c, g.items(s, g.rng.randrange(1, 3)), "created")
        bt = g.ticket(s, c, "billing", "open")
        g.ticket(s, c, "shipping", "open")
        g.ticket_distractors(s, c, "billing")
        return {"instruction": f"Customer {c['email']} asks to cancel their unpaid order {o['order_id']} (reason: "
                               f"customer_request) and to close their open billing ticket.",
                "params": {"email": c["email"], "order_id": o["order_id"]},
                "refs": {"order_id": o["order_id"], "ticket_id": bt["ticket_id"]}}

    def solve(api, inst) -> None:
        p = inst["params"]
        cid = api.call("find_customer", {"email": p["email"]})["customer_id"]
        api.call("cancel_order", {"order_id": p["order_id"], "reason": "customer_request"})
        ts = api.call("list_tickets", {"customer_id": cid})["tickets"]
        tid = next(t["ticket_id"] for t in ts if t["category"] == "billing" and t["status"] == "open")
        api.call("update_ticket", {"ticket_id": tid, "status": "closed"})

    def goals(inst, w) -> list:
        r = inst["refs"]
        out = g_cancelled(w, r["order_id"], "customer_request") + g_ticket_status(w, r["ticket_id"], "closed")
        for t in inst["setup"]["tickets"]:
            if t["ticket_id"] != r["ticket_id"]:
                out += g_unchanged_ticket(w, t)
        return out

    return sample, solve, goals, frozenset({"find_customer", "cancel_order", "list_tickets", "update_ticket"})


# T31: ship from W1 if it has the item, else W2 (random)
def t31():
    def sample(g: Gen, s: dict) -> dict:
        c = g.customer(s)
        qty = g.rng.randrange(1, 4)
        w1, w2 = g.rng.sample(list(WAREHOUSES), 2)
        enough = g.rng.random() < 0.5
        stock = {w: g.rng.randrange(qty, 40) for w in WAREHOUSES}
        stock[w1] = g.rng.randrange(qty, 40) if enough else g.rng.randrange(0, qty)
        stock[w2] = g.rng.randrange(qty, 40)
        prod = g.product(s, stock=stock)
        o = g.order(s, c, [{"sku": prod["sku"], "qty": qty}], "paid")
        carrier = g.rng.choice(CARRIERS)
        return {"instruction": f"Ship paid order {o['order_id']} with {CARRIER_NAMES[carrier]} from {wh_text(w1)} if "
                               f"that warehouse has enough stock of the order's item; otherwise ship it from "
                               f"{wh_text(w2)}.",
                "params": {"order_id": o["order_id"], "w1": w1, "w2": w2, "carrier": carrier},
                "refs": {"target": w1 if enough else w2}, "outcome": "w1" if enough else "w2"}

    def solve(api, inst) -> None:
        p = inst["params"]
        o = api.call("get_order", {"order_id": p["order_id"]})
        it = o["items"][0]
        a = api.call("check_stock", {"sku": it["sku"], "warehouse": p["w1"]})["available"]
        api.call("create_shipment", {"order_id": p["order_id"], "warehouse": p["w1"] if a >= it["qty"] else p["w2"],
                                     "carrier": p["carrier"], "insured": False})

    def goals(inst, w) -> list:
        p = inst["params"]
        return g_shipped(w, w.orders[p["order_id"]], inst["refs"]["target"], p["carrier"])

    return sample, solve, goals, frozenset({"get_order", "check_stock", "create_shipment"})


def _build() -> dict[str, Template]:
    specs = {
        "T01": _mk_cart(False, False, None), "T02": _mk_cart(True, False, None), "T03": _mk_cart(True, True, None),
        "T04": _mk_product_order(True, False, False), "T05": _mk_product_order(False, True, False),
        "T06": _mk_most_stock(2), "T07": t07(), "T08": _mk_pay_existing(False), "T09": t09(),
        "T10": _mk_pay_existing(True), "T11": _mk_refund(True, False, False, False),
        "T12": _mk_refund(False, True, False, False), "T13": _mk_cancel(True, False), "T14": _mk_ticket(False),
        "T15": _mk_ticket(True), "T16": t16(), "T17": t17(), "T18": _mk_delayed(False), "T19": _mk_delayed(True),
        "T20": _mk_cart(True, False, "low"), "T21": _mk_cart(True, True, "normal"),
        "T22": _mk_product_order(False, True, True), "T23": _mk_refund(False, False, True, False),
        "T24": _mk_cancel(False, True), "T25": t25(), "T26": t26(), "T27": t27(), "T28": t28(),
        "T29": _mk_most_stock(3), "T30": _mk_refund(True, False, False, True), "T31": t31(),
    }
    long_ids = {"T03", "T20", "T21", "T22", "T27", "T29", "T30"}
    return {tid: Template(tid, cov, smp, slv, gl, long=tid in long_ids) for tid, (smp, slv, gl, cov) in specs.items()}


TEMPLATES: dict[str, Template] = _build()


def sample_instance(tid: str, g: Gen, index: int) -> dict:
    setup: dict = {}
    out = TEMPLATES[tid].sample(g, setup)
    inst = {"template": tid, "index": index, "instruction": out["instruction"], "params": out["params"],
            "refs": out["refs"], "setup": setup}
    if "outcome" in out:
        inst["outcome"] = out["outcome"]
    return inst


def goals(inst: dict, world: World) -> list:
    return TEMPLATES[inst["template"]].goals(inst, world)
