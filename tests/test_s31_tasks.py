"""S3.1: templates, oracles, necessity, checker, isolation (spec sections 7, 11)."""
import copy
import json
import random

import pytest

from eem_acc.stage3.apisuite.oracle import (DocsOnlyAPI, NaiveAPI, OracleAPI, OracleError, check, min_calls,
                                            reference_plan, scratch_suite)
from eem_acc.stage3.apisuite.streams import sample_stream
from eem_acc.stage3.apisuite.suite import ApiSuite
from eem_acc.stage3.apisuite.templates import TEMPLATES, Gen, sample_instance
from eem_acc.stage3.apisuite.world import World

WRITES = {"create_order", "pay_order", "refund_payment", "cancel_order", "reserve_stock", "create_shipment",
          "create_ticket", "update_ticket", "add_comment"}
STREAMS = [(s, "dev", "rq1") for s in range(1000, 1050)] + [(s, "test", "rq1") for s in range(2000, 2050)] + \
          [(s, "test", "sp") for s in range(3000, 3050)] + [(s, "test", "rq3") for s in range(4000, 4050)]


def _instances(n_per_template=30):
    for tid in TEMPLATES:
        for k in range(n_per_template):
            yield sample_instance(tid, Gen(random.Random(f"t:{tid}:{k}"), 2), 0)


def test_31_templates_and_coverage():
    assert len(TEMPLATES) == 31
    from eem_acc.stage3.apisuite.ops import OP_NAMES
    for op in OP_NAMES:
        n = sum(op in t.covers for t in TEMPLATES.values())
        assert n >= 4, (op, n)                                                  # D70


def test_covers_hold_for_every_outcome():
    for inst in _instances():
        ops = {op for op, _ in reference_plan(inst)}
        assert TEMPLATES[inst["template"]].covers <= ops, inst["template"]


def test_conditional_templates_produce_both_outcomes():
    seen = {}
    for inst in _instances(60):
        if "outcome" in inst:
            seen.setdefault(inst["template"], set()).add(inst["outcome"])
    assert set(seen) == {"T07", "T09", "T25", "T26", "T31"}
    assert all(len(v) == 2 for v in seen.values())


def test_reference_plan_lengths():
    expected = {"T01": 3, "T02": 4, "T03": 5, "T04": 4, "T05": 4, "T06": 4, "T07": 3, "T08": 2, "T09": 2, "T10": 3,
                "T11": 3, "T12": 3, "T13": 3, "T14": 2, "T15": 3, "T16": 4, "T17": 4, "T18": 3, "T19": 4, "T20": 5,
                "T21": 6, "T22": 5, "T23": 4, "T24": 4, "T25": 4, "T27": 5, "T28": 4, "T29": 5, "T30": 5, "T31": 3}
    for inst in _instances(10):
        n = min_calls(inst)
        if inst["template"] == "T26":
            assert n == (3 if inst["outcome"] == "reserve" else 4)
        else:
            assert n == expected[inst["template"]], inst["template"]


def test_instance_solved_by_oracle_on_scratch_world_and_goals_hold():
    for inst in _instances():
        s = scratch_suite(inst)
        TEMPLATES[inst["template"]].solve(OracleAPI(s, 0), inst)
        ok, goals = check(inst, s.world)
        assert ok, (inst["template"], [g for g in goals if not g[1]])


def test_unsolved_instance_fails():
    for inst in _instances(3):
        ok, goals = check(inst, scratch_suite(inst).world)
        assert not ok, inst["template"]


def _run_stream(spec, api_cls):
    w = World.build(spec)
    suite = ApiSuite(w, spec)
    for t in spec["tasks"]:
        api = api_cls(suite, t["index"])
        TEMPLATES[t["template"]].solve(api, t)
        ok, goals = check(t, w)
        assert ok, (spec["seed"], t["template"], [g for g in goals if not g[1]])
        assert len(api.trace) == min_calls(t), (spec["seed"], t["template"])
    return suite


@pytest.mark.parametrize("seed,split,kind", STREAMS[::5])
def test_oracle_solves_full_streams_with_min_calls(seed, split, kind):
    _run_stream(sample_stream(seed, split, kind=kind), OracleAPI)


def test_oracle_all_streams():
    for seed, split, kind in STREAMS:
        _run_stream(sample_stream(seed, split, kind=kind), OracleAPI)


def test_docs_only_oracle_solves_all_streams():
    for seed, split, kind in STREAMS:
        _run_stream(sample_stream(seed, split, kind=kind), DocsOnlyAPI)


class _NaiveSolve(NaiveAPI):
    """Runs the template with documented-form calls and stops at the first failure."""

    class Stop(Exception):
        pass

    def call(self, op, doc_args):
        r = self.call_raw(op, doc_args)
        if r.status != 200 or (isinstance(r.body, dict) and r.body.get("status") == "error"):
            raise self.Stop()
        return r.body


def test_naive_documented_calls_fail_exactly_on_drifted_operations():
    for seed, split, kind in STREAMS:
        spec = sample_stream(seed, split, kind=kind)
        w = World.build(spec)
        suite = ApiSuite(w, spec)
        for t in spec["tasks"]:
            api = _NaiveSolve(suite, t["index"])
            try:
                TEMPLATES[t["template"]].solve(api, t)
            except _NaiveSolve.Stop:
                pass
            active = {p.op: p for p in suite.active_pairs(t["index"])}
            for (op, _), r in zip(api.trace, api.responses):
                failed = r.status != 200 or (isinstance(r.body, dict) and r.body.get("status") == "error")
                assert failed == (op in active), (seed, t["template"], op, r.status)
                if failed:
                    assert r.truth.kind in ("drift", "silent_drift")
                    assert {pid.split(":")[0] for pid in r.truth.pair_ids} == {op}


def test_necessity_writes_and_reads():
    for inst in _instances(10):
        plan = reference_plan(inst)
        # every write is needed: replaying the plan without it fails a sub-goal
        for j, (op, args) in enumerate(plan):
            if op not in WRITES:
                continue
            s = scratch_suite(inst)
            for k, (op2, a2) in enumerate(plan):
                if k != j:
                    s.execute(op2, copy.deepcopy(a2), 0)
            assert not check(inst, s.world)[0], (inst["template"], op)
        # every read returns a value used later as an argument, or decides a branch / comparison
        s = scratch_suite(inst)
        bodies = [s.execute(op, copy.deepcopy(a), 0).body for op, a in plan]
        for j, (op, _) in enumerate(plan):
            if op in WRITES:
                continue
            vals = _scalars(bodies[j]) - {"True", "False", "None"}
            later = set().union(*[_scalars(a) for _, a in plan[j + 1:]]) if plan[j + 1:] else set()
            used = bool(vals & later)
            decides = op in DECISION_READS.get(inst["template"], ())
            assert used or decides, (inst["template"], op)


# reads whose result decides a comparison or branch rather than feeding an argument
DECISION_READS = {"T06": ("check_stock",), "T29": ("check_stock",), "T07": ("check_stock",),
                  "T26": ("check_stock",), "T31": ("check_stock",), "T25": ("track_shipment",), "T09": ("get_order",)}


def _scalars(x):
    if isinstance(x, dict):
        return set().union(*[_scalars(v) for v in x.values()]) if x else set()
    if isinstance(x, list):
        return set().union(*[_scalars(v) for v in x]) if x else set()
    return {str(x)}


# -------------------------------------------------------------------------------------------------------------- checker
def _solved(tid, k=0):
    inst = sample_instance(tid, Gen(random.Random(f"neg:{tid}:{k}"), 2), 0)
    s = scratch_suite(inst)
    TEMPLATES[tid].solve(OracleAPI(s, 0), inst)
    assert check(inst, s.world)[0]
    return inst, s


def _failed(inst, s):
    ok, goals = check(inst, s.world)
    return [n for n, g in goals if not g]


def test_checker_negatives():
    inst, s = _solved("T02")                                               # duplicate order
    s.execute("create_order", {"customer_id": inst["refs"]["customer_id"], "items": inst["refs"]["items"],
                               "shipping_method": inst["params"]["method"]}, 0)
    assert "exactly one new order" in _failed(inst, s)
    inst, s = _solved("T02")                                               # wrong payment method
    pid = s.world.created[-1][1]
    s.world.payments[pid]["method"] = next(m for m in ("card", "transfer", "wallet") if m != inst["params"]["pay"])
    assert _failed(inst, s) == ["payment method"]
    inst, s = _solved("T03")                                               # wrong warehouse
    sid = next(i for k, i in s.world.created if k == "shipments")
    s.world.shipments[sid]["warehouse"] = next(w for w in ("JKT", "SBY", "BDG") if w != inst["params"]["wh"])
    assert _failed(inst, s) == ["warehouse"]
    inst, s = _solved("T15")                                               # missing comment
    tid = next(i for k, i in s.world.created if k == "tickets")
    s.world.tickets[tid]["comments"].clear()
    assert "exactly one new comment" in _failed(inst, s)
    inst, s = _solved("T16")                                               # a non-open ticket touched
    other = next(t for t in inst["setup"]["tickets"] if t["ticket_id"] not in inst["refs"]["open"]
                 and t["customer_id"] == inst["refs"].get("customer_id", t["customer_id"]))
    s.execute("update_ticket", {"ticket_id": other["ticket_id"], "status": "open" if other["status"] != "open"
                                else "closed"}, 0)
    assert "other ticket unchanged" in _failed(inst, s)
    inst, s = _solved("T14")                                               # subject must match
    tid = next(i for k, i in s.world.created if k == "tickets")
    s.world.tickets[tid]["subject"] += " now"
    assert _failed(inst, s) == ["ticket subject"]


def test_checker_counts_enum_by_meaning():
    inst = sample_instance("T01", Gen(random.Random("enum"), 2), 0)
    spec = {"styles": {op: "informative" for op in __import__("eem_acc.stage3.apisuite.ops", fromlist=["OPS"]).OPS},
            "drift_schedule": [[0, ["create_order:enum"]]], "tasks": [inst]}
    w = World()
    w.add_setup(inst["setup"])
    s = ApiSuite(w, spec)
    TEMPLATES["T01"].solve(OracleAPI(s, 0), inst)
    o = next(w.orders[i] for k, i in w.created if k == "orders")
    assert o["shipping_method"] in ("standard", "express") and check(inst, w)[0]


# ------------------------------------------------------------------------------------------------------------ isolation
def _touched(before: dict, after: dict) -> set:
    out = set()
    for kind in ("customers", "products", "orders", "payments", "reservations", "shipments", "tickets"):
        for k, v in after[kind].items():
            if before[kind].get(k) != v:
                out.add((kind, k))
    return out


def _task_ids(t: dict) -> set:
    ids = set()
    for kind, ents in t["setup"].items():
        for e in ents:
            ids |= {v for kk, v in e.items() if isinstance(v, str) and kk.endswith(("_id", "sku"))}
    return ids


@pytest.mark.parametrize("seed", [2000, 2001, 4000])
def test_tasks_touch_only_their_own_entities_and_order_does_not_matter(seed):
    spec = sample_stream(seed, "test", kind="rq3" if seed >= 4000 else "rq1")
    w = World.build(spec)
    suite = ApiSuite(w, spec)
    results = {}
    for t in spec["tasks"]:
        before = w.snapshot()
        TEMPLATES[t["template"]].solve(OracleAPI(suite, t["index"]), t)
        after = w.snapshot()
        own = _task_ids(t)
        for kind, k in _touched(before, after):
            ent = after[kind][k]
            refs = {v for kk, v in ent.items() if isinstance(v, str) and kk.endswith(("_id", "sku"))}
            refs |= {i["sku"] for i in ent.get("items", [])}
            assert refs - {k} <= own | {i for _, i in after["created"]}, (t["template"], kind)
        results[t["index"]] = check(t, w)[0]
    # reversed order on a fresh world gives the same results
    w2 = World.build(spec)
    suite2 = ApiSuite(w2, spec)
    for t in reversed(spec["tasks"]):
        TEMPLATES[t["template"]].solve(OracleAPI(suite2, t["index"]), t)
    assert {t["index"]: check(t, w2)[0] for t in spec["tasks"]} == results


def test_checker_partial_refund_fails():
    inst, s = _solved("T11")
    w = s.world
    pid = inst["refs"]["payment_id"]
    w.payments[pid]["refunded_amount"] -= 1
    assert "refunded in full" in _failed(inst, s)


def _own_products(t):
    return {p["sku"]: p["name"] for p in t["setup"].get("products", [])}


@pytest.mark.parametrize("seed", [2000, 2003, 4001])
def test_partial_name_search_reaches_only_the_tasks_own_products(seed):       # D77
    spec = sample_stream(seed, "test", kind="rq3" if seed >= 4000 else "rq1")
    w = World.build(spec)
    suite = ApiSuite(w, spec)
    for t in spec["tasks"]:
        own = _own_products(t)
        for name in own.values():
            brand_item = name.rsplit(" ", 2)[0]
            r = suite.execute("search_products", {"query": brand_item, "limit": 100}, t["index"])
            if r.status == 200 and "products" in r.body:          # a silent drift returns an error payload
                assert {p["sku"] for p in r.body["products"]} <= set(own), (t["template"], brand_item)


@pytest.mark.parametrize("seed", [2000, 2005, 4002])
def test_stray_writes_in_one_task_do_not_change_other_tasks(seed):
    spec = sample_stream(seed, "test", kind="rq3" if seed >= 4000 else "rq1")
    for j in range(0, spec["n_tasks"], 4):
        w = World.build(spec)
        suite = ApiSuite(w, spec)
        tj = spec["tasks"][j]
        # an agent in task j drains every product it can reach and cancels/pays every order it can see
        for sku, p in [(p["sku"], p) for p in tj["setup"].get("products", [])]:
            for wh in ("JKT", "SBY", "BDG"):
                n = w.products[sku]["stock"][wh]
                if n:
                    suite.execute("reserve_stock", {"sku": sku, "qty": n, "warehouse": wh}, j)
        for o in tj["setup"].get("orders", []):
            suite.execute("cancel_order", {"order_id": o["order_id"], "reason": "out_of_stock"}, j)
        for t in spec["tasks"]:
            if t["index"] == j:
                continue
            TEMPLATES[t["template"]].solve(OracleAPI(suite, t["index"]), t)
            assert check(t, w)[0], (seed, j, t["template"])
