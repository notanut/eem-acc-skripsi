"""S3.2a: agent loop mechanics (spec sections 2-5, 6.5) with fake backends and dispatchers."""
import asyncio
import json

import pytest

from eem_acc.env.vtime import run_virtual
from eem_acc.stage3.agent import prompts as P
from eem_acc.stage3.agent.backend import T_LLM, BagOfWordsEmbedder, Gen, StubTokenizer, t_llm
from eem_acc.stage3.agent.loop import Agent, DirectDispatcher, LoopConfig, Outcome, run_rq1_unit
from eem_acc.stage3.agent.memory import EEMStore, make_memory
from eem_acc.stage3.agent.scripted import ScriptedBackend
from eem_acc.stage3.apisuite import ApiSuite, World, sample_stream
from eem_acc.stage3.apisuite.docs import documented_all

TOK = StubTokenizer()


class ListBackend:
    """Returns the given replies in order (action calls only); reflections get a fixed text."""
    id = "list"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages, kind, grammar, max_tokens):
        self.calls.append((kind, grammar, max_tokens, messages))
        text = "Fix it." if kind == "reflection" else self.replies.pop(0)
        return Gen(text, TOK.count_messages(messages), TOK.count(text), "stop", grammar)


def env(seed=1000):
    spec = sample_stream(seed, "dev")
    w = World.build(spec)
    return spec, w, ApiSuite(w, spec)


def agent_for(backend, suite, cond="C0", **cfgkw):
    mem = make_memory(cond, BagOfWordsEmbedder(), TOK)
    cfg = LoopConfig(condition=cond, **cfgkw)
    return Agent(0, backend, TOK, mem, DirectDispatcher(suite), cfg, lambda ti: documented_all()), mem


def env_json(action, args, thought="t"):
    return json.dumps({"thought": thought, "action": action, "args": args})


def test_fence_fallback_and_format_error():
    spec, w, s = env()
    inst = spec["tasks"][0]
    fenced = "```json\n" + env_json("finish", {"summary": "x"}) + "\n```"
    b = ListBackend([fenced])
    a, _ = agent_for(b, s)
    res = run_virtual(a.run_task(inst))
    assert res["steps"][0]["fence"] and res["finished"] and len(b.calls) == 1
    b = ListBackend(["garbage", env_json("finish", {"summary": "x"})])
    a, _ = agent_for(b, s)
    res = run_virtual(a.run_task(inst))
    assert res["steps"][0]["fallback"] and [c[1] for c in b.calls] == [None, "envelope_cap300"] and res["finished"]
    b = ListBackend(["garbage", "still garbage"] + ["x"] * 100)
    a, _ = agent_for(b, s)
    res = run_virtual(a.run_task(inst))
    assert res["steps"][0]["format_error"] and res["traj"][0] == ("still garbage", P.FORMAT_ERROR)


def test_grammar_always_and_action_max_tokens():
    spec, w, s = env()
    b = ListBackend(["garbage"] * 100)
    a, _ = agent_for(b, s, grammar_always=True, action_max_tokens=640)
    res = run_virtual(a.run_task(spec["tasks"][0]))
    assert all(c[1] == "envelope" and c[2] == 640 for c in b.calls)          # grammar-always: no thought cap
    assert len(b.calls) == res["max_steps"] and res["reason"] == "max_steps"        # no fallback call


def test_finish_required_and_wrong_state():
    spec, w, s = env()
    from eem_acc.stage3.agent.loop import task_summary
    inst = spec["tasks"][0]
    b = ListBackend([env_json("finish", {"summary": "nothing done"})])
    a, _ = agent_for(b, s)
    res = run_virtual(a.run_task(inst))
    summ = task_summary(inst, res, w, s)
    assert summ["reason"] == "wrong_state" and not summ["success"]


def test_context_overflow():
    spec, w, s = env()

    class Huge(StubTokenizer):
        def count_messages(self, messages):
            return 9000
    a, _ = agent_for(ListBackend([]), s)
    a.tok = Huge()
    res = run_virtual(a.run_task(spec["tasks"][0]))
    assert res["reason"] == "context_overflow" and res["n_llm"] == 0


def test_t_llm_per_mode_and_virtual_time():
    g1 = Gen("x", 1000, 10, "stop", None)
    g2 = Gen("x", 1000, 10, "stop", "envelope_cap300")
    assert t_llm(g1) == pytest.approx(0.99 + 0.446)
    assert t_llm(g2) == pytest.approx(1.09 + 0.705)
    spec, w, s = env()
    b = ListBackend([env_json("finish", {"summary": "x"})])
    a, _ = agent_for(b, s)

    async def go():
        t0 = asyncio.get_running_loop().time()
        r = await a.run_task(spec["tasks"][0])
        return r, asyncio.get_running_loop().time() - t0
    res, dt = run_virtual(go())
    assert dt == pytest.approx(res["t_llm"])


class TransientDispatcher:
    async def call(self, op_id, args, agent, k, task_index):
        return Outcome(503, {"error": "Service Unavailable"}, True, {"kind": "transient"}, op_id)


def test_transient_notice_no_eem():
    spec, w, s = env()
    b = ListBackend([env_json("get_order", {"order_id": "O-1"}), env_json("finish", {"summary": "x"})])
    mem = make_memory("C2", BagOfWordsEmbedder(), TOK)
    a = Agent(0, b, TOK, mem, TransientDispatcher(), LoopConfig(condition="C2"), lambda ti: documented_all())
    res = run_virtual(a.run_task(spec["tasks"][0]))
    assert res["traj"][0][1] == P.TRANSIENT and not mem.notes


def test_call_numbering_runs_across_tasks():
    spec, w, s = env()
    u = run_rq1_unit(spec, "C0", ScriptedBackend(TOK, "docs"), BagOfWordsEmbedder(), TOK, n_tasks=4)
    ks = [st["k"] for st in u["steps"] if "k" in st]
    assert ks == list(range(1, len(ks) + 1))


def test_get_api_docs_errors_create_no_notes_and_docs_shown_verbatim():
    spec, w, s = env()
    b = ListBackend([env_json("get_api_docs", {"operation": "nope"}),
                     env_json("get_api_docs", {"operation": "pay_order"}), env_json("finish", {"summary": "x"})])
    a, mem = agent_for(b, s, cond="C2")
    res = run_virtual(a.run_task(spec["tasks"][0]))
    assert not mem.notes
    assert res["traj"][1][1].startswith("Observation: 200\n") and "\npay_order" in res["traj"][1][1]


def test_prompts_identical_across_conditions_with_empty_memory():
    spec, w, s = env()
    inst = spec["tasks"][0]
    seen = {}
    for cond in ("C0", "C1", "C2", "C3", "F", "AN", "A1", "A2", "C2-OU"):
        b = ListBackend([env_json("finish", {"summary": "x"})])
        a, _ = agent_for(b, s, cond=cond)
        run_virtual(a.run_task(inst))
        seen[cond] = b.calls[0][3]
    assert all(v == seen["C0"] for v in seen.values())
    assert seen["C0"][0]["content"].endswith(P.DOCS_WORDING["W1"])


def test_r_shows_current_docs_same_rules():
    spec = sample_stream(1000, "dev")
    u0 = run_rq1_unit(spec, "R", ListBackend([env_json("finish", {"summary": "x"})]), BagOfWordsEmbedder(), TOK,
                      n_tasks=1)
    assert u0["stamp"]["condition"] == "R"


def test_two_agents_note_visibility_timing():                    # spec 6.5
    spec = sample_stream(1000, "dev")
    w = World.build(spec)
    s = ApiSuite(w, spec)
    mem = EEMStore(BagOfWordsEmbedder(), TOK)
    drifted = spec["drift_schedule"][0][1][0].split(":")[0]
    from test_s31_suite import DOC_ARGS
    bad = env_json(drifted, {"x": 1})
    ba = ListBackend([bad] + [env_json("finish", {"summary": "x"})])
    bb = ListBackend([env_json("get_api_docs", {"operation": drifted})] * 3 + [env_json("finish", {"summary": "x"})])
    cfg = LoopConfig(condition="C2")
    A = Agent(0, ba, TOK, mem, DirectDispatcher(s), cfg, lambda ti: documented_all())
    B = Agent(1, bb, TOK, mem, DirectDispatcher(s), cfg, lambda ti: documented_all())

    async def both():
        return await asyncio.gather(A.run_task(spec["tasks"][0]), B.run_task(spec["tasks"][1]))
    ra, rb = run_virtual(both())
    t_note = next(e["t"] for e in mem.events if e["event"] == "create")
    starts = [st["t_start"] for st in rb["steps"]]
    assert min(starts) < t_note <= max(starts)                   # B has steps before and after the note
    for st in rb["steps"]:
        visible = bool(st["retrieved"])
        assert visible == (st["t_start"] >= t_note), (st["t_start"], t_note)


def test_r_prompt_shows_current_docs_with_same_rules():
    spec = sample_stream(1000, "dev")
    seen = {}
    for cond in ("C0", "R"):
        b = ListBackend([env_json("finish", {"summary": "x"})])
        run_rq1_unit(spec, cond, b, BagOfWordsEmbedder(), TOK, n_tasks=1)
        seen[cond] = b.calls[0][3][0]["content"]
    assert seen["C0"] != seen["R"]
    assert seen["C0"].split("Operations:")[0] == seen["R"].split("Operations:")[0]
    assert seen["C0"].endswith(P.DOCS_WORDING["W1"]) and seen["R"].endswith(P.DOCS_WORDING["W1"])


def test_reflection_overflow_is_skipped_but_note_stored():
    spec, w, s = env()
    drifted = spec["drift_schedule"][0][1][0].split(":")[0]

    class Tight(StubTokenizer):
        def count_messages(self, messages):
            return 8150 if messages[0]["role"] == "user" else 100
    b = ListBackend([env_json(drifted, {"x": 1}), env_json("finish", {"summary": "x"})])
    a, mem = agent_for(b, s, cond="C2")
    a.tok = Tight()
    res = run_virtual(a.run_task(spec["tasks"][0]))
    assert res["steps"][0].get("reflection_skipped") and mem.notes and mem.notes[1].reflection is None


def test_skipped_reads_and_placeholder_args():
    from eem_acc.stage3.agent.loop import task_summary
    from eem_acc.stage3.apisuite.oracle import reference_plan
    spec, w, s = env()
    inst = spec["tasks"][0]
    b = ListBackend([env_json("get_order", {"order_id": "<order_id>"}), env_json("finish", {"summary": "x"})])
    a, _ = agent_for(b, s)
    res = run_virtual(a.run_task(inst))
    summ = task_summary(inst, res, w, s)
    reads = [op for op, _ in reference_plan(inst) if op in ("find_customer", "get_cart", "get_order",
                                                             "search_products", "check_stock", "track_shipment",
                                                             "list_tickets")]
    assert summ["skipped_reads"] >= len(reads) - 1 and summ["placeholder_args"] == 1


def test_echo_trap_from_truth():
    from eem_acc.stage3.agent.loop import _echo_trap
    assert _echo_trap({"pair_ids": ["reserve_stock:enum"]}, {"warehouse": "JKT"}, {"JKT"})
    assert not _echo_trap({"pair_ids": ["reserve_stock:enum"]}, {"warehouse": "jakarta"}, {"JKT"})
    assert not _echo_trap({"pair_ids": ["reserve_stock:rename"]}, {"warehouse": "JKT"}, {"JKT"})
