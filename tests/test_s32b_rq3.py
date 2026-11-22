"""S3.2b: RQ3 wiring (spec docs/s32b_spec.md)."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eem_acc import sim
from eem_acc.env import run_virtual
from eem_acc.scenarios import acc_method, build, fixed_method
from eem_acc.stage3.agent import BagOfWordsEmbedder, ScriptedBackend, StubTokenizer
from eem_acc.stage3.agent import prompts as P
from eem_acc.stage3.apisuite import ApiSuite, World, sample_stream
from eem_acc.stage3.rq3 import FROZEN, LLMRequest, SemanticServer, WARMUP, method_for, run_rq3_unit

REPO = Path(__file__).resolve().parents[1]
TOK = StubTokenizer()


def unit(cond="C2", win="acc", seed=5000, **kw):
    spec = sample_stream(seed, "dev", kind="rq3")
    strat = "docs" if cond == "C0" else "memory"
    return run_rq3_unit(spec, cond, win, ScriptedBackend(TOK, strat), BagOfWordsEmbedder(), TOK, **kw)


def test_configs_from_single_source():
    frozen = json.loads(FROZEN.read_text())
    assert method_for("acc") == acc_method(**frozen["acc"])
    assert method_for("none") == fixed_method(128) and method_for("fixed4") == fixed_method(4)
    sc = build("CAP-steady", 32, method_for("none"), 0)
    assert sc.acc.frozen and sc.acc.w0 == 128


@pytest.mark.parametrize("window", ["acc", "none"])
def test_scripted_only_run_bit_identical_to_mockserver(window, monkeypatch):
    sc = build("CAP-steady", 32, method_for(window), 0)
    ref = sim.run_scenario(sc)
    monkeypatch.setattr(sim, "MockServer", lambda cfg, seed: SemanticServer(cfg, seed, None))
    got = sim.run_scenario(sc)
    assert json.dumps(ref, sort_keys=True, default=str) == json.dumps(got, sort_keys=True, default=str)


def test_llm_request_is_the_plain_tuple_for_keyed_randomness():
    r = LLMRequest(1000, 7, "get_order", {"order_id": "O-1"}, 3)
    assert r == (1000, 7) and hash(r) == hash((1000, 7)) and tuple(r) == (1000, 7)
    agent, k = r
    assert (agent, k) == (1000, 7) and r.op == "get_order"
    from eem_acc.keyed import KeyedNormals
    kn = KeyedNormals(5, tag=1)
    assert kn.get(*r) == kn.get(1000, 7)


def _semantic_env(**server_kw):
    spec = sample_stream(5001, "dev", kind="rq3")
    w = World.build(spec)
    suite = ApiSuite(w, spec)
    sc = build("CAP-steady", 32, method_for("acc"), 0)
    import dataclasses
    srv = SemanticServer(dataclasses.replace(sc.server, **server_kw), 1, suite)
    return spec, suite, srv


def test_semantic_4xx_bookkeeping_and_200_latency():
    spec, suite, srv = _semantic_env()

    async def go():
        bad = await srv.handle(LLMRequest(1000, 1, "nope", {}, 0))
        inst = spec["tasks"][0]
        ok = await srv.handle(LLMRequest(1000, 2, "get_api_docs", {"operation": "get_order"}, 0))
        return bad, ok
    bad, ok = run_virtual(go())
    assert bad.status == 404 and bad.latency == srv.cfg.error_latency and srv.counts["sem4xx"] == 1
    assert ok.status == 200 and ok.latency > srv.cfg.error_latency and srv.counts[200] == 1
    assert [e[2] for e in srv.log] == [404, 200] and srv.inflight == 0


def test_idempotent_retry_after_timeout_changes_state_once():
    spec, suite, srv = _semantic_env(base_latency=3.0, latency_sigma=0.0)
    inst = next(t for t in spec["tasks"] if t["template"] in ("T14", "T15"))
    from eem_acc.stage3.apisuite.oracle import reference_plan
    cid = inst["refs"]["customer_id"]
    req_args = {"customer_id": cid, "subject": "x", "category": "billing", "priority": "low"}
    from eem_acc.agent import Gateway, RetryConfig
    from eem_acc.acc import ACCController
    sc = build("CAP-steady", 32, method_for("acc"), 0)
    gw = Gateway(srv, ACCController(sc.acc), RetryConfig(max_retries=2, backoff_base=0.1, backoff_cap=0.2,
                                                          timeout=1.0), 2)
    v = suite.views(inst["index"])["create_ticket"]
    from eem_acc.stage3.apisuite.drift import to_current

    async def go():
        rec = await gw.call(LLMRequest(1000, 1, v.route, to_current(v, req_args), inst["index"]))
        await gw.shutdown()
        return rec
    rec = run_virtual(go())
    from eem_acc.status import CLIENT_TIMEOUT
    assert rec.attempts == 3 and rec.status == CLIENT_TIMEOUT                # every attempt timed out
    created = [i for k, i in suite.world.created if k == "tickets"]
    assert len(created) == 1                                        # executed once despite 3 admissions
    assert suite.executed((1000, 1)) is not None


def test_until_done_and_cells_run():
    for cond, win in (("C0", "acc"), ("C2", "acc"), ("C2", "none"), ("C2", "fixed4"), ("C2-private", "acc")):
        u = unit(cond, win)
        assert len(u["tasks"]) == 20 and not u["capped"] and u["t_end"] < 3600
        assert all(t["t_start"] >= WARMUP for t in u["tasks"])
        assert all(s["t_start"] >= WARMUP for s in u["steps"])
        assert all(not t["unfinished_at_t_max"] for t in u["tasks"])


def test_t_max_marks_unfinished_tasks_with_partial_logs():
    u = unit("C2", "acc", t_max=60.0)
    assert u["capped"]
    unfinished = [t for t in u["tasks"] if t["unfinished_at_t_max"]]
    assert unfinished and all(not t["success"] for t in unfinished)
    assert any(t["steps"] > 0 for t in unfinished)


def test_manipulation_no_window_hurts_llm_calls():
    a = unit("C2", "acc")
    n = unit("C2", "none")
    assert n["llm_call_success_rate"] <= a["llm_call_success_rate"]
    assert n["llm_metrics"]["overload_rate"] > a["llm_metrics"]["overload_rate"]


def test_transient_path_and_hidden_executions():
    u = unit("C2", "none", server_override={"base_latency": 1.5, "latency_sigma": 0.3})
    assert any(c.get("transient") for c in u["llm_calls"])
    assert u["hidden_executions"] >= 1
    traj_obs = [s for s in u["steps"] if s.get("transient")]
    assert traj_obs


def test_private_store_no_cross_agent_notes():
    u = unit("C2-private", "acc")
    for i, notes in enumerate(u["final_notes"]):
        assert all(n["owner"][0] == 1000 + i for n in notes)


def test_shared_store_has_notes_from_several_agents():
    u = unit("C2", "acc")
    owners = {n["owner"][0] for n in u["final_notes"]}
    assert len(owners) >= 2


def test_determinism_same_unit_twice_and_hashseed():
    a = json.dumps(unit("C2", "acc", seed=5002), sort_keys=True, default=str)
    b = json.dumps(unit("C2", "acc", seed=5002), sort_keys=True, default=str)
    assert a == b
    code = ("import json,sys,hashlib;sys.path.insert(0,'src');sys.path.insert(0,'tests');"
            "from test_s32b_rq3 import unit;"
            "d=unit('C2','acc',seed=5002);d['stamp'].pop('code_commit');"
            "print(hashlib.sha256(json.dumps(d,sort_keys=True,default=str).encode()).hexdigest())")
    outs = set()
    for hs in ("0", "4242"):
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=hs), check=True)
        outs.add(r.stdout.strip())
    assert len(outs) == 1


def test_generate_does_not_advance_virtual_time():
    class Clocked(ScriptedBackend):
        def generate(self, messages, kind, grammar, max_tokens):
            t0 = asyncio.get_event_loop().time()
            g = super().generate(messages, kind, grammar, max_tokens)
            assert asyncio.get_event_loop().time() == t0
            return g
    spec = sample_stream(5003, "dev", kind="rq3")
    run_rq3_unit(spec, "C2", "acc", Clocked(TOK, "memory"), BagOfWordsEmbedder(), TOK)


def test_llm_time_scale_shortens_runs():
    from eem_acc.stage3.agent.loop import LoopConfig
    full = unit("C2", "acc", seed=5004)
    fast = unit("C2", "acc", seed=5004, cfg=LoopConfig(llm_time_scale=1 / 3))
    assert fast["t_end"] < full["t_end"] and fast["stamp"]["llm_time_scale"] == pytest.approx(1 / 3)


def test_server_log_matches_gateway_and_dispatcher_matches_steps():
    u = unit("C2", "acc", seed=5005)
    # every non-transient LLM call's final status equals the status the agent's step recorded
    by_k = {(c["agent"], c["k"]): c for c in u["llm_calls"]}
    for st in u["steps"]:
        if "k" in st and not st.get("transient"):
            assert by_k[(st["agent"], st["k"])]["status"] == st["status"]
    assert any(st.get("action") == "get_api_docs" for st in u["steps"])      # docs calls go through the gateway


def test_server_log_counts_equal_gateway_attempts():
    from eem_acc.agent import Gateway
    from eem_acc.acc import ACCController
    from eem_acc.status import CLIENT_TIMEOUT
    sc = build("CAP-steady", 32, method_for("none"), 3)
    srv_holder = {}

    def mk(cfg, seed):
        srv_holder["s"] = SemanticServer(cfg, seed, None)
        return srv_holder["s"]
    import eem_acc.sim as S
    orig = S.MockServer
    S.MockServer = mk
    try:
        gw = S.simulate(sc)
    finally:
        S.MockServer = orig
    srv = srv_holder["s"]
    timeout = sc.retry.timeout
    n_to_log = sum(1 for e in srv.log if e[2] == 200 and e[3] > timeout)
    n_200_log = sum(1 for e in srv.log if e[2] == 200 and e[3] <= timeout)
    cnt = {st: sum(a.status == st for a in gw.attempts) for st in (200, 429, 503, CLIENT_TIMEOUT)}
    assert n_200_log == cnt[200]
    assert sum(1 for e in srv.log if e[2] == 503) == cnt[503] and sum(1 for e in srv.log if e[2] == 429) == cnt[429]
    assert n_to_log <= cnt[CLIENT_TIMEOUT]          # abandoned attempts cancelled at the end are not logged


# ---------------------------------------------------------------------------------------- S3.2b code review (M/L)
import copy
import dataclasses
import pickle

from eem_acc import status as ST
from eem_acc.acc import ACCController
from eem_acc.agent import RetryConfig
from eem_acc.stage3 import rq3
from eem_acc.stage3.apisuite.drift import to_current
from eem_acc.stage3.rq3 import GatewayDispatcher, TaggedGateway, is_llm


def captured_unit(monkeypatch, cond="C2", win="acc", seed=5005, **kw):
    """A unit with the suite, server and gateway captured for checks against the internals."""
    got = {}

    class S(rq3.ApiSuite):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            got["suite"] = self

    class Srv(rq3.SemanticServer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            got["server"] = self

    class G(rq3.TaggedGateway):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            got["gw"] = self
            got["shutdown"] = 0

        async def shutdown(self):
            got["shutdown"] += 1
            await super().shutdown()

    orig_rv = rq3.run_virtual

    def rv(coro):
        async def wrap():
            r = await coro
            got["pending"] = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
            return r
        return orig_rv(wrap())
    monkeypatch.setattr(rq3, "ApiSuite", S)
    monkeypatch.setattr(rq3, "SemanticServer", Srv)
    monkeypatch.setattr(rq3, "TaggedGateway", G)
    monkeypatch.setattr(rq3, "run_virtual", rv)
    return unit(cond, win, seed=seed, **kw), got


def test_dispatcher_status_equals_suite_and_server_log_equals_gateway(monkeypatch):     # review M6
    u, got = captured_unit(monkeypatch)
    suite, srv, gw = got["suite"], got["server"], got["gw"]
    n = 0
    for c in u["llm_calls"]:
        if not c.get("transient"):
            assert suite.executed((c["agent"], c["k"])).status == c["status"]
            n += 1
    assert n > 50 and any(c["status"] in (400, 404, 410, 422) for c in u["llm_calls"])   # semantic 4xx present
    llm_att = [a for a, r in zip(gw.attempts, gw.attempt_requests) if is_llm(r)]
    llm_log = [e for e in srv.log if is_llm(e[0])]
    timeout = gw.retry.timeout
    for st in (429, 503, 404, 400, 422, 410):
        assert sum(a.status == st for a in llm_att) == sum(e[2] == st for e in llm_log), st
    assert sum(a.status == 200 for a in llm_att) == sum(e[2] == 200 and e[3] <= timeout for e in llm_log)
    assert sum(e[2] == 200 and e[3] > timeout for e in llm_log) <= sum(a.status == ST.CLIENT_TIMEOUT for a in llm_att)
    assert u["llm_metrics"]["n_attempts"] == len(llm_att) == sum(c["attempts"] for c in u["llm_calls"])
    # LLM metrics and J_rq3 recomputed independently from the tagged attempts (re-review M1)
    from eem_acc.scenarios import L_TARGET
    lm = u["llm_metrics"]
    over = sum(a.status in (429, 503, 504, ST.CLIENT_TIMEOUT) for a in llm_att) / len(llm_att)
    exc = sum(max(0.0, (a.latency - L_TARGET) / L_TARGET) for a in llm_att) / len(llm_att)
    assert lm["overload_rate"] == pytest.approx(over) and lm["mean_latency_excess"] == pytest.approx(exc)
    assert any(a.status != 200 for a in llm_att)                       # excess includes non-200 attempts
    succ = sum(t["success"] for t in u["tasks"]) / len(u["tasks"])
    assert u["J_rq3"] == pytest.approx(succ - 0.5 * exc - 0.5 * over)
    lat200 = sorted(a.latency for a in llm_att if a.status == 200)
    assert lm["lat_p95"] == pytest.approx(float(np.percentile(lat200, 95)))
    assert lm["attempts_per_call"] == pytest.approx(len(llm_att) / len(u["llm_calls"]))


def test_infra_metrics_background_only_and_window_has_j(monkeypatch):    # D117, review M2
    u, got = captured_unit(monkeypatch, cond="C0")
    gw = got["gw"]
    bg = [a for a, r in zip(gw.attempts, gw.attempt_requests) if not is_llm(r)]
    inf = u["infra"]
    assert inf["traffic"] == "background" and inf["n_attempts"] == len(bg)
    assert inf["n_calls_requested"] == sum(1 for c in gw.calls if not is_llm(c.request)) + sum(
        1 for _, r in gw.open_calls.values() if not is_llm(r))
    w = inf["windows"]["llm_phase"]
    for k in ("call_success_rate", "mean_latency_excess_all_attempts", "throttle_rate", "J", "goodput"):
        assert w[k] is not None, k
    assert w["J"] == pytest.approx(w["call_success_rate"] - 0.5 * w["mean_latency_excess_all_attempts"]
                                   - 0.5 * w["throttle_rate"])
    lm = u["llm_metrics"]
    for k in ("attempts_per_call", "e2e_p95_censored", "lat_p95", "overload_rate", "slo_violation_rate"):
        assert k in lm, k


def test_shutdown_called_no_pending_and_end_at_last_finish(monkeypatch):
    u, got = captured_unit(monkeypatch)
    assert got["shutdown"] == 1 and got["pending"] == []
    assert u["t_end"] == pytest.approx(max(t["t_finish"] for t in u["tasks"]))
    u2, got2 = captured_unit(monkeypatch, t_max=40.0)
    assert u2["capped"] and got2["shutdown"] == 1 and got2["pending"] == [] and u2["t_end"] == pytest.approx(40.0)


def test_agent_exception_ends_the_run_at_once(monkeypatch):            # review M4
    state = {"raised_at": None, "after": 0, "n": 0}

    class Boom(ScriptedBackend):
        def generate(self, messages, kind, grammar, max_tokens):
            now = asyncio.get_event_loop().time()
            if state["raised_at"] is not None and now > state["raised_at"]:
                state["after"] += 1
            state["n"] += 1
            if state["n"] == 30:
                state["raised_at"] = now
                raise RuntimeError("backend down")
            return super().generate(messages, kind, grammar, max_tokens)
    shut = {"n": 0}
    orig = rq3.TaggedGateway.shutdown

    async def sd(self):
        shut["n"] += 1
        await orig(self)
    monkeypatch.setattr(rq3.TaggedGateway, "shutdown", sd)
    spec = sample_stream(5006, "dev", kind="rq3")
    with pytest.raises(RuntimeError, match="backend down"):
        run_rq3_unit(spec, "C2", "acc", Boom(TOK, "memory"), BagOfWordsEmbedder(), TOK)
    assert state["after"] == 0 and shut["n"] == 1


def test_capped_units_count_drift_errors_and_censor_completion():    # review L1, L2; D118
    u = unit("C2", "acc", t_max=120.0)
    assert u["capped"]
    assert any(t["t_finish"] is None and t["t_start"] is not None and t["t_start"] > WARMUP + 1 for t in u["tasks"])
    assert any(t["t_start"] is None for t in u["tasks"])
    by_task = {}
    for st in u["steps"]:
        by_task.setdefault(st["task"], []).append(st)
    for t in u["tasks"]:
        steps = by_task.get(t["task"], [])
        assert t["drift_errors"] == sum(1 for s in steps if s.get("truth", {}).get("kind") in ("drift",
                                                                                              "silent_drift"))
        assert t["completion_time"] is not None
        assert t["completion_censored"] == (t["t_finish"] is None)
        assert t["never_started"] == (t["t_start"] is None)
        if t["t_start"] is None:
            assert t["completion_time"] == pytest.approx(u["t_end"] - WARMUP) and not t["success"]
        elif t["t_finish"] is None:                                    # cut off: from its own start (re-review M2)
            assert t["completion_time"] == pytest.approx(u["t_end"] - t["t_start"])
        else:
            assert t["completion_time"] == pytest.approx(t["t_finish"] - t["t_start"])
    assert u["drift_errors"] == sum(t["drift_errors"] for t in u["tasks"])
    assert any(t["unfinished_at_t_max"] and t["api_calls"] > 0 for t in u["tasks"])
    # a step cancelled inside its API call keeps its call number (L2) and is counted as an API call
    for t in u["tasks"]:
        assert t["api_calls"] == sum(1 for s in by_task.get(t["task"], []) if "k" in s)


def test_429_503_before_semantics_and_silent_200_is_a_latency_sample():
    from test_s31_suite import DOC_ARGS, _suite
    spec, suite, srv = _semantic_env()

    async def go():
        srv._tokens = 0.0
        r1 = await srv.handle(LLMRequest(1000, 1, "get_api_docs", {"operation": "get_order"}, 0))
        srv._tokens = 10.0
        srv.inflight = 10 ** 6
        r2 = await srv.handle(LLMRequest(1000, 2, "get_api_docs", {"operation": "get_order"}, 0))
        srv.inflight = 0
        return r1, r2
    r1, r2 = run_virtual(go())
    assert r1.status == 429 and r2.status == 503
    assert suite.executed((1000, 1)) is None and suite.executed((1000, 2)) is None   # the suite never ran
    s = _suite(["pay_order:silent"])
    sc = build("CAP-steady", 32, method_for("acc"), 0)
    srv2 = SemanticServer(sc.server, 1, s)
    r = run_virtual(srv2.handle(LLMRequest(1000, 3, "pay_order", DOC_ARGS["pay_order"], 0)))
    assert r.status == 200 and r.payload.body["status"] == "error" and r.payload.truth.kind == "silent_drift"
    assert r.latency > sc.server.error_latency and srv2.counts[200] == 1 and srv2.counts["sem4xx"] == 0


def test_request_subclass_same_noise_and_backoff_as_plain_tuple():
    sc = build("CAP-steady", 32, method_for("acc"), 0)
    a = SemanticServer(sc.server, 11, None)
    b = SemanticServer(sc.server, 11, None)
    req = LLMRequest(1000, 5, "get_order", {}, 0)
    assert a._noise(req) == b._noise((1000, 5))
    g = TaggedGateway(a, ACCController(sc.acc), sc.retry, 3)
    for k in range(3):
        assert g._backoff(k, req) == g._backoff(k, (1000, 5))


def test_llm_request_copy_deepcopy_pickle():                         # review L4
    r = LLMRequest(1000, 7, "get_order", {"order_id": "O-1"}, 3)
    for c in (copy.copy(r), copy.deepcopy(r), pickle.loads(pickle.dumps(r))):
        assert isinstance(c, LLMRequest) and c == r and (c.op, c.args, c.task_index) == ("get_order",
                                                                                          {"order_id": "O-1"}, 3)


def test_idempotent_retry_in_flight_gets_stored_response():          # review M6
    spec, suite, srv = _semantic_env(base_latency=1.5, latency_sigma=0.0)
    inst = next(t for t in spec["tasks"] if t["template"] in ("T14", "T15"))
    cid = inst["refs"]["customer_id"]
    req_args = {"customer_id": cid, "subject": "x", "category": "billing", "priority": "low"}
    sc = build("CAP-steady", 32, method_for("acc"), 0)
    gw = TaggedGateway(srv, ACCController(sc.acc), RetryConfig(max_retries=2, backoff_base=0.1, backoff_cap=0.1,
                                                               timeout=1.0), 2)
    v = suite.views(inst["index"])["create_ticket"]

    async def go():
        task = asyncio.ensure_future(gw.call(LLMRequest(1000, 1, v.route, to_current(v, req_args),
                                                        inst["index"])))
        await asyncio.sleep(0.5)
        srv.cfg = dataclasses.replace(srv.cfg, base_latency=0.05)   # the retry is fast; the first is still running
        rec = await task
        await gw.shutdown()
        return rec
    rec = run_virtual(go())
    assert rec.status == 200 and rec.attempts == 2
    assert srv.max_inflight_seen == 2                                  # both admissions counted in flight
    assert len([i for k, i in suite.world.created if k == "tickets"]) == 1
    assert rec.payload.body == suite.executed((1000, 1)).body          # the stored response


def test_state_changing_hidden_execution_counted():
    spec, suite, srv = _semantic_env(base_latency=3.0, latency_sigma=0.0)
    inst = next(t for t in spec["tasks"] if t["template"] in ("T14", "T15"))
    req_args = {"customer_id": inst["refs"]["customer_id"], "subject": "x", "category": "billing",
                "priority": "low"}
    sc = build("CAP-steady", 32, method_for("acc"), 0)
    gw = TaggedGateway(srv, ACCController(sc.acc), RetryConfig(max_retries=1, backoff_base=0.1, backoff_cap=0.1,
                                                               timeout=1.0), 2)
    disp = GatewayDispatcher(gw, suite)
    v = suite.views(inst["index"])["create_ticket"]

    async def go():
        out = await disp.call(v.route, to_current(v, req_args), 1000, 1, inst["index"])
        await gw.shutdown()
        return out
    out = run_virtual(go())
    assert out.transient and disp.calls[0]["hidden_execution"] and disp.calls[0]["hidden_state_change"]


def test_repeat_after_transient_notice():
    u = unit("C2", "none", server_override={"base_latency": 1.5, "latency_sigma": 0.3})
    by = {}
    for s in u["steps"]:
        by.setdefault((s["agent"], s["task"]), []).append(s)
    n = 0
    for steps in by.values():
        for a, b in zip(steps, steps[1:]):
            if a.get("transient") and "action" in b:
                assert (b["action"], b["arg_keys"]) == (a["action"], a["arg_keys"])
                n += 1
    assert n > 10


def test_shared_store_timing_and_cross_agent_retrieval():           # S3.2a timing with 4 agents
    u = unit("C2", "acc")
    created = {e["id"]: e for e in u["note_events"] if e["event"] == "create"}
    cross = 0
    for st in u["steps"]:
        for nid in st.get("retrieved", []):
            assert created[nid]["t"] <= st["t_start"] + 1e-9         # visible only after its reflection sleep
            if created[nid]["at"][0] != st["agent"]:
                cross += 1
    assert cross > 0


def test_private_store_no_agent_retrieves_another_agents_note():
    u = unit("C2-private", "acc")
    created = [{e["id"]: e["at"][0] for e in evs if e["event"] == "create"} for evs in u["note_events"]]
    for st in u["steps"]:
        i = st["agent"] - 1000
        for nid in st.get("retrieved", []):
            assert created[i][nid] == st["agent"]
    assert all(set(c.values()) <= {1000 + i} for i, c in enumerate(created))


def test_background_draws_identical_across_cells(monkeypatch):
    draws = {}
    orig = rq3._scripted_agent

    class Rec:
        def __init__(self, rng, log):
            self.rng, self.log = rng, log

        def exponential(self, *a):
            x = self.rng.exponential(*a)
            self.log.append(x)
            return x

        def uniform(self, *a):
            x = self.rng.uniform(*a)
            self.log.append(x)
            return x

    def wrapped(gw, think, jitter, rng, agent_id, stop):
        return orig(gw, think, jitter, Rec(rng, draws.setdefault(cell, {}).setdefault(agent_id, [])), agent_id, stop)
    monkeypatch.setattr(rq3, "_scripted_agent", wrapped)
    for cell in (("C0", "acc"), ("C2", "none")):
        unit(*cell, seed=5007)
    a, b = draws[("C0", "acc")], draws[("C2", "none")]
    assert a.keys() == b.keys() and len(a) == 32
    for i in a:
        m = min(len(a[i]), len(b[i]))
        assert m > 5 and a[i][:m] == b[i][:m]


def test_stamp_and_configs_equal_build():                           # review M5, L11
    for win, w in (("acc", None), ("none", 128), ("fixed4", 4)):
        u = unit("C2", win, seed=5008)
        sc = build("CAP-steady", 32, method_for(win), 5008)
        st = u["stamp"]
        assert st["method"] == method_for(win) and st["acc_config"] == dataclasses.asdict(sc.acc)
        assert st["server_config"] == dataclasses.asdict(sc.server) and st["retry_config"] == dataclasses.asdict(sc.retry)
        assert {"action_max_tokens", "grammar_always"} <= set(st["params"]) and st["rq2_frozen_sha256"]
        if w is not None:
            assert sc.acc.frozen and sc.acc.w0 == w and u["window_trace"][-1][1] == w


def test_scripted_only_attempt_lists_identical(monkeypatch):           # review L11
    sc = build("CAP-steady", 32, method_for("acc"), 1)
    ref = sim.simulate(sc)
    monkeypatch.setattr(sim, "MockServer", lambda cfg, seed: SemanticServer(cfg, seed, None))
    monkeypatch.setattr(sim, "Gateway", TaggedGateway)                 # the RQ3 gateway too (re-review L12)
    got = sim.simulate(sc)
    assert isinstance(got, TaggedGateway) and len(got.attempt_requests) == len(got.attempts)
    assert [dataclasses.astuple(a) for a in ref.attempts] == [dataclasses.astuple(a) for a in got.attempts]
    assert [(c.t_request, c.t_done, c.status, c.attempts) for c in ref.calls] == \
        [(c.t_request, c.t_done, c.status, c.attempts) for c in got.calls]


def test_notes_created_counted():                                    # review L10
    u = unit("C2", "acc")
    assert u["notes_created"] == sum(1 for e in u["note_events"] if e["event"] == "create") > 0
    p = unit("C2-private", "acc")
    assert p["notes_created"] == sum(1 for evs in p["note_events"] for e in evs if e["event"] == "create")
    assert unit("C0", "acc")["notes_created"] == 0


def test_step_cancelled_inside_its_call_keeps_k():                   # review L2
    from eem_acc.stage3.agent.loop import Agent, LoopConfig
    from eem_acc.stage3.agent.memory import make_memory
    from eem_acc.stage3.apisuite.docs import documented_all

    class Hang:
        async def call(self, *a, **k):
            await asyncio.Event().wait()

    spec = sample_stream(5009, "dev", kind="rq3")
    ag = Agent(1000, ScriptedBackend(TOK, "memory"), TOK, make_memory("C2", BagOfWordsEmbedder(), TOK), Hang(),
               LoopConfig(condition="C2"), lambda ti: documented_all())
    ScriptedBackend.register_tasks(ag.backend, spec["tasks"])

    async def go():
        t = asyncio.ensure_future(ag.run_task(spec["tasks"][0]))
        await asyncio.sleep(120)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
    run_virtual(go())
    last = ag.live[-1]
    assert last.get("action") not in (None, "finish") and "status" not in last and last["k"] == 1


def test_scripted_agent_error_ends_the_run_at_once(monkeypatch):     # re-review L1
    orig = rq3._scripted_agent
    state = {"raised_at": None, "after": 0, "shutdown": 0}

    async def bad(gw, think, jitter, rng, agent_id, stop):
        if agent_id == 0:
            await asyncio.sleep(25.0)                                  # after the warm-up: LLM agents are working
            state["raised_at"] = asyncio.get_event_loop().time()
            raise RuntimeError("scripted agent failed")
        await orig(gw, think, jitter, rng, agent_id, stop)

    class Count(ScriptedBackend):
        def generate(self, messages, kind, grammar, max_tokens):
            if state["raised_at"] is not None and asyncio.get_event_loop().time() > state["raised_at"]:
                state["after"] += 1
            return super().generate(messages, kind, grammar, max_tokens)
    orig_sd = rq3.TaggedGateway.shutdown

    async def sd(self):
        state["shutdown"] += 1
        await orig_sd(self)
    monkeypatch.setattr(rq3, "_scripted_agent", bad)
    monkeypatch.setattr(rq3.TaggedGateway, "shutdown", sd)
    spec = sample_stream(5010, "dev", kind="rq3")
    with pytest.raises(RuntimeError, match="scripted agent failed"):
        run_rq3_unit(spec, "C2", "acc", Count(TOK, "memory"), BagOfWordsEmbedder(), TOK)
    assert state["raised_at"] == pytest.approx(25.0) and state["after"] == 0 and state["shutdown"] == 1
