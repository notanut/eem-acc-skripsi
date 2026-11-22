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
