"""S3.3b: T4 helpers (docs/s33b_spec.md r2) on CPU: the tiny qwen2 GGUF for llama paths, the scripted backend for runner
paths. The real T4 behaviour is checked in the notebook sessions."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eem_acc.stage3.agent import BagOfWordsEmbedder, ScriptedBackend, StubTokenizer
from eem_acc.stage3.llm import runner as R
from eem_acc.stage3.llm import t4
from eem_acc.stage3.llm.cache import CachedBackend, UnitCache
from eem_acc.stage3.llm.ledger import Ledger

REPO = Path(__file__).resolve().parents[1]
TOK = StubTokenizer()
TINY = Path.home() / ".cache" / "eem-acc" / "tiny-qwen2.gguf"


class Killed(BaseException):
    pass


def kill():
    raise Killed()


def header():
    h = R.make_header({"backend": "scripted"}, "bow-sha1-256", {"python": "3", "numpy": "2", "gpu": "cpu"}, 0, "fp0")
    return {**h, "code_commit": "c" * 40, "code_dirty": False}


def runner(root, backend, hdr=None):
    ctx = R.RunContext(backend, BagOfWordsEmbedder(), TOK, hdr or header())
    return R.Runner(Path(root), ctx, Ledger(root, "s", "rt1"), log=lambda m: None)


def payload(root, u):
    return R.split_result(json.loads((Path(root) / "units" / u.phase / f"{u.unit_id}.json").read_text()))


# ----------------------------------------------------------------------------------------------- pure helpers
def test_smoke_plan_order_and_ids():
    ref, units = t4.smoke_plan()
    assert [u.condition for u in units] == ["C2", "C0", "C2"] and units[2].rq == "rq3" and units[2].window == "acc"
    assert ref[0].phase == "smoke-ref" and (ref[0].rq, ref[0].seed, ref[0].condition, ref[0].n_tasks) == \
        (units[0].rq, units[0].seed, units[0].condition, units[0].n_tasks)
    assert len({u.unit_id for u in ref + units}) == 4


def test_model_digest_and_verify_parts(tmp_path):
    a, b = tmp_path / "p1.gguf", tmp_path / "p2.gguf"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    exp = {"p1.gguf": t4.sha256_file(a), "p2.gguf": t4.sha256_file(b)}
    assert t4.verify_parts([a, b], exp)["equal_s30"]
    assert not t4.verify_parts([a], exp)["equal_s30"]
    assert t4.model_digest(exp) == t4.model_digest(dict(reversed(list(exp.items()))))


def test_timing_check_flag():
    recs = [{"n_in": 1000, "n_out": 50, "grammar": None, "seconds": 1.0}]
    from eem_acc.stage3.agent.backend import Gen, t_llm
    pred = t_llm(Gen("", 1000, 50, "stop", None))
    assert not t4.timing_check([{**recs[0], "seconds": pred}])["flag"]
    assert t4.timing_check([{**recs[0], "seconds": pred * 1.3}])["flag"]


# ------------------------------------------------------------------------------------- kills, drive, comparison
U_REF, U_C2, U_C0, U_RQ3 = (*t4.smoke_plan()[0], *t4.smoke_plan()[1])


def _plans(root):
    R.write_plan(root, "smoke-ref", [U_REF], 1)
    R.write_plan(root, "smoke", [U_C2, U_C0], 1)


def test_kill_a_then_resume_in_new_runtime_equals_reference(tmp_path):
    _plans(tmp_path)
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("smoke-ref")
    r1 = runner(tmp_path, None)
    assert t4.arm_kill_a(r1, U_C2)
    r1.ctx.backend = t4.KillAfter(ScriptedBackend(TOK, "memory"), t4.KILL_A_AFTER, kill)
    with pytest.raises(Killed):
        r1.run_phase("smoke")
    assert not t4.arm_kill_a(r1, U_C2)                           # never armed twice (cache exists)
    ref_cache = r1.cache_path(U_REF)
    st = t4.drive_state(r1.cache_path(U_C2))
    assert st["records_present"] == t4.records_at_kill(ref_cache, t4.KILL_A_AFTER) and not st["last_line_torn"]
    r2 = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r2.ledger = Ledger(tmp_path, "s", "rt2")
    r2.run_phase("smoke")
    p, run = payload(tmp_path, U_C2)
    assert run["resumed"]
    pref, _ = payload(tmp_path, U_REF)
    cmp = t4.compare_units(r2.cache_path(U_C2), ref_cache, p, pref)
    assert cmp["verdict"] == "pass" and cmp["llm_equal"] and cmp["embeddings_equal"] and cmp["payload_equal"]


def test_kill_b_result_written_then_counted_done(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    assert t4.arm_kill_b(r, U_C0, kill)
    with pytest.raises(Killed):
        r.run_phase("smoke")
    st = t4.drive_state(r.cache_path(U_C0), r.result_path(U_C0))
    assert st["result_present"] and not st["cache_short"] and st["records_present"] == st["result_cache_records"]
    assert not t4.arm_kill_b(r, U_C0, kill)                      # result exists: not armed again
    r2 = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    out = r2.run_phase("smoke")
    assert r2.is_done(U_C0) and out["done"] == 2                  # counted done, no rerun of C0


def test_drive_state_reads_without_writing(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("smoke-ref")
    cp = r.cache_path(U_REF)
    raw = cp.read_bytes()
    cp.write_bytes(raw[:-30])
    before = cp.read_bytes()
    st = t4.drive_state(cp)
    assert st["last_line_torn"] and cp.read_bytes() == before


def test_compare_units_embedding_difference_is_reported_not_failed(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("smoke-ref")
    r.run_phase("smoke")
    a, b = r.cache_path(U_C2), r.cache_path(U_REF)
    lines = a.read_text().split("\n")
    from eem_acc.stage3.llm.cache import crc
    from eem_acc.stage3.llm.fsio import canon
    j = next(i for i, ln in enumerate(lines) if ln and json.loads(ln).get("kind") == "embed")
    rec = json.loads(lines[j])
    rec["vector"] = [v + 1e-7 for v in rec["vector"]]
    rec["crc"] = crc(rec)
    lines[j] = canon(rec)
    a.write_text("\n".join(lines))
    pa, pb = payload(tmp_path, U_C2)[0], payload(tmp_path, U_REF)[0]
    cmp = t4.compare_units(a, b, {**pa, "x": 1}, pb)
    assert cmp["llm_equal"] and not cmp["embeddings_equal"] and cmp["first_differing_vector"] is not None
    assert cmp["verdict"] == "pass"                               # payload differs, but embeddings differ too
    cmp2 = t4.compare_units(b, b, {**pb, "x": 1}, pb)
    assert cmp2["verdict"] == "fail"                              # equal embeddings, different payload


def test_claims_of_dead_runtime_broken_by_lease_holder(tmp_path):
    old = R.Claims(tmp_path, "rtOld")
    old.claim(U_C2.unit_id)
    new = R.Claims(tmp_path, "rtNew")
    mine = R.Claims(tmp_path, "rtNew")
    mine.claim("other")
    with pytest.raises(R.LeaseHeld):                              # without holding the lease: refused
        new.break_other_runtimes(R.Lease(tmp_path, "rtNew"))
    lease = R.Lease(tmp_path, "rtNew")
    lease.acquire()
    assert new.break_other_runtimes(lease) == [U_C2.unit_id]
    assert (tmp_path / "claims" / "other.claim").exists() and new.claim(U_C2.unit_id)


def test_smoke_stats_and_append_latency(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("smoke")
    results = [payload(tmp_path, u)[0] for u in (U_C2, U_C0)]
    s = t4.smoke_stats(results, lambda u: r.cache_path(R.Unit.from_json(u)))
    assert set(s) == {"smoke-rq1-C2", "smoke-rq1-C0"}
    assert all(v["llm_calls"] > 0 and v["calls_per_task"] > 0 and v["embed_records"] >= 0 for v in s.values())
    lat = t4.append_latency(tmp_path / "lat.jsonl", n=5)
    assert lat["n"] == 5 and not (tmp_path / "lat.jsonl").exists()


def test_benchmark_workload_dedup_stratified_deterministic(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("smoke-ref")
    r.run_phase("smoke")
    caches = [r.cache_path(u) for u in (U_REF, U_C2, U_C0)]
    w = t4.benchmark_workload(caches, n=30)
    assert w == t4.benchmark_workload(caches, n=30) and 0 < len(w) <= 30
    assert len({x["prompt"] for x in w}) == len(w)               # ref and C2 share prompts: deduplicated
    kinds = {(x["kind"], x["grammar"]) for x in w}
    assert ("reflection", None) in kinds and ("action", None) in kinds


# ----------------------------------------------------------------------------------------------- llama paths
llama_cpp = pytest.importorskip("llama_cpp")
needs_tiny = pytest.mark.skipif(not TINY.exists(), reason="tiny GGUF missing: run tools/fetch_test_models.py")


@pytest.fixture(scope="module")
def llm():
    return llama_cpp.Llama(model_path=str(TINY), n_ctx=8192, n_batch=512, seed=0, verbose=False)


def backend(llm, snapshot=False):
    return t4.LlamaCppBackend(llm, "d" * 64, "w" * 64, snapshot=snapshot)


def chat_path(llm):
    from eem_acc.stage3 import feasibility as F

    def run(messages, grammar):
        llm.reset()
        return F.make_generate(llm, grammar=grammar, reset=True, max_tokens=t4.S30_MAX_TOKENS)(messages)
    return run


@needs_tiny
def test_s30_equality_and_interpretation_rule(llm, monkeypatch):
    from eem_acc.stage3 import feasibility as F
    prompts = F.sample_prompts(20, seed=0)[:4]
    cp = chat_path(llm)
    recs = {"free": [], "grammar": []}
    for i, m in enumerate(prompts):
        for kind in ("free", "grammar"):
            o = cp(m, kind == "grammar")                          # S3.0's real format: "prompt" is always 0
            recs[kind].append({"prompt": 0, "text": o["text"][:500], "n_in": o["n_in"], "n_out": o["n_out"],
                               "finish_reason": o["finish_reason"]})
    with pytest.raises(ValueError, match="records"):
        t4.s30_equality(backend(llm), {"free": recs["free"][:3], "grammar": recs["grammar"]}, prompts, cp)
    b = backend(llm)
    res = t4.s30_equality(b, recs, prompts, cp)
    assert res["all_equal"] and res["compared"] == 8 and len(res["records"]) == 8
    assert res["coverage"]["n_in_max"] == max(r["n_in"] for r in recs["free"])
    bad = json.loads(json.dumps(recs))
    bad["free"][1]["text"] = "something S3.0 never said"
    res2 = t4.s30_equality(b, bad, prompts, cp)
    assert not res2["all_equal"] and res2["differences"][0]["classification"].startswith("platform change")
    orig = b.generate
    monkeypatch.setattr(b, "generate", lambda *a, **k: (lambda g: type(g)("!" + g.text, g.n_in, g.n_out,
                                                                           g.finish_reason, g.grammar))(orig(*a, **k)))
    res3 = t4.s30_equality(b, recs, prompts, cp)
    assert res3["differences"][0]["classification"].startswith("code bug")
    # D145: the first 2 prompts only = 2 free + 2 grammar records, S3.0's order (free, grammar, grammar, free)
    monkeypatch.setattr(b, "generate", orig)
    res4 = t4.s30_equality(b, recs, prompts, cp, limit=2)
    assert res4["all_equal"] and res4["compared"] == 4
    assert [(r["kind"], r["prompt"]) for r in res4["records"]] == [("free", 0), ("grammar", 0), ("grammar", 1), ("free", 1)]
    bad2 = json.loads(json.dumps(recs))
    bad2["free"][3]["text"] = "beyond the limit"
    assert t4.s30_equality(b, bad2, prompts, cp, limit=2)["all_equal"]


@needs_tiny
def test_d47_check_on_tiny(llm):
    from eem_acc.stage3.llm.canary import load_reference_prompts
    from eem_acc.stage3.llm.llama_backend import LlamaTokenizer
    b = backend(llm)
    tok = LlamaTokenizer(b)
    ps = sorted(load_reference_prompts()[0], key=lambda p: -p["n_tokens"])[:2]
    ps = [{**p, "max_tokens": 16} for p in ps]
    fresh = lambda: backend(llama_cpp.Llama(model_path=str(TINY), n_ctx=8192, n_batch=512, seed=0, verbose=False))
    res = t4.d47_check(b, fresh, ps, tok)
    assert res["all_equal"] and all(r["unrelated_tokens"] >= r["n_tokens"] + 320 for r in res["prompts"])


@needs_tiny
def test_canary_counts_and_template_check(llm):
    from eem_acc.stage3.llm.canary import load_reference_prompts
    from eem_acc.stage3.llm.llama_backend import LlamaTokenizer
    c = t4.canary_counts(LlamaTokenizer(backend(llm)), load_reference_prompts()[0])
    assert c["equal"]
    tc = t4.template_check(llm)
    assert tc["equals_s30"] and tc["text"].startswith("{%- if tools %}")         # the real template (CPU fixture)
    lp = t4.load_params_report(llm, {"n_ctx": 8192, "n_batch": 512})
    assert lp["n_ctx_equals_s30"] and lp["n_batch_equals_s30"] and lp["threads"]["n_threads"]


def _llama_cache(llm, path):
    b = backend(llm)
    cache = UnitCache(path, {}, "w")
    cb = CachedBackend(b, cache)
    msgs = [
        [{"role": "system", "content": "You are an API agent."}, {"role": "user", "content": "Task one."}],
        [{"role": "system", "content": "You are an API agent."}, {"role": "user", "content": "Task two, longer text."}],
        [{"role": "user", "content": "Reflect on the error."}],
    ]
    for m in msgs:
        cb.generate(m, "action" if m[0]["role"] == "system" else "reflection", None, 12)
    cb.generate(msgs[0], "action", "envelope_cap300", 12)
    return path


@needs_tiny
def test_benchmark_ab_on_tiny(llm, tmp_path):
    cp = _llama_cache(llm, tmp_path / "c.jsonl")
    w = t4.benchmark_workload([cp], n=10)
    assert len(w) == 4
    res = t4.benchmark_ab(backend(llm, snapshot=True), w, telemetry=lambda: {})
    assert res["A_equals_recorded"] == 4 and res["B_repeatable"]
    assert res["B_used_snapshot"] == 3                            # the reflection ran on the fresh path
    bad = [{**w[0], "n_in": w[0]["n_in"] + 1}]
    with pytest.raises(AssertionError, match="n_in"):
        t4.TokenRunner(backend(llm)).run(bad[0], False)


@needs_tiny
def test_snapshot_ram_probe_on_tiny(llm):
    b = backend(llm, snapshot=True)
    res = t4.snapshot_ram_probe(b, ["Rules W1.", "Rules W2, worded otherwise.", "Rules W3, a third wording."])
    assert res["states"] == 3 and res["first_eval_at_prefix"] and len(res["rss_after_each_state"]) == 3
    t4.drop_snapshot_states(b)
    assert not b._states


@needs_tiny
def test_unit_worker_two_processes_on_tiny(tmp_path, llm):
    from eem_acc.stage3.llm.llama_backend import LlamaTokenizer
    units = [R.Unit("bench2", "rq1", "dev", "rq1", 1000, c, n_tasks=2) for c in ("C2", "C0")]
    R.write_plan(tmp_path, "bench2", units, 1)
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"tiny": "0" * 64}))
    override = json.dumps({"code_commit": "c" * 40, "code_dirty": False})
    procs = []
    for k in range(2):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "eem_acc.stage3.llm.unit_worker", "--root", str(tmp_path), "--phase", "bench2",
             "--shard", f"{k}/2", "--model", str(TINY), "--parts", str(parts), "--wheel-sha", "w" * 64,
             "--session", "t", "--out", str(tmp_path / f"w{k}.json"), "--n-gpu-layers", "0", "--scripted",
             "--header-override", override],
            cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    for p in procs:
        out, err = p.communicate(timeout=900)
        assert p.returncode == 0, err.decode()[-2000:]
    outs = [json.loads((tmp_path / f"w{k}.json").read_text()) for k in range(2)]
    assert [o["run"]["ran"] for o in outs] == [1, 1] and outs[0]["canary_fingerprint"] == outs[1]["canary_fingerprint"]
    # the same units in one process give the same payloads (without `unit`)
    ref_root = tmp_path / "ref"
    ref_units = [R.Unit("ref", "rq1", "dev", "rq1", 1000, c, n_tasks=2) for c in ("C2", "C0")]
    R.write_plan(ref_root, "ref", ref_units, 1)
    w0 = json.loads(UnitCache(tmp_path / "cache" / f"{units[0].unit_id}.jsonl", {}, "r", readonly=True).path
                    .read_text().split("\n")[0])
    hdr = {k: v for k, v in w0.items() if k not in ("type", "format", "writer", "unit", "unit_id")}
    same = t4.LlamaCppBackend(llm, t4.model_digest({"tiny": "0" * 64}), "w" * 64)    # the workers' model identity
    rr = R.Runner(ref_root, R.RunContext(ScriptedBackend(LlamaTokenizer(same), "memory"), BagOfWordsEmbedder(),
                                         LlamaTokenizer(same), hdr), Ledger(ref_root, "r", "rt"), log=lambda m: None)
    rr.run_phase("ref")
    strip = lambda p: {k: v for k, v in p.items() if k != "unit"}
    for u, v in zip(units, ref_units):
        pa, pb = strip(payload(tmp_path, u)[0]), strip(payload(ref_root, v)[0])
        diff = {k: (pa["stamp"].get(k), pb["stamp"].get(k)) for k in set(pa["stamp"]) | set(pb["stamp"])
                if pa["stamp"].get(k) != pb["stamp"].get(k)}
        assert pa == pb, diff



# ----------------------------------------------------------------------------------------------- code review fixes
class Saved(t4.Sections):
    pass


def test_kill_after_never_retried_when_kill_returns(tmp_path):                           # review H2
    _plans(tmp_path)
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("smoke-ref")
    calls = []
    r = runner(tmp_path, t4.KillAfter(ScriptedBackend(TOK, "memory"), 3, lambda: calls.append(1)))
    with pytest.raises(t4.KillFailed):
        r.run_phase("smoke")
    assert calls == [1]                                           # called once; nothing retried afterwards
    assert not (tmp_path / "failures").exists() and not r.is_done(U_C0)
    assert issubclass(t4.KillFailed, BaseException) and not issubclass(t4.KillFailed, Exception)


def test_make_kill_saves_request_first(tmp_path):
    S = t4.Sections(tmp_path / "s.json")
    seen = []
    k = t4.make_kill(S, "kill_a", "rtX", kill=lambda: seen.append(json.loads((tmp_path / "s.json").read_text())))
    k()
    assert seen[0]["kill_a_requested"]["runtime"] == "rtX"


def test_arming_recorded_once_and_loss_recorded_d141(tmp_path):
    _plans(tmp_path)
    S = t4.Sections(tmp_path / "s.json")
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    assert t4.arm_kill_a(r, U_C2, S, "rt1") and S.data["kill_a_armed"]["runtime"] == "rt1"
    assert not t4.arm_kill_a(r, U_C2, S, "rt2")                 # armed before, cache missing -> lost, not re-armed
    assert S.data["kill_a_lost"]["runtime"] == "rt2"
    assert t4.arm_kill_b(r, U_C0, kill, S, "rt2") and r.after_result is not None
    assert not t4.arm_kill_b(r, U_C0, kill, S, "rt3") and S.done("kill_b_lost") and r.after_result is None


def test_token_runner_timing_includes_state_load_and_kwargs(tmp_path):                   # review H3 + survivors
    class FakeLlm:
        def __init__(self):
            self.calls, self.n_tokens = [], 0

        def token_eos(self):
            return 2

        def tokenize(self, b, add_bos, special):
            return [int(x) for x in b.decode().split()]

        def reset(self):
            self.calls.append("reset")

        def create_completion(self, **kw):
            self.calls.append(("complete", kw["stop"], kw["stopping_criteria"], kw["prompt"]))
            return {"choices": [{"text": "t"}], "usage": {"completion_tokens": 1}}

    class FakeBackend:
        def __init__(self):
            import time as _t
            self.llm, self._states, self.loaded = FakeLlm(), {}, []
            self._t = _t

        def formatter(self, messages):
            return type("R", (), {"stop": ["<|im_end|>"], "stopping_criteria": "CRIT"})()

        def grammar(self, gid):
            return None, None

        def _load_prefix_state(self, prefix):
            self.loaded.append(list(prefix))
            self._states[tuple(prefix)] = 1
            self._t.sleep(0.05)                                   # a costly state load must be timed
    b = FakeBackend()
    tr = t4.TokenRunner(b)
    item = {"prompt": "5 6 2 7 2 8", "kind": "action", "grammar": None, "max_tokens": 4, "n_in": 6}
    res = tr.run(item, snapshot=True)
    assert b.loaded == [[5, 6, 2]]                               # prefix at the FIRST eos
    assert res["seconds"] >= 0.05 and res["created_state"]
    assert b.llm.calls[-1][1:3] == (["<|im_end|>"], "CRIT")      # formatter's stop and stopping criteria
    tr.run({**item, "kind": "reflection"}, snapshot=True)
    assert b.llm.calls[-2] == "reset" and len(b.loaded) == 1       # reflections: fresh path


def test_workload_trims_strata_and_rejects_empty(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("smoke")
    caches = [r.cache_path(u) for u in (U_C2, U_C0)]
    for n in (2, 3, 5):
        w = t4.benchmark_workload(caches, n=n)
        kinds = {x["kind"] for x in w}
        assert len(w) <= max(n, 2) and "reflection" in kinds and "action" in kinds   # no stratum dropped
    import random as _r
    full = t4.benchmark_workload(caches, n=1000)
    by_kind = {k: sum(1 for x in full if x["kind"] == k) for k in ("action", "reflection")}
    w = t4.benchmark_workload(caches, n=10)
    share = sum(1 for x in w if x["kind"] == "reflection") / len(w)
    assert abs(share - by_kind["reflection"] / sum(by_kind.values())) < 0.25            # stratified, not uniform
    empty = tmp_path / "empty.jsonl"
    UnitCache(empty, {}, "w")
    with pytest.raises(ValueError, match="no recorded"):
        t4.benchmark_workload([empty])


def test_first_full_wall_skips_repairs(tmp_path):
    led = Ledger(tmp_path, "s", "rt")
    led.event("unit", unit="u1", wall_seconds=5.0, calls_fresh=3, calls_replayed=7)
    led.event("unit", unit="u1", wall_seconds=40.0, calls_fresh=10, calls_replayed=0)
    assert t4.first_full_wall(tmp_path, "u1")[0] == 40.0 and t4.first_full_wall(tmp_path, "u2")[0] is None


def test_drive_state_crc_and_retrieval_equal(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("smoke-ref")
    cp = r.cache_path(U_REF)
    n = t4.drive_state(cp)["records_present"]
    lines = cp.read_text().split("\n")
    rec = json.loads(lines[-2])
    rec["text"] = "changed without a new crc"
    lines[-2] = json.dumps(rec)
    cp.write_text("\n".join(lines))
    st = t4.drive_state(cp)
    assert st["records_present"] == n - 1 and st["last_line_torn"]
    pa = payload(tmp_path, U_REF)[0]
    pb = json.loads(json.dumps(pa))
    step = next(x for x in pb["steps"] if x.get("retrieved"))
    step["retrieved"] = step["retrieved"][::-1] + [999]
    assert not t4.compare_units(cp, cp, pa, pb)["retrieval_equal"]


def test_budget_uses_the_first_benchmark_start(tmp_path):
    t = [0.0]
    a = Ledger(tmp_path, "s", "rtA", clock=lambda: t[0])
    a.start(None, vm_boot=0.0)
    t[0] = 1000.0
    a.event("benchmark_start")
    t[0] = 2000.0
    a.event("benchmark_start")                                    # a second one must not move the start
    t[0] = 4600.0
    a.heartbeat()
    from eem_acc.stage3.llm.ledger import summarise
    assert summarise(tmp_path)["gpu_hours"] == pytest.approx(1.0, abs=1e-3)


def test_smoke_stats_new_fields(tmp_path):
    _plans(tmp_path)
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("smoke")
    results = [{**payload(tmp_path, u)[0], "run": payload(tmp_path, u)[1]} for u in (U_C2, U_C0)]
    fails = [{"unit": U_C2.unit_id, "error": "ValueError: message content adds 1 special (control) tokens"}]
    s = t4.smoke_stats(results, lambda u: r.cache_path(R.Unit.from_json(u)), fails)
    assert s["smoke-rq1-C2"]["control_token_firings"] == 1 and s["smoke-rq1-C0"]["control_token_firings"] == 0
    assert all({"format_errors", "drift_errors", "transient_notices", "run"} <= set(v) for v in s.values())
    assert s["smoke-rq1-C2"]["run"]["cache_bytes"] > 0


@needs_tiny
def test_s30_order_and_warmup_and_d47_call_sequence(llm):
    from eem_acc.stage3 import feasibility as F
    b = backend(llm)
    seq = []
    orig = b.generate

    def spy(m, kind, gid, mt):
        seq.append((F.sample_prompts(20, seed=0).index(m) if m in F.sample_prompts(20, seed=0) else -1, gid))
        return orig(m, kind, gid, mt)
    b.generate = spy
    prompts = F.sample_prompts(20, seed=0)[:3]
    recs = {k: [{"prompt": 0, "text": "", "n_in": 0, "n_out": 0, "finish_reason": "stop"}] * 3 for k in ("free", "grammar")}
    t4.s30_equality(b, recs, prompts)
    assert seq[:2] == [(0, None), (0, "envelope_cap300")]         # S3.0's warm-up
    assert seq[2:] == [(0, None), (0, "envelope_cap300"), (1, "envelope_cap300"), (1, None), (2, None),
                       (2, "envelope_cap300")]                    # alternating by i % 2
    from eem_acc.stage3.llm.canary import load_reference_prompts
    from eem_acc.stage3.llm.llama_backend import LlamaTokenizer
    tok = LlamaTokenizer(orig.__self__)
    calls = []
    class Spy:
        def __init__(self, inner):
            self.inner = inner
            self.llm = inner.llm
        def generate(self, m, kind, gid, mt):
            calls.append((tok.count_messages(m), gid))
            return self.inner.generate(m, kind, gid, mt)
    p = sorted(load_reference_prompts()[0], key=lambda x: -x["n_tokens"])[1]
    p = {**p, "max_tokens": 8}
    fresh = lambda: backend(llama_cpp.Llama(model_path=str(TINY), n_ctx=8192, n_batch=512, seed=0, verbose=False))
    t4.d47_check(Spy(orig.__self__), fresh, [p], tok)
    assert calls[0][0] >= p["n_tokens"] + 320 and calls[2][1] == "envelope_cap300" and len(calls) == 4


@needs_tiny
def test_t4session_wiring_and_worker_epoch_identity(tmp_path, llm):
    old = R.Claims(tmp_path, "dead-runtime")
    old.claim("some-unit")
    ses = t4.T4Session(tmp_path, llm, {"tiny": "0" * 64}, "w" * 64, BagOfWordsEmbedder(), "s", 1.0,
                       heartbeat_seconds=60, log=lambda m: None)
    try:
        assert ses.epoch == 0 and ses.status == "new" and ses.broken_claims == ["some-unit"]
        assert ses.counts["equal"] and ses.header["platform_fingerprint"] == ses.fingerprint
        assert ses.runner.claims is ses.claims and ses.runner.session_heartbeat is ses.heartbeat
        assert json.loads((tmp_path / "lease.json").read_text())["runtime"] == ses.runtime
        hdr_parent = dict(ses.header)
    finally:
        ses.close()
    # a worker on the same platform finds epoch 0 (registered by the parent) and builds the same header
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"tiny": "0" * 64}))
    units = [R.Unit("bench2", "rq1", "dev", "rq1", 1000, "C2", n_tasks=2)]
    R.write_plan(tmp_path, "bench2", units, 1)
    r = subprocess.run([sys.executable, "-m", "eem_acc.stage3.llm.unit_worker", "--root", str(tmp_path), "--phase",
                        "bench2", "--shard", "0/1", "--model", str(TINY), "--parts", str(parts), "--wheel-sha", "w" * 64,
                        "--session", "t", "--out", str(tmp_path / "w.json"), "--n-gpu-layers", "0", "--scripted"],
                       cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads((tmp_path / "w.json").read_text())
    assert out["canary_epoch"] == 0 and out["canary_fingerprint"] == hdr_parent["platform_fingerprint"]
    from eem_acc.stage3.llm.cache import read_header
    wh = read_header(tmp_path / "cache" / f"{units[0].unit_id}.jsonl")
    for k in ("platform_epoch", "platform_fingerprint", "embedder", "code_commit", "prompt_sha256", "model_digest"):
        if k in hdr_parent:
            assert wh.get(k) == hdr_parent[k], k
    assert not list((tmp_path / "claims").glob("*.claim"))        # the worker claimed and released its unit


# ------------------------------------------------------------------------------- session-1 evidence and pins
def test_t4_pins_file():
    import json
    pins = json.loads((Path(__file__).resolve().parents[1] / "configs" / "t4_pins.json").read_text())
    assert pins["minilm_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert "libraries" not in pins                       # added only after session 2 (D142)


def test_session1_evidence_matches_the_template_fixture():
    import hashlib
    import json
    root = Path(__file__).resolve().parents[1]
    ev = json.loads((root / "docs" / "evidence" / "s33b_session1_2026-11-23.json").read_text())
    fixture = (root / "tests" / "fixtures" / "qwen25_chat_template.jinja").read_bytes()
    assert ev["template"]["text"].encode() == fixture and ev["template"]["equals_s30"]
    assert hashlib.sha256(fixture).hexdigest() == t4.S30_TEMPLATE_SHA
    assert ev["versions"]["minilm_revision"] == json.loads((root / "configs" / "t4_pins.json").read_text())["minilm_revision"]
