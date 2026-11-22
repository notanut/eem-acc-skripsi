"""S3.3a: unit runner, resume, completeness, failure policy, plans, lease, ledger, verify_replay, audit (spec 6-8)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eem_acc.stage3.agent import BagOfWordsEmbedder, ScriptedBackend, StubTokenizer
from eem_acc.stage3.llm import runner as R
from eem_acc.stage3.llm.audit import audit_scrub
from eem_acc.stage3.llm.cache import ReplayDivergence, UnitCache, content_hash_of, crc
from eem_acc.stage3.llm.canary import Session
from eem_acc.stage3.llm.fsio import canon, read_jsonl
from eem_acc.stage3.llm.ledger import Ledger, summarise

REPO = Path(__file__).resolve().parents[1]
TOK = StubTokenizer()


class Crash(BaseException):
    """Like a Colab interrupt: not caught by the runner's failure policy."""


class CrashAfter(ScriptedBackend):
    def __init__(self, n, *a, **k):
        super().__init__(*a, **k)
        self.n, self.k = n, 0

    def generate(self, messages, kind, grammar, max_tokens):
        self.k += 1
        if self.k == self.n:
            raise Crash()
        return super().generate(messages, kind, grammar, max_tokens)


class Suffix(ScriptedBackend):
    """Same id, different outputs: a platform whose generations changed."""

    def generate(self, messages, kind, grammar, max_tokens):
        g = super().generate(messages, kind, grammar, max_tokens)
        return type(g)(g.text + " ", g.n_in, g.n_out, g.finish_reason, g.grammar)


def header():
    return R.make_header({"backend": "scripted"}, "bow-sha1-256", {"python": "3", "numpy": "2", "gpu": "cpu"}, 0)


def runner(root, backend, session=None, rerun=None, shard=(0, 1), hdr=None):
    ctx = R.RunContext(backend, BagOfWordsEmbedder(), TOK, hdr or header(), session, rerun)
    return R.Runner(Path(root), ctx, Ledger(root, "s1", "rt1", str(shard[0])), shard=shard, log=lambda m: None)


U1 = R.Unit("pilot", "rq1", "dev", "rq1", 1000, "C2", n_tasks=6)
U3 = R.Unit("pilot", "rq3", "dev", "rq3", 5000, "C2", "acc")


def plan(root, units, **kw):
    return R.write_plan(root, "pilot", units, 1, **kw)


def strategy(u):
    return "docs" if u.condition == "C0" else "memory"


def payload(root, u):
    return R.split_result(json.loads((Path(root) / "units" / u.phase / f"{u.unit_id}.json").read_text()))


@pytest.mark.parametrize("u", [U1, U3], ids=["rq1", "rq3"])
def test_interrupted_and_resumed_equals_uninterrupted(tmp_path, u):
    a, b = tmp_path / "a", tmp_path / "b"
    plan(a, [u])
    plan(b, [u])
    runner(a, ScriptedBackend(TOK, strategy(u))).run_phase("pilot")
    rb = runner(b, CrashAfter(9, TOK, strategy(u)))
    with pytest.raises(Crash):
        rb.run_phase("pilot")
    assert not rb.is_done(u) and (b / "cache" / f"{u.unit_id}.jsonl").exists()
    runner(b, ScriptedBackend(TOK, strategy(u))).run_phase("pilot")
    pa, ra = payload(a, u)
    pb, rbm = payload(b, u)
    assert pa == pb and ra["payload_sha256"] == rbm["payload_sha256"]
    assert rbm["resumed"] and rbm["calls_replayed"] >= 8 and not ra["resumed"]
    assert ra["cache_content_hash"] == rbm["cache_content_hash"]
    assert any(r.get("kind") == "embed" for r in UnitCache(b / "cache" / f"{u.unit_id}.jsonl", {}, "v",
                                                           readonly=True).records)


def test_payload_independent_of_hashseed(tmp_path):
    code = ("import sys,json;sys.path[:0]=['src','tests'];from pathlib import Path;"
            "import test_s33_runner as T;r=Path(sys.argv[1]);T.plan(r,[T.U1]);"
            "T.runner(r,T.ScriptedBackend(T.TOK,'memory')).run_phase('pilot');"
            "print(T.payload(r,T.U1)[1]['payload_sha256'])")
    outs = set()
    for hs in ("0", "77"):
        root = tmp_path / f"h{hs}"
        r = subprocess.run([sys.executable, "-c", code, str(root)], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=hs), check=True)
        outs.add(r.stdout.strip().splitlines()[-1])
    assert len(outs) == 1


def test_done_requires_complete_cache_and_repairs_lost_tail(tmp_path):
    plan(tmp_path, [U1])
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("pilot")
    sha = payload(tmp_path, U1)[1]["payload_sha256"]
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().splitlines()
    cp.write_text("\n".join(lines[:-6]) + "\n")                  # Drive lost the cache tail after the result
    assert not r.is_done(U1)
    r.run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert run["payload_sha256"] == sha and run["resumed"] and r.is_done(U1)


def test_result_written_but_manifest_missing_counts_as_done(tmp_path):
    plan(tmp_path, [U1])
    r = runner(tmp_path, ScriptedBackend(TOK, "memory"))
    r.run_phase("pilot")
    for f in (tmp_path / "manifest").rglob("*.jsonl"):
        f.unlink()
    assert r.is_done(U1)
    assert r.run_phase("pilot")["ran"] == 0


def test_stamp_mismatch_archives_and_restarts(tmp_path):
    plan(tmp_path, [U1])
    with pytest.raises(Crash):
        runner(tmp_path, CrashAfter(9, TOK, "memory")).run_phase("pilot")
    h = {**header(), "code_commit": "other"}
    runner(tmp_path, ScriptedBackend(TOK, "memory"), hdr=h).run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert not run["resumed"] and run["restarts"] and run["restarts"][0].startswith("stamp")
    idx = read_jsonl(tmp_path / "cache" / "archive" / "index.jsonl")
    assert idx[0]["reason"] == "stamp" and "code_commit" in idx[0]["fields"]


def test_test_phase_refuses_other_commit(tmp_path):
    R.write_plan(tmp_path, "test", [R.Unit("test", "rq1", "test", "rq1", 2000, "C2", n_tasks=4)], 1,
                 frozen_commit="abc")
    with pytest.raises(R.TestPhaseCommitMismatch):
        runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("test")


def test_divergence_restarts_once_then_halts(tmp_path, monkeypatch):
    plan(tmp_path, [U1])
    with pytest.raises(Crash):
        runner(tmp_path, CrashAfter(9, TOK, "memory")).run_phase("pilot")
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["key"] = "f" * 64
    rec["crc"] = crc(rec)
    lines[3] = canon(rec)
    cp.write_text("\n".join(lines) + "\n")
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    p, run = payload(tmp_path, U1)
    assert run["restarts"][0].startswith("divergence") and not run["resumed"]
    orig = UnitCache.call

    def always(self, kind, key, compute, extra):
        raise ReplayDivergence(self.pos, {"kind": kind, "key": key}, kind, key)
    monkeypatch.setattr(UnitCache, "call", always)
    U = R.Unit("pilot", "rq1", "dev", "rq1", 1001, "C2", n_tasks=4)
    plan(tmp_path, [U1, U], ) if False else R.write_plan(tmp_path, "pilot", [U1, U], 2)
    with pytest.raises(R.RunHalted):
        runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    monkeypatch.setattr(UnitCache, "call", orig)


def test_two_identical_exceptions_mark_failed_and_are_skipped(tmp_path):
    class Boom(ScriptedBackend):
        def generate(self, *a, **k):
            raise ValueError("boom")
    plan(tmp_path, [U1])
    r = runner(tmp_path, Boom(TOK, "memory"))
    r.run_phase("pilot")
    assert U1.unit_id in r.failed_units("pilot")
    fails = read_jsonl(tmp_path / "failures" / "pilot.jsonl")
    assert len(fails) == 2 and fails[0]["error"] == "ValueError: boom"
    assert r.run_phase("pilot")["ran"] == 0


def test_resume_mismatch_restarts_and_second_enters_replay_only(tmp_path):
    units = [U1, R.Unit("pilot", "rq1", "dev", "rq1", 1001, "C2", n_tasks=4),
             R.Unit("pilot", "rq1", "dev", "rq1", 1002, "C2", n_tasks=4)]
    plan(tmp_path, units)
    for u in units:                                             # every unit interrupted
        R.write_plan(tmp_path, "pilot", [u], 1) if False else None
    for u in units:
        r = runner(tmp_path, CrashAfter(7, TOK, "memory"))
        with pytest.raises(Crash):
            r.run_unit(u, R.read_plan(tmp_path, "pilot"))
    sess = Session("fresh", 0, "pass")
    calls = []
    r = runner(tmp_path, Suffix(TOK, "memory"), session=sess, rerun=lambda: (calls.append(1), (0, "pass"))[1])
    r.run_phase("pilot")
    assert sess.mismatches == 2 and sess.replay_only_mode and calls == [1]
    runs = [payload(tmp_path, u)[1] for u in units]
    assert runs[0]["restarts"][0].startswith("resume_mismatch") and not runs[0]["resumed"]
    assert runs[1]["restarts"][0].startswith("resume_mismatch") and not runs[1]["resumed"]   # the trigger restarts
    assert runs[2]["resumed"] and runs[2]["replay_only"]                                    # later: replay only


def test_backend_busy_refused(tmp_path):
    class Busy(ScriptedBackend):
        busy = True
    plan(tmp_path, [U1])
    from eem_acc.stage3.llm.llama_backend import BackendBusy
    with pytest.raises(BackendBusy):
        runner(tmp_path, Busy(TOK, "memory")).run_unit(U1, R.read_plan(tmp_path, "pilot"))


def test_lease_refuses_second_runtime(tmp_path):
    t = [1000.0]
    a = R.Lease(tmp_path, "rtA", clock=lambda: t[0])
    b = R.Lease(tmp_path, "rtB", clock=lambda: t[0])
    a.acquire()
    with pytest.raises(R.LeaseHeld):
        b.acquire()
    t[0] += 500
    b.acquire()                                                  # stale lease taken over
    a2 = R.Lease(tmp_path, "rtA", clock=lambda: t[0])
    with pytest.raises(R.LeaseHeld):
        a2.acquire()


def test_plan_rules_and_unit_ids(tmp_path):
    R.write_plan(tmp_path, "pilot", [U1], 1)
    with pytest.raises(ValueError, match="new plan version"):
        R.write_plan(tmp_path, "pilot", [U1, U3], 1)
    R.write_plan(tmp_path, "pilot", [U1, U3], 2)
    with pytest.raises(ValueError, match="back"):
        R.write_plan(tmp_path, "pilot", [U1], 1)
    with pytest.raises(ValueError, match="duplicate"):
        R.write_plan(tmp_path, "x", [U1, U1], 1)
    ids = {R.Unit("dev", "rq1", "dev", "rq1", 1001, "C2", tag=f"cfg{i}").unit_id for i in range(3)}
    assert len(ids) == 3 and U3.unit_id == "pilot-rq3-s5000-C2-acc"
    assert R.Unit("test", "rq1", "test", "sp", 3000, "C2").unit_id == "test-rq1-sps3000-C2"
    assert R.Unit.from_json(U1.to_json()) == U1


def test_test_phase_order_d63():
    units = R.test_phase_units(list(range(2000, 2014)), list(range(4000, 4010)), 1)
    rq1 = [u for u in units if u.rq == "rq1"]
    rq3 = [u for u in units if u.rq == "rq3"]
    assert len({u.seed for u in rq1}) == 14 and len({u.seed for u in rq3}) == 10
    assert sum(u.condition == "C2-OU" for u in rq1) == 5
    assert {u.seed for u in rq1 if u.condition == "C2-OU"} == set(range(2000, 2005))
    seq = []
    for u in units:
        key = (u.rq, u.seed)
        if not seq or seq[-1] != key:
            seq.append(key)
    kinds = "".join("1" if k[0] == "rq1" else "3" for k in seq)
    assert kinds.startswith("113" * 7) and kinds == "113" * 7 + "333"
    again = R.test_phase_units(list(range(2000, 2014)), list(range(4000, 4010)), 1)
    assert [u.unit_id for u in again] == [u.unit_id for u in units]
    other = R.test_phase_units(list(range(2000, 2014)), list(range(4000, 4010)), 2)
    assert [u.unit_id for u in other] != [u.unit_id for u in units]   # the shuffle depends on the plan version
    assert len({u.unit_id for u in units}) == len(units)


def test_sharding_splits_whole_units(tmp_path):
    units = [R.Unit("pilot", "rq1", "dev", "rq1", s, "C2", n_tasks=3) for s in (1000, 1001, 1002)]
    plan(tmp_path, units)
    r0 = runner(tmp_path, ScriptedBackend(TOK, "memory"), shard=(0, 2))
    r1 = runner(tmp_path, ScriptedBackend(TOK, "memory"), shard=(1, 2))
    assert r0.run_phase("pilot")["total"] == 2 and r1.run_phase("pilot")["total"] == 1
    assert all(r0.is_done(u) for u in units)
    assert (tmp_path / "manifest" / "pilot" / "0.jsonl").exists() and (tmp_path / "manifest" / "pilot" / "1.jsonl").exists()


def test_ledger_gpu_hours_from_heartbeats(tmp_path):
    t = [0.0]
    a = Ledger(tmp_path, "s1", "rtA", "0", clock=lambda: t[0])
    b = Ledger(tmp_path, "s1", "rtA", "1", clock=lambda: t[0])     # second process, same runtime
    a.start(100.0)
    b.start(None)
    t[0] = 1800.0
    a.event("unit", unit="u", calls_fresh=10, calls_replayed=2, llm_seconds=50.0)
    t[0] = 3600.0
    b.heartbeat()
    c = Ledger(tmp_path, "s2", "rtB", "0", clock=lambda: t[0])
    c.start(90.0)
    t[0] = 5400.0
    c.event("discarded", unit="u", reason="epoch", calls=7)
    s = summarise(tmp_path)
    assert s["gpu_hours"] == pytest.approx(1.0 + 0.5) and s["calls_fresh"] == 10
    assert s["seconds_per_fresh_call"] == pytest.approx(5.0) and s["discarded_calls"] == {"epoch": 7}


@pytest.mark.parametrize("u", [U1, U3], ids=["rq1", "rq3"])
def test_verify_replay(tmp_path, u):
    plan(tmp_path, [u])
    runner(tmp_path, ScriptedBackend(TOK, strategy(u))).run_phase("pilot")
    key_source = ScriptedBackend(TOK, strategy(u))
    assert R.verify_replay(tmp_path, "pilot", u.unit_id, key_source, TOK)
    res = json.loads((tmp_path / "units" / "pilot" / f"{u.unit_id}.json").read_text())
    res["success_rate"] = -1.0                                   # a payload that does not match its cache
    (tmp_path / "units" / "pilot" / f"{u.unit_id}.json").write_text(json.dumps(res))
    assert not R.verify_replay(tmp_path, "pilot", u.unit_id, key_source, TOK)


def test_verify_replay_never_generates(tmp_path):
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    cp = tmp_path / "cache" / f"{U1.unit_id}.jsonl"
    lines = cp.read_text().splitlines()
    cp.write_text("\n".join(lines[:-3]) + "\n")
    before = cp.read_text()
    with pytest.raises(ReplayDivergence):
        R.verify_replay(tmp_path, "pilot", U1.unit_id, ScriptedBackend(TOK, "memory"), TOK)
    assert cp.read_text() == before


def test_scrub_audit_finds_planted_leak(tmp_path):
    plan(tmp_path, [U1])
    runner(tmp_path, ScriptedBackend(TOK, "memory")).run_phase("pilot")
    res = json.loads((tmp_path / "units" / "pilot" / f"{U1.unit_id}.json").read_text())
    clean = audit_scrub([res])
    assert clean["hits"] == [] and clean["units"] == 1
    from eem_acc.stage3.apisuite import sample_stream
    from eem_acc.stage3.llm.audit import stream_entities
    sp = sample_stream(1000, "dev")
    ent = sorted(stream_entities({**sp, "tasks": sp["tasks"][:6]}))[0]
    res["final_notes"][0]["reflection"] = f"use {ent}"
    leak = audit_scrub([res])
    assert leak["hits"] and leak["hits_by_condition"]["C2"] == 1 and leak["hits"][0]["value"] == ent
