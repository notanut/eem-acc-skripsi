"""S3.2a end to end with the scripted backend (plumbing tests, never evidence; spec section 11)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eem_acc.stage3.agent import BagOfWordsEmbedder, CONDITIONS, ScriptedBackend, StubTokenizer, run_rq1_unit
from eem_acc.stage3.apisuite import sample_stream
from eem_acc.stage3.apisuite.leakcheck import last_working_change, render_items
from eem_acc.stage3.apisuite.ops import OPS

REPO = Path(__file__).resolve().parents[1]
TOK = StubTokenizer()
SEEDS = range(1000, 1020)
STRATEGY = {"C0": "docs", "C1": "docs", "F": "docs", "R": "prompt_docs", "A1": "docs", "C2-OU": "docs"}


def unit(seed, cond, strategy=None, **kw):
    spec = sample_stream(seed, "dev")
    return run_rq1_unit(spec, cond, ScriptedBackend(TOK, strategy or STRATEGY.get(cond, "memory"), **kw),
                        BagOfWordsEmbedder(), TOK)


@pytest.fixture(scope="module")
def runs():
    return {(s, c): unit(s, c) for s in SEEDS for c in ("C0", "C2", "C3")}


def test_no_wrong_state_and_memory_reduces_repeat_drift_errors(runs):
    rep = {"C0": 0, "C2": 0, "C3": 0}
    for (seed, cond), u in runs.items():
        assert all(t["reason"] != "wrong_state" for t in u["tasks"]), (seed, cond)
        seen = set()
        for t in u["tasks"]:
            for op, out in t["first_call"].items():
                if op in seen and out in ("drift", "silent_drift"):
                    rep[cond] += 1
                seen.add(op)
    for seed in SEEDS:
        ok0 = {t["task"] for t in runs[(seed, "C0")]["tasks"] if t["success"]}
        for c in ("C2", "C3"):
            okc = {t["task"] for t in runs[(seed, c)]["tasks"] if t["success"]}
            assert ok0 <= okc, (seed, c, ok0 - okc)
    assert rep["C2"] < rep["C0"] and rep["C3"] < rep["C0"], rep


def test_naive_c0_fails_on_drifted_operations():
    u = unit(1000, "C0", "naive")
    assert sum(t["success"] for t in u["tasks"]) <= 2
    assert all(t["drift_errors"] > 0 for t in u["tasks"] if t["exposures_planned"])


@pytest.mark.parametrize("cond", CONDITIONS)
def test_every_condition_runs(cond):
    u = unit(1001, cond)
    assert len(u["tasks"]) == 20 and u["stamp"]["condition"] == cond
    if cond in ("C2", "C3", "AN", "A2"):
        assert any(n["line"] for n in u["final_notes"])
    if cond in ("C2-OU", "A1"):
        assert all(n.get("line") is None for n in u["final_notes"])


def test_close_canonical_op_equals_suite_op(runs):                  # r4 review M1 (aliases learned since creation)
    n = 0
    for (seed, cond), u in runs.items():
        for e in u["note_events"]:
            if e["event"] == "close" and "success_args" in e and not e.get("no_schema"):
                assert e["canonical_op"] == e["suite_op"], e
                n += 1
    assert n > 50


def test_shown_lines_parse_back_to_items(runs):                     # D116 compact form round trip
    from eem_acc.stage3.agent.memory import EEMStore
    from eem_acc.stage3.agent.scripted import parse_line
    key = lambda it: json.dumps(it, sort_keys=True)
    n = 0
    for (seed, cond), u in runs.items():
        for e in u["note_events"]:
            if e["event"] == "close" and e["items"]:
                shown = "Last working change: " + "; ".join(EEMStore.display_items(e["items"]))
                assert sorted(map(key, parse_line(shown))) == sorted(map(key, e["items"])), e["items"]
                n += 1
    assert n > 50


def _entities(spec):
    out = set()
    for t in spec["tasks"]:
        for kind, ents in t["setup"].items():
            for e in ents:
                for k, v in e.items():
                    if k in ("customer_id", "email", "name", "sku", "order_id", "payment_id", "shipment_id",
                             "ticket_id", "subject") and isinstance(v, str):
                        out.add(v)
                    if k in ("price", "total", "amount") and isinstance(v, int) and v >= 1000:
                        out.add(str(v))
    return out


@pytest.mark.parametrize("cond", ["C2", "C1", "A1", "AN"])
def test_no_entity_value_in_stored_notes_even_with_echoing_reflections(cond):
    for seed in (1000, 1003):
        spec = sample_stream(seed, "dev")
        u = run_rq1_unit(spec, cond, ScriptedBackend(TOK, STRATEGY.get(cond, "memory"), echo_reflections=True),
                         BagOfWordsEmbedder(), TOK)
        ents = _entities(spec)
        for n in u["final_notes"]:
            shown = json.dumps({k: n.get(k) for k in ("kt", "reflection", "text_raw", "line", "text")})
            for e in ents:
                assert e not in shown, (cond, seed, e, shown)
        if cond in ("C2", "C1", "AN"):
            assert sum(t["scrubbed"] for t in u["tasks"]) > 0


def test_malformed_replies_are_handled():
    u = unit(1002, "C2", malformed_every=3)
    assert sum(t["fallbacks"] for t in u["tasks"]) > 0
    assert any(s.get("fence") for s in u["steps"])


def test_determinism_same_unit_twice():
    a = json.dumps(unit(1004, "C2"), sort_keys=True)
    b = json.dumps(unit(1004, "C2"), sort_keys=True)
    assert a == b


def test_determinism_across_pythonhashseed():
    code = ("import json,sys,hashlib;sys.path.insert(0,'src');sys.path.insert(0,'tests');"
            "from test_s32_e2e import unit;"
            "print(hashlib.sha256(json.dumps(unit(1005,'C2'),sort_keys=True).encode()).hexdigest())")
    outs = set()
    for hs in ("0", "999"):
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=hs), check=True)
        outs.add(r.stdout.strip())
    assert len(outs) == 1


def test_close_lines_recomputed_independently(runs):             # review M5
    n = 0
    for (seed, cond), u in runs.items():
        if cond == "C0":
            continue
        for e in u["note_events"]:
            if e["event"] != "close" or "success_args" not in e:
                continue
            ref = last_working_change(e["trigger_args"], e["success_args"], OPS[e["canonical_op"]].schema,
                                      moved_to=e["moved_to"], error_keys=frozenset(e["writer_error_keys"]),
                                      known_keys=frozenset(e["known_keys"]))
            new = e.get("new_items", e["items"])
            if new or ref:
                assert render_items(new) == ref, e                 # the new diff equals the S3.1 reference
                n += 1
            kept = e.get("kept_items", [])                           # line = items kept at a merge + new (D113)
            assert e["items"] == kept + [it for it in new if it not in kept], e
    assert n > 50


def _entities_full(spec):
    out = _entities(spec)
    for t in spec["tasks"]:
        for p in t["setup"].get("products", []):
            out.add(p["name"])
        for tk in t["setup"].get("tickets", []):
            out.add(tk["subject"])
        for k in ("subject", "text"):
            if k in t["params"]:
                out.add(t["params"][k])
    return out


@pytest.mark.parametrize("cond", CONDITIONS)
def test_scrub_all_conditions_extended_entities(cond):
    for seed in (1006, 1007):
        spec = sample_stream(seed, "dev")
        u = run_rq1_unit(spec, cond, ScriptedBackend(TOK, STRATEGY.get(cond, "memory"), echo_reflections=True),
                         BagOfWordsEmbedder(), TOK)
        ents = _entities_full(spec)
        for n in u["final_notes"]:
            shown = json.dumps({k: n.get(k) for k in ("kt", "reflection", "text_raw", "line", "text")})
            for e in ents:
                assert e not in shown, (cond, seed, e)


def test_kt_never_carries_entities_over_many_streams():
    for seed in range(1000, 1040):
        spec = sample_stream(seed, "dev")
        u = run_rq1_unit(spec, "C2", ScriptedBackend(TOK, "memory"), BagOfWordsEmbedder(), TOK)
        ents = _entities_full(spec)
        for s_ in u["steps"]:
            if "kt" in s_:
                blob = json.dumps(s_["kt"])
                for e in ents:
                    assert e not in blob, (seed, e, blob)


def test_shown_note_prefix_through_the_loop():
    for cond, prefix in (("C2", "\nNote: "), ("C1", "\nReflection: "), ("A2", "\nNote: ")):
        u = unit(1000, cond)
        obs = [o for traj in u["trajectories"] for _, o in traj if o.startswith("Observation: 4")]
        assert obs and any(prefix in o for o in obs), cond
    u = unit(1000, "C0")
    obs = [o for traj in u["trajectories"] for _, o in traj if o.startswith("Observation: 4")]
    assert obs and not any("\nNote: " in o or "\nReflection: " in o for o in obs)


def test_sp_stream_runs_end_to_end():
    spec = sample_stream(3000, "test", kind="sp")
    for cond in ("C0", "C2"):
        u = run_rq1_unit(spec, cond, ScriptedBackend(TOK, STRATEGY.get(cond, "memory")), BagOfWordsEmbedder(), TOK)
        assert all(t["reason"] != "wrong_state" for t in u["tasks"])


def test_note_stats_logged_per_step_with_retrieved_notes(runs):
    n = 0
    for (seed, cond), u in runs.items():
        for st in u["steps"]:
            if st.get("retrieved"):
                assert [x["id"] for x in st["note_stats"]] == st["retrieved"]
                assert all(0 <= x["items_shown"] <= x["items_total"] for x in st["note_stats"])
                n += 1
            else:
                assert "note_stats" not in st
    assert n > 100
