"""S3.1: stream sampler (spec section 8, D70): exposure rule, stratification, SP, prefixes, determinism, mix."""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from eem_acc.stage3.apisuite.drift import POOLS, get_pair, legal_together
from eem_acc.stage3.apisuite.ops import OP_NAMES
from eem_acc.stage3.apisuite.streams import (SEED_RANGES, exposures, prefix, sample_stream, spec_hash)
from eem_acc.stage3.apisuite.templates import TEMPLATES

REPO = Path(__file__).resolve().parents[1]
ALL = [(kind, split, seed) for (kind, split), r in SEED_RANGES.items() for seed in r]


@pytest.fixture(scope="module")
def streams():
    return {(k, sp, s): sample_stream(s, sp, kind=k) for k, sp, s in ALL}


def _rule_ok(spec: dict, op: str) -> bool:
    pos = exposures(spec, op)
    return bool(pos) and pos[0] < 7 and sum(p < 15 for p in pos) >= 3


def test_sampler_never_fails_and_exposure_rule_holds_on_full_and_prefixes(streams):
    assert len(streams) == 2200
    for key, spec in streams.items():
        for n in (20, 16, 15):
            sp = prefix(spec, n) if n < 20 else spec
            for op in spec["drifted_ops"]:
                assert _rule_ok(sp, op), (key, n, op)


def test_drift_sets_stratified_and_legal(streams):
    for (kind, split, seed), spec in streams.items():
        ids = spec["drift_schedule"][0][1]
        pairs = [get_pair(i) for i in ids]
        assert len(pairs) == 10 and len(spec["drifted_ops"]) == 8
        assert set(i for i in ids if not i.endswith(":sp1")) <= set(POOLS[split])
        by_op = Counter(p.op for p in pairs)
        assert sorted(by_op.values()) == [1] * 6 + [2] * 2
        for op, n in by_op.items():
            if n == 2:
                a, b = [p for p in pairs if p.op == op]
                assert legal_together(a, b), (seed, a.id, b.id)
        types = {p.type for p in pairs if not p.id.endswith(":sp1")}
        assert types == {"rename", "type", "new_required", "enum", "moved", "silent"}, (seed, types)


def test_error_styles_four_four(streams):
    for spec in streams.values():
        d = spec["drifted_ops"]
        st = spec["styles"]
        assert sum(st[o] == "informative" for o in d) == 4
        assert sum(st[o] == "informative" for o in OP_NAMES if o not in d) == 4


def test_sp_streams(streams):
    for (kind, split, seed), spec in streams.items():
        if kind != "sp":
            continue
        op = spec["sp_op"]
        assert f"{op}:sp1" in spec["drift_schedule"][0][1]
        for n in (20, 16, 15):
            sp = prefix(spec, n) if n < 20 else spec
            mid = n // 2
            assert sp["drift_schedule"][1][0] == mid and f"{op}:sp2" in sp["drift_schedule"][1][1]
            pos = exposures(sp, op)
            assert sum(p < mid for p in pos) >= 1 and sum(p >= mid for p in pos) >= 2, (seed, n)


def test_rq3_dealing(streams):
    for (kind, _, _), spec in streams.items():
        if kind == "rq3":
            assert spec["agents"] == [[0, 4, 8, 12, 16], [1, 5, 9, 13, 17], [2, 6, 10, 14, 18], [3, 7, 11, 15, 19]]


def test_seed_ranges_disjoint_and_enforced():
    rs = list(SEED_RANGES.values())
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            assert not set(rs[i]) & set(rs[j])
    with pytest.raises(ValueError):
        sample_stream(2000, "dev")


def test_template_mix_within_tolerance(streams):
    for (kind, split) in [("rq1", "dev"), ("rq1", "test")]:
        c = Counter(t["template"] for (k, sp, _), s in streams.items() if (k, sp) == (kind, split)
                    for t in s["tasks"])
        n = sum(c.values())
        share = {t: c[t] / n * len(TEMPLATES) for t in TEMPLATES}
        # the exposure rule tilts the mix towards multi-operation templates (measured 0.43-1.79 x uniform)
        assert 0.4 <= min(share.values()) and max(share.values()) <= 1.9, share


def test_pilot_knobs_do_not_change_drift_or_styles():
    a = sample_stream(1000, "dev")
    b = sample_stream(1000, "dev", n_distractors=4, long_share=0.4)
    assert a["drift_schedule"] == b["drift_schedule"] and a["styles"] == b["styles"]


def test_unique_product_names_match_only_themselves():
    for seed in range(2000, 2020):
        spec = sample_stream(seed, "test")
        names = [p["name"] for t in spec["tasks"] for p in t["setup"].get("products", [])]
        assert len(names) == len(set(names))
        for nm in names:
            assert [m for m in names if nm.lower() in m.lower()] == [nm]


def test_deterministic_and_hash():
    a = sample_stream(2042, "test")
    assert a == sample_stream(2042, "test") and a["sha256"] == spec_hash(a)
    p = prefix(a, 16)
    assert p["prefix_of"] == a["sha256"] and p["tasks"] == a["tasks"][:16]


def test_deterministic_across_pythonhashseed():
    code = ("import json,sys;sys.path.insert(0,'src');from eem_acc.stage3.apisuite import sample_stream;"
            "print(json.dumps([sample_stream(s,'test')['sha256'] for s in (2000,2001,2002)]))")
    here = [sample_stream(s, "test")["sha256"] for s in (2000, 2001, 2002)]
    for hs in ("0", "12345"):
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                             env=dict(os.environ, PYTHONHASHSEED=hs), check=True)
        assert json.loads(out.stdout) == here
