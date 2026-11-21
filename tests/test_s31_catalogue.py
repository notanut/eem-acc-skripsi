"""S3.1: drift catalogue, pools, legality rules, docs rendering (docs/s31_spec.md sections 3, 6, 9)."""
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from eem_acc.stage3.apisuite import drift, docs
from eem_acc.stage3.apisuite.drift import (ALL_PAIRS, CATALOGUE, POOLS, SILENT, SP_RENAMES, effective, get_pair,
                                           legal_together, op_view, sp_pairs)
from eem_acc.stage3.apisuite.ops import OPS, OP_NAMES
from eem_acc.stage3.apisuite.schema import schema_paths
from eem_acc.stage3.apisuite.suite import SEMANTIC_STATUSES
from eem_acc.status import RETRYABLE

REPO = Path(__file__).resolve().parents[1]


def test_catalogue_counts_match_spec():
    c = Counter(p.type for p in CATALOGUE.values())
    assert c == {"rename": 16, "type": 5, "new_required": 12, "enum": 9, "moved": 16}
    assert len(CATALOGUE) == 58 and len(SILENT) == 16 and len(ALL_PAIRS) == 74
    assert set(p.op for p in CATALOGUE.values()) == set(OP_NAMES)


def test_new_required_fields_are_enum_boolean_or_constant():
    for p in CATALOGUE.values():
        if p.type == "new_required":
            assert p.new_field.kind in ("enum", "boolean", "const"), p.id
            assert p.new_field.required


def test_identifier_renames_only_to_id_names():        # D69
    for p in CATALOGUE.values():
        if p.type == "rename":
            f = schema_paths(OPS[p.op].schema)[p.path]
            if f.kind == "id":
                assert p.new_name.endswith("_id"), p.id


def test_drift_only_on_required_documented_fields():
    for p in CATALOGUE.values():
        if p.type in ("rename", "type", "enum"):
            assert schema_paths(OPS[p.op].schema)[p.path].required, p.id


def test_enum_remaps_cover_all_values():
    for p in CATALOGUE.values():
        if p.type == "enum":
            f = schema_paths(OPS[p.op].schema)[p.path]
            assert tuple(m[0] for m in p.mapping) == f.values
            assert len({m[1] for m in p.mapping}) == len(f.values)


def test_moved_ids_unique_and_not_existing_ops():
    new = [p.new_op for p in CATALOGUE.values() if p.type == "moved"]
    assert len(set(new)) == 16 and not set(new) & set(OP_NAMES)


def test_silent_underlying_is_own_base_change_in_same_pool():   # D71
    for sid, s in SILENT.items():
        u = CATALOGUE[s.underlying]
        assert u.op == s.op and u.type in drift.BASE_TYPES
        for split in ("dev", "test"):
            assert (sid in POOLS[split]) == (u.id in POOLS[split])


def test_pools_disjoint_cover_everything_and_every_op_in_both():
    dev, test = set(POOLS["dev"]), set(POOLS["test"])
    assert not dev & test
    assert dev | test == set(ALL_PAIRS)
    for op in OP_NAMES:
        assert any(i.startswith(op + ":") for i in dev) and any(i.startswith(op + ":") for i in test)


def test_pools_stratified_by_type():
    for t in ("rename", "moved", "type", "new_required", "enum"):
        n_dev = sum(1 for i in POOLS["dev"] if get_pair(i).type == t)
        n_test = sum(1 for i in POOLS["test"] if get_pair(i).type == t)
        assert abs(n_dev - n_test) <= 2, t


def test_pools_match_committed_evidence():
    ev = json.loads((REPO / "docs/evidence/s31_drift_pools.json").read_text())
    assert ev["pools"] == POOLS
    assert ev["pairs"] == {i: ALL_PAIRS[i].as_json() for i in sorted(ALL_PAIRS)}


def test_legal_together_rules():
    assert not legal_together(CATALOGUE["pay_order:rename"], CATALOGUE["pay_order:type"])       # same field
    assert not legal_together(CATALOGUE["pay_order:rename"], CATALOGUE["pay_order:new_required"])  # D69
    assert legal_together(CATALOGUE["pay_order:rename"], CATALOGUE["pay_order:enum"])
    assert legal_together(CATALOGUE["pay_order:moved"], CATALOGUE["pay_order:rename"])
    s = SILENT["pay_order:silent"]
    assert not legal_together(s, CATALOGUE[s.underlying])


@pytest.mark.parametrize("split", ["dev", "test"])
def test_every_pool_has_enough_two_pair_operations(split):
    pairs = [get_pair(i) for i in POOLS[split]]
    combos = [(p, q) for p, q in itertools.combinations(pairs, 2) if legal_together(p, q)]
    # a stream needs 2 double operations; at least 4 operations and 10 combinations keep the doubles varied
    # (measured: dev 6 operations / 15 combinations, test 8 / 10)
    assert len({p.op for p, _ in combos}) >= 4 and len(combos) >= 10


def test_status_set_disjoint_from_retryable():
    assert not SEMANTIC_STATUSES & set(RETRYABLE)


def test_docs_render_every_operation_and_documented_is_stable():
    text = docs.documented_all()
    for op in OP_NAMES:
        assert text.count(f"\n{op} - ") + text.startswith(f"{op} - ") == 1
    assert "identifier" in text and "enum: standard | express" in text and "(default 10)" in text


def test_current_docs_reflect_each_pair():
    for p in ALL_PAIRS.values():
        v = op_view(p.op, [p])
        t = docs.render(v)
        e = effective(p)
        if e.type == "moved":
            assert t.startswith(e.new_op + " - ") and f"(formerly {p.op})" in t
        elif e.type == "rename":
            assert f" {e.new_name} " in t
        elif e.type == "new_required":
            assert e.new_field.name in t
            if e.new_field.kind == "const":
                assert "constant: " in t
        elif e.type == "enum":
            assert " | ".join(m[1] for m in e.mapping) in t
        elif e.type == "type":
            assert docs.KIND_WORDS[e.new_kind].split(" (")[0] in t
    assert docs.documented("pay_order") == docs.render(op_view("pay_order", []))


def test_sp_renames_are_enum_fields_and_render():
    for op, (field, n1, n2) in SP_RENAMES.items():
        assert schema_paths(OPS[op].schema)[field].kind == "enum"
        p1, p2 = sp_pairs(op)
        assert f" {n2} " in docs.render(op_view(op, [p2]))
        v2 = op_view(op, [p2])
        assert n1 in v2.drift_paths and n2 in v2.drift_paths and field in v2.drift_paths


def test_sp_stage_names_differ_from_catalogue_renames():
    cat = {(p.op, p.new_name) for p in CATALOGUE.values() if p.type == "rename"}
    for op, (_, n1, n2) in SP_RENAMES.items():
        assert (op, n1) not in cat and (op, n2) not in cat
