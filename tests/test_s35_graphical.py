"""S3.5 spec section 5: the graphical procedure [BMBP09] and section 4's sign-flip statistics."""
import itertools
import math

import numpy as np
import pytest

from eem_acc.stage3.analysis import graphical as gr
from eem_acc.stage3.analysis import stats as st

A = 0.025


def holm_graph(m):
    off = 1 / (m - 1)
    return gr.Graph(tuple(f"H{i}" for i in range(m)), tuple([1 / m] * m),
                    tuple(tuple(0.0 if i == k else off for k in range(m)) for i in range(m)))


def holm_reject(p, alpha):
    order = sorted(range(len(p)), key=lambda i: p[i])
    out = set()
    for r, i in enumerate(order):
        if p[i] <= alpha / (len(p) - r):
            out.add(i)
        else:
            break
    return out


def random_graph(rng, m):
    """Random valid graphs, including zero initial weights and rows short of 1 (code review L9, L10)."""
    w = rng.dirichlet(np.ones(m)) * rng.choice([1.0, rng.uniform(0.5, 1.0)])
    if m > 2 and rng.uniform() < 0.5:
        z = rng.choice(m, size=int(rng.integers(1, m - 1)), replace=False)
        w[z] = 0.0
        w = w / w.sum() * rng.choice([1.0, 0.8])
    G = []
    for i in range(m):
        row = rng.dirichlet(np.ones(m - 1)) * rng.choice([1.0, rng.uniform(0, 1)])
        if rng.uniform() < 0.2:
            row[int(rng.integers(0, m - 1))] = 0.0
        row = list(row)
        row.insert(i, 0.0)
        G.append(tuple(row))
    return gr.Graph(tuple(f"H{i}" for i in range(m)), tuple(w), tuple(G))


def random_p(rng, names):
    p = {}
    for n in names:
        u = rng.uniform()
        p[n] = None if u < 0.08 else (0.0 if u < 0.12 else float(rng.uniform(0, 0.05)))
    return p


# ---------------------------------------------------------------------------------------------- known procedures
def test_equal_weights_full_transitions_is_holm():
    rng = np.random.default_rng(1)
    for m in (2, 3, 4, 5):
        g = holm_graph(m)
        for _ in range(300):
            p = rng.uniform(0, 0.06, m)
            got = {g.index(n) for n in gr.reject(g, dict(zip(g.names, p)), A).rejected}
            assert got == holm_reject(list(p), A)


def test_adjusted_p_equals_holm_adjusted():
    rng = np.random.default_rng(2)
    g = holm_graph(4)
    for _ in range(100):
        p = rng.uniform(0, 0.2, 4)
        adj = gr.adjusted_p(g, dict(zip(g.names, p)))
        order = sorted(range(4), key=lambda i: p[i])
        run, ref = 0.0, {}
        for r, i in enumerate(order):
            run = max(run, min(1.0, (4 - r) * p[i]))
            ref[i] = run
        for i in range(4):
            assert adj[f"H{i}"] == pytest.approx(ref[i], abs=1e-12)


def test_fixed_sequence():
    g = gr.Graph(("a", "b", "c"), (1.0, 0.0, 0.0), ((0, 1, 0), (0, 0, 1), (0, 0, 0)))
    assert gr.reject(g, {"a": 0.01, "b": 0.02, "c": 0.03}, A).rejected == ["a", "b"]
    assert gr.reject(g, {"a": 0.03, "b": 0.001, "c": 0.001}, A).rejected == []
    assert gr.reject(g, {"a": 0.025, "b": 0.025, "c": 0.0}, A).rejected == ["a", "b", "c"]


def test_fallback():
    # Wiens' fallback: H1 at a1, H2 at a2 (+a1 if H1 rejected), H3 at a3 (+ what H2 had if H2 rejected)
    g = gr.Graph(("H1", "H2", "H3"), (0.5, 0.3, 0.2), ((0, 1, 0), (0, 0, 1), (0, 0, 0)))
    assert gr.reject(g, {"H1": 0.02, "H2": 0.02, "H3": 0.02}, A).rejected == []
    r = gr.reject(g, {"H1": 0.012, "H2": 0.02, "H3": 0.025}, A).rejected
    assert r == ["H1", "H2", "H3"]                 # 0.0125, then 0.02, then 0.025
    r = gr.reject(g, {"H1": 0.5, "H2": 0.0075, "H3": 0.0125}, A).rejected
    assert r == ["H2", "H3"]                       # H2 at 0.0075, H3 at 0.005 + 0.0075


# ---------------------------------------------------------------------------------------------- properties
def test_order_independence_and_closed_test_equivalence():
    rng = np.random.default_rng(3)
    for _ in range(150):
        m = int(rng.integers(2, 6))
        g = random_graph(rng, m)
        p = random_p(rng, g.names)
        sets = {frozenset(gr.reject(g, p, A, order=list(o)).rejected) for o in itertools.permutations(g.names)}
        assert len(sets) == 1
        adj = gr.adjusted_p(g, p)
        assert next(iter(sets)) == frozenset(n for n in g.names if adj[n] is not None and adj[n] <= A + 1e-12)


def test_invariants_after_every_step():
    rng = np.random.default_rng(4)
    for _ in range(200):
        m = int(rng.integers(2, 7))
        g = random_graph(rng, m)
        w, G = gr._state(g)
        for j in list(rng.permutation(m)):
            gr._remove(w, G, int(j))
            assert sum(w.values()) <= 1 + 1e-9 and all(v >= -1e-12 for v in w.values())
            for l in G:
                assert G[l][l] == 0.0
                assert sum(G[l].values()) <= 1 + 1e-9
                assert all(v >= -1e-12 for v in G[l].values())


def test_graph_validation():
    with pytest.raises(ValueError):
        gr.Graph(("a", "b"), (0.7, 0.7), ((0, 1), (1, 0)))
    with pytest.raises(ValueError):
        gr.Graph(("a", "b"), (0.5, 0.5), ((0.5, 0.5), (1, 0)))
    with pytest.raises(ValueError):
        gr.Graph(("a", "b"), (0.5, 0.5), ((0, 0.7), (1.2, 0)))
    with pytest.raises(ValueError):
        gr.Graph(("a", "a"), (0.5, 0.5), ((0, 1), (1, 0)))


# ---------------------------------------------------------------------------------------------- the frozen graphs
def p1(**kw):
    base = {n: 1.0 for n in gr.rq1_graph().names}
    base.update({k.replace("_", "-"): v for k, v in kw.items()})
    return base


def test_rq1_graph_matches_the_design_table_entry_by_entry():
    """Design section 8, every one of the 36 transitions (code review H2)."""
    g = gr.rq1_graph()
    assert g.names == ("NI1", "S1", "E2", "NI3", "S3", "E4")
    assert g.weights == (0.25, 0.0, 0.25, 0.25, 0.0, 0.25)
    t = 1 / 3
    design = {"NI1": {"S1": 1.0}, "NI3": {"S3": 1.0},
              "S1": {"E2": t, "NI3": t, "E4": t}, "S3": {"E4": t, "NI1": t, "E2": t},
              "E2": {"NI1": t, "NI3": t, "E4": t}, "E4": {"NI1": t, "NI3": t, "E2": t}}
    for a in g.names:
        for b in g.names:
            assert g.transitions[g.index(a)][g.index(b)] == pytest.approx(design[a].get(b, 0.0)), (a, b)


def test_rq3_graph_matches_the_design_entry_by_entry():
    g = gr.rq3_graph()
    assert g.names == ("NI-R1", "S-R1", "R2") and g.weights == (0.5, 0.0, 0.5)
    design = {"NI-R1": {"S-R1": 1.0}, "S-R1": {"R2": 1.0}, "R2": {"NI-R1": 1.0}}
    for a in g.names:
        for b in g.names:
            assert g.transitions[g.index(a)][g.index(b)] == design[a].get(b, 0.0), (a, b)


def test_rq1_hand_cases():
    g = gr.rq1_graph()
    assert gr.reject(g, p1(), A).rejected == []
    assert set(gr.reject(g, {n: 1e-4 for n in g.names}, A).rejected) == set(g.names)
    assert gr.reject(g, p1(NI1=0.006), A).rejected == ["NI1"]                     # 0.00625; S1 needs its own p
    assert gr.reject(g, p1(S1=1e-6), A).rejected == []                            # S1 has no alpha before NI1
    assert gr.reject(g, p1(NI1=0.006, S1=0.006), A).rejected == ["NI1", "S1"]     # S1 inherits 0.00625
    assert gr.reject(g, p1(E2=0.007), A).rejected == []                           # 0.007 > 0.00625
    # NI1 and S1 rejected -> S1's 1/4 split 1/3 each to E2, NI3, E4: E2 at 0.025 * (1/4 + 1/12) = 0.00833
    assert gr.reject(g, p1(NI1=0.001, S1=0.001, E2=0.008), A).rejected == ["NI1", "S1", "E2"]
    assert gr.reject(g, p1(NI1=0.001, S1=0.001, E2=0.0084), A).rejected == ["NI1", "S1"]
    # E2 rejected first passes 1/3 of its 1/4 to NI1: NI1 at 0.025 * (1/4 + 1/12)
    assert gr.reject(g, p1(E2=0.006, NI1=0.008), A).rejected == ["E2", "NI1"]
    # the C1 side, mirrored (code review H2): NI3 -> S3 -> E4, NI1, E2
    assert gr.reject(g, p1(NI3=0.006, S3=0.006), A).rejected == ["NI3", "S3"]
    assert gr.reject(g, p1(NI3=0.001, S3=0.001, E4=0.008), A).rejected == ["NI3", "S3", "E4"]
    # S3 passes 1/3 of its 1/4 to NI1 and to E2: each at 0.025 * (1/4 + 1/12) = 0.00833
    assert gr.reject(g, p1(NI3=0.001, S3=0.001, NI1=0.008), A).rejected == ["NI3", "S3", "NI1"]
    assert gr.reject(g, p1(NI3=0.001, S3=0.001, E2=0.008), A).rejected == ["NI3", "S3", "E2"]
    # E4 rejected first passes 1/3 of its 1/4 to NI3
    assert gr.reject(g, p1(E4=0.006, NI3=0.008), A).rejected == ["E4", "NI3"]
    assert gr.reject(g, p1(E4=0.006, NI3=0.0084), A).rejected == ["E4"]
    # S1 rejected passes 1/3 to NI3
    assert gr.reject(g, p1(NI1=0.001, S1=0.001, NI3=0.008), A).rejected == ["NI1", "S1", "NI3"]
    assert gr.reject(g, p1(E2=0.006, NI1=0.0084), A).rejected == ["E2"]


def test_rq3_hand_cases():
    g = gr.rq3_graph()
    assert gr.reject(g, {"NI-R1": 0.01, "S-R1": 0.012, "R2": 0.02}, A).rejected == ["NI-R1", "S-R1", "R2"]
    assert gr.reject(g, {"NI-R1": 0.013, "S-R1": 0.0, "R2": 0.012}, A).rejected == ["R2", "NI-R1", "S-R1"]
    assert gr.reject(g, {"NI-R1": 0.013, "S-R1": 0.0, "R2": 0.013}, A).rejected == []
    assert gr.reject(g, {"NI-R1": 0.01, "S-R1": 0.02, "R2": 0.5}, A).rejected == ["NI-R1"]


def test_missing_p_is_never_rejected_and_keeps_its_weight():
    g = gr.rq3_graph()
    r = gr.reject(g, {"NI-R1": None, "S-R1": None, "R2": 0.013}, A)
    assert r.rejected == []
    adj = gr.adjusted_p(g, {"NI-R1": None, "S-R1": None, "R2": 0.01})
    assert adj["NI-R1"] is None and adj["R2"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------------------------- sign-flip
def test_sign_flip_one_sided_by_hand():
    assert st.sign_flip_one_sided([1, 1, 1]) == pytest.approx(1 / 8)
    assert st.sign_flip_one_sided([1, -1]) == pytest.approx(3 / 4)
    assert st.sign_flip_one_sided([-1, -1]) == pytest.approx(1.0)
    assert st.sign_flip_one_sided([2, 1, 1]) == pytest.approx(1 / 8)      # means: only all-+ reaches 4/3
    assert st.sign_flip_one_sided([3, 1, 1]) == pytest.approx(1 / 8)      # only (+,+,+) reaches 5/3
    # x = (3, 1, -1), mean 1: (+,+,+) 1, (+,+,-) 5/3, (+,-,-) 1 reach it; the other five do not -> 3/8
    assert st.sign_flip_one_sided([3, 1, -1]) == pytest.approx(3 / 8)
    x = np.array([3.0, 1.0, 1.0])
    pats = np.array(list(itertools.product((1, -1), repeat=3)))
    assert st.sign_flip_one_sided(x) == pytest.approx(np.mean((pats * x).mean(1) >= x.mean() - 1e-12))


def test_ties_floor_and_shift():
    assert st.sign_flip_one_sided([1, 0, 0]) == pytest.approx(1 / 2)
    assert st.sign_flip_one_sided([1] * 8 + [0] * 6) == pytest.approx(2 ** -8)
    d = np.array([-0.05] * 14)                      # C2 one task worse in every 20-task stream: all ties in NI
    x = st.directional(d, "greater", shift=1 / 20)
    assert st.untied(x) == 0 and st.sign_flip_one_sided(x) == pytest.approx(1.0)
    assert list(st.directional([1.0, -2.0], "less")) == [-1.0, 2.0]


def test_sign_flip_limits():
    with pytest.raises(ValueError):
        st.sign_flip_one_sided([])
    with pytest.raises(ValueError):
        st.sign_flip_one_sided([1] * 21)


def test_two_sided_matches_stage2():
    from eem_acc.analysis import sign_flip_test
    rng = np.random.default_rng(5)
    for _ in range(20):
        d = rng.normal(size=8)
        assert st.sign_flip_two_sided(d) == pytest.approx(sign_flip_test(d))


def test_t_ci_and_mdd():
    from scipy import stats
    d = np.array([0.1, 0.0, 0.05, -0.05, 0.1, 0.15, 0.0, 0.05])
    lo, hi = st.t_ci(d)
    h = stats.t.ppf(0.975, 7) * d.std(ddof=1) / math.sqrt(8)
    assert (lo, hi) == pytest.approx((d.mean() - h, d.mean() + h))
    m = st.mdd(d, 0.00625)
    assert m == pytest.approx((stats.t.ppf(1 - 0.00625, 7) + stats.t.ppf(0.8, 7)) * d.std(ddof=1) / math.sqrt(8))
    assert math.isnan(st.mdd([0.1], 0.00625))


def test_bootstrap_ratio_seeded():
    a = st.bootstrap_ratio([10, 12, 9, 11], [12, 12, 10, 14])
    b = st.bootstrap_ratio([10, 12, 9, 11], [12, 12, 10, 14])
    assert a == b and a[0] == pytest.approx(42 / 48) and a[1] <= a[0] <= a[2]


# ---------------------------------------------------------------------------------------------- BMBP09 examples
import json
from pathlib import Path

EX = json.loads((Path(__file__).parent / "data" / "bmbp09_examples.json").read_text())


def _num(v, eps):
    """An entry like '1-eps' or '0.8*eps' (no builtins available to the expression, r3 second opinion L7)."""
    return float(eval(v, {"__builtins__": {}}, {"eps": eps})) if isinstance(v, str) else float(v)


def test_eps_is_far_above_the_zero_rule_tolerance():
    # _remove treats a denominator 1 - g_lj g_jl <= TOL as 0; with eps near TOL, 1 - (1 - eps) would hit that rule
    assert EX["eps"] >= 1e4 * gr.TOL


def _graph(ex):
    eps = EX["eps"]
    G = tuple(tuple(_num(v, eps) for v in row) for row in ex["transitions"])
    return gr.Graph(tuple(ex["names"]), tuple(ex["weights"]), G)


@pytest.mark.parametrize("ex", EX["examples"], ids=lambda e: e["id"])
def test_bmbp09_worked_examples(ex):
    """Every numerical example of the paper (spec section 5): the rejections, their order and the local levels
    (epsilon edges change levels by at most eps * alpha)."""
    g = _graph(ex)
    r = gr.reject(g, ex["p"], ex["alpha"], order=None)
    # the paper takes j = argmin p_i / alpha_i at each step; reproduce that order for the comparison
    w, G = gr._state(g)
    order, levels = [], []
    pv = {i: ex["p"][n] for i, n in enumerate(g.names)}
    while True:
        cand = {i: pv[i] / w[i] for i in w if w[i] > 1e-12}
        if not cand:
            break
        j = min(cand, key=cand.get)
        if pv[j] > w[j] * ex["alpha"] + 1e-12:
            break
        order.append(g.names[j]); levels.append(w[j] * ex["alpha"])
        gr._remove(w, G, j)
    assert order == ex["rejected_in_order"]
    assert levels == pytest.approx(ex["levels"], abs=1e-6)
    assert set(r.rejected) == set(ex["rejected_in_order"])          # our index-priority order: same set
    if "adjusted_p" in ex:
        assert gr.adjusted_p_sequential(g, ex["p"]) == pytest.approx(ex["adjusted_p"], abs=1e-12)
        assert gr.adjusted_p(g, ex["p"]) == pytest.approx(ex["adjusted_p"], abs=1e-12)


def test_fig10_levels_after_h1_h2():
    ex = next(e for e in EX["examples"] if e["id"] == "fig10_weighted_gatekeeper")
    g = _graph(ex)
    w, G = gr._state(g)
    gr._remove(w, G, g.index("H2"))
    gr._remove(w, G, g.index("H1"))
    got = {g.names[i]: w[i] * ex["alpha"] for i in w}
    assert got == pytest.approx(ex["levels_after_H1_H2"], abs=1e-6)


def test_algorithm2_equals_closed_test_adjusted_p():
    rng = np.random.default_rng(6)
    for _ in range(200):
        m = int(rng.integers(2, 6))
        g = random_graph(rng, m)
        p = random_p(rng, g.names)
        a, b = gr.adjusted_p(g, p), gr.adjusted_p_sequential(g, p)
        for n in g.names:
            assert (a[n] is None) == (b[n] is None)
            if a[n] is not None:
                assert a[n] == pytest.approx(b[n], abs=1e-9), (n, p)
    for graph in (gr.rq1_graph(), gr.rq3_graph()):
        for _ in range(100):
            p = dict(zip(graph.names, rng.uniform(0, 0.05, len(graph.names))))
            assert gr.adjusted_p(graph, p) == pytest.approx(gr.adjusted_p_sequential(graph, p), abs=1e-12)



# ------------------------------------------------------------------------- second-opinion L4 (2026-11-28)
def test_adjusted_p_is_capped_at_one():
    g = holm_graph(4)
    adj = gr.adjusted_p(g, {"H0": 0.5, "H1": 0.6, "H2": 0.7, "H3": 0.9})
    assert all(v == 1.0 for v in adj.values())
    adj2 = gr.adjusted_p_sequential(g, {"H0": 0.5, "H1": 0.6, "H2": 0.7, "H3": 0.9})
    assert all(v == 1.0 for v in adj2.values())


def test_bootstrap_seed_is_frozen_at_zero():
    """The pre-registered bootstrap (seed 0, 10,000 resamples) is pinned to its output. This re-states the resampling
    (rng.integers with shape (10000, n)), so it also pins that implementation detail (r3 review L4)."""
    # 14 streams with varied values, so the interval depends on the seed (6 streams gave the same quantiles for
    # seeds 0 and 1, which would let a changed seed pass)
    num = [37, 52, 41, 66, 29, 58, 44, 71, 35, 49, 62, 31, 55, 46]
    den = [40, 49, 47, 61, 38, 57, 52, 64, 33, 56, 60, 39, 50, 53]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 14, size=(10_000, 14))
    rs = np.asarray(num)[idx].sum(1) / np.asarray(den)[idx].sum(1)
    want = (sum(num) / sum(den), float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975)))
    assert st.bootstrap_ratio(num, den) == pytest.approx(want, abs=0)
    assert st.bootstrap_ratio(num, den, seed=1) != st.bootstrap_ratio(num, den)
