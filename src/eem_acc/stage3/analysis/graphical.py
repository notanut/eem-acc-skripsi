"""The graphical sequentially rejective multiple test procedure of Bretz, Maurer, Brannath and Posch (2009), with
weighted Bonferroni tests (S3.5 spec section 5).

Weights are fractions of alpha (sum <= 1); G[j][l] is the share of H_j's weight passed to H_l when H_j is rejected
(G[j][j] = 0, row sums <= 1). When H_j is rejected (Algorithm 1):
    w_l <- w_l + w_j * G[j][l]                                         for every remaining l
    G[l][k] <- (G[l][k] + G[l][j] * G[j][k]) / (1 - G[l][j] * G[j][l])  for remaining l != k, if G[l][j]G[j][l] < 1
    G[l][k] <- 0                                                        otherwise
The rejection set does not depend on the order of rejections (BMBP09); `reject` takes the smallest index, and the
tests check every order. Adjusted p-values come from the procedure's closed-testing representation: H_i's adjusted p
is the maximum over intersections J containing i of min_{j in J} p_j / w_j(J), where w_j(J) are the weights left on
J after removing every hypothesis outside J by the same update (the graph's intersection weights).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

TOL = 1e-12


@dataclass(frozen=True)
class Graph:
    names: tuple[str, ...]
    weights: tuple[float, ...]
    transitions: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        m = len(self.names)
        if len(set(self.names)) != m or len(self.weights) != m or len(self.transitions) != m:
            raise ValueError("names, weights and transitions must have the same length (unique names)")
        if any(w < -TOL for w in self.weights) or sum(self.weights) > 1 + 1e-9:
            raise ValueError("weights must be >= 0 and sum to at most 1")
        for j, row in enumerate(self.transitions):
            if len(row) != m:
                raise ValueError("the transition matrix must be square")
            if abs(row[j]) > TOL:
                raise ValueError("G[j][j] must be 0")
            if any(g < -TOL for g in row) or sum(row) > 1 + 1e-9:
                raise ValueError("transition rows must be >= 0 and sum to at most 1")

    def index(self, name: str) -> int:
        return self.names.index(name)


def _remove(w: dict[int, float], g: dict[int, dict[int, float]], j: int) -> None:
    """Algorithm 1's update for removing (rejecting) H_j, in place."""
    rest = [l for l in w if l != j]
    wj = w[j]
    for l in rest:
        w[l] = w[l] + wj * g[j][l]
    new = {l: {} for l in rest}
    for l in rest:
        for k in rest:
            if l == k:
                new[l][k] = 0.0
                continue
            den = 1.0 - g[l][j] * g[j][l]
            new[l][k] = (g[l][k] + g[l][j] * g[j][k]) / den if den > TOL else 0.0
    del w[j]
    g.clear()
    g.update(new)


def _state(graph: Graph):
    m = len(graph.names)
    w = {i: float(graph.weights[i]) for i in range(m)}
    g = {i: {k: float(graph.transitions[i][k]) for k in range(m)} for i in range(m)}
    return w, g


@dataclass
class Result:
    rejected: list[str]                       # in the order they were rejected
    steps: list[dict] = field(default_factory=list)   # per rejection: the hypothesis, its p, level, weights after

    def is_rejected(self, name: str) -> bool:
        return name in self.rejected


def reject(graph: Graph, p: dict[str, float | None], alpha: float, order: list[str] | None = None) -> Result:
    """Run the procedure. A hypothesis with p None (no data) is never rejected but keeps its place in
    the graph (its weight is not passed on). `order` (tests only): a priority among rejectable hypotheses."""
    w, g = _state(graph)
    names = graph.names
    pri = [graph.index(n) for n in order] if order is not None else list(range(len(names)))
    res = Result(rejected=[])
    while True:
        cand = [j for j in pri if j in w and p.get(names[j]) is not None and p[names[j]] <= w[j] * alpha + TOL
                and w[j] > TOL]
        if not cand:
            return res
        j = cand[0]
        level = w[j] * alpha
        _remove(w, g, j)
        res.rejected.append(names[j])
        res.steps.append({"rejected": names[j], "p": p[names[j]], "level": level,
                          "weights_after": {names[i]: w[i] for i in sorted(w)}})


def intersection_weights(graph: Graph, subset: frozenset[int]) -> dict[int, float]:
    """w_j(J): the weights left on J after removing every hypothesis outside J (order does not matter)."""
    w, g = _state(graph)
    for j in sorted(set(w) - set(subset)):
        _remove(w, g, j)
    return w


def adjusted_p(graph: Graph, p: dict[str, float | None]) -> dict[str, float | None]:
    """Closed-testing adjusted p-values (H_i rejected at alpha iff adjusted p_i <= alpha). None for a hypothesis
    without a p-value; hypotheses without a p-value are treated as p = 1 inside intersections."""
    m = len(graph.names)
    pv = [1.0 if p.get(n) is None else float(p[n]) for n in graph.names]
    out: dict[str, float | None] = {}
    inter = {}
    for r in range(1, m + 1):
        for J in itertools.combinations(range(m), r):
            wj = intersection_weights(graph, frozenset(J))
            vals = [pv[j] / wj[j] for j in J if wj[j] > TOL]
            inter[J] = min(vals) if vals else float("inf")
    for i, n in enumerate(graph.names):
        if p.get(n) is None:
            out[n] = None
            continue
        out[n] = min(1.0, max(v for J, v in inter.items() if i in J))
    return out


def adjusted_p_sequential(graph: Graph, p: dict[str, float | None]) -> dict[str, float | None]:
    """BMBP09 Algorithm 2 (section 3.1): sequential adjusted p-values. Used as an independent check of
    `adjusted_p` (closed test); both must agree. A hypothesis without a p-value is treated as p = 1 and reported
    as None."""
    w, g = _state(graph)
    pv = {i: (1.0 if p.get(n) is None else float(p[n])) for i, n in enumerate(graph.names)}
    pmax, out = 0.0, {}
    while w:
        ratios = {i: (pv[i] / w[i] if w[i] > TOL else float("inf")) for i in w}
        j = min(ratios, key=lambda i: (ratios[i], i))
        pmax = max(ratios[j], pmax)
        out[j] = min(1.0, pmax)
        _remove(w, g, j)
    return {n: (None if p.get(n) is None else out[i]) for i, n in enumerate(graph.names)}


# ------------------------------------------------------------------------------------- the pre-registered graphs
def rq1_graph() -> Graph:
    """Design section 8: NI1, S1, E2, NI3, S3, E4; alpha/4 on NI1, E2, NI3, E4."""
    names = ("NI1", "S1", "E2", "NI3", "S3", "E4")
    t = 1 / 3
    edges = {"NI1": {"S1": 1.0}, "NI3": {"S3": 1.0},
             "S1": {"E2": t, "NI3": t, "E4": t}, "S3": {"E4": t, "NI1": t, "E2": t},
             "E2": {"NI1": t, "NI3": t, "E4": t}, "E4": {"NI1": t, "NI3": t, "E2": t}}
    G = tuple(tuple(edges.get(a, {}).get(b, 0.0) for b in names) for a in names)
    return Graph(names, (0.25, 0.0, 0.25, 0.25, 0.0, 0.25), G)


def rq3_graph() -> Graph:
    """Design section 8: NI-R1, S-R1, R2; alpha/2 on NI-R1 and R2; NI-R1 -> S-R1 -> R2 -> NI-R1 (1 each)."""
    names = ("NI-R1", "S-R1", "R2")
    G = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    return Graph(names, (0.5, 0.0, 0.5), G)
