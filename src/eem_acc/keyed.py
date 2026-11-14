"""Order-independent ("keyed") random draws for common random numbers (CRN).

`KeyedNormals(seed).get(agent, k)` returns the k-th standard-normal draw of agent `agent`'s own stream,
whatever order the (agent, k) pairs are requested in. Each agent's stream is one sequential numpy
Generator seeded by (seed, tag, agent) whose values are cached in a table that grows in blocks; this
gives the order independence of a per-draw keyed generator at O(1) amortised cost.
"""

from __future__ import annotations

import numpy as np

_BLOCK = 256


class _KeyedTable:
    def __init__(self, seed: int, tag: int, draw):
        self.seed, self.tag, self._draw = int(seed), int(tag), draw
        self._gens: dict[int, np.random.Generator] = {}
        self._tabs: dict[int, np.ndarray] = {}

    def get(self, agent: int, idx: int) -> float:
        if idx < 0 or agent < 0:
            raise ValueError("agent and index must be non-negative")
        tab = self._tabs.get(agent)
        if tab is None or idx >= len(tab):
            g = self._gens.setdefault(agent, np.random.default_rng([self.seed, self.tag, agent]))
            have = 0 if tab is None else len(tab)
            need = max(_BLOCK, (idx + 1 - have + _BLOCK - 1) // _BLOCK * _BLOCK)
            new = self._draw(g, need)
            tab = new if tab is None else np.concatenate([tab, new])
            self._tabs[agent] = tab
        return float(tab[idx])


class KeyedNormals(_KeyedTable):
    def __init__(self, seed: int, tag: int = 0):
        super().__init__(seed, tag, lambda g, n: g.standard_normal(n))


class KeyedUniforms(_KeyedTable):
    def __init__(self, seed: int, tag: int = 0):
        super().__init__(seed, tag, lambda g, n: g.random(n))
