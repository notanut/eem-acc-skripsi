"""Adaptive Concurrency Controller (AIMD), implementing revised Section III-C.

Once every `n_c` completed attempts (one control interval) the window W is updated
from that interval's attempts only:

    F   = fraction of attempts that returned 429 or 503
    dL  = (p95 server latency of the interval's *successful* attempts - L_target) / L_target
          (0 if the interval has no successful attempt)

    if use_throttle_branch and F > 0:          W <- max(W_min, floor(W * (1 - beta * F)))
    elif use_latency_branch and dL > epsilon:  W <- max(W_min, floor(W * (1 - alpha * dL)))
    elif (not use_latency_branch) or dL <= 0:  W <- min(W + 1, W_max)
    else:                                      hold   (tolerance band 0 < dL <= epsilon)

IMPLEMENTATION ASSUMPTION (not in the paper text, flagged in HANDOFF.md): the latency
percentile uses successful (HTTP 200) attempts only, because 429/503 responses carry
no service time.

Config flags exist so the same class also implements the ablations/baselines
(no-throttle-branch, no-latency-branch, additive-only, fixed window via `frozen`).
There are NO default values for the hyperparameters: they are experimental decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


OVERLOAD_STATUSES = (429, 503, 504)   # 504 = client timeout (D21): an overload signal like 429/503
_FLOOR_EPS = 1e-9   # guards floor() against float error, e.g. 12*(1-0.5*25/30) = 6.999999999999999 (exactly 7)


@dataclass(frozen=True)
class ACCConfig:
    alpha: float
    beta: float
    epsilon: float
    w0: int
    w_min: int
    w_max: int
    n_c: int
    l_target: float
    use_throttle_branch: bool = True
    use_latency_branch: bool = True
    multiplicative_decrease: bool = True   # False -> "additive-only" ablation (never decreases)
    frozen: bool = False                   # True -> fixed window baseline (W stays at w0)
    kappa: float = 1.0                     # controller target L_ctrl = kappa * l_target (l_target = the SLO); 1 = paper's base

    def __post_init__(self) -> None:
        if not (1 <= self.w_min <= self.w0 <= self.w_max):
            raise ValueError("need 1 <= w_min <= w0 <= w_max")
        if self.n_c < 1:
            raise ValueError("n_c must be >= 1")
        if self.l_target <= 0:
            raise ValueError("l_target must be > 0")
        if not (0 < self.kappa <= 1):
            raise ValueError("kappa must be in (0, 1]")

    @property
    def l_ctrl(self) -> float:
        return self.kappa * self.l_target


class ACCController:
    def __init__(self, cfg: ACCConfig):
        self.cfg = cfg
        self.window = int(cfg.w0)
        self._buf: list[tuple[int, float]] = []   # (status, server latency) of current interval
        self.n_updates = 0
        self.trace: list[tuple[float, int, float, float]] = []  # (t, W after update, F, dL)

    # -- signals -------------------------------------------------------------
    @staticmethod
    def throttle_rate(buf: list[tuple[int, float]]) -> float:
        return sum(1 for s, _ in buf if s in OVERLOAD_STATUSES) / len(buf)

    def latency_degradation(self, buf: list[tuple[int, float]]) -> float:
        ok = [lat for s, lat in buf if s == 200]
        if not ok:
            return 0.0
        p95 = float(np.percentile(ok, 95))
        return (p95 - self.cfg.l_ctrl) / self.cfg.l_ctrl

    # -- update --------------------------------------------------------------
    def next_window(self, w: int, f: float, dl: float) -> int:
        c = self.cfg
        if c.use_throttle_branch and f > 0:
            if not c.multiplicative_decrease:
                return w
            return max(c.w_min, math.floor(w * (1.0 - c.beta * f) + _FLOOR_EPS))
        if c.use_latency_branch and dl > c.epsilon:
            if not c.multiplicative_decrease:
                return w
            return max(c.w_min, math.floor(w * (1.0 - c.alpha * dl) + _FLOOR_EPS))
        if (not c.use_latency_branch) or dl <= 0:
            return min(w + 1, c.w_max)
        return w

    def record(self, status: int, latency: float, now: float = 0.0) -> bool:
        """Feed one completed attempt. Returns True if the window was updated."""
        self._buf.append((status, float(latency)))
        if len(self._buf) < self.cfg.n_c:
            return False
        buf, self._buf = self._buf, []
        f = self.throttle_rate(buf)
        dl = self.latency_degradation(buf)
        if not self.cfg.frozen:
            self.window = self.next_window(self.window, f, dl)
        self.n_updates += 1
        self.trace.append((float(now), self.window, f, dl))
        return True

    # -- checkpointing -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"window": self.window, "buf": list(self._buf), "n_updates": self.n_updates,
                "trace": list(self.trace)}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.window = int(s["window"])
        self._buf = [tuple(x) for x in s["buf"]]
        self.n_updates = int(s["n_updates"])
        self.trace = [tuple(x) for x in s["trace"]]
