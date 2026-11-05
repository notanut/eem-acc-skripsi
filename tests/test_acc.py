"""ACC controller tests. All numeric values here are TEST-ONLY, not experimental choices."""

import pytest

from eem_acc.acc import ACCConfig, ACCController


def cfg(**kw):
    base = dict(alpha=0.5, beta=0.5, epsilon=0.1, w0=10, w_min=1, w_max=20, n_c=4, l_target=1.0)
    base.update(kw)
    return ACCConfig(**base)


def feed(c, items):
    """items: list of (status, latency); returns number of updates triggered."""
    return sum(c.record(s, l, now=float(i)) for i, (s, l) in enumerate(items))


OK = (200, 0.5)       # healthy: latency below target
SLOW = (200, 2.0)     # dL = +1.0
BAND = (200, 1.05)    # dL = +0.05, inside tolerance band (eps = 0.1)


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(w0=0)
    with pytest.raises(ValueError):
        cfg(w0=30)
    with pytest.raises(ValueError):
        cfg(n_c=0)
    with pytest.raises(ValueError):
        cfg(l_target=0)


def test_updates_only_once_per_interval():
    c = ACCController(cfg(n_c=4))
    assert feed(c, [OK] * 3) == 0 and c.window == 10
    assert feed(c, [OK]) == 1 and c.window == 11
    assert c.n_updates == 1


def test_additive_increase_and_cap():
    c = ACCController(cfg(w0=19, w_max=20))
    feed(c, [OK] * 4); assert c.window == 20
    feed(c, [OK] * 4); assert c.window == 20     # capped at w_max


def test_throttle_decrease_multiplicative_floor():
    c = ACCController(cfg(beta=0.5))
    feed(c, [(429, 0.0), OK, OK, OK])             # F = 0.25 -> W = floor(10 * (1 - 0.125)) = 8
    assert c.window == 8


def test_503_counts_as_throttle():
    c = ACCController(cfg(beta=0.5))
    feed(c, [(503, 0.0), (503, 0.0), OK, OK])     # F = 0.5 -> floor(10 * 0.75) = 7
    assert c.window == 7


def test_latency_decrease():
    c = ACCController(cfg(alpha=0.25))
    feed(c, [SLOW] * 4)                           # dL = 1 -> floor(10 * 0.75) = 7
    assert c.window == 7


def test_tolerance_band_holds():
    c = ACCController(cfg())
    feed(c, [BAND] * 4)
    assert c.window == 10


def test_throttle_has_priority_over_latency():
    c = ACCController(cfg(alpha=0.9, beta=0.1))
    feed(c, [(429, 0.0), SLOW, SLOW, SLOW])       # F=0.25 -> beta branch: floor(10*(1-0.025)) = 9
    assert c.window == 9


def test_never_below_w_min():
    c = ACCController(cfg(w0=2, w_min=2, beta=1.0))
    feed(c, [(429, 0.0)] * 4)                     # F = 1 -> 0 -> clamped
    assert c.window == 2


def test_huge_latency_does_not_go_negative():
    c = ACCController(cfg(alpha=5.0, w0=5, w_min=1))
    feed(c, [(200, 100.0)] * 4)
    assert c.window == 1


def test_interval_without_success_has_zero_latency_signal():
    c = ACCController(cfg())
    assert c.latency_degradation([(429, 0.0)] * 4) == 0.0


def test_p95_ignores_error_latencies():
    c = ACCController(cfg())
    buf = [(200, 2.0), (429, 0.0), (429, 0.0), (429, 0.0)]
    assert c.latency_degradation(buf) == pytest.approx(1.0)


def test_ablation_no_throttle_branch_ignores_429():
    c = ACCController(cfg(use_throttle_branch=False))
    feed(c, [(429, 0.0), OK, OK, OK])             # F ignored, dL <= 0 -> increase
    assert c.window == 11


def test_ablation_no_latency_branch_ignores_slowness():
    c = ACCController(cfg(use_latency_branch=False))
    feed(c, [SLOW] * 4)                           # no 429s -> increase
    assert c.window == 11


def test_ablation_additive_only_never_decreases():
    c = ACCController(cfg(multiplicative_decrease=False))
    feed(c, [(429, 0.0), OK, OK, OK]); assert c.window == 10    # hold, no decrease
    feed(c, [OK] * 4); assert c.window == 11


def test_frozen_window_is_fixed():
    c = ACCController(cfg(frozen=True))
    feed(c, [(429, 0.0)] * 8 + [OK] * 8)
    assert c.window == 10 and c.n_updates == 4


def test_state_dict_roundtrip_mid_interval():
    a = ACCController(cfg())
    feed(a, [OK] * 4 + [(429, 0.0), OK])          # one update done, 2 attempts buffered
    b = ACCController(cfg())
    b.load_state_dict(a.state_dict())
    rest = [OK, OK]
    feed(a, rest); feed(b, rest)
    assert (a.window, a.n_updates, a.trace) == (b.window, b.n_updates, b.trace)
