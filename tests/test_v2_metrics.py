"""Design v2 metrics on hand-built logs with known answers."""

from types import SimpleNamespace as NS

import pytest

from eem_acc.agent import AttemptRecord as A, CallRecord as C
from eem_acc.metrics import AnalysisConfig, _queue_mean, compute_metrics_v2, objective_J, recovery_time

L = 0.5


def gw(attempts=(), calls=(), open_calls=(), queue=((0.0, 0),)):
    return NS(attempts=list(attempts), calls=list(calls), open_calls={i: (t, r) for i, (t, r) in enumerate(open_calls)},
              queue_trace=list(queue), controller=NS(window=7), server=NS(max_inflight_seen=3))


def test_accounting_over_requested_calls_504_and_censoring():
    g = gw(attempts=[A(0.0, 200, 0.4, 4, 1), A(1.0, 200, 0.6, 4, 1), A(2.0, 504, 1.0, 4, 1), A(3.5, 429, 0.02, 4, 1),
                     A(4.0, 200, 0.3, 4, 1)],
           calls=[C(0.0, 0.4, 200, 1, 0.0), C(1.0, 1.6, 200, 1, 0.0), C(2.0, 4.3, 200, 3, 0.0)],
           open_calls=[(8.0, (0, 9)), (9.0, (1, 9))])
    m = compute_metrics_v2(g, L, horizon=10.0)
    assert m["n_calls_requested"] == 5 and m["n_calls_ok"] == 3 and m["backlog_at_T"] == 2
    assert m["call_success_rate"] == pytest.approx(3 / 5)
    assert m["goodput"] == pytest.approx(0.3)
    assert m["throttle_rate"] == pytest.approx(2 / 5)                      # 504 + 429
    assert m["slo_violation_rate"] == pytest.approx(2 / 4)                 # 0.6 s success + the 504, of 3 ok + 1 504
    assert m["n_504"] == 1 and m["wasted_attempts_per_ok_call"] == pytest.approx(2 / 3)
    assert sorted([0.4, 0.6, 2.3, 2.0, 1.0])[2] == pytest.approx(m["e2e_p50_censored"])   # censored e2e: 2.0 and 1.0
    J = objective_J(m, 0.5, 0.5)
    assert J == pytest.approx(0.6 - 0.5 * m["mean_latency_excess_all_attempts"] - 0.5 * 0.4)


def test_queue_time_weighted_mean():
    trace = [(0.0, 0), (1.0, 4), (3.0, 2), (6.0, 0)]
    assert _queue_mean(trace, 0.0, 10.0) == pytest.approx((4 * 2 + 2 * 3) / 10)
    assert _queue_mean(trace, 2.0, 4.0) == pytest.approx((4 * 1 + 2 * 1) / 2)
    assert _queue_mean(trace, 7.0, 9.0) == 0.0
    assert _queue_mean(trace, 5.0, 5.0) is None


def test_window_membership_by_dispatch_and_request_time():
    g = gw(attempts=[A(1.99, 200, 0.9, 4, 1), A(2.0, 200, 0.9, 4, 1), A(3.99, 504, 1.0, 4, 1), A(4.0, 200, 0.1, 4, 1)],
           calls=[C(1.99, 3.0, 200, 1, 0.0), C(2.0, 3.0, 200, 1, 0.0), C(3.99, 6.0, 200, 2, 0.0)])
    m = compute_metrics_v2(g, L, 10.0, AnalysisConfig(windows=(("w", 2.0, 4.0),)))
    w = m["windows"]["w"]
    assert w["n_attempts"] == 2 and w["n_calls"] == 2
    assert w["slo_violation_rate"] == 1.0 and w["throttle_rate"] == 0.5
    assert w["goodput"] == pytest.approx(2 / 2.0)              # calls COMPLETED in [2, 4): t_done 3.0, 3.0
    assert w["arrival_rate"] == pytest.approx(2 / 2.0)         # requested in [2, 4): 2.0, 3.99


def _steady(t0, t1, rate, lat, status=200):
    """`rate` calls/s, one attempt each, latency `lat`."""
    n = int(round((t1 - t0) * rate))
    ts = [t0 + i / rate for i in range(n)]
    return ([A(t, status, lat, 8, 1) for t in ts], [C(t, t + lat, status, 1, 0.0) for t in ts])


def build(segments):
    att, calls = [], []
    for seg in segments:
        a, c = _steady(*seg)
        att += a
        calls += c
    return gw(att, calls)


CFG = AnalysisConfig(event=(40.0, 80.0), pre=(20.0, 40.0))


def _rec(g, T=120.0):
    from eem_acc.metrics import _calls_with_censoring
    return recovery_time(g, _calls_with_censoring(g, T), L, T, CFG)


def test_recovery_no_degradation():
    assert _rec(build([(0, 120, 10, 0.2)]))["status"] == "no_degradation"


def test_recovery_time_exact():
    # slow (latency 0.9 > L) during the event and until 90 s, then good again
    g = build([(0, 40, 10, 0.2), (40, 90, 10, 0.9), (90, 120, 10, 0.2)])
    r = _rec(g)
    # trailing window (t-5, t] is all-good from t = 95 (window (90, 95]); 5 consecutive points 95..99
    assert r == {"status": "recovered", "time": pytest.approx(15.0)}


def test_recovery_by_goodput_drop():
    g = build([(0, 40, 10, 0.2), (40, 100, 2, 0.2), (100, 120, 10, 0.2)])
    assert _rec(g) == {"status": "recovered", "time": pytest.approx(25.0)}


def test_recovery_censored_and_no_reference():
    g = build([(0, 40, 10, 0.2), (40, 120, 10, 0.9)])
    assert _rec(g) == {"status": "censored", "time": pytest.approx(40.0)}
    g2 = build([(0, 20, 10, 0.2), (40, 120, 10, 0.2)])          # nothing in the pre window
    assert _rec(g2)["status"] == "no_reference"


def test_recovery_judges_goodput_only_when_few_successes():
    # (2026-11-14: latency step changed from 0.9 to 0.6 s when goodput became completion-based: a 0.7-s step really
    # delays completions enough to dip below 90%, which is correct behaviour, not what this test is about)
    # 3 calls/s -> 15 successes per 5-s window (< 20): latency (0.6 > L = 0.5) is ignored; completion-time goodput
    # stays >= 90% of the pre window (the 0.4-s latency step shifts at most one completion out of a window)
    g = build([(0, 40, 3, 0.2), (40, 80, 3, 0.6), (80, 120, 3, 0.2)])
    assert _rec(g)["status"] == "no_degradation"


def test_recovery_reference_is_the_methods_own_pre_window():
    # a method that is already slow before the event is judged against its own pre-window p95 (0.9), not L
    g = build([(0, 120, 10, 0.9)])
    assert _rec(g)["status"] == "no_degradation"


def test_success_population_excludes_the_last_grace_seconds():
    g = gw(calls=[C(1.0, 2.0, 200, 1, 0.0), C(85.0, 86.0, 200, 1, 0.0)],
           open_calls=[(50.0, (0, 1)), (95.0, (1, 1)), (99.5, (2, 1))])
    m = compute_metrics_v2(g, L, 100.0, AnalysisConfig(success_grace=10.0))
    assert m["n_calls_success_population"] == 3                 # requested before 90: t=1, 85, 50
    assert m["call_success_rate"] == pytest.approx(2 / 3)       # the 50-s call never finished -> failure
    assert m["call_success_rate_all_requested"] == pytest.approx(2 / 5)
    assert m["backlog_at_T"] == 3 and m["backlog_older_than_grace"] == 1



def test_window_goodput_counts_completions_not_requests():
    # 10 calls requested in [0, 1) that all complete in [5, 6): throughput in [0, 2) is 0, in [4, 6) it is 10/2
    g = gw(calls=[C(0.1 * i, 5.0 + 0.1 * i, 200, 1, 0.0) for i in range(10)])
    m = compute_metrics_v2(g, L, 10.0, AnalysisConfig(windows=(("a", 0.0, 2.0), ("b", 4.0, 6.0))))
    assert m["windows"]["a"]["goodput"] == 0.0 and m["windows"]["a"]["arrival_rate"] == pytest.approx(5.0)
    assert m["windows"]["b"]["goodput"] == pytest.approx(5.0)


def test_recovery_sees_damage_that_starts_after_the_event():
    g = build([(0, 80, 10, 0.2), (80, 95, 10, 0.9), (95, 120, 10, 0.2)])
    # bad windows from t=81 until (t-5, t] no longer contains a 0.9-s attempt dispatched before 95 -> good from t=100
    assert _rec(g) == {"status": "recovered", "time": pytest.approx(20.0)}



def test_recovery_judges_latency_when_enough_successes():
    # same shape at 10 calls/s (50 successes per window): the 0.6-s latency is judged -> degradation, then recovery
    g = build([(0, 40, 10, 0.2), (40, 80, 10, 0.6), (80, 120, 10, 0.2)])
    assert _rec(g) == {"status": "recovered", "time": pytest.approx(5.0)}


def _queued(segs):
    """(t0, t1, rate, server latency, e2e): calls whose end-to-end time includes client-side queueing."""
    att, calls = [], []
    for t0, t1, rate, lat, e2e in segs:
        for i in range(int(round((t1 - t0) * rate))):
            t = t0 + i / rate
            att.append(A(t + e2e - lat, 200, lat, 8, 1))
            calls.append(C(t, t + e2e, 200, 1, e2e - lat))
    return gw(att, calls)


def test_recovery_e2e_condition_sees_client_side_queueing():
    """Server latency stays 0.2 s, but calls requested in [40, 100) wait 2.8 s in the client queue (e2e 3 s).
    Server-side signals recover at the event end; with the e2e condition (2 x pre e2e p95 = 0.4 s) recovery
    comes only when calls requested in the trailing window are fast again: t = 105 -> 25 s after the end."""
    from eem_acc.metrics import _calls_with_censoring
    g = _queued([(0, 40, 10, 0.2, 0.2), (40, 100, 10, 0.2, 3.0), (100, 120, 10, 0.2, 0.2)])
    calls = _calls_with_censoring(g, 120.0)
    base = dict(event=(40.0, 80.0), pre=(20.0, 40.0))
    assert recovery_time(g, calls, L, 120.0, AnalysisConfig(**base))["time"] == pytest.approx(0.0)
    assert recovery_time(g, calls, L, 120.0, AnalysisConfig(**base, recovery_e2e_factor=2.0)) == \
        {"status": "recovered", "time": pytest.approx(25.0)}


def test_recovery_e2e_window_without_requests_is_bad():
    """Calls requested in [70, 80) wait 10 s (they complete in [80, 90), so goodput and server latency look fine there);
    nobody requests anything in [80, 90). Windows with no requested call must count as bad: recovery only when fast
    requests (from t = 90) enter the trailing window (t - 5, t], closed on the right: t = 90 -> 10 s after the event
    end. (If empty windows counted as good, recovery would be reported at t = 85.)"""
    from eem_acc.metrics import _calls_with_censoring
    g = _queued([(0, 70, 10, 0.2, 0.2), (70, 80, 10, 0.2, 10.0), (90, 120, 10, 0.2, 0.2)])
    r = recovery_time(g, _calls_with_censoring(g, 120.0), L, 120.0,
                      AnalysisConfig(event=(40.0, 80.0), pre=(20.0, 40.0), recovery_e2e_factor=2.0))
    assert r == {"status": "recovered", "time": pytest.approx(10.0)}
