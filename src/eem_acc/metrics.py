"""Infrastructure metrics from a finished simulation (RQ2).

Latency metrics use *successful* attempts' server-side latency (the quantity the
SLO and the ACC refer to). End-to-end metrics include queueing and retries.
The combined objective J (Eq. 1 of the paper) needs lambda/gamma (undecided), so only its
components are stored; `objective_J` combines them exactly as Eq. 1:
    J = call_success_rate - lambda * mean_latency_excess_all_attempts - gamma * throttle_rate
where the latency excess averages max(0, (L_j - L_target)/L_target) over ALL attempts j (a 429/503
contributes its short error latency, i.e. ~0 excess), as in the paper. `mean_latency_excess` is the
variant over successful attempts only (it does not dilute under heavy throttling); report both and say which is J.
`slo_violation_rate` and lat_p* use successful attempts only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agent.gateway import Gateway


def _pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if x else None


def compute_metrics(gw: Gateway, l_target: float, horizon: float | None = None,
                    analysis: "AnalysisConfig | None" = None) -> dict[str, Any]:
    """v1 (horizon None): as before. v2 (horizon T): see `compute_metrics_v2`, whose keys are added on top."""
    if horizon is not None:
        return compute_metrics_v2(gw, l_target, horizon, analysis)
    att, calls = gw.attempts, gw.calls
    n_att = len(att)
    n200 = sum(a.status == 200 for a in att)
    n429 = sum(a.status == 429 for a in att)
    n503 = sum(a.status == 503 for a in att)
    n_to = sum(a.status == CLIENT_TIMEOUT for a in att)     # never > 0 in v1 runs (no client timeout)
    lat = [a.latency for a in att if a.status == 200]
    e2e = [c.t_done - c.t_request for c in calls]
    ok_calls = [c for c in calls if c.ok]
    makespan = (max(c.t_done for c in calls) - min(c.t_request for c in calls)) if calls else 0.0
    w = [a.window for a in att]
    excess_all = [max(0.0, (a.latency - l_target) / l_target) for a in att]
    return {
        "n_calls": len(calls),
        "n_calls_ok": len(ok_calls),
        "call_success_rate": len(ok_calls) / len(calls) if calls else None,
        "n_exhausted_calls": len(calls) - len(ok_calls),
        "n_attempts": n_att,
        "n_200": n200, "n_429": n429, "n_503": n503,
        "throttle_rate": (n429 + n503 + n_to) / n_att if n_att else None,
        "attempts_per_call": n_att / len(calls) if calls else None,
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "slo_violation_rate": (sum(x > l_target for x in lat) / len(lat)) if lat else None,
        "mean_latency_excess": float(np.mean([max(0.0, (x - l_target) / l_target) for x in lat])) if lat else None,
        "mean_latency_excess_all_attempts": float(np.mean(excess_all)) if excess_all else None,   # J's definition
        "wasted_attempts_per_ok_call": (n_att - len(ok_calls)) / len(ok_calls) if ok_calls else None,
        "e2e_p50": _pct(e2e, 50), "e2e_p95": _pct(e2e, 95), "e2e_p99": _pct(e2e, 99),
        "mean_queue_wait": float(np.mean([c.queue_wait for c in calls])) if calls else None,
        "makespan": makespan,
        "goodput": len(ok_calls) / makespan if makespan > 0 else None,
        "window_mean": float(np.mean(w)) if w else None,
        "window_min": min(w) if w else None,
        "window_max": max(w) if w else None,
        "window_final": gw.controller.window,
        "server_max_inflight": gw.server.max_inflight_seen,
    }


def objective_J(m: dict[str, Any], lam: float, gam: float) -> float | None:
    """Eq. (1) of the paper from stored metrics (None if a component is undefined)."""
    parts = (m.get("call_success_rate"), m.get("mean_latency_excess_all_attempts"), m.get("throttle_rate"))
    if any(p is None for p in parts):
        return None
    return parts[0] - lam * parts[1] - gam * parts[2]



# ======================================================================================= design v2 (D21)
from .status import CLIENT_TIMEOUT, OVERLOAD


@dataclass(frozen=True)
class AnalysisConfig:
    """Event windows of a v2 scenario. `windows`: ((name, t0, t1), ...). `event`: (start, end) of the disturbance,
    `pre`: (t0, t1) of the reference window for recovery (None = no event)."""
    windows: tuple = ()
    event: tuple | None = None
    pre: tuple | None = None
    recovery_span: float = 5.0        # trailing window (s)
    recovery_grid: float = 1.0        # grid step (s)
    recovery_hold: int = 5            # consecutive good grid points
    recovery_min_ok: int = 20         # fewer successes in a window -> judge goodput only
    recovery_goodput_frac: float = 0.9
    success_grace: float = 0.0        # call success is measured over calls requested before T - grace (see below)
    recovery_e2e_factor: float | None = None   # if set: recovery also needs e2e p95 <= factor x pre-window e2e p95


def _slo_bad(a, l_target: float) -> bool:
    return a.status == CLIENT_TIMEOUT or (a.status == 200 and a.latency > l_target)


def _calls_with_censoring(gw: Gateway, horizon: float):
    """[(t_request, ok, e2e, finished, t_done)] for every call REQUESTED before T; unfinished: e2e censored at T,
    t_done None."""
    out = [(c.t_request, c.ok, c.t_done - c.t_request, True, c.t_done) for c in gw.calls]
    out += [(t, False, horizon - t, False, None) for t, _ in gw.open_calls.values()]
    return out


def _queue_mean(trace: list[tuple[float, int]], t0: float, t1: float) -> float | None:
    """Time-weighted mean of a step function given as [(t, value)] (value holds from t on) over [t0, t1)."""
    if t1 <= t0:
        return None
    area, cur, last = 0.0, 0, t0
    for t, v in trace:
        if t <= t0:
            cur = v
            continue
        if t >= t1:
            break
        area += cur * (t - last)
        cur, last = v, t
    area += cur * (t1 - last)
    return area / (t1 - t0)


def _completed_ok(calls, t0: float, t1: float, closed_right: bool = False) -> int:
    if closed_right:
        return sum(1 for c in calls if c[1] and t0 < c[4] <= t1)
    return sum(1 for c in calls if c[1] and t0 <= c[4] < t1)


def _window_stats(gw: Gateway, calls, l_target: float, t0: float, t1: float) -> dict[str, Any]:
    """Goodput = successful calls COMPLETED in the window / its length (throughput, as the headline goodput);
    e2e and arrivals use calls REQUESTED in the window; attempt metrics use attempts DISPATCHED in the window."""
    att = [a for a in gw.attempts if t0 <= a.t_dispatch < t1]
    cw = [c for c in calls if t0 <= c[0] < t1]
    judged = [a for a in att if a.status in (200, CLIENT_TIMEOUT)]
    e2e = [c[2] for c in cw]
    return {
        "goodput": _completed_ok(calls, t0, t1) / (t1 - t0),
        "arrival_rate": len(cw) / (t1 - t0),
        "slo_violation_rate": (sum(_slo_bad(a, l_target) for a in judged) / len(judged)) if judged else None,
        "throttle_rate": (sum(a.status in OVERLOAD for a in att) / len(att)) if att else None,
        "e2e_p95_censored": _pct(e2e, 95),
        "queue_mean": _queue_mean(gw.queue_trace, t0, t1),
        "n_calls": len(cw), "n_attempts": len(att),
    }


def recovery_time(gw: Gateway, calls, l_target: float, horizon: float, cfg: AnalysisConfig) -> dict[str, Any]:
    """Section 7 of docs/rq2_design_v2.md. Trailing windows (t - span, t] on a grid; good = p95 latency of successful
    attempts dispatched in the window <= max(L_SLO, own pre-window p95) [skipped if < min_ok successes] AND goodput
    (successful calls COMPLETED in the window / span) >= frac x own pre-window goodput. Grid points run from
    event start + span to T. Recovered at the first grid point t* >= event end from which `hold` consecutive points
    are good; time = t* - event end.
    status: 'no_degradation' (no bad point between event start + span and t*: damage AFTER the event counts too) |
    'recovered' | 'censored' (never `hold` good points after the end; time = lower bound T - end) | 'no_reference'
    (no success in the pre window). The trailing window adds up to `span` s of lag to every recovery time.
    If `recovery_e2e_factor` is set (D26), a point is also bad unless the e2e p95 of calls REQUESTED in the window
    (unfinished calls censored at T) is <= factor x the pre-window e2e p95; a window with no requested call is bad
    (every agent is stuck)."""
    (e0, e1), (p0, p1) = cfg.event, cfg.pre
    pre_ok = [a.latency for a in gw.attempts if p0 <= a.t_dispatch < p1 and a.status == 200]
    pre_gp = _completed_ok(calls, p0, p1) / (p1 - p0)
    if not pre_ok or pre_gp <= 0:
        return {"status": "no_reference", "time": None}
    lat_bound = max(l_target, float(np.percentile(pre_ok, 95)))
    span = cfg.recovery_span
    e2e_bound = None
    if cfg.recovery_e2e_factor is not None:
        pre_e2e = [c[2] for c in calls if p0 <= c[0] < p1]
        if not pre_e2e:
            return {"status": "no_reference", "time": None}
        e2e_bound = cfg.recovery_e2e_factor * float(np.percentile(pre_e2e, 95))

    def good(t: float) -> bool:
        lo = t - span
        ok_lat = [a.latency for a in gw.attempts if lo < a.t_dispatch <= t and a.status == 200]
        gp = _completed_ok(calls, lo, t, closed_right=True) / span
        if gp < cfg.recovery_goodput_frac * pre_gp:
            return False
        if len(ok_lat) >= cfg.recovery_min_ok and float(np.percentile(ok_lat, 95)) > lat_bound:
            return False
        if e2e_bound is not None:
            e2e = [c[2] for c in calls if lo < c[0] <= t]
            if not e2e or float(np.percentile(e2e, 95)) > e2e_bound:
                return False
        return True

    step = cfg.recovery_grid
    grid = [float(t) for t in np.arange(e0 + span, horizon + 1e-9, step)]
    flags = [good(t) for t in grid]
    for i, t in enumerate(grid):
        if t < e1 - 1e-9 or i + cfg.recovery_hold > len(grid):
            continue
        if all(flags[i:i + cfg.recovery_hold]):
            if all(flags[:i]):
                return {"status": "no_degradation", "time": None}
            return {"status": "recovered", "time": t - e1}
    return {"status": "censored", "time": horizon - e1}


def compute_metrics_v2(gw: Gateway, l_target: float, horizon: float,
                       analysis: AnalysisConfig | None = None) -> dict[str, Any]:
    """Design v2 accounting (D21): success population = calls requested before T - grace (unfinished = not successful);
    attempts =
    those completed by T (the run stops at T, so in-flight attempts have no outcome); a client timeout is an
    overload signal (throttle rate, f_j of Eq. 1) and an SLO violation."""
    att = gw.attempts
    calls = _calls_with_censoring(gw, horizon)
    n_req = len(calls)
    n_ok = sum(c[1] for c in calls)
    n_att = len(att)
    cnt = {s: sum(a.status == s for a in att) for s in (200, 429, 503, CLIENT_TIMEOUT)}
    judged = [a for a in att if a.status in (200, CLIENT_TIMEOUT)]
    lat = [a.latency for a in att if a.status == 200]
    w = [a.window for a in att]
    excess_all = [max(0.0, (a.latency - l_target) / l_target) for a in att]
    grace = analysis.success_grace if analysis is not None else 0.0
    pop = [c for c in calls if c[0] < horizon - grace]      # every call in `pop` had >= grace s to finish
    out = {
        "n_calls_requested": n_req, "n_calls_ok": n_ok, "n_calls_finished": len(gw.calls),
        "backlog_at_T": len(gw.open_calls),
        "backlog_older_than_grace": sum(1 for t, _ in gw.open_calls.values() if t < horizon - grace),
        "success_grace": grace, "n_calls_success_population": len(pop),
        # closed-loop agents almost always have a call running at T; counting those as failures would cap
        # success at ~1 - M/requested for EVERY method, so the population excludes the last `grace` seconds
        "call_success_rate": (sum(c[1] for c in pop) / len(pop)) if pop else None,
        "call_success_rate_all_requested": n_ok / n_req if n_req else None,
        "goodput": n_ok / horizon,
        "n_attempts": n_att, "n_200": cnt[200], "n_429": cnt[429], "n_503": cnt[503], "n_timeout": cnt[CLIENT_TIMEOUT],
        # "overload rate" in the paper (429 + 503 + client timeouts per attempt); key kept for compatibility
        "throttle_rate": (cnt[429] + cnt[503] + cnt[CLIENT_TIMEOUT]) / n_att if n_att else None,
        "attempts_per_call": n_att / n_req if n_req else None,
        "wasted_attempts_per_ok_call": (n_att - n_ok) / n_ok if n_ok else None,
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "slo_violation_rate": (sum(_slo_bad(a, l_target) for a in judged) / len(judged)) if judged else None,
        "mean_latency_excess": float(np.mean([max(0.0, (x - l_target) / l_target) for x in lat])) if lat else None,
        "mean_latency_excess_all_attempts": float(np.mean(excess_all)) if excess_all else None,
        "e2e_p50_censored": _pct([c[2] for c in calls], 50), "e2e_p95_censored": _pct([c[2] for c in calls], 95),
        "queue_mean": _queue_mean(gw.queue_trace, 0.0, horizon),
        "queue_max": max(v for _, v in gw.queue_trace),
        "window_mean": float(np.mean(w)) if w else None,
        "window_min": min(w) if w else None, "window_max": max(w) if w else None,
        "window_final": gw.controller.window,
        "server_max_inflight": gw.server.max_inflight_seen,
        "horizon": horizon,
    }
    if analysis is not None:
        out["windows"] = {name: _window_stats(gw, calls, l_target, t0, t1) for name, t0, t1 in analysis.windows}
        if analysis.event is not None:
            out["recovery"] = recovery_time(gw, calls, l_target, horizon, analysis)
    return out
