"""Infrastructure metrics from a finished simulation (RQ2).

Latency metrics use *successful* attempts' server-side latency (the quantity the
SLO and the ACC refer to). End-to-end metrics include queueing and retries.
The combined objective J needs lambda/gamma (undecided), so only its components
are reported here: success rate, mean_latency_excess, throttle_rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agent.gateway import Gateway


def _pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if x else None


def compute_metrics(gw: Gateway, l_target: float) -> dict[str, Any]:
    att, calls = gw.attempts, gw.calls
    n_att = len(att)
    n200 = sum(a.status == 200 for a in att)
    n429 = sum(a.status == 429 for a in att)
    n503 = sum(a.status == 503 for a in att)
    lat = [a.latency for a in att if a.status == 200]
    e2e = [c.t_done - c.t_request for c in calls]
    ok_calls = [c for c in calls if c.ok]
    makespan = (max(c.t_done for c in calls) - min(c.t_request for c in calls)) if calls else 0.0
    w = [a.window for a in att]
    return {
        "n_calls": len(calls),
        "n_calls_ok": len(ok_calls),
        "call_success_rate": len(ok_calls) / len(calls) if calls else None,
        "n_exhausted_calls": len(calls) - len(ok_calls),
        "n_attempts": n_att,
        "n_200": n200, "n_429": n429, "n_503": n503,
        "throttle_rate": (n429 + n503) / n_att if n_att else None,
        "attempts_per_call": n_att / len(calls) if calls else None,
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "slo_violation_rate": (sum(x > l_target for x in lat) / len(lat)) if lat else None,
        "mean_latency_excess": float(np.mean([max(0.0, (x - l_target) / l_target) for x in lat])) if lat else None,
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
