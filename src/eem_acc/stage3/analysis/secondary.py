"""S3.5 spec sections 6-7: secondary analyses (outside the graphical procedure; estimates, intervals and unadjusted
p-values only). Every function takes the normalised units (data.UnitData) and the plan phase(s) to read."""
from __future__ import annotations

import math
import warnings
from collections import defaultdict

import numpy as np

from . import stats as st
from .data import PRIMARY_RQ1, UnitData, complete, index
from .endpoints import _n, _solved_n, api_diff, success_diff

T_PHASE = "s35-T"


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(float(v)) else float(v)
    return v


def summary(seeds: list[int], d: list[float | None], two_sided: bool = True) -> dict:
    """Per-seed paired differences -> n, mean, 95% t interval and (optionally) the exact two-sided sign-flip p."""
    keep = [(s, x) for s, x in zip(seeds, d) if x is not None]
    x = np.array([v for _, v in keep], dtype=float)
    out = {"n": len(x), "seeds": [s for s, _ in keep], "left_out": [s for s, v in zip(seeds, d) if v is None],
           "differences": x.tolist(), "mean": _clean(float(x.mean())) if len(x) else None,
           "ci95": [_clean(v) for v in st.t_ci(x)] if len(x) >= 2 else [None, None]}
    if two_sided:
        out["p_two_sided"] = st.sign_flip_two_sided(x) if len(x) else None
    return out


# ------------------------------------------------------------------------------------------------ RQ1 secondary
def c1_vs_c0(units: list[UnitData], phase=T_PHASE) -> dict:
    """Cross-task memory: C1 - C0 in success and in API calls over jointly solved tasks (two-sided, unadjusted)."""
    by = index(units, "rq1", "rq1", phase)
    seeds, _ = complete(by, PRIMARY_RQ1)
    return {"success": summary(seeds, [success_diff(by[s]["C1"], by[s]["C0"]) for s in seeds]),
            "api_calls": summary(seeds, [api_diff(by[s]["C1"], by[s]["C0"]) for s in seeds])}


def c3_vs_c2_tost(units: list[UnitData], phase=T_PHASE) -> dict:
    """TOST, margin one task per stream, one-sided 0.025 per side, from integer counts (as the NI tests):
    p_a tests mean((solved_C3 - solved_C2 + 1) / n) > 0, p_b tests mean((solved_C2 - solved_C3 + 1) / n) > 0.
    Equivalence at 0.025 per side iff max(p_a, p_b) <= 0.025; the 95% t interval inside +-1/n is the paired-t check."""
    by = index(units, "rq1", "rq1", phase)
    seeds, _ = complete(by, PRIMARY_RQ1)
    if not seeds:
        return {"n": 0, "p_tost": None, "equivalent": None}
    d = [success_diff(by[s]["C3"], by[s]["C2"]) for s in seeds]
    up = [(_solved_n(by[s]["C3"]) - _solved_n(by[s]["C2"]) + 1) / _n(by[s]["C3"], by[s]["C2"]) for s in seeds]
    dn = [(_solved_n(by[s]["C2"]) - _solved_n(by[s]["C3"]) + 1) / _n(by[s]["C3"], by[s]["C2"]) for s in seeds]
    p_a, p_b = st.sign_flip_one_sided(up), st.sign_flip_one_sided(dn)
    ci = st.t_ci(d) if len(d) >= 2 else (math.nan, math.nan)
    margin = 1.0 / _n(by[seeds[0]]["C3"], by[seeds[0]]["C2"])
    return {"n": len(seeds), "margin": margin, "mean": float(np.mean(d)), "p_lower": p_a, "p_upper": p_b,
            "p_tost": max(p_a, p_b), "equivalent": max(p_a, p_b) <= 0.025,
            "ci95": [_clean(ci[0]), _clean(ci[1])],
            "t_check_equivalent": bool(-margin < ci[0] and ci[1] < margin) if not math.isnan(ci[0]) else None}


def population(units: list[UnitData], condition: str, phase=T_PHASE) -> list[UnitData]:
    """The done, untagged RQ1 units of one condition. For C0-C3 only the complete streams (all four done), so the
    primary conditions are compared over the same streams (code review M5); other arms: all their done units."""
    phases = (phase,) if isinstance(phase, str) else tuple(phase)
    us = [u for u in units if u.rq == "rq1" and u.kind == "rq1" and u.condition == condition and u.status == "done"
          and (u.tag or "") == "" and u.phase in phases]
    if condition in PRIMARY_RQ1:
        seeds, _ = complete(index(units, "rq1", "rq1", phases), PRIMARY_RQ1)
        us = [u for u in us if u.seed in set(seeds)]
    return us


def cost_per_solved(units: list[UnitData], conditions, phase=T_PHASE) -> dict:
    """EEM's own cost: LLM calls and prompt tokens per solved task, per condition (pooled over its population)."""
    out = {}
    for c in conditions:
        us = population(units, c, phase)
        solved = sum(_solved_n(u) for u in us)
        llm = sum(t.get("llm_calls", 0) for u in us for t in u.tasks)
        tok = sum(t.get("tok_in", 0) for u in us for t in u.tasks)
        out[c] = {"units": len(us), "solved": solved, "llm_calls_per_solved": (llm / solved) if solved else None,
                  "tok_in_per_solved": (tok / solved) if solved else None}
    return out


def ratio_of_sums(units: list[UnitData], phase=T_PHASE, n_boot: int = 10_000, seed: int = 0) -> dict:
    """E2/E4 as ratios of pooled API-call sums over jointly solved tasks (C2/C0, C2/C1), stream bootstrap 95%."""
    by = index(units, "rq1", "rq1", phase)
    seeds, _ = complete(by, PRIMARY_RQ1)
    out = {}
    for comp in ("C0", "C1"):
        num, den = [], []
        for s in seeds:
            a = {t["pos"]: t for t in by[s]["C2"].tasks if t["success"]}
            b = {t["pos"]: t for t in by[s][comp].tasks if t["success"]}
            joint = set(a) & set(b)
            num.append(sum(a[p]["api_calls"] for p in joint))
            den.append(sum(b[p]["api_calls"] for p in joint))
        if not seeds or sum(den) == 0:
            out[f"C2/{comp}"] = {"n": len(seeds), "ratio": None, "ci95": [None, None]}
            continue
        est, lo, hi = st.bootstrap_ratio(num, den, n_boot=n_boot, seed=seed)
        out[f"C2/{comp}"] = {"n": len(seeds), "ratio": est, "ci95": [_clean(lo), _clean(hi)]}
    return out


def error_splits(units: list[UnitData], conditions, phase=T_PHASE) -> dict:
    """Per condition, per task means: format errors, budget exhaustion (MaxSteps), context overflow, drift and agent
    errors (design section 8)."""
    out = {}
    for c in conditions:
        ts = [t for u in population(units, c, phase) for t in u.tasks]
        n = len(ts)
        out[c] = {"tasks": n} | ({} if not n else {
            "format_errors": sum(t.get("format_errors", 0) for t in ts) / n,
            "budget_exhausted": sum(t.get("reason") == "max_steps" for t in ts) / n,
            "context_overflow": sum(t.get("reason") == "context_overflow" for t in ts) / n,
            "drift_errors": sum(t.get("drift_errors", 0) for t in ts) / n,
            "agent_errors": sum(t.get("agent_errors", 0) for t in ts) / n})
    return out


# ------------------------------------------------------------------------------------------------ stream structure
def active_pairs(spec: dict, task_index: int) -> list[str]:
    """Drift pair ids in force at a task (the last schedule entry starting at or before it)."""
    cur: list[str] = []
    for start, ids in sorted(spec["drift_schedule"], key=lambda e: e[0]):
        if start <= task_index:
            cur = list(ids)
    return cur


def op_pairs(spec: dict, task_index: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for pid in active_pairs(spec, task_index):
        op, _, typ = pid.partition(":")
        out[op].append(typ)
    return dict(out)


def exposures(units: list[UnitData], conditions, phase=T_PHASE) -> dict:
    """Per exposure (a task's first call to a drifted operation, `first_call`): the share whose first call
    succeeded, per condition, by drift type and by error style (review L6: type and style belong to an operation).
    An operation with two pairs counts under each of its types."""
    out = {}
    for c in conditions:
        by_type, by_style = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for u in population(units, c, phase):
            if not u.spec:
                continue
            for t in u.tasks:
                ops = op_pairs(u.spec, t["pos"])
                for op, kind in (t.get("first_call") or {}).items():
                    if op not in ops:
                        continue
                    ok = kind == "ok"
                    for typ in ops[op]:
                        by_type[typ][0] += ok
                        by_type[typ][1] += 1
                    style = u.spec["styles"].get(op)
                    by_style[style][0] += ok
                    by_style[style][1] += 1
        out[c] = {"by_drift_type": {k: {"n": v[1], "first_call_ok": v[0] / v[1]} for k, v in sorted(by_type.items())},
                  "by_style": {str(k): {"n": v[1], "first_call_ok": v[0] / v[1]} for k, v in sorted(by_style.items(),
                                                                                                key=lambda kv: str(kv[0]))}}
    return out


def drifted_call_share(units: list[UnitData], conditions, phase=T_PHASE) -> dict:
    """Share of all API calls (steps with a call counter) that hit an operation drifted at that task."""
    out = {}
    for c in conditions:
        hit = tot = 0
        for u in population(units, c, phase):
            if not u.spec:
                continue
            for s in u.raw.get("steps", []):
                if "k" not in s or s.get("action") == "get_api_docs":
                    continue
                tot += 1
                hit += s.get("canonical_op") in op_pairs(u.spec, s["task"])
        out[c] = {"calls": tot, "share_drifted": (hit / tot) if tot else None}
    return out


def learning_curves(units: list[UnitData], phase=T_PHASE) -> dict:
    """Success by position (0..n-1), per primary condition, over the complete streams."""
    by = index(units, "rq1", "rq1", phase)
    seeds, _ = complete(by, PRIMARY_RQ1)
    out = {}
    for c in PRIMARY_RQ1:
        if not seeds:
            out[c] = []
            continue
        n = len(by[seeds[0]][c].tasks)
        out[c] = [float(np.mean([any(t["pos"] == p and t["success"] for t in by[s][c].tasks) for s in seeds]))
                  for p in range(n)]
    return out


# ------------------------------------------------------------------------------------------------ mixed model
def mixed_model(units: list[UnitData], phase=T_PHASE) -> dict:
    """D187: success ~ condition x centred position, crossed random intercepts for stream and template (statsmodels
    BinomialBayesMixedGLM: VB, with the MAP fit as a check); GEE (logit, exchangeable, clustered by stream,
    bias-reduced covariance) with the same fixed effects. Descriptive; errors are reported, never raised."""
    import pandas as pd
    by = index(units, "rq1", "rq1", phase)
    seeds, _ = complete(by, PRIMARY_RQ1)
    rows = [{"success": int(t["success"]), "condition": c, "pos": t["pos"], "stream": s,
             "template": t.get("template", "?")} for s in seeds for c in PRIMARY_RQ1 for t in by[s][c].tasks]
    out: dict = {"n_streams": len(seeds), "n_rows": len(rows), "position_centre": None}
    if len(seeds) < 2:
        out["error"] = "fewer than 2 complete streams"
        out["quotable"] = {"quote_mixed_model": False, "reasons": [out["error"]]}
        return out
    df = pd.DataFrame(rows)
    for col in ("condition", "template"):          # pandas 3's string dtype is not understood by patsy 1.0.1
        df[col] = df[col].astype(object).astype("category")
    out["position_centre"] = float(df["pos"].mean())
    df["pos_c"] = df["pos"] - out["position_centre"]
    formula = "success ~ C(condition, Treatment('C0')) * pos_c"
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        m = BinomialBayesMixedGLM.from_formula(formula, {"stream": "0 + C(stream)", "template": "0 + C(template)"},
                                               df, fe_p=2.0, vcp_p=1.0)
        for name, fit in (("vb", m.fit_vb), ("map", m.fit_map)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = fit()
            sd = np.sqrt(np.diag(r.cov_params())) if name == "map" else r.fe_sd
            k = len(m.exog_names)
            out[name] = {"priors": {"fe_p": 2.0, "vcp_p": 1.0},
                         "fixed": {n: {"mean": float(r.params[i]), "lo95": float(r.params[i] - 1.96 * sd[i]),
                                       "hi95": float(r.params[i] + 1.96 * sd[i])}
                                   for i, n in enumerate(m.exog_names[:k])},
                         "warnings": sorted({str(w.message)[:200] for w in caught})}
        # the dry run showed VB and MAP can disagree badly on near-separated data (MAP's Laplace step then warns);
        # the gap is reported so a reader can judge whether the descriptive model is usable
        if "vb" in out and "map" in out:
            out["vb_map_max_abs_diff"] = max(abs(out["vb"]["fixed"][n]["mean"] - out["map"]["fixed"][n]["mean"])
                                             for n in out["vb"]["fixed"])
    except Exception as e:                                          # pragma: no cover - reported, not raised
        out["bayes_error"] = f"{type(e).__name__}: {e}"
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from scipy import stats as sps
        g = smf.gee(formula, groups="stream", data=df, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type="bias_reduced")
        q = float(sps.t.ppf(0.975, len(seeds) - 1))     # t with df = clusters - 1, not normal (review L4)
        out["gee"] = {"cov_type": g.cov_type, "interval": f"t, df = {len(seeds) - 1}",
                      "fixed": {n: {"mean": float(g.params[n]), "se": float(g.bse[n]),
                                    "lo95": float(g.params[n] - q * g.bse[n]),
                                    "hi95": float(g.params[n] + q * g.bse[n])} for n in g.params.index}}
    except Exception as e:                                          # pragma: no cover - reported, not raised
        out["gee_error"] = f"{type(e).__name__}: {e}"
    out["quotable"] = quotable(out, df)
    return out


GAP_MAX = 0.5
EXTREME = 0.02


def quotable(mm: dict, df) -> dict:
    """D208, frozen before any test data: Chapter IV quotes the mixed model only if the MAP fit converged with no
    warning, the largest VB-MAP gap is at most 0.5 logits, and no condition's success is >= 0.98 or <= 0.02.
    Otherwise GEE alone is reported, with the reasons."""
    reasons = []
    if "vb" not in mm or "map" not in mm:
        reasons.append("a Bayesian fit failed")
    else:
        if mm["map"]["warnings"]:
            reasons.append("the MAP fit warned: " + "; ".join(mm["map"]["warnings"]))
        gap = mm.get("vb_map_max_abs_diff")
        if gap is None or not (gap <= GAP_MAX):         # a NaN gap must not pass (delta review)
            reasons.append(f"VB-MAP gap {gap} is not <= {GAP_MAX} logits")
    rates = df.groupby("condition", observed=True)["success"].mean()
    for c, r in rates.items():
        if r >= 1 - EXTREME or r <= EXTREME:
            reasons.append(f"{c} success {r:.3f} is within {EXTREME} of 0 or 1")
    return {"quote_mixed_model": not reasons, "reasons": reasons,
            "rule": "D208: MAP converged without warning, VB-MAP gap <= 0.5 logits, every condition's success in (0.02, 0.98)"}


# ------------------------------------------------------------------------------------------------ RQ3 secondary
# D206: every RQ2 metric for both traffics, labelled. LLM traffic = the agents' calls (rq3 `llm_metrics`, capacity
# path, D122; J_rq3 = success - 0.5 latency excess - 0.5 overload); background = the scripted traffic
# (`infra.windows.llm_phase`, D117; its J is objective_J(0.5, 0.5)). LLM goodput is computed (llm_goodput).
RQ2_KEYS = (("llm_overload_rate", ("llm_metrics", "overload_rate")),
            ("llm_slo_violation_rate", ("llm_metrics", "slo_violation_rate")),
            ("llm_mean_latency_excess", ("llm_metrics", "mean_latency_excess")),
            ("llm_J", ("J_rq3",)),
            ("background_overload_rate", ("infra", "windows", "llm_phase", "throttle_rate")),
            ("background_slo_violation_rate", ("infra", "windows", "llm_phase", "slo_violation_rate")),
            ("background_goodput", ("infra", "windows", "llm_phase", "goodput")),
            ("background_J", ("infra", "windows", "llm_phase", "J")))


def llm_goodput(u: UnitData) -> float | None:
    """LLM-traffic goodput, defined as the background one (metrics._completed_ok): calls with status 200 completed
    in [warmup, t_end), per second (Part 1 second opinion M2: 4xx answers are not good calls)."""
    calls, t_end = u.raw.get("llm_calls"), u.raw.get("t_end")
    warm = (u.raw.get("stamp") or {}).get("warmup")
    if calls is None or t_end is None or warm is None or t_end <= warm:
        return None
    ok = sum(1 for c in calls if c.get("status") == 200 and c.get("t_done") is not None
             and warm <= c["t_done"] < t_end)
    return ok / (t_end - warm)


def _dig(d: dict, path: tuple):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def completion(u: UnitData) -> dict:
    """Spec section 6: mean completion time over finished tasks (unfinished have none), and the censored version
    (`completion_time` as the payload writes it, D120: never-started tasks excluded)."""
    fin = [t["completion_time"] for t in u.tasks if not t.get("completion_censored") and not t.get("never_started")
           and t.get("completion_time") is not None]
    cens = [t["completion_time"] for t in u.tasks if not t.get("never_started") and t.get("completion_time") is not None]
    return {"finished_mean": float(np.mean(fin)) if fin else None, "censored_mean": float(np.mean(cens)) if cens else None,
            # started but unfinished at T_max (never-started tasks are counted on their own, code review L3)
            "unfinished": sum(bool(t.get("unfinished_at_t_max")) and not t.get("never_started") for t in u.tasks),
            "never_started": sum(bool(t.get("never_started")) for t in u.tasks)}


def rq3_secondary(units: list[UnitData], phases=(T_PHASE,)) -> dict:
    by = index(units, "rq3", "rq3", phases)
    cells = sorted({c for s in by for c in by[s]})
    per_cell = {}
    for c in cells:
        us = [by[s][c] for s in sorted(by) if c in by[s] and by[s][c].status == "done"]
        comp = [completion(u) for u in us]
        per_cell[c] = {"seeds": len(us),
                       "success": float(np.mean([u.success_rate() for u in us])) if us else None,
                       "completion_finished_mean": _mean([x["finished_mean"] for x in comp]),
                       "completion_censored_mean": _mean([x["censored_mean"] for x in comp]),
                       "unfinished": sum(x["unfinished"] for x in comp),
                       "never_started": sum(x["never_started"] for x in comp),
                       "drift_errors": _mean([u.raw.get("drift_errors") for u in us]),
                       "notes_created": _mean([u.raw.get("notes_created") for u in us])}
        n_values = {}
        for name, path in RQ2_KEYS:
            vals = [_dig(u.raw, path) for u in us]
            per_cell[c][name] = _mean(vals)
            n_values[name] = sum(v is not None for v in vals)
        gp = [llm_goodput(u) for u in us]
        per_cell[c]["llm_goodput"] = _mean(gp)
        n_values["llm_goodput"] = sum(v is not None for v in gp)
        # how many units each mean is over (a metric can be undefined for a unit, e.g. no judged calls; review L3)
        per_cell[c]["n_values"] = n_values

    def paired(a, b, fn):
        seeds = [s for s in sorted(by) if a in by[s] and b in by[s] and by[s][a].status == "done"
                 and by[s][b].status == "done"]
        vals = [fn(by[s][a], by[s][b]) for s in seeds]
        return summary(seeds, vals, two_sided=False)

    def comp_diff(key):
        def f(x, y):
            a, b = completion(x)[key], completion(y)[key]
            return None if a is None or b is None else a - b
        return f

    out = {"cells": per_cell, "paired": {}}
    for a, b in (("C2+acc", "C0+acc"), ("C2+acc", "C2+none"), ("C2+fixed4", "C2+acc")):
        if any(a in by[s] for s in by) and any(b in by[s] for s in by):
            out["paired"][f"{a} - {b}"] = {
                "success": paired(a, b, lambda x, y: x.success_rate() - y.success_rate()),
                "completion_finished": paired(a, b, comp_diff("finished_mean")),
                "completion_censored": paired(a, b, comp_diff("censored_mean"))}
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None
