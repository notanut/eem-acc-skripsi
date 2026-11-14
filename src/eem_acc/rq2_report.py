"""Chapter IV tables for RQ2 (design v2). Pure functions of the stored results + the frozen configuration; every
table is descriptive except the pre-registered tests in analysis.primary_comparisons.

Methods shown in the detailed tables ("key methods"): frozen ACC, its kappa = 1 ablation, the best kappa = 1 ACC,
the global baselines (D34), the secondary baselines (D35) and the per-scenario best fixed windows (D16).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import scenarios as S
from .analysis import _T975, _worst_fill
from .metrics import objective_J

CORE = ["goodput", "norm_goodput", "slo_violation_rate", "call_success_rate", "throttle_rate", "e2e_p95_censored",
        "queue_mean", "J_05", "J_10"]


def key_methods(frozen: dict, extended: bool = False) -> dict[str, str]:
    """{label: tag}, in display order, without duplicates: a method that plays several roles gets ONE label
    "<name> (<role>; <role>)". extended=True adds the 3 ablations and fixed W {4, 16, 64} (spec section 16)."""
    a = frozen["acc"]
    acc_t = S.method_tag(S.acc_method(**a))
    rows: list[tuple[str, str, str]] = [("ACC", "frozen", acc_t)]
    if a.get("kappa", 1.0) != 1.0:
        rows.append(("ACC", "kappa = 1 ablation", S.method_tag(S.acc_method(**{**a, "kappa": 1.0}))))
    k1 = frozen.get("acc_kappa1")
    if k1:
        feas = "" if k1.get("feasible") else f", dev: infeasible, {k1.get('n_feasible_cells', '?')}/{k1.get('n_loaded_cells', '?')}"
        rows.append(("ACC", f"best kappa = 1{feas}", S.method_tag(S.acc_method(**k1["config"]))))
    if extended:
        for name, flags in (("no 429 branch", S.ABLATIONS["acc_no429"]), ("no latency branch", S.ABLATIONS["acc_nolat"]),
                            ("additive only", S.ABLATIONS["acc_additive"])):
            rows.append(("ACC", f"ablation: {name}", S.method_tag(S.acc_method(**a, **flags))))
    fb = frozen.get("baseline_fallback", {})
    note = (lambda f: f", dev: closest to feasible, {fb[f]['n_feasible_cells']}/{fb[f]['n_loaded_cells']}" if f in fb else "")
    rows.append((f"Fixed W={frozen['best_fixed_w']}", "global" + note("fixed"), S.method_tag(S.fixed_method(frozen["best_fixed_w"]))))
    rows.append((f"Token bucket {frozen['best_pacer_rate']:g}/s", "global" + note("tokenbucket"),
                 S.method_tag(S.pacer_method(frozen["best_pacer_rate"]))))
    sec = frozen.get("secondary_global_baselines", {})
    if "fixed" in sec:
        rows.append((f"Fixed W={sec['fixed']['value']}", "secondary global", S.method_tag(S.fixed_method(sec["fixed"]["value"]))))
    if "tokenbucket" in sec:
        rows.append((f"Token bucket {sec['tokenbucket']['value']:g}/s", "secondary global",
                     S.method_tag(S.pacer_method(sec["tokenbucket"]["value"]))))
    for scn, w in frozen["best_fixed_w_per_scenario"].items():
        rows.append((f"Fixed W={w}", f"best for {scn}", S.method_tag(S.fixed_method(w))))
    if extended:
        for w in S.FIXED_W_REPORTED:
            rows.append((f"Fixed W={w}", "reported", S.method_tag(S.fixed_method(w))))
    names: dict[str, str] = {}
    roles: dict[str, list[str]] = {}
    for name, role, tag in rows:
        names.setdefault(tag, name)
        roles.setdefault(tag, [])
        if role not in roles[tag]:
            roles[tag].append(role)
    return {f"{names[t]} ({'; '.join(r)})": t for t, r in roles.items()}


def _with_derived(df: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    d = _worst_fill(df)
    dn = frozen["denominators"]
    d["norm_goodput"] = [g / dn[f"{s}|{int(m)}"] if f"{s}|{int(m)}" in dn else np.nan
                         for g, s, m in zip(d["goodput"], d["scenario"], d["M"])]
    d["J_05"] = [objective_J(r, 0.5, 0.5) for r in d.to_dict("records")]
    d["J_10"] = [objective_J(r, 1.0, 1.0) for r in d.to_dict("records")]
    # J with the latency excess over successful attempts only (spec section 16: reported next to J)
    d["J_05_success_only"] = [r["call_success_rate"] - 0.5 * (r.get("mean_latency_excess") or 0.0) - 0.5 * r["throttle_rate"]
                              for r in d.to_dict("records")]
    if "n_timeout" not in d.columns:
        d["n_timeout"] = 0
    return d


def _ci_half(x: pd.Series) -> float:
    x = x.dropna()
    n = len(x)
    if n < 2:
        return math.nan
    q = _T975.get(n - 1) or (_T975[max(k for k in _T975 if k <= n - 1)] if n - 1 < 30 else 1.96)
    return float(q * x.std(ddof=1) / math.sqrt(n))


def cell_table(test: pd.DataFrame, frozen: dict, loaded_only: bool = True, extended: bool = True) -> pd.DataFrame:
    """Per (scenario, M) cell and key method: seed means of the core metrics (+ 95% t half-width of goodput), the
    feasibility flag, and the client-side cost (censored e2e p95, mean gateway queue) next to every SLO number."""
    km = key_methods(frozen, extended=extended)
    inv = {t: lab for lab, t in km.items()}
    d = _with_derived(test[test["tag"].isin(km.values())], frozen)
    loaded = {tuple(x.split("|")) for x in frozen["criterion"]["loaded_cells"]}
    d["loaded"] = [(s, str(int(m))) in loaded for s, m in zip(d["scenario"], d["M"])]
    if loaded_only:
        d = d[d["loaded"]]
    g = d.groupby(["scenario", "M", "tag"])
    t = g[CORE + ["J_05_success_only", "n_timeout"]].mean()
    t["goodput_ci95"] = g["goodput"].apply(_ci_half)
    t["n_seeds"] = g.size()
    t = t.reset_index()
    t["method"] = t["tag"].map(inv)
    t["feasible"] = ((t["slo_violation_rate"] <= S.MAX_SLO_VIOLATION) & (t["call_success_rate"] >= S.MIN_CALL_SUCCESS)
                     & (t["throttle_rate"] <= S.MAX_THROTTLE_RATE))
    order = {lab: i for i, lab in enumerate(km)}
    t["_o"] = t["method"].map(order)
    t = t.sort_values(["scenario", "M", "_o"]).drop(columns="_o")
    return t[["scenario", "M", "method", "tag", "n_seeds", "goodput", "goodput_ci95", "norm_goodput",
              "slo_violation_rate", "call_success_rate", "throttle_rate", "feasible", "e2e_p95_censored", "queue_mean",
              "J_05", "J_10", "J_05_success_only", "n_timeout"]].reset_index(drop=True)


def event_table(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Event windows (pre / during / post) of the scenarios with an event, for the key methods: seed means of
    completion-time goodput, SLO violation, overload rate, censored e2e p95 and mean queue length."""
    km = key_methods(frozen)
    inv = {t: lab for lab, t in km.items()}
    d = test[test["tag"].isin(km.values())]
    rows = []
    for scn, sc in S.SCENARIOS.items():
        an = sc.get("analysis")
        if an is None or an.event is None:
            continue
        for (m, tag), grp in d[d["scenario"] == scn].groupby(["M", "tag"]):
            for name, _, _ in an.windows:
                rec = {"scenario": scn, "M": m, "method": inv[tag], "tag": tag, "window": name, "n_seeds": len(grp)}
                for k in ("goodput", "slo_violation_rate", "throttle_rate", "e2e_p95_censored", "queue_mean"):
                    col = f"win_{name}_{k}"
                    rec[k] = float(grp[col].mean()) if col in grp else math.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def recovery_table(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Recovery after the event, per scenario x M x key method: how many seeds recovered / never degraded / were
    censored, and the median and max recovery time over the recovered seeds."""
    km = key_methods(frozen)
    inv = {t: lab for lab, t in km.items()}
    d = test[test["tag"].isin(km.values()) & test.get("recovery_status", pd.Series(index=test.index)).notna()]
    rows = []
    for (scn, m, tag), grp in d.groupby(["scenario", "M", "tag"]):
        st = grp["recovery_status"]
        rec_t = grp.loc[st == "recovered", "recovery_time"]
        bound = grp.loc[st.isin(["recovered", "censored"]), "recovery_time"]      # censored: lower bound T - end
        rows.append({"scenario": scn, "M": m, "method": inv[tag], "tag": tag, "n_seeds": len(grp),
                     "median_incl_censored_s": float(bound.median()) if len(bound) else math.nan,
                     "median_incl_censored_is_lower_bound": bool((st == "censored").sum() * 2 >= len(bound)) if len(bound) else False,
                     "recovered": int((st == "recovered").sum()), "no_degradation": int((st == "no_degradation").sum()),
                     "censored": int((st == "censored").sum()), "no_reference": int((st == "no_reference").sum()),
                     "median_recovery_s": float(rec_t.median()) if len(rec_t) else math.nan,
                     "max_recovery_s": float(rec_t.max()) if len(rec_t) else math.nan})
    return pd.DataFrame(rows)


def loso_table(frozen: dict) -> pd.DataFrame:
    rows = []
    for scn, e in frozen.get("loso", {}).items():
        for fam, v in e.items():
            if not isinstance(v, dict):
                continue
            rows.append({"left_out": scn, "family": fam, "chosen": v.get("tag"),
                         "feasible_on_training": v.get("feasible_on_training"),
                         "feasible_on_left_out": v.get("feasible_on_left_out"),
                         "score_on_left_out": v.get("score_on_left_out"),
                         "slo_on_left_out": v.get("mean_slo_violation"),
                         "overload_on_left_out": v.get("mean_throttle_rate")})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Plain GitHub-style table (no extra dependency)."""
    cols = list(df.columns)

    def fmt(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if math.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def reporting_facts(frozen: dict, dev: pd.DataFrame | None, test: pd.DataFrame) -> pd.DataFrame:
    """Spec section 16 items that are single facts: tuning budget, kappa (1 + eps) of the frozen ACC, the kappa / grid
    corner evidence (feasible dev ACC configs by kappa, top-5 score spread), the best-scoring infeasible fixed window on
    dev next to P1a, and test timeout counts."""
    from .analysis import score_methods
    a = frozen["acc"]
    rows = [("tuning budget (dev configurations)", f"ACC {len(S.acc_grid_methods())}, fixed W {len(S.FIXED_W_DEV)}, "
                                                     f"token bucket {len(S.PACER_RATES_DEV)}"),
            ("frozen ACC kappa*(1+eps)", f"{a['kappa'] * (1 + a['epsilon']):.3f} (band within the SLO iff <= 1)")]
    if dev is not None and len(dev):
        loaded = [(s, int(m)) for s, m in (x.split("|") for x in frozen["criterion"]["loaded_cells"])]
        sc = score_methods(dev, loaded, frozen["denominators"])
        acc = sc[sc["m_kind"] == "acc"]
        feas = acc[acc["feasible"]]
        rows.append(("feasible dev ACC configs by kappa", str(feas["m_kappa"].value_counts().sort_index().to_dict())))
        top = feas.sort_values("score", ascending=False).head(5)["score"]
        rows.append(("top-5 feasible dev ACC score spread", f"{top.max():.4f} - {top.min():.4f}"))
        fx = sc[sc["m_kind"] == "fixed"].sort_values("score", ascending=False).iloc[0]
        rows.append(("best-scoring fixed W on dev (infeasible)" if not fx["feasible"] else "best-scoring fixed W on dev",
                     f"{fx['tag']}: score {fx['score']:.3f}, mean SLO {fx['mean_slo_violation']:.3f}, "
                     f"mean overload {fx['mean_throttle_rate']:.3f}, feasible cells {int(fx['n_feasible_cells'])}/{len(loaded)}"))
    if "n_timeout" in test.columns:
        rows.append(("client timeouts on test (all runs)", f"{int(test['n_timeout'].sum())} of {int(test['n_attempts'].sum())} attempts"))
    return pd.DataFrame(rows, columns=["fact", "value"])


def pareto_table(frozen: dict) -> pd.DataFrame:
    """The dev Pareto front stored in the frozen file (spec section 6)."""
    return pd.DataFrame(frozen.get("pareto_front", []))
