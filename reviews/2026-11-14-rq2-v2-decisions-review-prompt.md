# Review request: RQ2 v2 decisions D24-D28 (2026-11-14)

You are an independent reviewer (experimental statistics and simulation). Be concrete.

## Context
https://github.com/notanut/eem-acc-skripsi, RQ2 design v2 (docs/rq2_design_v2.md). The user decided: D24 add fixed
windows 12 and 20 to the dev grid; D25 test the "about equal" endpoint (ACC vs best fixed window, RATE-steady,
normalised goodput) with an equivalence test, margin +-0.05 (TOST with one-sided sign-flip tests over 10 paired test
seeds, plus a paired-t 90% CI check); D26 recovery after an event additionally requires the e2e p95 of calls requested
in the trailing 5-s window to be <= 2 x the method's own pre-event e2e p95 (empty window = not recovered); D27 the best
kappa = 1 ACC configuration selected on dev is also run in the test phase (fallback when none is feasible: the one
feasible in most loaded cells); D28 success measured over calls requested before T - 20 s.

## Checklist
- TOST correctness (directions, shifts, ties, exactness assumptions), combining TOST and two-sided p-values under Holm.
- Recovery e2e rule: fairness across methods (bound relative to each method's own pre-event e2e), empty windows,
  censoring near T.
- kappa = 1 selection and its fallback; reporting in the robustness matrix.
- Tests that would pass even if the code were wrong.

## Output
Findings by severity (location, explanation, fix), then a verdict.

## Diff against the previous state of main
```diff
diff --git a/src/eem_acc/analysis.py b/src/eem_acc/analysis.py
index 5293b71..173bca6 100644
--- a/src/eem_acc/analysis.py
+++ b/src/eem_acc/analysis.py
@@ -195,6 +195,17 @@ def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
         else:
             res["ok"] = False
     res["best_fixed_w_per_scenario"] = per
+    # D27: the best kappa = 1 ACC config (the paper's base algorithm at its best); if none is feasible, the best-scoring
+    # one is taken and marked infeasible (it is a reference method, not a candidate for the frozen ACC)
+    k1 = scored[(scored["m_kind"] == "acc") & (scored["m_kappa"] == 1.0)]
+    if len(k1):
+        row = choose(k1, "acc")
+        feasible = row is not None
+        if row is None:                    # closest to feasible first (not the most aggressive), complete rows only
+            c = k1[k1["complete"]] if k1["complete"].any() else k1
+            row = c.sort_values(["n_feasible_cells", "score", "mean_slo_violation", "tag"],
+                                ascending=[False, False, True, True]).iloc[0]
+        res["acc_kappa1"] = {"config": _params(row, "acc"), "feasible": bool(feasible), "score": float(row["score"])}
     res["pareto_front"] = pareto_front(scored)
     res["loso"] = leave_one_scenario_out(df)
     return res
@@ -246,6 +257,11 @@ def robustness_matrix(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
     rows = {"ACC (frozen)": S.method_tag(S.acc_method(**frozen["acc"])),
             f"fixed W={frozen['best_fixed_w']} (global)": S.method_tag(S.fixed_method(frozen["best_fixed_w"])),
             f"token bucket {frozen['best_pacer_rate']:g}/s": S.method_tag(S.pacer_method(frozen["best_pacer_rate"]))}
+    a = frozen["acc"]
+    if a.get("kappa", 1.0) != 1.0:
+        rows["ACC (frozen, kappa = 1: ablation)"] = S.method_tag(S.acc_method(**{**a, "kappa": 1.0}))
+    if frozen.get("acc_kappa1"):
+        rows["ACC (best kappa = 1 on dev)"] = S.method_tag(S.acc_method(**frozen["acc_kappa1"]["config"]))
     for scn, w in frozen["best_fixed_w_per_scenario"].items():
         rows[f"fixed W={w} (best for {scn})"] = S.method_tag(S.fixed_method(w))
     out = []
@@ -276,6 +292,39 @@ _T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          20: 2.086, 25: 2.060, 30: 2.042}
 
 
+def tost_sign_flip(d: np.ndarray, margin: float) -> float:
+    """Equivalence (TOST) with one-sided sign-flip tests: H0a mean <= -margin, H0b mean >= +margin.
+    p = max(p_a, p_b); equivalence is shown at level a if p <= a. Exact only under SYMMETRY of the paired differences
+    about the boundary (unlike the test of no difference, which only needs exchangeable pairs); the paired-t TOST
+    (90% CI inside +-margin) is reported next to it as a robustness check."""
+    d = np.asarray(d, dtype=float)
+    n = len(d)
+    if n == 0 or n > 20:
+        raise ValueError("need 1..20 paired differences")
+    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
+    up, dn = d + margin, d - margin
+    p_a = float(np.mean((signs * up).mean(axis=1) >= up.mean() - 1e-12))    # small if mean(d) clearly > -margin
+    p_b = float(np.mean((signs * dn).mean(axis=1) <= dn.mean() + 1e-12))    # small if mean(d) clearly < +margin
+    return max(p_a, p_b)
+
+
+_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
+        11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729,
+        20: 1.725, 25: 1.708, 30: 1.697}
+
+
+def t_ci90(d: np.ndarray) -> tuple[float, float]:
+    """90% paired t interval (the TOST at alpha = 0.05 shows equivalence iff it lies inside +-margin)."""
+    d = np.asarray(d, dtype=float)
+    n = len(d)
+    if n < 2:
+        return (math.nan, math.nan)
+    df = n - 1
+    q = _T95.get(df) or (_T95[max(k for k in _T95 if k <= df)] if df < 30 else 1.645)
+    h = q * d.std(ddof=1) / math.sqrt(n)
+    return (float(d.mean() - h), float(d.mean() + h))
+
+
 def t_ci(d: np.ndarray) -> tuple[float, float]:
     d = np.asarray(d, dtype=float)
     n = len(d)
@@ -332,11 +381,23 @@ def primary_comparisons(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
     for name, (base, cells_, metric, exp) in ends.items():
         a, b = _per_seed(test, acc, cells_, dn, metric), _per_seed(test, base, cells_, dn, metric)
         d = (a - b).dropna()
-        p = sign_flip_test(d.values) if len(d) else math.nan
+        equivalence = exp == "about equal"                      # D25: TOST with +-EQUIVALENCE_MARGIN
+        if not len(d):
+            p = math.nan
+        elif equivalence:
+            p = tost_sign_flip(d.values, S.EQUIVALENCE_MARGIN)
+        else:
+            p = sign_flip_test(d.values)
         pv[name] = p
         lo, hi = t_ci(d.values)
-        rows.append({"endpoint": name, "expected": exp, "n_seeds": len(d), "mean_diff": float(d.mean()) if len(d) else math.nan,
-                     "ci95_low": lo, "ci95_high": hi, "p_signflip": p})
+        row = {"endpoint": name, "expected": exp, "test": f"TOST +-{S.EQUIVALENCE_MARGIN:g}" if equivalence else "two-sided",
+               "n_seeds": len(d), "mean_diff": float(d.mean()) if len(d) else math.nan,
+               "ci95_low": lo, "ci95_high": hi, "p_value": p}
+        if equivalence:                     # a TOST rejection at 0.05 means "equivalent"; t robustness check
+            l90, h90 = t_ci90(d.values)
+            row.update(ci90_low=l90, ci90_high=h90,
+                       t_tost_equivalent=bool(-S.EQUIVALENCE_MARGIN < l90 and h90 < S.EQUIVALENCE_MARGIN))
+        rows.append(row)
     adj = holm({k: v for k, v in pv.items() if not math.isnan(v)})
     for r in rows:
         r["p_holm"] = adj.get(r["endpoint"], math.nan)
diff --git a/src/eem_acc/metrics.py b/src/eem_acc/metrics.py
index 25fc74d..72cc207 100644
--- a/src/eem_acc/metrics.py
+++ b/src/eem_acc/metrics.py
@@ -93,6 +93,7 @@ class AnalysisConfig:
     recovery_min_ok: int = 20         # fewer successes in a window -> judge goodput only
     recovery_goodput_frac: float = 0.9
     success_grace: float = 0.0        # call success is measured over calls requested before T - grace (see below)
+    recovery_e2e_factor: float | None = None   # if set: recovery also needs e2e p95 <= factor x pre-window e2e p95
 
 
 def _slo_bad(a, l_target: float) -> bool:
@@ -156,7 +157,10 @@ def recovery_time(gw: Gateway, calls, l_target: float, horizon: float, cfg: Anal
     are good; time = t* - event end.
     status: 'no_degradation' (no bad point between event start + span and t*: damage AFTER the event counts too) |
     'recovered' | 'censored' (never `hold` good points after the end; time = lower bound T - end) | 'no_reference'
-    (no success in the pre window). The trailing window adds up to `span` s of lag to every recovery time."""
+    (no success in the pre window). The trailing window adds up to `span` s of lag to every recovery time.
+    If `recovery_e2e_factor` is set (D26), a point is also bad unless the e2e p95 of calls REQUESTED in the window
+    (unfinished calls censored at T) is <= factor x the pre-window e2e p95; a window with no requested call is bad
+    (every agent is stuck)."""
     (e0, e1), (p0, p1) = cfg.event, cfg.pre
     pre_ok = [a.latency for a in gw.attempts if p0 <= a.t_dispatch < p1 and a.status == 200]
     pre_gp = _completed_ok(calls, p0, p1) / (p1 - p0)
@@ -164,6 +168,12 @@ def recovery_time(gw: Gateway, calls, l_target: float, horizon: float, cfg: Anal
         return {"status": "no_reference", "time": None}
     lat_bound = max(l_target, float(np.percentile(pre_ok, 95)))
     span = cfg.recovery_span
+    e2e_bound = None
+    if cfg.recovery_e2e_factor is not None:
+        pre_e2e = [c[2] for c in calls if p0 <= c[0] < p1]
+        if not pre_e2e:
+            return {"status": "no_reference", "time": None}
+        e2e_bound = cfg.recovery_e2e_factor * float(np.percentile(pre_e2e, 95))
 
     def good(t: float) -> bool:
         lo = t - span
@@ -173,6 +183,10 @@ def recovery_time(gw: Gateway, calls, l_target: float, horizon: float, cfg: Anal
             return False
         if len(ok_lat) >= cfg.recovery_min_ok and float(np.percentile(ok_lat, 95)) > lat_bound:
             return False
+        if e2e_bound is not None:
+            e2e = [c[2] for c in calls if lo < c[0] <= t]
+            if not e2e or float(np.percentile(e2e, 95)) > e2e_bound:
+                return False
         return True
 
     step = cfg.recovery_grid
diff --git a/src/eem_acc/scenarios.py b/src/eem_acc/scenarios.py
index 895ae15..39419d8 100644
--- a/src/eem_acc/scenarios.py
+++ b/src/eem_acc/scenarios.py
@@ -21,8 +21,7 @@ HORIZON = 120.0                      # s of simulated time per run
 THINK_MEAN = 0.1                     # s, exponential
 START_JITTER = 1.0                   # s
 RETRY = RetryConfig(max_retries=6, backoff_base=0.1, backoff_cap=4.0, timeout=1.0)
-SUCCESS_GRACE = 20.0                 # s; success over calls requested before T - 20 s (implementation finding,
-                                     # pending the user's confirmation: see HANDOFF / report 2026-11-14)
+SUCCESS_GRACE = 20.0                 # s; success over calls requested before T - 20 s (confirmed by the user, D28)
 
 # ---- server regimes and scenarios (D16, D18) ---------------------------------
 _COMMON = dict(base_latency=0.2, latency_sigma=0.25, degradation=1.5, retry_after=0.5, error_latency=0.02)
@@ -80,7 +79,7 @@ ACC_GRID = dict(alpha=(0.25, 0.5, 0.75), beta=(0.25, 0.5, 0.75), epsilon=(0.05,
                 kappa=(0.8, 0.9, 1.0))
 ACC_START = dict(alpha=0.5, beta=0.5, epsilon=0.1, n_c=16, kappa=0.9)    # centre of the grid (illustrations only)
 FIXED_W_REPORTED = (4, 16, 64)
-FIXED_W_DEV = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96)
+FIXED_W_DEV = (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96)   # 12 and 20 added (D24)
 PACER_RATES_DEV = (10.0, 14.0, 15.0, 16.0, 17.0, 18.0, 22.0, 27.0, 33.0, 40.0, 50.0, 65.0, 80.0, 100.0, 130.0)
 PACER_BURST = 1
 UNCONSTRAINED_W = 128
@@ -97,6 +96,8 @@ MAX_SLO_VIOLATION = 0.05
 MIN_CALL_SUCCESS = 0.99
 MAX_THROTTLE_RATE = 0.10
 TIE_TOLERANCE = 0.005                # candidates with score >= (1 - tol) * best are tied
+EQUIVALENCE_MARGIN = 0.05            # +-5% normalised goodput, TOST for 'about equal' endpoints (D25)
+RECOVERY_E2E_FACTOR = 2.0            # recovery also needs e2e p95 <= 2 x pre-window e2e p95 (D26)
 
 
 # ---- method specs (plain dicts -> JSON-able, named via tag) --------------------
@@ -151,7 +152,7 @@ def build(scenario: str, m_agents: int, method: dict, seed: int) -> Scenario:
         pacer = PacerConfig(rate=method["rate"], burst=PACER_BURST)
     else:
         raise ValueError(f"unknown method kind: {k}")
-    analysis = dataclasses.replace(sc["analysis"], success_grace=SUCCESS_GRACE)
+    analysis = dataclasses.replace(sc["analysis"], success_grace=SUCCESS_GRACE, recovery_e2e_factor=RECOVERY_E2E_FACTOR)
     return Scenario(server=server, workload=wl, retry=RETRY, acc=acc, seed=seed, pacer=pacer, analysis=analysis)
 
 
@@ -166,13 +167,15 @@ def dev_methods() -> list[dict]:
 
 
 def test_methods(frozen: dict) -> list[dict]:
-    """Test-phase methods from the frozen dev selection: frozen ACC; the same ACC with kappa = 1 (the paper's base
-    algorithm, D22); 3 ablations; fixed {4,16,64}; the global best fixed W; the per-scenario best fixed W values
+    """Test-phase methods from the frozen dev selection: frozen ACC; the same ACC with kappa = 1 (kappa ablation);
+    the best kappa = 1 ACC config selected on dev (the paper's base algorithm at its best, D27); 3 ablations; fixed {4,16,64}; the global best fixed W; the per-scenario best fixed W values
     (robustness matrix, D16); the tuned token bucket. Duplicates are removed (order kept)."""
     a = frozen["acc"]
     ms = [acc_method(**a)]
     if a.get("kappa", 1.0) != 1.0:
-        ms.append(acc_method(**{**a, "kappa": 1.0}))
+        ms.append(acc_method(**{**a, "kappa": 1.0}))              # kappa ablation (same config, no margin)
+    if frozen.get("acc_kappa1"):
+        ms.append(acc_method(**frozen["acc_kappa1"]["config"]))   # best kappa = 1 config on dev (D27)
     ms += [acc_method(**a, **flags) for flags in ABLATIONS.values()]
     ws = list(FIXED_W_REPORTED) + [frozen["best_fixed_w"]] + sorted(set(frozen["best_fixed_w_per_scenario"].values()))
     ms += [fixed_method(w) for w in ws]
diff --git a/tests/test_rq2.py b/tests/test_rq2.py
index 672ce09..9f3279f 100644
--- a/tests/test_rq2.py
+++ b/tests/test_rq2.py
@@ -71,9 +71,9 @@ def test_tokenbucket_scenario_respects_rate():
 # ---------- design wiring (v2, docs/rq2_design_v2.md) ----------
 def test_design_counts_match_approved_v2():
     assert len(S.acc_grid_methods()) == 81 * 3                      # kappa {0.8, 0.9, 1.0} (D22)
-    assert len(S.FIXED_W_DEV) == 10 and len(S.PACER_RATES_DEV) == 15
+    assert len(S.FIXED_W_DEV) == 12 and {12, 20} <= set(S.FIXED_W_DEV) and len(S.PACER_RATES_DEV) == 15
     dev = S.units(S.dev_methods(), S.DEV_SEEDS)
-    assert len(dev) == (243 + 10 + 15) * 4 * 3 * 5 == 16080
+    assert len(dev) == (243 + 12 + 15) * 4 * 3 * 5 == 16200
     assert len({u[0] for u in dev}) == len(dev)                     # unique unit ids
 
 
@@ -321,10 +321,10 @@ def test_primary_comparisons_on_synthetic_test_data(monkeypatch):
     test = pd.DataFrame(rows)
     pc = primary_comparisons(test, frozen).set_index("endpoint")
     p1a = pc.loc["P1a ACC - global fixed W"]
-    assert p1a["mean_diff"] > 0 and p1a["p_signflip"] == pytest.approx(2 / 1024) and p1a["n_seeds"] == 10
+    assert p1a["mean_diff"] > 0 and p1a["p_value"] == pytest.approx(2 / 1024) and p1a["n_seeds"] == 10
     p2a = pc.loc["P2 A: ACC - best fixed W=64"]
     assert p2a["mean_diff"] < 0
-    assert (pc["p_holm"] >= pc["p_signflip"] - 1e-12).all()
+    assert (pc["p_holm"] >= pc["p_value"] - 1e-12).all()
     rm = robustness_matrix(test, frozen)
     assert set(rm["method"]) >= {"ACC (frozen)", "fixed W=64 (best for A)"}
 
@@ -358,7 +358,7 @@ def test_primary_comparisons_pair_by_seed(monkeypatch):
         rows += _rows("tb_r40", "tokenbucket", {("A", 16): (base, 0, 1)}, seeds=(s,), rate=40.0)
     pc = primary_comparisons(pd.DataFrame(rows), frozen).set_index("endpoint")
     r = pc.loc["P1a ACC - global fixed W"]
-    assert r["mean_diff"] == pytest.approx(1.0) and r["p_signflip"] == pytest.approx(2 / 1024)
+    assert r["mean_diff"] == pytest.approx(1.0) and r["p_value"] == pytest.approx(2 / 1024)
     assert r["ci95_low"] == pytest.approx(1.0) and r["ci95_high"] == pytest.approx(1.0)
 
 
@@ -368,3 +368,59 @@ def test_select_rerun_with_identical_result_is_not_refused(tmp_path, monkeypatch
     run_rq2.run_phase("dev", tmp_path)
     assert run_rq2.do_select(tmp_path) == 0
     assert run_rq2.do_select(tmp_path) == 0
+
+
+
+def test_best_kappa1_config_is_added_to_the_test_methods():
+    f = {**FROZEN, "acc_kappa1": {"config": dict(alpha=0.75, beta=0.25, epsilon=0.05, n_c=32, kappa=1.0),
+                                  "feasible": False, "score": 0.9}}
+    tags = [S.method_tag(m) for m in S.test_methods(f)]
+    assert tags[:3] == ["acc_a0.5_b0.5_e0.1_n16_k0.8", "acc_a0.5_b0.5_e0.1_n16_k1", "acc_a0.75_b0.25_e0.05_n32_k1"]
+    assert len(tags) == len(set(tags)) == 13
+
+
+def test_select_reports_best_kappa1_config(synthetic_design):
+    df = _synthetic()
+    k08 = df[df["tag"] == "accB"].copy()
+    k08["tag"], k08["m_kappa"], k08["goodput"] = "accB_k08", 0.8, k08["goodput"] * 1.05
+    res = select_frozen(pd.concat([df, k08]))
+    assert res["acc"]["kappa"] == 0.8                                  # the kappa 0.8 variant wins overall
+    k1 = res["acc_kappa1"]
+    assert k1["config"] == acc(a=.25) and k1["feasible"] is True
+    sc = score_methods(pd.concat([df, k08]), LOADED, res["denominators"]).set_index("tag")
+    assert k1["score"] == pytest.approx(sc.loc["accB", "score"])
+
+
+def test_best_kappa1_falls_back_to_best_score_when_infeasible(synthetic_design):
+    df = _synthetic()
+    df.loc[df["tag"].str.startswith("acc"), "slo_violation_rate"] = 0.2
+    df.loc[(df["tag"] == "accC") & ~((df["scenario"] == "A") & (df["M"] == 64)), "slo_violation_rate"] = 0.0
+    res = select_frozen(df)                         # accA scores highest but is infeasible in all cells; accC in one
+    assert res["acc_kappa1"]["feasible"] is False and res["acc_kappa1"]["config"] == acc(a=.5)
+
+
+def test_tost_exact_values():
+    from eem_acc.analysis import tost_sign_flip
+    assert tost_sign_flip([0.0] * 10, 0.05) == pytest.approx(1 / 1024)       # clearly inside the margin
+    assert tost_sign_flip([0.04] * 10, 0.05) == pytest.approx(1 / 1024)
+    assert tost_sign_flip([0.1] * 10, 0.05) == 1.0                           # clearly outside
+    assert tost_sign_flip([0.2, -0.2] * 5, 0.05) > 0.05                      # noisy: equivalence not shown
+
+
+def test_about_equal_endpoint_uses_tost(monkeypatch):
+    from eem_acc.analysis import primary_comparisons
+    frozen = {"acc": acc(), "best_fixed_w": 4, "best_pacer_rate": 40.0, "best_fixed_w_per_scenario": {"RATE-steady": 32},
+              "criterion": {"loaded_cells": ["RATE-steady|96"]}, "denominators": {"RATE-steady|96": 100.0}}
+    monkeypatch.setattr(S, "SCENARIOS", {"RATE-steady": {}})
+    rows = []
+    for s in range(100, 110):
+        jitter = 0.5 if s % 2 else -0.5
+        rows += _rows(S.method_tag(S.acc_method(**acc())), "acc", {("RATE-steady", 96): (97 + jitter, 0, 1, 0.05)}, seeds=(s,))
+        rows += _rows("fixed_w32", "fixed", {("RATE-steady", 96): (97, 0, 1, 0.0)}, seeds=(s,), w=32)
+        rows += _rows("fixed_w4", "fixed", {("RATE-steady", 96): (19, 0, 1)}, seeds=(s,), w=4)
+        rows += _rows("tb_r40", "tokenbucket", {("RATE-steady", 96): (40, 0, 1)}, seeds=(s,), rate=40.0)
+    pc = primary_comparisons(pd.DataFrame(rows), frozen).set_index("endpoint")
+    r = pc.loc["P2 RATE-steady: ACC - best fixed W=32"]
+    assert r["test"] == "TOST +-0.05" and r["p_value"] < 0.01         # differences +-0.005 << 0.05 margin
+    assert r["t_tost_equivalent"] and -0.05 < r["ci90_low"] < r["ci90_high"] < 0.05
+    assert pc.loc["P1a ACC - global fixed W", "test"] == "two-sided"
diff --git a/tests/test_v2_metrics.py b/tests/test_v2_metrics.py
index 4913973..cc6aa37 100644
--- a/tests/test_v2_metrics.py
+++ b/tests/test_v2_metrics.py
@@ -144,3 +144,39 @@ def test_recovery_judges_latency_when_enough_successes():
     # same shape at 10 calls/s (50 successes per window): the 0.6-s latency is judged -> degradation, then recovery
     g = build([(0, 40, 10, 0.2), (40, 80, 10, 0.6), (80, 120, 10, 0.2)])
     assert _rec(g) == {"status": "recovered", "time": pytest.approx(5.0)}
+
+
+def _queued(segs):
+    """(t0, t1, rate, server latency, e2e): calls whose end-to-end time includes client-side queueing."""
+    att, calls = [], []
+    for t0, t1, rate, lat, e2e in segs:
+        for i in range(int(round((t1 - t0) * rate))):
+            t = t0 + i / rate
+            att.append(A(t + e2e - lat, 200, lat, 8, 1))
+            calls.append(C(t, t + e2e, 200, 1, e2e - lat))
+    return gw(att, calls)
+
+
+def test_recovery_e2e_condition_sees_client_side_queueing():
+    """Server latency stays 0.2 s, but calls requested in [40, 100) wait 2.8 s in the client queue (e2e 3 s).
+    Server-side signals recover at the event end; with the e2e condition (2 x pre e2e p95 = 0.4 s) recovery
+    comes only when calls requested in the trailing window are fast again: t = 105 -> 25 s after the end."""
+    from eem_acc.metrics import _calls_with_censoring
+    g = _queued([(0, 40, 10, 0.2, 0.2), (40, 100, 10, 0.2, 3.0), (100, 120, 10, 0.2, 0.2)])
+    calls = _calls_with_censoring(g, 120.0)
+    base = dict(event=(40.0, 80.0), pre=(20.0, 40.0))
+    assert recovery_time(g, calls, L, 120.0, AnalysisConfig(**base))["time"] == pytest.approx(0.0)
+    assert recovery_time(g, calls, L, 120.0, AnalysisConfig(**base, recovery_e2e_factor=2.0)) == \
+        {"status": "recovered", "time": pytest.approx(25.0)}
+
+
+def test_recovery_e2e_window_without_requests_is_bad():
+    """Calls requested in [70, 80) wait 10 s (they complete in [80, 90), so goodput and server latency look fine there);
+    nobody requests anything in [80, 90). Windows with no requested call must count as bad: recovery only when fast
+    requests (from t = 90) enter the trailing window (t - 5, t], closed on the right: t = 90 -> 10 s after the event
+    end. (If empty windows counted as good, recovery would be reported at t = 85.)"""
+    from eem_acc.metrics import _calls_with_censoring
+    g = _queued([(0, 70, 10, 0.2, 0.2), (70, 80, 10, 0.2, 10.0), (90, 120, 10, 0.2, 0.2)])
+    r = recovery_time(g, _calls_with_censoring(g, 120.0), L, 120.0,
+                      AnalysisConfig(event=(40.0, 80.0), pre=(20.0, 40.0), recovery_e2e_factor=2.0))
+    assert r == {"status": "recovered", "time": pytest.approx(10.0)}
diff --git a/tests/test_viz_monitor.py b/tests/test_viz_monitor.py
index 45d464d..0b1be67 100644
--- a/tests/test_viz_monitor.py
+++ b/tests/test_viz_monitor.py
@@ -22,14 +22,14 @@ def test_simulate_returns_gateway_with_logs():
 
 def test_snapshot_empty_then_partial(tmp_path):
     snap = snapshot(tmp_path, "dev")
-    assert snap["done"] == 0 and snap["total"] == 16080 and not snap["running"]
+    assert snap["done"] == 0 and snap["total"] == 16200 and not snap["running"]
     run_rq2.run_phase("dev", tmp_path, limit=20)
     snap = snapshot(tmp_path, "dev")
     assert snap["done"] == 20
     assert sum(snap["cell_done"].values()) == 20
-    assert sum(snap["cell_total"].values()) == 16080
+    assert sum(snap["cell_total"].values()) == 16200
     assert snap["cell_done"][("CAP-steady", 8)] == 20            # first 20 units are all in that cell
-    assert "20/16080" in summary_line(snap)
+    assert "20/16200" in summary_line(snap)
 
 
 def test_snapshot_running_state_and_eta(tmp_path):
@@ -44,7 +44,7 @@ def test_snapshot_running_state_and_eta(tmp_path):
     # all 10 finished runs have M=8 -> weighted throughput 8 work-units/s; remaining work = sum(M) of the rest
     total_w = sum(u[2] for u in S.units(S.dev_methods(), S.DEV_SEEDS))
     assert abs(snap["eta_s"] - (total_w - 10 * 8) / 8.0) < 1e-6
-    assert snap["eta_s"] > (16080 - 10) * 2                      # far above the naive runs/s estimate
+    assert snap["eta_s"] > (16200 - 10) * 2                      # far above the naive runs/s estimate
     assert not snapshot(tmp_path, "dev", now=t0 + 500)["running"]   # idle for > 60 s
 
 
```
