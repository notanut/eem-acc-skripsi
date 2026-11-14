# Review request: pre-test safeguards for RQ2 (D35-D37, infeasibility rule), 2026-11-14

You are an independent reviewer (experimental methodology). Be concrete; no praise.

## Context
https://github.com/notanut/eem-acc-skripsi, design docs/rq2_design_v2.md (section 12: D34-D37 and the pre-registered
infeasibility rule; section 16: reporting). After a second opinion on the dev-phase selection, BEFORE any test-seed
run, the author added: D35 a secondary global baseline counted per scenario (W=12, 33/s) reported outside the Holm
family; D36 every fixed window and token-bucket rate on the test seeds as descriptive results, with the descriptive
analysis (test_descriptive) frozen in advance; D37 a boundary check on DEV seeds (frozen ACC one step past each grid
edge), sensitivity only; and the frozen ACC's dev margins with the rule that a test-phase infeasibility is reported
as-is without re-selection.

## Checklist
1. Are these safeguards sufficient against the objections "strawman baseline", "tuned to the grid corner" and
   "results chosen after seeing them"? What else would you pre-register now?
2. Code correctness of the diff (secondary baselines, margins, boundary phase isolation from test seeds, Holm family,
   descriptive table).
3. Anything that should change before the test seeds (100-109) are opened.

## Output
Findings by severity (location, explanation, fix), then a verdict.

## Code diff against main
```diff
diff --git a/experiments/run_rq2.py b/experiments/run_rq2.py
index 6fef3d2..8cdf1aa 100644
--- a/experiments/run_rq2.py
+++ b/experiments/run_rq2.py
@@ -5,8 +5,10 @@ Phases (approved design v2, docs/rq2_design_v2.md, D16-D23):
           4 scenarios x M in {8, 32, 96}, dev seeds 0-4
   select  choose the frozen configuration from dev results (per-loaded-cell constraints) -> configs/rq2_frozen.json
           (also: per-scenario best fixed W, Pareto front, leave-one-scenario-out)
-  test    frozen ACC (+ kappa = 1 variant) + 3 ablations + fixed {4,16,64} + global and per-scenario best fixed W
-          + tuned token bucket, test seeds 100-109
+  boundary  DEV seeds only (D37): frozen ACC with one parameter one step past each grid edge; sensitivity only
+  test    frozen ACC (+ kappa = 1 ablation and best kappa = 1) + 3 ablations + fixed {4,16,64} + global (D34),
+          secondary (D35) and per-scenario best fixed W + token buckets + EVERY dev fixed W and rate (D36,
+          descriptive): 33 methods x 12 cells x test seeds 100-109 = 3,960 runs
 
 Usage (repo root):
   PYTHONPATH=src python experiments/run_rq2.py --phase dev    --root <ROOT>
@@ -42,6 +44,9 @@ def phase_units(phase: str, frozen: dict | None):
     if phase == "test":
         assert frozen is not None, "test phase needs configs/rq2_frozen.json (run --phase select first)"
         return S.units(S.test_methods(frozen), S.TEST_SEEDS), True
+    if phase == "boundary":                     # D37: DEV seeds only, sensitivity, never used to re-select
+        assert frozen is not None, "boundary phase needs the frozen configuration"
+        return S.units(S.boundary_methods(frozen), S.DEV_SEEDS), False
     raise ValueError(phase)
 
 
@@ -177,7 +182,7 @@ def do_select(root: Path, force: bool = False) -> int:
 
 def main() -> int:
     ap = argparse.ArgumentParser()
-    ap.add_argument("--phase", choices=["dev", "select", "test"], required=True)
+    ap.add_argument("--phase", choices=["dev", "select", "boundary", "test"], required=True)
     ap.add_argument("--root", default="outputs", help="Drive folder (Colab) or local output folder")
     ap.add_argument("--limit", type=int, default=None, help="only the first N units (smoke tests)")
     ap.add_argument("--force", action="store_true", help="select: replace a different configs/rq2_frozen.json")
@@ -186,12 +191,12 @@ def main() -> int:
     if args.phase == "select":
         return do_select(root, args.force)
     frozen = None
-    if args.phase == "test":
+    if args.phase in ("test", "boundary"):
         frozen, msg = load_frozen_for_test(root)
         if frozen is None:
-            print(f"[test] refusing: {msg}")
+            print(f"[{args.phase}] refusing: {msg}")
             return 7
-        print(f"[test] {msg}")
+        print(f"[{args.phase}] {msg}")
     run_phase(args.phase, root, args.limit, frozen)
     return 0
 
diff --git a/src/eem_acc/analysis.py b/src/eem_acc/analysis.py
index 89b6d5e..80a2ba5 100644
--- a/src/eem_acc/analysis.py
+++ b/src/eem_acc/analysis.py
@@ -147,6 +147,64 @@ def _params(row: pd.Series, family: str) -> Any:
     return float(row["m_rate"])
 
 
+def secondary_global_baselines(df: pd.DataFrame, loaded: list[tuple[str, int]], denoms: dict) -> dict[str, Any]:
+    """D35: the closest-to-feasible global fixed W / token bucket counted per SCENARIO (a scenario is passed when the
+    value is feasible in all its loaded cells), then score; reported as a pre-registered secondary P1 comparison,
+    because counting per cell (D34) depends on how many loaded cells each scenario happens to have."""
+    cells = cell_means(df)
+    cells = cells[_loaded_mask(cells, loaded)]
+    scored = score_methods(df, loaded, denoms).set_index("tag")
+    scns = sorted({s for s, _ in loaded})
+    out = {}
+    for fam in ("fixed", "tokenbucket"):
+        rows = []
+        for tag, r in scored[scored["m_kind"] == fam].iterrows():
+            if not r["complete"]:
+                continue
+            ok_scn = sum(bool(cells[(cells["tag"] == tag) & (cells["scenario"] == s)]["cell_feasible"].all()) for s in scns)
+            rows.append({"tag": tag, "n_feasible_scenarios": ok_scn, "score": float(r["score"]),
+                         "mean_slo_violation": float(r["mean_slo_violation"]), **{c: r[c] for c, _ in FAMILY_ORDER[fam]}})
+        if not rows:
+            continue
+        t = pd.DataFrame(rows)
+        order = [c for c, _ in FAMILY_ORDER[fam]]
+        best = t.sort_values(["n_feasible_scenarios", "score", "mean_slo_violation"] + order,
+                             ascending=[False, False, True] + [a for _, a in FAMILY_ORDER[fam]]).iloc[0]
+        out[fam] = {"value": int(best["m_w"]) if fam == "fixed" else float(best["m_rate"]), "tag": best["tag"],
+                    "n_feasible_scenarios": int(best["n_feasible_scenarios"]), "n_scenarios": len(scns),
+                    "feasible": bool(best["n_feasible_scenarios"] == len(scns)), "score": float(best["score"])}
+    return out
+
+
+def dev_margins(df: pd.DataFrame, tag: str, loaded: list[tuple[str, int]]) -> dict[str, Any]:
+    """M4 (pre-registered 2026-11-14): per loaded cell, the frozen ACC's dev seed-mean and worst-seed values of the three
+    constraint metrics and the margins to the limits. If the frozen ACC is infeasible in a TEST cell, that is
+    reported as-is; there is no re-selection."""
+    d = _worst_fill(df[df["tag"] == tag])
+    out = {}
+    for s, m in loaded:
+        c = d[(d["scenario"] == s) & (d["M"] == m)]
+        if c.empty:
+            continue
+        out[f"{s}|{m}"] = {
+            "slo_mean": float(c["slo_violation_rate"].mean()), "slo_worst_seed": float(c["slo_violation_rate"].max()),
+            "success_mean": float(c["call_success_rate"].mean()), "success_worst_seed": float(c["call_success_rate"].min()),
+            "overload_mean": float(c["throttle_rate"].mean()), "overload_worst_seed": float(c["throttle_rate"].max()),
+            "margin_slo": S.MAX_SLO_VIOLATION - float(c["slo_violation_rate"].mean()),
+            "margin_success": float(c["call_success_rate"].mean()) - S.MIN_CALL_SUCCESS,
+            "margin_overload": S.MAX_THROTTLE_RATE - float(c["throttle_rate"].mean())}
+    return out
+
+
+def boundary_report(dev: pd.DataFrame, boundary: pd.DataFrame, frozen: dict) -> pd.DataFrame:
+    """D37: frozen ACC vs the one-step-past-the-edge variants on DEV seeds (sensitivity only)."""
+    loaded = [(s, int(m)) for s, m in (x.split("|") for x in frozen["criterion"]["loaded_cells"])]
+    both = pd.concat([dev[dev["tag"] == S.method_tag(S.acc_method(**frozen["acc"]))], boundary])
+    sc = score_methods(both, loaded, frozen["denominators"])
+    return sc[["tag", "score", "feasible", "n_feasible_cells", "mean_slo_violation", "mean_throttle_rate",
+               "mean_call_success"]].reset_index(drop=True)
+
+
 def pareto_front(scored: pd.DataFrame) -> list[dict[str, Any]]:
     """Non-dominated methods on (higher score, lower mean SLO violation), complete methods only."""
     t = scored[scored["complete"]].dropna(subset=["score"])
@@ -191,8 +249,10 @@ def closest_to_feasible(scored: pd.DataFrame, family: str, loaded: list[tuple[st
     t = scored[(scored["m_kind"] == family) & scored["complete"]]
     if t.empty:
         return None
-    row = t.sort_values(["n_feasible_cells", "score", "mean_slo_violation", "tag"],
-                        ascending=[False, False, True, True]).iloc[0]
+    order = [c for c, _ in FAMILY_ORDER[family] if c in t.columns]
+    asc = [a for c, a in FAMILY_ORDER[family] if c in t.columns]
+    row = t.sort_values(["n_feasible_cells", "score", "mean_slo_violation"] + order,
+                        ascending=[False, False, True] + asc).iloc[0]
     score = float(row["score"]) if pd.notna(row["score"]) else None
     return {"value": _params(row, family), "tag": row["tag"], "feasible": False,
             "n_feasible_cells": int(row["n_feasible_cells"]), "n_loaded_cells": len(loaded),
@@ -231,7 +291,11 @@ def select_frozen(df: pd.DataFrame) -> dict[str, Any]:
             c = k1[k1["complete"]] if k1["complete"].any() else k1
             row = c.sort_values(["n_feasible_cells", "score", "mean_slo_violation", "tag"],
                                 ascending=[False, False, True, True]).iloc[0]
-        res["acc_kappa1"] = {"config": _params(row, "acc"), "feasible": bool(feasible), "score": float(row["score"])}
+        res["acc_kappa1"] = {"config": _params(row, "acc"), "feasible": bool(feasible), "score": float(row["score"]),
+                             "n_feasible_cells": int(row["n_feasible_cells"]), "n_loaded_cells": len(loaded)}
+    res["secondary_global_baselines"] = secondary_global_baselines(df, loaded, res["denominators"])
+    if "acc" in res:
+        res["acc_dev_margins"] = dev_margins(df, S.method_tag(S.acc_method(**res["acc"])), loaded)
     res["pareto_front"] = pareto_front(scored)
     res["loso"] = leave_one_scenario_out(df)
     return res
@@ -256,15 +320,17 @@ def leave_one_scenario_out(dev: pd.DataFrame) -> dict[str, Any]:
         dn = denominators(cells, held)
         for fam, key in (("acc", "acc"), ("fixed", "best_fixed_w"), ("tokenbucket", "best_pacer_rate")):
             if key not in sel:
-                ev[fam] = {"config": None, "training_ok": bool(sel["ok"]),
-                           "reason": "no feasible candidate on the training scenarios"}
+                ev[fam] = {"config": None, "feasible_on_training": False,
+                           "reason": ("no feasible ACC configuration on the training scenarios" if fam == "acc"
+                                      else "no complete candidate on the training scenarios")}
                 continue
             tag = S.method_tag(S.acc_method(**sel[key]) if fam == "acc" else
                                S.fixed_method(sel[key]) if fam == "fixed" else S.pacer_method(sel[key]))
             c = cells[cells["tag"] == tag].copy()
             c["norm"] = [g / dn.get(f"{a}|{int(m)}", np.nan) for g, a, m in zip(c["goodput"], c["scenario"], c["M"])]
-            ev[fam] = {"config": sel[key], "tag": tag, "training_ok": bool(sel["ok"]),
-                       "training_fallback": sel.get("baseline_fallback", {}).get(fam),
+            fb = sel.get("baseline_fallback", {}).get(fam)
+            ev[fam] = {"config": sel[key], "tag": tag, "feasible_on_training": fb is None,
+                       "training_fallback": fb,
                        "score_on_left_out": float(c["norm"].mean()) if len(c) else None,
                        "feasible_on_left_out": bool(c["cell_feasible"].all()) if len(c) else None,
                        "mean_slo_violation": float(c["slo_violation_rate"].mean()) if len(c) else None,
@@ -284,8 +350,10 @@ def robustness_matrix(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
     dn = frozen["denominators"]
     cells["norm"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(cells["goodput"], cells["scenario"], cells["M"])]
     fb = frozen.get("baseline_fallback", {})
-    w_lab = " (global, closest to feasible: infeasible in some loaded cells)" if "fixed" in fb else " (global)"
-    r_lab = " (closest to feasible: infeasible in some loaded cells)" if "tokenbucket" in fb else ""
+    def _lab(f):
+        return (f" (global, closest to feasible; dev: feasible in {fb[f]['n_feasible_cells']}/{fb[f]['n_loaded_cells']}"
+                f" loaded cells)") if f in fb else " (global)"
+    w_lab, r_lab = _lab("fixed"), _lab("tokenbucket")
     rows = {"ACC (frozen)": S.method_tag(S.acc_method(**frozen["acc"])),
             f"fixed W={frozen['best_fixed_w']}{w_lab}": S.method_tag(S.fixed_method(frozen["best_fixed_w"])),
             f"token bucket {frozen['best_pacer_rate']:g}/s{r_lab}": S.method_tag(S.pacer_method(frozen["best_pacer_rate"]))}
@@ -294,12 +362,20 @@ def robustness_matrix(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
         rows["ACC (frozen, kappa = 1: ablation)"] = S.method_tag(S.acc_method(**{**a, "kappa": 1.0}))
     if frozen.get("acc_kappa1"):
         rows["ACC (best kappa = 1 on dev)"] = S.method_tag(S.acc_method(**frozen["acc_kappa1"]["config"]))
+    sec = frozen.get("secondary_global_baselines", {})
+    for f, mk in (("fixed", S.fixed_method), ("tokenbucket", S.pacer_method)):
+        if f in sec:
+            t = S.method_tag(mk(sec[f]["value"]))
+            name = f"fixed W={sec[f]['value']}" if f == "fixed" else f"token bucket {sec[f]['value']:g}/s"
+            lab = (f"{name} (secondary global, D35; dev: feasible in {sec[f]['n_feasible_scenarios']}/"
+                   f"{sec[f]['n_scenarios']} scenarios)")
+            rows[lab] = t
     by_tag = {t: lab for lab, t in rows.items()}
     for scn, w in frozen["best_fixed_w_per_scenario"].items():
         t = S.method_tag(S.fixed_method(w))
         if t in by_tag:                                   # same method as a row above: one row, merged label
             old = by_tag[t]
-            rows[f"{old}; best for {scn}"] = rows.pop(old)
+            rows = {(f"{k}; best for {scn}" if k == old else k): v for k, v in rows.items()}   # keep row order
             by_tag[t] = f"{old}; best for {scn}"
         else:
             rows[f"fixed W={w} (best for {scn})"] = t
@@ -315,6 +391,26 @@ def robustness_matrix(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
     return pd.DataFrame(out)
 
 
+def test_descriptive(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
+    """D36 (frozen before the test phase): every test method scored like the dev selection, but with the FROZEN dev
+    denominators: score (mean normalised goodput over scored loaded cells), number of loaded cells in which it is
+    feasible on the TEST seeds, mean and worst-cell SLO violation, mean overload rate, mean call success. Descriptive
+    only (no significance tests); this is the static frontier shown next to the frozen ACC."""
+    loaded = [(s, int(m)) for s, m in (x.split("|") for x in frozen["criterion"]["loaded_cells"])]
+    cells = cell_means(test)
+    cells = cells[_loaded_mask(cells, loaded)].copy()
+    dn = frozen["denominators"]
+    cells["norm"] = [g / dn.get(f"{s}|{int(m)}", np.nan) for g, s, m in zip(cells["goodput"], cells["scenario"], cells["M"])]
+    g = cells.groupby("tag")
+    out = pd.DataFrame({"score": g["norm"].mean(), "n_feasible_cells": g["cell_feasible"].sum().astype(int),
+                        "n_cells": g.size(), "mean_slo_violation": g["slo_violation_rate"].mean(),
+                        "worst_cell_slo_violation": g["slo_violation_rate"].max(),
+                        "mean_overload_rate": g["throttle_rate"].mean(), "mean_call_success": g["call_success_rate"].mean()})
+    out["feasible_all_loaded_cells"] = out["n_feasible_cells"] == len(loaded)
+    out["kind"] = g["m_kind"].first() if "m_kind" in cells.columns else None
+    return out.reset_index().sort_values(["kind", "score"], ascending=[True, False]).reset_index(drop=True)
+
+
 def sign_flip_test(d: np.ndarray) -> float:
     """Exact two-sided paired randomisation test of mean(d) = 0: all 2^n sign patterns (n <= 20)."""
     d = np.asarray(d, dtype=float)
@@ -441,4 +537,19 @@ def primary_comparisons(test: pd.DataFrame, frozen: dict) -> pd.DataFrame:
     adj = holm({k: v for k, v in pv.items() if not math.isnan(v)})
     for r in rows:
         r["p_holm"] = adj.get(r["endpoint"], math.nan)
+        r["family"] = "primary"
+    # D35: pre-registered SECONDARY P1 comparisons (outside the Holm family), per-scenario-counted global baselines
+    sec = frozen.get("secondary_global_baselines", {})
+    for key, mk, lab in (("fixed", S.fixed_method, "P1a' ACC - secondary global fixed W"),
+                         ("tokenbucket", S.pacer_method, "P1b' ACC - secondary token bucket")):
+        if key not in sec:
+            continue
+        base = S.method_tag(mk(sec[key]["value"]))
+        a, b = _per_seed(test, acc, scored, dn, "norm_goodput"), _per_seed(test, base, scored, dn, "norm_goodput")
+        d = (a - b).dropna()
+        lo, hi = t_ci(d.values)
+        rows.append({"endpoint": f"{lab}={sec[key]['value']:g}", "expected": "ACC higher", "test": "two-sided",
+                     "n_seeds": len(d), "mean_diff": float(d.mean()) if len(d) else math.nan, "ci95_low": lo,
+                     "ci95_high": hi, "p_value": sign_flip_test(d.values) if len(d) else math.nan,
+                     "p_holm": math.nan, "family": "secondary"})
     return pd.DataFrame(rows)
diff --git a/src/eem_acc/scenarios.py b/src/eem_acc/scenarios.py
index 39419d8..c91ed66 100644
--- a/src/eem_acc/scenarios.py
+++ b/src/eem_acc/scenarios.py
@@ -168,8 +168,9 @@ def dev_methods() -> list[dict]:
 
 def test_methods(frozen: dict) -> list[dict]:
     """Test-phase methods from the frozen dev selection: frozen ACC; the same ACC with kappa = 1 (kappa ablation);
-    the best kappa = 1 ACC config selected on dev (the paper's base algorithm at its best, D27); 3 ablations; fixed {4,16,64}; the global best fixed W; the per-scenario best fixed W values
-    (robustness matrix, D16); the tuned token bucket. Duplicates are removed (order kept)."""
+    the best kappa = 1 ACC config selected on dev (D27); 3 ablations; fixed {4,16,64}; the global best fixed W and
+    token bucket (D34) and the secondary ones (D35); the per-scenario best fixed W values (robustness matrix, D16);
+    and (D36) EVERY dev fixed window and token-bucket rate, as descriptive results. Duplicates removed, order kept."""
     a = frozen["acc"]
     ms = [acc_method(**a)]
     if a.get("kappa", 1.0) != 1.0:
@@ -180,6 +181,12 @@ def test_methods(frozen: dict) -> list[dict]:
     ws = list(FIXED_W_REPORTED) + [frozen["best_fixed_w"]] + sorted(set(frozen["best_fixed_w_per_scenario"].values()))
     ms += [fixed_method(w) for w in ws]
     ms.append(pacer_method(frozen["best_pacer_rate"]))
+    sec = frozen.get("secondary_global_baselines", {})
+    if "fixed" in sec:
+        ms.append(fixed_method(sec["fixed"]["value"]))
+    if "tokenbucket" in sec:
+        ms.append(pacer_method(sec["tokenbucket"]["value"]))
+    ms += [fixed_method(w) for w in FIXED_W_DEV] + [pacer_method(r) for r in PACER_RATES_DEV]   # D36
     seen, out = set(), []
     for m in ms:
         t = method_tag(m)
@@ -189,6 +196,18 @@ def test_methods(frozen: dict) -> list[dict]:
     return out
 
 
+BOUNDARY_STEPS = dict(alpha=0.125, beta=0.125, epsilon=0.025, n_c=64, kappa=0.7)   # one step past each grid edge (D37)
+
+
+def boundary_methods(frozen: dict) -> list[dict]:
+    """D37: the frozen ACC with ONE parameter moved one step past the grid edge it sits on (dev seeds only,
+    sensitivity analysis; never used to re-select)."""
+    a = frozen["acc"]
+    edge = {"alpha": min(ACC_GRID["alpha"]), "beta": min(ACC_GRID["beta"]), "epsilon": min(ACC_GRID["epsilon"]),
+            "n_c": max(ACC_GRID["n_c"]), "kappa": min(ACC_GRID["kappa"])}
+    return [acc_method(**{**a, k: BOUNDARY_STEPS[k]}) for k in BOUNDARY_STEPS if a[k] == edge[k]]
+
+
 def units(methods: list[dict], seeds: tuple[int, ...]) -> list[tuple[str, str, int, dict, int]]:
     """(unit_id, scenario, M, method, seed) in a fixed order (light cells first within a seed)."""
     out = []
```
