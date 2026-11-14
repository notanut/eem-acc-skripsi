"""Revert each review fix in a temp copy and check that the test suite catches it.
Usage: python tools/mutation_check.py <scratch_dir>   (every line should start with CAUGHT)"""
import shutil, subprocess, sys, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
W = pathlib.Path(sys.argv[1]) / "work"
M = {
 # ---- design v2 (2026-11-14)
 "v2: kappa ignored": ("src/eem_acc/acc/controller.py", "return (p95 - self.cfg.l_ctrl) / self.cfg.l_ctrl", "return (p95 - self.cfg.l_target) / self.cfg.l_target"),
 "v2: 504 not in F": ("src/eem_acc/acc/controller.py", "OVERLOAD_STATUSES = (429, 503, 504)", "OVERLOAD_STATUSES = (429, 503)"),
 "v2: timeout kills server work": ("src/eem_acc/agent/gateway.py", "        self._abandon(task)                                # client gives up; the server keeps working", "        task.cancel(); self._abandon(task)"),
 "v2: no shutdown at T": ("src/eem_acc/agent/scripted.py", "    await gw.shutdown()", "    pass"),
 "v2: noise ignores admission": ("src/eem_acc/env/server.py", "z = self._keyed.get(int(agent), int(k) * ADMISSION_STRIDE + a)", "z = self._keyed.get(int(agent), int(k) * ADMISSION_STRIDE)"),
 "v2: burst outside window": ("src/eem_acc/agent/scripted.py", "        if loop.time() >= b.t_end:\n            return", "        if loop.time() >= b.t_end + 5:\n            return"),
 "v2: success grace ignored": ("src/eem_acc/metrics.py", "    pop = [c for c in calls if c[0] < horizon - grace]", "    pop = calls"),
 "v2: 504 not an SLO violation": ("src/eem_acc/metrics.py", "    return a.status == 504 or (a.status == 200 and a.latency > l_target)", "    return a.status == 200 and a.latency > l_target"),
 "v2: feasibility on all-but-one cell": ("src/eem_acc/analysis.py", 'out["feasible"] = out["complete"] & (out["n_feasible_cells"] == len(loaded))', 'out["feasible"] = out["complete"] & (out["n_feasible_cells"] >= len(loaded) - 1)'),
 "v2: no throttle constraint": ("src/eem_acc/analysis.py", "                            & (out[\"throttle_rate\"] <= S.MAX_THROTTLE_RATE))", "                            )"),
 "v2: denominators from any method": ("src/eem_acc/analysis.py", '        c = cells[(cells["scenario"] == s) & (cells["M"] == m) & cells["cell_feasible"]]', '        c = cells[(cells["scenario"] == s) & (cells["M"] == m)]'),
 "v2: no tie tolerance": ("src/eem_acc/analysis.py", '    tied = t[t["score"] >= (1.0 - S.TIE_TOLERANCE) * best]', '    tied = t[t["score"] >= best]'),
 "v2: one-sided sign flip": ("src/eem_acc/analysis.py", "    means = np.abs((signs * d).mean(axis=1))", "    means = (signs * d).mean(axis=1)"),
 "v2: recovery ignores post-event damage": ("src/eem_acc/metrics.py", "            if all(flags[:i]):", "            if all(flags[:max(0, i - 25)]):"),
 "v2: window goodput by request time": ("src/eem_acc/metrics.py", '        "goodput": _completed_ok(calls, t0, t1) / (t1 - t0),', '        "goodput": sum(c[1] for c in cw) / (t1 - t0),'),
 "v2: TOST replaced by two-sided": ("src/eem_acc/analysis.py", "        equivalence = exp == \"about equal\"                      # D25: TOST with +-EQUIVALENCE_MARGIN", "        equivalence = False"),
 "v2: recovery ignores e2e": ("src/eem_acc/metrics.py", "        if e2e_bound is not None:\n            e2e = ", "        if False:\n            e2e = "),
 "v2: kappa1 config not tested": ("src/eem_acc/scenarios.py", "    if frozen.get(\"acc_kappa1\"):", "    if False:"),
 "v2: queue mean unweighted": ("src/eem_acc/metrics.py", "        area += cur * (t - last)", "        area += cur"),
 "release before shrink": ("src/eem_acc/agent/gateway.py",
   "                if self.controller.record(resp.status, resp.latency, loop.time()):\n                    self.slots.set_limit(self.controller.window)\n            finally:\n                self.slots.release()",
   "                self.slots.release()\n                if self.controller.record(resp.status, resp.latency, loop.time()):\n                    self.slots.set_limit(self.controller.window)\n            finally:\n                pass"),
 "window logged at resume": ("src/eem_acc/agent/gateway.py",
   "window_now, in_use = await self.slots.acquire()",
   "await self.slots.acquire(); window_now, in_use = self.slots.limit, self.slots.in_use"),
 "no release on exception": ("src/eem_acc/agent/gateway.py",
   "            finally:\n                self.slots.release()",
   "            except ZeroDivisionError:\n                pass\n            if True:\n                self.slots.release()"),
 "unkeyed jitter": ("src/eem_acc/agent/gateway.py", "if request is None:\n            u = float(self.rng.random())", "if True:\n            u = float(self.rng.random())"),
 "think draw only if mean>0": ("src/eem_acc/agent/scripted.py",
   "        u = float(rng.exponential(1.0))", "        u = float(rng.exponential(1.0)) if _think_mean(wl.phases, loop.time()) > 0 else 0.0"),
 "no floor eps": ("src/eem_acc/acc/controller.py", "_FLOOR_EPS = 1e-9", "_FLOOR_EPS = 0.0"),
 "fname no hash": ("src/eem_acc/utils/results.py", "    if safe != uid:", "    if False:"),
 "load no unit check": ("src/eem_acc/utils/results.py", "        if d.get(\"unit_id\") != str(unit_id):\n            raise", "        if False:\n            raise"),
 "unkeyed server noise": ("src/eem_acc/env/server.py", "        if request is None:\n            z = self.rng", "        if True:\n            z = self.rng"),
 "cancel no release": ("src/eem_acc/agent/gateway.py", "                self.release()          # the slot had", "                pass          # the slot had"),
 "select no stamp check": ("experiments/run_rq2.py", "        hashes = check_results(\"dev\", root, None)",
   "        hashes = {u[0]: '' for u in S.units(S.dev_methods(), S.DEV_SEEDS)}"),
 "resume no stamp check": ("experiments/run_rq2.py", "    check_results(phase, root, frozen)   # refuse", "    pass   # refuse"),
 "test-phase no digest check": ("experiments/run_rq2.py", "    if frozen.get(\"dev_digest\") != digest(dev):", "    if False:"),
 "select overwrite": ("experiments/run_rq2.py", "    if FROZEN_PATH.exists() and not force:", "    if False:"),
}
only = sys.argv[2] if len(sys.argv) > 2 else ""
for name, (f, old, new) in M.items():
    if only and not name.startswith(only):
        continue
    if W.exists(): shutil.rmtree(W)
    shutil.copytree(REPO, W, ignore=shutil.ignore_patterns(".git", "__pycache__", "outputs"))
    p = W / f; s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=W,
                       env={"PYTHONPATH": "src:.", "MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1]
    print(("CAUGHT " if r.returncode else "MISSED ") + f"{name:28s} | {last}")
