"""Revert each review fix in a temp copy and check that the test suite catches it.
Usage: python tools/mutation_check.py <scratch_dir>   (every line should start with CAUGHT)"""
import shutil, subprocess, sys, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
W = pathlib.Path(sys.argv[1]) / "work"
M = {
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
for name, (f, old, new) in M.items():
    if W.exists(): shutil.rmtree(W)
    shutil.copytree(REPO, W, ignore=shutil.ignore_patterns(".git", "__pycache__", "outputs"))
    p = W / f; s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=W,
                       env={"PYTHONPATH": "src:.", "MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1]
    print(("CAUGHT " if r.returncode else "MISSED ") + f"{name:28s} | {last}")
