import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from eem_acc import scenarios as S
from eem_acc.monitor import fmt_duration, snapshot, summary_line
from eem_acc.sim import simulate
from eem_acc.viz import demo_figure, monitor_figure

_spec = importlib.util.spec_from_file_location(
    "run_rq2", Path(__file__).resolve().parents[1] / "experiments" / "run_rq2.py")
run_rq2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(run_rq2)


def test_simulate_returns_gateway_with_logs():
    gw = simulate(S.build("CAP-steady", 4, S.fixed_method(4), 0))
    assert len(gw.calls) == 4 * S.CALLS_PER_AGENT and gw.attempts


def test_snapshot_empty_then_partial(tmp_path):
    snap = snapshot(tmp_path, "dev")
    assert snap["done"] == 0 and snap["total"] == 5520 and not snap["running"]
    run_rq2.run_phase("dev", tmp_path, limit=20)
    snap = snapshot(tmp_path, "dev")
    assert snap["done"] == 20
    assert sum(snap["cell_done"].values()) == 20
    assert sum(snap["cell_total"].values()) == 5520
    assert snap["cell_done"][("CAP-steady", 4)] == 20            # first 20 units are all in that cell
    assert "20/5520" in summary_line(snap)


def test_snapshot_running_state_and_eta(tmp_path):
    import os
    run_rq2.run_phase("dev", tmp_path, limit=10)
    files = sorted((tmp_path / "results" / "rq2" / "dev" / "units").glob("*.json"))
    t0 = 1_000_000.0
    for i, f in enumerate(files):
        os.utime(f, (t0 + i, t0 + i))                            # one result per second
    snap = snapshot(tmp_path, "dev", now=t0 + 12)
    assert snap["running"] and abs(snap["rate_per_s"] - 1.0) < 1e-9
    # all 10 finished runs have M=4 -> weighted throughput 4 work-units/s; remaining work = sum(M) of the rest
    total_w = sum(u[2] for u in S.units(S.dev_methods(), S.DEV_SEEDS))
    assert abs(snap["eta_s"] - (total_w - 10 * 4) / 4.0) < 1e-6
    assert snap["eta_s"] > (5520 - 10) * 2                       # far above the naive runs/s estimate
    assert not snapshot(tmp_path, "dev", now=t0 + 500)["running"]   # idle for > 60 s


def test_fmt_duration():
    assert fmt_duration(None) == "?" and fmt_duration(65) == "1m05s" and fmt_duration(3725) == "1h02m05s"


def test_monitor_figure_renders(tmp_path):
    run_rq2.run_phase("dev", tmp_path, limit=5)
    fig = monitor_figure(snapshot(tmp_path, "dev"))
    fig.savefig(tmp_path / "monitor.png", dpi=60)
    assert (tmp_path / "monitor.png").stat().st_size > 1000


def test_demo_figure_renders(tmp_path):
    fig = demo_figure()
    assert len(fig.axes) == 3
    fig.savefig(tmp_path / "demo.png", dpi=60)
    assert (tmp_path / "demo.png").stat().st_size > 1000
