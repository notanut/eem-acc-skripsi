"""Regression (2026-11-14, user hit 'Cannot run the event loop while another loop is running' in Colab Part A):
Jupyter/Colab cells execute while an asyncio loop is running in the main thread. run_virtual must still work there."""

import asyncio

import pytest

from eem_acc import scenarios as S
from eem_acc.env import run_virtual
from eem_acc.sim import run_scenario


def _inside_running_loop(fn):
    async def host():
        return fn()          # synchronous call made while a loop is running, exactly like a notebook cell
    return asyncio.run(host())


def test_run_virtual_works_when_a_loop_is_already_running():
    async def sleepy():
        await asyncio.sleep(3600)
        return asyncio.get_running_loop().time()
    assert _inside_running_loop(lambda: run_virtual(sleepy())) == pytest.approx(3600.0)


def test_scenario_result_identical_inside_and_outside_a_running_loop():
    sc = S.build("CAP-drop", 16, S.acc_method(**S.ACC_START), 3)
    assert _inside_running_loop(lambda: run_scenario(sc)) == run_scenario(sc)


def test_errors_propagate_from_the_worker_thread():
    async def boom():
        raise ValueError("inside the simulation")
    with pytest.raises(ValueError, match="inside the simulation"):
        _inside_running_loop(lambda: run_virtual(boom()))


def test_demo_figure_works_inside_running_loop():
    import matplotlib
    matplotlib.use("Agg")
    from eem_acc.viz import demo_figure
    fig = _inside_running_loop(lambda: demo_figure("CAP-drop", 16, seed=0, fixed_w=16))
    assert fig is not None
