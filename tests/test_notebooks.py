"""Executes the Colab notebooks headlessly (outside Colab). Slow; excluded from the fast suite.
Run explicitly:  pytest -m notebook   (needs: pip install nbclient ipykernel ipywidgets)
It does NOT prove the notebook works inside Colab (Drive mount, Colab runtime) - that must be checked in Colab."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.notebook
def test_live_notebook_executes_with_small_limit():
    nb = nbformat.read(NB_DIR / "02_live.ipynb", as_version=4)
    n_patched = 0
    for c in nb.cells:
        if "LIMIT = None" in c.source:
            c.source = c.source.replace("LIMIT = None", "LIMIT = 12"); n_patched += 1
    assert n_patched == 1
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_DIR)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors

