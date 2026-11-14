"""Static checks of the Colab notebooks (stdlib json only, so they run in the fast suite)."""

import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SRC_LINE = 'sys.path.insert(0, f"{REPO_DIR}/src")'


def _code(name):
    nb = json.loads((NB_DIR / f"{name}.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("name", ["00_setup", "01_rq2", "02_live"])
def test_colab_setup_makes_package_importable_without_restart(name):
    """Regression (2026-11-14, user hit ModuleNotFoundError in Colab): `pip install -e` inside a running kernel is not
    importable until a restart (its .pth is read only at interpreter start), so every notebook that pip-installs the
    repo must put src/ on sys.path at top level (also in Colab), BEFORE the first eem_acc import."""
    code = _code(name)
    assert "pip" in code and "-e" in code
    lines = code.splitlines()
    i_path = next(i for i, l in enumerate(lines) if l.strip() == SRC_LINE)
    guard = lines[i_path - 1]
    assert guard.startswith("if ") and not guard.startswith(" "), "src/ must be added at top level, also in Colab"
    i_import = next(i for i, l in enumerate(lines) if "eem_acc" in l and "import" in l)
    assert i_path < i_import


def test_live_notebook_warns_to_restart_after_pulling_new_code():
    code = _code("02_live")
    assert "rev-parse" in code and "Restart session" in code
