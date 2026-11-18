"""Static checks of the Colab notebooks (stdlib json only, so they run in the fast suite)."""

import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SRC_LINE = 'sys.path.insert(0, f"{REPO_DIR}/src")'


def _code(name):
    nb = json.loads((NB_DIR / f"{name}.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("name", ["00_setup", "01_rq2", "02_live", "03_s30_feasibility"])
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


def test_s30_notebook_pins_llama_cpp_and_targets_the_t4():
    """S3.0 (design r4 section 9: llama.cpp build is part of the integrity stamp): exact version, CUDA arch of the T4,
    wheel cached on Drive, every section saved atomically so a disconnect loses nothing."""
    import ast
    code = _code("03_s30_feasibility")
    ast.parse(code)
    assert 'LLAMA_VERSION = "0.3.36"' in code and "-DCMAKE_CUDA_ARCHITECTURES=75" in code
    assert "WHEEL_DIR" in code and "atomic_write_json" in code
    for section in ("timing", "determinism", "toy"):
        assert f'if not done("{section}")' in code
    assert "sha256" in code and "repo_commit" in code and "STAMP" in code and "os.replace(tmp" in code
    assert code.index('if not done("toy")') < code.index('if not done("determinism")')   # replay needs the toy run
    assert "llm.close" in code or "].close()" in code


def test_s30_wheel_selection_accepts_the_source_build_tag(tmp_path):
    """Regression (2026-11-18, user's Colab run): a source build of llama-cpp-python is tagged py3-none-linux_x86_64
    (ctypes, no CPython extension); the cell looked for cpXY only, so the 62-min build was never copied to Drive."""
    import zipfile
    nb = json.loads((NB_DIR / "03_s30_feasibility.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "Install llama-cpp-python" in "".join(c["source"]))
    ns = {"Path": Path, "zipfile": zipfile, "LLAMA_VERSION": "0.3.36", "PYTAG": "cp312", "WHEEL_DIR": tmp_path}
    exec(cell[cell.index("def compatible"):cell.index("def installed_version")], ns)
    for name in ["llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl", "llama_cpp_python-0.3.36-cp312-cp312-linux_x86_64.whl",
                 "llama_cpp_python-0.3.36-cp311-cp311-linux_x86_64.whl", "llama_cpp_python-0.3.35-py3-none-linux_x86_64.whl"]:
        with zipfile.ZipFile(tmp_path / name, "w") as z:
            z.writestr("x", "y")
    (tmp_path / "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl.tmp").write_text("partial copy")
    (tmp_path / "llama_cpp_python-0.3.36-py3-none-manylinux_2_17_aarch64.whl").write_text("not a zip")
    assert [w.name for w in ns["good_wheels"]()] == ["llama_cpp_python-0.3.36-cp312-cp312-linux_x86_64.whl",
                                                    "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl"]
    (tmp_path / "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl").write_text("truncated")    # broken zip
    assert [w.name for w in ns["good_wheels"]()] == ["llama_cpp_python-0.3.36-cp312-cp312-linux_x86_64.whl"]
    assert "save_to_drive" in cell and 'raise RuntimeError(f"llama-cpp-python' in cell
