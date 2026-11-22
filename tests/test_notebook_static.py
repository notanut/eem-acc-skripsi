"""Static checks of the Colab notebooks (stdlib json only, so they run in the fast suite)."""

import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SRC_LINE = 'sys.path.insert(0, f"{REPO_DIR}/src")'


def _code(name):
    nb = json.loads((NB_DIR / f"{name}.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("name", ["00_setup", "01_rq2", "02_live", "03_s30_feasibility", "04_stage3"])
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
    assert "sha256" in code and "repo_commit" in code and "STAMP" in code and "FEAS_SHA256" in code and "os.replace(tmp" in code
    assert code.index('if not done("toy")') < code.index('if not done("determinism")')   # replay needs the toy run
    assert "llm.close" in code or "].close()" in code


def test_s30_wheel_selection_accepts_the_source_build_tag(tmp_path):
    """Regression (2026-11-18, user's Colab run): a source build of llama-cpp-python is tagged py3-none-linux_x86_64
    (ctypes, no CPython extension); the cell looked for cpXY only, so the 62-min build was never copied to Drive."""
    import zipfile
    nb = json.loads((NB_DIR / "03_s30_feasibility.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "Install llama-cpp-python" in "".join(c["source"]))
    import zlib
    ns = {"Path": Path, "zipfile": zipfile, "zlib": zlib, "LLAMA_VERSION": "0.3.36", "PYTAG": "cp312", "WHEEL_DIR": tmp_path}
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


def _s30_install_cell():
    nb = json.loads((NB_DIR / "03_s30_feasibility.ipynb").read_text())
    return next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "Install llama-cpp-python" in "".join(c["source"]))


@pytest.mark.parametrize("state", ["fresh", "built_not_saved", "on_drive", "installed_and_cached",
                                   "installed_nothing_cached", "other_python_only"])
def test_s30_install_cell_states(tmp_path, state):
    """Cell 2 run with a fake subprocess in every state the review listed: builds only when nothing usable exists and
    the library is not installed; never claims success without an importable 0.3.36."""
    import hashlib, os, shutil, subprocess, sys, time, zipfile
    drive, build = tmp_path / "drive", tmp_path / "build"
    (drive / "wheels").mkdir(parents=True)
    build.mkdir()
    py3 = "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl"
    def make(folder, name):
        with zipfile.ZipFile(folder / name, "w") as z:
            z.writestr("llama_cpp/__init__.py", "")
    if state == "built_not_saved":
        make(build, py3)
    if state in ("on_drive", "installed_and_cached"):
        make(drive / "wheels", py3)
    if state == "other_python_only":
        make(drive / "wheels", "llama_cpp_python-0.3.36-cp39-cp39-linux_x86_64.whl")
    log = {"installed": state.startswith("installed"), "built": 0, "pip_install": 0}

    class R:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, "" if rc == 0 else "ModuleNotFoundError: llama_cpp"

    def fake_run(cmd, **kw):
        if cmd[1] == "-c":
            return R(0, "0.3.36\n") if log["installed"] else R(1)
        if cmd[3] == "wheel":
            log["built"] += 1
            make(Path(cmd[cmd.index("-w") + 1]), py3)
        elif cmd[3] == "install" and cmd[-1].endswith(".whl"):
            log["pip_install"] += 1
            log["installed"] = True
        return R(0)

    fake_subprocess = type(sys)("fake_subprocess")
    fake_subprocess.run = fake_run
    ns = {"sys": sys, "os": os, "time": time, "shutil": shutil, "hashlib": hashlib, "glob": __import__("glob"),
          "Path": Path, "subprocess": fake_subprocess, "IN_COLAB": True, "DRIVE_ROOT": str(drive),
          "BUILD_DIR_OVERRIDE": str(build)}
    exec(_s30_install_cell(), ns)
    assert log["installed"]
    assert log["built"] == (1 if state in ("fresh", "other_python_only") else 0)
    assert log["pip_install"] == (0 if state.startswith("installed") else 1)
    assert (ns["WHEEL_SHA256"] is None) == (state == "installed_nothing_cached")
    if state != "installed_nothing_cached":
        assert (drive / "wheels" / py3).exists()


def test_s30_install_cell_raises_if_library_not_importable(tmp_path):
    import hashlib, os, shutil, sys, time, zipfile
    drive = tmp_path / "drive"
    (drive / "wheels").mkdir(parents=True)
    with zipfile.ZipFile(drive / "wheels" / "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl", "w") as z:
        z.writestr("x", "y")
    class R:
        returncode, stdout, stderr = 1, "", "ImportError: libcudart.so.12: cannot open shared object file"
    fake = type(sys)("fake_subprocess")
    fake.run = lambda cmd, **kw: R()
    ns = {"sys": sys, "os": os, "time": time, "shutil": shutil, "hashlib": hashlib, "glob": __import__("glob"),
          "Path": Path, "subprocess": fake, "IN_COLAB": True, "DRIVE_ROOT": str(drive),
          "BUILD_DIR_OVERRIDE": str(tmp_path / "b")}
    with pytest.raises(RuntimeError, match="libcudart"):
        exec(_s30_install_cell(), ns)


def test_s30_legacy_stamp_maps_to_that_commits_measurement_code():
    """Sections saved by the first notebook version (stamped with the repo commit) are reused only because that
    commit's feasibility.py is byte-identical to the hash recorded here."""
    import hashlib, re, shutil, subprocess
    if shutil.which("git") is None:
        pytest.skip("git not available")
    code = _code("03_s30_feasibility")
    commit, sha = re.search(r'LEGACY_COMMITS = \{"([0-9a-f]{40})": "([0-9a-f]{64})"\}', code).groups()
    repo = NB_DIR.parent
    r = subprocess.run(["git", "show", f"{commit}:src/eem_acc/stage3/feasibility.py"], cwd=repo, capture_output=True)
    if r.returncode != 0:
        pytest.skip("commit not in this clone")
    assert hashlib.sha256(r.stdout).hexdigest() == sha


def test_s33b_notebook_structure():
    """S3.3b (docs/s33b_spec.md r2): pinned tag checkout and a clean tree in every runtime (never git pull), the S3.0
    wheel cell reused, sessions guarded, kills only through t4.colab_kill and armed by Drive state, the model closed
    before benchmark C, workers as fresh interpreters, results downloaded at the end."""
    import ast
    code = _code("04_stage3")
    ast.parse(code)
    assert 'git("checkout", "--detach", target)' in code and "SESSION_TAG" in code and '"pull"' not in code
    assert "--untracked-files=no" in code and "assert not dirty" in code
    assert 'LLAMA_VERSION = "0.3.36"' in code and "WHEEL_SHA256" in code
    assert code.count("if SESSION == 1") >= 3 and code.count("if SESSION == 2") >= 5
    assert "t4.KillAfter(SES.backend, t4.KILL_A_AFTER, t4.colab_kill)" in code and "t4.arm_kill_a(RUN, U_C2)" in code
    assert "t4.arm_kill_b(RUN, U_C0, t4.colab_kill)" in code
    assert code.index('S.save("kill_a_drive"') < code.index('out = RUN.run_phase("smoke")')     # read before reopening
    assert code.index("SES.backend.llm.close()") < code.index("subprocess.Popen")
    assert "multiprocessing" not in code and "fork" not in code
    assert "verify_parts" in code and "equal_s30" in code and "files.download" in code
    assert 'transformers is already imported' in code


def test_s33b_notebook_is_generated_from_the_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_nb04", Path(__file__).resolve().parents[1] / "tools" / "make_nb04.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    built = m.build()
    stored = json.loads((NB_DIR / "04_stage3.ipynb").read_text())
    assert [c["source"] for c in built["cells"]] == [c["source"] for c in stored["cells"]]
