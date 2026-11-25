"""S3.3b session-2 embedder setup (D142-D145; docs/s33b_spec.md section 11) against a fake pip / Colab image.

The fake applies pip's commands to a dict of installed distributions, serves wheel files for `pip download`, refuses
`--no-index` installs it cannot satisfy from `--find-links`, and answers the embedder probe from the installed state.
Real pip resolution and the real embedder run only on Colab (session 2)."""
import json
import sys
from pathlib import Path

import pytest

from eem_acc.stage3.llm import embed_setup as ES
from eem_acc.stage3.llm.embed import CPU_PATHS_ENV, cpu_paths_tag
from eem_acc.stage3.llm.t4 import Sections

PY = sys.executable
COLAB = {"torch": "2.9.0+cu126", "torchvision": "0.24.0+cu126", "transformers": "5.22.1", "huggingface-hub": "1.33.0",
         "tokenizers": "0.22.1", "safetensors": "0.6.2", "scikit-learn": "1.7.2", "scipy": "1.16.3", "numpy": "2.4.6",
         "timm": "1.0.20"}
REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PINS = {"minilm_revision": REV}


class World:
    """One Colab VM: installed distributions, the package indexes, the environment and the CPU."""
    n = 0

    def __init__(self, tmp, latest=None, cpu_torch="2.9.1+cpu", vectors="v-avx2", capability="AVX2", installed=None):
        self.inst = dict(COLAB if installed is None else installed)
        self.latest = {"sentence-transformers": "6.3.0", **(latest or {})}
        self.cpu_torch = cpu_torch
        self.vectors, self.capability = vectors, capability
        self.environ = {}
        self.calls = []
        self.minilm_downloads = 0
        self.probe_stdout = None
        self.fail_install = 0                              # fail the n-th pip install (1-based), once
        self.fail_download = False
        self.libtorch = None                               # a fake libtorch_cuda.so path (D145 ldd)
        self.tmp = Path(tmp)
        World.n += 1
        self.vm = f"vm{World.n}"                          # a unique runtime id per VM

    # -- the shell
    def sh(self, argv):
        self.calls.append(argv)
        if argv[:2] == [PY, "-c"]:
            ok = "+cpu" in (self.inst.get("torch") or "")
            return (0, "", "") if ok else (1, "", "Traceback ...\nImportError: libtorch_cuda.so: undefined symbol: ncclCommResume")
        if argv[:3] == [PY, "-m", "eem_acc.stage3.llm.embed"]:
            return self._probe()
        if argv[0] == "ldd":
            return 0, "\tlibnccl.so.2 => /usr/lib/libnccl.so.2 (0x7f)\n\tlibc.so.6 => /lib/libc.so.6\n", ""
        assert argv[:3] == [PY, "-m", "pip"], argv
        args = argv[3:]
        if args[0] == "uninstall":
            for n in args[3:]:
                self.inst.pop(n, None)
            return 0, "", ""
        if args[0] == "check":
            return 1, "timm 1.0.20 requires torchvision, which is not installed.\n", ""
        if args[0] == "download":
            if self.fail_download:
                return 1, "", "ERROR: Could not find a version that satisfies the requirement (sdist only)"
            return self._download(args)
        if args[0] == "install":
            n = len([c for c in self.calls if c[3:4] == ["install"]])
            if n == self.fail_install:
                self.fail_install = 0
                return 1, "", "ERROR: network unreachable"
            return self._install(args)
        raise AssertionError(args)

    def _opts(self, args):
        index = args[args.index("--index-url") + 1] if "--index-url" in args else None
        links = Path(args[args.index("--find-links") + 1]) if "--find-links" in args else None
        skip = {"--index-url", "--find-links", "-d"}
        specs, i = [], 1
        while i < len(args):
            if args[i] in skip:
                i += 2
                continue
            if not args[i].startswith("-") and args[i] != ":all:":
                specs.append(args[i])
            i += 1
        return index, links, specs

    def _resolve(self, name, ver, index):
        if name == "torch":
            if index != ES.CPU_TORCH_INDEX:
                return None                               # PyPI only has CUDA torch: not offered here
            return ver or self.cpu_torch
        if index == ES.CPU_TORCH_INDEX:
            return None
        return ver or self.latest.get(name)

    def _install(self, args):
        if "--dry-run" in args:                           # resolve only: the installed state is unchanged
            keep = dict(self.inst)
            try:
                return self._apply([a for a in args if a != "--dry-run"])
            finally:
                self.inst = keep
        return self._apply(args)

    def _apply(self, args):
        index, links, specs = self._opts(args)
        assert ES.TORCH_HOLD not in specs
        for spec in specs:
            name, _, ver = spec.partition("==")
            if "--no-index" in args:
                have = {(ES.norm(f.name.split("-")[0]), f.name.split("-")[1]) for f in links.glob("*.whl")}
                if self.inst.get(name) != ver and (ES.norm(name), ver) not in have:
                    return 1, "", f"ERROR: No matching distribution found for {spec}"
                self.inst[name] = ver
                continue
            if ver and self.inst.get(name) == ver:
                continue
            if not ver and self.inst.get(name):
                continue                                  # already satisfied
            got = self._resolve(name, ver or None, index)
            if got is None:
                return 1, "", f"ERROR: No matching distribution found for {spec}"
            self.inst[name] = got
            if name == "sentence-transformers":
                self.inst.setdefault("tqdm", "4.67.1")    # a dependency Colab lacked
        return 0, "", ""

    def _download(self, args):
        index, _, specs = self._opts(args)
        out = Path(args[args.index("-d") + 1])
        assert "--no-deps" in args
        for spec in specs:
            name, _, ver = spec.partition("==")
            if "+cpu" in ver and index != ES.CPU_TORCH_INDEX:
                return 1, "", f"ERROR: No matching distribution found for {spec}"
            (out / f"{name.replace('-', '_')}-{ver}-py3-none-any.whl").write_bytes(f"{name} {ver}".encode())
        return 0, "", ""

    def _probe(self):
        if self.probe_stdout is not None:
            return 0, self.probe_stdout, ""
        t = self.inst.get("torch") or ""
        p = {"torch": t, "torch_cuda": None if "+cpu" in t else "12.6",
             "sentence-transformers": self.inst.get("sentence-transformers"), "transformers": self.inst.get("transformers"),
             "num_threads": 1, "cpu_capability": self.capability, "mkl": True, "mkldnn": True,
             "env": {k: self.environ.get(k) for k in CPU_PATHS_ENV}, "embedder_id": "minilm-...-isa-compat-avx2", "dim": 384,
             "vectors_sha256": self.vectors, "cpu": {"model": "Intel Xeon @ 2.00GHz", "avx2": True}}
        return 0, "Loading weights ...\n" + json.dumps(p) + "\n", "UserWarning: something\n"

    # -- the rest of the environment
    def versions(self, names=ES.EMBED_LIBRARIES):
        return {n: self.inst.get(n) for n in names}

    def freeze(self):
        return dict(self.inst)

    def download_minilm(self, local, rev):
        self.minilm_downloads += 1
        for f in ("modules.json", "config.json", "model.safetensors", "tokenizer.json", "1_Pooling/config.json"):
            (local / f).parent.mkdir(parents=True, exist_ok=True)
            (local / f).write_text(f"{f} at {rev}")

    def run(self, sections, attempts, pins=PINS, loaded=(), drive=None):
        return ES.ensure_embed(sections, attempts, pins, drive or self.tmp / "drive", self.tmp / f"local-{self.vm}",
                               self.sh, self.download_minilm, loaded=loaded, versions=self.versions,
                               freeze=self.freeze, runtime=self.vm, python=PY, environ=self.environ,
                               locate_libtorch=lambda: self.libtorch)


def files(tmp):
    return (Sections(tmp / "session2.json", stamp={"commit": "c0ffee", "nb_version": "abc123"}),
            Sections(tmp / "attempts.json"))


def pip_installs(world):
    return [c[3:] for c in world.calls if c[:4] == [PY, "-m", "pip", "install"] and "--dry-run" not in c]


# ------------------------------------------------------------------------------------------- the whole sequence
def test_first_runtime_resolves_records_and_caches(tmp_path):
    S, A = files(tmp_path)
    w = World(tmp_path)
    e = w.run(S, A)
    assert e["mode"] == "resolve" and e["from"] == "resolved"
    ins = pip_installs(w)
    assert ins[0] == ["install", "-q", "--index-url", ES.CPU_TORCH_INDEX, "torch"]
    assert ins[1][:4] == ["install", "-q", "sentence-transformers", "torch==2.9.1+cpu"]        # torch held (L2)
    assert set(ins[1][4:]) == {f"{n}=={COLAB[n]}" for n in ES.HELD}                           # Colab's kept
    assert w.inst["torch"] == "2.9.1+cpu" and "torchvision" not in w.inst
    assert w.environ == CPU_PATHS_ENV                                                          # D143
    saved = json.loads((tmp_path / "session2.json").read_text())
    lib = saved["libraries"]
    assert saved["stamp"] == {"commit": "c0ffee", "nb_version": "abc123"}
    assert lib["versions"]["sentence-transformers"] == "6.3.0"
    assert lib["colab_torch"]["import_returncode"] == 1 and "ncclCommResume" in lib["colab_torch"]["import_error"][-1]
    assert lib["pip_check"]["returncode"] == 1 and "torchvision" in lib["pip_check"]["output"][0]
    assert lib["freeze"]["tqdm"] == "4.67.1" and set(lib["durations"]) >= {"install", "probe", "minilm", "wheel_cache"}
    # D144: the wheels of everything the install added or changed, sha256-verified on Drive
    wheels = {w_["name"]: w_["version"] for w_ in lib["wheels"]["wheels"].values()}
    assert wheels == {"torch": "2.9.1+cpu", "sentence_transformers": "6.3.0", "tqdm": "4.67.1", "transformers": "5.22.1",
                      "huggingface_hub": "1.33.0", "tokenizers": "0.22.1", "safetensors": "0.6.2", "scikit_learn": "1.7.2",
                      "scipy": "1.16.3", "numpy": "2.4.6"}          # D148: all nine + what the install added
    wdir = tmp_path / "drive" / "wheels" / "embed" / f"{ES.python_tag()}-{ES.versions_key(lib['versions'])}"
    assert ES.read_manifest(wdir)["versions"] == lib["versions"] and len(list(wdir.glob("*.whl"))) == 10
    assert not list((tmp_path / "drive" / "wheels" / "embed").glob("*.tmp-*"))
    mdir = tmp_path / "drive" / "models" / "minilm" / REV
    assert ES.read_manifest(mdir)["revision"] == REV and Path(e["minilm_path"], "model.safetensors").exists()
    assert saved["embed_probes"][0]["equal_first"] and saved["embed_probes"][0]["cpu_capability"] == "AVX2"
    att = json.loads((tmp_path / "attempts.json").read_text())["attempts"]
    assert att[0]["mode"] == "resolve" and "ok" not in att[0] and att[-1]["ok"] is True
    assert "stamp" not in json.loads((tmp_path / "attempts.json").read_text())


def test_later_runtime_installs_offline_exactly_the_recorded_versions(tmp_path):
    S, A = files(tmp_path)
    first = World(tmp_path).run(S, A)
    # a fresh VM after a kill: Colab's CUDA torch again, newer releases on the indexes, another CPU type
    w2 = World(tmp_path, latest={"sentence-transformers": "6.4.0"}, cpu_torch="2.10.0+cpu", vectors="v-other")
    S2, A2 = Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json")
    e2 = w2.run(S2, A2)
    assert e2["mode"] == "offline" and e2["from"] == "session file" and e2["versions"] == first["versions"]
    ins = pip_installs(w2)
    assert len(ins) == 1 and "--no-index" in ins[0] and "--index-url" not in ins[0]
    assert "torch==2.9.1+cpu" in ins[0] and "sentence-transformers==6.3.0" in ins[0] and "tqdm==4.67.1" in ins[0]
    assert w2.minilm_downloads == 0 and Path(e2["minilm_path"], "config.json").exists()       # MiniLM from Drive
    assert not [c for c in w2.calls if c[3:4] == ["download"]]                                  # nothing re-cached
    saved = json.loads((tmp_path / "session2.json").read_text())
    assert saved["libraries"]["runtime"] == first["runtime"]                                    # the first record stays
    assert [r["mode"] for r in saved["libraries_runtimes"]] == ["resolve", "offline"]
    assert [p["equal_first"] for p in saved["embed_probes"]] == [True, False]                  # recorded, not fatal
    assert "colab_torch" not in e2                                                              # recorded once


def test_rerun_in_the_same_runtime_skips_pip_and_records_nothing_new(tmp_path):
    S, A = files(tmp_path)
    w = World(tmp_path)
    w.run(S, A)
    n = len(pip_installs(w))
    e = w.run(S, A, loaded=["torch", "numpy", "sentence-transformers"])
    assert e["mode"] == "in_place" and e["pip"] == [] and len(pip_installs(w)) == n
    saved = json.loads((tmp_path / "session2.json").read_text())
    assert len(saved["libraries_runtimes"]) == 1 and len(saved["embed_probes"]) == 1


def test_record_without_cache_installs_exactly_from_the_internet_and_builds_the_cache(tmp_path):
    S, A = files(tmp_path)
    first = World(tmp_path).run(S, A)
    import shutil
    shutil.rmtree(tmp_path / "drive" / "wheels")
    w2 = World(tmp_path, latest={"sentence-transformers": "6.4.0"}, cpu_torch="2.10.0+cpu")
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "network_exact" and e2["versions"] == first["versions"]
    ins = pip_installs(w2)
    assert ins[0] == ["install", "-q", "--index-url", ES.CPU_TORCH_INDEX, "torch==2.9.1+cpu"]
    assert "torch==2.9.1+cpu" in ins[1] and "sentence-transformers==6.3.0" in ins[1]
    assert ES.read_manifest(tmp_path / "drive" / "wheels" / "embed" /
                            f"{ES.python_tag()}-{ES.versions_key(first['versions'])}") is not None


def test_committed_pins_take_precedence(tmp_path):
    S, A = files(tmp_path)
    S.save("libraries", {"versions": {"torch": "9.9+cpu"}})                  # a stale session record is ignored
    rec = {n: COLAB[n] for n in ES.HELD}
    rec.update({"torch": "2.8.0+cpu", "sentence-transformers": "6.2.0"})
    w = World(tmp_path)
    e = w.run(S, A, pins={**PINS, "libraries": rec})
    assert e["from"] == "configs/t4_pins.json" and e["versions"] == rec and e["mode"] == "network_exact"


# ---------------------------------------------------------------------------------------------- refusals, errors
def test_loaded_modules_block_only_what_would_change(tmp_path):
    S, A = files(tmp_path)
    w = World(tmp_path)
    w.run(S, A, loaded=["numpy", "scikit-learn"])                            # held on the first runtime: fine
    # a later VM whose image has another numpy than the recorded one, with numpy already imported: refuse
    other = dict(COLAB, numpy="2.5.0")
    w2 = World(tmp_path, installed=other)
    with pytest.raises(ES.SetupError, match=r"\['numpy'\] already imported"):
        w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"), loaded=["numpy"])
    assert pip_installs(w2) == [] and not [c for c in w2.calls if c[3:4] == ["uninstall"]]
    # sentence-transformers imported on a first runtime: refuse
    w3 = World(tmp_path / "x")
    (tmp_path / "x").mkdir()
    with pytest.raises(ES.SetupError, match="already imported"):
        w3.run(*files(tmp_path / "x"), loaded=["sentence-transformers"])
    # torch imported before the CPU code paths were set: refuse (D143)
    w4 = World(tmp_path / "x")
    with pytest.raises(ES.SetupError, match="D143"):
        w4.run(*files(tmp_path / "x"), loaded=["torch"])


def test_a_failed_probe_records_nothing_binding(tmp_path):
    for kw, msg in (({"cpu_torch": "2.9.1+cu126"}, "CPU-only"),
                    ({"capability": "AVX512"}, "AVX2 path")):
        d = tmp_path / msg[:3]
        d.mkdir()
        w = World(d, **kw)
        S, A = files(d)
        with pytest.raises(ES.SetupError, match=msg):
            w.run(S, A)
        assert not (d / "session2.json").exists()                          # no record, no commit pinned
        att = json.loads((d / "attempts.json").read_text())["attempts"]
        assert att[-1]["ok"] is False and msg in att[-1]["error"]
        assert not (d / "drive" / "wheels").exists()                       # nothing cached either
    w = World(tmp_path)
    w.probe_stdout = ""
    with pytest.raises(ES.SetupError, match="(?s)probe failed.*Disconnect and delete runtime"):
        w.run(*files(tmp_path))


def test_a_damaged_wheel_cache_is_set_aside_before_pip_and_rebuilt(tmp_path):     # second opinion M1(b)
    S, A = files(tmp_path)
    first = World(tmp_path).run(S, A)
    whl = next((tmp_path / "drive" / "wheels" / "embed").glob("*/torch-*.whl"))
    whl.write_bytes(b"damaged")
    w2 = World(tmp_path)
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "network_exact" and "sha256 mismatch" in e2["broken_cache"]["error"]
    assert not [c for c in w2.calls if c[3:4] == ["install"] and "--no-index" in c]      # decided before pip
    wdir = tmp_path / "drive" / "wheels" / "embed" / f"{ES.python_tag()}-{ES.versions_key(first['versions'])}"
    assert ES.cache_complete(wdir, ES.read_manifest(wdir))                             # rebuilt
    assert e2["versions"] == first["versions"]


def test_a_changed_colab_image_installs_offline_from_the_cache(tmp_path):    # D148 (was: falls back, M1(c))
    S, A = files(tmp_path)
    first = World(tmp_path).run(S, A)
    w2 = World(tmp_path, installed=dict(COLAB, scipy="1.17.0"))           # Colab's scipy moved on
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "offline" and "image_changed" not in e2           # the pinned scipy wheel is cached
    assert e2["versions"] == first["versions"] and w2.inst["scipy"] == "1.16.3"
    assert "broken_cache" not in e2 and not list((tmp_path / "drive" / "wheels" / "embed").glob("*.broken-*"))


def test_preflight_still_falls_back_for_an_uncached_library(tmp_path):       # M1(c) path, cache without scipy
    S, A = files(tmp_path)
    first = World(tmp_path).run(S, A)
    wdir = next(p for p in (tmp_path / "drive" / "wheels" / "embed").iterdir())
    man = ES.read_manifest(wdir)
    f = next(k for k, w in man["wheels"].items() if w["name"] == "scipy")
    (wdir / f).unlink()
    del man["wheels"][f]
    (wdir / "manifest.json").write_text(json.dumps(man))
    w2 = World(tmp_path, installed=dict(COLAB, scipy="1.17.0"))
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "network_exact" and e2["image_changed"] == {"scipy": ["1.17.0", "1.16.3"]}
    assert e2["versions"] == first["versions"]


# ------------------------------------------------------------------------------------------------------ helpers
def test_plans_and_would_change():
    before = {**{n: None for n in ES.EMBED_LIBRARIES + ES.REMOVE}, **{n: COLAB[n] for n in ES.HELD},
              "torch": "2.9.0+cu126", "torchvision": "0.24.0+cu126"}
    p = ES.network_plan(None, before)
    assert p[0] == ["uninstall", "-y", "-q", "torch", "torchvision"] and ES.TORCH_HOLD in p[2]
    assert ES.would_change(None, before) == {"torch", "torchvision", "sentence-transformers"}
    rec = dict(before, torch="2.9.1+cpu", **{"sentence-transformers": "6.3.0"})
    rec = {n: rec[n] for n in ES.EMBED_LIBRARIES}
    assert ES.would_change(rec, before) == {"torch", "torchvision", "sentence-transformers"}
    assert ES.would_change(rec, dict(before, numpy="2.5.0")) >= {"numpy"}
    with pytest.raises(ValueError, match="incomplete"):
        ES.network_plan({**rec, "scipy": None}, before)
    man = {"wheels": {"sentence_transformers-6.3.0-py3-none-any.whl": {"name": "sentence_transformers", "version": "6.3.0"},
                      "tqdm-4.67.1-py3-none-any.whl": {"name": "tqdm", "version": "4.67.1"}}}
    op = ES.offline_plan(rec, man, Path("/w"), before)
    specs = op[-1][5:]
    assert op[-1][:5] == ["install", "-q", "--no-index", "--find-links", "/w"]
    assert len(specs) == len(ES.EMBED_LIBRARIES) + 1 and "tqdm==4.67.1" in specs        # no duplicate spelling
    bad = {"wheels": {"x": {"name": "sentence_transformers", "version": "6.4.0"}}}
    with pytest.raises(ES.SetupError, match="wheel cache has"):
        ES.offline_plan(rec, bad, Path("/w"), before)


def test_loaded_libraries_maps_import_names():
    assert ES.loaded_libraries({"sklearn": 1, "huggingface_hub": 1, "os": 1}) == ["huggingface-hub", "scikit-learn"]
    assert ES.norm("Sentence_Transformers") == ES.norm("sentence-transformers") == "sentence-transformers"


def test_installed_versions_reads_metadata_without_importing():
    from importlib import metadata

    def fake(name):
        if name == "missing-lib":
            raise metadata.PackageNotFoundError(name)
        return {"torch": "2.9.1+cpu"}[name]
    assert ES.installed_versions(("torch", "missing-lib"), fake) == {"torch": "2.9.1+cpu", "missing-lib": None}
    assert ES.installed_versions(("pytest",))["pytest"]                     # real metadata


def test_cpu_paths_tag():
    assert cpu_paths_tag(dict(CPU_PATHS_ENV)) == "isa-compat-avx2"
    assert cpu_paths_tag({}) == "isa-default"
    assert cpu_paths_tag({"MKL_CBWR": "COMPATIBLE"}) == "isa-custom"
    assert cpu_paths_tag({**CPU_PATHS_ENV, "MKL_CBWR": "AVX2,STRICT"}) == "isa-custom"
    assert CPU_PATHS_ENV == {"MKL_CBWR": "COMPATIBLE", "ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2"}


def test_check_embed_install():
    got = {n: "1.0" for n in ES.EMBED_LIBRARIES}
    got["torch"] = "2.9.1+cpu"
    ok = {"torch": "2.9.1+cpu", "torch_cuda": None, "sentence-transformers": "1.0", "transformers": "1.0",
          "num_threads": 1, "vectors_sha256": "s", "dim": 384, "env": dict(CPU_PATHS_ENV), "cpu_capability": "AVX2"}
    env = dict(CPU_PATHS_ENV)
    ES.check_embed_install(ok, got, None, env)
    ES.check_embed_install(ok, got, dict(got), env)
    for bad, msg in (({"torch_cuda": "12.6"}, "CPU-only"), ({"transformers": "2.0"}, "imported"),
                     ({"num_threads": 2}, "probe is wrong"), ({"dim": 0}, "probe is wrong"),
                     ({"env": {}}, "not pinned"), ({"cpu_capability": "AVX512"}, "AVX2 path")):
        with pytest.raises(ES.SetupError, match=msg):
            ES.check_embed_install({**ok, **bad}, got, None, env)
    with pytest.raises(ES.SetupError, match="not installed"):
        ES.check_embed_install(ok, {**got, "scipy": None}, None, env)
    with pytest.raises(ES.SetupError, match="differ from the recorded"):
        ES.check_embed_install(ok, got, {**got, "numpy": "9.9"}, env)
    with pytest.raises(ES.SetupError, match="not pinned"):
        ES.check_embed_install(ok, got, None, {})


def test_ensure_minilm_refusals_and_partial_drive_folder(tmp_path):
    drive, local = tmp_path / "drive" / REV, tmp_path / "local"
    with pytest.raises(ES.SetupError, match="incomplete"):
        ES.ensure_minilm(drive, local, REV, lambda d, r: (d / "config.json").write_text("{}"))
    drive.mkdir(parents=True)                                           # a partial folder without a manifest
    (drive / "junk").write_text("x")
    w = World(tmp_path)
    man = ES.ensure_minilm(drive, local, REV, w.download_minilm)
    assert man["revision"] == REV and not (drive / "junk").exists() and ES.read_manifest(drive) == man
    other = tmp_path / "drive2"
    ES.publish_dir(local, man["files"], other, {**man, "revision": "other"})
    with pytest.raises(ES.SetupError, match="holds revision other"):
        ES.ensure_minilm(other, tmp_path / "l2", REV, w.download_minilm)
    (local / "config.json").write_text("tampered")                      # a bad local copy is replaced from Drive
    ES.ensure_minilm(drive, local, REV, w.download_minilm)
    assert (local / "config.json").read_text() == f"config.json at {REV}"


def test_module_imports_no_heavy_library():
    import subprocess
    code = ("import sys, eem_acc.stage3.llm.embed_setup, eem_acc.stage3.llm.embed;"
            "print([m for m in ('torch','transformers','sentence_transformers','numpy','sklearn') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"



def test_publish_dir_never_replaces_a_published_cache(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.whl").write_bytes(b"one")
    sha = ES.sha256_file(src / "a.whl")
    ES.publish_dir(src, {"a.whl": sha}, tmp_path / "cache", {"v": 1})
    (src / "a.whl").write_bytes(b"two")
    ES.publish_dir(src, {"a.whl": ES.sha256_file(src / "a.whl")}, tmp_path / "cache", {"v": 2})
    assert ES.read_manifest(tmp_path / "cache") == {"v": 1} and (tmp_path / "cache" / "a.whl").read_bytes() == b"one"


# ------------------------------------------------------------------------------------------- code review fixes
def test_a_runtime_that_already_attempted_is_refused(tmp_path):      # review H1
    S, A = files(tmp_path)
    w = World(tmp_path)
    w.fail_install = 2                                                  # the sentence-transformers install fails
    with pytest.raises(ES.SetupError, match="network unreachable"):
        w.run(S, A)
    assert w.inst["torch"] == "2.9.1+cpu"                               # half installed: no longer Colab's image
    with pytest.raises(ES.SetupError, match="already attempted the install"):
        w.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert not (tmp_path / "session2.json").exists() and not (tmp_path / "drive" / "wheels").exists()
    # a fresh VM succeeds and caches torch too
    e = World(tmp_path).run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert {w_["name"] for w_ in e["wheels"]["wheels"].values()} >= {"torch", "sentence_transformers"}


def test_cache_wheels_requires_the_changed_libraries():
    with pytest.raises(ES.SetupError, match=r"did not change \['torch'\]"):
        ES.cache_wheels(lambda a: (0, "", ""), PY, {"torch": "2.9.1+cpu", "numpy": "2.4.6"},
                        {"torch": "2.9.1+cpu", "sentence-transformers": "6.3.0", "numpy": "2.4.6"},
                        {"torch": "2.9.1+cpu", "sentence-transformers": "6.3.0", "numpy": "2.4.6"},
                        Path("/nonexistent-local"), Path("/nonexistent-drive"))


def test_an_incomplete_cache_is_set_aside_and_rebuilt(tmp_path):      # review M1
    S, A = files(tmp_path)
    first = World(tmp_path).run(S, A)
    wdir = tmp_path / "drive" / "wheels" / "embed" / f"{ES.python_tag()}-{ES.versions_key(first['versions'])}"
    next(wdir.glob("torch-*.whl")).unlink()                             # the large wheel never finished uploading
    w2 = World(tmp_path)
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "network_exact" and ".broken-" in e2["broken_cache"]["folder"]
    assert ES.cache_complete(wdir, ES.read_manifest(wdir)) and Path(e2["broken_cache"]["folder"]).exists()


def test_a_cache_failure_is_recorded_not_fatal(tmp_path):             # review M2
    S, A = files(tmp_path)
    w = World(tmp_path)
    w.fail_download = True
    e = w.run(S, A)
    assert "sdist only" in e["wheels"]["error"]
    assert json.loads((tmp_path / "session2.json").read_text())["libraries"]["versions"] == e["versions"]
    e2 = World(tmp_path).run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "network_exact" and "wheels" in e2["wheels"]   # the next runtime builds the cache


def test_colab_torch_evidence_with_ldd(tmp_path):                      # D145
    lib = tmp_path / "libtorch_cuda.so"
    lib.write_bytes(b"x")
    w = World(tmp_path)
    w.libtorch = lib
    e = w.run(*files(tmp_path))
    assert e["colab_torch"]["ldd_nccl"] == ["libnccl.so.2 => /usr/lib/libnccl.so.2 (0x7f)"]
    assert set(e["colab_torch"]["nccl"]) == {"nvidia-nccl-cu12", "nvidia-nccl-cu13"}


def test_stale_temporary_cache_folders_are_removed(tmp_path):         # review L3
    stale = tmp_path / "drive" / "models" / "minilm" / (REV + ".tmp-1-2")
    stale.mkdir(parents=True)
    (stale / "model.safetensors").write_text("half")
    World(tmp_path).run(*files(tmp_path))
    assert not stale.exists()


def test_probe_vectors_is_deterministic_and_timed():
    from eem_acc.stage3.llm.embed import PROBE_TEXTS, probe_vectors

    class E:
        calls = 0

        def embed(self, t):
            E.calls += 1
            return tuple(float(len(t) + i) / 100 for i in range(384))
    a, b = probe_vectors(E()), probe_vectors(E())
    assert a["vectors_sha256"] == b["vectors_sha256"] and a["dim"] == 384 and a["seconds_per_embed"] >= 0
    assert E.calls == 2 * (len(PROBE_TEXTS) + 1)                       # one warm-up call each


def test_a_failure_before_pip_does_not_block_the_runtime(tmp_path):    # second opinion L1
    S, A = files(tmp_path)
    w = World(tmp_path)
    real = w.freeze
    w.freeze = lambda: None                                              # pip list fails: before any pip install
    with pytest.raises(ES.SetupError, match="pip list failed"):
        w.run(S, A)
    assert pip_installs(w) == [] and w.inst["torch"] == "2.9.0+cu126"
    w.freeze = real
    e = w.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))   # same runtime: fine
    assert e["mode"] == "resolve"


def test_a_rerun_after_pip_finished_resumes_without_pip(tmp_path):     # D147 (second opinion J3)
    S, A = files(tmp_path)
    w = World(tmp_path)
    real = w.download_minilm
    w.download_minilm = lambda d, r: (_ for _ in ()).throw(ES.SetupError("Hub: 503 Service Unavailable"))
    with pytest.raises(ES.SetupError, match="503"):
        w.run(S, A)
    assert w.inst["torch"] == "2.9.1+cpu" and not (tmp_path / "session2.json").exists()
    n = len(pip_installs(w))
    w.download_minilm = real
    e = w.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e["mode"] == "resumed" and len(pip_installs(w)) == n and e["pip"]          # no pip; commands recorded
    names = {x["name"] for x in e["wheels"]["wheels"].values()}
    assert {"torch", "sentence_transformers", "tqdm"} <= names                         # diff vs Colab's original
    assert "colab_torch" in json.loads((tmp_path / "attempts.json").read_text())["attempts"][0]
    # the versions changed since pip finished (someone installed something): refuse
    w2 = World(tmp_path / "y")
    (tmp_path / "y").mkdir()
    S2, A2 = files(tmp_path / "y")
    w2.download_minilm = lambda d, r: (_ for _ in ()).throw(ES.SetupError("Hub down"))
    with pytest.raises(ES.SetupError):
        w2.run(S2, A2)
    w2.inst["numpy"] = "9.9"
    with pytest.raises(ES.SetupError, match="changed since pip finished"):
        w2.run(Sections(tmp_path / "y" / "session2.json"), Sections(tmp_path / "y" / "attempts.json"))


def test_a_cut_minilm_upload_is_set_aside_before_pip_and_downloaded_again(tmp_path):   # second opinion M1(a)
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)
    (tmp_path / "drive" / "models" / "minilm" / REV / "model.safetensors").unlink()
    w2 = World(tmp_path)
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert w2.minilm_downloads == 1 and "cache file missing" in e2["broken_minilm"]["error"]
    assert e2["mode"] == "offline" and ES.read_manifest(tmp_path / "drive" / "models" / "minilm" / REV)


def test_setup_logs_one_line_per_step(tmp_path):                       # second opinion J2
    lines = []
    w = World(tmp_path)
    ES.ensure_embed(*files(tmp_path), PINS, tmp_path / "drive", tmp_path / "local", w.sh, w.download_minilm,
                    versions=w.versions, freeze=w.freeze, runtime="vmX", python=PY, environ=w.environ, log=lines.append)
    text = "\n".join(lines)
    for step in ("mode resolve", "pip uninstall", "pip install", "MiniLM", "fresh interpreter", "caching", "done"):
        assert step in text, step



def test_unfinished_pip_is_refused_even_if_versions_match_the_record(tmp_path):   # review M1
    S, A = files(tmp_path)
    first = World(tmp_path).run(S, A)
    import shutil
    shutil.rmtree(tmp_path / "drive" / "wheels")                                   # -> network_exact
    w2 = World(tmp_path)
    S2, A2 = Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json")
    real_install = w2._install

    def install_then_die(args):                       # the packages land, then pip is killed (no pip_done)
        real_install(args)
        if "sentence-transformers==6.3.0" in args:
            return 1, "", "Killed"
        return 0, "", ""
    w2._install = install_then_die
    with pytest.raises(ES.SetupError, match="Killed"):
        w2.run(S2, A2)
    assert w2.versions() == first["versions"]                                       # looks like the record
    with pytest.raises(ES.SetupError, match="pip did not finish"):
        w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))


def test_resumed_network_exact_builds_the_cache_and_keeps_the_evidence(tmp_path):  # review M2
    S, A = files(tmp_path)
    w = World(tmp_path)
    w.libtorch = tmp_path / "libtorch_cuda.so"
    w.libtorch.write_bytes(b"x")
    real = w.download_minilm
    w.download_minilm = lambda d, r: (_ for _ in ()).throw(ES.SetupError("Hub: 503"))
    with pytest.raises(ES.SetupError, match="re-run this cell first"):             # review L2: the right hint
        w.run(S, A)
    w.download_minilm = real
    e = w.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    lib = json.loads((tmp_path / "session2.json").read_text())["libraries"]
    assert e["mode"] == "resumed" and lib["colab_torch"]["import_returncode"] == 1
    assert lib["before"]["torch"] == "2.9.0+cu126" and lib["pip_check"]["returncode"] == 1   # Colab's original
    # a later runtime: network_exact (cache deleted), MiniLM fails once, the re-run resumes and builds the cache
    import shutil
    shutil.rmtree(tmp_path / "drive" / "wheels")
    w2 = World(tmp_path)
    w2.download_minilm = lambda d, r: (_ for _ in ()).throw(ES.SetupError("Drive: I/O error"))
    shutil.rmtree(tmp_path / "drive" / "models")
    S2, A2 = Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json")
    with pytest.raises(ES.SetupError, match="I/O error"):
        w2.run(S2, A2)
    w2.download_minilm = World(tmp_path).download_minilm
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "resumed" and "torch" in {x["name"] for x in e2["wheels"]["wheels"].values()}
    rts = json.loads((tmp_path / "session2.json").read_text())["libraries_runtimes"]
    assert [r["mode"] for r in rts] == ["resumed", "resumed"] and rts[-1]["before"]["torch"] == "2.9.0+cu126"


def test_offline_dry_run_failure_falls_back_before_uninstalling(tmp_path):        # review L3
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)
    w2 = World(tmp_path)
    real = w2._install
    w2._install = lambda args: ((1, "", "ERROR: No matching distribution found for sympy>=1.14")
                                if "--no-index" in args else real(args))
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "network_exact" and "sympy" in e2["offline_dry_run"][-1]
    assert not [c for c in w2.calls if "--no-index" in c and "--dry-run" not in c]


def test_a_new_session_file_needs_the_notebook_fingerprint(tmp_path):            # outdated-notebook guard (code side)
    with pytest.raises(RuntimeError, match="no cell fingerprint"):
        Sections(tmp_path / "session2.json", stamp={"commit": "c"})
    assert not (tmp_path / "session2.json").exists()
    Sections(tmp_path / "session2.json", stamp={"commit": "c", "nb_version": "v"}).save("x", 1)
    assert Sections(tmp_path / "session2.json", stamp={"commit": "c", "nb_version": "v"}).data["x"] == 1   # same copy
    # (another copy, or one without a fingerprint, is refused in later runtimes: test_s33b_t4, third review M1)
    Sections(tmp_path / "attempts.json")                                           # no stamp: no requirement



# ------------------------------------------------------------------------------- third review + session-2 finding
def test_cache_built_when_colab_already_has_sentence_transformers(tmp_path):   # session 2: ST 5.7.0 preinstalled
    colab = dict(COLAB, **{"sentence-transformers": "5.7.0"})
    S, A = files(tmp_path)
    e = World(tmp_path, installed=colab).run(S, A)
    assert e["versions"]["sentence-transformers"] == "5.7.0" and "error" not in e["wheels"]
    assert {w["name"] for w in e["wheels"]["wheels"].values()} == {"torch", "sentence_transformers", "transformers",
                                                                     "huggingface_hub", "tokenizers", "safetensors",
                                                                     "scikit_learn", "scipy", "numpy"}   # D148
    w2 = World(tmp_path, installed=colab)
    e2 = w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e2["mode"] == "offline" and e2["versions"] == e["versions"]


def test_an_offline_cache_that_failed_after_pip_started_is_set_aside(tmp_path):   # third review L2
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)
    w2 = World(tmp_path)
    real = w2._install
    w2._install = lambda args: ((1, "", "ERROR: file conflict while installing torch")
                                if "--no-index" in args and "--dry-run" not in args else real(args))
    with pytest.raises(ES.SetupError, match="file conflict"):
        w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    w3 = World(tmp_path)
    e3 = w3.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e3["mode"] == "network_exact" and "while pip was running" in e3["broken_cache"]["error"]
    # the rebuilt cache (newer than the failure) is used by the next VM: no loop (review of the fix, #1)
    e4 = World(tmp_path).run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e4["mode"] == "offline" and "broken_cache" not in e4
    assert len(list((tmp_path / "drive" / "wheels" / "embed").glob("*.broken-*"))) == 1


def test_a_failure_after_pip_finished_does_not_condemn_the_cache(tmp_path):   # review of the L2 fix, #2
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)
    w2 = World(tmp_path)
    real = w2.download_minilm
    import shutil
    shutil.rmtree(tmp_path / "drive" / "models")                      # MiniLM must be downloaded again ...
    w2.download_minilm = lambda d, r: (_ for _ in ()).throw(ES.SetupError("Hub: 503"))   # ... and the Hub is down
    with pytest.raises(ES.SetupError, match="503"):
        w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    w2.download_minilm = real
    assert w2.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))["mode"] == "resumed"
    e3 = World(tmp_path).run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert e3["mode"] == "offline" and "broken_cache" not in e3


def test_a_drive_error_in_preflight_gets_a_record_and_the_hint(tmp_path, monkeypatch):   # third review L3
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)

    def boom(*a, **k):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(ES, "copy_verified", boom)
    with pytest.raises(ES.SetupError, match="(?s)pre-flight on Drive failed.*Disconnect and delete runtime"):
        World(tmp_path).run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    last = json.loads((tmp_path / "attempts.json").read_text())["attempts"][-1]
    assert last["event"] == "failed" and "Input/output error" in last["error"]



def test_a_drive_error_in_the_minilm_preflight_gets_a_record(tmp_path, monkeypatch):     # third review L3
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)

    def boom(*a, **k):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(ES, "preflight_minilm", boom)
    with pytest.raises(ES.SetupError, match="pre-flight on Drive failed"):
        World(tmp_path).run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert json.loads((tmp_path / "attempts.json").read_text())["attempts"][-1]["event"] == "failed"



# ------------------------------------------------------------------------------------------- fourth review
def test_offline_is_disabled_when_a_rebuilt_cache_fails_too(tmp_path):      # fourth review L1
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)
    modes = []
    for _ in range(5):
        w = World(tmp_path)
        real = w._install
        w._install = lambda args, real=real: ((1, "", "ERROR: No space left on device")
                                              if "--no-index" in args and "--dry-run" not in args else real(args))
        try:
            modes.append(w.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))["mode"])
        except ES.SetupError:
            modes.append("failed")
    assert modes == ["failed", "network_exact", "failed", "network_exact", "network_exact"]
    att = json.loads((tmp_path / "attempts.json").read_text())["attempts"]
    assert any("offline_disabled" in a for a in att if a.get("event") == "start")
    assert len(list((tmp_path / "drive" / "wheels" / "embed").glob("*.broken-*"))) == 1


def test_a_manifest_read_error_is_not_a_missing_cache(tmp_path):           # fourth review L2
    d = tmp_path / "cache"
    d.mkdir()
    assert ES.read_manifest(d) is None                                       # missing
    (d / "manifest.json").write_text("{not json")
    assert ES.read_manifest(d) is None                                       # not JSON
    (d / "manifest.json").unlink()
    (d / "manifest.json").mkdir()                                            # an I/O error (IsADirectoryError)
    with pytest.raises(OSError):
        ES.read_manifest(d)
    S, A = files(tmp_path)
    World(tmp_path).run(S, A)
    wdir = next(p for p in (tmp_path / "drive" / "wheels" / "embed").iterdir())
    (wdir / "manifest.json").unlink()
    (wdir / "manifest.json").mkdir()
    with pytest.raises(ES.SetupError, match="pre-flight on Drive failed"):
        World(tmp_path).run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))
    assert (wdir / "manifest.json").is_dir() and not list(wdir.parent.glob("*.broken-*"))   # nothing set aside



def test_a_read_error_right_after_publishing_is_not_a_failure(tmp_path, monkeypatch):   # fourth-round review, Low 1
    real = ES.read_manifest

    def flaky(folder):
        if Path(folder).name == REV and Path(folder).exists():      # MiniLM on Drive, after it was published
            raise OSError(5, "Input/output error")
        return real(folder)
    monkeypatch.setattr(ES, "read_manifest", flaky)
    e = World(tmp_path).run(*files(tmp_path))
    assert e["minilm"]["revision"] == REV and e["mode"] == "resolve"


def test_a_preflight_error_after_pip_finished_says_rerun(tmp_path, monkeypatch):   # fourth-round review, Low 2
    S, A = files(tmp_path)
    w = World(tmp_path)
    real = w.download_minilm
    w.download_minilm = lambda d, r: (_ for _ in ()).throw(ES.SetupError("Hub: 503"))
    with pytest.raises(ES.SetupError):
        w.run(S, A)
    w.download_minilm = real

    def boom(*a, **k):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(ES, "preflight_minilm", boom)
    with pytest.raises(ES.SetupError, match="re-run this cell first"):
        w.run(Sections(tmp_path / "session2.json"), Sections(tmp_path / "attempts.json"))   # mode resumed
