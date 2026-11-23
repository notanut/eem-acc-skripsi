"""Session-2 setup of the embedder's libraries and model (docs/s33b_spec.md section 11; D142-D145).

Colab's preinstalled torch (a CUDA build) does not import (session 1: undefined symbol ncclCommResume) and
sentence-transformers is absent. `ensure_embed` therefore, in every runtime and before anything imports torch:

1. sets D143's CPU code paths (`embed.CPU_PATHS_ENV`);
2. replaces Colab's torch with PyTorch's CPU-only build and installs sentence-transformers. Modes:
   - `resolve` (the session's first runtime): newest CPU torch, then sentence-transformers with every other library of
     `EMBED_LIBRARIES` held at Colab's installed version (torch held too);
   - `offline` (D144): exactly the recorded versions from the sha256-verified wheel cache on Drive, `--no-index`;
   - `network_exact`: exactly the recorded versions from the internet (a record without a cache);
   - `in_place`: the recorded versions are already installed (a re-run in the same runtime): no pip;
   - `resumed` (D147): in this runtime pip finished (attempts file) but a later step failed; the installed versions
     equal what pip installed: no pip, the rest continues.
   Before pip touches anything, the Drive caches are checked (second-opinion M1): a broken wheel cache or MiniLM
   snapshot is set aside and rebuilt; a Colab image whose uncached libraries differ from the record falls back to
   `network_exact`. A runtime where pip started and did not finish is refused (a fresh VM is Colab's original image).
   A runtime that installed from the internet caches the wheels of every distribution it added or changed on Drive;
3. puts the MiniLM snapshot at the pinned revision on Drive (sha256 manifest) and copies it to local disk (D144);
4. embeds `embed.PROBE_TEXTS` in a fresh interpreter (`python -m eem_acc.stage3.llm.embed --probe`) and checks it;
5. only then saves `libraries` in the session file (the versions become binding there; also pins the commit), appends
   `libraries_runtimes` for a runtime that installed and `embed_probes` once per runtime.
Every attempt is logged first to a separate attempts file (no stamp, so a failed attempt pins nothing). Before the
first uninstall, why Colab's torch fails is recorded (D145).

Nothing here imports torch, transformers or sentence-transformers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .embed import CPU_PATHS_ENV

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
EMBED_LIBRARIES = ("torch", "sentence-transformers", "transformers", "huggingface-hub", "tokenizers", "safetensors",
                   "scikit-learn", "scipy", "numpy")
HELD = tuple(n for n in EMBED_LIBRARIES if n not in ("torch", "sentence-transformers"))   # Colab's versions kept
REMOVE = ("torch", "torchvision", "torchaudio")          # an installed torch would satisfy "torch" as is
IMPORT_NAMES = {"torch": "torch", "sentence-transformers": "sentence_transformers", "transformers": "transformers",
                "huggingface-hub": "huggingface_hub", "tokenizers": "tokenizers", "safetensors": "safetensors",
                "scikit-learn": "sklearn", "scipy": "scipy", "numpy": "numpy", "torchvision": "torchvision",
                "torchaudio": "torchaudio"}
TORCH_HOLD = "torch==@INSTALLED@"                       # replaced by the CPU torch version just installed
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_FILES = ["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json", "config.json",
                "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
                "1_Pooling/config.json", "README.md"]
HINT = ("Runtime -> Disconnect and delete runtime, reconnect, run from the top; if it happens again, upload "
        "session2_library_attempts.json (and session2.json, if it exists) from Drive Skripsi/stage3/s33b to the chat.")
HINT_RERUN = ("The install itself finished and is kept: re-run this cell first (D147); if it fails again, "
              "Runtime -> Disconnect and delete runtime, reconnect, run from the top, and upload "
              "session2_library_attempts.json from Drive Skripsi/stage3/s33b to the chat.")

Sh = Callable[[list[str]], "tuple[int, str, str]"]      # argv -> (returncode, stdout, stderr)


class SetupError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- inspection
def installed_versions(names: tuple = EMBED_LIBRARIES, version: Callable[[str], str] | None = None) -> dict:
    """Distribution versions read from the installed metadata (nothing is imported); None when not installed."""
    from importlib import metadata
    version = version or metadata.version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def loaded_libraries(modules: dict | None = None) -> list[str]:
    """Distributions (of `IMPORT_NAMES`) whose module this interpreter has imported."""
    modules = sys.modules if modules is None else modules
    return [d for d, m in IMPORT_NAMES.items() if m in modules]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def versions_key(versions: dict) -> str:
    return hashlib.sha256(json.dumps({n: versions.get(n) for n in EMBED_LIBRARIES}, sort_keys=True).encode()).hexdigest()[:12]


def norm(name: str) -> str:
    """PEP 503 normalised distribution name (wheel files use underscores, pip list hyphens)."""
    import re
    return re.sub(r"[-_.]+", "-", name).lower()


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


# --------------------------------------------------------------------------------------------------------- plans
def network_plan(recorded: dict | None, before: dict) -> list[list[str]]:
    """`pip` argument lists. Without a record: newest CPU torch, then sentence-transformers with torch (TORCH_HOLD) and
    every installed library of HELD at its current version. With one: exactly the recorded versions."""
    cmds = []
    present = [n for n in REMOVE if before.get(n)]
    if present:
        cmds.append(["uninstall", "-y", "-q", *present])
    if recorded:
        _complete(recorded)
        cmds.append(["install", "-q", "--index-url", CPU_TORCH_INDEX, f"torch=={recorded['torch']}"])
        cmds.append(["install", "-q", *[f"{n}=={recorded[n]}" for n in EMBED_LIBRARIES]])
    else:
        cmds.append(["install", "-q", "--index-url", CPU_TORCH_INDEX, "torch"])
        cmds.append(["install", "-q", "sentence-transformers", TORCH_HOLD,
                     *[f"{n}=={before[n]}" for n in HELD if before.get(n)]])
    return cmds


def offline_plan(recorded: dict, manifest: dict, wheel_dir: Path, before: dict) -> list[list[str]]:
    """Exactly the recorded versions from the local copy of the Drive wheel cache (D144); libraries not in the cache are
    Colab's own and must already be at the recorded version (else pip fails loudly: no index)."""
    _complete(recorded)
    cmds = []
    present = [n for n in REMOVE if before.get(n)]
    if present:
        cmds.append(["uninstall", "-y", "-q", *present])
    pins = {norm(n): recorded[n] for n in EMBED_LIBRARIES}
    for w in manifest["wheels"].values():
        if pins.setdefault(norm(w["name"]), w["version"]) != w["version"]:
            raise SetupError(f"the wheel cache has {w['name']} {w['version']}, the record {pins[norm(w['name'])]}")
    cmds.append(["install", "-q", "--no-index", "--find-links", str(wheel_dir), *[f"{n}=={v}" for n, v in pins.items()]])
    return cmds


def _complete(recorded: dict) -> None:
    missing = [n for n in EMBED_LIBRARIES if not recorded.get(n)]
    if missing:
        raise ValueError(f"recorded library versions are incomplete: {missing}")


def would_change(recorded: dict | None, before: dict) -> set[str]:
    """Distributions the install may replace (a loaded module among them must stop the setup: second opinion L1)."""
    out = {n for n in REMOVE if before.get(n)}
    if recorded:
        return out | {n for n in EMBED_LIBRARIES if before.get(n) != recorded.get(n)}
    return out | {"torch", "sentence-transformers"} | {n for n in EMBED_LIBRARIES if not before.get(n)}


# ------------------------------------------------------------------------------------------------- Drive caches
def read_manifest(folder: Path) -> dict | None:
    """None only when there is no manifest or it is not JSON; a Drive I/O error propagates (fourth review L2: it must
    not be taken for a missing cache, which would set a valid one aside unrecorded)."""
    try:
        return json.loads((Path(folder) / "manifest.json").read_text())
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None


def copy_verified(src_dir: Path, files: dict[str, str], dst_dir: Path) -> None:
    """Copy `files` ({relative path: sha256}) from src_dir to dst_dir, verifying every copy (a local copy that already
    verifies is kept)."""
    for rel, sha in files.items():
        dst = Path(dst_dir) / rel
        if dst.exists() and sha256_file(dst) == sha:
            continue
        src = Path(src_dir) / rel
        if not src.exists():
            raise SetupError(f"cache file missing: {src} (delete {src_dir} on Drive to rebuild it)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        got = sha256_file(tmp)
        if got != sha:
            tmp.unlink()
            raise SetupError(f"sha256 mismatch for {src} ({got} != {sha}): delete {src_dir} on Drive to rebuild it")
        os.replace(tmp, dst)


def publish_dir(local_dir: Path, files: dict[str, str], drive_dir: Path, manifest: dict) -> None:
    """Copy a verified folder to Drive under a temporary name, write the manifest last, then rename it into place (a
    half-copied cache never carries the real name)."""
    drive_dir = Path(drive_dir)
    if read_manifest(drive_dir) is not None:
        return
    for stale in drive_dir.parent.glob(drive_dir.name + ".tmp-*"):   # left by a killed runtime (runtimes are sequential)
        shutil.rmtree(stale, ignore_errors=True)
    tmp = drive_dir.with_name(drive_dir.name + f".tmp-{os.getpid()}-{int(time.time())}")
    copy_verified(local_dir, files, tmp)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    if drive_dir.exists():                               # an earlier partial folder without a manifest
        set_aside(drive_dir)
    os.rename(tmp, drive_dir)


def set_aside(folder: Path) -> Path:
    """Rename a broken cache folder out of the way (kept for inspection, never reused)."""
    dst = Path(folder).with_name(Path(folder).name + f".broken-{time.time_ns()}")
    os.rename(folder, dst)
    return dst


def cache_complete(folder: Path, manifest: dict) -> bool:
    """Every wheel of the manifest is present (Drive may hold the manifest of a folder whose large files never finished
    uploading) and the CPU torch is among them (sentence-transformers may be Colab's own, session 2)."""
    names = {norm(w["name"]) for w in manifest.get("wheels", {}).values()}
    return "torch" in names and all((Path(folder) / f).exists() for f in manifest["wheels"])


def ensure_minilm(drive_dir: Path, local_dir: Path, revision: str,
                  download: Callable[[Path, str], None]) -> dict:
    """The MiniLM snapshot at `revision`: from Drive (verified) or downloaded once (`download(local_dir, revision)`)
    and published to Drive. Returns the manifest {"revision", "files": {path: sha256}}."""
    man = read_manifest(drive_dir)
    if man is not None:
        if man.get("revision") != revision:
            raise SetupError(f"{drive_dir} holds revision {man.get('revision')}, not {revision}")
        copy_verified(drive_dir, man["files"], local_dir)
        return man
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    download(Path(local_dir), revision)
    files = {}
    for rel in MINILM_FILES:
        p = Path(local_dir) / rel
        if p.exists():
            files[rel] = sha256_file(p)
    for rel in ("modules.json", "config.json", "model.safetensors", "tokenizer.json"):
        if rel not in files:
            raise SetupError(f"MiniLM snapshot incomplete: {rel} missing in {local_dir}")
    man = {"revision": revision, "repo": MINILM_REPO, "files": files}
    publish_dir(local_dir, files, drive_dir, man)
    return _reread(drive_dir, man)


# --------------------------------------------------------------------------------------------------- the checks
def check_embed_install(probe: dict, got: dict, recorded: dict | None, environ: dict | None = None) -> None:
    """Raises unless torch is a CPU build, the embedder loads and embeds on one thread, the imported versions are the
    installed ones, every library is installed, the CPU code paths are pinned (D143) and, when there is a record,
    every version equals it."""
    environ = os.environ if environ is None else environ
    if probe.get("torch_cuda") is not None:
        raise SetupError(f"torch is not the CPU-only build (CUDA {probe['torch_cuda']})")
    for k in ("torch", "sentence-transformers", "transformers"):
        if probe.get(k) != got.get(k):
            raise SetupError(f"{k}: imported {probe.get(k)} but installed {got.get(k)}")
    missing = [n for n in EMBED_LIBRARIES if not got.get(n)]
    if missing:
        raise SetupError(f"not installed: {missing}")
    if probe.get("num_threads") != 1 or not probe.get("vectors_sha256") or probe.get("dim") != 384:
        raise SetupError(f"the embedder probe is wrong: threads {probe.get('num_threads')}, dim {probe.get('dim')}")
    if probe.get("env") != CPU_PATHS_ENV or {k: environ.get(k) for k in CPU_PATHS_ENV} != CPU_PATHS_ENV:
        raise SetupError(f"the CPU code paths are not pinned (D143): {probe.get('env')}")
    if str(probe.get("cpu_capability", "")).upper() != "AVX2":
        raise SetupError(f"ATen does not use the AVX2 path (D143): {probe.get('cpu_capability')}")
    if recorded:
        diff = {n: (recorded.get(n), got.get(n)) for n in EMBED_LIBRARIES if recorded.get(n) != got.get(n)}
        if diff:
            raise SetupError(f"library versions differ from the recorded ones (recorded, installed): {diff}")


def libtorch_cuda() -> Path | None:
    try:
        from importlib import metadata
        return Path(metadata.distribution("torch").locate_file("torch/lib/libtorch_cuda.so"))
    except Exception:          # noqa: BLE001
        return None


def colab_torch_failure(sh: Sh, python: str = sys.executable,
                        locate: Callable[[], Path | None] = libtorch_cuda) -> dict:
    """D145: why Colab's preinstalled torch fails, recorded before it is uninstalled (for the thesis)."""
    rc, _, err = sh([python, "-c", "import torch"])
    res: dict[str, Any] = {"import_returncode": rc, "import_error": err.strip().splitlines()[-3:] if err else [],
                           "nccl": installed_versions(("nvidia-nccl-cu12", "nvidia-nccl-cu13"))}
    lib = locate()
    if lib is not None and lib.exists():
        _, ldd, _ = sh(["ldd", str(lib)])
        res["ldd_nccl"] = [ln.strip() for ln in ldd.splitlines() if "nccl" in ln]
    else:
        res["ldd_nccl"] = None
    return res


# ------------------------------------------------------------------------------------------------- the setup step
def preflight_minilm(drive_dir: Path, local_dir: Path, revision: str, evidence: dict) -> dict | None:
    """Before pip runs (second-opinion M1): copy and verify the Drive snapshot; a broken one (missing or damaged files,
    a wrong revision) is set aside and recorded, and None returned so it is downloaded again after the install."""
    man = read_manifest(drive_dir)
    if man is None:
        return None
    try:
        if man.get("revision") != revision:
            raise SetupError(f"{drive_dir} holds revision {man.get('revision')}, not {revision}")
        copy_verified(drive_dir, man["files"], local_dir)
        return man
    except SetupError as e:
        evidence["broken_minilm"] = {"folder": str(set_aside(drive_dir)), "error": str(e)[-500:]}
        return None


def preflight_wheels(wheel_dir: Path, rec: dict, before: dict, local_wheels: Path, evidence: dict,
                     failed_offline: list[dict] = ()) -> dict | None:
    """Before pip runs (second-opinion M1): the offline install is chosen only if the cache is complete, every wheel
    copies with its sha256, and every library not in the cache is already at the recorded version in Colab's image.
    A broken cache is set aside (rebuilt later); a changed image keeps the (valid) cache; both are recorded and the
    caller falls back to installing the recorded versions from the internet."""
    man = read_manifest(wheel_dir)
    if man is None:
        return None
    # third review L2: an offline install from THIS copy of the cache (failure newer than its manifest) failed while pip
    # was running -> set it aside; failures after pip finished (Hub, probe, ...) and older copies do not count
    created = man.get("created", 0)
    mine = [f for f in failed_offline if f.get("cache") == Path(wheel_dir).name]
    if len(mine) >= 2:                 # fourth review L1: a rebuilt copy failed too -> not the cache; stop alternating
        evidence["offline_disabled"] = f"{len(mine)} offline installs from this cache failed while pip was running"
        return None
    if any(f.get("t", 0) > created for f in mine):
        evidence["broken_cache"] = {"folder": str(set_aside(wheel_dir)),
                                    "error": "an earlier offline install from this cache failed while pip was running"}
        return None
    problem = None if cache_complete(wheel_dir, man) else "incomplete: a wheel or the CPU torch is missing"
    if problem is None:
        try:
            copy_verified(wheel_dir, {f: w["sha256"] for f, w in man["wheels"].items()}, local_wheels)
        except SetupError as e:
            problem = str(e)[-500:]
    if problem is not None:
        evidence["broken_cache"] = {"folder": str(set_aside(wheel_dir)), "error": problem}
        return None
    cached = {norm(w["name"]) for w in man["wheels"].values()}
    changed = {n: [before.get(n), rec.get(n)] for n in EMBED_LIBRARIES
               if norm(n) not in cached and before.get(n) != rec.get(n)}
    if changed:
        evidence["image_changed"] = changed
        return None
    return man


def ensure_embed(sections, attempts, pins: dict, drive_root: Path, local_root: Path, sh: Sh,
                 download_minilm: Callable[[Path, str], None], loaded: list[str] | tuple = (),
                 versions: Callable[..., dict] = installed_versions,
                 freeze: Callable[[], dict | None] = lambda: None, runtime: str | None = None,
                 python: str = sys.executable, environ: dict | None = None,
                 locate_libtorch: Callable[[], Path | None] = libtorch_cuda,
                 log: Callable[[str], Any] = lambda m: None) -> dict:
    """See the module docstring. `sections`: the session file (t4.Sections); `attempts`: a stamp-less Sections for the
    attempt log (events start / pip_started / pip_done / ok / failed); `sh(argv) -> (rc, stdout, stderr)`; `loaded`:
    `loaded_libraries()`; `freeze() -> {name: version}` from `pip list`; `log`: one line per step. Returns the entry
    (with `minilm_path` and `probe`)."""
    environ = os.environ if environ is None else environ
    env_now = {k: environ.get(k) for k in CPU_PATHS_ENV}
    if "torch" in loaded and env_now != CPU_PATHS_ENV:
        raise SetupError("torch was imported before the CPU code paths were set (D143). " + HINT)
    environ.update(CPU_PATHS_ENV)
    rev = pins["minilm_revision"]
    rec = pins.get("libraries") or sections.data.get("libraries", {}).get("versions")
    before = versions(EMBED_LIBRARIES + REMOVE)
    now = {n: before.get(n) for n in EMBED_LIBRARIES}
    mine = [a for a in attempts.data.get("attempts", []) if a.get("runtime") == runtime]
    last = {}
    for i, a in enumerate(mine):
        last[a.get("event")] = i
    if last.get("pip_started", -1) > last.get("pip_done", -1):   # pip ran here and did not finish (review M1)
        raise SetupError("this runtime already attempted the install and pip did not finish, so its environment is "
                         "not Colab's original. " + HINT)
    resumed = None
    if last.get("pip_done", -1) > last.get("ok", -1):            # D147: pip finished, a later step failed
        done = mine[last["pip_done"]]
        if done.get("versions") != now:
            raise SetupError("the installed versions changed since pip finished in this runtime "
                             f"({done.get('versions')} -> {now}). " + HINT)
        resumed = done
    in_place = resumed is None and bool(rec) and now == {n: rec.get(n) for n in EMBED_LIBRARIES}
    change = set() if (in_place or resumed) else would_change(rec, before)
    clash = sorted(change & set(loaded))
    if clash:
        raise SetupError(f"{clash} already imported and would be replaced. " + HINT)
    evidence: dict[str, Any] = {}
    local_wheels = Path(local_root) / "embed_wheels"
    manifest = None
    if in_place:
        mode = "in_place"
    elif resumed:
        mode = "resumed"
        before = resumed.get("before", before)                  # Colab's original state and evidence (review M2)
        evidence.update(resumed.get("evidence", {}))
    else:
        if rec:
            log("embedder setup: checking the wheel cache on Drive ...")
            failed_offline = [a for a in attempts.data.get("attempts", [])
                              if a.get("event") == "failed" and a.get("mode") == "offline" and a.get("pip_started")
                              and not a.get("pip_finished")]
            wheel_dir = Path(drive_root) / "wheels" / "embed" / f"{python_tag()}-{versions_key(rec)}"
            try:
                manifest = preflight_wheels(wheel_dir, rec, before, local_wheels, evidence, failed_offline)
            except OSError as e:       # third review L3: a Drive I/O error gets a record and the hint
                attempts.append("attempts", {"event": "failed", "runtime": runtime, "t": time.time(), "ok": False,
                                             "error": f"pre-flight: {type(e).__name__}: {e}"[-3000:]})
                raise SetupError(f"pre-flight on Drive failed: {e}\n{HINT}") from e
        if manifest:                   # review L3: a dependency Colab's image no longer satisfies -> fall back
            rc, out, err = sh([python, "-m", "pip", "install", "--dry-run", "-q", "--no-index", "--find-links",
                               str(local_wheels), *offline_plan(rec, manifest, local_wheels, before)[-1][5:]])
            if rc:
                evidence["offline_dry_run"] = (out + err).strip().splitlines()[-5:]
                manifest = None
        mode = "offline" if manifest else "network_exact" if rec else "resolve"
    mdir = Path(local_root) / "minilm" / rev
    mdrive = Path(drive_root) / "models" / "minilm" / rev
    try:
        minilm = preflight_minilm(mdrive, mdir, rev, evidence)
    except OSError as e:               # third review L3
        attempts.append("attempts", {"event": "failed", "runtime": runtime, "t": time.time(), "ok": False,
                                     "error": f"pre-flight: {type(e).__name__}: {e}"[-3000:]})
        hint = HINT_RERUN if mode in ("resumed", "in_place") else HINT    # pip already done here: a re-run suffices
        raise SetupError(f"pre-flight on Drive failed: {e}\n{hint}") from e
    if mode not in ("in_place", "resumed") and not sections.done("libraries") and before.get("torch") \
            and "+cpu" not in before["torch"]:
        evidence["colab_torch"] = colab_torch_failure(sh, python, locate_libtorch)
    log(f"embedder setup: mode {mode}" + (f" ({', '.join(evidence)})" if evidence else ""))
    attempts.append("attempts", {"event": "start", "runtime": runtime, "t": time.time(), "mode": mode,
                                 "before": before, "loaded": list(loaded), **evidence})
    durations: dict[str, float] = {}
    cmds_run: list[list[str]] = []
    pip_finished = resumed is not None or mode == "in_place"
    started_pip = False
    cache_name = (Path(drive_root) / "wheels" / "embed" / f"{python_tag()}-{versions_key(rec)}").name if rec else None
    try:
        def pip(args: list[str], key: str) -> None:
            log(f"embedder setup: pip {' '.join(a for a in args if not a.startswith('-'))[:160]}")
            t0 = time.time()
            rc, out, err = sh([python, "-m", "pip", *args])
            durations[key] = round(durations.get(key, 0.0) + time.time() - t0, 1)
            if rc:
                raise SetupError(f"pip {' '.join(args[:3])} ... failed ({rc}):\n{(out + err)[-2000:]}")

        network = mode in ("resolve", "network_exact") or (resumed is not None and
                                                           resumed.get("mode") in ("resolve", "network_exact"))
        if resumed is not None:
            freeze_before = resumed.get("freeze_before")
        else:
            freeze_before = freeze() if network else None
        if network and freeze_before is None:
            raise SetupError("pip list failed: the installed wheels could not be cached")
        if mode == "offline":
            plan = offline_plan(rec, manifest, local_wheels, before)
        elif mode in ("resolve", "network_exact"):
            plan = network_plan(rec, before)
        else:
            plan = []
        if plan:
            attempts.append("attempts", {"event": "pip_started", "runtime": runtime, "t": time.time(), "mode": mode})
            started_pip = True
        for args in plan:
            if TORCH_HOLD in args:
                args = [f"torch=={versions(('torch',))['torch']}" if a == TORCH_HOLD else a for a in args]
            pip(args, args[0])
            cmds_run.append(args)
        rc, chk, _ = sh([python, "-m", "pip", "check"]) if cmds_run else (None, "", "")
        pip_check = {"returncode": rc, "output": chk.strip().splitlines()[-30:]}
        if resumed is not None:
            pip_check = resumed.get("pip_check", pip_check)
        if plan:
            attempts.append("attempts", {"event": "pip_done", "runtime": runtime, "t": time.time(), "mode": mode,
                                         "versions": versions(EMBED_LIBRARIES), "pip": cmds_run,
                                         "freeze_before": freeze_before, "before": before, "evidence": evidence,
                                         "pip_check": pip_check})
            pip_finished = True
        t0 = time.time()
        if minilm is None:
            log("embedder setup: downloading the MiniLM snapshot to Drive ...")
            minilm = ensure_minilm(mdrive, mdir, rev, download_minilm)
        durations["minilm"] = round(time.time() - t0, 1)
        log("embedder setup: checking the embedder in a fresh interpreter ...")
        t0 = time.time()
        rc, out, err = sh([python, "-m", "eem_acc.stage3.llm.embed", "--probe", "--path", str(mdir), "--revision", rev])
        durations["probe"] = round(time.time() - t0, 1)
        lines = (out or "").strip().splitlines()
        if rc or not lines:
            raise SetupError(f"the embedder probe failed ({rc}):\n{((out or '') + (err or '') or '(no output)')[-3000:]}")
        probe = json.loads(lines[-1])
        got = versions(EMBED_LIBRARIES)
        check_embed_install(probe, got, rec, environ)
        freeze_after = freeze() if (cmds_run or resumed) else None
        wheels = manifest
        if network:
            log("embedder setup: caching the wheels on Drive ...")
            t0 = time.time()
            try:                       # review M2: the libraries passed every check; a cache failure is recorded
                wheels = cache_wheels(sh, python, freeze_before, freeze_after, got,
                                      Path(local_root) / "embed_wheels_new",
                                      Path(drive_root) / "wheels" / "embed" / f"{python_tag()}-{versions_key(got)}")
            except (SetupError, OSError) as ce:   # (a later runtime installs from the internet and tries again)
                wheels = {"error": str(ce)[-2000:]}
            durations["wheel_cache"] = round(time.time() - t0, 1)
    except Exception as e:     # noqa: BLE001
        attempts.append("attempts", {"event": "failed", "runtime": runtime, "t": time.time(), "ok": False,
                                     "error": str(e)[-3000:], "durations": durations, "pip": cmds_run, "mode": mode,
                                     "pip_started": started_pip, "pip_finished": pip_finished, "cache": cache_name})
        raise SetupError(f"{e}\n{HINT_RERUN if pip_finished else HINT}") from e
    pip_cmds = cmds_run or (resumed or {}).get("pip", [])
    entry = {"versions": got, "mode": mode, "pip": pip_cmds, "durations": durations, "before": before,
             "pip_check": pip_check, "probe": probe, "wheels": wheels, "minilm": minilm, "runtime": runtime,
             "from": "configs/t4_pins.json" if pins.get("libraries") else ("session file" if rec else "resolved"),
             **evidence}
    if not sections.done("libraries"):
        sections.save("libraries", {**entry, "freeze": freeze_after})
    if cmds_run or resumed:
        sections.append("libraries_runtimes", {"runtime": runtime, "mode": mode, "pip": pip_cmds,
                                               "durations": durations, "before": before, "freeze": freeze_after,
                                               **evidence})
    probes = sections.data.get("embed_probes", [])
    if not any(p.get("runtime") == runtime for p in probes):
        first = probes[0]["vectors_sha256"] if probes else probe["vectors_sha256"]
        sections.append("embed_probes", {"runtime": runtime, "cpu": probe.get("cpu"),
                                         "cpu_capability": probe.get("cpu_capability"),
                                         "vectors_sha256": probe["vectors_sha256"],
                                         "seconds_per_embed": probe.get("seconds_per_embed"),
                                         "equal_first": probe["vectors_sha256"] == first})
    attempts.append("attempts", {"event": "ok", "runtime": runtime, "t": time.time(), "ok": True, "mode": mode,
                                 "durations": durations})
    log(f"embedder setup: done ({mode}; {sum(durations.values()):.0f} s)")
    return {**entry, "minilm_path": str(mdir)}


def cache_wheels(sh: Sh, python: str, freeze_before: dict, freeze_after: dict | None, got: dict, local_dir: Path,
                 drive_dir: Path) -> dict:
    """D144: download (no dependencies) the wheel of every distribution the install added or changed (CPU torch from
    the CPU index, the rest from PyPI), verify, and publish them to Drive with a sha256 manifest."""
    existing = read_manifest(drive_dir)
    if existing is not None:
        return existing
    if freeze_after is None:
        raise SetupError("pip list failed after the install: the wheels could not be cached")
    fb = {norm(n): v for n, v in freeze_before.items()}
    changed = {n: v for n, v in freeze_after.items() if fb.get(norm(n)) != v}
    by = {norm(n): n for n in changed}
    need = {"torch"} | {norm(n) for n in EMBED_LIBRARIES if fb.get(norm(n)) != got.get(n)}   # torch: always replaced
    if not need <= set(by):
        raise SetupError(f"the install did not change {sorted(need - set(by))}: the environment was not Colab's original")
    shutil.rmtree(local_dir, ignore_errors=True)
    Path(local_dir).mkdir(parents=True)
    for name, ver in sorted(changed.items()):
        idx = ["--index-url", CPU_TORCH_INDEX] if "+cpu" in ver else []
        rc, out, err = sh([python, "-m", "pip", "download", "-q", "--no-deps", "--only-binary", ":all:", *idx,
                           "-d", str(local_dir), f"{name}=={ver}"])
        if rc:
            raise SetupError(f"pip download {name}=={ver} failed ({rc}):\n{(out + err)[-1500:]}")
    wheels = {}
    for f in sorted(Path(local_dir).glob("*.whl")):
        dist, ver = f.name.split("-")[:2]
        wheels[f.name] = {"name": dist, "version": ver.replace("%2B", "+"), "sha256": sha256_file(f)}
    have = {(norm(w["name"]), w["version"]) for w in wheels.values()}
    lost = [f"{n}=={v}" for n, v in changed.items() if (norm(n), v) not in have]   # e.g. an sdist-only package
    if lost:
        raise SetupError(f"no wheel downloaded for {lost}")
    man = {"python": python_tag(), "versions": {n: got[n] for n in EMBED_LIBRARIES}, "wheels": wheels,
           "created": time.time()}
    publish_dir(local_dir, {f: w["sha256"] for f, w in wheels.items()}, drive_dir, man)
    return _reread(drive_dir, man)


def _reread(drive_dir: Path, man: dict) -> dict:
    """The manifest as published (another runtime's, if it won); a read error right after a successful publish is not
    a failure (review of the fourth-round fixes, Low 1)."""
    try:
        return read_manifest(drive_dir) or man
    except OSError:
        return man
