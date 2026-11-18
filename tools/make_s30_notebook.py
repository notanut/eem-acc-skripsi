"""Generates notebooks/03_s30_feasibility.ipynb. Usage: python tools/make_s30_notebook.py notebooks/03_s30_feasibility.ipynb"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# 03 - S3.0 feasibility check of the local LLM on a T4 (Stage 3)
Measures what the Stage 3 design (`docs/stage3_design.md`, section 10) needs before the scale is fixed:
LLM speed (the timing model `t = a + b_in*n_in + b_out*n_out`) with and without the JSON grammar, GPU memory,
JSON validity, whether repeated greedy outputs are identical, and a toy 3-step task with one changed API field.
**Nothing here is a thesis result.**

**Runtime:** Runtime -> Change runtime type -> **T4 GPU**. Run the cells top to bottom.
- First time: building llama-cpp-python with CUDA takes about an hour (it is then cached on Drive, so later
  sessions install in seconds); downloading the model (~4.7 GB) takes a few minutes and is also cached on Drive.
- The measurements themselves take about 10-15 min.
- Every section saves into `Skripsi/stage3/s30/s30_results.json` on Drive; if Colab disconnects, run the cells again
  and finished sections are skipped.

**At the end:** download `s30_results.json` (path printed in the last cell) and upload it to the chat.
""")

code(r'''
# 1) Setup: GPU check, Drive, repo   (run once per session)
import os, sys, subprocess, tempfile, json, time, glob, shutil, hashlib
from pathlib import Path

REPO_URL = "https://github.com/notanut/eem-acc-skripsi.git"
IN_COLAB = "google.colab" in sys.modules
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True)
print("GPU:", gpu.stdout.strip() or "NONE - switch the runtime to T4 GPU")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_DIR = "/content/eem-acc-skripsi"
    DRIVE_ROOT = "/content/drive/MyDrive/Skripsi"
    def git(*args, cwd=None):
        subprocess.run(["git", *args], cwd=cwd, check=True)
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        head = lambda: subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True).stdout
        before = head()
        git("pull", "--ff-only", cwd=REPO_DIR)
        if head() != before and "eem_acc" in sys.modules:
            print("NEW CODE WAS PULLED, but the old version is still loaded.\n"
                  ">>> Runtime -> Restart session, then run this cell again. <<<")
    else:
        git("clone", REPO_URL, REPO_DIR)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt", "-e", REPO_DIR], check=True)
else:                                   # running outside Colab (local test): use this checkout + a temp folder
    REPO_DIR = str(Path.cwd().parent) if Path.cwd().name == "notebooks" else str(Path.cwd())
    DRIVE_ROOT = tempfile.mkdtemp(prefix="eem_acc_")
import importlib
if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")
importlib.invalidate_caches()
from eem_acc.stage3 import feasibility as F
from eem_acc.utils.atomic import atomic_write_json

OUT_DIR = Path(DRIVE_ROOT) / "stage3" / "s30"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_DIR / "s30_results.json"
FORCE = False          # True = redo every measurement section even if a matching one is saved
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
REPO_COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
# Saved sections are reused only if measured with the same measurement code, build and model. The code is identified
# by the hash of feasibility.py (not the repo commit, so unrelated commits do not invalidate finished sections).
FEAS_SHA256 = hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest()
STAMP = {"feasibility_sha256": FEAS_SHA256}  # completed in cell 4 with the llama.cpp build and the model sha256
def save(section, value):
    results[section] = {**value, "stamp": dict(STAMP)}
    atomic_write_json(RESULTS, results)
    print(f"saved section '{section}' -> {RESULTS}")
# Sections saved by the first notebook version carry the repo commit instead; map it to that commit's feasibility.py hash.
LEGACY_COMMITS = {"e4594b9882bf295ee9383ceb33b1435ae73133d3": "cb2e4dcf7f385c615e574bf31da05c3d818616eb5a31a0cee6dae4119c86fd65"}
def done(section):
    """A saved section is reused only if it was measured with the same code, build and model (else it is redone)."""
    old = results.get(section)
    if FORCE or old is None:
        return False
    stamp = dict(old.get("stamp", {}))
    if stamp.get("repo_commit") in LEGACY_COMMITS:
        stamp["feasibility_sha256"] = LEGACY_COMMITS[stamp.pop("repo_commit")]
    if stamp != STAMP:
        print(f"section '{section}' was measured with a different code/build/model -> redoing it")
        return False
    return True
save("env", {"gpu": gpu.stdout.strip(), "python": sys.version.split()[0], "repo_commit": REPO_COMMIT})
print("saved sections:", [k for k in results if k != "env"])
''')

code(r'''
# 2) Install llama-cpp-python with CUDA (pinned; the built wheel is cached on Drive)
import zipfile, zlib
LLAMA_VERSION = "0.3.36"           # released 2026-10-01 (PyPI); pinned for reproducibility
PYTAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
WHEEL_DIR = Path(DRIVE_ROOT) / "wheels"
WHEEL_DIR.mkdir(parents=True, exist_ok=True)
BUILD_DIR = Path(globals().get("BUILD_DIR_OVERRIDE", "/content/wheels"))   # override only in tests
def compatible(name):
    """llama-cpp-python loads its library through ctypes, so a source build is tagged py3-none-linux_x86_64;
    prebuilt wheels may carry the CPython tag instead. Accept both, never another Python version."""
    parts = name[:-4].split("-")          # dist, version, python tag, abi tag, platform
    return len(parts) == 5 and parts[2] in (PYTAG, "py3") and parts[4].startswith("linux_x86_64")
def good_wheels(folder=None):
    out = []
    for w in sorted(Path(folder or WHEEL_DIR).glob(f"llama_cpp_python-{LLAMA_VERSION}-*.whl")):
        if not compatible(w.name):
            continue
        try:
            with zipfile.ZipFile(w) as z:
                bad = z.testzip()
            if bad is None:
                out.append(w)
            else:
                print(f"ignoring a damaged wheel {w.name} (bad member {bad})")
        except (zipfile.BadZipFile, OSError, EOFError, zlib.error) as e:
            print(f"ignoring a broken wheel {w.name}: {e}")
    return out
def installed_version():
    r = subprocess.run([sys.executable, "-c", "import llama_cpp; print(llama_cpp.__version__)"], capture_output=True, text=True)
    return (r.stdout or "").strip() if r.returncode == 0 else None
def import_error():
    r = subprocess.run([sys.executable, "-c", "import llama_cpp"], capture_output=True, text=True)
    return (r.stderr or "").strip().splitlines()[-1:] if r.returncode != 0 else []
def save_to_drive(wheels):
    for w in wheels:
        tmp = WHEEL_DIR / (w.name + ".tmp")
        shutil.copy(w, tmp)
        os.replace(tmp, WHEEL_DIR / w.name)                 # never leave a half-copied wheel under the real name
    saved = good_wheels()
    if not saved:
        raise RuntimeError(f"copying the wheel to Drive failed; it is still in {BUILD_DIR}")
    return saved
already = installed_version() == LLAMA_VERSION
cached = good_wheels()
if IN_COLAB and not cached and good_wheels(BUILD_DIR):
    cached = save_to_drive(good_wheels(BUILD_DIR))          # an earlier build in this session that was not saved
    print(f"saved an earlier build from {BUILD_DIR} to Drive")
if IN_COLAB and not cached and not already:
    t0 = time.time()
    print("Building llama-cpp-python with CUDA for the T4 (sm_75). This takes about an hour the first time ...")
    env = dict(os.environ, CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75", CMAKE_BUILD_PARALLEL_LEVEL="4")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", f"llama-cpp-python=={LLAMA_VERSION}",
                    "-w", str(BUILD_DIR)], env=env, check=True)
    built = good_wheels(BUILD_DIR)
    if not built:
        raise RuntimeError(f"build finished but no usable wheel in {BUILD_DIR}: {os.listdir(BUILD_DIR)}")
    cached = save_to_drive(built)
    print(f"built in {(time.time()-t0)/60:.1f} min")
print("cached on Drive:", [w.name for w in cached])
if cached and not already:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(cached[-1])], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "typing-extensions", "diskcache", "jinja2"], check=True)
if already and not cached:
    print("WARNING: llama-cpp-python is installed but no wheel is cached on Drive; wheel sha256 unknown")
# sha256 of the cached wheel (if the library was installed from elsewhere earlier in this session, it may differ)
WHEEL_SHA256 = hashlib.sha256(Path(cached[-1]).read_bytes()).hexdigest() if cached else None
print("llama-cpp-python:", installed_version(), "| wheel sha256:", WHEEL_SHA256)
if IN_COLAB and installed_version() != LLAMA_VERSION:
    hint = f" (delete {cached[-1]} on Drive to force a rebuild)" if cached else ""
    raise RuntimeError(f"llama-cpp-python {LLAMA_VERSION} is not importable: {import_error()}{hint}")
''')

code(r'''
# 3) Download the model to Drive (cached) and record its sha256
from huggingface_hub import list_repo_files, hf_hub_download
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
QUANT = "q4_k_m"
MODEL_DIR = Path(DRIVE_ROOT) / "models"
def fetch_model(repo, quant):
    files = sorted(f for f in list_repo_files(repo) if f.lower().endswith(".gguf") and quant in f.lower())
    assert files, f"no {quant} file in {repo}"
    paths, digests = [], {}
    for f in files:                                # a split model has several parts; llama.cpp loads the first
        p = Path(hf_hub_download(repo, f, local_dir=MODEL_DIR / repo.replace("/", "__")))
        side = p.with_suffix(p.suffix + ".sha256.json")
        meta = json.loads(side.read_text()) if side.exists() else {}
        if meta.get("size") != p.stat().st_size:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            meta = {"size": p.stat().st_size, "sha256": h.hexdigest()}
            side.write_text(json.dumps(meta))
        paths.append(p); digests[f] = meta
    return paths, digests
t0 = time.time()
MODEL_PATHS, MODEL_DIGESTS = fetch_model(MODEL_REPO, QUANT)
print(f"model ready in {time.time()-t0:.0f} s:", [p.name for p in MODEL_PATHS])
save("model", {"repo": MODEL_REPO, "files": MODEL_DIGESTS})
''')

code(r'''
# 4) Load the model on the GPU and record memory
import gc
from llama_cpp import Llama
import llama_cpp
N_CTX, N_BATCH = 8192, 512
def load(path):
    g = globals()
    for name in ("llm", "llm3"):                     # free a model loaded by an earlier run of this cell
        if g.get(name) is not None:
            try:
                g[name].close()
            except Exception:
                pass
            g[name] = None
    gc.collect()
    t0 = time.time()
    m = Llama(model_path=str(path), n_gpu_layers=-1, n_ctx=N_CTX, n_batch=N_BATCH, seed=0, verbose=False)
    return m, time.time() - t0
llm, load_s = load(MODEL_PATHS[0])
vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
template = (llm.metadata or {}).get("tokenizer.chat_template", "")
STAMP.update({"llama_cpp_python": llama_cpp.__version__, "wheel_sha256": WHEEL_SHA256,
              "model_sha256": {k: v["sha256"] for k, v in MODEL_DIGESTS.items()}, "n_ctx": N_CTX, "n_batch": N_BATCH,
              "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()})
print(f"loaded in {load_s:.0f} s; GPU memory used/total: {vram}")
save("load", {"load_seconds": load_s, "vram_used_total_after_load": vram,
              "chat_format": getattr(llm, "chat_format", None),
              "system_info": llama_cpp.llama_print_system_info().decode(errors="replace")})
''')

code(r'''
# 5) Speed: timing model with and without the JSON grammar (fresh KV cache per call = worst case).
#    Free and grammar calls are interleaved (GPU throttling then affects both alike). Decode probes at several
#    max_tokens identify the per-output-token cost: free-text probes for the free fit, grammar probes for the grammar fit.
PROMPTS = F.sample_prompts(20, seed=0)
if not done("timing"):
    gens = {"free": F.make_generate(llm, grammar=False, reset=True), "grammar": F.make_generate(llm, grammar=True, reset=True)}
    gens["free"](PROMPTS[0]); gens["grammar"](PROMPTS[0])              # warm-up, not recorded
    out = {"free": [], "grammar": []}
    for i, msgs in enumerate(PROMPTS):
        for k in (("free", "grammar") if i % 2 == 0 else ("grammar", "free")):
            out[k] += F.bench_prompts(gens[k], [msgs])
    probes = {"free": [], "grammar": []}
    for n in (16, 48, 160):
        for msgs in PROMPTS[::4]:                                         # 5 prompts of different lengths
            for k, grammar in (("free", False), ("grammar", True)):
                g = F.make_generate(llm, grammar=grammar, reset=True, max_tokens=n)
                r = g(msgs if grammar else msgs + [{"role": "user", "content": "Now explain your reasoning in a long, detailed essay."}])
                probes[k].append({"seconds": r["seconds"], "n_in": r["n_in"], "n_out": r["n_out"], "max_tokens": n,
                                  "finish_reason": r.get("finish_reason")})
    out["probe_free"], out["probe_grammar"] = probes["free"], probes["grammar"]
    out["system_prefix_reuse"] = F.bench_prompts(F.make_generate(llm, grammar=True, reset=False), PROMPTS)
    fits = {"free": F.fit_timing(out["free"] + probes["free"]),
            "grammar": F.fit_timing(out["grammar"] + probes["grammar"])}
    def mean(recs, k): return sum(r[k] for r in recs) / len(recs)
    summary = {k: {"mean_s": mean(out[k], "seconds"), "mean_n_in": mean(out[k], "n_in"), "mean_n_out": mean(out[k], "n_out"),
                   "valid_envelope_rate": mean(out[k], "valid_envelope")} for k in ("free", "grammar", "system_prefix_reuse")}
    vram_peak = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()
    save("timing", {"records": out, "fits": fits, "summary": summary, "vram_used_after_timing": vram_peak})
for k, v in results["timing"]["summary"].items():
    print(f"{k:20s} mean {v['mean_s']:.2f} s/call  n_in {v['mean_n_in']:.0f}  n_out {v['mean_n_out']:.0f}  valid JSON {v['valid_envelope_rate']:.2f}")
for k, f in results["timing"]["fits"].items():
    se = f["se"] or {}
    print(f"fit {k:8s} a={f['a']:.3f}s (se {se.get('a', float('nan')):.3f})  b_in={1000*f['b_in']:.3f} ms/tok  "
          f"b_out={1000*f['b_out']:.1f} ms/tok (se {1000*se.get('b_out', float('nan')):.1f})  R2={f['r2']:.3f}  "
          f"n_in {f['n_in_range']}  n_out {f['n_out_range']}")
''')

code(r'''
# 6) Toy task: 3 operations, `create_order` changed `currency` -> `currency_code` (docs are outdated).
#    Half the instances get an informative error, half a terse "400 Bad Request". MaxSteps = 3 calls + finish + 6 = 10.
#    NOT a result.
if not done("toy"):
    gen = F.make_generate(llm, grammar=True, reset=False)           # KV-prefix reuse, as in a real agent run
    insts = F.make_instances(6, seed=0)
    eps = [F.run_toy_episode(gen, inst) for inst in insts]
    save("toy", {"summary": F.summarise_episodes(eps), "max_steps": F.TOY_MAX_STEPS,
                 "episodes": [{"error_style": i.error_style, "success": e.success, "steps": e.steps,
                               "docs_calls": e.docs_calls, "format_errors": e.format_errors,
                               "transcript": e.transcript} for i, e in zip(insts, eps)]})
print(results["toy"]["summary"])
for e in results["toy"]["episodes"]:
    print(f"  {e['error_style']:11s} success={e['success']!s:5s} steps={e['steps']} docs={e['docs_calls']} format_errors={e['format_errors']}")
''')

code(r'''
# 7) Determinism. (a) Same prompt twice with a fresh KV cache. (b) The informative one for resume-by-replay: regenerate
#    every step of the toy episodes (recorded WITH KV-prefix reuse) from the same messages with a FRESH cache.
if not done("determinism"):
    det = F.determinism_check(F.make_generate(llm, grammar=True, reset=True), PROMPTS[:8], repeats=2)
    fresh = F.make_generate(llm, grammar=True, reset=True)
    insts = F.make_instances(6, seed=0)
    det["replay"] = [F.replay_compare(fresh, inst, ep["transcript"]) for inst, ep in zip(insts, results["toy"]["episodes"])]
    save("determinism", det)
print(results["determinism"])
''')

code(r'''
# 8) OPTIONAL: the same speed and toy checks for Qwen2.5-3B-Instruct (only useful if the 7B is slower than ~6 s per call)
RUN_3B = False
if RUN_3B and not done("model_3b"):
    paths3, dig3 = fetch_model("Qwen/Qwen2.5-3B-Instruct-GGUF", QUANT)
    llm3, s3 = load(paths3[0])                      # load() closes the 7B first; re-run cell 4 to get it back
    gen3 = F.make_generate(llm3, grammar=True, reset=True)
    gen3(PROMPTS[0])
    recs3 = F.bench_prompts(gen3, PROMPTS)
    eps3 = [F.run_toy_episode(F.make_generate(llm3, grammar=True, reset=False), i) for i in F.make_instances(6, seed=0)]
    save("model_3b", {"files": dig3, "load_seconds": s3, "fit_grammar": F.fit_timing(recs3),
                      "valid_envelope_rate": sum(r["valid_envelope"] for r in recs3) / len(recs3),
                      "toy_summary": F.summarise_episodes(eps3)})
    print(results["model_3b"]["fit_grammar"], results["model_3b"]["toy_summary"])
''')

code(r'''
# 9) Done: upload this file to the chat
print("Results file:", RESULTS)
print("sections:", list(results))
if IN_COLAB:
    from google.colab import files
    files.download(str(RESULTS))
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(sys.argv[1], "w"), indent=1)
