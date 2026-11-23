"""Builds notebooks/04_stage3.ipynb (S3.3b, docs/s33b_spec.md r4). The S3.0 wheel cell is reused verbatim from
notebooks/03_s30_feasibility.ipynb so the two notebooks install the same build the same way.

    python tools/make_nb04.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S30 = json.loads((ROOT / "notebooks" / "03_s30_feasibility.ipynb").read_text())


def s30_cell(marker: str) -> str:
    return next("".join(c["source"]) for c in S30["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"]))


MD_INTRO = """# 04 - S3.3b: T4 verification, epoch 0, smoke run, throughput benchmark (Stage 3)
Implements `docs/s33b_spec.md`. **Nothing here is a thesis result.** Runtime: **T4 GPU**. Run the cells top to bottom.

**Session 1 (about 20-25 min):** set `SESSION = 1` below. Verifies the platform (chat template, equality with S3.0,
fresh KV cache, Colab tests). At the end a results file downloads: **upload it to the chat** and wait until I say the
session-2 code is pushed.

**Session 2 (about 2 h, three runtimes):** set `SESSION = 2`. The setup cell first installs the embedder's libraries
(CPU-only PyTorch + sentence-transformers; about 5 min in the first runtime, less later: they are cached on Drive).
Then it registers platform epoch 0, runs the smoke units, and
**kills the runtime twice on purpose** to test resume and Drive. When the runtime disappears:
1. Wait about 10 seconds, then **Runtime -> Disconnect and delete runtime** (if it is not already gone), then
   **Connect** (T4 GPU again; if no T4 is offered, wait and retry - never continue on CPU).
2. Run the cells again from the top. Each cell knows what is already done and continues where it stopped.

After the second kill and the benchmark, a results file downloads: **upload it to the chat**.

In session 2, **never use "Restart session"** to recover: always **Disconnect and delete runtime**, then reconnect (a
restarted session stays on the same machine, which spoils the resume test). If a cell says "Restart session", that
applies only before the smoke run has started.

Never edit files inside `/content/eem-acc-skripsi` (a changed checkout stops the run: it must stay clean)."""

SETTINGS = """# 0) Settings
SESSION = 1          # 1 = verification; 2 = epoch 0 + smoke + kills + benchmark (only after I push the session-2 code)
NB_VERSION = "@NB_VERSION@"   # fingerprint of this notebook's cells; the setup cell refuses an outdated copy"""

SETUP = """# 1) Setup: timeline, GPU, Drive, pinned code (every runtime)
import os, sys, subprocess, tempfile, json, time, glob, shutil, hashlib
from pathlib import Path
T_FIRST_CELL = time.time()
REPO_URL = "https://github.com/notanut/eem-acc-skripsi.git"
IN_COLAB = "google.colab" in sys.modules
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", gpu.stdout.strip() or "NONE")
if IN_COLAB and "T4" not in gpu.stdout:
    raise RuntimeError("No T4 GPU: Runtime -> Change runtime type -> T4 GPU (if none is offered, wait and retry).")
if IN_COLAB:
    from google.colab import drive
    t0 = time.time(); drive.mount("/content/drive"); T_MOUNT = time.time() - t0
    REPO_DIR = "/content/eem-acc-skripsi"
    DRIVE_ROOT = "/content/drive/MyDrive/Skripsi"
    def git(*args, cwd=REPO_DIR, capture=False):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=capture, text=True)
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        git("clone", REPO_URL, REPO_DIR, cwd=None)
    git("fetch", "origin")
    # the session's code is pinned to the commit its FIRST runtime used (recorded on Drive); later runtimes check it out
    sess_file = Path(DRIVE_ROOT) / "stage3" / "s33b" / f"session{SESSION}.json"
    pinned = json.loads(sess_file.read_text()).get("stamp", {}).get("commit") if sess_file.exists() else None
    target = pinned or git("rev-parse", "origin/main", capture=True).stdout.strip()
    head = git("rev-parse", "HEAD", capture=True).stdout.strip()
    if head != target:
        if "eem_acc" in sys.modules:
            raise RuntimeError("Other code is loaded: Runtime -> Restart session, then run from the top.")
        git("checkout", "--detach", target)
    dirty = git("status", "--porcelain", "--untracked-files=no", capture=True).stdout.strip()
    assert not dirty, f"the checkout is not clean: {dirty}"
    nb_src = "".join(json.loads((Path(REPO_DIR) / "notebooks" / "04_stage3.ipynb").read_text())["cells"][1]["source"])
    if f'NB_VERSION = "{NB_VERSION}"' not in nb_src:
        raise RuntimeError("This notebook is an OUTDATED copy (its cells differ from the code it would run). Close it and "
                           f"open the link I gave in the chat (commit {target[:7]}), then run from the top.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt", "-e", REPO_DIR], check=True)
else:                                   # local test run: this checkout + a temp folder
    T_MOUNT = 0.0
    REPO_DIR = str(Path.cwd().parent) if Path.cwd().name == "notebooks" else str(Path.cwd())
    DRIVE_ROOT = tempfile.mkdtemp(prefix="eem_acc_")
import importlib
if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")
importlib.invalidate_caches()
PINS_FILE = Path(REPO_DIR) / "configs" / "t4_pins.json"
PINS = json.loads(PINS_FILE.read_text()) if PINS_FILE.exists() else {}
from eem_acc.stage3.llm import t4                      # imports neither torch nor transformers
ROOT = Path(DRIVE_ROOT) / "stage3"                     # the campaign root (runner, epochs, ledger, lease, claims)
OUT = ROOT / "s33b"; OUT.mkdir(parents=True, exist_ok=True)
REPO_COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
S = t4.Sections(OUT / f"session{SESSION}.json", stamp={"commit": REPO_COMMIT, "nb_version": NB_VERSION})   # pins the code
if S.data.get("stamp", {}).get("commit") not in (None, REPO_COMMIT):
    raise RuntimeError(f"this session was started at another commit ({S.data['stamp']['commit']}); ask in the chat")
if SESSION == 2 and IN_COLAB:          # D142-D145: embedder libraries + MiniLM, BEFORE anything imports torch
    from eem_acc.stage3.llm import embed_setup as ES
    from eem_acc.stage3.llm.ledger import runtime_id
    def _sh(argv):
        r = subprocess.run(argv, capture_output=True, text=True)
        importlib.invalidate_caches()
        return r.returncode, r.stdout or "", r.stderr or ""
    def _freeze():
        rc, out, _ = _sh([sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"])
        return {d["name"]: d["version"] for d in json.loads(out.strip().splitlines()[-1])} if rc == 0 and out.strip() else None
    def _minilm(local_dir, rev):
        from huggingface_hub import snapshot_download
        snapshot_download(ES.MINILM_REPO, revision=rev, local_dir=str(local_dir), allow_patterns=ES.MINILM_FILES)
    LIBS = ES.ensure_embed(S, t4.Sections(OUT / f"session{SESSION}_library_attempts.json"), PINS, Path(DRIVE_ROOT),
                           Path("/content"), _sh, _minilm, loaded=ES.loaded_libraries(), freeze=_freeze,
                           runtime=runtime_id(), log=print)
    print("embedder:", LIBS["mode"], LIBS["versions"], LIBS["durations"], "| probe equal to runtime 1:",
          S.data["embed_probes"][-1]["equal_first"], "| CPU:", LIBS["probe"]["cpu"])
from eem_acc.stage3 import feasibility as F
from eem_acc.stage3.llm import runner as R
from eem_acc.stage3.llm.ledger import vm_boot_time
RUNTIME_INFO = {"vm_boot": vm_boot_time(), "first_cell": T_FIRST_CELL, "mount_seconds": T_MOUNT, "commit": REPO_COMMIT,
                "gpu": gpu.stdout.strip()}
print("commit", REPO_COMMIT[:12], "| saved sections:", [k for k in S.data if k != "stamp"])"""

MODEL = """# 3) Model: Drive cache (S3.0), local copy verified against S3.0's sha256 (every runtime)
from huggingface_hub import list_repo_files, hf_hub_download
MODEL_REPO, QUANT = "Qwen/Qwen2.5-7B-Instruct-GGUF", "q4_k_m"
MODEL_DIR = Path(DRIVE_ROOT) / "models" / MODEL_REPO.replace("/", "__")
files = sorted(f for f in list_repo_files(MODEL_REPO) if f.lower().endswith(".gguf") and QUANT in f.lower())
LOCAL = Path("/content/models") if IN_COLAB else Path(tempfile.mkdtemp())
LOCAL.mkdir(parents=True, exist_ok=True)
for f in files:
    src = Path(hf_hub_download(MODEL_REPO, f, local_dir=MODEL_DIR))
    dst = LOCAL / f
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy(src, str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst)
MODEL_PATHS = [LOCAL / f for f in files]
PARTS = t4.verify_parts(MODEL_PATHS)
assert PARTS["equal_s30"], f"the model differs from S3.0's: {PARTS}"
PART_SHA = PARTS["sha256"]
print("model verified:", list(PART_SHA))"""

LOAD = """# 4) Load the model on the GPU (every runtime)
import gc
from llama_cpp import Llama
for name in ("llm",):
    if globals().get(name) is not None:
        globals()[name].close(); globals()[name] = None
gc.collect()
t0 = time.time()
llm = Llama(model_path=str(MODEL_PATHS[0]), n_gpu_layers=-1, n_ctx=8192, n_batch=512, seed=0, verbose=False)
LOAD_SECONDS = time.time() - t0
from eem_acc.stage3.llm.llama_backend import load_params_of
from eem_acc.stage3.llm.canary import session_env
if not S.done("stamps"):
    S.save("stamps", {"wheel_sha256": WHEEL_SHA256, "wheel_equals_s30": WHEEL_SHA256 == t4.S30_WHEEL_SHA256,
                      "part_sha256": PART_SHA, "model_digest": t4.model_digest(PART_SHA),
                      "template_sha256": t4.template_check(llm)["sha256"], "load_params": load_params_of(llm),
                      "env": session_env(), "cuda_libraries": t4.cuda_libraries()})
print(f"loaded in {LOAD_SECONDS:.0f} s;", t4.gpu_telemetry(), "| wheel equals S3.0:", WHEEL_SHA256 == t4.S30_WHEEL_SHA256)"""

S1_VERIFY = """# 5) SESSION 1: verification (template, S3.0 equality, load params, libraries, canary counts, versions, timing)
if SESSION == 1:
    from eem_acc.stage3.llm.llama_backend import LlamaCppBackend, LlamaTokenizer
    from eem_acc.stage3.llm.canary import load_reference_prompts
    s30 = json.loads((Path(REPO_DIR) / "docs/evidence/s30_results_2026-11-21.json").read_text())
    B = LlamaCppBackend(llm, t4.model_digest(PART_SHA), WHEEL_SHA256)
    TOKZ = LlamaTokenizer(B)
    S.save("runtime", RUNTIME_INFO)
    if not S.done("template"):
        S.save("template", t4.template_check(llm))
    if not S.done("s30_equality"):
        chat = lambda m, g: (llm.reset(), F.make_generate(llm, grammar=g, reset=True, max_tokens=t4.S30_MAX_TOKENS)(m))[1]
        S.save("s30_equality", t4.s30_equality(B, s30["timing"]["records"], F.sample_prompts(20, seed=0), chat))
    if not S.done("load_params"):
        S.save("load_params", {**t4.load_params_report(llm, s30["load"]["stamp"]), "cuda_libraries": t4.cuda_libraries()})
    if not S.done("canary_counts"):
        S.save("canary_counts", t4.canary_counts(TOKZ, load_reference_prompts()[0]))
    if not S.done("versions"):
        S.save("versions", {"libraries": t4.library_versions(), "minilm_revision": t4.minilm_revision()})
    if not S.done("timing"):
        S.save("timing", t4.timing_check(S.data["s30_equality"]["records"]))
    eq = S.data["s30_equality"]
    print("template equals S3.0:", S.data["template"]["equals_s30"], "| S3.0 equality:", eq["equal"], "/", eq["compared"])
    print("timing:", S.data["timing"], "| canary counts equal:", S.data["canary_counts"]["equal"])
    if not (S.data["template"]["equals_s30"] and eq["all_equal"]):
        print(">>> STOP: the platform check failed. Upload the results file (last cell) to the chat. <<<")"""

S1_D47 = """# 6) SESSION 1: D47 on the GPU (reset path vs a freshly constructed model; at most two models at a time)
if SESSION == 1 and not S.done("d47"):
    longest = sorted(load_reference_prompts()[0], key=lambda p: -p["n_tokens"])[:2]
    fresh = lambda: LlamaCppBackend(Llama(model_path=str(MODEL_PATHS[0]), n_gpu_layers=-1, n_ctx=8192, n_batch=512,
                                          seed=0, verbose=False), t4.model_digest(PART_SHA), WHEEL_SHA256)
    S.save("d47", t4.d47_check(B, fresh, longest, TOKZ))
    gc.collect()
if SESSION == 1:
    print("D47:", S.data["d47"])"""

S1_TESTS = """# 7) SESSION 1: Colab-marked tests (their own process loads one more model: two in total)
if SESSION == 1 and not S.done("colab_tests"):
    env = dict(os.environ, EEM_GGUF=str(MODEL_PATHS[0]), EEM_MINILM_REV=S.data["versions"]["minilm_revision"] or "",
               PYTHONPATH=f"{REPO_DIR}/src:{REPO_DIR}/tests")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "colab", "-p", "no:cacheprovider",
                        "tests/test_s33_colab.py"], cwd=REPO_DIR, env=env, capture_output=True, text=True)
    S.save("colab_tests", {"returncode": r.returncode, "tail": (r.stdout + r.stderr)[-4000:]})
if SESSION == 1:
    print(S.data["colab_tests"]["tail"][-800:])"""

S2_OPEN = """# 8) SESSION 2: open the runtime (canary -> epoch; lease; claims of dead runtimes broken) - every runtime
if SESSION == 2:
    assert WHEEL_SHA256, "no cached wheel sha256 (cell 2): ask in the chat"
    if globals().get("SES") is not None:                 # a re-run of this cell in the same kernel
        SES.close()
    if not S.done("minilm_test"):  # session 1's failed Colab test, re-run once with the D142 libraries (no GGUF load)
        env = dict(os.environ, EEM_MINILM_REV=PINS["minilm_revision"], EEM_MINILM_PATH=LIBS["minilm_path"],
                   PYTHONPATH=f"{REPO_DIR}/src:{REPO_DIR}/tests")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "colab", "-p", "no:cacheprovider",
                            "tests/test_s33_colab.py::test_minilm_repeatable_across_processes"],
                           cwd=REPO_DIR, env=env, capture_output=True, text=True)
        res = {"returncode": r.returncode, "tail": (r.stdout + r.stderr)[-4000:]}
        if r.returncode != 0:                            # only a pass is kept; each failure is recorded
            S.append("minilm_test_failures", res)
            print(res["tail"][-1500:])
            raise RuntimeError("the MiniLM test failed: stop here and upload session2.json from Drive (Skripsi/stage3/s33b) to the chat")
        S.save("minilm_test", res)
    from eem_acc.stage3.llm.embed import MiniLMEmbedder
    EMB = MiniLMEmbedder(PINS["minilm_revision"], path=LIBS["minilm_path"])
    assert EMB.cpu_capability == "AVX2" and EMB.id.endswith("isa-compat-avx2"), (EMB.cpu_capability, EMB.id)   # D143
    EMB.embed("warm-up: torch, MKL and OpenMP initialised as during the run")
    from eem_acc.stage3.llm.embed import probe_vectors
    from eem_acc.stage3.llm.ledger import runtime_id
    KP = {**probe_vectors(EMB), "runtime": runtime_id(), "check_sha256": LIBS["probe"]["vectors_sha256"]}
    KP["equal"] = KP["vectors_sha256"] == KP["check_sha256"]
    if not any(k["runtime"] == KP["runtime"] for k in S.data.get("kernel_probes", [])):
        S.append("kernel_probes", KP)
    if not KP["equal"]:                  # D146: the in-process embedder must equal the setup's fresh-process check
        raise RuntimeError("the kernel's embedder differs from the setup check: stop here and upload session2.json from Drive to the chat")
    if not S.done("torch_llama_check"):  # D145: S3.0 equality again (2 free + 2 grammar) with torch loaded and run
        from eem_acc.stage3.llm.llama_backend import LlamaCppBackend
        s30 = json.loads((Path(REPO_DIR) / "docs/evidence/s30_results_2026-11-21.json").read_text())
        chat = lambda m, g: (llm.reset(), F.make_generate(llm, grammar=g, reset=True, max_tokens=t4.S30_MAX_TOKENS)(m))[1]
        eq = t4.s30_equality(LlamaCppBackend(llm, t4.model_digest(PART_SHA), WHEEL_SHA256), s30["timing"]["records"],
                             F.sample_prompts(20, seed=0), chat, limit=2)
        S.save("torch_llama_check", {"s30_equality": eq, "torch_loaded": "torch" in sys.modules})
    if not S.data["torch_llama_check"]["s30_equality"]["all_equal"]:
        raise RuntimeError("S3.0 equality fails with torch loaded: stop here and upload session2.json from Drive to the chat")
    cl = t4.cuda_libraries()                # every runtime: llama.cpp's CUDA libraries vs session 1 and runtime 1
    if cl.get("error") or not cl.get("linked"):
        raise RuntimeError(f"could not read llama.cpp's CUDA libraries ({cl.get('error')}): re-run this cell")
    s1 = json.loads((Path(REPO_DIR) / "docs/evidence/s33b_session1_2026-11-23.json").read_text())
    CC = {"runtime": runtime_id(), "linked": t4.linked_paths(cl), "fingerprint": t4.cuda_fingerprint(cl)}
    CC["equal_session1"] = CC["linked"] == t4.linked_paths(s1["stamps"]["cuda_libraries"])
    prev = S.data.get("cuda_checks", [])
    CC["equal_first"] = CC["fingerprint"] == (prev[0]["fingerprint"] if prev else CC["fingerprint"])
    if not any(c["runtime"] == CC["runtime"] for c in prev):
        S.append("cuda_checks", CC)
    if not (CC["equal_session1"] and CC["equal_first"]):
        raise RuntimeError("llama.cpp's CUDA libraries differ from session 1 or runtime 1: stop here and upload session2.json from Drive to the chat")
    SES = t4.T4Session(ROOT, llm, PART_SHA, WHEEL_SHA256, EMB, session_id="s33b-2", load_seconds=LOAD_SECONDS)
    RUN = SES.runner
    S.append("runtimes", {**RUNTIME_INFO, "runtime": SES.runtime, "epoch": SES.epoch, "canary": SES.status,
                          "fingerprint": SES.fingerprint, "broken_claims": SES.broken_claims,
                          "canary_seconds": SES.canary_seconds, "load_seconds": LOAD_SECONDS})
    print(f"runtime {len(S.data['runtimes'])}: epoch {SES.epoch} ({SES.status})")
    if len(S.data["runtimes"]) > 1 and SES.status != "pass":
        print(">>> the canary did not pass in this runtime: the resume test is inconclusive (it is recorded). <<<")
    REF, (U_C2, U_C0, U_RQ3) = t4.smoke_plan()
    for phase, units in (("smoke-ref", REF), ("smoke", [U_C2, U_C0, U_RQ3])):
        if not (ROOT / "plans" / f"{phase}.json").exists():
            R.write_plan(ROOT, phase, units, 1)"""

S2_SMOKE = """# 9) SESSION 2: smoke run with the two forced kills (re-run this cell after each reconnect)
if SESSION == 2:
    RT = SES.runtime
    RUN.run_phase("smoke-ref")                                   # the uninterrupted reference, first
    # read BEFORE the cache is opened; keyed on the arming record (saved long before the kill; the "requested" record,
    # written just before it, may itself be lost on Drive)
    if S.done("kill_a_armed") and S.data["kill_a_armed"]["runtime"] != RT and not S.done("kill_a_drive"):
        S.save("kill_a_drive", {**t4.drive_state(RUN.cache_path(U_C2)), "runtime": RT,
                                "records_expected": t4.records_at_kill(RUN.cache_path(REF[0]), t4.KILL_A_AFTER),
                                "requested_record_reached_drive": S.done("kill_a_requested"),
                                "unit_done_before_resume": RUN.is_done(U_C2)})
    if S.done("kill_b_armed") and S.data["kill_b_armed"]["runtime"] != RT and not S.done("kill_b_drive"):
        st = t4.drive_state(RUN.cache_path(U_C0), RUN.result_path(U_C0))
        sha = (R.split_result(json.loads(RUN.result_path(U_C0).read_text()))[1].get("payload_sha256")
               if st.get("result_present") else None)
        S.save("kill_b_drive", {**st, "is_done": RUN.is_done(U_C0, full=True), "payload_sha256": sha, "runtime": RT,
                                "requested_record_reached_drive": S.done("kill_b_requested")})
    if t4.arm_kill_a(RUN, U_C2, S, RT):
        print(f">>> KILL (a): the runtime will be killed after {t4.KILL_A_AFTER} LLM calls of {U_C2.unit_id}. <<<")
        RUN.ctx.backend = t4.KillAfter(SES.backend, t4.KILL_A_AFTER, t4.make_kill(S, "kill_a", RT))
        RUN.after_result = None
    else:
        RUN.ctx.backend = SES.backend
        if t4.arm_kill_b(RUN, U_C0, t4.make_kill(S, "kill_b", RT), S, RT):
            print(f">>> KILL (b): the runtime will be killed right after {U_C0.unit_id}'s result is written. <<<")
        else:
            RUN.after_result = None
    out = RUN.run_phase("smoke")
    S.append("smoke_runs", {"runtime": RT, **out})
    for name in ("kill_a", "kill_b"):                             # an armed kill that never fired is a failed test
        if (S.done(f"{name}_armed") and S.data[f"{name}_armed"]["runtime"] == RT and not S.done(f"{name}_not_fired")
                and not S.done(f"{name}_requested")):
            S.save(f"{name}_not_fired", {"runtime": RT})
    if RUN.is_done(U_C2) and not S.done("kill_a_resumed"):
        r2 = R.split_result(json.loads(RUN.result_path(U_C2).read_text()))[1]
        S.save("kill_a_resumed", {"resumed": r2["resumed"], "restarts": r2["restarts"], "runtime": RT,
                                  "cross_cpu_resume": r2.get("cross_cpu_resume"),
                                  "resumed_in_new_runtime": RT != S.data.get("kill_a_armed", {}).get("runtime")})
    if S.done("kill_b_drive") and RUN.is_done(U_C0) and not S.done("kill_b_verdict"):
        now = R.split_result(json.loads(RUN.result_path(U_C0).read_text()))[1]
        S.save("kill_b_verdict", {"counted_done_without_rerun": not now["restarts"] and
                                  now["payload_sha256"] == S.data["kill_b_drive"]["payload_sha256"],
                                  "payload_identical": now["payload_sha256"] == S.data["kill_b_drive"]["payload_sha256"],
                                  "restarts": now["restarts"]})
    print("smoke:", out)
    for k in ("kill_a_drive", "kill_a_resumed", "kill_b_drive", "kill_b_verdict", "kill_a_lost", "kill_b_lost",
              "kill_a_not_fired", "kill_b_not_fired"):
        if S.done(k):
            print(k, S.data[k])"""

S2_CHECKS = """# 10) SESSION 2: smoke checks (interrupted vs reference, verify_replay on CPU, audit, statistics, Drive latency)
if SESSION == 2 and not S.done("smoke_checks"):
    assert all(RUN.is_done(u) for u in REF + [U_C2, U_C0, U_RQ3]), "run cell 9 until the smoke run is complete"
    from eem_acc.stage3.llm.audit import audit_phase
    from eem_acc.stage3.llm.fsio import read_jsonl
    from eem_acc.stage3.llm.llama_backend import key_source_from_header, LlamaTokenizer
    from eem_acc.stage3.llm.cache import read_header
    load = lambda u: R.split_result(json.loads(RUN.result_path(u).read_text()))[0]
    cmp = t4.compare_units(RUN.cache_path(U_C2), RUN.cache_path(REF[0]), load(U_C2), load(REF[0]))
    vocab = Llama(model_path=str(MODEL_PATHS[0]), vocab_only=True, verbose=False)
    verified = {}
    for u in REF + [U_C2, U_C0, U_RQ3]:
        ks = key_source_from_header(read_header(RUN.cache_path(u)), vocab)
        try:
            verified[u.unit_id] = R.verify_replay(ROOT, u.phase, u.unit_id, ks, LlamaTokenizer(ks))
        except Exception as e:
            verified[u.unit_id] = f"{type(e).__name__}: {e}"
    results = [{**load(u), "run": R.split_result(json.loads(RUN.result_path(u).read_text()))[1]}
               for u in REF + [U_C2, U_C0, U_RQ3]]
    fails = read_jsonl(ROOT / "failures" / "smoke.jsonl")
    S.save("smoke_checks", {"interrupted_vs_reference": cmp, "verify_replay": verified,
                            "audit": {k: v for k, v in audit_phase(ROOT, "smoke").items() if k != "hits"} | {
                                "hits": audit_phase(ROOT, "smoke")["hits"][:20]},
                            "stats": t4.smoke_stats(results, lambda uj: RUN.cache_path(R.Unit.from_json(uj)), fails),
                            "append_latency": t4.append_latency(ROOT / "s33b" / "latency.jsonl")})
    heavy = ("steps", "note_events", "llm_calls", "final_notes", "embedding_cache", "window_trace")
    S.save("smoke_results", [{k: v for k, v in r.items() if k not in heavy} for r in results])   # slim; caches on Drive
if SESSION == 2:
    print(json.dumps({k: S.data["smoke_checks"][k] for k in ("interrupted_vs_reference", "verify_replay")}, indent=1))"""

S2_BENCH_AB = """# 11) SESSION 2: D53 benchmark A (fresh) vs B (static-prefix snapshot) - the GPU-hour counter starts here
if SESSION == 2 and not S.done("bench_ab"):
    assert S.done("smoke_checks"), "run cell 10 first (the budget counter must not start before the smoke run ends)"
    from eem_acc.stage3.llm.llama_backend import LlamaCppBackend
    from eem_acc.stage3.llm.canary import run_canary, fingerprint
    if not S.done("benchmark_start"):
        SES.ledger.event("benchmark_start"); S.save("benchmark_start", time.time())
    caches = [RUN.cache_path(u) for u in REF + [U_C2, U_C0, U_RQ3]]
    W = t4.benchmark_workload(caches, n=60)
    SNAP = LlamaCppBackend(llm, t4.model_digest(PART_SHA), WHEEL_SHA256, snapshot=True)
    ab = t4.benchmark_ab(SNAP, W)
    t4.drop_snapshot_states(SNAP)
    ab["snapshot_canary_fingerprint"] = fingerprint(run_canary(
        lambda p: SNAP.generate(p["messages"], p["kind"], p["grammar"], p["max_tokens"]), SES.prompts))
    ab["fresh_canary_fingerprint"] = SES.fingerprint
    ab["ram_probe"] = t4.snapshot_ram_probe(SNAP, t4.wording_system_prompts(SES.prompts))
    t4.drop_snapshot_states(SNAP)
    S.save("bench_ab", ab)
if SESSION == 2:
    print({k: S.data["bench_ab"][k] for k in ("n", "seconds_A", "seconds_B", "saving_B", "B_repeatable", "A_all_equal_recorded")})
    if not S.data["bench_ab"]["A_all_equal_recorded"]:
        print(">>> A did not reproduce every recorded smoke output: report this in the chat. <<<")"""

S2_BENCH_C = """# 12) SESSION 2: D53 benchmark C - two worker processes on real units (the kernel's model is closed first)
if SESSION == 2 and not S.done("bench_c"):
    assert S.done("bench_ab"), "run cell 11 first"
    assert WHEEL_SHA256, "no wheel sha256"
    if SES.backend.llm is not None:
        SES.backend.llm.close(); SES.backend.llm = None
    llm = None; gc.collect(); time.sleep(5)
    tele = t4.gpu_telemetry()
    assert int(float(tele.get("memory.used", "0"))) < 1500, f"GPU memory not freed: {tele}"
    attempt = len(S.data.get("bench_c_attempts", [])) + 1        # a fresh phase per attempt (no stale results)
    PH = f"bench2-a{attempt}"
    B2 = [R.Unit(PH, u.rq, u.split, u.kind, u.seed, u.condition, n_tasks=u.n_tasks) for u in (U_C2, U_C0)]
    R.write_plan(ROOT, PH, B2, 1)
    parts_file = OUT / "parts.json"; parts_file.write_text(json.dumps(PART_SHA))
    start_at = time.time() + 480                                  # synchronised start after both loads and canaries
    S.append("bench_c_attempts", {"phase": PH, "start_at": start_at, "runtime": SES.runtime})
    logs = [open(f"/content/w{k}.log", "w") for k in range(2)]
    procs = [subprocess.Popen([sys.executable, "-m", "eem_acc.stage3.llm.unit_worker", "--root", str(ROOT),
                               "--phase", PH, "--shard", f"{k}/2", "--model", str(MODEL_PATHS[0]),
                               "--parts", str(parts_file), "--wheel-sha", WHEEL_SHA256,
                               "--minilm-rev", PINS["minilm_revision"], "--minilm-path", LIBS["minilm_path"],
                               "--session", f"s33b-2c-a{attempt}",
                               "--start-at", str(start_at), "--out", f"/content/w{k}.json"],
                              cwd=REPO_DIR, env=dict(os.environ, PYTHONPATH=f"{REPO_DIR}/src"),
                              stdout=logs[k], stderr=subprocess.STDOUT) for k in range(2)]
    peak, t_end = [], time.time() + 4 * 3600
    while any(p.poll() is None for p in procs) and time.time() < t_end:
        peak.append(t4.gpu_telemetry()); time.sleep(5)
    for p in procs:
        if p.poll() is None:
            p.kill()
    for f in logs:
        f.close()
    outs = [json.loads(Path(f"/content/w{k}.json").read_text()) for k in range(2) if Path(f"/content/w{k}.json").exists()]
    ok = len(outs) == 2 and all(o["run"]["ran"] == 1 and o["run"]["failed"] == 0 for o in outs)
    walls = {u.unit_id: t4.first_full_wall(ROOT, u.unit_id) for u in (REF[0], U_C0)}
    base = sum(w for w, _ in walls.values()) if all(w for w, _ in walls.values()) else None
    load = lambda u: R.split_result(json.loads(RUN.result_path(u).read_text()))[0]
    wall = (max(o["end"] for o in outs) - min(o["start"] for o in outs)) if ok else None
    res = {"ok": ok, "workers": outs, "logs_tail": [Path(f"/content/w{k}.log").read_text()[-2000:] for k in range(2)],
           "late_start_s": [round(o["start"] - start_at, 1) for o in outs], "wall_two_processes": wall,
           "wall_single_process": base, "baseline_source": walls, "gain": (1 - wall / base) if (wall and base) else None,
           "vram_peak_mib": max((int(float(t.get("memory.used", 0))) for t in peak if t), default=None),
           "canary_equal_epoch0": len(outs) == 2 and all(o["canary_fingerprint"] == SES.fingerprint for o in outs)}
    if ok:
        res["payload_vs_smoke"] = {u.unit_id: t4.compare_units(RUN.cache_path(u), RUN.cache_path(v), load(u), load(v))
                                   for u, v in zip(B2, (REF[0], U_C0))}
        S.save("bench_c", res)
    else:
        S.append("bench_c_failed", res)
        print(">>> benchmark C failed (see bench_c_failed in the results); you can re-run this cell once. <<<")
if SESSION == 2 and S.done("bench_c"):
    print({k: S.data["bench_c"][k] for k in ("wall_two_processes", "wall_single_process", "gain", "canary_equal_epoch0")})"""

END = """# 13) Done: stop the heartbeat, release the lease, download the results file - upload it to the chat
if SESSION == 2 and globals().get("SES") is not None:
    SES.close()
    from eem_acc.stage3.llm.ledger import summarise
    S.save("ledger", {k: v for k, v in summarise(ROOT).items() if k != "unit_wall_seconds"})
print("results file:", S.path, "| sections:", [k for k in S.data if k != "stamp"])
if IN_COLAB:
    from google.colab import files
    files.download(str(S.path))
RELEASE_RUNTIME = False      # True = end the Colab runtime after the download (saves GPU time)
if IN_COLAB and RELEASE_RUNTIME:
    time.sleep(20)
    from google.colab import runtime
    runtime.unassign()"""


def cell(src: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def build() -> dict:
    wheel = s30_cell("Install llama-cpp-python").replace("# 2) Install", "# 2) Install", 1)
    others = [MD_INTRO, SETUP, wheel, MODEL, LOAD, S1_VERIFY, S1_D47, S1_TESTS, S2_OPEN, S2_SMOKE, S2_CHECKS,
              S2_BENCH_AB, S2_BENCH_C, END]
    import hashlib
    nb_version = hashlib.sha256(json.dumps([SETTINGS] + others).encode()).hexdigest()[:12]
    cells = [cell(MD_INTRO, "markdown"), cell(SETTINGS.replace("@NB_VERSION@", nb_version)), cell(SETUP), cell(wheel),
             cell(MODEL), cell(LOAD), cell(S1_VERIFY), cell(S1_D47), cell(S1_TESTS), cell(S2_OPEN), cell(S2_SMOKE),
             cell(S2_CHECKS), cell(S2_BENCH_AB), cell(S2_BENCH_C), cell(END)]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    out = ROOT / "notebooks" / "04_stage3.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(out)
