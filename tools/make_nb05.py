"""Build notebooks/05_s34.ipynb: the S3.4 automatic pilot chain on the Colab T4 (docs/s34_spec.md r2).

Cells 1-5 are notebook 04's setup, wheel, model and load cells (fingerprint guard, pinned commit, embedder setup), with
the session file replaced by `stage3/s34/chain.json`; cell 6 opens the runtime (embedder, D146 kernel probe, CUDA
check recorded per D149, canary -> epoch); cell 7 runs the chain; cell 8 writes and downloads the bundle.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_nb04 as NB4  # noqa: E402

MD_INTRO = """# 05 - S3.4: the automatic pilot chain (Stage 3)

Runs the S3.4 chain of `docs/s34_spec.md`: docs wording -> pilot (switches, acceptance, recalibration ladder) -> RQ3
wiring check -> budget gate -> freeze -> EEM selection (D60) -> report. It takes many Colab sessions (about 10
GPU-hours, more if a switch or a ladder step fires); every session continues where the last one stopped.

**Every session:** Runtime -> Change runtime type -> **T4 GPU**; open this notebook from the commit-pinned link in the
chat; **Run all**. Keep the tab open. If the runtime is lost, reconnect (T4) and Run all again.

**The runtime ends by itself** after the bundle (`RELEASE_RUNTIME`); the bundle is also saved on Drive
(`Skripsi/stage3/s34/s34_bundle.json`) if the download is blocked.

**The chain stops by itself** only when a rule hands the decision to the user (a PAUSE line says why), when units
failed once (a RETRY line: the next session retries them in a fresh runtime), or when it is done. Either way the last
cell downloads `s34_bundle.json`: upload it to the chat.

**Settings (cell 0)** change only when the chat says so: `USER_DECISION` + `EXPECT_LOG` answer a pause; `REPIN` moves
the chain to a fixed commit (finished units are kept when they replay identically under the fix)."""

SETTINGS = """# 0) Settings
NB_VERSION = "@NB_VERSION@"   # fingerprint of this notebook's cells; the setup cell refuses an outdated copy
USER_DECISION = None          # the chat's answer to a pause, e.g. {"answers": 7, "action": "accept"} (spec section 11)
EXPECT_LOG = None             # the last bundle's log, e.g. "8:3fa2c91b04de": refuses an older chain file on Drive
REPIN = False                 # or the commit the chat names after a code fix (D166; it must descend from the chain's)
RELEASE_RUNTIME = True        # end the Colab runtime after the bundle (D170; the bundle is also on Drive)
CHAIN_VERSION = 1             # fixed: a code fix is adopted by replay (REPIN), not a new chain version (D166)
CHAIN_FILE = "chain.json\""""


TARGET_LINE = '    target = pinned or git("rev-parse", "origin/main", capture=True).stdout.strip()\n'
REPIN_BLOCK = """    if REPIN:                           # D166: the fixed commit the chat names; it must descend from the chain's commit
        target = git("rev-parse", REPIN, capture=True).stdout.strip()
        if pinned and subprocess.run(["git", "merge-base", "--is-ancestor", pinned, target], cwd=REPO_DIR).returncode:
            raise RuntimeError(f"REPIN {REPIN} does not descend from the chain's commit {pinned[:7]}: ask in the chat")
"""


def setup_cell() -> str:
    s = NB4.SETUP
    reps = [
        ('sess_file = Path(DRIVE_ROOT) / "stage3" / "s33b" / f"session{SESSION}.json"',
         'sess_file = Path(DRIVE_ROOT) / "stage3" / "s34" / CHAIN_FILE'),
        ('"notebooks" / "04_stage3.ipynb"', '"notebooks" / "05_s34.ipynb"'),
        ('''        raise RuntimeError(f"This notebook's cells differ from SESSION {SESSION}'s pinned commit {target[:7]}. Check that "
                           f"SESSION is right; otherwise open the Colab link for commit {target[:7]} from the chat and "
                           "run from the top.")''',
         '''        raise RuntimeError(f"This notebook's cells differ from the S3.4 chain's pinned commit {target[:7]}: open "
                           f"the Colab link for commit {target[:7]} from the chat and run from the top.")'''),
        ('OUT = ROOT / "s33b"', 'OUT = ROOT / "s34"'),
        ('S = t4.Sections(OUT / f"session{SESSION}.json", stamp=', 'S = t4.Sections(OUT / CHAIN_FILE, stamp='),
        ('"nb_version": NB_VERSION})   # pins the code',
         '"nb_version": NB_VERSION}, repin=bool(REPIN),\n                expect_log=EXPECT_LOG)   # pins the code; M6: before any write'),
        (TARGET_LINE, TARGET_LINE + REPIN_BLOCK),
        ('if SESSION == 2 and IN_COLAB:          # D142-D145', 'if IN_COLAB:                           # D142-D148'),
        ('t4.Sections(OUT / f"session{SESSION}_library_attempts.json")', 't4.Sections(OUT / "library_attempts.json")'),
        ('"this session was started at another commit', '"the S3.4 chain was started at another commit'),
    ]
    for a, b in reps:
        assert a in s, a
        s = s.replace(a, b)
    assert "SESSION" not in s, "a SESSION reference is left in the setup cell"
    return s


OPEN = """# 6) Open the runtime: embedder, D146 kernel probe, CUDA check (recorded, D149), canary -> epoch (every runtime)
assert IN_COLAB, "S3.4 runs on the Colab T4 only"
assert os.path.ismount("/content/drive"), "Drive is not mounted (the session ended): Run all from the top"
assert WHEEL_SHA256, "no cached wheel sha256 (cell 2): ask in the chat"
STOP = None                                              # a reason to stop before the chain (the bundle still runs)
if globals().get("SES") is not None:                     # a re-run of this cell in the same kernel
    SES.close()
from eem_acc.stage3.llm.embed import MiniLMEmbedder, probe_vectors
from eem_acc.stage3.llm.ledger import runtime_id
EMB = MiniLMEmbedder(PINS["minilm_revision"], path=LIBS["minilm_path"])
assert EMB.cpu_capability == "AVX2" and EMB.id.endswith("isa-compat-avx2"), (EMB.cpu_capability, EMB.id)   # D143
EMB.embed("warm-up: torch, MKL and OpenMP initialised as during the run")
KP = t4.kernel_probe_check(probe_vectors(EMB), LIBS["probe"]["vectors_sha256"], runtime_id())
if not any(k["runtime"] == KP["runtime"] for k in S.data.get("kernel_probes", [])):
    S.append("kernel_probes", KP)
if not KP["equal"]:                      # D146 (strict): the in-process embedder must equal the setup's check
    STOP = "the kernel's embedder differs from the setup check (D146)"
s1 = json.loads((Path(REPO_DIR) / "docs/evidence/s33b_session1_2026-11-23.json").read_text())
prev = S.data.get("cuda_checks", [])
CC = t4.cuda_check(t4.cuda_libraries(), s1["stamps"]["cuda_libraries"], prev, runtime_id())
if not any(c["runtime"] == CC["runtime"] for c in prev):
    S.append("cuda_checks", CC)
if not CC["ok"]:                         # D149: recorded, not a stop; the canary decides the epoch
    print(">>> llama.cpp's CUDA libraries differ from session 1 or an earlier runtime (recorded; the canary decides) <<<")
if STOP is None:
    try:                                 # second opinion L2: a start-up failure stops (the session closes itself)
        SES = t4.T4Session(ROOT, llm, PART_SHA, WHEEL_SHA256, EMB, session_id="s34", load_seconds=LOAD_SECONDS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        SES, STOP = None, f"the runtime did not start: {type(e).__name__}: {e}"
if STOP is None:
    RUN = SES.runner
    S.append("runtimes", {**RUNTIME_INFO, "runtime": SES.runtime, "epoch": SES.epoch, "canary": SES.status,
                          "fingerprint": SES.fingerprint, "broken_claims": SES.broken_claims,
                          "canary_seconds": SES.canary_seconds, "load_seconds": LOAD_SECONDS})
    print(f"runtime {len(S.data['runtimes'])}: epoch {SES.epoch} ({SES.status})")
    UD = USER_DECISION or {}
    if UD.get("action") == "allow_epoch" and UD.get("epoch") == SES.epoch and SES.epoch not in S.data.get(
            "allowed_epochs", []):           # the user's answer to an epoch pause (D165)
        S.append("allowed_epochs", SES.epoch)
    if SES.epoch != 0 and SES.epoch not in S.data.get("allowed_epochs", []):     # spec section 11.5: the user decides
        STOP = f"the canary put this runtime in epoch {SES.epoch}, not 0 (spec section 11.5)"
if STOP:
    print(f">>> PAUSE: {STOP}: the chain does not run; the bundle cell still runs - upload the bundle to the chat <<<")"""

CHAIN = """# 7) The S3.4 chain (re-run this cell to continue after an interruption in the same runtime)
from eem_acc.stage3 import s34 as S34
from eem_acc.stage3.s34_platform import Platform
from eem_acc.stage3.llm.cache import read_header
from eem_acc.stage3.llm.ledger import summarise, profile
from eem_acc.stage3.llm.llama_backend import key_source_from_header, LlamaTokenizer
VOCAB = Llama(model_path=str(MODEL_PATHS[0]), vocab_only=True, verbose=False)

def replay_tools(u):                                     # verify_replay's key source: the unit's cache header
    ks = key_source_from_header(read_header(RUN.cache_path(u)), VOCAB)
    return ks, LlamaTokenizer(ks)

assert not IN_COLAB or os.path.ismount("/content/drive"), "Drive is not mounted (the session ended): Run all"
CHAIN_STATE = None
if STOP:
    CHAIN_STATE = f"PAUSE: {STOP}"
else:
    PF = Platform(ROOT, RUN, REPO_COMMIT, replay_tools, version=CHAIN_VERSION, verified=S.data.get("verified"))
    CH = S34.Chain(S, ROOT, PF.run_units, PF.metrics, PF.checks, lambda: summarise(ROOT)["gpu_hours"],
                   lambda: profile(ROOT, "s34"), PF.freeze_extra, log=print, version=CHAIN_VERSION,
                   runtime=SES.runtime, adopt=PF.adopt)
    try:
        if USER_DECISION and USER_DECISION.get("action") != "allow_epoch":
            CH.apply_user_decision(USER_DECISION, EXPECT_LOG)
        CHAIN_STATE = CH.run()
    except S34.ChainRetry as e:                          # second opinion M3: no pause; the next session retries
        CHAIN_STATE = f"RETRY: {e}"
    except S34.ChainPause as e:
        CHAIN_STATE = f"PAUSE: {e}"
    except Exception as e:                               # never skip the bundle or leave the heartbeat on (review M5)
        import traceback
        traceback.print_exc()
        CHAIN_STATE = f"ERROR: {type(e).__name__}: {e}"
    except KeyboardInterrupt:                            # second opinion L2: an interrupt still writes the bundle
        CHAIN_STATE = "INTERRUPTED (the units already finished are kept; Run all continues the chain)"
print(">>>", CHAIN_STATE, "<<<")"""

END = """# 8) End of session: per-unit replay checks and audit, stop the heartbeat, write and download the bundle
import hashlib
from eem_acc.stage3 import s34 as S34
from eem_acc.stage3.llm.ledger import summarise
assert not IN_COLAB or os.path.ismount("/content/drive"), "Drive is not mounted (the session ended): Run all"
VERIFIED = dict(S.data.get("verified", {}))              # review M3: every finished unit, once, kept across sessions
try:
    if globals().get("PF") is not None:
        VERIFIED = PF.verify_finished(VERIFIED)
        S.save("verified", VERIFIED)
except Exception as e:                                   # the bundle is still written (fix review L1)
    import traceback
    traceback.print_exc()
    CHAIN_STATE = f"{globals().get('CHAIN_STATE')} | verify: {type(e).__name__}: {e}"
finally:
    if globals().get("SES") is not None:
        SES.close()
ff = ROOT / "s34" / S34.freeze_file(CHAIN_VERSION)
LOG_FP = S34.log_fingerprint(S.data.get("log", []))      # second opinion M6: the chat quotes it in EXPECT_LOG
bundle = {"chain": S.data, "freeze": json.loads(ff.read_text()) if ff.exists() else None,
          "ledger": {k: v for k, v in summarise(ROOT).items() if k != "unit_wall_seconds"},
          "state": globals().get("CHAIN_STATE"), "log_fingerprint": LOG_FP, "commit": REPO_COMMIT,
          "chain_sha256": hashlib.sha256((OUT / CHAIN_FILE).read_bytes()).hexdigest(),
          "not_verified": sorted(k for k, v in VERIFIED.items() if v.get("verify_replay") is not True),
          "audit_hits": sum(len(v.get("audit_hits", [])) for v in VERIFIED.values())}
BUNDLE = OUT / "s34_bundle.json"                          # the canonical copy, on Drive
BUNDLE.write_text(json.dumps(bundle, indent=1, default=str))
print("bundle:", BUNDLE, "| state:", bundle["state"], "| log:", LOG_FP, "| GPU-hours:", bundle["ledger"]["gpu_hours"])
if IN_COLAB:
    import shutil
    from google.colab import drive, files
    LOCAL_BUNDLE = shutil.copy(BUNDLE, "/content/s34_bundle.json")     # the download reads the VM's local copy
    drive.flush_and_unmount()                            # 2nd opinion #2, M4: block until Drive has every write
    print("Drive flushed: every write of this session is on Drive (cells 6-8 now need Run all from the top)")
    files.download(LOCAL_BUNDLE)
if IN_COLAB and RELEASE_RUNTIME:
    time.sleep(20)                                       # the browser fetches the download
    from google.colab import runtime
    runtime.unassign()"""


def build() -> dict:
    wheel = NB4.s30_cell("Install llama-cpp-python")
    setup = setup_cell()
    others = [MD_INTRO, setup, wheel, NB4.MODEL, NB4.LOAD, OPEN, CHAIN, END]
    nb_version = hashlib.sha256(json.dumps([SETTINGS] + others).encode()).hexdigest()[:12]
    cells = [NB4.cell(MD_INTRO, "markdown"), NB4.cell(SETTINGS.replace("@NB_VERSION@", nb_version)), NB4.cell(setup),
             NB4.cell(wheel), NB4.cell(NB4.MODEL), NB4.cell(NB4.LOAD), NB4.cell(OPEN), NB4.cell(CHAIN), NB4.cell(END)]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    out = ROOT / "notebooks" / "05_s34.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(out)
