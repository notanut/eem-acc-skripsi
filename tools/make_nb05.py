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

**The chain stops by itself** only when a rule hands the decision to the user (a PAUSE line says why), or when it is
done. Either way the last cell downloads `s34_bundle.json`: upload it to the chat."""

SETTINGS = """# 0) Settings
NB_VERSION = "@NB_VERSION@"   # fingerprint of this notebook's cells; the setup cell refuses an outdated copy
CHAIN_VERSION = 1             # changes only for a new chain version after a code fix (the chat says so; spec section 2)
CHAIN_FILE = "chain.json" if CHAIN_VERSION == 1 else f"chain-v{CHAIN_VERSION}.json"
RELEASE_RUNTIME = False       # True = end the Colab runtime after the download (saves GPU time)"""


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
    SES = t4.T4Session(ROOT, llm, PART_SHA, WHEEL_SHA256, EMB, session_id="s34", load_seconds=LOAD_SECONDS)
    RUN = SES.runner
    S.append("runtimes", {**RUNTIME_INFO, "runtime": SES.runtime, "epoch": SES.epoch, "canary": SES.status,
                          "fingerprint": SES.fingerprint, "broken_claims": SES.broken_claims,
                          "canary_seconds": SES.canary_seconds, "load_seconds": LOAD_SECONDS})
    print(f"runtime {len(S.data['runtimes'])}: epoch {SES.epoch} ({SES.status})")
    if SES.epoch != 0:                   # spec section 11.5: the user decides
        STOP = f"the canary put this runtime in epoch {SES.epoch}, not 0 (spec section 11.5)"
if STOP:
    print(f">>> PAUSE: {STOP}: the chain does not run; the bundle cell still runs - upload the bundle to the chat <<<")"""

CHAIN = """# 7) The S3.4 chain (re-run this cell to continue after an interruption in the same runtime)
import hashlib
from eem_acc.stage3 import s34 as S34
from eem_acc.stage3.agent import prompts as P
from eem_acc.stage3.apisuite.docs import documented_all
from eem_acc.stage3.apisuite.streams import SUITE_VERSION
from eem_acc.stage3.llm.audit import audit_result
from eem_acc.stage3.llm.cache import UnitCache, read_header
from eem_acc.stage3.llm.ledger import summarise, profile
from eem_acc.stage3.llm.llama_backend import key_source_from_header, LlamaTokenizer
VOCAB = Llama(model_path=str(MODEL_PATHS[0]), vocab_only=True, verbose=False)

def _load(u):
    p = RUN.result_path(u)
    if not RUN.is_done(u) or not p.exists():
        return None
    payload, run = R.split_result(json.loads(p.read_text()))
    return {**payload, "run": run}

def run_units(phase, units, meta):
    plan_file = ROOT / "plans" / f"{phase}.json"
    if plan_file.exists():                               # a plan is written once; a reload must name the same units
        plan = R.read_plan(ROOT, phase)
        assert [R.Unit.from_json(u).unit_id for u in plan["units"]] == [u.unit_id for u in units], phase
        assert meta is None or plan.get("meta") == meta, f"{phase}: the plan's meta differs"     # review L8
    else:
        R.write_plan(ROOT, phase, units, CHAIN_VERSION, frozen_commit=REPO_COMMIT, meta=meta)
    try:
        RUN.run_phase(phase)
    except (R.RunHalted, R.TestPhaseCommitMismatch) as e:    # a code fix (or a new chain version) is needed
        print(f">>> {type(e).__name__}: {e} <<<")
    failed = RUN.failed_units(phase)
    out = []
    for u in units:
        r = _load(u)
        out.append(r if r is not None else {"unit": u.to_json(),
                                            "status": "failed" if u.unit_id in failed else "not_done"})
    return out

def metrics(r):
    if "run" not in r:
        return S34.unit_metrics(r, [])
    u = R.Unit.from_json(r["unit"])
    return S34.unit_metrics(r, UnitCache(RUN.cache_path(u), {}, "s", readonly=True).records)

def checks(results):
    ok, hits = True, 0
    for r in results:
        u = R.Unit.from_json(r["unit"])
        try:
            ks = key_source_from_header(read_header(RUN.cache_path(u)), VOCAB)
            ok = ok and R.verify_replay(ROOT, u.phase, u.unit_id, ks, LlamaTokenizer(ks)) is True
        except Exception as e:
            print(f"verify_replay {u.unit_id}: {type(e).__name__}: {e}")
            ok = False
        hits += len(audit_result(r))
    return ok, hits

PREFIX = "s34-" if CHAIN_VERSION == 1 else f"s34v{CHAIN_VERSION}-"

def freeze_extra(s):
    plans, payloads = {}, {}
    for pf in sorted((ROOT / "plans").glob(f"{PREFIX}[ABC]-r*.json")):
        plan = json.loads(pf.read_text())
        plans[pf.stem] = plan["sha256"]
        for uj in plan["units"]:                         # every step A-C result's payload sha256 (review M2)
            u = R.Unit.from_json(uj)
            payloads[u.unit_id] = R.split_result(json.loads(RUN.result_path(u).read_text()))[1]["payload_sha256"]
    return {"suite_version": SUITE_VERSION, "template_set_sha256": S34.template_set_sha(),
            "system_prompt_sha256": hashlib.sha256(P.system_prompt(documented_all(), s.wording).encode()).hexdigest(),
            "plans_sha256": plans, "payload_sha256": payloads}

CHAIN_STATE = None
if STOP:
    CHAIN_STATE = f"PAUSE: {STOP}"
else:
    CH = S34.Chain(S, ROOT, run_units, metrics, checks, lambda: summarise(ROOT)["gpu_hours"],
                   lambda: profile(ROOT, "s34"), freeze_extra, log=print, version=CHAIN_VERSION)
    try:
        CHAIN_STATE = CH.run()
    except S34.ChainPause as e:
        CHAIN_STATE = f"PAUSE: {e}"
    except Exception as e:                               # never skip the bundle or leave the heartbeat on (review M5)
        import traceback
        traceback.print_exc()
        CHAIN_STATE = f"ERROR: {type(e).__name__}: {e}"
print(">>>", CHAIN_STATE, "<<<")"""

END = """# 8) End of session: per-unit replay checks and audit, stop the heartbeat, write and download the bundle
from eem_acc.stage3 import s34 as S34
from eem_acc.stage3.llm.ledger import summarise
VERIFIED = dict(S.data.get("verified", {}))              # review M3: every finished unit, once, kept across sessions
if globals().get("RUN") is not None:
    from eem_acc.stage3.llm.audit import audit_result
    from eem_acc.stage3.llm.cache import read_header
    from eem_acc.stage3.llm.llama_backend import key_source_from_header, LlamaTokenizer
    VOCAB = globals().get("VOCAB") or Llama(model_path=str(MODEL_PATHS[0]), vocab_only=True, verbose=False)
    PREFIX = "s34-" if CHAIN_VERSION == 1 else f"s34v{CHAIN_VERSION}-"
    for pf in sorted((ROOT / "plans").glob(f"{PREFIX}*.json")):
        for uj in json.loads(pf.read_text())["units"]:
            u = R.Unit.from_json(uj)
            if u.unit_id in VERIFIED or not RUN.is_done(u):
                continue
            try:
                ks = key_source_from_header(read_header(RUN.cache_path(u)), VOCAB)
                v = R.verify_replay(ROOT, u.phase, u.unit_id, ks, LlamaTokenizer(ks))
            except Exception as e:
                v = f"{type(e).__name__}: {e}"
            payload, run = R.split_result(json.loads(RUN.result_path(u).read_text()))
            VERIFIED[u.unit_id] = {"verify_replay": v, "audit_hits": audit_result({**payload, "run": run})[:5]}
    S.save("verified", VERIFIED)
if globals().get("SES") is not None:
    SES.close()
ff = ROOT / "s34" / S34.freeze_file(CHAIN_VERSION)
bundle = {"chain": S.data, "freeze": json.loads(ff.read_text()) if ff.exists() else None,
          "ledger": {k: v for k, v in summarise(ROOT).items() if k != "unit_wall_seconds"},
          "state": globals().get("CHAIN_STATE"),
          "not_verified": sorted(k for k, v in VERIFIED.items() if v["verify_replay"] is not True),
          "audit_hits": sum(len(v["audit_hits"]) for v in VERIFIED.values())}
BUNDLE = OUT / "s34_bundle.json"
BUNDLE.write_text(json.dumps(bundle, indent=1, default=str))
print("bundle:", BUNDLE, "| state:", bundle["state"], "| GPU-hours:", bundle["ledger"]["gpu_hours"])
if IN_COLAB:
    from google.colab import files
    files.download(str(BUNDLE))
if IN_COLAB and RELEASE_RUNTIME:
    time.sleep(20)
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
