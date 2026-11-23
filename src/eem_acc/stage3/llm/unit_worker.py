"""D53 configuration C (D139): one worker process with its own model running one runner shard of a phase.

    python -m eem_acc.stage3.llm.unit_worker --root /content/drive/MyDrive/Skripsi/stage3 --phase bench2 --shard 0/2 \
        --model /content/models/qwen...-00001-of-00002.gguf --parts parts.json --wheel-sha <sha> --minilm-rev <sha> \
        --session s33b-2 --start-at <unix time> --out /content/w0.json [--n-threads 1] [--scripted]

Started as a fresh interpreter (never fork). The worker runs the canary (not registering an epoch: only the parent does)
while the other worker generates, then waits for `--start-at` (a synchronised start after both loads), runs its shard
with claims (no lease: the parent holds it), and writes timings and its canary fingerprint to --out. `--scripted` uses
the scripted policy and a bag-of-words embedder on the tiny model (CPU tests)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", required=True, help="JSON file {part name: sha256}")
    ap.add_argument("--wheel-sha", required=True)
    ap.add_argument("--minilm-rev", default=None)
    ap.add_argument("--session", required=True)
    ap.add_argument("--start-at", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-threads", type=int, default=1)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--header-override", default=None, help="JSON of header fields to override (tests only)")
    a = ap.parse_args(argv)
    import llama_cpp

    from .canary import EpochStore, Session, fingerprint, load_reference_prompts, run_canary, session_env
    from .ledger import Ledger, runtime_id
    from .llama_backend import LlamaCppBackend, LlamaTokenizer
    from .runner import Claims, RunContext, Runner, make_header
    from .t4 import model_digest
    k, n = (int(x) for x in a.shard.split("/"))
    t_load = time.time()
    llm = llama_cpp.Llama(model_path=a.model, n_gpu_layers=a.n_gpu_layers, n_ctx=8192, n_batch=512, seed=0,
                          n_threads=a.n_threads, n_threads_batch=a.n_threads, verbose=False)
    load_s = time.time() - t_load
    parts = json.loads(Path(a.parts).read_text())
    backend = LlamaCppBackend(llm, model_digest(parts), a.wheel_sha)
    tok = LlamaTokenizer(backend)
    if a.scripted:
        from ..agent import BagOfWordsEmbedder, ScriptedBackend
        emb = BagOfWordsEmbedder()
        run_backend = ScriptedBackend(tok, "memory")
    else:
        from .embed import MiniLMEmbedder
        emb = MiniLMEmbedder(a.minilm_rev)
        run_backend = backend
    root = Path(a.root)
    env = session_env()
    prompts, psha = load_reference_prompts()
    gen = lambda p: backend.generate(p["messages"], p["kind"], p["grammar"], p["max_tokens"])
    t0 = time.time()
    outs = run_canary(gen, prompts)
    canary_s = time.time() - t0
    fp = fingerprint(outs)
    epochs = EpochStore(root).read().get("fresh", [])
    match = next((e for e in epochs if e.get("fingerprint") == fp and e["prompts_sha256"] == psha), None)
    epoch = match["epoch"] if match else None
    header = make_header(backend.stamp(), emb.id, env, epoch, fp)   # the platform's identity (also in scripted tests)
    if a.header_override:
        header.update(json.loads(a.header_override))
    ledger = Ledger(root, a.session, runtime_id(), shard=f"w{k}")
    ledger.start(load_s)
    claims = Claims(root, runtime_id())
    runner = Runner(root, RunContext(run_backend, emb, tok, header, Session("fresh", epoch if epoch is not None else -1,
                                                                              "worker", fp), None),
                    ledger, shard=(k, n), claims=claims, log=lambda m: None)
    while time.time() < a.start_at:
        time.sleep(0.2)
    start = time.time()
    res = runner.run_phase(a.phase)
    end = time.time()
    out = {"shard": a.shard, "load_seconds": load_s, "canary_seconds": canary_s, "canary_fingerprint": fp,
           "canary_epoch": epoch, "start": start, "end": end, "wall": end - start, "run": res}
    Path(a.out).write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
