"""all-MiniLM-L6-v2 through sentence-transformers on CPU (D127): one thread, one text per call, normalised, rounded.
Recorded per unit in the replay cache (D129, `cache.CachedEmbedder`).

D143: the CPU code paths of MKL (COMPATIBLE), ATen and oneDNN (AVX2) are pinned through `CPU_PATHS_ENV`, set before torch is first
imported (the notebook's setup cell; worker processes inherit it), so vectors do not depend on which CPU a Colab VM has.
The setting is part of the embedder id. D144: the model is loaded from a sha256-verified local snapshot (`path`).

    python -m eem_acc.stage3.llm.embed --probe --path /content/minilm/<rev> --revision <rev>   # D142 setup check
"""
from __future__ import annotations

import os

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# MKL: COMPATIBLE is the only branch Intel documents as reproducible on Intel and non-Intel CPUs (other values fall
# back to AUTO silently on AMD); ATen and oneDNN: the AVX2 paths, which both vendors run.
CPU_PATHS_ENV = {"MKL_CBWR": "COMPATIBLE", "ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2"}
PROBE_TEXTS = ("pay_order rename amount", "Observation: 422 {\"error\": \"field 'amount' is required\"}",
               "Reflect on the error in one sentence.", "x y")


def cpu_paths_tag(environ=None) -> str:
    """`isa-compat-avx2` when D143's settings are all in place, `isa-default` when none is set, `isa-custom` otherwise."""
    environ = os.environ if environ is None else environ
    got = {k: environ.get(k) for k in CPU_PATHS_ENV}
    if got == CPU_PATHS_ENV:
        return "isa-compat-avx2"
    return "isa-default" if all(v is None for v in got.values()) else "isa-custom"


class MiniLMEmbedder:
    def __init__(self, revision: str, model_name: str = MODEL, path: str | None = None):
        import sentence_transformers
        import torch
        import transformers
        torch.set_num_threads(1)
        src = str(path) if path else model_name
        kw = {} if path else {"revision": revision}
        self.model = sentence_transformers.SentenceTransformer(src, device="cpu", **kw)
        self.cpu_capability = _cpu_capability(torch)
        self.id = (f"minilm-l6-v2-{revision[:8]}-st{sentence_transformers.__version__}-torch{torch.__version__}"
                   f"-tf{transformers.__version__}-{cpu_paths_tag()}")

    def embed(self, text: str) -> tuple:
        v = self.model.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)[0]
        return tuple(round(float(x), 8) for x in v)


def _cpu_capability(torch) -> str | None:
    try:
        return str(torch.backends.cpu.get_cpu_capability())
    except Exception:          # noqa: BLE001
        return None


def cpu_info() -> dict:
    """CPU model, vendor and the instruction-set flags that decide MKL/ATen code paths."""
    out = {"model": None, "vendor": None, "avx2": None, "avx512f": None}
    try:
        txt = open("/proc/cpuinfo").read()
    except OSError:
        return out
    for ln in txt.splitlines():
        k, _, v = ln.partition(":")
        k, v = k.strip(), v.strip()
        if k == "model name" and out["model"] is None:
            out["model"] = v
        elif k == "vendor_id" and out["vendor"] is None:
            out["vendor"] = v
        elif k == "flags" and out["avx2"] is None:
            flags = set(v.split())
            out["avx2"], out["avx512f"] = "avx2" in flags, "avx512f" in flags
    return out


def probe(path: str, revision: str) -> dict:
    """Load the embedder in this (fresh) interpreter and embed `PROBE_TEXTS`: proves the installed libraries can embed
    at the pinned revision (second opinion M1) and records what decides the vectors (D143)."""
    import hashlib
    import json

    import sentence_transformers
    import torch
    import transformers
    e = MiniLMEmbedder(revision, path=path)
    vecs = [e.embed(t) for t in PROBE_TEXTS]
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "sentence-transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
            "num_threads": torch.get_num_threads(), "cpu_capability": e.cpu_capability,
            "mkl": bool(torch.backends.mkl.is_available()), "mkldnn": bool(torch.backends.mkldnn.is_available()),
            "env": {k: os.environ.get(k) for k in CPU_PATHS_ENV}, "embedder_id": e.id, "dim": len(vecs[0]),
            "vectors_sha256": hashlib.sha256(json.dumps(vecs).encode()).hexdigest(), "cpu": cpu_info()}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--revision", required=True)
    a = ap.parse_args()
    print(json.dumps(probe(a.path, a.revision)))
