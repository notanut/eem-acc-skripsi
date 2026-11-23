"""Test fixtures for the S3.3a llama.cpp tests (never committed): the Qwen2 vocabulary GGUF from the pinned
llama-cpp-python 0.3.36 sdist (PyPI), and a tiny random-weight qwen2 GGUF written with the sdist's gguf-py,
carrying the real Qwen2.5 chat template (tests/fixtures/qwen25_chat_template.jinja, sha256-checked).

    python tools/fetch_test_models.py          # -> ~/.cache/eem-acc/{ggml-vocab-qwen2.gguf, tiny-qwen2.gguf}
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

CACHE = Path.home() / ".cache" / "eem-acc"
SDIST = "llama_cpp_python-0.3.36"
SDIST_SHA256 = "832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e"   # pinned (PyPI sdist of 0.3.36)
TEMPLATE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qwen25_chat_template.jinja"
TEMPLATE_SHA256 = "d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4"   # = S3.0's, captured on the T4 (S3.3b session 1)


def template_text() -> str:
    import hashlib
    raw = TEMPLATE.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != TEMPLATE_SHA256:
        raise SystemExit(f"template fixture sha256 {got} != {TEMPLATE_SHA256}")
    return raw.decode()


def fetch_vocab() -> Path:
    out = CACHE / "ggml-vocab-qwen2.gguf"
    gguf_py = CACHE / "gguf-py"
    if out.exists() and gguf_py.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:", "-q",
                        "llama-cpp-python==0.3.36", "-d", td], check=True)
        import hashlib
        got = hashlib.sha256((Path(td) / f"{SDIST}.tar.gz").read_bytes()).hexdigest()
        if got != SDIST_SHA256:
            raise SystemExit(f"sdist sha256 {got} != pinned {SDIST_SHA256}")
        with tarfile.open(Path(td) / f"{SDIST}.tar.gz") as tf:
            base = f"{SDIST}/vendor/llama.cpp/"
            for m in tf.getmembers():
                if m.name == base + "models/ggml-vocab-qwen2.gguf":
                    out.write_bytes(tf.extractfile(m).read())
                elif m.name.startswith(base + "gguf-py/") and m.isfile():
                    dest = gguf_py / m.name[len(base + "gguf-py/"):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tf.extractfile(m).read())
    return out


def make_tiny(vocab: Path) -> Path:
    """84 MB qwen2 GGUF: real Qwen2 vocabulary, random weights, the real Qwen2.5 chat template (S3.0's sha256).
    A cached file with another template (the S3.3a reconstruction) is rebuilt."""
    out = CACHE / "tiny-qwen2.gguf"
    sys.path.insert(0, str(CACHE / "gguf-py"))
    import numpy as np
    from gguf import GGUFReader, GGUFWriter
    template = template_text()
    if out.exists():
        field = GGUFReader(str(out)).fields.get("tokenizer.chat_template")
        if field is not None and bytes(field.parts[field.data[0]]).decode() == template:
            return out
    r = GGUFReader(str(vocab))
    f = r.fields
    w = GGUFWriter(str(out) + ".tmp", "qwen2")
    E, L, H, HKV, FF = 64, 2, 4, 2, 128
    V = len(f["tokenizer.ggml.tokens"].data)
    w.add_name("tiny")
    w.add_block_count(L)
    w.add_context_length(8192)
    w.add_embedding_length(E)
    w.add_feed_forward_length(FF)
    w.add_head_count(H)
    w.add_head_count_kv(HKV)
    w.add_rope_freq_base(1e6)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(0)

    def strs(name):
        return [bytes(f[name].parts[i]) for i in f[name].data]

    def ints(name):
        return [int(f[name].parts[i][0]) for i in f[name].data]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(strs("tokenizer.ggml.tokens"))
    w.add_token_types(ints("tokenizer.ggml.token_type"))
    w.add_token_merges(strs("tokenizer.ggml.merges"))
    w.add_eos_token_id(151645)
    w.add_bos_token_id(151643)
    w.add_pad_token_id(151643)
    w.add_add_bos_token(False)
    w.add_chat_template(template)
    rng = np.random.default_rng(0)

    def t(name, *shape):
        w.add_tensor(name, (rng.standard_normal(shape) * 0.05).astype(np.float32))
    t("token_embd.weight", V, E)
    w.add_tensor("output_norm.weight", np.ones(E, np.float32))
    t("output.weight", V, E)
    D = E // H
    for i in range(L):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", np.ones(E, np.float32))
        w.add_tensor(p + "ffn_norm.weight", np.ones(E, np.float32))
        t(p + "attn_q.weight", E, E)
        t(p + "attn_k.weight", HKV * D, E)
        t(p + "attn_v.weight", HKV * D, E)
        t(p + "attn_q.bias", E)
        t(p + "attn_k.bias", HKV * D)
        t(p + "attn_v.bias", HKV * D)
        t(p + "attn_output.weight", E, E)
        t(p + "ffn_gate.weight", FF, E)
        t(p + "ffn_up.weight", FF, E)
        t(p + "ffn_down.weight", E, FF)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    Path(str(out) + ".tmp").replace(out)
    return out


if __name__ == "__main__":
    v = fetch_vocab()
    print(v, make_tiny(v))
