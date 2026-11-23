"""S3.3a Colab-marked tests (run on the T4 in S3.3b; never claimed as passed from the CPU sandbox).

Environment: EEM_GGUF = path of the first GGUF part on Drive; EEM_MINILM_REV = the pinned MiniLM revision;
EEM_MINILM_PATH (optional, S3.3b session 2) = the verified local snapshot of that revision (D144)."""
import hashlib
import json
import os
import random
import subprocess
import sys

import pytest

pytestmark = pytest.mark.colab
S30_TEMPLATE_SHA = "d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4"


@pytest.fixture(scope="module")
def real():
    import llama_cpp
    from eem_acc.stage3.llm.llama_backend import LlamaCppBackend, LlamaTokenizer
    llm = llama_cpp.Llama(model_path=os.environ["EEM_GGUF"], n_ctx=8192, n_batch=512, n_gpu_layers=-1, seed=0,
                          verbose=False)
    b = LlamaCppBackend(llm, "colab", "colab")
    return llm, b, LlamaTokenizer(b)


def test_real_template_is_s30s(real):
    llm, b, _ = real
    assert b.template_sha256 == S30_TEMPLATE_SHA


def test_count_messages_equals_n_in_and_chat_path_equal(real):
    from eem_acc.stage3.llm.canary import load_reference_prompts
    from eem_acc.stage3.llm.llama_backend import SCHEMAS
    llm, b, tok = real
    for p in load_reference_prompts()[0][:4]:
        g = b.generate(p["messages"], p["kind"], p["grammar"], 64)
        assert tok.count_messages(p["messages"]) == g.n_in
        kw = dict(temperature=0.0, top_k=1, max_tokens=64, seed=0)
        if p["grammar"]:
            kw["response_format"] = {"type": "json_object", "schema": SCHEMAS[p["grammar"]]}
        llm.reset()
        assert llm.create_chat_completion(messages=p["messages"], **kw)["choices"][0]["message"]["content"] == g.text


def test_note_render_caps_with_real_tokenizer(real):
    from eem_acc.stage3.agent import BagOfWordsEmbedder
    from eem_acc.stage3.agent.memory import EEMStore, NOTE_CAP
    from test_s32_memory import DOC_ARGS, _err, _suite
    _, _, tok = real
    rnd = random.Random(0)
    for mode in ("eem", "a2", "a1"):
        for i in range(400):
            m = EEMStore(BagOfWordsEmbedder(), tok, mode=mode)
            s = _suite(["pay_order:rename"])
            _err(m, s, "pay_order", DOC_ARGS["pay_order"], reflection=" ".join(rnd.choice(
                ["use", "the", "field", "total", "amount", "integer", "instead", "of", "'method'", "CARD"])
                for _ in range(rnd.randint(0, 90))))
            n = m.notes[1]
            n.items = [{"kind": "rename", "old": f"field_{j}", "new": f"value_{j}", "type": "string"}
                       for j in range(rnd.randint(0, 8))]
            text, _ = m.note_render(n)
            assert tok.count(text) <= NOTE_CAP


def test_snapshot_path_repeatable(real):
    """Design section 9: the snapshot path may differ from the fresh path (other batch boundaries), so it is its own
    epoch series; what must hold is repeatability within the path."""
    from eem_acc.stage3.llm.canary import load_reference_prompts
    from eem_acc.stage3.llm.llama_backend import LlamaCppBackend
    llm, _, _ = real
    snap = LlamaCppBackend(llm, "colab", "colab", snapshot=True)
    for p in load_reference_prompts()[0][:3]:
        assert snap.generate(p["messages"], p["kind"], p["grammar"], 48) == \
            snap.generate(p["messages"], p["kind"], p["grammar"], 48)


def test_minilm_repeatable_across_processes():
    code = ("import os,json;from eem_acc.stage3.llm.embed import MiniLMEmbedder as M;"
            "e=M(os.environ['EEM_MINILM_REV'],path=os.environ.get('EEM_MINILM_PATH'));"
            "print(json.dumps([e.embed('pay_order rename amount'),e.id]))")
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
            for _ in range(2)]
    assert outs[0] == outs[1]
    from eem_acc.stage3.llm.embed import MiniLMEmbedder
    e = MiniLMEmbedder(os.environ["EEM_MINILM_REV"], path=os.environ.get("EEM_MINILM_PATH"))
    assert e.embed("x y") == e.embed("x y") and json.loads(outs[0])[1] == e.id
