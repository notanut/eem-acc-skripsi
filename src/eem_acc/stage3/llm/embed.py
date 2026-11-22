"""all-MiniLM-L6-v2 through sentence-transformers on CPU (D127): one thread, one text per call, normalised, rounded.
Recorded per unit in the replay cache (D129, `cache.CachedEmbedder`)."""
from __future__ import annotations

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbedder:
    def __init__(self, revision: str, model_name: str = MODEL):
        import sentence_transformers
        import torch
        import transformers
        torch.set_num_threads(1)
        self.model = sentence_transformers.SentenceTransformer(model_name, revision=revision, device="cpu")
        self.id = (f"minilm-l6-v2-{revision[:8]}-st{sentence_transformers.__version__}-torch{torch.__version__}"
                   f"-tf{transformers.__version__}")

    def embed(self, text: str) -> tuple:
        v = self.model.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)[0]
        return tuple(round(float(x), 8) for x in v)
