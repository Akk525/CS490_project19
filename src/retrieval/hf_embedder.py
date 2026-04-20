from typing import Dict, List

from src.retrieval.embedder_base import EmbedderBase


class HuggingFaceEmbedder(EmbedderBase):
    def __init__(self, backend_cfg: Dict, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceEmbedder requires sentence-transformers. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self.backend_cfg = backend_cfg
        self.model_name = model_name
        self.batch_size = int(backend_cfg.get('batch_size', 32))
        device = backend_cfg.get('device', 'auto')
        if device == 'auto':
            device = None
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embeddings.tolist()
