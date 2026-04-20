from typing import Dict, List, Tuple

from src.retrieval.reranker_base import RerankerBase


class HuggingFaceReranker(RerankerBase):
    def __init__(self, backend_cfg: Dict, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceReranker requires sentence-transformers. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self.backend_cfg = backend_cfg
        self.model_name = model_name
        self.batch_size = int(backend_cfg.get('batch_size', 16))
        device = backend_cfg.get('device', 'auto')
        if device == 'auto':
            device = None
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [(idx, float(score)) for idx, score in enumerate(scores)]
